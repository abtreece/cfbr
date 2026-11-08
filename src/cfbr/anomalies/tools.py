"""Tools for the investigation agent.

`query` runs agent-written SQL on a snapshot of cfb.duckdb. What is enforced:
- Each query runs in its own short-lived child process (multiprocessing "spawn", never "fork": DuckDB
  is multithreaded). The parent holds no DuckDB connection. The child is killed at
  query_timeout_seconds, which also stops work DuckDB's interrupt can't reach (the optimizer on
  deeply nested subqueries). A child that dies without answering (abort, out-of-memory kill,
  rlimit) is reported as a tool error, never raised; so is any failure to create the pipe, start the
  child or read its message.
- Memory: the parent's is bounded by the row/char caps. The child's is bounded by DuckDB's
  memory_limit for DuckDB itself and, on Linux, by RLIMIT_AS at 2x that; Python conversion of a
  fetched batch in the child is not separately bounded. The parent's `__main__` must be guarded by
  `if __name__ == "__main__"` (spawn re-imports it; the cfbr console script and `python -m cfbr.cli`
  are guarded).
- The child's connection is read-only, has external access disabled (no read_csv on .keys, no
  ATTACH, COPY or INSTALL), configuration locked (no SET to undo any of this), an empty
  temp_directory (so DuckDB's automatic allowance for `<db>.tmp/` is gone; `<db>.wal` is refused by
  the read-only, no-external-access connection), and memory_limit/threads from
  query_memory_limit/query_threads, all set at connect time. On Linux the child also sets
  RLIMIT_AS to twice query_memory_limit, best effort (ignored where the OS rejects it).
- Before executing, the child checks the SQL using only that restricted connection (never
  module-level `duckdb.*`, which uses an unrestricted default connection): it must be exactly one
  statement of type SELECT (DuckDB also classifies DESCRIBE, SHOW and SUMMARIZE as SELECT); it must
  serialize to a parse tree (PRAGMA forms don't, so they are refused); every table function in that
  tree must be in ALLOWED_TABLE_FUNCTIONS (a table-function node without a readable name is
  refused); and no scalar function may be in DENIED_SCALAR_FUNCTIONS (the ones that turn a string
  into SQL). The tree is walked iteratively, so deep nesting can't overflow Python's stack. This
  is an allowlist on the parse tree, so quoting, comments, schema qualification, CTEs and
  subqueries can't hide a function.
- Results are fetched in batches and rendering stops at query_max_rows or query_max_chars, with
  each cell cut to query_max_chars first, so a huge result is never materialized in Python.
Refusals are returned as tool errors. `read_output` serves a fixed set of `rank` output files by
name, never by path. Both tools number their results in one sequence; a verdict's evidence cites
those ids and the grounding check (agent.grounded) looks them up in `results`.
"""

from __future__ import annotations

import json
import multiprocessing
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

if TYPE_CHECKING:  # the query child imports this module; keep pydantic out of its startup
    from cfbr.config import Investigate

OUTPUT_FILES = {
    "rankings.csv": "this week's ratings: rank, team, rating, offense_rating, defense_rating, rating_change, prior_share, ...",
    "rankings_prev.csv": "the previous week's rankings.csv",
    "predictions.csv": "next week's games: pred_margin (model home margin) and closing_spread (current consensus, home perspective)",
    "accuracy.csv": "season-to-date accuracy line, one row per week",
}


ALLOWED_TABLE_FUNCTIONS = {"range", "generate_series", "unnest", "duckdb_tables", "duckdb_columns"}
DENIED_SCALAR_FUNCTIONS = {"json_serialize_plan", "json_execute_serialized_sql", "query", "query_table"}

_FETCH_BATCH = 50
_UNITS = {"b": 1, "kb": 1000, "mb": 1000**2, "gb": 1000**3, "tb": 1000**4,
          "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4}


def _parse_bytes(size: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-z]*)\s*", size.lower())
    if match is None or match.group(2) not in _UNITS | {"": 1}:
        raise ValueError(f"can't parse memory size {size!r}")
    return int(float(match.group(1)) * _UNITS.get(match.group(2), 1))


def _functions(tree) -> tuple[list[str], list[str]]:
    """Table-function and scalar-function names in a json_serialize_sql tree, walked iteratively."""
    table_functions: list[str] = []
    scalar_functions: list[str] = []
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if node.get("type") == "TABLE_FUNCTION":
                function = node.get("function")
                # Fail closed: a node whose name can't be read is recorded as "" and refused.
                table_functions.append(str(function.get("function_name", "")) if isinstance(function, dict) else "")
            elif node.get("class") == "FUNCTION":
                scalar_functions.append(str(node.get("function_name", "")))
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return table_functions, scalar_functions


def _refusal(con: duckdb.DuckDBPyConnection, sql: str) -> str | None:
    try:
        statements = con.extract_statements(sql)
    except Exception as e:  # noqa: BLE001 - parser message goes back to the model
        return str(e)[:300]
    if len(statements) != 1 or statements[0].type.name != "SELECT":
        return "only a single SELECT statement is allowed"
    try:
        tree = json.loads(con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    except Exception as e:  # noqa: BLE001
        return f"statement can't be checked: {str(e)[:300]}"
    if tree.get("error"):
        return f"statement can't be checked: {str(tree.get('error_message', ''))[:300]}"
    table_functions, scalar_functions = _functions(tree)
    for name in table_functions:
        if name.lower() not in ALLOWED_TABLE_FUNCTIONS:
            return f"table function not allowed: {name}"
    for name in scalar_functions:
        if name.lower() in DENIED_SCALAR_FUNCTIONS:
            return f"function not allowed: {name}"
    return None


def _cell(v, limit: int) -> str:
    text = "NULL" if v is None else str(v)
    return text if len(text) <= limit else text[:limit]


def _render(cur: duckdb.DuckDBPyConnection, max_rows: int, max_chars: int) -> tuple[str, int]:
    """Header plus up to max_rows rows; stops fetching once the text passes max_chars."""
    cols = [d[0] for d in cur.description] if cur.description else []
    lines = ["\t".join(cols)]
    length = len(lines[0])
    shown = 0
    more = False
    while length <= max_chars and not more:
        batch = cur.fetchmany(_FETCH_BATCH)
        if not batch:
            break
        for row in batch:
            if shown == max_rows:
                more = True
                break
            line = "\t".join(_cell(v, max_chars) for v in row)
            lines.append(line)
            length += len(line) + 1
            shown += 1
            if length > max_chars:
                break
    body = "\n".join(lines)
    if more:
        body += f"\n… more than {max_rows} rows; aggregate or add LIMIT"
    # One past the budget, so the parent's cap still sees the overflow and marks it.
    return body[: max_chars + 1], shown


def _query_child(conn, db_path: str, sql: str, settings: dict) -> None:
    """Runs in a spawned process: gate, execute, render, send exactly one message, exit."""
    try:
        if sys.platform.startswith("linux"):
            try:
                import resource

                limit = 2 * _parse_bytes(settings["memory_limit"])
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except Exception:  # noqa: BLE001, S110 - best effort; DuckDB's memory_limit still applies
                pass
        # Key order matters in duckdb 1.5: with memory_limit/threads present, connect raises
        # "Modifying the temp_directory has been disabled" unless enable_external_access and
        # memory_limit come before temp_directory and lock_configuration. A test asserts the
        # effective settings.
        con = duckdb.connect(db_path, read_only=True,
                             config={"enable_external_access": False, "memory_limit": settings["memory_limit"],
                                     "threads": settings["threads"], "temp_directory": "",
                                     "lock_configuration": True})
        try:
            if (reason := _refusal(con, sql)) is not None:
                conn.send(("refused", reason))
                return
            body, shown = _render(con.execute(sql), settings["max_rows"], settings["max_chars"])
            conn.send(("ok", body, shown))
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 - every failure goes back to the model as a tool error
        conn.send(("error", f"{type(e).__name__}: {e}"[:2000]))
    finally:
        conn.close()


class ToolSession:
    def __init__(self, db_path: Path, out_dir: Path, season: int, week: int, cfg: Investigate):
        self.db_path, self.out_dir, self.season, self.week, self.cfg = db_path, out_dir, season, week, cfg
        # No DuckDB connection lives in this process; each query opens its own in a child.
        # TIMESTAMPTZ renders in the host zone (javelin's clock is UTC).
        self.results: dict[int, str] = {}
        self.log: list[dict] = []

    def _next_id(self) -> int:
        return len(self.log) + 1

    def _cap(self, text: str) -> str:
        limit = self.cfg.query_max_chars
        return text if len(text) <= limit else text[:limit] + "\n… truncated"

    def _done(self, rid: int, tool: str, inp: str, text: str, ok: bool) -> tuple[str, bool]:
        self.log.append({"id": rid, "tool": tool, "input": inp, "ok": ok, "text": text})
        if ok:
            self.results[rid] = text
        return text, not ok

    def _run_child(self, sql: str) -> tuple:
        settings = {"max_rows": self.cfg.query_max_rows, "max_chars": self.cfg.query_max_chars,
                    "memory_limit": self.cfg.query_memory_limit, "threads": self.cfg.query_threads}
        receiver = sender = proc = None
        message = None
        try:
            ctx = multiprocessing.get_context("spawn")
            receiver, sender = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_query_child, args=(sender, str(self.db_path), sql, settings), daemon=True)
            proc.start()
            sender.close()  # only the child writes; EOF on the receiver then means the child is gone
            if not receiver.poll(self.cfg.query_timeout_seconds):
                return ("error", f"timed out after {self.cfg.query_timeout_seconds:g}s")
            try:
                message = receiver.recv()
            except (EOFError, OSError):
                message = None  # child died before or while sending (a partial message raises OSError)
        except Exception as e:
            return ("error", f"query process failed: {type(e).__name__}: {e}"[:2000])
        finally:
            if proc is not None and proc.pid is not None:
                proc.kill()
                proc.join()
            for end in (sender, receiver):
                if end is not None:
                    try:
                        end.close()
                    except OSError:
                        pass
        if message is None:
            return ("error", f"query process exited with code {proc.exitcode}")
        return message

    def query(self, sql: str) -> tuple[str, bool]:
        rid = self._next_id()
        message = self._run_child(sql)
        if message[0] == "refused":
            return self._done(rid, "query", sql, f"result_id {rid}: error: refused: {message[1]}", False)
        if message[0] == "error":
            return self._done(rid, "query", sql, f"result_id {rid}: error: {message[1]}"[:2000], False)
        _, body, shown = message
        return self._done(rid, "query", sql, self._cap(f"result_id {rid} ({shown} rows)\n{body}"), True)

    def read_output(self, name: str) -> tuple[str, bool]:
        rid = self._next_id()
        season_dir = self.out_dir / str(self.season)
        paths = {"rankings.csv": season_dir / f"w{self.week:02d}" / "rankings.csv",
                 "rankings_prev.csv": season_dir / f"w{self.week - 1:02d}" / "rankings.csv",
                 "predictions.csv": season_dir / f"w{self.week:02d}" / "predictions.csv",
                 "accuracy.csv": season_dir / "accuracy.csv"}
        if name not in paths:
            return self._done(rid, "read_output", name,
                              f"result_id {rid}: error: unknown file {name!r}; allowed: {', '.join(paths)}", False)
        path = paths[name]
        if not path.exists():
            return self._done(rid, "read_output", name,
                              f"result_id {rid}: error: {name} is not available for {self.season} week {self.week}", False)
        return self._done(rid, "read_output", name, self._cap(f"result_id {rid} ({name})\n{path.read_text()}"), True)

    def close(self) -> None:
        """No-op: the parent holds no connection. Kept so callers don't change."""
