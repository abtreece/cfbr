"""Tools for the investigation agent.

`query` runs agent-written SQL on a snapshot of cfb.duckdb. What is enforced:
- The connection is read-only, has external access disabled (no read_csv on .keys, no ATTACH, COPY
  or INSTALL), configuration locked (no SET to undo that), an empty temp_directory (so DuckDB's
  automatic allowance for `<db>.tmp/` is gone; `<db>.wal` is refused by the read-only,
  no-external-access connection), all set at connect time.
- Before executing, the SQL is checked using only that restricted connection (never module-level
  `duckdb.*`, which uses an unrestricted default connection): it must be exactly one statement of
  type SELECT (DuckDB also classifies DESCRIBE, SHOW and SUMMARIZE as SELECT); it must serialize to
  a parse tree (PRAGMA forms don't, so they are refused); and every table function in that tree
  must be in ALLOWED_TABLE_FUNCTIONS. This is an allowlist on the parse tree, so quoting, comments,
  schema qualification, CTEs and subqueries can't hide a function.
Refusals are returned as tool errors. `read_output` serves a fixed set of `rank` output files by
name, never by path. Both tools number their results in one sequence; a verdict's evidence cites
those ids and the grounding check (agent.grounded) looks them up in `results`.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import duckdb

from cfbr.config import Investigate

OUTPUT_FILES = {
    "rankings.csv": "this week's ratings: rank, team, rating, offense_rating, defense_rating, rating_change, prior_share, ...",
    "rankings_prev.csv": "the previous week's rankings.csv",
    "predictions.csv": "next week's games: pred_margin (model home margin) and closing_spread (current consensus, home perspective)",
    "accuracy.csv": "season-to-date accuracy line, one row per week",
}


ALLOWED_TABLE_FUNCTIONS = {"range", "generate_series", "unnest", "duckdb_tables", "duckdb_columns"}


def _table_functions(node, found: list[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "TABLE_FUNCTION":
            function = node.get("function")
            if isinstance(function, dict):
                found.append(str(function.get("function_name", "")))
        for v in node.values():
            _table_functions(v, found)
    elif isinstance(node, list):
        for v in node:
            _table_functions(v, found)


def _cell(v) -> str:
    return "NULL" if v is None else str(v)


class ToolSession:
    def __init__(self, db_path: Path, out_dir: Path, season: int, week: int, cfg: Investigate):
        self.db_path, self.out_dir, self.season, self.week, self.cfg = db_path, out_dir, season, week, cfg
        # Every restriction is applied at connect time, so no agent-reachable moment exists
        # before the lock. TIMESTAMPTZ renders in the host zone (javelin's clock is UTC).
        self.con = duckdb.connect(str(db_path), read_only=True,
                                  config={"enable_external_access": False, "lock_configuration": True,
                                          "temp_directory": ""})
        self.results: dict[int, str] = {}
        self.log: list[dict] = []

    def _refusal(self, sql: str) -> str | None:
        try:
            statements = self.con.extract_statements(sql)
        except Exception as e:  # noqa: BLE001 - parser message goes back to the model
            return str(e)[:300]
        if len(statements) != 1 or statements[0].type.name != "SELECT":
            return "only a single SELECT statement is allowed"
        try:
            tree = json.loads(self.con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
        except Exception as e:  # noqa: BLE001
            return f"statement can't be checked: {str(e)[:300]}"
        if tree.get("error"):
            return f"statement can't be checked: {str(tree.get('error_message', ''))[:300]}"
        found: list[str] = []
        _table_functions(tree, found)
        for name in found:
            if name.lower() not in ALLOWED_TABLE_FUNCTIONS:
                return f"table function not allowed: {name}"
        return None

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

    def query(self, sql: str) -> tuple[str, bool]:
        rid = self._next_id()
        if (reason := self._refusal(sql)) is not None:
            return self._done(rid, "query", sql, f"result_id {rid}: error: refused: {reason}", False)
        timed_out = threading.Event()

        def stop() -> None:
            timed_out.set()
            self.con.interrupt()

        timer = threading.Timer(self.cfg.query_timeout_seconds, stop)
        timer.start()
        try:
            cur = self.con.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(self.cfg.query_max_rows + 1)
        except Exception as e:  # noqa: BLE001 - every failure goes back to the model as a tool error
            why = f"timed out after {self.cfg.query_timeout_seconds:g}s" if timed_out.is_set() else f"{type(e).__name__}: {e}"
            return self._done(rid, "query", sql, f"result_id {rid}: error: {why}"[:2000], False)
        finally:
            timer.cancel()
        shown = rows[: self.cfg.query_max_rows]
        body = "\n".join(["\t".join(cols)] + ["\t".join(_cell(v) for v in r) for r in shown])
        if len(rows) > self.cfg.query_max_rows:
            body += f"\n… more than {self.cfg.query_max_rows} rows; aggregate or add LIMIT"
        return self._done(rid, "query", sql, self._cap(f"result_id {rid} ({len(shown)} rows)\n{body}"), True)

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
        self.con.close()
