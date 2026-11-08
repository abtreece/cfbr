"""Tools for the investigation agent.

`query` enforces, in order: (1) the SQL must parse as exactly one statement of type SELECT
(DuckDB also classifies PRAGMA, DESCRIBE, SHOW and SUMMARIZE as SELECT); (2) it must not call a
function on DENIED_FUNCTIONS; then it runs on a snapshot of cfb.duckdb opened read-only with
external access disabled (no file reads such as read_csv on .keys, no ATTACH, COPY or INSTALL),
configuration locked (no SET to undo that) and temp_directory empty (so DuckDB's automatic
allowance for `<db>.tmp/` is gone; `<db>.wal` is refused by the read-only, no-external-access
connection). Refusals are returned as tool errors. `read_output` serves a fixed set of `rank`
output files by name, never by path. Both tools number their results in one sequence; a
verdict's evidence cites those ids and the grounding check (agent.grounded) looks them up in
`results`.
"""

from __future__ import annotations

import re
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


# enable_logging/disable_logging/truncate_duckdb_logs: calling them aborts the whole process
# (uncaught C++ PermissionException); query/query_table: run string-built SQL that bypasses the
# single-SELECT statement gate.
DENIED_FUNCTIONS = ("enable_logging", "disable_logging", "truncate_duckdb_logs", "query", "query_table")
_DENIED_CALL = re.compile(r"\b(" + "|".join(DENIED_FUNCTIONS) + r")\s*\(", re.IGNORECASE)


def _refusal(sql: str) -> str | None:
    try:
        statements = duckdb.extract_statements(sql)
    except Exception as e:  # noqa: BLE001 - parser message goes back to the model
        return str(e)
    if len(statements) != 1 or statements[0].type.name != "SELECT":
        return "exactly one SELECT statement is allowed"
    if m := _DENIED_CALL.search(sql):
        return f"{m.group(1).lower()}() is not allowed"
    return None


def _cell(v) -> str:
    return "NULL" if v is None else str(v)


class ToolSession:
    def __init__(self, db_path: Path, out_dir: Path, season: int, week: int, cfg: Investigate):
        self.db_path, self.out_dir, self.season, self.week, self.cfg = db_path, out_dir, season, week, cfg
        self.con = duckdb.connect(str(db_path), read_only=True,
                                  config={"enable_external_access": False, "lock_configuration": True,
                                          "temp_directory": ""})
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

    def query(self, sql: str) -> tuple[str, bool]:
        rid = self._next_id()
        if (reason := _refusal(sql)) is not None:
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
