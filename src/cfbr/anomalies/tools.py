"""Tools for the investigation agent.

`query` runs SQL on a snapshot of cfb.duckdb opened read-only, with external access disabled
and configuration locked: no writes, no file reads (read_csv on .keys), no ATTACH, COPY or
INSTALL, and no SET to undo that. `read_output` serves a fixed set of `rank` output files by
name, never by path. Both tools number their results in one sequence; a verdict's evidence cites
those ids and the grounding check (agent.grounded) looks them up in `results`.
"""

from __future__ import annotations

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


def _cell(v) -> str:
    return "NULL" if v is None else str(v)


class ToolSession:
    def __init__(self, db_path: Path, out_dir: Path, season: int, week: int, cfg: Investigate):
        self.db_path, self.out_dir, self.season, self.week, self.cfg = db_path, out_dir, season, week, cfg
        self.con = duckdb.connect(str(db_path), read_only=True,
                                  config={"enable_external_access": False, "lock_configuration": True})
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
