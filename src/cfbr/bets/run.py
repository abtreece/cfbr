"""Weekly paper-bet step: settle finished bets, then record new ones."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import consensus_spread
from cfbr.bets.ledger import (add_bets, candidate_bets, read_ledger, settle, snapshot_frame, summarize,
                              summary_markdown)
from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.ingest.fetch import latest_lines_snapshot

log = logging.getLogger(__name__)


def run_bets(cfg: Config, ds: Dataset, season: int, week: int) -> Path | None:
    """``week`` is the last completed week; new bets come from that week's
    predictions.csv (the next week's games) against the newest line snapshot."""
    out_dir = cfg.paths.out_dir / str(season)
    ledger_path = out_dir / "bets.csv"
    ledger = read_ledger(ledger_path)

    games = ds.games.filter(pl.col("season") == season)
    closing = consensus_spread(ds.lines, cfg.backtest.spread_consensus).select(
        "game_id", closing_line="spread")
    ledger = settle(ledger, games.select("game_id", "completed", "home_points", "away_points"), closing)

    snap = latest_lines_snapshot(cfg.paths.raw_dir, season)
    preds_path = out_dir / f"w{week:02d}" / "predictions.csv"
    added = 0
    if snap is None:
        log.warning("no live line snapshot for %d; no new bets recorded", season)
    elif not preds_path.exists():
        log.warning("%s missing; run `cfbr rank` first", preds_path)
    else:
        placed_at, lines_json = snap
        preds = pl.read_csv(preds_path)
        new = candidate_bets(preds, snapshot_frame(lines_json), placed_at, cfg.bets)
        new = (new.drop("season", "week")
               .join(games.select("game_id", pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)),
                     on="game_id", how="left"))
        ledger, added = add_bets(ledger, new)

    out_dir.mkdir(parents=True, exist_ok=True)
    ledger.sort("kickoff_ts", "game_id").write_csv(ledger_path)
    (out_dir / "bets.md").write_text(summary_markdown(ledger, cfg.bets, season))
    m = summarize(ledger, cfg.bets)
    log.info("paper bets: %d new, %d open, %d settled (%d-%d-%d), units %+.2f, avg CLV %s",
             added, m["open"], m["settled"], m["won"], m["lost"], m["push"], m["units"],
             "n/a" if m["n_clv"] == 0 else f"{m['avg_clv']:+.2f}")
    return ledger_path
