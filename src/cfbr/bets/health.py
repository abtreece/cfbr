"""Post-run health check for the paper-bet tracker (run at the end of ``cfbr weekly``).

Returns short, human-readable problems; an empty list means everything looks right.
It reads outputs only and never changes the ledger.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb
import polars as pl

from cfbr.bets.ledger import SETTLED, read_ledger
from cfbr.config import Config


def check(cfg: Config, season: int, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(UTC)
    problems: list[str] = []
    path = cfg.paths.out_dir / str(season) / "bets.csv"
    if not path.exists():
        return problems  # nothing recorded yet this season
    try:
        led = read_ledger(path)
    except Exception as e:  # noqa: BLE001
        return [f"bets.csv unreadable: {e}"]
    dup = led.filter(pl.col("game_id").is_duplicated())["game_id"].unique().to_list()
    if dup:
        problems.append(f"{len(dup)} games bet twice in the ledger")
    settled = led.filter(pl.col("status").is_in(SETTLED))
    if settled.filter(pl.col("units").is_null()).height:
        problems.append(f"{settled.filter(pl.col('units').is_null()).height} settled bets have no units")
    opn = led.filter(pl.col("status") == "open")
    if opn.height and cfg.paths.db_path.exists():
        with duckdb.connect(str(cfg.paths.db_path), read_only=True) as con:
            done = con.execute("SELECT game_id FROM games WHERE completed AND season = ?", [season]).pl()
        stuck = opn.filter(pl.col("game_id").is_in(done["game_id"].to_list())).height
        if stuck:
            problems.append(f"{stuck} bets on finished games are still open (settlement failed?)")
    stale = opn.filter(pl.col("kickoff_ts").str.to_datetime(time_zone="UTC") < now - timedelta(days=4)).height
    if stale:
        problems.append(f"{stale} open bets kicked off over 4 days ago")
    if cfg.odds.enabled:
        from cfbr.ingest.odds import credits_left, latest_odds_snapshot, odds_api_key

        if odds_api_key() is not None:
            snap = latest_odds_snapshot(cfg.paths.raw_dir, season)
            if snap is None or now - snap[0] > timedelta(hours=3):
                problems.append("no fresh Odds API snapshot this run (bets priced from CFBD lines only)")
            if credits_left(cfg.paths.raw_dir, cfg.odds.monthly_quota) < cfg.odds.quota_reserve + 10:
                problems.append("The Odds API credits nearly used up this month")
    if cfg.oddspapi.enabled:
        from cfbr.ingest.oddspapi import oddspapi_key, requests_left

        if oddspapi_key() is not None and requests_left(cfg) < 30:
            problems.append(f"OddsPapi (bet365): only {max(requests_left(cfg), 0)} requests left this month")
    return problems
