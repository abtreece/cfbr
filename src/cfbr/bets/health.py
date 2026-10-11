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
    """Never raises: an unexpected error becomes a problem string."""
    now = now or datetime.now(UTC)
    try:
        return _check(cfg, season, now)
    except Exception as e:  # noqa: BLE001
        return [f"health check itself failed: {type(e).__name__}: {e}"]


def _guarded(problems: list[str], what: str, fn) -> None:
    """Run one read-only check; a missing file or table never stops the others."""
    try:
        problems.extend(fn())
    except Exception as e:  # noqa: BLE001
        problems.append(f"{what} check failed: {type(e).__name__}: {e}")


def _check(cfg: Config, season: int, now: datetime) -> list[str]:
    problems: list[str] = backup_problems(cfg, now)
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
    from cfbr.bets.integrity import verify

    problems += verify(cfg.paths.out_dir, season)
    recent = settled.filter(pl.col("kickoff_ts").str.to_datetime(time_zone="UTC") > now - timedelta(days=8))
    fallback = recent.filter(pl.col("closing_source") == "cfbd").height
    if cfg.odds.enabled and fallback:
        problems.append(f"{fallback} of {recent.height} bets settled this week were graded against CFBD's close "
                        "(no pre-kickoff snapshot; is the odds-snapshot cron running?)")
    _guarded(problems, "CFBD snapshot freshness", lambda: cfbd_snapshot_problems(cfg, season, now))
    if cfg.odds.enabled:
        _guarded(problems, "snapshot gap", lambda: snapshot_gap_problems(cfg, season))
        _guarded(problems, "Odds API match rate", lambda: match_rate_problems(cfg, season))
    _guarded(problems, "Sunday bets", lambda: sunday_bets_problems(cfg, season, led, now))
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


def backup_problems(cfg: Config, now: datetime) -> list[str]:
    from cfbr.backup import last_backup, target

    last = last_backup(cfg)
    if last is None:
        return [f"no backup found at {target(cfg)} (is the `cfbr backup` cron running?)"]
    age = now - last
    if age > timedelta(days=cfg.backup.max_age_days):
        return [f"last backup is {age.days} days old ({target(cfg)})"]
    return []


def cfbd_snapshot_problems(cfg: Config, season: int, now: datetime) -> list[str]:
    """(a) The CFBD lines snapshot every run takes should be fresh."""
    from cfbr.ingest.fetch import latest_lines_snapshot

    snap = latest_lines_snapshot(cfg.paths.raw_dir, season)
    if snap is None:
        return ["no CFBD lines snapshot this season (bets can't be priced)"]
    age = now - snap[0]
    if age > timedelta(hours=3):
        return [f"newest CFBD lines snapshot is {age.total_seconds() / 3600:.0f} h old (ingest didn't take one?)"]
    return []


def snapshot_gap_problems(cfg: Config, season: int) -> list[str]:
    """(b) An Odds API snapshot too far from the CFBD one is discarded by ``current_books``,
    and bets are then silently priced from CFBD's books only."""
    from cfbr.ingest.fetch import latest_lines_snapshot
    from cfbr.ingest.odds import latest_odds_snapshot

    snap = latest_lines_snapshot(cfg.paths.raw_dir, season)
    odds = latest_odds_snapshot(cfg.paths.raw_dir, season)
    if snap is None or odds is None:
        return []
    gap = abs((odds[0] - snap[0]).total_seconds()) / 3600
    if gap > cfg.odds.max_snapshot_gap_hours:
        return [f"Odds API snapshot is {gap:.1f} h from the CFBD snapshot (limit {cfg.odds.max_snapshot_gap_hours:g} h); "
                "it was discarded and bets were priced from CFBD lines only"]
    return []


def match_rate_problems(cfg: Config, season: int, min_rate: float = 0.9) -> list[str]:
    """(d) Most Odds API events should match a CFBD game; a drop means names or dates changed."""
    import json

    from cfbr.bets.closing import cached_games_frame
    from cfbr.ingest.odds import latest_odds_snapshot, match_stats, team_aliases

    odds = latest_odds_snapshot(cfg.paths.raw_dir, season)
    games = cached_games_frame(cfg.paths.raw_dir, season)
    if odds is None or games is None or not odds[1]:
        return []
    teams_path = cfg.paths.raw_dir / "teams_fbs" / str(season) / "season.json"
    aliases = team_aliases(json.loads(teams_path.read_text())) if teams_path.exists() else {}
    matched, total = match_stats(odds[1], games, aliases)
    if matched < min_rate * total:
        return [f"only {matched} of {total} Odds API events matched a CFBD game in the newest snapshot"]
    return []


def sunday_bets_problems(cfg: Config, season: int, led: pl.DataFrame, now: datetime) -> list[str]:
    """(c) A Sunday run with games in the coming week should record bets (unless none qualify,
    which this check can't tell apart; the alert is a prompt to look)."""
    from cfbr.bets.closing import cached_games_frame

    if now.weekday() != 6 or not cfg.bets.enabled:
        return []
    games = cached_games_frame(cfg.paths.raw_dir, season)
    if games is None:
        return []
    upcoming = games.filter(~pl.col("completed") & (pl.col("kickoff_ts") > now)
                            & (pl.col("kickoff_ts") <= now + timedelta(days=7))).height
    if upcoming == 0:
        return []
    when = pl.coalesce("recorded_at", "placed_at")
    today = led.filter(when.str.slice(0, 10) == now.strftime("%Y-%m-%d")).height
    if today == 0:
        return [f"no bets recorded today although {upcoming} games kick off in the next 7 days"]
    return []
