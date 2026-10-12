"""Dead-man's switch for the unattended weekly run (review O2).

``cfbr weekly`` records each successful run in ``out/last_success.json``. ``cfbr check``
(a daily cron) alerts when, during the season, no success has been recorded since the
most recent Sunday run window, so a crontab that silently stopped firing is noticed.
Failures inside a run already alert on their own.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from cfbr.config import Config

log = logging.getLogger(__name__)
STAMP = "last_success.json"
CENTRAL = ZoneInfo("America/Chicago")
# The last weekly run of the week starts at 3 PM Central on Sunday; by this hour it has finished.
SUNDAY_DEADLINE_HOUR = 18


def record_success(cfg: Config, season: int, code: int, now: datetime | None = None) -> Path:
    now = now or datetime.now(UTC)
    path = cfg.paths.out_dir / STAMP
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"finished_at": now.isoformat(timespec="seconds"), "season": season, "exit": code}))
    tmp.replace(path)
    return path


def last_success(cfg: Config) -> datetime | None:
    path = cfg.paths.out_dir / STAMP
    if not path.exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(path.read_text())["finished_at"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def season_active(games: pl.DataFrame | None, now: datetime, slack_days: int = 10) -> bool:
    """A season is active when a game kicked off in the last ``slack_days`` and another kicks
    off in the next ``slack_days`` (``games``: the cached games frame with ``kickoff_ts``)."""
    if games is None or games.height == 0:
        return False
    k = games["kickoff_ts"].drop_nulls()
    if k.len() == 0:
        return False
    recent = (k >= now - timedelta(days=slack_days)) & (k <= now)
    upcoming = (k > now) & (k <= now + timedelta(days=slack_days))
    return bool(recent.any() and upcoming.any())


def expected_since(now: datetime) -> datetime | None:
    """Start of the most recent Sunday whose run window has closed (Sunday 00:00 Central), or
    None when no window has closed yet this week (Sunday before the deadline hour)."""
    local = now.astimezone(CENTRAL)
    days_back = (local.weekday() - 6) % 7          # Sunday = 6
    sunday = (local - timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)
    if days_back == 0 and local.hour < SUNDAY_DEADLINE_HOUR:
        sunday -= timedelta(days=7)
    return sunday


def heartbeat_problems(cfg: Config, season: int, now: datetime | None = None) -> list[str]:
    """One problem string when the season is on and no successful weekly run is recorded since
    the last Sunday window; empty otherwise. Never raises."""
    from cfbr.bets.closing import cached_games_frame

    now = now or datetime.now(UTC)
    try:
        games = cached_games_frame(cfg.paths.raw_dir, season)
    except Exception as e:  # noqa: BLE001
        return [f"heartbeat check could not read the cached games: {type(e).__name__}: {e}"]
    if not season_active(games, now):
        return []
    since = expected_since(now)
    last = last_success(cfg)
    if last is None:
        return [f"no successful `cfbr weekly` run recorded at all ({cfg.paths.out_dir / STAMP} missing)"]
    if last < since:
        return [f"no successful `cfbr weekly` run since {since:%a %Y-%m-%d} (last {last:%Y-%m-%d %H:%M} UTC); "
                "is the Sunday cron firing?"]
    return []


def daily_check(cfg: Config, season: int, now: datetime | None = None) -> list[str]:
    """Everything `cfbr check` reports: the heartbeat plus the backup age."""
    from cfbr.bets.health import backup_problems

    now = now or datetime.now(UTC)
    problems = heartbeat_problems(cfg, season, now)
    try:
        problems += backup_problems(cfg, now)
    except Exception as e:  # noqa: BLE001
        problems.append(f"backup check failed: {type(e).__name__}: {e}")
    return problems
