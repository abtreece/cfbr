"""Fetch CFBD datasets into the raw cache.

Cache layout: data/raw/{endpoint}/{year}/{key}.json

Completed seasons use fixed keys (``season``, ``both``, ``regular_w05``) so a
re-run is served entirely from cache. For the in-progress season, the
season-level files that change week to week (games, lines) are keyed by the
ingest week (``both_w07``) so a re-run of the same week also makes zero calls.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from cfbr.ingest.client import CFBDClient

log = logging.getLogger(__name__)

# endpoint -> required-year-only, fetched once per season
SEASON_ENDPOINTS = ["/teams/fbs", "/calendar", "/player/returning", "/talent", "/recruiting/teams"]


def current_season(now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    return now.year if now.month >= 8 else now.year - 1


def fetch_season_level(client: CFBDClient, year: int, live_week: int | None = None) -> None:
    for ep in SEASON_ENDPOINTS:
        client.get(ep, {"year": year}, year, "season")
    key = "both" if live_week is None else f"both_w{live_week:02d}"
    client.get("/games", {"year": year, "seasonType": "both"}, year, key)
    if live_week is None:
        client.get("/lines", {"year": year, "seasonType": "both"}, year, key)
    else:
        fetch_live_lines(client, year, key)
    fetch_rankings(client, year, live_week)


def fetch_live_lines(client: CFBDClient, year: int, key: str) -> list:
    """Live-season lines are re-fetched on every run (lines move and new games
    get posted between the 9 AM and 3 PM runs), and each fetch is also kept as
    a timestamped snapshot under lines/{year}/snapshots/ for the bet tracker."""
    params = {"year": year, "seasonType": "both"}
    was = client.refresh
    client.refresh = True
    try:
        data = client.get("/lines", params, year, key)
    finally:
        client.refresh = was
    snap_dir = client.cache_path(client.raw_dir, "/lines", year, key).parent / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(data))
    return data


def latest_lines_snapshot(client_or_raw, year: int):
    """(captured_at, lines) for the newest live-season snapshot, or None."""
    raw = getattr(client_or_raw, "raw_dir", client_or_raw)
    snaps = sorted((raw / "lines" / str(year) / "snapshots").glob("*.json"))
    if not snaps:
        return None
    ts = datetime.strptime(snaps[-1].stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return ts, json.loads(snaps[-1].read_text())


def fetch_rankings(client: CFBDClient, year: int, live_week: int | None = None) -> None:
    """All of a season's polls (AP, Coaches, CFP committee) in one call.

    Poll weeks run one ahead of game weeks: the poll released after week N's
    games is labeled week N+1. Polls come out Sunday afternoon (CFP: Tuesday
    night), so in a live season a cached file missing the week N+1 AP poll is
    re-fetched on the next run (one call per run until it appears).
    """
    if live_week is None:
        client.get("/rankings", {"year": year, "seasonType": "both"}, year, "both")
        return
    key = f"both_w{live_week:02d}"
    params = {"year": year, "seasonType": "both"}
    data = client.get("/rankings", params, year, key)
    if not has_poll(data, "AP Top 25", live_week + 1) and not client.refresh:
        client.refresh = True
        try:
            data = client.get("/rankings", params, year, key)
        finally:
            client.refresh = False
        if not has_poll(data, "AP Top 25", live_week + 1):
            log.warning("week %d AP poll not released yet; rankings will show the latest available poll",
                        live_week + 1)


def has_poll(weeks: list, poll: str, week: int) -> bool:
    return any(w["seasonType"] == "regular" and int(w["week"]) == week and any(p["poll"] == poll for p in w["polls"])
               for w in weeks)


# Player-level data for the roster-based prior: season s's prior uses the
# season-s roster (published preseason) and season s-1 player production.
PLAYER_SEASON_ENDPOINTS = ["/ppa/players/season", "/stats/player/season"]


def fetch_player_data(client: CFBDClient, year: int, live: bool, live_week: int | None = None) -> None:
    """Roster for ``year``; player production only for completed seasons.

    ``roster/{year}/season.json`` is the first snapshot and never changes: the
    model (roster prior) uses it, so roster updates can't move a prior mid-season.
    In a live season, ``live_week`` also stores a weekly snapshot
    (``roster/{year}/wNN.json``, one call per week) for display.
    """
    client.get("/roster", {"year": year}, year, "season")
    if live and live_week is not None:
        client.get("/roster", {"year": year}, year, f"w{live_week:02d}")
    if not live:
        for ep in PLAYER_SEASON_ENDPOINTS:
            client.get(ep, {"year": year}, year, "season")


def completed_play_weeks(games: list) -> list[tuple[str, int]]:
    """(seasonType, week) pairs in which every game involving an FBS team is
    completed. Fetching plays only for finished weeks keeps a mid-week run
    from caching a partial week forever."""
    status: dict[tuple[str, int], bool] = {}
    for g in games:
        if g["seasonType"] not in ("regular", "postseason"):
            continue
        if "fbs" not in (g.get("homeClassification"), g.get("awayClassification")):
            continue
        key = (g["seasonType"], int(g["week"]))
        status[key] = status.get(key, True) and bool(g.get("completed"))
    return sorted((k for k, done in status.items() if done), key=lambda k: (k[0] != "regular", k[1]))


def _season_games(client: CFBDClient, year: int) -> list:
    games = latest_cached(client, "/games", year)
    return games or []


def fetch_plays(client: CFBDClient, year: int, season_type: str, week: int) -> None:
    client.get(
        "/plays",
        {"year": year, "week": week, "seasonType": season_type},
        year,
        f"{season_type}_w{week:02d}",
    )


def fetch_game_players(client: CFBDClient, year: int, season_type: str, week: int) -> None:
    """Per-game player box scores (who played; QB pass attempts), one call per week."""
    client.get(
        "/games/players",
        {"year": year, "week": week, "seasonType": season_type},
        year,
        f"{season_type}_w{week:02d}",
    )


def backfill(client: CFBDClient, first: int, last: int, recruiting_lookback: int) -> None:
    """Full historical pull. Small season-level endpoints first, then plays.

    Resumable: everything already cached is skipped, and QuotaExceeded stops
    the run cleanly with partial progress kept on disk. For the in-progress
    season, use ``weekly`` instead (backfill only pulls its finished weeks
    from whatever games snapshot is cached).
    """
    for y in range(first - recruiting_lookback, first):
        client.get("/recruiting/teams", {"year": y}, y, "season")
    live = current_season()
    for year in range(first, last + 1):
        if year < live:
            fetch_season_level(client, year)
    for year in range(first, last + 1):
        fetch_player_data(client, year, live=year >= live)
    for year in range(first, last + 1):
        for st, wk in completed_play_weeks(_season_games(client, year)):
            fetch_plays(client, year, st, wk)
    for year in range(first, last + 1):
        for st, wk in completed_play_weeks(_season_games(client, year)):
            fetch_game_players(client, year, st, wk)


def weekly(client: CFBDClient, year: int, week: int) -> None:
    """Incremental ingest through regular-season ``week`` of an in-progress
    season: refresh games/lines (keyed by week), then any finished weeks' plays
    not yet cached. About 4 calls per week on a cold cache (games, lines,
    polls, plays); a re-run costs 1 (lines are always re-fetched for the bet
    tracker) plus 1 while that week's polls are pending."""
    fetch_season_level(client, year, live_week=week)
    fetch_player_data(client, year, live=True, live_week=week)
    for st, wk in completed_play_weeks(_season_games(client, year)):
        if st == "regular" and wk > week:
            continue
        fetch_plays(client, year, st, wk)
        fetch_game_players(client, year, st, wk)
    unfinished = {wk for wk in range(1, week + 1)} - {
        wk for st, wk in completed_play_weeks(_season_games(client, year)) if st == "regular"}
    if unfinished:
        log.warning("season %d weeks %s not finished; plays not fetched yet", year, sorted(unfinished))


def latest_cached(client: CFBDClient, endpoint: str, year: int) -> list | None:
    """Most complete cached season-level games/lines file for a year."""
    d = client.cache_path(client.raw_dir, endpoint, year, "x").parent
    if not d.exists():
        return None
    if (d / "both.json").exists():
        return json.loads((d / "both.json").read_text())
    weekly_files = sorted(d.glob("both_w*.json"))
    return json.loads(weekly_files[-1].read_text()) if weekly_files else None
