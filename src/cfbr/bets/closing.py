"""Closing lines from pre-kickoff Odds API snapshots.

A bet's closing line is the consensus (median home spread across every book, offshore
included, same as the bet consensus) from the latest Odds API snapshot taken before
its kickoff and no more than ``odds.closing_max_age_hours`` earlier. Lightweight
snapshots are taken by ``cfbr odds-snapshot`` shortly before kickoffs. Games without
such a snapshot fall back to CFBD's post-game consensus; each bet records which source
graded its CLV.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from cfbr.config import Config
from cfbr.ingest.odds import odds_frame, team_aliases

log = logging.getLogger(__name__)


def _snapshots(raw_dir: Path, season: int) -> list[tuple[datetime, Path]]:
    return [(datetime.strptime(p.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC), p)
            for p in sorted((raw_dir / "odds" / str(season) / "snapshots").glob("*.json"))]


def snapshot_closing(cfg: Config, games: pl.DataFrame, season: int) -> pl.DataFrame:
    """game_id, closing_line, closing_books, closing_best_home, closing_best_away for ``games``
    (game_id, home_team, away_team, kickoff_ts) that have a qualifying pre-kickoff snapshot
    (the newest one in the window that carries the game; older ones fill in when a book pulled it).
    The best closing numbers come from the pricing books only (the best-line record's close);
    they are null when none posted."""
    from cfbr.bets.ledger import book_key

    schema = {"game_id": pl.Int64, "closing_line": pl.Float64, "closing_books": pl.Int32,
              "closing_best_home": pl.Float64, "closing_best_away": pl.Float64}
    off = {book_key(b) for b in cfg.bets.offshore_books}
    allowed = {book_key(b) for b in cfg.bets.pricing_books}
    priceable = pl.col("provider").map_elements(
        lambda b: book_key(b) not in off and (not allowed or book_key(b) in allowed), return_dtype=pl.Boolean)
    snaps = _snapshots(cfg.paths.raw_dir, season)
    if not snaps or games.height == 0:
        return pl.DataFrame(schema=schema)
    max_age = timedelta(hours=cfg.odds.closing_max_age_hours)
    # Qualifying snapshots per game, newest first: a book pulling a market (or an event not
    # matching) in the last snapshot falls back to the previous one in the window.
    cands: dict[int, list[Path]] = {}
    for gid, kick in games.select("game_id", "kickoff_ts").iter_rows():
        kick = kick.astimezone(UTC) if kick.tzinfo else kick.replace(tzinfo=UTC)
        before = [p for ts, p in snaps if kick - max_age <= ts < kick]
        if before:
            cands[gid] = before[::-1]
    teams_path = cfg.paths.raw_dir / "teams_fbs" / str(season) / "season.json"
    aliases = team_aliases(json.loads(teams_path.read_text())) if teams_path.exists() else {}
    need: dict[Path, list[int]] = {}
    for gid, paths in cands.items():
        for path in paths:
            need.setdefault(path, []).append(gid)
    frames: dict[Path, pl.DataFrame] = {}

    def frame(path: Path) -> pl.DataFrame:
        if path not in frames:
            gids = need[path]
            f = odds_frame(json.loads(path.read_text()), games.filter(pl.col("game_id").is_in(gids)), aliases)
            frames[path] = f.filter(pl.col("game_id").is_in(gids))
        return frames[path]

    parts = []
    for gid, paths in cands.items():
        for i, path in enumerate(paths):
            f = frame(path).filter(pl.col("game_id") == gid)
            if f.height == 0:
                continue
            if i:
                log.info("closing line for game %d from an earlier snapshot (%s; missing from %d newer)",
                         gid, path.stem, i)
            parts.append(f.with_columns(_ok=priceable).group_by("game_id").agg(
                closing_line=pl.col("spread").median(), closing_books=pl.len().cast(pl.Int32),
                closing_best_home=pl.col("spread").filter(pl.col("_ok")).max(),
                closing_best_away=pl.col("spread").filter(pl.col("_ok")).min()))
            break
    return pl.concat(parts).cast(schema) if parts else pl.DataFrame(schema=schema)


def cached_games(raw_dir: Path, season: int) -> list | None:
    """CFBD /games rows from the newest cached season file (``both.json``, else the latest
    ``both_w*.json``), or None when nothing is cached. No API call."""
    d = raw_dir / "games" / str(season)
    if (d / "both.json").exists():
        return json.loads((d / "both.json").read_text())
    weekly = sorted(d.glob("both_w*.json"))
    return json.loads(weekly[-1].read_text()) if weekly else None


def cached_games_frame(raw_dir: Path, season: int) -> pl.DataFrame | None:
    """The cached games as the frame ``match_events`` expects (game_id, home_team, away_team,
    kickoff_ts in UTC, completed), or None when nothing is cached."""
    rows = cached_games(raw_dir, season)
    if rows is None:
        return None
    recs = [{"game_id": g["id"], "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
             "kickoff_ts": _iso_utc(g.get("startDate")), "completed": bool(g.get("completed"))} for g in rows]
    return pl.DataFrame(recs, schema={"game_id": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8,
                                      "kickoff_ts": pl.Datetime("us", "UTC"), "completed": pl.Boolean})


def _iso_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (t if t.tzinfo else t.replace(tzinfo=UTC)).astimezone(UTC)


def games_kicking_off_cached(raw_dir: Path, season: int, within_hours: float, now: datetime | None = None,
                             db_path: Path | None = None) -> int:
    """How many not-yet-started games kick off in the next ``within_hours``, from the newest
    cached CFBD games file (refreshed by every ingest, so mid-week kickoff changes count).
    Falls back to the DuckDB store when nothing is cached and ``db_path`` is given."""
    now = now or datetime.now(UTC)
    g = cached_games_frame(raw_dir, season)
    if g is None:
        return games_kicking_off(db_path, season, within_hours, now) if db_path and db_path.exists() else 0
    return g.filter(~pl.col("completed") & (pl.col("kickoff_ts") > now)
                    & (pl.col("kickoff_ts") <= now + timedelta(hours=within_hours))).height


def games_kicking_off(db_path: Path, season: int, within_hours: float, now: datetime | None = None) -> int:
    """How many not-yet-started games kick off in the next ``within_hours`` (from the
    local store; no API call)."""
    import duckdb

    now = now or datetime.now(UTC)
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute("SELECT count(*) FROM games WHERE season = ? AND NOT completed "
                           "AND kickoff_ts > ? AND kickoff_ts <= ?",
                           [season, now, now + timedelta(hours=within_hours)]).fetchone()[0]
