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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from cfbr.config import Config
from cfbr.ingest.odds import odds_frame, team_aliases


def _snapshots(raw_dir: Path, season: int) -> list[tuple[datetime, Path]]:
    return [(datetime.strptime(p.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC), p)
            for p in sorted((raw_dir / "odds" / str(season) / "snapshots").glob("*.json"))]


def snapshot_closing(cfg: Config, games: pl.DataFrame, season: int) -> pl.DataFrame:
    """game_id, closing_line, closing_books, closing_best_home, closing_best_away for ``games``
    (game_id, home_team, away_team, kickoff_ts) that have a qualifying pre-kickoff snapshot.
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
    pick: dict[Path, list[int]] = {}
    for gid, kick in games.select("game_id", "kickoff_ts").iter_rows():
        kick = kick.astimezone(UTC) if kick.tzinfo else kick.replace(tzinfo=UTC)
        before = [(ts, p) for ts, p in snaps if kick - max_age <= ts < kick]
        if before:
            pick.setdefault(before[-1][1], []).append(gid)
    teams_path = cfg.paths.raw_dir / "teams_fbs" / str(season) / "season.json"
    aliases = team_aliases(json.loads(teams_path.read_text())) if teams_path.exists() else {}
    parts = []
    for path, gids in pick.items():
        f = odds_frame(json.loads(path.read_text()), games.filter(pl.col("game_id").is_in(gids)), aliases)
        parts.append(f.filter(pl.col("game_id").is_in(gids)).with_columns(_ok=priceable).group_by("game_id").agg(
            closing_line=pl.col("spread").median(), closing_books=pl.len().cast(pl.Int32),
            closing_best_home=pl.col("spread").filter(pl.col("_ok")).max(),
            closing_best_away=pl.col("spread").filter(pl.col("_ok")).min()))
    return pl.concat(parts).cast(schema) if parts else pl.DataFrame(schema=schema)


def games_kicking_off(db_path: Path, season: int, within_hours: float, now: datetime | None = None) -> int:
    """How many not-yet-started games kick off in the next ``within_hours`` (from the
    local store; no API call)."""
    import duckdb

    now = now or datetime.now(UTC)
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute("SELECT count(*) FROM games WHERE season = ? AND NOT completed "
                           "AND kickoff_ts > ? AND kickoff_ts <= ?",
                           [season, now, now + timedelta(hours=within_hours)]).fetchone()[0]
