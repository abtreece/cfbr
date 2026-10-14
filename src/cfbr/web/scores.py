"""Midweek final scores for the site, between Sunday's ``cfbr weekly`` runs.

``cfbr weekly`` is the only ingest that reaches the store, so without this a Tuesday game
would show as unplayed until Sunday (and as "postponed" from Saturday evening, once it is
STALE_GAME past kickoff). ``cfbr scores`` (cron, nightly) makes one CFBD ``/games`` call when
a game on the site should be over but isn't marked final, saves the response as
``data/raw/games/{season}/scores.json`` and fills in those games' scores in
``out/web/{season}/games.parquet`` (:func:`apply_scores`). ``web build`` applies the same file,
so a rebuild between Sundays keeps them.

Display only. ``scores.json`` is outside the files the store reads (``both.json`` /
``both_w*.json``), so it never reaches a model or the paper bets: ratings stay "after week N"
until Sunday's run, and only that run settles bets. The overlay only fills in games the site
has as not completed; it never changes a score the store already has.
"""

from __future__ import annotations

import fcntl
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from cfbr.config import Config
from cfbr.data import STALE_GAME

log = logging.getLogger(__name__)

# How long after kickoff a game is assumed over and worth a CFBD call for.
GAME_LENGTH = timedelta(hours=3, minutes=30)


def scores_path(raw_dir: Path, season: int) -> Path:
    return raw_dir / "games" / str(season) / "scores.json"


def read_scores(raw_dir: Path, season: int) -> list | None:
    p = scores_path(raw_dir, season)
    return json.loads(p.read_text()) if p.exists() else None


def due_games(games: pl.DataFrame, now: datetime) -> pl.DataFrame:
    """Games the site still has as not completed whose kickoff was GAME_LENGTH to STALE_GAME ago:
    over by now unless postponed. Older ones are the site's postponed/cancelled games."""
    kickoff = pl.col("kickoff_ts").dt.convert_time_zone("UTC")
    return games.filter(~pl.col("completed") & (kickoff <= now - GAME_LENGTH) & (kickoff > now - STALE_GAME))


def apply_scores(games: pl.DataFrame, rows: list | None) -> pl.DataFrame:
    """``games`` with CFBD's final scores (``rows``: a /games response) filled in for games not
    yet completed: completed, home_points, away_points, margin. Games CFBD doesn't have as
    completed with both scores, and games already completed, are unchanged."""
    final = [{"game_id": g["id"], "_hp": g["homePoints"], "_ap": g["awayPoints"]} for g in rows or []
             if g.get("completed") and g.get("homePoints") is not None and g.get("awayPoints") is not None]
    if not final:
        return games
    f = pl.DataFrame(final, schema={"game_id": pl.Int64, "_hp": pl.Int64, "_ap": pl.Int64})
    out = games.join(f, on="game_id", how="left")
    fill = ~pl.col("completed") & pl.col("_hp").is_not_null()
    return out.with_columns(
        completed=pl.col("completed") | fill,
        home_points=pl.when(fill).then(pl.col("_hp")).otherwise(pl.col("home_points")).cast(games.schema["home_points"]),
        away_points=pl.when(fill).then(pl.col("_ap")).otherwise(pl.col("away_points")).cast(games.schema["away_points"]),
        margin=pl.when(fill).then((pl.col("_hp") - pl.col("_ap")).cast(pl.Float64))
        .otherwise(pl.col("margin")).cast(games.schema["margin"]),
    ).drop("_hp", "_ap")


def refresh_scores(cfg: Config, season: int, fetch: Callable[[], list],
                   now: datetime | None = None) -> tuple[int, int] | None:
    """Fill in final scores on the site for games that should be over. ``fetch`` makes the CFBD
    call (and saves the response to :func:`scores_path`); it runs only when a game is due, and
    only while holding the weekly lock (non-blocking, as ``refresh_lines`` does), so it never
    spends a call while ``cfbr weekly`` or a deploy is writing the site. Rewrites
    ``games.parquet`` then ``meta.json`` (the server's reload signal). Returns None when no
    call was made (no web data, a finished season, nothing due, lock held), else
    (games due, games filled in)."""
    from cfbr.web.build import web_dir

    now = now or datetime.now(UTC)
    d = web_dir(cfg) / str(season)
    games_path, meta_path = d / "games.parquet", d / "meta.json"
    if not games_path.exists() or not meta_path.exists():
        return None
    with open(cfg.paths.db_path.parent / ".weekly.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("weekly lock held; scores refresh skipped (the next run or rebuild applies it)")
            return None
        meta = json.loads(meta_path.read_text())
        if not meta.get("live", True):
            return None
        games = pl.read_parquet(games_path)
        due = due_games(games, now)
        if due.height == 0:
            return None
        rows = fetch()
        out = apply_scores(games, rows)
        filled = out.filter(pl.col("completed")).height - games.filter(pl.col("completed")).height
        if filled:
            tmp = d / ".games.parquet.tmp"
            out.write_parquet(tmp)
            tmp.replace(games_path)
            meta["scores_as_of"] = now.isoformat(timespec="seconds")
            tmp = d / ".meta.json.tmp"
            tmp.write_text(json.dumps(meta, indent=2, default=str))
            tmp.replace(meta_path)
    still = due.join(out.filter(~pl.col("completed")).select("game_id"), on="game_id", how="semi")
    for r in still.iter_rows(named=True):
        log.info("not final on CFBD yet: %s @ %s (kickoff %s)", r["away_team"], r["home_team"], r["kickoff_ts"])
    log.info("scores refresh %d: %d game(s) due, %d filled in", season, due.height, filled)
    return due.height, filled
