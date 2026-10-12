"""Fresher Vegas lines for the site's unplayed games, from the Odds API snapshots.

``cfbr weekly`` (Sundays) builds the site's ``games.spread`` from CFBD's lines; ``cfbr
odds-snapshot`` (Tue-Sat, near kickoffs) saves Odds API snapshots for paper-bet closes. Between
them the site's line for a Saturday game could be six days old. :func:`apply_lines` overlays the
newest Odds API consensus onto unplayed games, both in ``cfbr web build`` and right after each
snapshot (:func:`refresh_lines`).

``games.parquet`` keeps each game's CFBD line in ``cfbd_spread``, so every refresh recomputes
``spread`` from scratch: a game the newest snapshot no longer carries falls back to CFBD instead
of keeping a stale Odds API line.

Display only: lines are never model inputs, nothing here touches ratings, predictions or a
ledger, and finished games keep their closing line from CFBD.
"""

from __future__ import annotations

import fcntl
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from cfbr.config import Config
from cfbr.ingest.odds import latest_odds_snapshot, odds_frame, team_aliases

log = logging.getLogger(__name__)


def odds_consensus(raw_dir: Path, season: int, games: pl.DataFrame) -> tuple[datetime, pl.DataFrame] | None:
    """(snapshot time, game_id + spread) from the newest Odds API snapshot: the median home
    spread across every book that posted the game, offshore included, the same consensus the
    paper bets use. ``games`` needs game_id, home_team, away_team, kickoff_ts. None without a
    snapshot."""
    snap = latest_odds_snapshot(raw_dir, season)
    if snap is None:
        return None
    ts, events = snap
    teams_path = raw_dir / "teams_fbs" / str(season) / "season.json"
    aliases = team_aliases(json.loads(teams_path.read_text())) if teams_path.exists() else {}
    books = odds_frame(events, games.select("game_id", "home_team", "away_team", "kickoff_ts"), aliases)
    return ts, books.group_by("game_id").agg(pl.col("spread").median())


def apply_lines(games: pl.DataFrame, fresh: tuple[datetime, pl.DataFrame] | None,
                cfbd_as_of: str | None) -> pl.DataFrame:
    """Set ``spread`` from ``cfbd_spread`` (CFBD's line), then, for games still unplayed when
    the snapshot was taken, from the Odds API consensus in ``fresh`` where it has the game.
    Adds per-game ``line_source`` ("odds"/"cfbd") and ``line_as_of`` (ISO time). A game that
    kicked off before the snapshot keeps CFBD's line (a snapshot after kickoff carries live
    lines); finished games keep CFBD's closing line."""
    has_cfbd = pl.col("cfbd_spread").is_not_null()
    out = games.with_columns(spread=pl.col("cfbd_spread"),
                             line_source=pl.when(has_cfbd).then(pl.lit("cfbd")),
                             line_as_of=pl.when(has_cfbd).then(pl.lit(cfbd_as_of)))
    if fresh is None:
        return out
    ts, cons = fresh
    out = out.join(cons.rename({"spread": "_fresh"}), on="game_id", how="left")
    # kickoff_ts is local time during a build and UTC once stored; compare in UTC either way.
    kickoff = pl.col("kickoff_ts").dt.convert_time_zone("UTC")
    use = (~pl.col("completed")) & (kickoff > ts.astimezone(UTC)) & pl.col("_fresh").is_not_null()
    return out.with_columns(
        spread=pl.when(use).then(pl.col("_fresh")).otherwise(pl.col("spread")),
        line_source=pl.when(use).then(pl.lit("odds")).otherwise(pl.col("line_source")),
        line_as_of=pl.when(use).then(pl.lit(ts.isoformat())).otherwise(pl.col("line_as_of")),
    ).drop("_fresh")


STALE_GAP = timedelta(hours=1)


def older_line(line_as_of: str | None, lines_as_of: str | None) -> bool:
    """Whether a game's line is meaningfully older than the page's "as of" time: an hour or
    more (CFBD and Odds API snapshots in the same run land seconds apart)."""
    if not line_as_of or not lines_as_of:
        return False
    try:
        return datetime.fromisoformat(lines_as_of) - datetime.fromisoformat(line_as_of) >= STALE_GAP
    except ValueError:
        return False


def refresh_lines(cfg: Config, season: int) -> int | None:
    """Re-apply the newest Odds API lines to ``out/web/{season}/games.parquet``, then rewrite
    ``meta.json`` last (the server's reload signal), as ``web build`` does. Takes the weekly lock
    (non-blocking) so it never interleaves with a rebuild or a deploy. Returns None when it did
    nothing (lock held, no web data, data built before ``cfbd_spread`` existed, or no snapshot),
    else how many games now show an Odds API line."""
    from cfbr.web.build import web_dir

    d = web_dir(cfg) / str(season)
    games_path, meta_path = d / "games.parquet", d / "meta.json"
    if not games_path.exists() or not meta_path.exists():
        return None
    with open(cfg.paths.db_path.parent / ".weekly.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("weekly lock held; lines refresh skipped (the next snapshot or rebuild applies it)")
            return None
        meta = json.loads(meta_path.read_text())
        games = pl.read_parquet(games_path)
        if "cfbd_spread" not in games.columns:
            log.info("web data for %d predates line refresh; rebuild it once (cfbr web build)", season)
            return None
        fresh = odds_consensus(cfg.paths.raw_dir, season, games)
        if fresh is None:
            return None
        out = apply_lines(games.drop([c for c in ("line_source", "line_as_of") if c in games.columns]),
                          fresh, meta.get("cfbd_lines_as_of"))
        tmp = d / ".games.parquet.tmp"
        out.write_parquet(tmp)
        tmp.replace(games_path)
        meta["lines_as_of"] = fresh[0].isoformat()
        tmp = d / ".meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        tmp.replace(meta_path)
    n = out.filter(pl.col("line_source") == "odds").height
    log.info("lines refresh %d: %d games on Odds API lines as of %s", season, n, fresh[0].isoformat())
    return n
