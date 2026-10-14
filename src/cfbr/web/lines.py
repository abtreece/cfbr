"""Fresher Vegas lines for the site's unplayed games, from the Odds API snapshots.

``cfbr weekly`` (Sundays) builds the site's ``games.spread`` from CFBD's lines; ``cfbr
odds-snapshot`` (Tue-Sat, near kickoffs) saves Odds API snapshots for paper-bet closes. Between
them the site's line for a Saturday game could be six days old. :func:`apply_lines` overlays the
newest Odds API consensus onto unplayed games, both in ``cfbr web build`` and right after each
snapshot (:func:`refresh_lines`).

``games.parquet`` keeps each game's CFBD line in ``cfbd_spread``, so every refresh recomputes
``spread`` from scratch: a game the newest snapshot no longer carries falls back to CFBD instead
of keeping a stale Odds API line. ``book_lines.parquet`` is rewritten alongside it with each
book's line for those same games (:func:`book_lines`), for the shortlist's "now" numbers.

A game that kicked off after CFBD's lines were fetched has no CFBD closing line yet (CFBD's is
from before the week), so until the next ``cfbr weekly`` it shows the consensus from the newest
Odds API snapshot before its kickoff (:func:`kickoff_closes`); midweek scores
(:mod:`cfbr.web.scores`) are then graded against the line the game actually closed near.

Display only: lines are never model inputs, nothing here touches ratings, predictions or a
ledger, and finished games keep their closing line from CFBD once CFBD has one.
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


BOOK_LINES_SCHEMA = {"game_id": pl.Int64, "book": pl.Utf8, "home_spread": pl.Float64,
                     "home_price": pl.Int64, "away_price": pl.Int64, "as_of": pl.Utf8}


def _aliases(raw_dir: Path, season: int) -> dict[str, set[str]]:
    teams_path = raw_dir / "teams_fbs" / str(season) / "season.json"
    return team_aliases(json.loads(teams_path.read_text())) if teams_path.exists() else {}


def odds_books(raw_dir: Path, season: int, games: pl.DataFrame) -> tuple[datetime, pl.DataFrame] | None:
    """(snapshot time, per-book home spreads and prices) from the newest Odds API snapshot.
    ``games`` needs game_id, home_team, away_team, kickoff_ts. None without a snapshot."""
    snap = latest_odds_snapshot(raw_dir, season)
    if snap is None:
        return None
    ts, events = snap
    return ts, odds_frame(events, games.select("game_id", "home_team", "away_team", "kickoff_ts"),
                          _aliases(raw_dir, season))


CLOSES_SCHEMA = {"game_id": pl.Int64, "close": pl.Float64, "close_as_of": pl.Utf8}


def kickoff_closes(raw_dir: Path, season: int, games: pl.DataFrame, cfbd_as_of: str | None,
                   now: datetime | None = None) -> pl.DataFrame:
    """game_id, close (median home spread across books), close_as_of (snapshot time) for each
    game that kicked off after CFBD's lines were fetched, from the newest Odds API snapshot
    taken between that fetch and its kickoff that carries the game. Empty without
    ``cfbd_as_of`` (a finished season, or no live CFBD lines yet)."""
    try:
        after = datetime.fromisoformat(cfbd_as_of) if cfbd_as_of else None
    except ValueError:
        after = None
    if after is None:
        return pl.DataFrame(schema=CLOSES_SCHEMA)
    now = now or datetime.now(UTC)
    kickoff = pl.col("kickoff_ts").dt.convert_time_zone("UTC")
    g = games.filter((kickoff > after) & (kickoff <= now)).select(
        "game_id", "home_team", "away_team", kickoff.alias("kickoff_ts"))
    snaps = [(datetime.strptime(p.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC), p)
             for p in sorted((raw_dir / "odds" / str(season) / "snapshots").glob("*.json"))]
    snaps = [(ts, p) for ts, p in snaps if ts > after]
    if g.height == 0 or not snaps:
        return pl.DataFrame(schema=CLOSES_SCHEMA)
    aliases = _aliases(raw_dir, season)
    frames: dict[Path, pl.DataFrame] = {}
    rows = []
    for gid, kick in g.select("game_id", "kickoff_ts").iter_rows():
        for ts, p in reversed([(ts, p) for ts, p in snaps if ts < kick]):
            if p not in frames:
                frames[p] = odds_frame(json.loads(p.read_text()), g, aliases)
            books = frames[p].filter(pl.col("game_id") == gid)
            if books.height:
                rows.append({"game_id": gid, "close": books["spread"].median(), "close_as_of": ts.isoformat()})
                break
    return pl.DataFrame(rows, schema=CLOSES_SCHEMA)


def consensus(books: tuple[datetime, pl.DataFrame] | None) -> tuple[datetime, pl.DataFrame] | None:
    """(snapshot time, game_id + spread): the median home spread across every book that posted
    the game, offshore included, the same consensus the paper bets use."""
    if books is None:
        return None
    ts, frame = books
    return ts, frame.group_by("game_id").agg(pl.col("spread").median())


def book_lines(games: pl.DataFrame, books: tuple[datetime, pl.DataFrame] | None) -> pl.DataFrame:
    """Each book's line and prices for the games :func:`apply_lines` gave an Odds API line
    (``line_source == "odds"``), so the shortlist can say what the pick's own book offers now.
    Empty when ``books`` is None, so an older file never outlives the lines it came from."""
    if books is None:
        return pl.DataFrame(schema=BOOK_LINES_SCHEMA)
    ts, frame = books
    fresh_ids = games.filter(pl.col("line_source") == "odds").select("game_id")
    return (frame.join(fresh_ids, on="game_id", how="semi")
            .select("game_id", pl.col("provider").alias("book"), pl.col("spread").alias("home_spread"),
                    "home_price", "away_price", as_of=pl.lit(ts.isoformat()))
            .cast(BOOK_LINES_SCHEMA)
            .sort("game_id", "book"))


def newer_than(fresh: tuple[datetime, pl.DataFrame] | None,
               cfbd_as_of: str | None) -> tuple[datetime, pl.DataFrame] | None:
    """``fresh`` only if its snapshot is newer than CFBD's lines; an older Odds API snapshot
    (e.g. Sunday's Odds API fetch failed after the CFBD one succeeded) would roll lines back."""
    if fresh is None or not cfbd_as_of:
        return fresh
    try:
        return fresh if fresh[0] > datetime.fromisoformat(cfbd_as_of) else None
    except ValueError:
        return fresh


def apply_lines(games: pl.DataFrame, fresh: tuple[datetime, pl.DataFrame] | None,
                cfbd_as_of: str | None, closes: pl.DataFrame | None = None) -> pl.DataFrame:
    """Set ``spread`` from ``cfbd_spread`` (CFBD's line), then, for games still unplayed when
    the snapshot was taken, from the Odds API consensus in ``fresh`` where it has the game.
    Adds per-game ``line_source`` ("odds"/"cfbd") and ``line_as_of`` (ISO time). A game that
    kicked off before the snapshot keeps CFBD's line (a snapshot after kickoff carries live
    lines), unless ``closes`` (:func:`kickoff_closes`) has its pre-kickoff Odds API close;
    finished games keep CFBD's closing line (``closes`` only holds games CFBD has none for)."""
    has_cfbd = pl.col("cfbd_spread").is_not_null()
    out = games.with_columns(spread=pl.col("cfbd_spread"),
                             line_source=pl.when(has_cfbd).then(pl.lit("cfbd")),
                             line_as_of=pl.when(has_cfbd).then(pl.lit(cfbd_as_of)))
    if closes is not None and closes.height:
        out = out.join(closes, on="game_id", how="left")
        hit = pl.col("close").is_not_null()
        out = out.with_columns(
            spread=pl.when(hit).then(pl.col("close")).otherwise(pl.col("spread")),
            line_source=pl.when(hit).then(pl.lit("odds")).otherwise(pl.col("line_source")),
            line_as_of=pl.when(hit).then(pl.col("close_as_of")).otherwise(pl.col("line_as_of")),
        ).drop("close", "close_as_of")
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
        if not meta.get("live", True):
            return None  # a finished season keeps its closing lines
        books = odds_books(cfg.paths.raw_dir, season, games)
        fresh = newer_than(consensus(books), meta.get("cfbd_lines_as_of"))
        if fresh is None:
            return None
        closes = kickoff_closes(cfg.paths.raw_dir, season, games, meta.get("cfbd_lines_as_of"))
        out = apply_lines(games.drop([c for c in ("line_source", "line_as_of") if c in games.columns]),
                          fresh, meta.get("cfbd_lines_as_of"), closes)
        for name, df in (("games", out), ("book_lines", book_lines(out, books))):
            tmp = d / f".{name}.parquet.tmp"
            df.write_parquet(tmp)
            tmp.replace(d / f"{name}.parquet")
        meta["lines_as_of"] = fresh[0].isoformat()
        tmp = d / ".meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=2, default=str))
        tmp.replace(meta_path)
    n = out.filter(pl.col("line_source") == "odds").height
    log.info("lines refresh %d: %d games on Odds API lines as of %s", season, n, fresh[0].isoformat())
    return n
