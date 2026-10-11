"""Precompute what the web app shows: week-by-week ratings and per-game
predictions for each season, written to ``out/web/{season}/``.

The server only reads these files; it never fits a model. Ratings "after week
N" use the same as-of rule as ``cfbr rank`` so the two always agree.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import consensus_spread
from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.ingest.fetch import current_season
from cfbr.model.engine import RATING_COLS, Engine, team_ratings_points
from cfbr.report.rank import POLLS, poll_ranks, records, week_as_of

log = logging.getLogger(__name__)

PRESEASON = 0
FINAL = 99
# A game still not completed this long after kickoff is treated as cancelled,
# so it doesn't hold its week out of the timeline forever.
STALE_GAME = timedelta(days=4)


def web_dir(cfg: Config) -> Path:
    return cfg.paths.out_dir / "web"


def _stale_cutoff(games: pl.DataFrame, now: datetime) -> pl.Expr:
    return pl.lit(now - STALE_GAME).cast(games.schema["kickoff_ts"])


def completed_weeks(eng: Engine, season: int, now: datetime) -> list[int]:
    """Regular-season weeks whose games are all final (ignoring stale, never-completed games)."""
    g = eng.games.filter((pl.col("season") == season) & (pl.col("season_type") == "regular"))
    wk = g.group_by("week").agg(
        done=pl.col("completed").sum(),
        pending=(~pl.col("completed") & (pl.col("kickoff_ts") > _stale_cutoff(g, now))).sum(),
    )
    return sorted(wk.filter((pl.col("done") > 0) & (pl.col("pending") == 0))["week"].to_list())


def season_over(eng: Engine, season: int, now: datetime) -> bool:
    g = eng.games.filter(pl.col("season") == season)
    post = g.filter(pl.col("season_type") == "postseason")
    pending = g.filter(~pl.col("completed") & (pl.col("kickoff_ts") > _stale_cutoff(g, now)))
    return post.height > 0 and pending.height == 0


def timeline(eng: Engine, season: int, now: datetime) -> list[tuple[int, str, datetime]]:
    """(week, label, as_of) points: preseason, after each completed week, final."""
    over = season_over(eng, season, now)
    pts = [(PRESEASON, "Preseason", eng.slot_as_of(season, "r01"))]
    for w in completed_weeks(eng, season, now):
        if eng.slot_as_of(season, f"r{w + 1:02d}") is not None or not over:
            as_of = week_as_of(eng, season, w)  # identical to `cfbr rank`
        else:
            # Completed season, no next regular week: stop before the bowls
            # (week_as_of would fall through to after the last game).
            later = eng.season_slots(season).filter(pl.col("slot_order") > w)
            as_of = later["as_of"].min() if later.height else week_as_of(eng, season, w)
        pts.append((w, f"Week {w}", as_of))
    if over:
        last = eng.games.filter((pl.col("season") == season) & pl.col("completed"))["kickoff_ts"].max()
        pts.append((FINAL, "Final", last + timedelta(days=1)))
    return pts


def _snapshot(eng: Engine, season: int, as_of) -> pl.DataFrame:
    """Reuse the engine's per-slot snapshot when the as-of time is a slot's."""
    slots = eng.season_slots(season).filter(pl.col("as_of") == as_of)
    if slots.height:
        return eng.snapshots(season)[slots["slot"][0]]
    return eng.snapshot(season, as_of)


def timeline_polls(polls: pl.DataFrame, season: int, week: int) -> tuple[pl.DataFrame, dict]:
    """Poll ranks shown beside a timeline point, with the poll week each one came from.

    Weekly tables use the same poll as ``cfbr rank`` (the poll released after
    week N's games). The Final table uses each poll's final postseason edition
    when there is one ("final" in the returned dict).
    """
    ranks, used = poll_ranks(polls, season, week)
    if week == FINAL:
        post = polls.filter((pl.col("season") == season) & (pl.col("season_type") == "postseason"))
        for col, name in POLLS.items():
            p = post.filter(pl.col("poll") == name)
            if p.height:
                p = p.filter(pl.col("week") == p["week"].max()).select("team", pl.col("rank").alias(col))
                ranks = ranks.drop(col, strict=False).join(p, on="team", how="full", coalesce=True)
                used[col] = "final"
    for col in POLLS:
        if col not in ranks.columns:
            ranks = ranks.with_columns(pl.lit(None, pl.Int32).alias(col))
    return ranks.select("team", *[pl.col(c).cast(pl.Int32) for c in POLLS]), used


def game_polls(polls: pl.DataFrame, season: int, games: pl.DataFrame) -> pl.DataFrame:
    """Each team's poll ranks going into the game: for week N games, the poll
    released after week N-1; for postseason games, the last regular-season poll."""
    out = []
    for slot in games["slot"].unique().to_list():
        before = int(slot[1:]) - 1 if slot.startswith("r") else FINAL
        ranks, _ = poll_ranks(polls, season, before)
        g = games.filter(pl.col("slot") == slot).select("game_id", "home_team", "away_team")
        for side in ("home", "away"):
            r = ranks.rename({c: f"{side}_{c}" for c in POLLS if c in ranks.columns})
            g = g.join(r, left_on=f"{side}_team", right_on="team", how="left")
        out.append(g)
    cols = [f"{side}_{c}" for side in ("home", "away") for c in POLLS]
    df = pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame({"game_id": []}, schema={"game_id": pl.Int64})
    for c in cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, pl.Int32).alias(c))
    return df.select("game_id", *[pl.col(c).cast(pl.Int32) for c in cols])


def season_ratings(eng: Engine, season: int, now: datetime) -> pl.DataFrame:
    cal = eng.calibration(season)
    fbs = eng.fbs(season)
    conf = eng.ds.team_status.filter(pl.col("season") == season).select("team", "conference")
    parts = []
    for week, label, as_of in timeline(eng, season, now):
        snap = _snapshot(eng, season, as_of)
        t = (team_ratings_points(snap, cal, eng.cfg)
             .join(snap.select("team", *RATING_COLS), on="team")
             .with_columns(fbs=pl.col("team").is_in(fbs)))
        rank = t.filter(pl.col("fbs")).select(
            "team", rank=pl.col("rating").rank("ordinal", descending=True).cast(pl.Int32))
        rec = records(eng, season, as_of).select("team", "w", "l")
        polls, _ = timeline_polls(eng.ds.polls, season, week)
        parts.append(t.join(rank, on="team", how="left").join(rec, on="team", how="left")
                     .join(conf, on="team", how="left").join(polls, on="team", how="left")
                     .with_columns(season=pl.lit(season, pl.Int32), week=pl.lit(week, pl.Int32),
                                   label=pl.lit(label), as_of=pl.lit(as_of),
                                   w=pl.col("w").fill_null(0), l=pl.col("l").fill_null(0)))
    df = pl.concat(parts).sort("team", "week")
    return (df.with_columns(
        rating_change=pl.col("rating") - pl.col("rating").shift(1).over("team"),
        rank_change=pl.col("rank").shift(1).over("team") - pl.col("rank"),
    ).select("season", "week", "label", "as_of", "team", "conference", "fbs", "rank", "w", "l",
             "rating", "offense", "defense", "prior_share", "rating_change", "rank_change", *POLLS, *RATING_COLS)
            .sort("week", "rank", "team", nulls_last=True))


def season_games(eng: Engine, season: int) -> pl.DataFrame:
    pts = eng.games.filter(pl.col("season") == season).select(
        "game_id", "home_points", "away_points", "home_conference", "away_conference")
    spread = consensus_spread(eng.ds.lines, eng.cfg.backtest.spread_consensus)
    g = (eng.predict(season)
         .join(pts, on="game_id", how="left")
         .join(spread.select("game_id", "spread"), on="game_id", how="left"))
    return g.join(game_polls(eng.ds.polls, season, g), on="game_id", how="left").sort("kickoff_ts", "game_id")


def build_season(eng: Engine, season: int, out_root: Path, now: datetime | None = None) -> Path:
    now = now or datetime.now(UTC)
    out = out_root / str(season)
    out.mkdir(parents=True, exist_ok=True)
    ratings = season_ratings(eng, season, now)
    games = season_games(eng, season)
    # Write to temp names then rename, so a running server never reads a half-written file.
    for name, df in (("ratings", ratings), ("games", games)):
        tmp = out / f".{name}.parquet.tmp"
        df.write_parquet(tmp)
        tmp.replace(out / f"{name}.parquet")
    meta = {
        "season": season,
        "built_at": now.isoformat(timespec="seconds"),
        "config": eng.cfg.source,
        "live": not season_over(eng, season, now),
        "weeks": sorted(set(ratings["week"].to_list())),
        "labels": dict(ratings.select("week", "label").unique().sort("week").iter_rows()),
        # Poll week shown beside each table (int, "final", or null if not released yet).
        "polls": {w: timeline_polls(eng.ds.polls, season, w)[1] for w in sorted(set(ratings["week"].to_list()))},
    }
    tmp = out / ".meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    tmp.replace(out / "meta.json")
    log.info("web data %d: %d rating rows, %d games -> %s", season, ratings.height, games.height, out)
    return out


def build(cfg: Config, seasons: list[int] | None = None, eng: Engine | None = None) -> list[Path]:
    """Build web data for ``seasons`` (default: every season that has a prior)."""
    if eng is None:
        eng = Engine(Dataset.load(cfg), cfg, last_completed=current_season() - 1)
    seasons = seasons or list(range(eng.first_eval, current_season() + 1))
    return [build_season(eng, s, web_dir(cfg)) for s in seasons]
