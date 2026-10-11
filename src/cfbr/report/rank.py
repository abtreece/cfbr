"""Weekly ranking output (Phase 7)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import consensus_spread, summarize
from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.model.engine import Engine
from scipy.stats import norm

log = logging.getLogger(__name__)


def week_as_of(eng: Engine, season: int, week: int):
    """As-of time for ratings 'after week N': the first kickoff of week N+1,
    else just after the last game played so far (end of regular season)."""
    nxt = eng.slot_as_of(season, f"r{week + 1:02d}")
    if nxt is not None:
        return nxt
    g = eng.games.filter((pl.col("season") == season) & pl.col("completed"))
    return g["kickoff_ts"].max() + timedelta(days=1)


def records(eng: Engine, season: int, as_of) -> pl.DataFrame:
    g = eng.games.filter((pl.col("season") == season) & pl.col("completed") & (pl.col("kickoff_ts") < as_of)
                         & pl.col("margin").is_not_null())
    home = g.select(team="home_team", w=(pl.col("margin") > 0).cast(pl.Int32), l=(pl.col("margin") < 0).cast(pl.Int32))
    away = g.select(team="away_team", w=(pl.col("margin") < 0).cast(pl.Int32), l=(pl.col("margin") > 0).cast(pl.Int32))
    return (pl.concat([home, away]).group_by("team").agg(pl.col("w").sum(), pl.col("l").sum())
            .with_columns(record=pl.format("{}-{}", "w", "l")))


def fbs_table(eng: Engine, season: int, as_of) -> pl.DataFrame:
    t = eng.ratings(season, as_of=as_of).filter(pl.col("fbs"))
    return t.with_columns(rank=pl.col("rating").rank("ordinal", descending=True).cast(pl.Int32))


def build_rankings(eng: Engine, season: int, week: int) -> tuple[pl.DataFrame, object]:
    as_of = week_as_of(eng, season, week)
    cur = fbs_table(eng, season, as_of)
    if week > 1:
        prev = fbs_table(eng, season, week_as_of(eng, season, week - 1)).select(
            "team", prev_rating="rating", prev_rank="rank")
    else:
        prev = fbs_table(eng, season, eng.slot_as_of(season, "r01")).select(
            "team", prev_rating="rating", prev_rank="rank")
    conf = eng.ds.team_status.filter(pl.col("season") == season).select("team", "conference")
    out = (cur.join(prev, on="team", how="left")
           .join(conf, on="team", how="left")
           .join(records(eng, season, as_of).select("team", "record"), on="team", how="left")
           .with_columns(record=pl.col("record").fill_null("0-0"),
                         rating_change=pl.col("rating") - pl.col("prev_rating"),
                         rank_change=pl.col("prev_rank") - pl.col("rank"))
           .sort("rank")
           .select("rank", "team", "conference", "record",
                   pl.col("rating").round(2), pl.col("offense").round(2).alias("offense_rating"),
                   pl.col("defense").round(2).alias("defense_rating"),
                   pl.col("rating_change").round(2), "rank_change", pl.col("prior_share").round(3)))
    return out, as_of


POLLS = {"ap_rank": "AP Top 25", "coaches_rank": "Coaches Poll", "cfp_rank": "Playoff Committee Rankings"}


def poll_ranks(polls: pl.DataFrame, season: int, week: int) -> tuple[pl.DataFrame, dict[str, int | None]]:
    """Each poll's ranks to show beside the 'after week N' ratings.

    Poll weeks run one ahead of game weeks (the poll released after week N's
    games is labeled N+1), so the target is poll week N+1. If it isn't out yet,
    the latest earlier regular-season poll is used; the returned dict gives the
    poll week actually used (None if that poll has no ranking yet this season).
    """
    target = week + 1
    reg = polls.filter((pl.col("season") == season) & (pl.col("season_type") == "regular")
                       & (pl.col("week") <= target))
    out = pl.DataFrame(schema={"team": pl.Utf8})
    used: dict[str, int | None] = {}
    for col, name in POLLS.items():
        p = reg.filter(pl.col("poll") == name)
        if p.height == 0:
            used[col] = None
            continue
        wk = p["week"].max()
        used[col] = wk
        ranks = p.filter(pl.col("week") == wk).select("team", pl.col("rank").alias(col))
        out = out.join(ranks, on="team", how="full", coalesce=True)
    return out, used


def poll_note(used: dict[str, int | None], week: int) -> str:
    labels = {"ap_rank": "AP", "coaches_rank": "Coaches", "cfp_rank": "CFP"}
    parts = []
    for col, wk in used.items():
        if wk is None:
            parts.append(f"{labels[col]}: not released yet this season")
        elif wk == week + 1:
            parts.append(f"{labels[col]}: week {wk} poll (after week {week} games)")
        else:
            parts.append(f"{labels[col]}: week {wk} poll — **week {week + 1} not released yet**")
    return "Polls shown: " + "; ".join(parts) + "."


def add_polls(rankings: pl.DataFrame, polls: pl.DataFrame, season: int, week: int):
    ranks, used = poll_ranks(polls, season, week)
    out = rankings.join(ranks, on="team", how="left")
    for col in POLLS:
        if col not in out.columns:
            out = out.with_columns(pl.lit(None, pl.Int32).alias(col))
    return out, used


def build_predictions(eng: Engine, season: int, week: int, as_of) -> pl.DataFrame:
    """Next week's games, predicted from ratings as of ``as_of``."""
    g = eng.games.filter((pl.col("season") == season) & (pl.col("slot") == f"r{week + 1:02d}"))
    if g.height == 0:
        g = eng.games.filter((pl.col("season") == season) & ~pl.col("completed") & (pl.col("kickoff_ts") >= as_of))
    snap = eng.snapshot(season, as_of)
    cal = eng.calibration(season)
    from cfbr.model.calibrate import predict_margin

    att = eng._attach(g, snap)
    pred = predict_margin(cal, att)
    spread = consensus_spread(eng.ds.lines, eng.cfg.backtest.spread_consensus)
    return (att.select("game_id", "kickoff_ts", "home_team", "away_team", "neutral_site")
            .with_columns(pred_margin=pl.Series(pred).round(1),
                          home_win_prob=pl.Series(norm.cdf(pred / cal.sigma)).round(3))
            .join(spread.select("game_id", closing_spread="spread"), on="game_id", how="left")
            .sort("kickoff_ts"))


def to_markdown(df: pl.DataFrame, title: str, top: int = 25, note: str = "") -> str:
    def table(d: pl.DataFrame) -> list[str]:
        cols = d.columns
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in d.iter_rows():
            lines.append("| " + " | ".join("" if v is None else (f"{v:+.2f}" if isinstance(v, float) and c == "rating_change"
                                                                else str(v)) for c, v in zip(cols, row)) + " |")
        return lines

    sections = [f"# {title}", ""]
    if note:
        sections += [note, ""]
    sections += ["## Top 25", "", *table(df.head(top)), ""]
    if "ap_rank" in df.columns:
        poll_cols = [c for c in POLLS if df[c].is_not_null().any()]
        if poll_cols:
            outside = df.filter((pl.col("rank") > top) & pl.any_horizontal(pl.col(c).is_not_null() for c in poll_cols))
            if outside.height:
                sections += ["## Poll-ranked teams outside the model's top 25", "",
                             *table(outside.select("rank", "team", "record", "rating", *poll_cols)), ""]
    sections += ["## All FBS teams", "", *table(df), ""]
    return "\n".join(sections)


def season_accuracy(eng: Engine, season: int, week: int) -> dict:
    preds = eng.predict(season).filter(pl.col("completed") & (pl.col("slot_order") <= week))
    preds = preds.join(consensus_spread(eng.ds.lines, eng.cfg.backtest.spread_consensus), on="game_id", how="left")
    return summarize(preds)


def append_accuracy(path: Path, season: int, week: int, m: dict) -> None:
    row = pl.DataFrame([{
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"), "season": season, "through_week": week,
        "n_games": m.get("n_games", 0), "mae": m.get("mae"), "rmse": m.get("rmse"),
        "su_accuracy": m.get("su_accuracy"), "log_loss": m.get("log_loss"), "brier": m.get("brier"),
        "n_lined": m.get("n_lined"), "model_mae_lined": m.get("model_mae_lined"), "line_mae": m.get("line_mae"),
        "ats_win_rate": m.get("ats_win_rate"), "ats_ci95_low": m.get("ats_ci95_low"),
        "ats_ci95_high": m.get("ats_ci95_high"), "massey_mae": m.get("massey_mae"),
    }])
    if path.exists():
        old = pl.read_csv(path)
        # Re-running a week replaces its line instead of duplicating it.
        old = old.filter(~((pl.col("season") == season) & (pl.col("through_week") == week)))
        row = pl.concat([old, row], how="diagonal_relaxed")
    row.sort("season", "through_week").write_csv(path)


def run_rank(cfg: Config, season: int, week: int, last_completed: int, ds: Dataset | None = None) -> Path:
    ds = ds or Dataset.load(cfg)
    eng = Engine(ds, cfg, last_completed=last_completed)
    rankings, as_of = build_rankings(eng, season, week)
    rankings, used = add_polls(rankings, ds.polls, season, week)
    note = poll_note(used, week)
    log.info("%s", note.replace("**", ""))
    out = cfg.paths.out_dir / str(season) / f"w{week:02d}"
    out.mkdir(parents=True, exist_ok=True)
    rankings.write_csv(out / "rankings.csv")
    (out / "rankings.md").write_text(
        to_markdown(rankings, f"{season} ratings after week {week} (as of {as_of:%Y-%m-%d})", note=note))
    build_predictions(eng, season, week, as_of).write_csv(out / "predictions.csv")
    log.info("wrote %s", out)
    return out
