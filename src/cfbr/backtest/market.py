"""Market-implied power ratings: a diagnostic, not a model input (review B5).

For each prediction slot the closing consensus lines of the season's games up to and
including that slot (lines are known before kickoff) are inverted with the Massey
ridge (margin = -spread, no cap) into a "Vegas rating" per team, and compared with
the model's published ratings for the same slot. Systematic disagreements by
conference, week, FCS status or home field show up here long before they show up in
a win rate. Betting lines never feed the ratings themselves.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from cfbr.backtest.metrics import consensus_spread
from cfbr.model.engine import Engine
from cfbr.model.massey import fit_massey

NO_CAP = 1e9


def market_ratings(eng: Engine, season: int, alpha: float = 1.0, max_slots: int | None = None) -> pl.DataFrame:
    """One row per (slot, team): market rating, model rating, their difference and the
    market's home-field term, for every slot with at least 20 lined games so far."""
    cons = consensus_spread(eng.ds.lines, eng.cfg.backtest.spread_consensus).select("game_id", "spread")
    g = (eng.games.filter(pl.col("season") == season).join(cons, on="game_id", how="inner")
         .with_columns(margin=-pl.col("spread")))
    teams = eng.entities(season)
    fbs = eng.fbs(season)
    conf = eng.ds.team_status.filter(pl.col("season") == season).select("team", "conference")
    out = []
    now = datetime.now(UTC)
    for r in eng.season_slots(season).iter_rows(named=True):
        past = g.filter(pl.col("slot_order") <= r["slot_order"])
        as_of = r["as_of"] if r["as_of"].tzinfo else r["as_of"].replace(tzinfo=UTC)
        if past.height < 20 or as_of > now:   # a live season: only slots that have started
            continue
        if max_slots is not None and len(out) >= max_slots:
            break
        fit = fit_massey(past, teams, alpha, NO_CAP, fbs)
        market = pl.DataFrame({"team": fit.teams, "market": fit.rating})
        model = eng.ratings(season, as_of=r["as_of"]).select("team", "fbs", "prior_share", model="rating")
        out.append(model.join(market, on="team", how="left").with_columns(
            season=pl.lit(season, pl.Int32), slot=pl.lit(r["slot"]), slot_order=pl.lit(r["slot_order"], pl.Int32),
            n_lined=pl.lit(past.height, pl.Int32), market_home=pl.lit(fit.home), diff=pl.col("model") - pl.col("market"))
            .join(conf, on="team", how="left"))
    if not out:
        return pl.DataFrame()
    return pl.concat(out).select("season", "slot", "slot_order", "team", "conference", "fbs", "prior_share",
                                 "model", "market", "diff", "market_home", "n_lined").sort("slot_order", "team")


def summary(df: pl.DataFrame, b_home: float) -> dict:
    """Tables for the markdown report: by week, by conference, FCS pool, home field."""
    d = df.filter(pl.col("market").is_not_null())
    if d.height == 0:
        return {}
    by_slot = (d.group_by("slot", "slot_order").agg(
        n=pl.len(), corr=pl.corr("model", "market"), mean_diff=pl.col("diff").mean(),
        mean_abs=pl.col("diff").abs().mean(), sd_model=pl.col("model").std(), sd_market=pl.col("market").std(),
        market_home=pl.col("market_home").first()).sort("slot_order"))
    last = d.filter(pl.col("slot_order") == d["slot_order"].max())
    by_conf = (last.filter(pl.col("fbs")).group_by("conference").agg(
        n=pl.len(), mean_diff=pl.col("diff").mean(), mean_abs=pl.col("diff").abs().mean(),
        model=pl.col("model").mean(), market=pl.col("market").mean()).sort("mean_diff", descending=True))
    fcs = last.filter(~pl.col("fbs")).select("team", "model", "market", "diff")
    worst = last.filter(pl.col("fbs")).sort("diff").select("team", "conference", "model", "market", "diff")
    return {"by_slot": by_slot, "by_conference": by_conf, "fcs": fcs, "most_under": worst.head(10),
            "most_over": worst.tail(10).reverse(), "b_home": b_home, "last_slot": str(last["slot"][0])}


def _t(df: pl.DataFrame, cols: list[str], fmt: dict[str, str]) -> list[str]:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows(named=True):
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cells.append("–")
            elif isinstance(v, float):
                cells.append(format(v, fmt.get(c, ".2f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def markdown(season: int, s: dict) -> str:
    if not s:
        return f"# {season} market-implied ratings\n\nNo lined games yet.\n"
    lines = [f"# {season} market-implied ratings", "",
             "Diagnostic only. The market rating inverts the closing consensus lines of the season so far "
             "(Massey ridge on −spread, no cap) into a rating per team; the model rating is the published "
             "point-in-time rating for the same slot. diff = model − market (points). Lines never feed the model.", "",
             "## By slot", "",
             *_t(s["by_slot"], ["slot", "n", "corr", "mean_diff", "mean_abs", "sd_model", "sd_market", "market_home"],
                 {"corr": ".3f"}), "",
             f"Home field: market {float(s['by_slot']['market_home'][-1]):.2f} pts at the last slot vs the model's "
             f"calibrated home constant {s['b_home']:.2f}.", "",
             f"## By conference (after {s['last_slot']})", "",
             *_t(s["by_conference"], ["conference", "n", "model", "market", "mean_diff", "mean_abs"], {}), ""]
    if s["fcs"].height:
        lines += ["## Pooled FCS", "", *_t(s["fcs"], ["team", "model", "market", "diff"], {}), ""]
    lines += [f"## Model most below the market (after {s['last_slot']})", "",
              *_t(s["most_under"], ["team", "conference", "model", "market", "diff"], {}), "",
              f"## Model most above the market (after {s['last_slot']})", "",
              *_t(s["most_over"], ["team", "conference", "model", "market", "diff"], {}), ""]
    return "\n".join(lines)


def run_market_ratings(eng: Engine, season: int, out_dir: Path) -> Path:
    df = market_ratings(eng, season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"market_ratings_{season}.parquet"
    if df.height:
        df.write_parquet(path)
    (out_dir / f"market_ratings_{season}.md").write_text(markdown(season, summary(df, eng.calibration(season).b_home)))
    return path
