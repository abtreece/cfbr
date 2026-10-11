"""Shadow comparison of the published prior (v1) against the roster prior.

Both are point-in-time predictions for every game of the season, under
otherwise identical frozen settings. Only the published config feeds the
rankings; the roster predictions exist only in these comparison files.
The decision rule is in DECISIONS.md ("Prior shadow comparison").
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import polars as pl

from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.model.engine import Engine

log = logging.getLogger(__name__)

WINDOW_SEASON = 2026
WINDOW_FIRST_WEEK = 7          # regular-season week 7 onward, plus the postseason
GAIN_CI_FLOOR = -0.10          # adopt roster only if the 95% CI lower bound clears this


def in_window() -> pl.Expr:
    return (pl.col("season") == WINDOW_SEASON) & (
        (pl.col("season_type") == "postseason")
        | ((pl.col("season_type") == "regular") & (pl.col("week") >= WINDOW_FIRST_WEEK)))


def comparison(cfg: Config, ds: Dataset, season: int, last_completed: int) -> pl.DataFrame:
    shadow_cfg = cfg.model_copy(deep=True)
    shadow_cfg.prior.features = "roster"
    base = Engine(ds, cfg, last_completed=last_completed).predict(season)
    alt = Engine(ds, shadow_cfg, last_completed=last_completed).predict(season)
    return (base.select("game_id", "season", "week", "season_type", "kickoff_ts", "completed", "home_team",
                        "away_team", "margin", pred_v1="pred_margin")
            .join(alt.select("game_id", pred_roster="pred_margin"), on="game_id", how="left")
            .with_columns(in_window=in_window())
            .sort("kickoff_ts", "game_id"))


def paired_gain(df: pl.DataFrame) -> dict:
    d = df.filter(pl.col("completed") & pl.col("margin").is_not_null() & pl.col("pred_roster").is_not_null())
    if d.height == 0:
        return {"n": 0, "gain": math.nan, "ci95": (math.nan, math.nan), "mae_v1": math.nan, "mae_roster": math.nan}
    e1 = (d["pred_v1"] - d["margin"]).abs().to_numpy()
    e2 = (d["pred_roster"] - d["margin"]).abs().to_numpy()
    x = e1 - e2
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else math.nan
    return {"n": int(len(x)), "gain": float(x.mean()), "ci95": (float(x.mean() - half), float(x.mean() + half)),
            "mae_v1": float(e1.mean()), "mae_roster": float(e2.mean())}


def summary_markdown(df: pl.DataFrame, season: int) -> str:
    def row(label: str, sub: pl.DataFrame) -> str:
        g = paired_gain(sub)
        if g["n"] == 0:
            return f"| {label} | 0 | – | – | – |"
        return (f"| {label} | {g['n']} | {g['mae_v1']:.2f} | {g['mae_roster']:.2f} | "
                f"{g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")

    w = paired_gain(df.filter(pl.col("in_window")))
    if w["n"] == 0:
        status = "No window games completed yet."
    else:
        passing = w["gain"] > 0 and w["ci95"][0] > GAIN_CI_FLOOR
        status = (f"Window so far: gain {w['gain']:+.3f} (95% CI lower bound {w['ci95'][0]:+.3f}). "
                  + ("**Currently meets** the adoption rule." if passing else "**Does not currently meet** the adoption rule.")
                  + " The decision is made once, before 2027 week 1.")
    lines = [
        f"# {season} prior comparison: v1 (published) vs roster (shadow)", "",
        "Both columns are point-in-time predictions with every other setting frozen. Gain = v1 error − roster error "
        "(positive favors roster). Rule (DECISIONS.md): adopt roster before 2027 week 1 if the window gain is > 0 and "
        f"its 95% CI lower bound is > {GAIN_CI_FLOOR:+.2f}.", "",
        status, "",
        "| games | n | v1 MAE | roster MAE | gain [95% CI] |", "|---|---|---|---|---|",
        row("**window (wk 7+ & postseason)**", df.filter(pl.col("in_window"))),
        row("weeks 1–6 (used to build the rule; excluded)", df.filter(~pl.col("in_window"))),
        "",
    ]
    nxt = df.filter(~pl.col("completed") & pl.col("in_window")).head(80)
    if nxt.height:
        lines += ["## Upcoming window games", "", "| kickoff (UTC) | game | v1 home margin | roster home margin | diff |",
                  "|---|---|---|---|---|"]
        for r in nxt.iter_rows(named=True):
            diff = None if r["pred_roster"] is None else r["pred_roster"] - r["pred_v1"]
            lines.append(f"| {r['kickoff_ts']:%Y-%m-%d %H:%M} | {r['away_team']} @ {r['home_team']} | "
                         f"{r['pred_v1']:+.1f} | {'–' if r['pred_roster'] is None else f'{r['pred_roster']:+.1f}'} | "
                         f"{'–' if diff is None else f'{diff:+.1f}'} |")
        lines.append("")
    return "\n".join(lines)


def run_shadow(cfg: Config, ds: Dataset, season: int, last_completed: int) -> Path:
    df = comparison(cfg, ds, season, last_completed)
    out = cfg.paths.out_dir / str(season)
    out.mkdir(parents=True, exist_ok=True)
    csv, md = out / "prior_comparison.csv", out / "prior_comparison.md"
    tmp = csv.with_suffix(".csv.tmp")
    df.write_csv(tmp)
    tmp.replace(csv)
    tmp = md.with_suffix(".md.tmp")
    tmp.write_text(summary_markdown(df, season))
    tmp.replace(md)
    w = paired_gain(df.filter(pl.col("in_window")))
    log.info("prior shadow: %d window games settled, gain %s", w["n"],
             "n/a" if w["n"] == 0 else f"{w['gain']:+.3f} [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}]")
    return csv
