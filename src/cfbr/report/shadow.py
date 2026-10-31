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


# -- Totals v2 shadow (rule: DECISIONS.md "Point totals v2") ---------------------------

def totals_gain(df: pl.DataFrame) -> dict:
    """Paired total MAE gain of v2 over the points-for/against baseline (positive favors v2)."""
    d = df.filter(pl.col("completed") & pl.col("total").is_not_null() & pl.col("v2_total").is_not_null()
                  & pl.col("base_total").is_not_null())
    if d.height == 0:
        return {"n": 0, "gain": math.nan, "ci95": (math.nan, math.nan), "mae_base": math.nan, "mae_v2": math.nan}
    e1 = (d["base_total"] - d["total"]).abs().to_numpy()
    e2 = (d["v2_total"] - d["total"]).abs().to_numpy()
    x = e1 - e2
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else math.nan
    return {"n": int(len(x)), "gain": float(x.mean()), "ci95": (float(x.mean() - half), float(x.mean() + half)),
            "mae_base": float(e1.mean()), "mae_v2": float(e2.mean())}


def totals_comparison(ds: Dataset, cfg: Config, season: int, last_completed: int) -> pl.DataFrame:
    from cfbr.backtest.metrics import consensus_spread
    from cfbr.model.totals import TotalsV2

    eng = Engine(ds, cfg, last_completed=last_completed)
    df = TotalsV2(eng).predict_season(season)
    line = consensus_spread(ds.lines, cfg.backtest.spread_consensus, col="over_under").select(
        "game_id", line_total="over_under")
    return df.join(line, on="game_id", how="left").with_columns(in_window=in_window())


def totals_markdown(df: pl.DataFrame, season: int) -> str:
    w = totals_gain(df.filter(pl.col("in_window")))
    if w["n"] == 0:
        status = "No window games completed yet."
    else:
        ok = w["gain"] > 0 and w["ci95"][0] > GAIN_CI_FLOOR
        status = (f"Window so far: gain {w['gain']:+.3f} (95% CI lower bound {w['ci95'][0]:+.3f}). "
                  + ("**Currently meets**" if ok else "**Does not currently meet**")
                  + " the publish rule. The decision is made once, after the 2026 postseason.")
    lw = df.filter(pl.col("in_window") & pl.col("completed") & pl.col("line_total").is_not_null()
                   & pl.col("total").is_not_null())
    line_txt = (f"Closing total on the same window games: MAE {(lw['line_total'] - lw['total']).abs().mean():.2f} vs "
                f"v2 {(lw['v2_total'] - lw['total']).abs().mean():.2f} (reported, not gated)." if lw.height else "")
    rows = ["| games | n | baseline MAE | v2 MAE | gain [95% CI] |", "|---|---|---|---|---|"]
    for label, sub in (("**window (wk 7+ & postseason)**", df.filter(pl.col("in_window"))),
                       ("weeks 1–6 (outside the window)", df.filter(~pl.col("in_window")))):
        g = totals_gain(sub)
        rows.append(f"| {label} | {g['n']} | – | – | – |" if g["n"] == 0 else
                    f"| {label} | {g['n']} | {g['mae_base']:.2f} | {g['mae_v2']:.2f} | "
                    f"{g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")
    return "\n".join([
        f"# {season} totals shadow: v2 points ratings vs baseline", "",
        "Point-in-time predicted totals; nothing here is published. Gain = baseline error − v2 error (positive favors v2). "
        f"Rule (DECISIONS.md): publish v2 totals after the 2026 postseason if the window gain is > 0 and its 95% CI lower "
        f"bound is > {GAIN_CI_FLOOR:+.2f}.", "", status, "", line_txt, "", *rows, ""])


def run_totals_shadow(cfg: Config, ds: Dataset, season: int, last_completed: int) -> Path:
    df = totals_comparison(ds, cfg, season, last_completed)
    out = cfg.paths.out_dir / str(season)
    out.mkdir(parents=True, exist_ok=True)
    csv, md = out / "totals_comparison.csv", out / "totals_comparison.md"
    tmp = csv.with_suffix(".csv.tmp")
    df.write_csv(tmp)
    tmp.replace(csv)
    tmp = md.with_suffix(".md.tmp")
    tmp.write_text(totals_markdown(df, season))
    tmp.replace(md)
    w = totals_gain(df.filter(pl.col("in_window")))
    log.info("totals shadow: %d window games settled, gain %s", w["n"],
             "n/a" if w["n"] == 0 else f"{w['gain']:+.3f} [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}]")
    return csv


# -- Calibration v2 shadow (rule: DECISIONS.md "M2+M3 calibration v2") ------------------

def calibration_comparison(eng: Engine, season: int) -> pl.DataFrame:
    """Published (base calibration) vs calibration v2 game predictions, point-in-time. ``eng`` is the
    published engine; v2 is its second stage applied on top, so no extra ridge fits are needed."""
    from cfbr.model.calibrate import stage2_design

    g, base = eng.base_predictions(season)
    v2 = base + stage2_design(g, base) @ eng.stage2(season)
    return (g.select("game_id", "season", "week", "season_type", "kickoff_ts", "completed", "home_team",
                     "away_team", "margin")
            .with_columns(pred_base=pl.Series(base), pred_v2=pl.Series(v2), in_window=in_window())
            .sort("kickoff_ts", "game_id"))


def calibration_gain(df: pl.DataFrame) -> dict:
    """Paired MAE gain of v2 over the base calibration (positive favors v2)."""
    d = df.filter(pl.col("completed") & pl.col("margin").is_not_null())
    if d.height == 0:
        return {"n": 0, "gain": math.nan, "ci95": (math.nan, math.nan), "mae_base": math.nan, "mae_v2": math.nan}
    e1 = (d["pred_base"] - d["margin"]).abs().to_numpy()
    e2 = (d["pred_v2"] - d["margin"]).abs().to_numpy()
    x = e1 - e2
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else math.nan
    return {"n": int(len(x)), "gain": float(x.mean()), "ci95": (float(x.mean() - half), float(x.mean() + half)),
            "mae_base": float(e1.mean()), "mae_v2": float(e2.mean())}


def calibration_markdown(df: pl.DataFrame, season: int) -> str:
    w = calibration_gain(df.filter(pl.col("in_window")))
    if w["n"] == 0:
        status = "No window games completed yet."
    else:
        ok = w["gain"] > 0 and w["ci95"][0] > GAIN_CI_FLOOR
        status = (f"Window so far: gain {w['gain']:+.3f} (95% CI lower bound {w['ci95'][0]:+.3f}). "
                  + ("**Currently meets**" if ok else "**Does not currently meet**")
                  + " the adoption rule. The decision is made once, before 2027 week 1.")
    rows = ["| games | n | base MAE | v2 MAE | gain [95% CI] |", "|---|---|---|---|---|"]
    for label, sub in (("**window (wk 7+ & postseason)**", df.filter(pl.col("in_window"))),
                       ("weeks 1–6 (outside the window)", df.filter(~pl.col("in_window")))):
        g = calibration_gain(sub)
        rows.append(f"| {label} | {g['n']} | – | – | – |" if g["n"] == 0 else
                    f"| {label} | {g['n']} | {g['mae_base']:.2f} | {g['mae_v2']:.2f} | "
                    f"{g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")
    return "\n".join([
        f"# {season} calibration shadow: base (published) vs v2", "",
        "v2 adds a second stage to each game prediction: slope terms that depend on how much of the two ratings "
        "is still prior, a home term scaled the same way, and a pooled-FCS term. Ratings are unchanged; only game "
        "predictions differ, and nothing here is published. Gain = base error − v2 error (positive favors v2). "
        f"Rule (DECISIONS.md): adopt v2 before 2027 week 1 if the window gain is > 0 and its 95% CI lower bound is > "
        f"{GAIN_CI_FLOOR:+.2f}.", "", status, "", *rows, ""])


def run_calibration_shadow(eng: Engine, season: int) -> Path:
    df = calibration_comparison(eng, season)
    out = eng.cfg.paths.out_dir / str(season)
    out.mkdir(parents=True, exist_ok=True)
    csv, md = out / "calibration_comparison.csv", out / "calibration_comparison.md"
    tmp = csv.with_suffix(".csv.tmp")
    df.write_csv(tmp)
    tmp.replace(csv)
    tmp = md.with_suffix(".md.tmp")
    tmp.write_text(calibration_markdown(df, season))
    tmp.replace(md)
    w = calibration_gain(df.filter(pl.col("in_window")))
    log.info("calibration shadow: %d window games settled, gain %s", w["n"],
             "n/a" if w["n"] == 0 else f"{w['gain']:+.3f} [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}]")
    return csv


def paired_mae_gain(df: pl.DataFrame, a: str, b: str) -> dict:
    """Paired MAE gain of column b over column a on settled games (positive favors b)."""
    d = df.filter(pl.col("completed") & pl.col("margin").is_not_null()
                  & pl.col(a).is_not_null() & pl.col(b).is_not_null())
    if d.height == 0:
        return {"n": 0, "gain": math.nan, "ci95": (math.nan, math.nan), "mae_a": math.nan, "mae_b": math.nan}
    ea = (d["margin"] - d[a]).abs().to_numpy()
    eb = (d["margin"] - d[b]).abs().to_numpy()
    x = ea - eb
    half = 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else math.nan
    return {"n": int(len(x)), "gain": float(x.mean()), "ci95": (float(x.mean() - half), float(x.mean() + half)),
            "mae_a": float(ea.mean()), "mae_b": float(eb.mean())}


# -- Portal-era roster prior shadow (rule: DECISIONS.md "Roster prior, portal-era variant") --

PORTAL_MIN_SEASON = 2021       # roster regression trains and applies from this season on


def in_window_portal() -> pl.Expr:
    return ((pl.col("season") == 2026) & (pl.col("season_type") == "postseason")) | (pl.col("season") == 2027)


def portal_comparison(cfg: Config, ds: Dataset, season: int, last_completed: int,
                      published: pl.DataFrame) -> pl.DataFrame:
    """Published model and plain roster prior (from run_shadow's CSV) plus the portal-era variant."""
    portal_cfg = cfg.model_copy(deep=True)
    portal_cfg.prior.features = "roster"
    portal_cfg.prior.roster_min_season = PORTAL_MIN_SEASON
    alt = Engine(ds, portal_cfg, last_completed=last_completed).predict(season)
    return (published.rename({"pred_v1": "pred_published"})
            .join(alt.select("game_id", pred_portal="pred_margin"), on="game_id", how="left")
            .with_columns(in_window=in_window_portal())
            .sort("kickoff_ts", "game_id"))


def portal_markdown(df: pl.DataFrame, season: int) -> str:
    def row(label: str, sub: pl.DataFrame, a: str, b: str) -> str:
        g = paired_mae_gain(sub, a, b)
        if g["n"] == 0:
            return f"| {label} | 0 | – | – | – |"
        return (f"| {label} | {g['n']} | {g['mae_a']:.2f} | {g['mae_b']:.2f} | "
                f"{g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")

    win = df.filter(pl.col("in_window"))
    w = paired_mae_gain(win, "pred_published", "pred_portal")
    r = paired_mae_gain(win, "pred_roster", "pred_portal")
    if w["n"] == 0:
        status = "No window games completed yet (the window opens with the 2026 postseason)."
    else:
        passing = (w["gain"] > 0 and w["ci95"][0] > GAIN_CI_FLOOR and r["n"] > 0 and r["gain"] > 0)
        status = (f"Window so far: gain vs published {w['gain']:+.3f} (95% CI lower bound {w['ci95'][0]:+.3f}), "
                  f"gain vs plain roster {r['gain']:+.3f}. "
                  + ("**Currently meets** the adoption rule." if passing else "**Does not currently meet** the adoption rule.")
                  + " The decision is made once, before 2028 week 1.")
    lines = [
        f"# {season} prior comparison: published vs roster vs portal-era roster (shadow)", "",
        "All columns are point-in-time predictions with every other setting frozen. The portal variant trains and "
        f"applies the roster regression only from {PORTAL_MIN_SEASON} on. Gain = first error − second error "
        "(positive favors the second). Rule (DECISIONS.md): adopt portal before 2028 week 1 if its window gain vs "
        f"the published model is > 0 with 95% CI lower bound > {GAIN_CI_FLOOR:+.2f} and it also beats the plain "
        "roster prior over the window.", "",
        status, "",
        "| window games | n | first MAE | second MAE | gain [95% CI] |", "|---|---|---|---|---|",
        row("**portal vs published (primary)**", win, "pred_published", "pred_portal"),
        row("portal vs plain roster (must be > 0)", win, "pred_roster", "pred_portal"),
        row("plain roster vs published (context)", win, "pred_published", "pred_roster"),
        "",
    ]
    return "\n".join(lines)


def run_portal_shadow(cfg: Config, ds: Dataset, season: int, last_completed: int) -> Path:
    out = cfg.paths.out_dir / str(season)
    src = out / "prior_comparison.csv"
    if not src.exists():
        raise RuntimeError(f"{src} missing; run_shadow writes it first")
    published = pl.read_csv(src, try_parse_dates=True)
    df = portal_comparison(cfg, ds, season, last_completed, published)
    csv, md = out / "prior_portal_comparison.csv", out / "prior_portal_comparison.md"
    tmp = csv.with_suffix(".csv.tmp")
    df.write_csv(tmp)
    tmp.replace(csv)
    tmp = md.with_suffix(".md.tmp")
    tmp.write_text(portal_markdown(df, season))
    tmp.replace(md)
    w = paired_mae_gain(df.filter(pl.col("in_window")), "pred_published", "pred_portal")
    log.info("portal prior shadow: %d window games settled, gain %s", w["n"],
             "n/a" if w["n"] == 0 else f"{w['gain']:+.3f} [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}]")
    return csv
