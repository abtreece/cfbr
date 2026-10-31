"""Calibration v3 gate (DECISIONS "Calibration v3"): does a pooled-FCS slope term improve on v2?

Paired by game, leave-one-season-out on the tuning seasons under the hybrid scheme, from one
engine's base predictions (stage 2 is refit per design; the ratings are shared). This module
stays as the record of the test whether or not it passes (cf. model/home_field.py)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.ingest.fetch import current_season
from cfbr.model.calibrate import stage2_design, stage2_terms
from cfbr.model.engine import Engine

GATE_MIN_GAIN = 0.03  # points of MAE; the M4 ceiling threshold (DECISIONS)


def gate_frame(eng: Engine, seasons: list[int]) -> pl.DataFrame:
    parts = []
    for s in seasons:
        g, base = eng.base_predictions(s)
        v2 = base + stage2_design(g, base) @ eng.stage2(s, fcs_slope=False)
        v3 = base + stage2_design(g, base, fcs_slope=True) @ eng.stage2(s, fcs_slope=True)
        parts.append(g.select("game_id", "season", "week", "season_type", "completed", "neutral_site",
                              "home_fcs", "away_fcs", "margin")
                      .with_columns(pred_base=pl.Series(base), pred_v2=pl.Series(v2), pred_v3=pl.Series(v3)))
    return pl.concat(parts).filter(pl.col("completed") & pl.col("margin").is_not_null())


def paired_gain(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    """Mean of err_a − err_b (positive favours b) with a normal 95% CI."""
    d = err_a - err_b
    n = int(d.size)
    if n < 2:
        return {"n": n, "gain": math.nan, "ci95": (math.nan, math.nan)}
    m = float(d.mean())
    half = 1.96 * float(d.std(ddof=1)) / math.sqrt(n)
    return {"n": n, "gain": m, "ci95": (m - half, m + half)}


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x)) if x.size >= 3 and np.var(x) > 0 else math.nan


def gate_report(df: pl.DataFrame, eng: Engine, seasons: list[int]) -> dict:
    y = df["margin"].to_numpy()
    err = {k: np.abs(y - df[f"pred_{k}"].to_numpy()) for k in ("base", "v2", "v3")}
    primary = paired_gain(err["v2"], err["v3"])
    fcs = (df["home_fcs"] | df["away_fcs"]).fill_null(False).to_numpy()
    fcs_games = {k: {"mae": float(err[k][fcs].mean()) if fcs.any() else math.nan,
                     "slope": _slope(df[f"pred_{k}"].to_numpy()[fcs], y[fcs])} for k in err}
    coefs = {str(s): dict(zip(stage2_terms(True), (float(c) for c in eng.stage2(s, fcs_slope=True)))) for s in seasons}
    return {"seasons": seasons, "n_games": int(df.height), "n_fcs_games": int(fcs.sum()),
            "primary": {**primary, "threshold": GATE_MIN_GAIN,
                        "passes": bool(primary["gain"] >= GATE_MIN_GAIN and primary["ci95"][0] > 0)},
            "base_vs_v2": paired_gain(err["base"], err["v2"]), "base_vs_v3": paired_gain(err["base"], err["v3"]),
            "mae": {k: float(v.mean()) for k, v in err.items()}, "fcs_games": fcs_games,
            "pred_fcs_by_season": {s: c["pred_fcs"] for s, c in coefs.items()}}


def gate_markdown(r: dict) -> str:
    p = r["primary"]

    def g(x: dict) -> str:
        return f"{x['gain']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"

    lines = [f"# Calibration v3 gate: pooled-FCS slope, {r['seasons'][0]}–{r['seasons'][-1]}", "",
             f"Paired MAE gain v2 → v3 over {r['n_games']:,} games: **{g(p)}**. Rule: gain ≥ +{p['threshold']:.2f} "
             f"and 95% CI lower bound > 0 → **{'passes' if p['passes'] else 'fails'}**.", "",
             "| model | MAE all | MAE FCS games | slope on FCS games |", "|---|---|---|---|"]
    for k in ("base", "v2", "v3"):
        lines.append(f"| {k} | {r['mae'][k]:.3f} | {r['fcs_games'][k]['mae']:.3f} | {r['fcs_games'][k]['slope']:.3f} |")
    lines += ["", f"Reference: base → v2 {g(r['base_vs_v2'])}; base → v3 {g(r['base_vs_v3'])}. "
                  f"FCS games: {r['n_fcs_games']:,}.", "",
              "pred_fcs coefficient by season: " + ", ".join(f"{s}: {c:+.3f}" for s, c in r["pred_fcs_by_season"].items())]
    return "\n".join(lines) + "\n"


def run_gate(cfg: Config, ds: Dataset, seasons: list[int], tag: str = "calib-v3-gate") -> Path:
    eng = Engine(ds, cfg, last_completed=current_season() - 1)
    df = gate_frame(eng, seasons)
    rep = gate_report(df, eng, seasons)
    out = cfg.paths.results_dir / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{tag}")
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "frame.parquet")
    (out / "gate.json").write_text(json.dumps(rep, indent=2, default=float))
    (out / "gate.md").write_text(gate_markdown(rep))
    return out
