"""Backtest metrics."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

EPS = 1e-6


def week_bucket() -> pl.Expr:
    return (
        pl.when(pl.col("season_type") == "postseason").then(pl.lit("post"))
        .when(pl.col("week") <= 3).then(pl.lit("w01-03"))
        .when(pl.col("week") <= 7).then(pl.lit("w04-07"))
        .otherwise(pl.lit("w08+"))
    )


def wilson_ci(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def consensus_spread(lines: pl.DataFrame, method: str = "median") -> pl.DataFrame:
    """Home-perspective closing spread per game (negative = home favored)."""
    if method != "median":
        raise ValueError(f"unsupported spread_consensus {method!r}")
    return (lines.filter(pl.col("spread").is_not_null())
            .group_by("game_id").agg(spread=pl.col("spread").median(), n_providers=pl.len()))


def summarize(df: pl.DataFrame) -> dict:
    """Metrics over completed games in ``df`` (needs margin, pred_margin,
    win_prob, massey_pred, prior_pred, spread)."""
    d = df.filter(pl.col("margin").is_not_null())
    if d.height == 0:
        return {"n_games": 0}
    y = d["margin"].to_numpy()
    pred = d["pred_margin"].to_numpy()
    p = np.clip(d["win_prob"].to_numpy(), EPS, 1 - EPS)
    won = (y > 0).astype(float)
    decided = y != 0
    out = {
        "n_games": int(d.height),
        "mae": float(np.mean(np.abs(pred - y))),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "log_loss": float(-np.mean(won[decided] * np.log(p[decided]) + (1 - won[decided]) * np.log(1 - p[decided]))),
        "brier": float(np.mean((p[decided] - won[decided]) ** 2)),
        "su_accuracy": float(np.mean((pred[decided] > 0) == (y[decided] > 0))),
        "massey_mae": float(np.mean(np.abs(d["massey_pred"].to_numpy() - y))),
        "prior_mae": float(np.mean(np.abs(d["prior_pred"].to_numpy() - y))),
    }
    lined = d.filter(pl.col("spread").is_not_null())
    out["n_lined"] = int(lined.height)
    if lined.height:
        ly = lined["margin"].to_numpy()
        vegas = -lined["spread"].to_numpy()
        lp = lined["pred_margin"].to_numpy()
        out["model_mae_lined"] = float(np.mean(np.abs(lp - ly)))
        out["line_mae"] = float(np.mean(np.abs(vegas - ly)))
        # ATS: model side vs the line; pushes and exact agreement excluded.
        result = np.sign(ly - vegas)
        pick = np.sign(lp - vegas)
        mask = (result != 0) & (pick != 0)
        wins = int(np.sum(result[mask] == pick[mask]))
        n = int(mask.sum())
        lo, hi = wilson_ci(wins, n)
        out |= {"ats_n": n, "ats_wins": wins, "ats_win_rate": wins / n if n else math.nan,
                "ats_ci95_low": lo, "ats_ci95_high": hi}
    return out


def report(preds: pl.DataFrame) -> dict:
    preds = preds.with_columns(bucket=week_bucket())
    res = {"overall": summarize(preds), "by_season": {}, "by_bucket": {}, "by_season_bucket": {}}
    for (s,), g in preds.group_by("season", maintain_order=True):
        res["by_season"][str(s)] = summarize(g)
    for (b,), g in preds.group_by("bucket", maintain_order=True):
        res["by_bucket"][b] = summarize(g)
    for (s, b), g in preds.group_by("season", "bucket", maintain_order=True):
        res["by_season_bucket"][f"{s}:{b}"] = summarize(g)
    res["by_season"] = dict(sorted(res["by_season"].items()))
    res["by_bucket"] = dict(sorted(res["by_bucket"].items()))
    return res
