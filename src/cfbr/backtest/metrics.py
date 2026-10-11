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


# CFBD "providers" that are not sportsbooks. numberfire is a projection model,
# not a market price, and is never used. teamrankings and "consensus" aggregate
# sportsbook lines; they are used only when no sportsbook line exists for a game.
NON_BOOKS = {"consensus", "teamrankings", "numberfire"}
SOURCE_TIERS = [("books", None), ("consensus", "consensus"), ("teamrankings", "teamrankings")]


def consensus_spread(lines: pl.DataFrame, method: str = "median", col: str = "spread") -> pl.DataFrame:
    """Home-perspective market spread per game (negative = home favored).

    Median across real sportsbooks when any posted one; otherwise CFBD's
    ``consensus`` aggregate; otherwise teamrankings. numberfire is never used.
    ``source`` says which tier priced each game.
    """
    if method != "median":
        raise ValueError(f"unsupported spread_consensus {method!r}")
    prov = pl.col("provider").str.to_lowercase()
    l = lines.filter(pl.col(col).is_not_null()).with_columns(
        tier=pl.when(~prov.is_in(list(NON_BOOKS))).then(0)
        .when(prov == "consensus").then(1)
        .when(prov == "teamrankings").then(2)
        .otherwise(None)
    ).filter(pl.col("tier").is_not_null())
    best = l.group_by("game_id").agg(tier=pl.col("tier").min())
    names = {i: name for i, (name, _) in enumerate(SOURCE_TIERS)}
    return (l.join(best, on=["game_id", "tier"])
            .group_by("game_id").agg(pl.col(col).median().alias(col), n_providers=pl.len(), tier=pl.col("tier").first())
            .with_columns(source=pl.col("tier").replace_strict(names, return_dtype=pl.Utf8))
            .drop("tier"))


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
    res["prior_week1_mae_by_season"] = {
        str(s): float((g["prior_pred"] - g["margin"]).abs().mean())
        for (s,), g in preds.filter((pl.col("slot") == "r01") & pl.col("margin").is_not_null())
        .group_by("season", maintain_order=True)
    }
    res["prior_week1_mae_by_season"] = dict(sorted(res["prior_week1_mae_by_season"].items()))
    res["prior_influence_by_slot"] = [
        {"slot": r["slot"], "mean_prior_share": r["share"], "mae": r["mae"], "prior_only_mae": r["prior_mae"],
         "n_games": r["n"]}
        for r in preds.filter(pl.col("margin").is_not_null())
        .group_by("slot", "slot_order")
        .agg(share=((pl.col("prior_share_h") + pl.col("prior_share_a")) / 2).mean(),
             mae=(pl.col("pred_margin") - pl.col("margin")).abs().mean(),
             prior_mae=(pl.col("prior_pred") - pl.col("margin")).abs().mean(), n=pl.len())
        .sort("slot_order").iter_rows(named=True)
    ]
    res["by_season"] = dict(sorted(res["by_season"].items()))
    res["by_bucket"] = dict(sorted(res["by_bucket"].items()))
    return res
