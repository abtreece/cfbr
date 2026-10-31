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


def opening_spread(lines: pl.DataFrame, method: str = "median") -> pl.DataFrame:
    """Home-perspective opening spread per game: ``consensus_spread`` applied to CFBD's
    ``spreadOpen``. Only sportsbooks carry an opener (the aggregators' column is null), so this
    is a books-only median. Columns: game_id, spread_open, n_providers_open."""
    return (consensus_spread(lines, method, "spread_open")
            .select("game_id", "spread_open", n_providers_open="n_providers"))


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


# -- diagnostics (review M5) ---------------------------------------------------------
# Patterns that the headline MAE hides and that would have shown the calibration-v2
# effects (prior-share slope, week-1 bias) at tuning time.

DIAG_WEEK_BUCKETS = ["w01", "w02", "w03-04", "w05-08", "w09-12", "w13+", "post"]
DIAG_SHARE_BUCKETS = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]
DIAG_EDGE_BUCKETS = ["0-3", "3-6", "6-10", "10+"]
# Would-be bet threshold for the opener block: the paper-bet rule's Small tier (config
# [bets.tiers]); diagnostics take no config, so it is repeated here.
OPENER_MIN_EDGE = 3.0


def diag_week_bucket() -> pl.Expr:
    w = pl.col("week")
    return (pl.when(pl.col("season_type") == "postseason").then(pl.lit("post"))
            .when(w <= 1).then(pl.lit("w01")).when(w <= 2).then(pl.lit("w02"))
            .when(w <= 4).then(pl.lit("w03-04")).when(w <= 8).then(pl.lit("w05-08"))
            .when(w <= 12).then(pl.lit("w09-12")).otherwise(pl.lit("w13+")))


def prior_share_bucket() -> pl.Expr:
    ps = (pl.col("prior_share_h") + pl.col("prior_share_a")) / 2
    return (pl.when(ps < 0.3).then(pl.lit("<0.3")).when(ps < 0.5).then(pl.lit("0.3-0.5"))
            .when(ps < 0.7).then(pl.lit("0.5-0.7")).when(ps < 0.9).then(pl.lit("0.7-0.9"))
            .otherwise(pl.lit(">=0.9")))


def _bias_slope(d: pl.DataFrame) -> dict:
    """Bias (mean margin − pred) and the OLS slope of margin on pred (with intercept):
    slope > 1 means predictions are under-dispersed, < 1 over-dispersed."""
    y = d["margin"].to_numpy()
    x = d["pred_margin"].to_numpy()
    out = {"n": int(len(y)), "bias": float(np.mean(y - x)), "mae": float(np.mean(np.abs(y - x)))}
    if len(y) >= 3 and np.var(x) > 0:
        out["slope"] = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    else:
        out["slope"] = math.nan
    return out


def _by(d: pl.DataFrame, col: str, order: list[str]) -> list[dict]:
    rows = []
    for b in order:
        g = d.filter(pl.col(col) == b)
        if g.height:
            rows.append({"bucket": b, **_bias_slope(g)})
    return rows


def _ats_by_edge(y: np.ndarray, line: np.ndarray, pred: np.ndarray) -> list[dict]:
    """Model-side cover rate against ``line`` by |pred − line| bucket, pushes excluded, Wilson CIs."""
    edge = pred - line
    result = np.sign(y - line)
    pick = np.sign(edge)
    ok = (result != 0) & (pick != 0)
    ae = np.abs(edge)
    rows = []
    for label, lo, hi in (("0-3", 0, 3), ("3-6", 3, 6), ("6-10", 6, 10), ("10+", 10, np.inf)):
        m = ok & (ae >= lo) & (ae < hi)
        n = int(m.sum())
        wins = int(np.sum(result[m] == pick[m]))
        ci = wilson_ci(wins, n)
        rows.append({"bucket": label, "n": n, "wins": wins, "win_rate": wins / n if n else math.nan,
                     "ci95_low": ci[0], "ci95_high": ci[1]})
    return rows


def _opener(d: pl.DataFrame) -> dict | None:
    """The model against the opening line on games with both an opener and a close: each line's
    MAE, the would-be bets (|pred − opener| ≥ OPENER_MIN_EDGE) with their closing-line value
    (points the close moved toward the bet side), the share that beat the close, and how far the
    close moves toward the model per point of disagreement (OLS of close − open on pred − open,
    no intercept)."""
    o = d.filter(pl.col("spread_open").is_not_null() & pl.col("spread").is_not_null())
    if o.height < 2:
        return None
    y = o["margin"].to_numpy()
    mo = -o["spread_open"].to_numpy()
    mc = -o["spread"].to_numpy()
    pred = o["pred_margin"].to_numpy()
    edge = pred - mo
    move = mc - mo
    bet = np.abs(edge) >= OPENER_MIN_EDGE
    clv = (move * np.sign(edge))[bet]
    ss = float((edge * edge).sum())
    return {"n": int(o.height), "seasons": sorted(int(s) for s in o["season"].unique()),
            "mae_open": float(np.mean(np.abs(y - mo))), "mae_close": float(np.mean(np.abs(y - mc))),
            "mae_model": float(np.mean(np.abs(y - pred))),
            "n_bets": int(bet.sum()),
            "clv_mean": float(clv.mean()) if clv.size else math.nan,
            "clv_sd": float(clv.std(ddof=1)) if clv.size > 1 else math.nan,
            "beat_close": float(np.mean(clv > 0)) if clv.size else math.nan,
            "push_close": float(np.mean(clv == 0)) if clv.size else math.nan,
            "move_slope": float((edge * move).sum() / ss) if ss > 0 else math.nan}


def diagnostics(preds: pl.DataFrame) -> dict:
    """Bias/slope by week and prior-share bucket, ATS by edge bucket, the line+model blend
    weights and a win-probability reliability table, over completed games."""
    d = preds.filter(pl.col("margin").is_not_null())
    res: dict = {"n_games": int(d.height)}
    if d.height == 0:
        return res
    d = d.with_columns(diag_week=diag_week_bucket())
    res["by_week"] = _by(d, "diag_week", DIAG_WEEK_BUCKETS)
    if {"prior_share_h", "prior_share_a"} <= set(d.columns):
        res["by_prior_share"] = _by(d.with_columns(ps_bucket=prior_share_bucket()), "ps_bucket", DIAG_SHARE_BUCKETS)

    # Reliability: win_prob deciles vs the observed home-win rate (ties excluded).
    r = d.filter(pl.col("margin") != 0).with_columns(
        bin=(pl.col("win_prob") * 10).floor().clip(0, 9).cast(pl.Int32), won=(pl.col("margin") > 0).cast(pl.Float64))
    res["reliability"] = [
        {"bin": f"{b/10:.1f}-{(b+1)/10:.1f}", "n": r_["n"], "predicted": r_["p"], "observed": r_["o"]}
        for b, r_ in ((row["bin"], row) for row in
                      r.group_by("bin").agg(n=pl.len(), p=pl.col("win_prob").mean(), o=pl.col("won").mean())
                      .sort("bin").iter_rows(named=True))
    ]

    lined = d.filter(pl.col("spread").is_not_null()) if "spread" in d.columns else d.clear()
    if lined.height >= 3:
        y = lined["margin"].to_numpy()
        line = -lined["spread"].to_numpy()
        pred = lined["pred_margin"].to_numpy()
        # Blend: OLS of margin on (line, pred), no intercept. The model's weight is the
        # information it carries beyond the market; the paper bets implicitly bet on it.
        X = np.column_stack([line, pred])
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        res["blend"] = {"n": int(len(y)), "w_line": float(c[0]), "w_model": float(c[1]),
                        "mae_line": float(np.mean(np.abs(line - y))), "mae_model": float(np.mean(np.abs(pred - y))),
                        "mae_blend": float(np.mean(np.abs(X @ c - y)))}
        # ATS vs the closing line by |edge| bucket; pushes and exact agreement excluded.
        res["ats_by_edge"] = _ats_by_edge(y, line, pred)
    if "spread_open" in d.columns:
        op = d.filter(pl.col("spread_open").is_not_null())
        if op.height >= 2:
            res["ats_by_edge_open"] = _ats_by_edge(op["margin"].to_numpy(), -op["spread_open"].to_numpy(),
                                                   op["pred_margin"].to_numpy())
        res["opener"] = _opener(d)
    return res


def report(preds: pl.DataFrame) -> dict:
    preds = preds.with_columns(bucket=week_bucket())
    res = {"overall": summarize(preds), "by_season": {}, "by_bucket": {}, "by_season_bucket": {},
           "diagnostics": diagnostics(preds)}
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
