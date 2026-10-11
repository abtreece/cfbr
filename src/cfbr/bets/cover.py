"""Calibrated cover probability for paper bets (review B4).

The model's own Φ(edge/σ) is far too confident about covering the closing line
(DECISIONS "Bets"). This is a logistic regression of *actual* covers on the size of
the edge, the size of the line and the key-number term from the points-to-cents table,
fit on the 2015–2021 walk-forward predictions and frozen here. Ridge-shrunk toward
50% with no intercept. A reporting column only in 2026; see DECISIONS "B4".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from cfbr.bets.pricing import PricingTable

FEATURES = ["abs_edge", "abs_line", "key"]


@dataclass
class CoverModel:
    mean: list[float]
    sd: list[float]
    coef: list[float]
    lam: float = 10.0
    n: int = 0
    seasons: list[int] = field(default_factory=list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = ((X - np.array(self.mean)) / np.array(self.sd)) @ np.array(self.coef)
        return 1 / (1 + np.exp(-z))

    def cover_prob(self, side: str, line: float, model_margin: float, table: PricingTable) -> float:
        return float(self.predict(features(side, line, model_margin, table)[None, :])[0])


def features(side: str, line: float, model_margin: float, table: PricingTable) -> np.ndarray:
    """(|edge|, |line|, key) for a bet on ``side`` at home spread ``line`` with the model at
    ``model_margin`` (home). key = P(cover if the model were right) − 0.5."""
    edge = model_margin + line
    p_win, p_push = table.win_push(side, line, -model_margin)
    key = p_win / (1 - p_push) - 0.5 if p_push < 1 else 0.0
    return np.array([abs(edge), abs(line), key])


def bet_frame(preds: pl.DataFrame, table: PricingTable, threshold: float = 3.0) -> pl.DataFrame:
    """Backtest predictions -> one row per would-be bet: season, cover (0/1), and the features.
    Pushes and games without a line are dropped."""
    d = (preds.filter(pl.col("completed") & pl.col("margin").is_not_null() & pl.col("spread").is_not_null())
         .with_columns(edge=pl.col("pred_margin") + pl.col("spread"))
         .filter(pl.col("edge").abs() >= threshold)
         .with_columns(side=pl.when(pl.col("edge") > 0).then(pl.lit("home")).otherwise(pl.lit("away")),
                       res=pl.col("margin") + pl.col("spread"))
         .filter(pl.col("res") != 0))
    rows = [features(s, l, m, table) for s, l, m in d.select("side", "spread", "pred_margin").iter_rows()]
    X = np.array(rows) if rows else np.zeros((0, 3))
    return d.select("season", "game_id", cover=((pl.col("res") > 0) == (pl.col("edge") > 0)).cast(pl.Float64)) \
        .with_columns(**{f: pl.Series(X[:, i]) for i, f in enumerate(FEATURES)})


def fit_cover_model(bets: pl.DataFrame, lam: float = 10.0) -> CoverModel:
    """Ridge logistic regression (no intercept, standardized features) by Newton's method."""
    X = bets.select(FEATURES).to_numpy().astype(float)
    y = bets["cover"].to_numpy().astype(float)
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mean) / sd
    w = np.zeros(Z.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + lam * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    seasons = sorted(set(bets["season"].to_list())) if "season" in bets.columns else []
    return CoverModel(mean.tolist(), sd.tolist(), w.tolist(), lam, int(len(y)), seasons)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def loso_evaluation(bets: pl.DataFrame, lam: float = 10.0) -> dict:
    """Leave-one-season-out log loss vs a constant 50%, plus reliability by predicted-prob bin."""
    out_p, out_y = [], []
    for s in sorted(set(bets["season"].to_list())):
        m = fit_cover_model(bets.filter(pl.col("season") != s), lam)
        te = bets.filter(pl.col("season") == s)
        out_p.append(m.predict(te.select(FEATURES).to_numpy().astype(float)))
        out_y.append(te["cover"].to_numpy())
    p, y = np.concatenate(out_p), np.concatenate(out_y)
    bins = np.clip(((p - 0.40) / 0.04).astype(int), 0, 5)   # <0.44, 0.44-0.48, ..., >=0.60
    rel = [{"bin": b, "n": int((bins == b).sum()), "predicted": float(p[bins == b].mean()),
            "observed": float(y[bins == b].mean())} for b in range(6) if (bins == b).any()]
    return {"n": int(len(y)), "log_loss": log_loss(p, y), "log_loss_constant": log_loss(np.full(len(y), 0.5), y),
            "reliability": rel}


def evaluate(model: CoverModel, bets: pl.DataFrame) -> dict:
    p = model.predict(bets.select(FEATURES).to_numpy().astype(float))
    y = bets["cover"].to_numpy()
    return {"n": int(len(y)), "log_loss": log_loss(p, y), "log_loss_constant": log_loss(np.full(len(y), 0.5), y),
            "mean_p": float(p.mean()), "cover_rate": float(y.mean()),
            "above_breakeven": {"n": int((p > 110 / 210).sum()),
                                "cover_rate": float(y[p > 110 / 210].mean()) if (p > 110 / 210).any() else math.nan}}


# Frozen fit (DECISIONS "B4"), 2015-2021 walk-forward predictions, 3,202 would-be bets.
# It carries no usable information (LOSO log loss 0.6933 vs 0.6931 for a constant 50%);
# kept as the pre-registered reporting column, never refit mid-season.
FROZEN = CoverModel(mean=[6.943143759910701, 15.204169269206746, 0.1633204466702035],
                    sd=[4.000749782517164, 12.19004803233711, 0.09126640667116397],
                    coef=[-0.005448757837379266, 0.043454642685659135, 0.06826646496253497],
                    lam=10.0, n=3202, seasons=[2015, 2016, 2017, 2018, 2019, 2020, 2021])
