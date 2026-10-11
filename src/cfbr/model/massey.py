"""Score-only Massey/SRS baseline: ridge regression on capped game margins.

    margin_home = H * home_indicator + R[home] - R[away] + e
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp


@dataclass
class MasseyFit:
    teams: list[str]
    rating: np.ndarray
    home: float

    def lookup(self) -> dict[str, float]:
        return dict(zip(self.teams, self.rating))


def fit_massey(
    games: pl.DataFrame, teams: list[str], alpha: float, cap: float, center_on: list[str] | None = None
) -> MasseyFit:
    """``games`` needs home, away (pooled entity names), margin, neutral_site."""
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    m = games.height
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    if m:
        hi = games["home"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
        ai = games["away"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
        home = (~games["neutral_site"]).cast(pl.Float64).to_numpy()
        y = np.clip(games["margin"].to_numpy(), -cap, cap)
        rows = np.repeat(np.arange(m), 3)
        cols = np.column_stack([hi, ai, np.full(m, n)]).ravel()
        vals = np.column_stack([np.ones(m), -np.ones(m), home]).ravel()
        X = sp.csr_matrix((vals, (rows, cols)), shape=(m, n + 1))
        A = (X.T @ X).toarray()
        b = X.T @ y
    A[np.arange(n), np.arange(n)] += alpha
    A[n, n] += 1e-6
    coef = np.linalg.solve(A, b)
    r = coef[:n]
    if center_on:
        mask = np.isin(np.array(teams, dtype=object), center_on)
        if mask.any():
            r = r - r[mask].mean()
    return MasseyFit(teams, r, float(coef[n]))
