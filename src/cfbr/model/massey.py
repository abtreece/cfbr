"""Score-only Massey/SRS baseline: ridge regression on capped game margins.

    margin_home = H * home_indicator + R[home] - R[away] + e
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


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
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    for hm, aw, margin, neutral in games.select("home", "away", "margin", "neutral_site").iter_rows():
        x = np.zeros(n + 1)
        x[idx[hm]] += 1.0
        x[idx[aw]] -= 1.0
        x[n] = 0.0 if neutral else 1.0
        y = float(np.clip(margin, -cap, cap))
        A += np.outer(x, x)
        b += x * y
    A[np.arange(n), np.arange(n)] += alpha
    A[n, n] += 1e-6
    coef = np.linalg.solve(A, b)
    r = coef[:n]
    if center_on:
        mask = np.array([t in set(center_on) for t in teams])
        if mask.any():
            r = r - r[mask].mean()
    return MasseyFit(teams, r, float(coef[n]))
