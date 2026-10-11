"""Opponent-adjusted efficiency via ridge regression, shrunk toward a prior.

    y_play = c + O[offense] - D[defense] + H * home_indicator + e

Fit on residuals from the prior (y - (O0[off] - D0[def])), so the ridge pulls
each team toward its prior instead of toward zero. The system is small
(2 x teams + 2 columns), so we solve the normal equations directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp

UNPENALIZED_EPS = 1e-6


@dataclass
class EffFit:
    teams: list[str]
    o: np.ndarray
    d: np.ndarray
    home: float
    intercept: float
    n_off: np.ndarray  # plays on offense per team (for prior share)
    n_def: np.ndarray

    def frame(self, prefix: str = "") -> pl.DataFrame:
        return pl.DataFrame({
            "team": self.teams, f"{prefix}o": self.o, f"{prefix}d": self.d,
            f"{prefix}n_off": self.n_off, f"{prefix}n_def": self.n_def,
        })


def home_indicator(offense: pl.Series, defense: pl.Series, home_team: pl.Series, neutral: pl.Series) -> np.ndarray:
    h = np.where(offense.to_numpy() == home_team.to_numpy(), 1.0,
                 np.where(defense.to_numpy() == home_team.to_numpy(), -1.0, 0.0))
    return np.where(neutral.to_numpy(), 0.0, h)


def fit_efficiency(
    plays: pl.DataFrame,
    target: str,
    alpha: float,
    teams: list[str],
    prior_o: np.ndarray | None = None,
    prior_d: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    center_on: list[str] | None = None,
) -> EffFit:
    """Fit O/D for ``teams`` from ``plays`` (needs offense, defense, home_team,
    neutral_site and ``target`` columns). ``teams`` must contain every entity
    appearing in plays; teams with no plays keep their prior.

    ``center_on`` re-centers O and D to mean zero over those teams (FBS), with
    the intercept absorbing the shift, so 0 means an average FBS unit.
    """
    idx = {t: i for i, t in enumerate(teams)}
    off = plays["offense"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
    de = plays["defense"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
    h = home_indicator(plays["offense"], plays["defense"], plays["home_team"], plays["neutral_site"])
    y = plays[target].cast(pl.Float64).to_numpy()
    center = np.isin(np.array(teams, dtype=object), center_on) if center_on else None
    return fit_efficiency_arrays(off, de, h, y, teams, alpha, prior_o, prior_d, weights, center)


def fit_efficiency_arrays(
    off: np.ndarray,
    de: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    teams: list[str],
    alpha: float,
    prior_o: np.ndarray | None = None,
    prior_d: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    center: np.ndarray | None = None,
) -> EffFit:
    """Array form of :func:`fit_efficiency`: ``off``/``de`` are team indices,
    ``h`` the home indicator, ``center`` a boolean mask of FBS teams."""
    n = len(teams)
    p_o = np.zeros(n) if prior_o is None else np.asarray(prior_o, dtype=float)
    p_d = np.zeros(n) if prior_d is None else np.asarray(prior_d, dtype=float)

    m = len(y)
    if m:
        y = y - (p_o[off] - p_d[de])
        w = np.ones(m) if weights is None else np.asarray(weights, dtype=float)
        rows = np.repeat(np.arange(m), 4)
        cols = np.column_stack([off, n + de, np.full(m, 2 * n), np.full(m, 2 * n + 1)]).ravel()
        vals = np.column_stack([np.ones(m), -np.ones(m), h, np.ones(m)]).ravel()
        X = sp.csr_matrix((vals, (rows, cols)), shape=(m, 2 * n + 2))
        Xw = X.multiply(w[:, None]).tocsr()
        A = (X.T @ Xw).toarray()
        b = Xw.T @ y
        n_off = np.bincount(off, minlength=n).astype(float)
        n_def = np.bincount(de, minlength=n).astype(float)
    else:
        A = np.zeros((2 * n + 2, 2 * n + 2))
        b = np.zeros(2 * n + 2)
        n_off = np.zeros(n)
        n_def = np.zeros(n)

    reg = np.concatenate([np.full(2 * n, alpha), [UNPENALIZED_EPS, UNPENALIZED_EPS]])
    A[np.diag_indices_from(A)] += reg
    coef = np.linalg.solve(A, b)

    o = p_o + coef[:n]
    d = p_d + coef[n:2 * n]
    home, intercept = float(coef[2 * n]), float(coef[2 * n + 1])
    if center is not None and center.any():
        mo, md = o[center].mean(), d[center].mean()
        o, d = o - mo, d - md
        intercept += mo - md
    return EffFit(teams, o, d, home, intercept, n_off, n_def)


def recency_weights(kickoff_ns: np.ndarray, as_of_ns: int, half_life_weeks: float) -> np.ndarray | None:
    """Exponential decay by age at the as-of time; ``None`` when disabled."""
    if not half_life_weeks or half_life_weeks <= 0:
        return None
    age_weeks = (as_of_ns - kickoff_ns) / (7 * 86400 * 1e9)
    return np.power(0.5, age_weeks / half_life_weeks)
