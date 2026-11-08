"""Opponent-adjusted efficiency via ridge regression, shrunk toward a prior.

    y_play = c + O[offense] - D[defense] + H * home_indicator + e

Fit on residuals from the prior (y - (O0[off] - D0[def])), so the ridge pulls
each team toward its prior instead of toward zero. The system is small
(2 x teams + 2 columns), so we solve the normal equations directly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    # Only with keep_posterior=True (uncertainty): the penalized normal-equation
    # matrix A = XᵀWX + diag(reg), the penalty vector, and the play residuals.
    precision: np.ndarray | None = None
    reg: np.ndarray | None = None
    resid: np.ndarray | None = None

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
    keep_posterior: bool = False,
) -> EffFit:
    """Array form of :func:`fit_efficiency`: ``off``/``de`` are team indices,
    ``h`` the home indicator, ``center`` a boolean mask of FBS teams.

    ``keep_posterior`` also returns what the Gaussian posterior needs
    (see model/uncertainty.py); the point estimates are identical either way."""
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
    resid = (y - X @ coef) if (keep_posterior and m) else (np.zeros(0) if keep_posterior else None)
    if center is not None and center.any():
        mo, md = o[center].mean(), d[center].mean()
        o, d = o - mo, d - md
        intercept += mo - md
    if keep_posterior:
        return EffFit(teams, o, d, home, intercept, n_off, n_def, precision=A, reg=reg, resid=resid)
    return EffFit(teams, o, d, home, intercept, n_off, n_def)


FCS_LEVEL_PENALTY = 1e-6  # x alpha: keeps the shared FCS levels defined before any FCS play


def fit_fcs_stage2(
    off: np.ndarray,
    de: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    fcs: np.ndarray,
    alpha: float,
    stage1: EffFit,
    prior_o: np.ndarray | None = None,
    prior_d: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> EffFit:
    """Stage 2 of the FCS two-stage fit (DECISIONS "FCS games, two-stage fit").

    Rates pooled FCS entities (``fcs`` mask over teams) from plays with an FCS side, holding
    Stage 1's FBS ratings, home coefficient and intercept fixed:
        FCS offense:  y = c + H h + O[g] - D1[fbs]
        FCS defense:  y = c + H h + O1[fbs] - D[g]
    with O[g] = prior_o[g] + dL_o + dev_o[g] (same for D). dL (shared by all FCS groups) has a
    negligible penalty; dev has ``alpha``. With no prior (final()), groups shrink toward the FCS
    average, not toward an average FBS team. Returns a copy of ``stage1`` with FCS rows replaced."""
    n = len(stage1.teams)
    groups = np.flatnonzero(fcs)
    o, d = stage1.o.copy(), stage1.d.copy()
    if groups.size == 0:
        return replace(stage1, o=o, d=d)
    p_o = np.zeros(n) if prior_o is None else np.asarray(prior_o, dtype=float)
    p_d = np.zeros(n) if prior_d is None else np.asarray(prior_d, dtype=float)
    g = groups.size
    pos = np.full(n, -1)
    pos[groups] = np.arange(g)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    so = np.flatnonzero(fcs[off] & ~fcs[de])   # FCS on offense
    sd = np.flatnonzero(fcs[de] & ~fcs[off])   # FCS on defense
    c, hh = stage1.intercept, stage1.home
    target = np.concatenate([
        y[so] - (c + hh * h[so] - stage1.d[de[so]]) - p_o[off[so]],
        y[sd] - (c + hh * h[sd] + stage1.o[off[sd]]) + p_d[de[sd]],
    ])
    # Columns: [dL_o, dL_d, dev_o (g), dev_d (g)]
    X = np.zeros((so.size + sd.size, 2 + 2 * g))
    X[:so.size, 0] = 1.0
    X[np.arange(so.size), 2 + pos[off[so]]] = 1.0
    X[so.size:, 1] = -1.0
    X[so.size + np.arange(sd.size), 2 + g + pos[de[sd]]] = -1.0
    ww = np.concatenate([w[so], w[sd]])
    A = X.T @ (X * ww[:, None])
    b = X.T @ (ww * target)
    A[np.diag_indices_from(A)] += np.r_[[FCS_LEVEL_PENALTY * alpha] * 2, np.full(2 * g, alpha)]
    coef = np.linalg.solve(A, b)
    o[groups] = p_o[groups] + coef[0] + coef[2:2 + g]
    d[groups] = p_d[groups] + coef[1] + coef[2 + g:]
    return replace(stage1, o=o, d=d)


def recency_weights(kickoff_ns: np.ndarray, as_of_ns: int, half_life_weeks: float) -> np.ndarray | None:
    """Exponential decay by age at the as-of time; ``None`` when disabled."""
    if not half_life_weeks or half_life_weeks <= 0:
        return None
    age_weeks = (as_of_ns - kickoff_ns) / (7 * 86400 * 1e9)
    return np.power(0.5, age_weeks / half_life_weeks)


def play_weights(a, k: int, as_of_ns: int | None, m) -> np.ndarray | None:
    """Per-play weights for the first ``k`` plays of season arrays ``a`` (Engine.SeasonArrays):
    recency decay at ``as_of_ns`` (none when it is None), times ``m.fcs_weight`` for plays with a
    pooled FCS side when ``m.fcs_two_stage`` is on. ``None`` when nothing applies, so the fit is
    unweighted. Every fit that should match the published one calls this (engine, uncertainty, matchup)."""
    w = None if as_of_ns is None else recency_weights(a.kickoff_ns[:k], as_of_ns, m.half_life_weeks)
    if m.fcs_two_stage and m.fcs_weight != 1.0:
        fcs = ~a.center[a.off[:k]] | ~a.center[a.de[:k]]
        factor = np.where(fcs, m.fcs_weight, 1.0)
        w = factor if w is None else w * factor
    return w
