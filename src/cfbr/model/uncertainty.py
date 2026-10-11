"""Rating uncertainty, Step A: the exact Gaussian posterior of the existing fit.

The efficiency model is a ridge regression shrunk toward the preseason prior,
i.e. the posterior mode under a normal prior. Treating the recency weights as
precision multipliers, each fit's coefficient covariance is σ² A⁻¹ with
A = XᵀWX + diag(reg); the PPA and success-rate fits share plays, so their
cross-covariance is ρ σ₁ σ₂ A₁⁻¹ XᵀWX A₂⁻¹ (ρ = residual correlation). Team
net strength (O + D) is mapped to points with the calibration coefficients.

A game's predictive SD is sqrt(k² · v + s²), where v is the posterior variance
of the home-minus-away rating difference and (k, s) are fit by maximum
likelihood on training seasons (hybrid scheme). k corrects the scale of the
posterior (alpha was tuned for accuracy, not as a literal prior variance);
s is the per-game noise given the true ratings.

Interface for season simulations:
  rating_draws(eng, season, as_of, n_draws)  joint draws of every entity's rating
  game_noise(eng, season)                    home edge, per-game noise s, scale k
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import norm

from cfbr.model.engine import RATING_COLS, Engine, _ns
from cfbr.model.ridge import fit_efficiency_arrays, recency_weights
from cfbr.model.schemes import training_seasons


@dataclass
class SlotPosterior:
    teams: list[str]
    center: np.ndarray       # FBS mask
    cov_points: np.ndarray   # unscaled covariance of team net ratings, in points


def _wstats(r1: np.ndarray, r2: np.ndarray, w: np.ndarray | None) -> tuple[float, float, float]:
    w = np.ones(len(r1)) if w is None else w
    sw = w.sum()
    v1, v2 = (w * r1 * r1).sum() / sw, (w * r2 * r2).sum() / sw
    rho = (w * r1 * r2).sum() / sw / np.sqrt(v1 * v2)
    return float(v1), float(v2), float(rho)


class Posterior:
    """Per-engine cache of slot posteriors and fitted (k, s)."""

    def __init__(self, eng: Engine):
        self.eng = eng
        self._noise_stats: dict[int, tuple[float, float, float]] = {}
        self._slot: dict[tuple[int, int], SlotPosterior] = {}
        self._ks: dict[int, tuple[float, float]] = {}

    # -- play-level noise (needed when a slot has no plays yet) ----------------
    def noise_stats(self, season: int) -> tuple[float, float, float]:
        """(σ²_ppa, σ²_sr, ρ) from the previous season's full, no-prior fit."""
        if season not in self._noise_stats:
            a = self.eng.arrays(season - 1)
            m = self.eng.cfg.model
            f1 = fit_efficiency_arrays(a.off, a.de, a.home, a.ppa, a.teams, m.alpha_ppa, center=a.center,
                                       keep_posterior=True)
            f2 = fit_efficiency_arrays(a.off, a.de, a.home, a.success, a.teams, m.alpha_sr, center=a.center,
                                       keep_posterior=True)
            self._noise_stats[season] = _wstats(f1.resid, f2.resid, None)
        return self._noise_stats[season]

    def slot(self, season: int, as_of) -> SlotPosterior:
        key = (season, _ns(as_of))
        if key in self._slot:
            return self._slot[key]
        eng, m = self.eng, self.eng.cfg.model
        a = eng.arrays(season)
        k = a.n_before(as_of)
        n = len(a.teams)
        pr = pl.DataFrame({"team": a.teams}).join(eng.priors(season), on="team", how="left").fill_null(0.0)
        p = {c: pr[c].to_numpy() for c in RATING_COLS}
        w = recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)
        off, de, h = a.off[:k], a.de[:k], a.home[:k]
        f1 = fit_efficiency_arrays(off, de, h, a.ppa[:k], a.teams, m.alpha_ppa, p["o_ppa"], p["d_ppa"], w, a.center,
                                   keep_posterior=True)
        f2 = fit_efficiency_arrays(off, de, h, a.success[:k], a.teams, m.alpha_sr, p["o_sr"], p["d_sr"], w, a.center,
                                   keep_posterior=True)
        v1, v2, rho = _wstats(f1.resid, f2.resid, w) if k > 50 else self.noise_stats(season)
        B1, B2 = np.linalg.inv(f1.precision), np.linalg.inv(f2.precision)
        G = f1.precision - np.diag(f1.reg)              # XᵀWX (same design and weights in both fits)
        P = np.zeros((n, 2 * n + 2))
        P[np.arange(n), np.arange(n)] = 1.0            # O
        P[np.arange(n), n + np.arange(n)] = 1.0        # + D
        S1 = v1 * P @ B1 @ P.T
        S2 = v2 * P @ B2 @ P.T
        SX = rho * np.sqrt(v1 * v2) * P @ B1 @ G @ B2 @ P.T
        cal = eng.calibration(season)
        C = cal.b_ppa ** 2 * S1 + cal.b_sr ** 2 * S2 + cal.b_ppa * cal.b_sr * (SX + SX.T)
        out = SlotPosterior(a.teams, a.center, (C + C.T) / 2)
        self._slot[key] = out
        return out

    # -- per-game variance --------------------------------------------------------
    def game_variance(self, season: int) -> pl.DataFrame:
        """Unscaled posterior variance of the rating difference for every game of a season."""
        eng = self.eng
        rows = []
        for r in eng.season_slots(season).iter_rows(named=True):
            sp = self.slot(season, r["as_of"])
            idx = {t: i for i, t in enumerate(sp.teams)}
            g = eng.games.filter((pl.col("season") == season) & (pl.col("slot") == r["slot"]))
            for gid, hm, aw in g.select("game_id", "home", "away").iter_rows():
                i, j = idx[hm], idx[aw]
                C = sp.cov_points
                rows.append({"game_id": gid, "v": float(C[i, i] + C[j, j] - 2 * C[i, j])})
        return pl.DataFrame(rows, schema={"game_id": pl.Int64, "v": pl.Float64})

    def season_rows(self, season: int) -> pl.DataFrame:
        p = self.eng.predict(season).filter(pl.col("completed") & pl.col("margin").is_not_null())
        return p.join(self.game_variance(season), on="game_id", how="left")

    def fit_ks(self, season: int, available: list[int]) -> tuple[float, float]:
        """MLE of (k, s) on the training seasons for ``season``."""
        if season in self._ks:
            return self._ks[season]
        train = training_seasons(season, self.eng.cfg, available)
        rows = pl.concat([self.season_rows(s) for s in train])
        e = (rows["margin"] - rows["pred_margin"]).to_numpy()
        v = rows["v"].to_numpy()

        def nll(theta):
            k2, s2 = np.exp(theta)
            sd = np.sqrt(k2 * v + s2)
            return -norm.logpdf(e, 0, sd).sum()

        res = minimize(nll, x0=np.log([1.0, 14.0 ** 2]), method="Nelder-Mead")
        k, s = float(np.sqrt(np.exp(res.x[0]))), float(np.sqrt(np.exp(res.x[1])))
        self._ks[season] = (k, s)
        return k, s


def predictive_sd(rows: pl.DataFrame, k: float, s: float) -> np.ndarray:
    return np.sqrt(k * k * rows["v"].to_numpy() + s * s)


def rating_draws(eng: Engine, season: int, as_of, n_draws: int = 1000, seed: int = 0,
                 post: Posterior | None = None, available: list[int] | None = None) -> pl.DataFrame:
    """Joint draws of every entity's rating (points vs an average FBS team on a
    neutral field) as of ``as_of``: columns draw, team, rating."""
    post = post or Posterior(eng)
    available = available or list(range(eng.first_eval, eng.last_completed + 1))
    k, _s = post.fit_ks(season, available)
    sp = post.slot(season, as_of)
    mean = eng.ratings(season, as_of=as_of).select("team", "rating")
    mu = pl.DataFrame({"team": sp.teams}).join(mean, on="team", how="left")["rating"].fill_null(0.0).to_numpy()
    cov = (k * k) * sp.cov_points + 1e-9 * np.eye(len(sp.teams))
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mu, cov, size=n_draws, method="cholesky")
    x = x - x[:, sp.center].mean(axis=1, keepdims=True) + mu[sp.center].mean()  # keep FBS average at its mean
    return pl.DataFrame({
        "draw": np.repeat(np.arange(n_draws), len(sp.teams)),
        "team": np.tile(np.array(sp.teams, dtype=object), n_draws),
        "rating": x.ravel(),
    })


def game_noise(eng: Engine, season: int, post: Posterior | None = None,
               available: list[int] | None = None) -> dict[str, float]:
    post = post or Posterior(eng)
    available = available or list(range(eng.first_eval, eng.last_completed + 1))
    k, s = post.fit_ks(season, available)
    return {"home_edge": eng.calibration(season).b_home, "sigma": s, "scale": k}
