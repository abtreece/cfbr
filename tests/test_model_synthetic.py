import time

import numpy as np
import polars as pl

from cfbr.model.massey import fit_massey
from cfbr.model.ridge import fit_efficiency

RNG = np.random.default_rng(7)
N_TEAMS = 130


def _schedule(n_games_per_team=12):
    teams = [f"T{i}" for i in range(N_TEAMS)]
    games = []
    for _ in range(n_games_per_team * N_TEAMS // 2):
        a, b = RNG.choice(N_TEAMS, 2, replace=False)
        games.append((teams[a], teams[b], RNG.random() < 0.1))
    return teams, games


def _synthetic_plays(plays_per_side=70):
    # Realistic FBS spread (~0.15 PPA/play) and per-play noise (~1.2), full 13-game season.
    teams, games = _schedule(n_games_per_team=13)
    o_true = RNG.normal(0, 0.15, N_TEAMS)
    d_true = RNG.normal(0, 0.15, N_TEAMS)
    idx = {t: i for i, t in enumerate(teams)}
    rows = []
    for hm, aw, neutral in games:
        for off, de in ((hm, aw), (aw, hm)):
            h = 0.0 if neutral else (1.0 if off == hm else -1.0)
            mu = 0.05 + o_true[idx[off]] - d_true[idx[de]] + 0.02 * h
            ppa = mu + RNG.normal(0, 1.2, plays_per_side)
            rows.append(pl.DataFrame({
                "offense": [off] * plays_per_side, "defense": [de] * plays_per_side,
                "home_team": [hm] * plays_per_side, "neutral_site": [neutral] * plays_per_side,
                "ppa": ppa,
            }))
    return teams, pl.concat(rows), o_true, d_true


def test_efficiency_recovers_known_strengths():
    teams, plays, o_true, d_true = _synthetic_plays()
    fit = fit_efficiency(plays, "ppa", alpha=50.0, teams=teams, center_on=teams)
    assert np.corrcoef(fit.o, o_true)[0, 1] > 0.95
    assert np.corrcoef(fit.d, d_true)[0, 1] > 0.95
    assert np.corrcoef(fit.o - fit.d, o_true - d_true)[0, 1] > 0.95


def test_prior_pull_without_data_returns_prior():
    teams = ["A", "B"]
    empty = pl.DataFrame(schema={"offense": pl.Utf8, "defense": pl.Utf8, "home_team": pl.Utf8,
                                 "neutral_site": pl.Boolean, "ppa": pl.Float64})
    fit = fit_efficiency(empty, "ppa", 100.0, teams, prior_o=np.array([0.1, -0.1]), prior_d=np.array([0.0, 0.2]))
    np.testing.assert_allclose(fit.o, [0.1, -0.1])
    np.testing.assert_allclose(fit.d, [0.0, 0.2])


def test_fit_one_week_under_10s():
    teams, plays, _, _ = _synthetic_plays(plays_per_side=85)  # ~130k plays, a full season
    assert plays.height > 120_000
    t0 = time.perf_counter()
    for target in ("ppa", "ppa"):  # PPA + success-rate fit
        fit_efficiency(plays, target, 300.0, teams, center_on=teams)
    assert time.perf_counter() - t0 < 10


def test_massey_recovers_known_strengths():
    teams, games = _schedule(n_games_per_team=12)
    r_true = RNG.normal(0, 12, N_TEAMS)
    idx = {t: i for i, t in enumerate(teams)}
    rows = [{"home": h, "away": a, "neutral_site": n,
             "margin": r_true[idx[h]] - r_true[idx[a]] + (0 if n else 2.5) + RNG.normal(0, 8)}
            for h, a, n in games]
    fit = fit_massey(pl.DataFrame(rows), teams, alpha=1.0, cap=100, center_on=teams)
    assert np.corrcoef(fit.rating, r_true)[0, 1] > 0.95
