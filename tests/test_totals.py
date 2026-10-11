import numpy as np
import polars as pl
import pytest

from cfbr.model.totals import fit_pace, fit_points


def test_fit_points_recovers_offense_defense_and_environment():
    rng = np.random.default_rng(3)
    teams = [f"T{i}" for i in range(30)]
    off, de = rng.normal(0, 6, 30), rng.normal(0, 6, 30)
    rows = []
    for _ in range(600):
        a, b = rng.choice(30, 2, replace=False)
        rows.append({"team": teams[a], "opp": teams[b], "home": 1.0, "points": 27 + off[a] + de[b] + 1.5 + rng.normal(0, 3)})
        rows.append({"team": teams[b], "opp": teams[a], "home": -1.0, "points": 27 + off[b] + de[a] - 1.5 + rng.normal(0, 3)})
    df = pl.DataFrame(rows)
    z = np.zeros(30)
    mu, hh, po, pd_ = fit_points(df, teams, z, z, None, mu_fallback=99.0)
    assert mu == pytest.approx(27, abs=1.0) and hh == pytest.approx(1.5, abs=0.5)
    assert np.corrcoef(po, off)[0, 1] > 0.95 and np.corrcoef(pd_, de)[0, 1] > 0.95


def test_fit_points_uses_fallback_environment_early():
    df = pl.DataFrame({"team": ["A"], "opp": ["B"], "home": [1.0], "points": [40.0]})
    mu, hh, po, pd_ = fit_points(df, ["A", "B"], np.zeros(2), np.zeros(2), None, mu_fallback=27.0)
    assert mu == 27.0 and hh == 0.0           # too few games: environment fixed at last season's
    empty = df.head(0)
    assert fit_points(empty, ["A", "B"], np.ones(2), np.zeros(2), None, 27.0)[2].tolist() == [1.0, 1.0]


def test_fit_pace_shrinks_toward_prior_without_data():
    mu, po, pd_ = fit_pace(pl.DataFrame(schema={"team": pl.Utf8, "opp": pl.Utf8, "plays": pl.Int64}),
                           ["A", "B"], np.array([3.0, -3.0]), np.zeros(2), 68.0)
    assert mu == 68.0 and po.tolist() == [3.0, -3.0]


def test_totals_gain_sign():
    from cfbr.report.shadow import totals_gain

    df = pl.DataFrame({"completed": [True, True], "total": [50.0, 60.0], "v2_total": [52.0, 58.0],
                       "base_total": [55.0, 55.0]})
    g = totals_gain(df)
    assert g["n"] == 2 and g["gain"] == pytest.approx(3.0)   # baseline misses 5 each, v2 misses 2 each
