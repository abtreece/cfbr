import numpy as np

from cfbr.backtest.calib_gate import GATE_MIN_GAIN, gate_markdown, paired_gain


def test_paired_gain_direction_and_ci():
    a = np.array([10.0, 12.0, 8.0, 11.0])
    b = np.array([9.0, 11.0, 7.0, 10.0])  # b is one point better everywhere
    g = paired_gain(a, b)
    assert g["n"] == 4 and g["gain"] == 1.0 and g["ci95"] == (1.0, 1.0)


def test_gate_markdown_states_pass_or_fail():
    rep = {"seasons": [2015, 2021], "n_games": 100, "n_fcs_games": 10,
           "primary": {"n": 100, "gain": 0.05, "ci95": (0.01, 0.09), "threshold": GATE_MIN_GAIN, "passes": True},
           "base_vs_v2": {"n": 100, "gain": 0.04, "ci95": (0.0, 0.08)},
           "base_vs_v3": {"n": 100, "gain": 0.09, "ci95": (0.03, 0.15)},
           "mae": {"base": 13.4, "v2": 13.36, "v3": 13.31},
           "fcs_games": {k: {"mae": 14.0, "slope": s} for k, s in (("base", 0.71), ("v2", 0.75), ("v3", 0.98))},
           "pred_fcs_by_season": {"2015": -0.3, "2016": -0.28}}
    md = gate_markdown(rep)
    assert "**passes**" in md and "+0.0500 [+0.0100, +0.0900]" in md and "| v3 | 13.310 | 14.000 | 0.980 |" in md
    rep["primary"]["passes"] = False
    assert "**fails**" in gate_markdown(rep)


def test_gate_markdown_sanity_rendering():
    rep = {"seasons": [2022, 2025], "n_games": 100, "n_fcs_games": 10,
           "primary": {"n": 100, "gain": 0.02, "ci95": (-0.02, 0.06), "threshold": GATE_MIN_GAIN, "passes": False},
           "base_vs_v2": {"n": 100, "gain": 0.04, "ci95": (0.0, 0.08)},
           "base_vs_v3": {"n": 100, "gain": 0.09, "ci95": (0.03, 0.15)},
           "mae": {"base": 13.4, "v2": 13.36, "v3": 13.31},
           "fcs_games": {k: {"mae": 14.0, "slope": s} for k, s in (("base", 0.71), ("v2", 0.75), ("v3", 0.98))},
           "pred_fcs_by_season": {"2022": -0.3}}
    md = gate_markdown(rep, sanity=True)
    assert "Sanity criterion" in md and "**OK**" in md and "**fails**" not in md and "Calibration v3 sanity" in md
