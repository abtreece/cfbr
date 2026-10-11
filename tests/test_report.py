import polars as pl

from cfbr.backtest.metrics import summarize, wilson_ci
from cfbr.report.rank import append_accuracy


def test_accuracy_rerun_replaces_week(tmp_path):
    path = tmp_path / "accuracy.csv"
    append_accuracy(path, 2026, 6, {"n_games": 10, "mae": 12.0})
    append_accuracy(path, 2026, 7, {"n_games": 20, "mae": 12.5})
    append_accuracy(path, 2026, 7, {"n_games": 21, "mae": 12.4})
    df = pl.read_csv(path)
    assert df["through_week"].to_list() == [6, 7]
    assert df.filter(pl.col("through_week") == 7)["n_games"].item() == 21


def test_wilson_ci_brackets_rate():
    lo, hi = wilson_ci(530, 1000)
    assert lo < 0.53 < hi
    assert 0.49 < lo < 0.51 and 0.55 < hi < 0.57


def test_ats_excludes_pushes_and_ties():
    df = pl.DataFrame({
        "margin": [10.0, -3.0, 7.0, 4.0],
        "pred_margin": [8.0, 2.0, 7.0, 1.0],
        "win_prob": [0.7, 0.55, 0.65, 0.52],
        "massey_pred": [0.0] * 4, "prior_pred": [0.0] * 4,
        # vegas home margin = -spread: 3, 0, 7, 4
        "spread": [-3.0, 0.0, -7.0, -4.0],
    })
    m = summarize(df)
    # game 1: model 8 > 3, actual 10 > 3 -> win; game 2: model 2 > 0, actual -3 -> loss;
    # game 3: pick == line -> excluded; game 4: push -> excluded
    assert m["ats_n"] == 2 and m["ats_wins"] == 1
