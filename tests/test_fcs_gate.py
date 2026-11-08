import json

import polars as pl

from cfbr.backtest.fcs_gate import ats, gate_report, latest_gate


def _frame(p1, p2, fcs=None, spread=None):
    n = len(p1)
    return pl.DataFrame({"margin": [0.0] * n, "pred_v1": p1, "pred_v2": p2,
                         "fcs_game": fcs or [False] * n, "spread": spread or [None] * n},
                        schema_overrides={"spread": pl.Float64})


def test_gate1_needs_only_a_positive_gain_but_gates_2_and_3_need_the_ci_floor():
    noisy = _frame([2.0, 1.0] * 5, [1.0, 1.8] * 5)   # mean gain +0.1, wide CI
    assert gate_report(noisy, 1)["primary"]["passes"] is True
    assert gate_report(noisy, 2)["primary"]["passes"] is False
    assert gate_report(noisy, 3)["primary"]["passes"] is False
    tight = _frame([2.0] * 10, [1.5, 1.49] * 5)
    assert gate_report(tight, 2)["primary"]["passes"] is True


def test_negative_gain_fails_every_gate():
    worse = _frame([1.0] * 10, [2.0] * 10)
    assert not any(gate_report(worse, g)["primary"]["passes"] for g in (1, 2, 3))


def test_split_by_fcs_game():
    df = _frame([2.0, 2.0, 1.0], [1.0, 1.0, 1.0], fcs=[True, True, False])
    r = gate_report(df, 1)
    assert r["split"]["fbs_fcs"]["n"] == 2 and r["split"]["fbs_fbs"]["n"] == 1
    assert r["split"]["fbs_fcs"]["mae_v1"] == 2.0 and r["split"]["fbs_fcs"]["mae_v2"] == 1.0


def test_ats_skips_pushes_and_unlined_games():
    # Vegas home margin = -spread = 3. Margins 7 (cover), 3 (push), 0 (no cover), unlined.
    df = pl.DataFrame({"margin": [7.0, 3.0, 0.0, 5.0], "p": [5.0, 5.0, 5.0, 5.0],
                       "spread": [-3.0, -3.0, -3.0, None]})
    assert ats(df, "p") == {"n": 2, "wins": 1, "rate": 0.5}


def test_latest_gate_reads_the_newest_run(tmp_path):
    for ts, ok in (("20261108T000000Z", False), ("20261109T000000Z", True)):
        d = tmp_path / f"{ts}-fcs-gate-1"
        d.mkdir()
        (d / "gate.json").write_text(json.dumps({"primary": {"passes": ok}}))
    assert latest_gate(tmp_path, 1)["primary"]["passes"] is True
    assert latest_gate(tmp_path, 2) is None
