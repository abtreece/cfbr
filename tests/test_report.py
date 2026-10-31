import polars as pl
import pytest

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


def test_diagnostics_tables():
    import numpy as np

    from cfbr.backtest.metrics import diagnostics
    rng = np.random.default_rng(0)
    n = 400
    pred = rng.normal(0, 10, n)
    line = pred + rng.normal(0, 2, n)
    margin = 2.0 + 1.1 * pred + rng.normal(0, 12, n)   # under-dispersed, biased predictions
    df = pl.DataFrame({
        "season": [2020] * n, "season_type": ["regular"] * (n - 20) + ["postseason"] * 20,
        "week": [1] * 50 + [2] * 50 + [6] * 150 + [10] * 150,
        "margin": margin, "pred_margin": pred, "win_prob": 1 / (1 + np.exp(-pred / 8)),
        "prior_share_h": np.linspace(0, 1, n), "prior_share_a": np.linspace(0, 1, n), "spread": -line,
    })
    d = diagnostics(df)
    assert d["n_games"] == n
    assert [r["bucket"] for r in d["by_week"]] == ["w01", "w02", "w05-08", "w09-12", "post"]
    assert sum(r["n"] for r in d["by_week"]) == n
    overall_slope = np.cov(pred, margin, ddof=0)[0, 1] / np.var(pred)
    assert overall_slope > 1.0 and abs(sum(r["bias"] * r["n"] for r in d["by_week"]) / n - np.mean(margin - pred)) < 1e-9
    assert [r["bucket"] for r in d["by_prior_share"]] == ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">=0.9"]
    assert len(d["reliability"]) == 10 and sum(r["n"] for r in d["reliability"]) == int((margin != 0).sum())
    assert abs(d["blend"]["w_line"] + d["blend"]["w_model"] - 1.1) < 0.1   # recovers the 1.1 scale on the line
    assert [r["bucket"] for r in d["ats_by_edge"]] == ["0-3", "3-6", "6-10", "10+"]
    assert sum(r["n"] for r in d["ats_by_edge"]) <= n
    assert diagnostics(df.clear()) == {"n_games": 0}
    assert "blend" not in diagnostics(df.drop("spread"))


def _polls(rows):
    return pl.DataFrame(rows, schema={"season": pl.Int32, "season_type": pl.Utf8, "week": pl.Int32, "poll": pl.Utf8,
                                      "rank": pl.Int32, "team": pl.Utf8}, orient="row")


def test_poll_ranks_use_poll_released_after_the_week():
    from cfbr.report.rank import poll_ranks

    polls = _polls([
        (2026, "regular", 6, "AP Top 25", 1, "Alabama"),
        (2026, "regular", 7, "AP Top 25", 1, "Notre Dame"),
        (2026, "regular", 7, "AP Top 25", 2, "Alabama"),
        (2026, "regular", 8, "AP Top 25", 1, "Georgia"),  # future poll: never used for week 6
        (2026, "regular", 6, "Coaches Poll", 3, "Alabama"),
    ])
    ranks, used = poll_ranks(polls, 2026, 6)
    assert used == {"ap_rank": 7, "coaches_rank": 6, "cfp_rank": None}  # coaches falls back to latest
    d = dict(zip(ranks["team"], ranks["ap_rank"]))
    assert d["Notre Dame"] == 1 and d["Alabama"] == 2 and d.get("Georgia") is None


def test_rankings_markdown_flags_stale_polls():
    from cfbr.report.rank import add_polls, poll_note, to_markdown

    rankings = pl.DataFrame({"rank": [1, 26], "team": ["Alabama", "Army"], "record": ["5-1", "6-0"],
                             "rating": [28.5, 5.0]})
    polls = _polls([(2026, "regular", 6, "AP Top 25", 1, "Alabama"), (2026, "regular", 6, "AP Top 25", 20, "Army")])
    out, used = add_polls(rankings, polls, 2026, 6)
    assert out.columns[-3:] == ["ap_rank", "coaches_rank", "cfp_rank"]
    note = poll_note(used, 6)
    assert "week 7 not released yet" in note and "Coaches: not released" in note
    md = to_markdown(out, "t", note=note)
    assert "outside the model's top 25" in md and "| 26 | Army |" in md


def test_market_spread_prefers_books_and_never_uses_numberfire():
    from cfbr.backtest.metrics import consensus_spread

    lines = pl.DataFrame({
        "game_id": [1, 1, 1, 1, 2, 2, 2, 3, 3],
        "provider": ["DraftKings", "Bovada", "numberfire", "consensus", "consensus", "teamrankings", "numberfire",
                     "numberfire", "teamrankings"],
        "spread": [-3.0, -4.0, -20.0, -3.0, -7.0, -6.0, 5.0, 1.0, -2.5],
    })
    out = {r["game_id"]: r for r in consensus_spread(lines).iter_rows(named=True)}
    assert (out[1]["spread"], out[1]["source"], out[1]["n_providers"]) == (-3.5, "books", 2)
    assert (out[2]["spread"], out[2]["source"]) == (-7.0, "consensus")
    assert (out[3]["spread"], out[3]["source"]) == (-2.5, "teamrankings")


def test_shadow_window_and_paired_gain():
    from cfbr.report.shadow import in_window, paired_gain, summary_markdown

    df = pl.DataFrame({
        "season": [2026, 2026, 2026, 2026, 2025],
        "week": [6, 7, 1, 9, 9],
        "season_type": ["regular", "regular", "postseason", "regular", "regular"],
        "completed": [True, True, True, False, True],
        "margin": [10.0, -3.0, 7.0, None, 3.0],
        "pred_v1": [4.0, 2.0, 1.0, 5.0, 0.0],
        "pred_roster": [8.0, 0.0, 3.0, 6.0, 0.0],
    }).with_columns(in_window=in_window())
    assert df["in_window"].to_list() == [False, True, True, True, False]
    g = paired_gain(df.filter(pl.col("in_window")))
    # wk7: |2-(-3)|=5 vs |0-(-3)|=3 -> +2 ; post: |1-7|=6 vs |3-7|=4 -> +2 ; wk9 not completed
    assert g["n"] == 2 and g["gain"] == 2.0
    df = df.with_columns(kickoff_ts=pl.lit("2026-10-17T19:00:00").str.to_datetime(),
                         home_team=pl.lit("H"), away_team=pl.lit("A"))
    md = summary_markdown(df, 2026)
    assert "Currently meets" in md and "Upcoming window games" in md


def test_calibration_v2_stage2_and_shadow_gain():
    import numpy as np

    from cfbr.model.calibrate import fit_stage2, stage2_design
    from cfbr.report.shadow import calibration_gain, calibration_markdown, in_window

    rng = np.random.default_rng(0)
    n = 4000
    g = pl.DataFrame({"neutral_site": rng.random(n) < 0.1, "prior_share_h": rng.random(n),
                      "prior_share_a": rng.random(n), "home_fcs": rng.random(n) < 0.03,
                      "away_fcs": rng.random(n) < 0.08})
    pred = rng.normal(0, 14, n)
    true = np.array([-0.18, 0.32, -0.3, 3.0])
    g = g.with_columns(margin=pl.Series(pred + stage2_design(g, pred) @ true + rng.normal(0, 2, n)))
    assert np.allclose(fit_stage2(g, pred), true, atol=0.15)
    # fcs is signed: +1 for a pooled FCS visitor, -1 for a pooled FCS host
    x = stage2_design(pl.DataFrame({"neutral_site": [False, True, False], "prior_share_h": [1.0, 1.0, 0.0],
                                    "prior_share_a": [1.0, 0.0, 0.0], "home_fcs": [False, False, True],
                                    "away_fcs": [True, False, False]}), np.array([10.0, 10.0, 10.0]))
    assert x.tolist() == [[10.0, 10.0, 1.0, 1.0], [10.0, 5.0, 0.0, 0.0], [10.0, 0.0, 0.0, -1.0]]

    df = pl.DataFrame({"season": [2026] * 3, "week": [6, 7, 8], "season_type": ["regular"] * 3,
                       "completed": [True, True, False], "margin": [10.0, -3.0, None],
                       "pred_base": [4.0, 2.0, 1.0], "pred_v2": [8.0, 0.0, 3.0]}).with_columns(in_window=in_window())
    w = calibration_gain(df.filter(pl.col("in_window")))
    assert w["n"] == 1 and w["gain"] == 2.0
    assert "calibration shadow" in calibration_markdown(df, 2026)


def test_stage2_v3_adds_a_pred_times_fcs_column():
    import numpy as np
    import polars as pl
    from cfbr.model.calibrate import STAGE2_TERMS, STAGE2_TERMS_V3, fit_stage2, stage2_design, stage2_terms

    g = pl.DataFrame({"neutral_site": [False, False, True], "prior_share_h": [0.5, 0.2, 1.0],
                      "prior_share_a": [0.5, 0.4, 1.0], "home_fcs": [False, False, True], "away_fcs": [False, True, False],
                      "margin": [10.0, 30.0, -20.0]})
    pred = np.array([8.0, 20.0, -15.0])
    X2 = stage2_design(g, pred)
    X3 = stage2_design(g, pred, fcs_slope=True)
    assert X2.shape == (3, 4) and X3.shape == (3, 5)
    assert np.allclose(X3[:, :4], X2)
    assert X3[:, 4].tolist() == [0.0, 20.0, -15.0]  # pred on the two pooled-FCS games, 0 elsewhere
    assert stage2_terms() == STAGE2_TERMS and stage2_terms(True) == STAGE2_TERMS_V3 == STAGE2_TERMS + ["pred_fcs"]
    assert fit_stage2(g, pred, fcs_slope=True).shape == (5,)


def test_market_ratings_summary_and_markdown():
    from cfbr.backtest.market import markdown, summary

    df = pl.DataFrame({
        "season": [2025] * 6, "slot": ["r05"] * 3 + ["r06"] * 3, "slot_order": [5] * 3 + [6] * 3,
        "team": ["A", "B", "FCS"] * 2, "conference": ["SEC", "MAC", None] * 2, "fbs": [True, True, False] * 2,
        "prior_share": [0.5] * 6, "model": [10.0, -5.0, -20.0, 11.0, -6.0, -21.0],
        "market": [8.0, -4.0, -25.0, 9.0, -4.0, -24.0], "market_home": [2.5] * 6, "n_lined": [40] * 3 + [50] * 3,
    }).with_columns(diff=pl.col("model") - pl.col("market"))
    s = summary(df, b_home=2.8)
    assert s["last_slot"] == "r06" and s["by_slot"].height == 2
    assert s["by_conference"].filter(pl.col("conference") == "SEC")["mean_diff"].item() == pytest.approx(2.0)
    assert s["fcs"]["diff"].item() == pytest.approx(3.0)
    md = markdown(2025, s)
    assert "## By conference (after r06)" in md and "Pooled FCS" in md and "2.80" in md
    assert markdown(2025, {}).endswith("No lined games yet.\n")
