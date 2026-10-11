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
