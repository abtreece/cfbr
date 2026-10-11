import polars as pl
import pytest

from cfbr.config import Config, GarbageTime
from cfbr.prep.plays import FCS, garbage_time_mask, pool_fcs, presnap_scores, success_expr


def _success(down, distance, gained, play_type="Rush"):
    df = pl.DataFrame({"down": [down], "distance": [distance], "yards_gained": [gained], "play_type": [play_type]})
    return df.select(success_expr()).item()


@pytest.mark.parametrize(
    "down,distance,gained,expected",
    [
        (1, 10, 5, True), (1, 10, 4, False),
        (2, 10, 7, True), (2, 10, 6, False),
        (3, 4, 4, True), (3, 4, 3, False),
        (4, 1, 1, True), (4, 1, 0, False),
        (2, 5, 4, True),  # 3.5 needed
        (1, 10, -3, False),
    ],
)
def test_success_thresholds(down, distance, gained, expected):
    assert _success(down, distance, gained) is expected


def test_success_turnovers_and_tds():
    assert _success(1, 10, 15, "Fumble Recovery (Opponent)") is False
    assert _success(3, 10, 0, "Pass Interception Return") is False
    assert _success(3, 10, 2, "Rushing Touchdown") is True  # goal-line yardage quirks


@pytest.mark.parametrize(
    "period,margin,keep",
    [
        (1, 60, True),
        (2, 38, True), (2, 39, False),
        (3, 28, True), (3, 29, False),
        (4, 22, True), (4, 23, False),
        (5, 40, True),  # overtime kept
    ],
)
def test_garbage_time_thresholds(period, margin, keep):
    df = pl.DataFrame({"period": [period], "margin_at_snap": [margin]})
    assert df.select(garbage_time_mask(GarbageTime())).item() is keep


def test_garbage_time_disabled():
    df = pl.DataFrame({"period": [4], "margin_at_snap": [50]})
    assert df.select(garbage_time_mask(GarbageTime(enabled=False))).item() is True


def _pool_frame():
    return pl.DataFrame({
        "offense_team": ["Alabama", "Mercer", "Furman"],
        "defense_team": ["Mercer", "Alabama", "Clemson"],
        "offense_fcs": [False, True, True],
        "defense_fcs": [True, False, False],
        "offense_conference": ["SEC", "Southern", "Southern"],
        "defense_conference": ["Southern", "SEC", "ACC"],
    })


def test_fcs_single_pool():
    out = pool_fcs(_pool_frame(), "single")
    assert out["offense"].to_list() == ["Alabama", FCS, FCS]
    assert out["defense"].to_list() == [FCS, "Alabama", "Clemson"]


def test_fcs_conference_pool():
    out = pool_fcs(_pool_frame(), "conference")
    assert out["offense"].to_list() == ["Alabama", "FCS:Southern", "FCS:Southern"]


def test_fcs_unknown_strategy():
    with pytest.raises(ValueError):
        pool_fcs(_pool_frame(), "tiers")


def test_presnap_scores_lag_post_play_scores():
    # CFBD scores are post-play: the FG row already shows 3-0.
    df = pl.DataFrame({
        "game_id": [1, 1, 1, 1],
        "play_seq": [1, 2, 3, 4],
        "offense": ["A", "A", "B", "B"],
        "home": ["A"] * 4,
        "play_type": ["Rush", "Field Goal Good", "Rush", "Rush"],
        "offense_score": [0, 3, 0, 0],
        "defense_score": [0, 0, 3, 3],
    })
    out = presnap_scores(df)
    assert out["home_before"].to_list() == [0, 0, 3, 3]
    assert out["margin_at_snap"].to_list() == [0, 0, 3, 3]


def test_presnap_scores_skip_unreliable_rows():
    # A timeout row with swapped offense/defense must not corrupt the next snap.
    df = pl.DataFrame({
        "game_id": [1, 1, 1],
        "play_seq": [1, 2, 3],
        "offense": ["A", "A", "A"],
        "home": ["A"] * 3,
        "play_type": ["Rushing Touchdown", "Timeout", "Rush"],
        "offense_score": [7, 0, 7],
        "defense_score": [0, 7, 0],
    })
    out = presnap_scores(df)
    assert out["home_before"].to_list() == [0, 7, 7]
    assert out["away_before"].to_list() == [0, 0, 0]
