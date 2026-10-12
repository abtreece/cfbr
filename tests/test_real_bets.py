"""The owner's real-bet ledger: kept apart from the paper ledger, graded at its own price."""

from datetime import UTC, datetime

import polars as pl
import pytest

from cfbr.bets.real import (RealBetError, add_real, find_game, new_bet, read_real, settle_real, summarize_real,
                            win_dollars, write_real)

KICK = datetime(2026, 10, 17, 16, 0, tzinfo=UTC)
PLACED = datetime(2026, 10, 12, 4, 6, 17, tzinfo=UTC)
NOW = datetime(2026, 10, 12, 14, 0, tzinfo=UTC)


def _games() -> pl.DataFrame:
    return pl.DataFrame({
        "game_id": [1, 2, 3], "home_team": ["West Virginia", "Houston", "Cincinnati"],
        "away_team": ["Cincinnati", "Oklahoma State", "Baylor"],
        "kickoff_ts": [KICK, KICK.replace(hour=23, minute=30), KICK.replace(day=24)],
        "completed": [False, False, False]})


def _bet(game_id=1, team="Cincinnati", line=1.5, odds=-110, risk=10.0, placed=PLACED):
    game = _games().filter(pl.col("game_id") == game_id).to_dicts()[0]
    return new_bet(game, team, line, odds, risk, "DraftKings", placed, NOW, 2026, 7)


def test_find_game_takes_the_team_s_next_game():
    # Cincinnati plays game 1 (Oct 17) and game 3 (Oct 24): the bet is on the next one.
    assert find_game(_games(), "cincinnati", PLACED)["game_id"] == 1
    assert find_game(_games(), "Cincinnati", KICK)["game_id"] == 3  # after game 1 kicked off
    with pytest.raises(RealBetError, match="no 'Nobody' game"):
        find_game(_games(), "Nobody", PLACED)


def test_new_bet_stores_the_home_spread_and_refuses_bad_terms():
    away = _bet()  # Cincinnati +1.5 as the away team = home spread −1.5
    assert away["side"] == "away" and away["line"] == -1.5 and away["team"] == "Cincinnati"
    home = _bet(game_id=2, team="Houston", line=-4.5, odds=-105)
    assert home["side"] == "home" and home["line"] == -4.5
    with pytest.raises(RealBetError, match="after kickoff"):
        _bet(placed=KICK)
    with pytest.raises(RealBetError, match="not an American price"):
        _bet(odds=50)


def test_ledger_round_trips_and_refuses_duplicates(tmp_path):
    path = tmp_path / "2026" / "bets_real.csv"
    led = add_real(read_real(path), _bet())
    write_real(led, path)
    led = read_real(path)
    assert led.height == 1 and led["odds"][0] == -110 and led["risk_dollars"][0] == 10.0
    with pytest.raises(RealBetError, match="already recorded"):
        add_real(led, _bet())
    assert add_real(led, _bet(placed=PLACED.replace(minute=30))).height == 2  # a second bet, later


def test_settle_grades_at_the_bet_s_own_line_and_price(tmp_path):
    led = add_real(add_real(read_real(tmp_path / "none.csv"), _bet()),
                   _bet(game_id=2, team="Houston", line=-4.5, odds=-105))
    games = pl.DataFrame({"game_id": [1, 2], "completed": [True, True],
                          "home_points": [24, 30], "away_points": [23, 26]})
    closing = pl.DataFrame({"game_id": [1, 2], "closing_line": [-2.5, -5.0], "closing_source": ["snapshot", "cfbd"]})
    out = settle_real(led, games, closing).sort("game_id")
    cin, hou = out.to_dicts()
    # WVU won by 1; Cincinnati +1.5 covers: wins $10 × 100/110.
    assert cin["status"] == "won" and cin["dollars"] == pytest.approx(9.0909, abs=1e-3)
    # Houston won by 4; −4.5 loses the $10 risked.
    assert hou["status"] == "lost" and hou["dollars"] == -10.0
    # CLV against the close, from the bettor's side: Cincinnati +1.5 vs +2.5 close = −1; Houston −4.5 vs −5 = +0.5.
    assert cin["clv"] == -1.0 and hou["clv"] == 0.5 and hou["closing_source"] == "cfbd"
    m = summarize_real(out)
    assert (m["won"], m["lost"], m["open"]) == (1, 1, 0) and m["dollars"] == pytest.approx(-0.909, abs=1e-3)
    # Settled rows never change again.
    assert settle_real(out, games.with_columns(home_points=pl.lit(0)), closing).equals(out)


def test_win_dollars():
    assert win_dollars(-110, 10) == pytest.approx(9.0909, abs=1e-4)
    assert win_dollars(-105, 10) == pytest.approx(9.5238, abs=1e-4)
    assert win_dollars(150, 10) == 15.0
