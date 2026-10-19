"""The owner's real-bet ledger: kept apart from the paper ledger, graded at its own price."""

from datetime import UTC, datetime, timedelta

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


def _bet(game_id=1, team="Cincinnati", line=1.5, odds=-110, risk=10.0, placed=PLACED, market="spread", live=False):
    game = _games().filter(pl.col("game_id") == game_id).to_dicts()[0]
    return new_bet(game, team, line, odds, risk, "DraftKings", placed, NOW, 2026, 7, market=market, live=live)


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


def test_new_bet_refuses_a_team_not_in_the_game_and_non_finite_terms():
    # e.g. a mistyped --game-id: Houston isn't in game 1, so it must not become "away".
    with pytest.raises(RealBetError, match="isn't in Cincinnati @ West Virginia"):
        _bet(team="Houston")
    for line, risk in ((float("nan"), 10.0), (float("inf"), 10.0)):
        with pytest.raises(RealBetError, match="line must be a number"):
            _bet(line=line, risk=risk)
    for risk in (float("nan"), float("inf"), 0.0):
        with pytest.raises(RealBetError, match="stake must be a positive number"):
            _bet(risk=risk)


def test_only_an_identical_wager_is_a_duplicate(tmp_path):
    led = add_real(read_real(tmp_path / "none.csv"), _bet())
    # Same game, side and moment, but different price, line, stake or book: separate wagers.
    for other in (_bet(odds=-105), _bet(line=2.0), _bet(risk=20.0), _bet() | {"book": "FanDuel"}):
        led = add_real(led, other)
    assert led.height == 5
    with pytest.raises(RealBetError, match="already recorded"):
        add_real(led, _bet())


def test_update_real_holds_the_ledger_lock(tmp_path):
    import fcntl

    from cfbr.bets.real import update_real

    path = tmp_path / "2026" / "bets_real.csv"
    seen = {}

    def change(df):
        # While the change runs, another process can't take the lock.
        with open(path.with_suffix(".csv.lock")) as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                seen["free"] = True
            except BlockingIOError:
                seen["free"] = False
        return add_real(df, _bet())

    assert update_real(path, change).height == 1 and seen == {"free": False}
    assert read_real(path).height == 1


def test_weekly_settles_real_bets_with_paper_bets_off(monkeypatch):
    from types import SimpleNamespace

    from cfbr import cli
    from cfbr.bets import run

    calls = []
    monkeypatch.setattr(run, "run_bets", lambda *a: calls.append("paper"))
    monkeypatch.setattr(run, "run_real_bets", lambda *a: calls.append("real"))
    off = SimpleNamespace(bets=SimpleNamespace(enabled=False))
    assert cli._bets_step(off, None, 2026, 6) is False and calls == ["real"]
    calls.clear()
    on = SimpleNamespace(bets=SimpleNamespace(enabled=True))
    assert cli._bets_step(on, None, 2026, 6) is False and calls == ["paper", "real"]
    # A paper failure doesn't stop real settlement, and either failure flags the run (exit 6).
    monkeypatch.setattr(run, "run_bets", lambda *a: (_ for _ in ()).throw(RuntimeError("x")))
    calls.clear()
    assert cli._bets_step(on, None, 2026, 6) is True and calls == ["real"]


def test_live_bets_must_be_placed_during_the_game():
    in_game = KICK + timedelta(hours=3)
    bet = _bet(placed=in_game, live=True)
    assert bet["live"] is True and bet["market"] == "spread"
    with pytest.raises(RealBetError, match="before kickoff.*so not live"):
        _bet(placed=KICK - timedelta(minutes=1), live=True)
    with pytest.raises(RealBetError, match="more than 6 hours after kickoff"):
        _bet(placed=KICK + timedelta(hours=6, minutes=1), live=True)
    # Unmarked, an in-game time is still refused, with a pointer to the live flag.
    with pytest.raises(RealBetError, match="mark it live"):
        _bet(placed=in_game)
    # find_game: a live bet is on the team's latest game kicked off by then, not its next one.
    assert find_game(_games(), "Cincinnati", in_game, live=True)["game_id"] == 1
    with pytest.raises(RealBetError, match="kicked off by"):
        find_game(_games(), "Cincinnati", PLACED, live=True)


def test_moneyline_has_no_line_and_grades_on_the_winner(tmp_path):
    in_game = KICK.replace(hour=23, minute=30) + timedelta(hours=3)
    ok = _bet(game_id=2, team="Oklahoma State", line=None, odds=327, placed=in_game, market="moneyline", live=True)
    assert ok["line"] is None and ok["side"] == "away" and ok["market"] == "moneyline"
    with pytest.raises(RealBetError, match="not a bet type"):
        _bet(market="total")
    with pytest.raises(RealBetError, match="line must be a number"):
        _bet(line=None)  # a spread still needs one
    hou = _bet(game_id=2, team="Houston", line=None, odds=-400, placed=in_game, market="moneyline", live=True)
    pregame = _bet(game_id=2, team="Houston", line=-4.5, odds=-105)
    led = add_real(add_real(add_real(read_real(tmp_path / "none.csv"), ok), hou), pregame)
    games = pl.DataFrame({"game_id": [2], "completed": [True], "home_points": [35], "away_points": [32]})
    closing = pl.DataFrame({"game_id": [2], "closing_line": [-4.0], "closing_source": ["snapshot"]})
    houml, houats, okst = settle_real(led, games, closing).sort("odds").to_dicts()  # −400, −105, +327
    # Houston won by 3: the Oklahoma State moneyline loses the stake, Houston's pays at −400,
    # and Houston −4.5 loses at its own line.
    assert okst["status"] == "lost" and okst["dollars"] == -10.0
    assert houml["status"] == "won" and houml["dollars"] == pytest.approx(2.5)
    assert houats["status"] == "lost"
    # Only the pregame spread gets a close and CLV; the live moneylines get neither.
    assert houats["clv"] == -0.5 and houats["closing_line"] == -4.0
    assert okst["clv"] is None and okst["closing_line"] is None and houml["clv"] is None
    m = summarize_real(settle_real(led, games, closing))
    assert (m["won"], m["lost"], m["n_clv"]) == (1, 2, 1) and m["avg_clv"] == -0.5


def test_a_live_spread_gets_no_clv(tmp_path):
    led = add_real(read_real(tmp_path / "none.csv"), _bet(placed=KICK + timedelta(hours=1), live=True))
    games = pl.DataFrame({"game_id": [1], "completed": [True], "home_points": [24], "away_points": [23]})
    closing = pl.DataFrame({"game_id": [1], "closing_line": [-2.5], "closing_source": ["snapshot"]})
    r = settle_real(led, games, closing).to_dicts()[0]
    assert r["status"] == "won" and r["clv"] is None


def test_a_ledger_from_before_markets_reads_as_pregame_spreads(tmp_path):
    path = tmp_path / "bets_real.csv"
    old = _bet()
    del old["market"], old["live"]
    pl.DataFrame([old]).write_csv(path)
    led = read_real(path)
    assert led["market"].to_list() == ["spread"] and led["live"].to_list() == [False]
    # ...and still refuses the same bet recorded again.
    with pytest.raises(RealBetError, match="already recorded"):
        add_real(led, _bet())
    # A moneyline and a live bet on the same game are separate wagers.
    in_game = KICK + timedelta(hours=1)
    led = add_real(led, _bet(placed=in_game, live=True))
    led = add_real(led, _bet(placed=in_game, live=True, market="moneyline", line=None, odds=150))
    write_real(led, path)
    assert read_real(path).height == 3


def test_win_dollars():
    assert win_dollars(-110, 10) == pytest.approx(9.0909, abs=1e-4)
    assert win_dollars(-105, 10) == pytest.approx(9.5238, abs=1e-4)
    assert win_dollars(150, 10) == 15.0
