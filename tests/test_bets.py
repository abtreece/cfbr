from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from cfbr.bets.ledger import (LEDGER_SCHEMA, add_bets, candidate_bets, clv, grade, market, read_ledger, risk, settle,
                              snapshot_frame, summarize, summary_markdown)
from cfbr.config import Bets

CFG = Bets()
PLACED = datetime(2026, 10, 11, 14, 5, tzinfo=UTC)


def _snapshot():
    # Home-perspective spreads (negative = home favored).
    return snapshot_frame([
        {"id": 1, "lines": [{"provider": "A", "spread": -3.0}, {"provider": "B", "spread": -3.5},
                            {"provider": "C", "spread": -2.5}]},
        {"id": 2, "lines": [{"provider": "A", "spread": 7.0}, {"provider": "B", "spread": 6.5}]},
        {"id": 3, "lines": [{"provider": "A", "spread": -10.0}]},
        {"id": 4, "lines": [{"provider": "A", "spread": None}]},
    ])


def _preds():
    return pl.DataFrame({
        "game_id": [1, 2, 3, 4, 5],
        "kickoff_ts": ["2026-10-17T19:00:00.000000+0000", "2026-10-17T23:00:00.000000+0000",
                       "2026-10-11T12:00:00.000000+0000",  # kicked off before the snapshot
                       "2026-10-17T19:00:00.000000+0000", "2026-10-17T19:00:00.000000+0000"],
        "home_team": ["H1", "H2", "H3", "H4", "H5"], "away_team": ["A1", "A2", "A3", "A4", "A5"],
        "pred_margin": [8.0, -4.0, 30.0, 10.0, 10.0],
    })


def test_market_consensus_and_best_numbers():
    m = market(_snapshot()).sort("game_id")
    g1 = m.row(0, named=True)
    assert g1["line"] == -3.0 and g1["n_providers"] == 3
    assert g1["best_home"] == -2.5 and g1["best_home_provider"] == "C"  # home bettor wants the highest spread
    assert g1["best_away"] == -3.5 and g1["best_away_provider"] == "B"


def test_candidates_threshold_side_and_pre_kickoff_only():
    c = candidate_bets(_preds(), _snapshot(), PLACED, CFG).sort("game_id")
    # game 1: model home +8 vs market home +3 -> edge +5 -> home bet
    # game 2: model home -4 vs market home -6.75 -> edge +2.75 -> below threshold
    # game 3: already kicked off; game 4: no line; game 5: no line at all
    assert c["game_id"].to_list() == [1]
    r = c.row(0, named=True)
    assert r["side"] == "home" and r["team"] == "H1" and r["edge"] == pytest.approx(5.0)
    assert r["best_line"] == -2.5 and r["placed_at"] == "2026-10-11T14:05:00+00:00"


def test_away_side_when_model_below_market():
    preds = _preds().with_columns(pred_margin=pl.Series([-1.0, -4.0, 0.0, 0.0, 0.0]))
    c = candidate_bets(preds, _snapshot(), PLACED, CFG).filter(pl.col("game_id") == 1).row(0, named=True)
    assert c["side"] == "away" and c["team"] == "A1" and c["edge"] == pytest.approx(-4.0)
    assert c["best_line"] == -3.5  # away bettor wants the lowest home spread


@pytest.mark.parametrize("side,line,hp,ap,expected", [
    ("home", -3.0, 24, 20, "won"), ("home", -3.0, 23, 20, "push"), ("home", -3.0, 22, 20, "lost"),
    ("away", -3.0, 22, 20, "won"), ("away", 7.0, 10, 20, "won"), ("away", 7.0, 13, 20, "push"),
    ("away", 7.0, 30, 20, "lost"),  # home +7 = away laying 7; home won outright
])
def test_grading(side, line, hp, ap, expected):
    assert grade(side, line, hp, ap) == expected


def test_risk_and_clv():
    assert risk(-110, 1.0) == pytest.approx((1.1, 1.0))
    assert risk(120, 1.0) == pytest.approx((1.0, 1.2))
    assert clv("home", -3.0, -5.0) == 2.0   # bet home -3, closed -5: beat the close by 2
    assert clv("away", -3.0, -5.0) == -2.0


def test_ledger_is_append_only_and_settles(tmp_path):
    ledger = read_ledger(tmp_path / "bets.csv")
    new = candidate_bets(_preds(), _snapshot(), PLACED, CFG).with_columns(
        season=pl.lit(2026, pl.Int32), week=pl.lit(7, pl.Int32))
    ledger, n = add_bets(ledger, new)
    assert n == 1
    # A later snapshot with a different line must not change the recorded bet.
    later = candidate_bets(_preds(), snapshot_frame([{"id": 1, "lines": [{"provider": "A", "spread": -1.0}]}]),
                           PLACED, CFG)
    ledger, n2 = add_bets(ledger, later)
    assert n2 == 0 and ledger["line"].to_list() == [-3.0]

    games = pl.DataFrame({"game_id": [1], "completed": [True], "home_points": [31], "away_points": [21]})
    closing = pl.DataFrame({"game_id": [1], "closing_line": [-4.5]})
    ledger = settle(ledger, games, closing)
    r = ledger.row(0, named=True)
    assert r["status"] == "won" and r["units"] == pytest.approx(1.0) and r["clv"] == pytest.approx(1.5)

    path = tmp_path / "bets.csv"
    ledger.write_csv(path)
    back = read_ledger(path)
    assert back.row(0, named=True)["status"] == "won"
    m = summarize(back, CFG)
    assert m["settled"] == 1 and m["breakeven"] == pytest.approx(110 / 210)
    md = summary_markdown(back, CFG, 2026)
    assert "No conclusion yet" in md and "1-0-0" in md


def test_non_books_never_price_a_bet():
    snap = snapshot_frame([{"id": 1, "lines": [
        {"provider": "DraftKings", "spread": -3.0}, {"provider": "Bovada", "spread": -2.5},
        {"provider": "numberfire", "spread": -9.0}, {"provider": "consensus", "spread": -3.0},
        {"provider": "teamrankings", "spread": -3.0}]}])
    assert set(snap["provider"]) == {"DraftKings", "Bovada"}  # offshore books stay in; projections don't
    m = market(snap).row(0, named=True)
    assert m["books"] == "Bovada -2.5; DraftKings -3"


def test_best_line_graded_separately_from_consensus():
    snap = snapshot_frame([{"id": 1, "lines": [{"provider": "A", "spread": -3.5}, {"provider": "B", "spread": -2.5},
                                               {"provider": "C", "spread": -3.0}]}])
    new = candidate_bets(_preds(), snap, PLACED, CFG).with_columns(season=pl.lit(2026, pl.Int32),
                                                                   week=pl.lit(7, pl.Int32))
    ledger, _ = add_bets(read_ledger(Path("/nonexistent")), new)
    # Home wins by 3: a push at the -3 consensus, a win at the best -2.5 (book B).
    games = pl.DataFrame({"game_id": [1], "completed": [True], "home_points": [23], "away_points": [20]})
    r = settle(ledger, games, pl.DataFrame({"game_id": [1], "closing_line": [-3.5]})).row(0, named=True)
    assert (r["status"], r["best_status"]) == ("push", "won")
    assert r["clv"] == pytest.approx(0.5) and r["best_clv"] == pytest.approx(1.0)


def test_legacy_ledger_without_new_columns_loads(tmp_path):
    path = tmp_path / "bets.csv"
    legacy = [c for c in LEDGER_SCHEMA if c not in ("books", "best_status", "best_units", "best_clv")]
    new = candidate_bets(_preds(), _snapshot(), PLACED, CFG).with_columns(season=pl.lit(2026, pl.Int32),
                                                                          week=pl.lit(7, pl.Int32))
    add_bets(read_ledger(path), new)[0].select(legacy).write_csv(path)
    back = read_ledger(path)
    assert back.columns == list(LEDGER_SCHEMA) and back["books"].is_null().all()


def test_offshore_books_feed_consensus_but_never_price_a_bet():
    snap = snapshot_frame([
        {"id": 1, "lines": [{"provider": "DraftKings", "spread": -3.0}, {"provider": "Bovada", "spread": -1.0}]},
        {"id": 2, "lines": [{"provider": "Bovada", "spread": 2.0}]},
    ])
    m = {r["game_id"]: r for r in market(snap, CFG.offshore_books).iter_rows(named=True)}
    assert m[1]["line"] == -2.0                                      # Bovada counts toward the consensus...
    assert m[1]["best_home"] == -3.0 and m[1]["best_home_provider"] == "DraftKings"  # ...but never prices
    assert m[1]["books"] == "Bovada -1 (offshore); DraftKings -3"
    assert m[2]["best_home"] is None and m[2]["best_away"] is None   # offshore-only game: no best line

    preds = _preds().with_columns(pred_margin=pl.Series([8.0, 10.0, 0.0, 0.0, 0.0]))
    new = candidate_bets(preds, snap, PLACED, CFG).with_columns(season=pl.lit(2026, pl.Int32),
                                                                week=pl.lit(7, pl.Int32))
    ledger, _ = add_bets(read_ledger(Path("/nonexistent")), new)
    games = pl.DataFrame({"game_id": [1, 2], "completed": [True, True], "home_points": [30, 30],
                          "away_points": [20, 20]})
    out = {r["game_id"]: r for r in settle(ledger, games, pl.DataFrame({"game_id": [1, 2],
                                                                         "closing_line": [-3.0, 2.0]})).iter_rows(named=True)}
    assert out[2]["status"] == "won" and out[2]["best_status"] is None and out[2]["best_clv"] is None
    assert out[1]["best_status"] == "won"
