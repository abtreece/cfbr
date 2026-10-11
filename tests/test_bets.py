import math
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from cfbr.bets.ledger import (LEDGER_SCHEMA, add_bets, by_tier, candidate_bets, clv, grade, market, money, read_ledger,
                              risk, settle, snapshot_frame, summarize, summary_markdown, tier_expr, tiers, weekly)
from cfbr.config import Bets

CFG = Bets(pricing_books=[])  # synthetic book names below; the pricing list has its own test
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


def test_stale_snapshot_never_bets_a_game_already_kicked_off():
    # A snapshot from before game 1's kickoff, recorded after it (e.g. `cfbr bets` rerun on Saturday
    # against Sunday's snapshot): game 1 is out, and recorded_at is the wall clock, not the snapshot.
    now = datetime(2026, 10, 17, 20, 0, tzinfo=UTC)
    preds = _preds().with_columns(pred_margin=pl.Series([8.0, -12.0, 30.0, 10.0, 10.0]))
    assert set(candidate_bets(preds, _snapshot(), PLACED, CFG)["game_id"]) == {1, 2}
    c = candidate_bets(preds, _snapshot(), PLACED, CFG, now=now)
    assert c["game_id"].to_list() == [2]
    r = c.row(0, named=True)
    assert r["placed_at"] == "2026-10-11T14:05:00+00:00" and r["recorded_at"] == "2026-10-17T20:00:00+00:00"


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


def _settled(gid, week, edge, status, season_type="regular"):
    units = {"won": 1.0, "lost": -1.1, "push": 0.0, "open": None}[status]
    return {"game_id": gid, "season": 2026, "week": week, "side": "home", "edge": edge, "odds": -110, "stake": 1.0,
            "line": -3.0, "status": status, "units": units, "season_type": season_type,
            "kickoff_ts": "2026-10-17T19:00:00Z", "home_team": f"H{gid}", "away_team": f"A{gid}", "team": f"H{gid}",
            "model_margin": edge + 3.0}


def _ledger(rows):
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def test_tiers_from_edge_size():
    ts = tiers(CFG)
    assert [t["label"] for t in ts] == ["Small edge (3–6 pts)", "Medium edge (6–10 pts)", "Large edge (10+ pts)"]
    d = _ledger([_settled(i, 7, e, "open") for i, e in enumerate([3.0, -5.9, 6.0, -9.5, 10.0, 14.5])])
    assert d.select(tier_expr(CFG))["literal"].to_list() == ["Small", "Small", "Medium", "Medium", "Large", "Large"]
    out = {t["name"]: t for t in by_tier(d, CFG)}
    assert out["Small"]["bets"] == 2 and out["Large"]["open"] == 2
    # Every configured default tier has a backtest reference rate.
    assert all(t["backtest"] is not None for t in out.values())
    assert out["Small"]["backtest"]["rate"] == pytest.approx(860 / 1706)


def test_dollars_scale_units_by_unit_value():
    d = _ledger([_settled(1, 7, 4, "won"), _settled(2, 7, 4, "lost"), _settled(3, 7, 4, "open")])
    m = summarize(d, CFG)
    # $10 units (flat 1% of the $1,000 starting bankroll): +$10 - $11.
    assert m["dollars"] == pytest.approx(-1.0) and m["risked_dollars"] == pytest.approx(22.0)
    assert m["open_risk_dollars"] == pytest.approx(11.0) and m["balance"] == pytest.approx(999.0)
    assert summarize(d, Bets(unit_value=25.0))["dollars"] == pytest.approx(-2.5)
    assert money(-10.0) == "-$10" and money(1234.0) == "+$1,234" and money(0.0) == "$0"
    assert money(10.5) == "+$10.50" and money(-1.0000000001) == "-$1"  # cents only when not whole


def test_weekly_record_runs_in_calendar_order_with_postseason_last():
    d = _ledger([
        _settled(1, 1, 4, "won", "postseason"),  # CFBD restarts postseason weeks at 1
        _settled(2, 8, 4, "lost"), _settled(3, 7, 4, "won"), _settled(4, 7, 4, "won"),
        _settled(5, 1, 4, "push"), _settled(6, 9, 4, "open"),  # week 9 has nothing settled yet
    ])
    wk = weekly(d, CFG)
    assert [w["label"] for w in wk] == ["Week 1", "Week 7", "Week 8", "Postseason"]
    assert [(w["won"], w["lost"], w["push"]) for w in wk] == [(0, 0, 1), (2, 0, 0), (0, 1, 0), (1, 0, 0)]
    assert [w["dollars"] for w in wk] == pytest.approx([0.0, 20.0, -11.0, 10.0])
    assert [w["cum_dollars"] for w in wk] == pytest.approx([0.0, 20.0, 9.0, 19.0])
    assert [w["start_balance"] for w in wk] == pytest.approx([1000.0, 1000.0, 1020.0, 1009.0])
    assert [w["balance"] for w in wk] == pytest.approx([1000.0, 1020.0, 1009.0, 1019.0])
    assert not any(w["over_bankroll"] for w in wk)
    md = summary_markdown(d, CFG, 2026)
    assert "## Week by week" in md and "| Postseason | 1 | 1-0-0 |" in md and "$1,019" in md
    # A week whose stakes ($22 in week 7) exceed the bankroll going in is flagged.
    small = weekly(d, Bets(bankroll=21.0))
    assert [w["over_bankroll"] for w in small] == [False, True, False, False]
    assert "## By edge tier" in md


def test_bets_with_unknown_week_dont_break_the_summary():
    d = _ledger([_settled(1, None, 4, "won", None), _settled(2, 7, 4, "lost", None)])
    assert [w["label"] for w in weekly(d, CFG)] == ["Week 7", "Week ?"]
    summary_markdown(d, CFG, 2026)


def test_tracked_tests_split_the_ledger_by_fixed_rules():
    from cfbr.bets.experiments import clv_gap, tracked_tests

    def row(gid, week, line, fbs, placed, clv_, status="won", season_type="regular"):
        r = _settled(gid, week, 4, status, season_type)
        return r | {"line": line, "fbs_game": fbs, "placed_at": placed, "clv": clv_}

    early, late = "2026-10-11T14:05:00+00:00", "2026-10-11T20:05:00+00:00"
    d = _ledger([
        row(1, 7, -7.0, True, early, 1.0),                  # core, first capture
        row(2, 7, -24.5, True, early, -0.5, "lost"),        # line > 21: out
        row(3, 7, -3.0, False, late, 0.0, "lost"),          # FCS opponent: out; added at a later run
        row(4, 1, -3.0, True, early, 0.5),                  # week 1: out
        row(5, 1, -3.0, True, early, 0.5, "won", "postseason"),  # postseason "week 1" is not week 1
        row(6, 8, 2.5, None, late, 2.0),                    # unknown FBS status: out; first capture of week 8
    ])
    t1, t2 = tracked_tests(d, CFG)
    assert t1["key"] == "core_filter" and [g["settled"] for g in t1["groups"]] == [2, 4]
    assert t1["groups"][0]["won"] == 2 and t1["n_in"] == 2 and t1["n_out"] == 4
    assert t1["gap"] == pytest.approx(0.75 - 0.5)  # mean(1, .5) - mean(-.5, 0, .5, 2)
    # Second look: only game 3 came after its week's first capture (week 8's lone bet is its first).
    assert t2["key"] == "second_look" and [g["settled"] for g in t2["groups"]] == [5, 1]
    assert not t1["enough_data"] and not t2["enough_data"]
    g = clv_gap(pl.Series([1.0, 2.0, 3.0]), pl.Series([0.0, 1.0, 2.0]))
    assert g["gap"] == pytest.approx(1.0) and g["ci95"][0] < 1.0 < g["ci95"][1]
    assert math.isnan(clv_gap(pl.Series([1.0]), pl.Series([0.0, 1.0]))["gap"])
    md = summary_markdown(d, CFG, 2026)
    assert "## Tracked tests" in md and "### T1 · Core filter" in md and "**No conclusion yet**" in md
    assert "the pre-set minimum is 50 per group" in md


def test_only_pricing_books_set_the_best_line_but_all_books_feed_the_consensus():
    snap = snapshot_frame([{"id": 1, "lines": [
        {"provider": "DraftKings", "spread": -3.5}, {"provider": "FanDuel", "spread": -3.0},
        {"provider": "theScore Bet", "spread": -2.5}, {"provider": "Bovada", "spread": -1.5}]}])
    m = market(snap, Bets().offshore_books, -110, Bets().pricing_books).row(0, named=True)
    assert m["line"] == pytest.approx(-2.75)                                   # all four books
    assert (m["best_home"], m["best_home_provider"]) == (-3.0, "FanDuel")      # theScore's -2.5 can't price
    assert (m["best_away"], m["best_away_provider"]) == (-3.5, "DraftKings")
    only_other = snapshot_frame([{"id": 2, "lines": [{"provider": "theScore Bet", "spread": -2.5}]}])
    assert market(only_other, [], -110, ["DraftKings", "FanDuel", "bet365"])["best_home"].to_list() == [None]
    assert Bets().pricing_books == ["DraftKings", "FanDuel", "bet365"]


def test_what_if_restakes_the_same_bets():
    from cfbr.bets.ledger import what_if

    rows = [_settled(i, 7, 4, "won") for i in range(1, 4)] + [_settled(4, 7, 4, "lost")]   # week 7: 3-1
    rows += [_settled(5, 8, 4, "lost")]                                                      # week 8: one bet, lost
    d = _ledger(rows)
    flat1, flat2, allin = what_if(d, Bets())
    assert flat1["balance"] == pytest.approx(summarize(d, Bets())["balance"])  # the live plan is the record
    assert flat1["balance"] == pytest.approx(1000 + 30 - 11 - 11)
    assert flat2["balance"] == pytest.approx(1000 + 60 - 22 - 22) and not flat2["capped"]
    # All-in: week 7 stakes $250 a bet (3 wins at 10/11, 1 loss), then everything on one losing game.
    assert allin["path"][0] == pytest.approx(1000 + 3 * 250 * 10 / 11 - 250, abs=0.01)  # path is in cents
    assert allin["balance"] == 0 and allin["low"] == 0 and allin["bust"] == "Week 8"
    # A flat plan that would need more than the bankroll is capped, not allowed to go negative.
    tiny = what_if(d, Bets(bankroll=30.0))[1]
    assert tiny["capped"] == ["Week 7"]
    md = summary_markdown(d, Bets(), 2026)
    assert "## What if: staking plans" in md and "bust in Week 8" in md


def test_settle_records_the_closing_source():
    new = candidate_bets(_preds(), _snapshot(), PLACED, CFG).with_columns(season=pl.lit(2026, pl.Int32),
                                                                          week=pl.lit(7, pl.Int32))
    ledger, _ = add_bets(read_ledger(Path("/nonexistent")), new)
    games = pl.DataFrame({"game_id": [1], "completed": [True], "home_points": [31], "away_points": [21]})
    snap = pl.DataFrame({"game_id": [1], "closing_line": [-4.0], "closing_source": ["snapshot"]})
    r = settle(ledger, games, snap).row(0, named=True)
    assert (r["closing_line"], r["closing_source"], r["clv"]) == (-4.0, "snapshot", pytest.approx(1.0))
    r = settle(ledger, games, pl.DataFrame({"game_id": [1], "closing_line": [-4.0]})).row(0, named=True)
    assert r["closing_source"] is None   # older callers without a source
