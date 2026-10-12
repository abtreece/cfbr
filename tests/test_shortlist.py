import polars as pl
import pytest

from cfbr.bets.ledger import LEDGER_SCHEMA
from cfbr.bets.shortlist import BookLine, parse_books, shortlist, shortlist_ids
from cfbr.config import Bets

CFG = Bets()
FIRST, LATER = "2026-10-11T14:05:00+00:00", "2026-10-11T20:05:00+00:00"
REF = frozenset(CFG.pricing_books)


def _row(gid, side="home", line=-7.0, books="DraftKings -7 (-110/-110)", week=7, fbs=True, placed=FIRST,
         season_type="regular", kickoff=None):
    return {"game_id": gid, "season": 2026, "week": week, "season_type": season_type, "side": side,
            "team": f"H{gid}" if side == "home" else f"A{gid}", "home_team": f"H{gid}", "away_team": f"A{gid}",
            "line": line, "edge": 4.0, "model_margin": 11.0, "odds": -110, "stake": 1.0, "status": "open",
            "fbs_game": fbs, "placed_at": placed, "books": books,
            "kickoff_ts": kickoff or f"2026-10-{10 + gid:02d}T23:00:00Z"}


def _ledger(rows):
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def test_parse_books_reads_spread_prices_and_markers():
    got = parse_books("Bovada +3.5 (-115/-105) (offshore); DraftKings +3 (-110/-110); "
                      "bet365 +3.5 (-105/-115) (pricing only); BetMGM +3; garbage")
    assert got == [
        BookLine("Bovada", 3.5, -115, -105, True, False),
        BookLine("DraftKings", 3.0, -110, -110, False, False),
        BookLine("bet365", 3.5, -105, -115, False, True),
        BookLine("BetMGM", 3.0, None, None, False, False),
    ]
    assert parse_books(None) == [] and parse_books("") == []


def test_core_rule():
    d = _ledger([_row(1), _row(2, line=-24.5), _row(3, fbs=False), _row(4, fbs=None), _row(5, week=1),
                 _row(6, week=1, season_type="postseason")])
    assert shortlist(d, REF, CFG)["game_id"].to_list() == [1, 6]  # postseason "week 1" is not week 1


def test_first_capture_rule():
    d = _ledger([_row(1), _row(2, placed=LATER), _row(3, week=8, placed=LATER)])
    # Game 2 came after week 7's first capture; game 3 is week 8's only (so first) capture.
    assert shortlist(d, REF, CFG)["game_id"].to_list() == [1, 3]


def test_your_number_or_better_from_each_side():
    d = _ledger([
        _row(1, "home", -7.0, "DraftKings -7.5 (-110/-110)"),   # home wants a higher spread: -7.5 is worse
        _row(2, "home", -7.0, "DraftKings -6.5 (-110/-110)"),   # better
        _row(3, "away", 7.0, "DraftKings +7.5 (-110/-110)"),    # away wants a lower spread: +7.5 is worse
        _row(4, "away", 7.0, "DraftKings +6.5 (-110/-110)"),    # better: away lays 6.5 instead of 7
    ])
    s = shortlist(d, REF, CFG)
    assert s["game_id"].to_list() == [2, 4]
    assert s["bet_at"].to_list() == [-6.5, -6.5] and s["bet_at_book"].to_list() == ["DraftKings"] * 2


def test_only_the_viewers_books_and_never_offshore():
    books = "Bovada -6 (-110/-110) (offshore); BetMGM -6.5 (-110/-110); DraftKings -7.5 (-110/-110)"
    d = _ledger([_row(1, "home", -7.0, books), _row(2, "home", -7.0, None)])
    assert shortlist(d, REF, CFG).is_empty()                       # DK is short; Bovada never counts; null books out
    s = shortlist(d, {"BetMGM"}, CFG)
    assert s["game_id"].to_list() == [1] and s["bet_at_book"].to_list() == ["BetMGM"]
    assert shortlist(d, {"Bovada"}, CFG).is_empty()


def test_bet365_pricing_only_entries_count_and_names_match_by_key():
    d = _ledger([_row(1, "home", -7.0, "DraftKings -7.5 (-110/-110); bet365 -7 (-105/-115) (pricing only)")])
    s = shortlist(d, {"Bet365"}, CFG)
    assert s["bet_at_book"].to_list() == ["bet365"] and s["bet_at_odds"].to_list() == [-105]


def test_bet_at_prefers_points_then_price_without_a_table():
    books = "DraftKings -7 (-115/-105); FanDuel -7 (-105/-115); BetMGM -6.5 (-125/+105)"
    d = _ledger([_row(1, "home", -7.0, books)])
    s = shortlist(d, {"DraftKings", "FanDuel"}, CFG)
    assert (s["bet_at"][0], s["bet_at_book"][0], s["bet_at_odds"][0]) == (-7.0, "FanDuel", -105)
    s = shortlist(d, {"DraftKings", "FanDuel", "BetMGM"}, CFG)
    assert (s["bet_at"][0], s["bet_at_book"][0]) == (-6.5, "BetMGM")


def test_unpriced_entries_count_as_the_default_odds():
    d = _ledger([_row(1, "home", -7.0, "DraftKings -7; FanDuel -7 (-112/-108)")])
    s = shortlist(d, {"DraftKings", "FanDuel"}, CFG)
    assert s["bet_at_book"].to_list() == ["DraftKings"] and s["bet_at_odds"].to_list() == [None]


def test_bet_at_uses_expected_value_with_a_table():
    class Table:  # half a point is worth less than 20 cents here
        def expected_value(self, side, number, odds, center):
            payout = odds / 100 if odds > 0 else 100 / -odds
            return 0.5 * payout - 0.5 + (number - center) * 0.01

    books = "DraftKings -6.5 (-130/+110); FanDuel -7 (-105/-115)"
    d = _ledger([_row(1, "home", -7.0, books)])
    assert shortlist(d, {"DraftKings", "FanDuel"}, CFG)["bet_at_book"].to_list() == ["DraftKings"]
    assert shortlist(d, {"DraftKings", "FanDuel"}, CFG, Table())["bet_at_book"].to_list() == ["FanDuel"]


def test_shortlist_ids_use_the_reference_books():
    d = _ledger([_row(1), _row(2, books="BetMGM -6 (-110/-110)"), _row(3, line=-30.0)])
    assert shortlist_ids(d, CFG) == [1]


def test_sorted_by_kickoff_and_keeps_ledger_columns():
    d = _ledger([_row(1, kickoff="2026-10-18T00:00:00Z"), _row(2, kickoff="2026-10-17T16:00:00Z")])
    s = shortlist(d, REF, CFG)
    assert s["game_id"].to_list() == [2, 1]
    assert set(LEDGER_SCHEMA) <= set(s.columns) and not any(c.startswith("_") for c in s.columns)


def test_empty_pricing_books_means_any_non_offshore_book():
    cfg = Bets(pricing_books=[])
    d = _ledger([
        _row(1, books="BetMGM -7 (-110/-110); Bovada -6.5 (-110/-110) (offshore)"),
        _row(2, books="Bovada -6.5 (-110/-110) (offshore)"),
    ])
    assert shortlist_ids(d, cfg) == [1]
    assert shortlist(d, cfg.pricing_books, cfg)["bet_at_book"].to_list() == ["BetMGM"]
