import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cfbr.bets.ledger import (add_bets, book_key, candidate_bets, combine_books, market, read_ledger, settle,
                              snapshot_frame, summarize)
from cfbr.config import Bets, Config, Paths
from cfbr.ingest import odds
from cfbr.ingest.odds import credits_left, match_events, norm_name, odds_frame, team_aliases

CFG = Bets()
KICK = datetime(2026, 10, 17, 19, 30, tzinfo=UTC)
TEAMS = [
    {"school": "Alabama", "mascot": "Crimson Tide", "alternateNames": ["Bama"]},
    {"school": "Tennessee", "mascot": "Volunteers", "alternateNames": []},
    {"school": "Texas", "mascot": "Longhorns", "alternateNames": []},
    {"school": "Texas Tech", "mascot": "Red Raiders", "alternateNames": []},
    {"school": "San José State", "mascot": "Spartans", "alternateNames": []},
    {"school": "App State", "mascot": "Mountaineers", "alternateNames": ["Appalachian State"]},
    {"school": "Hawai'i", "mascot": "Rainbow Warriors", "alternateNames": []},
]
GAMES = pl.DataFrame({
    "game_id": [1, 2, 3, 4, 5],
    "home_team": ["Tennessee", "Texas Tech", "San José State", "App State", "Texas"],
    "away_team": ["Alabama", "Texas", "Hawai'i", "Delaware", "Texas Tech"],
    "kickoff_ts": [KICK, KICK, KICK, KICK, KICK + timedelta(days=60)],  # game 5: a later rematch
})


def _book(title, home_name, away_name, home_pt, home_price=-110, away_price=-110):
    return {"key": title.lower(), "title": title, "markets": [{"key": "spreads", "outcomes": [
        {"name": home_name, "point": home_pt, "price": home_price},
        {"name": away_name, "point": -home_pt, "price": away_price}]}]}


def _event(eid, home, away, books, t=KICK):
    return {"id": eid, "commence_time": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": home, "away_team": away,
            "bookmakers": books}


def test_name_normalization():
    assert norm_name("San José State Spartans") == "san jose state spartans"
    assert norm_name("Hawai'i Rainbow Warriors") == norm_name("Hawaii Rainbow Warriors")
    assert norm_name("Texas A&M Aggies") == "texas a and m aggies"


def test_events_match_games_by_both_teams_and_kickoff():
    al = team_aliases(TEAMS)
    ev = [
        _event("a", "Tennessee Volunteers", "Alabama Crimson Tide", []),
        _event("b", "Texas Tech Red Raiders", "Texas Longhorns", []),        # "Texas" is a prefix of both
        _event("c", "San Jose State Spartans", "Hawaii Rainbow Warriors", []),
        _event("d", "Appalachian State Mountaineers", "Delaware Blue Hens", []),  # alternate name; FCS by prefix
        _event("e", "Alabama Crimson Tide", "Tennessee Volunteers", []),     # neutral site, home/away swapped
        _event("f", "Ohio State Buckeyes", "Michigan Wolverines", []),       # not in CFBD games
    ]
    m = match_events(ev, GAMES, al)
    assert m == {"a": (1, False), "b": (2, False), "c": (3, False), "d": (4, False), "e": (1, True)}


def test_odds_frame_is_home_perspective_with_prices():
    al = team_aliases(TEAMS)
    ev = [_event("a", "Tennessee Volunteers", "Alabama Crimson Tide",
                 [_book("FanDuel", "Tennessee Volunteers", "Alabama Crimson Tide", 6.5, -105, -115)]),
          # Same game listed the other way round: Alabama -6.5 must still come out as home (Tennessee) +6.5.
          _event("e", "Alabama Crimson Tide", "Tennessee Volunteers",
                 [_book("BetRivers", "Alabama Crimson Tide", "Tennessee Volunteers", -6.5, -112, -108)])]
    f = {r["provider"]: r for r in odds_frame(ev, GAMES, al).iter_rows(named=True)}
    assert f["FanDuel"] == {"game_id": 1, "provider": "FanDuel", "spread": 6.5, "home_price": -105, "away_price": -115}
    assert f["BetRivers"]["spread"] == 6.5
    assert (f["BetRivers"]["home_price"], f["BetRivers"]["away_price"]) == (-108, -112)


def test_feeds_merge_one_row_per_book_priced_feed_first():
    priced = pl.DataFrame({"game_id": [1, 1], "provider": ["DraftKings", "BetOnline.ag"], "spread": [-3.0, -3.5],
                           "home_price": [-105, -110], "away_price": [-115, -110]})
    cfbd = snapshot_frame([{"id": 1, "lines": [{"provider": "DraftKings", "spread": -2.5},
                                               {"provider": "Bovada", "spread": -3.0}]}])
    c = {r["provider"]: r for r in combine_books(priced, cfbd).iter_rows(named=True)}
    assert set(c) == {"DraftKings", "BetOnline.ag", "Bovada"}
    assert c["DraftKings"]["spread"] == -3.0 and c["DraftKings"]["home_price"] == -105
    assert book_key("BetOnline.ag") == book_key("BetOnline")


def test_best_line_breaks_spread_ties_on_price_and_skips_offshore():
    snap = pl.DataFrame({
        "game_id": [1, 1, 1, 1], "provider": ["DraftKings", "FanDuel", "theScore Bet", "BetOnline.ag"],
        "spread": [-3.0, -3.0, -3.5, -2.0], "home_price": [-120, -105, -110, +100],
        "away_price": [+100, -115, -110, -120]}).cast({"home_price": pl.Int32, "away_price": pl.Int32})
    m = market(snap, CFG.offshore_books).row(0, named=True)
    # Home bettor: -3 beats -3.5; FanDuel's -105 beats DraftKings' -120. BetOnline's -2 is offshore.
    assert (m["best_home"], m["best_home_provider"], m["best_home_price"]) == (-3.0, "FanDuel", -105)
    # Away bettor: +3.5 at theScore beats +3 anywhere.
    assert (m["best_away"], m["best_away_provider"], m["best_away_price"]) == (-3.5, "theScore Bet", -110)
    assert m["line"] == pytest.approx(-3.0) and m["n_providers"] == 4
    assert "FanDuel -3 (-105/-115)" in m["books"] and "BetOnline.ag -2 (+100/-120) (offshore)" in m["books"]


def test_best_line_graded_at_the_books_price():
    snap = pl.DataFrame({"game_id": [1, 1], "provider": ["FanDuel", "DraftKings"], "spread": [-2.5, -3.0],
                         "home_price": [+105, -110], "away_price": [-125, -110]}
                        ).cast({"home_price": pl.Int32, "away_price": pl.Int32})
    preds = pl.DataFrame({"game_id": [1], "kickoff_ts": ["2026-10-17T19:30:00.000000+0000"], "home_team": ["H"],
                          "away_team": ["A"], "pred_margin": [9.0]})
    new = candidate_bets(preds, snap, datetime(2026, 10, 11, 14, tzinfo=UTC), CFG).with_columns(
        season=pl.lit(2026, pl.Int32), week=pl.lit(7, pl.Int32))
    r = new.row(0, named=True)
    assert (r["best_line"], r["best_provider"], r["best_odds"], r["odds"]) == (-2.5, "FanDuel", 105, -110)
    ledger, _ = add_bets(read_ledger(odds.REPO_ROOT / "nonexistent.csv"), new)
    games = pl.DataFrame({"game_id": [1], "completed": [True], "home_points": [24], "away_points": [21]})
    out = settle(ledger, games, pl.DataFrame({"game_id": [1], "closing_line": [-3.0]})).row(0, named=True)
    # Won by 3: a win at both -2.75 consensus and -2.5; the best line pays +105 (1 risked, 1.05 won).
    assert out["status"] == "won" and out["units"] == pytest.approx(1.0)
    assert out["best_status"] == "won" and out["best_units"] == pytest.approx(1.05)
    lost = settle(ledger, games.with_columns(home_points=pl.lit(20)), pl.DataFrame(
        {"game_id": [1], "closing_line": [-3.0]}))
    assert lost.row(0, named=True)["best_units"] == pytest.approx(-1.0)  # risked 1 at +105, not 1.1
    assert summarize(lost, CFG, "best")["risked"] == pytest.approx(1.0)
    assert summarize(lost, CFG)["risked"] == pytest.approx(1.1)


def _cfg(tmp_path):
    return Config(paths=Paths(raw_dir=tmp_path, db_path=tmp_path / "x.duckdb", results_dir=tmp_path,
                              out_dir=tmp_path / "out"))


def test_credits_left_and_quota_guard(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    assert credits_left(tmp_path, 500) == 500
    month = datetime.now(UTC).isoformat()
    (tmp_path / "_odds_calls.jsonl").write_text(
        json.dumps({"ts": "2020-01-01T00:00:00+00:00", "remaining": "3", "last": "2"}) + "\n"  # old month: ignored
        + json.dumps({"ts": month, "remaining": None, "last": "2"}) + "\n")
    assert credits_left(tmp_path, 500) == 498
    with open(tmp_path / "_odds_calls.jsonl", "a") as f:
        f.write(json.dumps({"ts": month, "remaining": "51", "last": "2"}) + "\n")
    assert credits_left(tmp_path, 500) == 51
    monkeypatch.setattr(odds, "_get", lambda *a, **k: pytest.fail("must not call the API under the reserve"))
    with pytest.raises(odds.OddsQuotaExceeded):
        odds.fetch_odds_snapshot(cfg, 2026)  # 51 - 2 credits would dip under the 50 reserve


def test_current_books_merges_feeds_only_when_close_in_time(tmp_path):
    from cfbr.bets.run import current_books

    cfg = _cfg(tmp_path)
    lines = tmp_path / "lines" / "2026" / "snapshots"
    oddsd = tmp_path / "odds" / "2026" / "snapshots"
    teams = tmp_path / "teams_fbs" / "2026"
    for d in (lines, oddsd, teams):
        d.mkdir(parents=True)
    (teams / "season.json").write_text(json.dumps(TEAMS))
    (lines / "20261011T140000Z.json").write_text(json.dumps(
        [{"id": 1, "lines": [{"provider": "DraftKings", "spread": 6.0}]}]))
    ev = [_event("a", "Tennessee Volunteers", "Alabama Crimson Tide",
                 [_book("FanDuel", "Tennessee Volunteers", "Alabama Crimson Tide", 6.5)])]
    (oddsd / "20261011T141500Z.json").write_text(json.dumps(ev))
    placed, books = current_books(cfg, GAMES, 2026)
    assert placed == datetime(2026, 10, 11, 14, 15, tzinfo=UTC)  # the later snapshot
    assert sorted(books["provider"]) == ["DraftKings", "FanDuel"]

    (oddsd / "20261011T141500Z.json").rename(oddsd / "20261010T060000Z.json")  # 8 h before CFBD's: stale
    placed, books = current_books(cfg, GAMES, 2026)
    assert placed == datetime(2026, 10, 11, 14, 0, tzinfo=UTC) and books["provider"].to_list() == ["DraftKings"]
