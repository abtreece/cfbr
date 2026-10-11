import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cfbr.bets.ledger import candidate_bets, market, snapshot_frame
from cfbr.config import Bets, Config, Paths
from cfbr.ingest import oddspapi
from cfbr.ingest.odds import team_aliases
from cfbr.ingest.oddspapi import main_spread, pricing_lines

KICK = datetime(2026, 10, 17, 19, 30, tzinfo=UTC)
CATALOG = {  # handicap is from participant 1's side; outcome "1" = participant 1
    "14312": {"handicap": 6.5, "period": "result", "outcomes": {"14312": "1", "14313": "2"}},
    "14314": {"handicap": 7.0, "period": "result", "outcomes": {"14314": "1", "14315": "2"}},
    "99": {"handicap": 3.5, "period": "1h", "outcomes": {"991": "1", "992": "2"}},
}


def _odds(slug="bet365", main=("14312", "14313"), suspended=False):
    def o(price, mainline):
        return {"players": {"0": {"priceAmerican": price, "mainLine": mainline, "active": True}}}
    return {"bookmakerOdds": {slug: {"bookmakerIsActive": True, "suspended": suspended, "markets": {
        "99": {"outcomes": {"991": o("-110", True), "992": o("-110", True)}},         # first half: ignored
        "14314": {"outcomes": {"14314": o("-140", False), "14315": o("+115", False)}},  # alternate line
        "14312": {"outcomes": {main[0]: o("-109", True), main[1]: o("-120", True)}},
    }}}}


def test_main_spread_reads_the_full_game_main_line():
    assert main_spread(_odds(), "bet365", CATALOG) == (6.5, -109, -120)
    assert main_spread(_odds(), "fanduel", CATALOG) is None
    assert main_spread(_odds(suspended=True), "bet365", CATALOG) is None


def test_pricing_only_lines_can_set_the_best_line_but_never_the_consensus():
    snap = snapshot_frame([{"id": 1, "lines": [{"provider": "DraftKings", "spread": 7.0},
                                               {"provider": "FanDuel", "spread": 6.5}]}])
    extra = pl.DataFrame({"game_id": [1], "provider": ["bet365"], "spread": [7.5], "home_price": [-120],
                          "away_price": [100]}).cast({"home_price": pl.Int32, "away_price": pl.Int32})
    cfg = Bets()
    m = market(snap, cfg.offshore_books, cfg.odds, cfg.pricing_books, extra).row(0, named=True)
    assert m["line"] == pytest.approx(6.75) and m["n_providers"] == 2       # bet365 left out
    assert (m["best_home"], m["best_home_provider"], m["best_home_price"]) == (7.5, "bet365", -120)
    assert "bet365 +7.5 (-120/+100) (pricing only)" in m["books"]
    preds = pl.DataFrame({"game_id": [1], "kickoff_ts": ["2026-10-17T19:30:00.000000+0000"], "home_team": ["H"],
                          "away_team": ["A"], "pred_margin": [-1.0]})   # model: home +1 vs market home -6.75 -> home bet
    r = candidate_bets(preds, snap, datetime(2026, 10, 11, tzinfo=UTC), cfg, extra).row(0, named=True)
    assert (r["side"], r["line"], r["best_line"], r["best_provider"], r["best_odds"]) == ("home", 6.75, 7.5, "bet365", -120)


def _cfg(tmp_path, **kw):
    cfg = Config(paths=Paths(raw_dir=tmp_path, db_path=tmp_path / "x.duckdb", results_dir=tmp_path,
                             out_dir=tmp_path / "out"))
    for k, v in kw.items():
        setattr(cfg.oddspapi, k, v)
    return cfg


def test_pricing_lines_match_fixtures_flip_neutral_sites_and_respect_the_budget(tmp_path, monkeypatch):
    calls = []
    fixtures = [
        {"fixtureId": "f1", "startTime": "2026-10-17T19:30:00.000Z", "hasOdds": True,
         "participant1Name": "Tennessee Volunteers", "participant2Name": "Alabama Crimson Tide"},
        {"fixtureId": "f2", "startTime": "2026-10-17T23:00:00.000Z", "hasOdds": True,   # listed the other way round
         "participant1Name": "Texas Longhorns", "participant2Name": "Oklahoma Sooners"},
    ]

    def fake_get(cfg, path, params):
        calls.append(path)
        with open(oddspapi._ledger(cfg.paths.raw_dir), "a") as f:
            f.write(json.dumps({"ts": datetime.now(UTC).isoformat(), "endpoint": path}) + "\n")
        return fixtures if path == "/fixtures" else _odds()

    monkeypatch.setattr(oddspapi, "_get", fake_get)
    (tmp_path / "oddspapi").mkdir()
    (tmp_path / "oddspapi" / "markets_spreads.json").write_text(json.dumps(CATALOG))
    aliases = team_aliases([{"school": s, "mascot": m, "alternateNames": []} for s, m in
                            (("Tennessee", "Volunteers"), ("Alabama", "Crimson Tide"), ("Texas", "Longhorns"),
                             ("Oklahoma", "Sooners"))])
    games = pl.DataFrame({"game_id": [1, 2], "home_team": ["Tennessee", "Oklahoma"], "away_team": ["Alabama", "Texas"],
                          "kickoff_ts": [KICK, KICK + timedelta(hours=3)]})
    now = datetime(2026, 10, 11, 14, tzinfo=UTC)
    out = {r["game_id"]: r for r in pricing_lines(_cfg(tmp_path), 2026, games, aliases, now).iter_rows(named=True)}
    assert calls == ["/fixtures", "/odds", "/odds"]
    assert out[1] == {"game_id": 1, "provider": "bet365", "spread": 6.5, "home_price": -109, "away_price": -120}
    # Game 2: OddsPapi lists Texas first; CFBD's home team is Oklahoma, so the sign and prices flip.
    assert (out[2]["spread"], out[2]["home_price"], out[2]["away_price"]) == (-6.5, -120, -109)
    snap = json.loads(next((tmp_path / "oddspapi" / "2026" / "snapshots").glob("*.json")).read_text())
    assert snap["fixtures"] == ["f1", "f2"] and len(snap["lines"]) == 2

    # Budget: 250 - 20 reserve - 3 used = 227; with a quota leaving room for the fixture list + 1 game only,
    # the soonest game is fetched and the other skipped.
    calls.clear()
    tight = _cfg(tmp_path, monthly_quota=20 + 3 + 2)
    got = pricing_lines(tight, 2026, games, aliases, now)
    assert calls == ["/fixtures", "/odds"] and got["game_id"].to_list() == [1]


def test_budget_guard_refuses_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDSPAPI_KEY", "test-key")
    monkeypatch.setattr(oddspapi.httpx, "get", lambda *a, **k: pytest.fail("no request past the budget"))
    cfg = _cfg(tmp_path)
    month = datetime.now(UTC).isoformat()
    oddspapi._ledger(tmp_path).write_text(
        json.dumps({"ts": "2020-01-01T00:00:00+00:00"}) + "\n"            # an old month doesn't count
        + "".join(json.dumps({"ts": month}) + "\n" for _ in range(230)))  # 250 - 20 reserve = 230 used
    assert oddspapi.requests_left(cfg) == 0
    with pytest.raises(oddspapi.OddsPapiBudget):
        oddspapi._get(cfg, "/fixtures", {})
