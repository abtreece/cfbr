"""Web app: data build agrees with `cfbr rank`; server pages, API, escaping, reload."""

import json
import math
import os
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from cfbr.config import load_config
from cfbr.web.app import create_app, default_slot, game_row
from cfbr.web.build import FINAL, build_season, web_dir
from cfbr.web.store import Store

SEASON = 2026
TEAMS = ["Texas A&M", "Hawai'i", "<b>Evil</b>", "Ohio State"]
T0 = datetime(2026, 9, 5, 16, tzinfo=UTC)


# -- synthetic web data -------------------------------------------------------------

def _ratings() -> pl.DataFrame:
    rows = []
    for week, label in ((0, "Preseason"), (1, "Week 1")):
        for i, team in enumerate(TEAMS):
            rows.append({
                "season": SEASON, "week": week, "label": label, "as_of": T0 + timedelta(days=7 * week),
                "team": team, "conference": "SEC" if i % 2 else "Big Ten", "fbs": True, "rank": i + 1,
                "w": week if i < 2 else 0, "l": 0 if i < 2 else week, "rating": 20.0 - 5 * i + week,
                "offense": 10.0 - i, "defense": 10.0 - 4 * i + week, "prior_share": 1.0 - 0.3 * week,
                "rating_change": None if week == 0 else 1.0, "rank_change": None if week == 0 else 0,
                "o_ppa": 0.1, "d_ppa": 0.1, "o_sr": 0.05, "d_sr": 0.05,
            })
    rows.append({**rows[-1], "team": "FCS:Big Sky", "conference": None, "fbs": False, "rank": None,
                 "rating": -20.0, "rating_change": 0.0, "rank_change": None})
    return pl.DataFrame(rows, schema_overrides={"rank": pl.Int32, "rank_change": pl.Int32,
                                                "rating_change": pl.Float64})


def _game(gid, slot, home, away, kickoff, completed, hp=None, ap=None, pred=3.0, spread=-2.5):
    order = int(slot[1:])
    return {
        "game_id": gid, "season": SEASON, "week": order, "season_type": "regular", "slot": slot,
        "slot_order": order, "as_of": kickoff, "kickoff_ts": kickoff, "completed": completed,
        "neutral_site": False, "home": home, "away": away, "home_team": home,
        "away_team": "Montana" if away.startswith("FCS") else away,
        "margin": None if hp is None else float(hp - ap), "pred_margin": pred,
        "win_prob": 1 / (1 + math.exp(-pred / 8)), "prior_pred": pred, "massey_pred": pred - 1, "sigma": 16.0,
        "prior_share_h": 0.5, "prior_share_a": 0.5, "home_points": hp, "away_points": ap,
        "home_conference": "SEC", "away_conference": "SEC", "spread": spread,
    }


def _games(now: datetime) -> pl.DataFrame:
    return pl.DataFrame([
        _game(1, "r01", "Texas A&M", "<b>Evil</b>", T0, True, 31, 10, pred=10.0, spread=-7.0),
        _game(2, "r01", "Hawai'i", "Ohio State", T0, True, 17, 24, pred=-4.0, spread=None),
        _game(3, "r02", "Ohio State", "Texas A&M", now + timedelta(days=2), False, pred=0.01, spread=-3.0),
        _game(4, "r02", "Hawai'i", "FCS:Big Sky", now + timedelta(days=2), False, pred=30.0, spread=float("nan")),
    ], schema_overrides={"home_points": pl.Int64, "away_points": pl.Int64, "margin": pl.Float64,
                         "spread": pl.Float64})


def _write_season(root, now, live=True):
    d = root / str(SEASON)
    d.mkdir(parents=True, exist_ok=True)
    _ratings().write_parquet(d / "ratings.parquet")
    _games(now).write_parquet(d / "games.parquet")
    (d / "meta.json").write_text(json.dumps({
        "season": SEASON, "built_at": now.isoformat(), "config": "test", "live": live,
        "weeks": [0, 1], "labels": {"0": "Preseason", "1": "Week 1"}}))
    return d


@pytest.fixture
def web_cfg(tmp_path):
    cfg = load_config()
    cfg.paths.out_dir = tmp_path / "out"
    cfg.paths.results_dir = tmp_path / "results"
    return cfg


@pytest.fixture
def client(web_cfg):
    _write_season(web_dir(web_cfg), datetime.now(UTC))
    run = web_cfg.paths.results_dir / "20260101T000000Z-test"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"seasons": [2015, 2021], "overall": {
        "n_games": 10, "mae": 13.0, "line_mae": 12.0, "massey_mae": 15.0, "ats_n": 9, "ats_win_rate": 0.5,
        "ats_ci95_low": 0.2, "ats_ci95_high": 0.8}}))
    return TestClient(create_app(web_cfg))


# -- pages ------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "/rankings", "/rankings?season=2026&week=0", "/rankings?season=2026&week=1&conf=SEC",
    "/team/2026/Texas%20A%26M", "/team/2026/Hawai%27i", "/games", "/games?season=2026&slot=r01", "/accuracy",
])
def test_pages_render(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text
    assert "<html" in r.text


def test_index_redirects_to_rankings(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/rankings"


def test_rankings_default_to_latest_week_and_link_teams(client):
    html = client.get("/rankings").text
    assert "2026 ratings · Week 1" in html
    assert 'href="/team/2026/Texas%20A%26M"' in html
    assert 'href="/team/2026/Hawai%27i"' in html
    assert "FCS:Big Sky" not in html  # pooled FCS entities are not ranked


def test_conference_filter(client):
    html = client.get("/rankings?season=2026&week=1&conf=SEC").text
    assert "Hawai&#39;i" in html and "Texas A&amp;M" not in html.split("<tbody>")[1]


def test_team_names_are_escaped(client):
    for url in ("/rankings", "/games?slot=r01", "/team/2026/%3Cb%3EEvil%3C%2Fb%3E"):
        html = client.get(url).text
        assert "<b>Evil</b>" not in html
        assert "&lt;b&gt;Evil&lt;/b&gt;" in html


def test_team_page_takes_team_perspective(client):
    html = client.get("/team/2026/Ohio%20State").text
    # Ohio State won 24-17 at Hawai'i, which was favored by 4 -> Ohio State's model margin is -4.0.
    assert "24–17" in html and "−4.0" in html


def test_upcoming_slot_is_default_in_live_season(client):
    html = client.get("/games").text
    assert "2026 · Week 2" in html
    assert "Toss-up" in html  # |pred| < 0.05
    assert "Final</th>" not in html  # no result columns before any game is final


def test_missing_things_404(client):
    for url in ("/team/2026/Nobody", "/rankings?season=1999", "/rankings?week=7", "/games?slot=r09"):
        assert client.get(url).status_code == 404, url
    r = client.get("/api/team/2026/Nobody")
    assert r.status_code == 404 and r.json()["detail"].startswith("No team")


def test_no_data_is_503(web_cfg):
    r = TestClient(create_app(web_cfg)).get("/rankings")
    assert r.status_code == 503 and "cfbr web build" in r.text


def test_accuracy_lists_backtest_runs(client):
    html = client.get("/accuracy").text
    assert "20260101T000000Z-test" in html
    assert "Model MAE" in html


# -- API ----------------------------------------------------------------------------

def test_api_rankings(client):
    body = client.get("/api/rankings?season=2026&week=1").json()
    assert body["week"] == 1 and body["label"] == "Week 1"
    assert [t["team"] for t in body["teams"]] == TEAMS
    assert "trend" not in body["teams"][0]


def test_api_games_is_valid_json_with_nan_as_null(client):
    r = client.get("/api/games?season=2026&slot=r02")
    assert r.status_code == 200
    spreads = {g["game_id"]: g["spread"] for g in r.json()["games"]}
    assert spreads[4] is None


def test_api_team(client):
    body = client.get("/api/team/2026/Ohio%20State").json()
    assert [w["week"] for w in body["weeks"]] == [0, 1]
    g = {x["game_id"]: x for x in body["games"]}
    assert g[2]["t_pred"] == 4.0 and g[2]["t_margin"] == 7.0 and not g[2]["is_home"]


# -- game rows, store ----------------------------------------------------------------

def _row(pred, spread, margin, completed=True):
    return game_row({"home_team": "H", "away_team": "A", "pred_margin": pred, "win_prob": 0.6, "spread": spread,
                     "margin": margin, "completed": completed})


def test_game_row_ats_grading():
    assert _row(8.0, -3.0, 10.0)["ats"] == "win"     # model H by 8, line H by 3, H won by 10
    assert _row(2.0, 0.0, -3.0)["ats"] == "loss"
    assert _row(7.0, -7.0, 3.0)["ats"] == "no pick"  # model agrees with the line
    assert _row(1.0, -4.0, 4.0)["ats"] == "push"
    assert _row(1.0, None, 4.0)["ats"] is None and _row(1.0, None, 4.0)["vegas_err"] is None
    assert _row(1.0, -4.0, None, completed=False)["model_err"] is None


def test_game_row_labels():
    r = _row(-6.5, 3.0, None, completed=False)
    assert r["model_pick"] == "A by 6.5" and r["vegas_pick"] == "A −3" and r["edge"] == "3.5 toward A"
    assert _row(1.0, 0.0, None, completed=False)["vegas_pick"] == "Pick'em"
    assert _row(-3.0, 3.0, None, completed=False)["edge"] == "agrees"


def test_default_slot_after_season_is_last(web_cfg):
    now = datetime.now(UTC)
    _write_season(web_dir(web_cfg), now, live=False)
    s = Store(web_dir(web_cfg), web_cfg.paths.results_dir, web_cfg.paths.out_dir).season(SEASON)
    assert default_slot(s, now) == "r02"


def test_store_reloads_when_meta_changes(web_cfg):
    root = web_dir(web_cfg)
    d = _write_season(root, datetime.now(UTC))
    store = Store(root, web_cfg.paths.results_dir, web_cfg.paths.out_dir)
    assert store.season(SEASON).ratings["rating"].max() == 21.0
    _ratings().with_columns(rating=pl.col("rating") + 100).write_parquet(d / "ratings.parquet")
    assert store.season(SEASON).ratings["rating"].max() == 21.0  # cached until meta.json changes
    meta = d / "meta.json"
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert store.season(SEASON).ratings["rating"].max() == 121.0


# -- real data: the web build matches `cfbr rank` -----------------------------------

@pytest.fixture(scope="module")
def engine(cfg, dataset):
    from cfbr.ingest.fetch import current_season
    from cfbr.model.engine import Engine

    return Engine(dataset, cfg, last_completed=current_season() - 1)


def test_web_ratings_match_rank_output(engine, tmp_path):
    from cfbr.report.rank import build_rankings

    season, week = 2023, 7
    build_season(engine, season, tmp_path, now=datetime.now(UTC))
    web = pl.read_parquet(tmp_path / str(season) / "ratings.parquet").filter(
        (pl.col("week") == week) & pl.col("fbs"))
    ref, _ = build_rankings(engine, season, week)
    m = ref.join(web, on="team", suffix="_web")
    assert m.height == ref.height == web.height
    assert (m["rank"] == m["rank_web"]).all()
    assert (m["rating"] - m["rating_web"].round(2)).abs().max() == 0
    assert (m["rating_change"] - m["rating_change_web"].round(2)).abs().max() == 0
    assert (m["rank_change"] == m["rank_change_web"]).all()
    assert (m["record"] == m["w"].cast(pl.Utf8) + "-" + m["l"].cast(pl.Utf8)).all()


def test_completed_season_timeline(engine, tmp_path):
    season = 2023
    build_season(engine, season, tmp_path, now=datetime.now(UTC))
    meta = json.loads((tmp_path / str(season) / "meta.json").read_text())
    assert meta["weeks"][0] == 0 and meta["weeks"][-1] == FINAL and not meta["live"]
    r = pl.read_parquet(tmp_path / str(season) / "ratings.parquet")
    as_of = r.group_by("week").agg(pl.col("as_of").first()).sort("week")["as_of"].to_list()
    assert as_of == sorted(as_of)
    g = pl.read_parquet(tmp_path / str(season) / "games.parquet")
    # The last regular-season table comes before the first bowl game.
    last_reg = r.filter(pl.col("week") == meta["weeks"][-2])["as_of"][0]
    assert last_reg <= g.filter(pl.col("season_type") == "postseason")["kickoff_ts"].min()
