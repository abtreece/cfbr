"""Web app: data build agrees with `cfbr rank`; server pages, API, escaping, reload."""

import json
import math
import os
from datetime import UTC, datetime, timedelta
from urllib.parse import quote
from zoneinfo import ZoneInfo

import polars as pl
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cfbr.config import load_config
from cfbr.web import MODEL_NAME
from cfbr.web.app import create_app, default_slot, game_row, poll_note
from cfbr.web.build import FINAL, build_season, timeline_polls, web_dir
from cfbr.web.store import Season, Store, _ensure

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


def _games(now: datetime, spreads: dict | None = None) -> pl.DataFrame:
    df = pl.DataFrame([
        _game(1, "r01", "Texas A&M", "<b>Evil</b>", T0, True, 31, 10, pred=10.0, spread=-7.0),
        _game(2, "r01", "Hawai'i", "Ohio State", T0, True, 17, 24, pred=-4.0, spread=None),
        _game(3, "r02", "Ohio State", "Texas A&M", now + timedelta(days=2), False, pred=0.01, spread=-3.0),
        _game(4, "r02", "Hawai'i", "FCS:Big Sky", now + timedelta(days=2), False, pred=30.0, spread=float("nan")),
    ], schema_overrides={"home_points": pl.Int64, "away_points": pl.Int64, "margin": pl.Float64,
                         "spread": pl.Float64})
    if spreads:
        df = df.with_columns(spread=pl.col("game_id").replace_strict(spreads, default=None).fill_null(pl.col("spread")))
    return df


AP = {"Texas A&M": 3, "Ohio State": 11}


def _write_season(root, now, live=True, polls=False, spreads=None):
    """Synthetic season; ``polls=False`` mimics web data built before poll columns existed."""
    d = root / str(SEASON)
    d.mkdir(parents=True, exist_ok=True)
    ratings, games = _ratings(), _games(now, spreads)
    meta = {"season": SEASON, "built_at": now.isoformat(), "config": "test", "live": live,
            "weeks": [0, 1], "labels": {"0": "Preseason", "1": "Week 1"}}
    if polls:
        ap = pl.DataFrame({"team": list(AP), "ap_rank": list(AP.values())}, schema_overrides={"ap_rank": pl.Int32})
        ratings = ratings.join(ap, on="team", how="left").with_columns(
            coaches_rank=pl.col("ap_rank") + 1, cfp_rank=pl.lit(None, pl.Int32))
        games = (games.join(ap.rename({"team": "home_team", "ap_rank": "home_ap_rank"}), on="home_team", how="left")
                 .join(ap.rename({"team": "away_team", "ap_rank": "away_ap_rank"}), on="away_team", how="left"))
        used = {"ap_rank": 2, "coaches_rank": 1, "cfp_rank": None}
        meta["polls"] = {"0": {**used, "ap_rank": 1}, "1": used}
    ratings.write_parquet(d / "ratings.parquet")
    games.write_parquet(d / "games.parquet")
    (d / "meta.json").write_text(json.dumps(meta))
    return d


@pytest.fixture
def web_cfg(tmp_path):
    cfg = load_config()
    cfg.paths.out_dir = tmp_path / "out"
    cfg.paths.results_dir = tmp_path / "results"
    return cfg


METRICS = {"seasons": [2015, 2021], "overall": {
    "n_games": 10, "mae": 13.0, "line_mae": 12.0, "massey_mae": 15.0, "ats_n": 9, "ats_win_rate": 0.5,
    "ats_ci95_low": 0.2, "ats_ci95_high": 0.8}, "diagnostics": {
    "n_games": 10, "by_week": [{"bucket": "w01", "n": 10, "bias": 3.1, "slope": 1.1, "mae": 14.0}],
    "by_prior_share": [{"bucket": ">=0.9", "n": 10, "bias": 3.1, "slope": 1.1, "mae": 14.0}],
    "ats_by_edge": [{"bucket": "0-3", "n": 9, "wins": 4, "win_rate": 0.444, "ci95_low": 0.2, "ci95_high": 0.7}],
    "blend": {"n": 9, "w_line": 0.95, "w_model": 0.05, "mae_line": 12.0, "mae_model": 13.0, "mae_blend": 12.0},
    "reliability": [{"bin": "0.5-0.6", "n": 10, "predicted": 0.55, "observed": 0.5}]}}


@pytest.fixture
def client(web_cfg):
    _write_season(web_dir(web_cfg), datetime.now(UTC))
    run = web_cfg.paths.results_dir / "20260101T000000Z-test"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps(METRICS))
    return TestClient(create_app(web_cfg))


# -- pages ------------------------------------------------------------------------

def test_lab_shows_run_diagnostics(client):
    html = client.get("/lab").text
    assert "Diagnostics" in html and "0.950·line + 0.050·model" in html and "w01" in html


@pytest.mark.parametrize("url", [
    "/rankings", "/rankings?season=2026&week=0", "/rankings?season=2026&week=1&conf=SEC",
    "/team/2026/Texas%20A%26M", "/team/2026/Hawai%27i", "/games", "/games?season=2026&slot=r01", "/accuracy",
])
def test_pages_render(client, url):
    r = client.get(url)
    assert r.status_code == 200, r.text
    assert "<html" in r.text


def test_static_urls_carry_a_content_hash(client, tmp_path):
    import hashlib

    from cfbr.web.app import HERE, asset_versions

    # A changed stylesheet gets a new URL, so a browser can't pair new markup with a cached old one.
    html = client.get("/rankings").text
    for name in ("style.css", "app.js"):
        v = hashlib.sha256((HERE / "static" / name).read_bytes()).hexdigest()[:10]
        assert f'/static/{name}?v={v}"' in html, name
        assert client.get(f"/static/{name}?v={v}").status_code == 200
    (tmp_path / "a.css").write_text("a{}")
    before = asset_versions(tmp_path)["a.css"]
    (tmp_path / "a.css").write_text("a{color:red}")
    assert asset_versions(tmp_path)["a.css"] != before


def test_index_redirects_to_this_week(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/week"


def test_this_week_page(client):
    html = " ".join(client.get("/week").text.split())
    # Upcoming slate (week 2), movers (through week 1) and last week's results (week 1).
    assert "2026 · Week 2" in html and "2 games" in html
    assert "Texas A&amp;M</a> @ " in html and "Ohio State</a>" in html and "Toss-up" in html
    assert 'href="/games?season=2026&amp;slot=r02"' in html
    assert "through week 1" in html and 'href="/team/2026/Texas%20A%26M"' in html
    assert "No rank changes yet" in html  # the fixture's week 1 table has rank_change 0 everywhere
    assert "Week 1 results" in html and 'Games final</div><div class="tile-value">2<' in html
    assert "Vegas MAE" in html and "on same games" in html
    assert "No upsets" in html  # both week 1 favorites won
    assert "FCS:Big Sky" not in html
    assert client.get("/week?season=1999").status_code == 404


def test_this_week_surprises_and_flags():
    from cfbr.web.app import this_week

    now = datetime.now(UTC)
    games = _games(now).with_columns(
        # Game 2: Hawai'i favored by 4 but lost 17-24 -> an upset. Game 3: the model is 5 points
        # off the line -> flagged, listed first. Game 4 keeps its NaN spread: no edge, no flag.
        pred_margin=pl.when(pl.col("game_id") == 2).then(4.0).when(pl.col("game_id") == 3).then(8.0)
        .otherwise(pl.col("pred_margin")),
        win_prob=pl.when(pl.col("game_id") == 2).then(0.62).when(pl.col("game_id") == 3).then(0.73)
        .otherwise(pl.col("win_prob")))
    s = Season({"season": SEASON, "live": True, "weeks": [0, 1], "labels": {"0": "Preseason", "1": "Week 1"}},
               _ensure(_ratings(), ("ap_rank", "coaches_rank", "cfp_rank")),
               _ensure(games, [f"{a}_{c}" for a in ("home", "away") for c in ("ap_rank", "coaches_rank", "cfp_rank")]))
    w = this_week(s, now, 3.0)
    assert w["last_slot"] == "r01" and w["next_slot"] == "r02"
    assert [u["game_id"] for u in w["upsets"]] == [2] and w["upsets"][0]["fav_prob"] == 0.62
    assert w["n_upcoming"] == 2 and w["n_flagged"] == 1
    assert [g["game_id"] for g in w["upcoming"]] == [3, 4] and w["upcoming"][0]["flag"] == "edge"
    assert w["movers"]["top"][0]["team"] == "Texas A&M" and w["results"]["n_games"] == 2
    # A locked viewer gets the schedule only.
    w = this_week(s, now, 3.0, locked=True)
    assert w["movers"] is None and w["results"] is None and w["upsets"] == [] and w["n_upcoming"] == 2


def test_this_week_midweek_keeps_the_slate_and_last_full_week():
    """Week 2's Tuesday game is final while Saturday's is pending: the slate still shows the
    pending game and the results section stays on week 1, the last week that is over."""
    from cfbr.web.app import this_week

    now = datetime.now(UTC)
    games = _games(now).with_columns(
        completed=pl.when(pl.col("game_id") == 3).then(True).otherwise(pl.col("completed")),
        kickoff_ts=pl.when(pl.col("game_id") == 3).then(now - timedelta(days=1)).otherwise(pl.col("kickoff_ts")),
        home_points=pl.when(pl.col("game_id") == 3).then(20).otherwise(pl.col("home_points")),
        away_points=pl.when(pl.col("game_id") == 3).then(17).otherwise(pl.col("away_points")),
        margin=pl.when(pl.col("game_id") == 3).then(3.0).otherwise(pl.col("margin")))
    s = Season({"season": SEASON, "live": True, "weeks": [0, 1], "labels": {"0": "Preseason", "1": "Week 1"}},
               _ensure(_ratings(), ("ap_rank", "coaches_rank", "cfp_rank")),
               _ensure(games, [f"{a}_{c}" for a in ("home", "away") for c in ("ap_rank", "coaches_rank", "cfp_rank")]))
    w = this_week(s, now, 3.0)
    assert w["next_slot"] == "r02" and [g["game_id"] for g in w["upcoming"]] == [4]
    assert w["last_slot"] == "r01" and w["results"]["n_games"] == 2
    # Once Saturday's game is also final, week 2 is the week that is over.
    s2 = Season(s.meta, s.ratings, s.games.with_columns(completed=True, kickoff_ts=now - timedelta(days=1)))
    w = this_week(s2, now, 3.0)
    assert w["next_slot"] is None and w["last_slot"] == "r02"


def test_this_week_after_the_season(web_cfg):
    from cfbr.web.app import this_week

    now = datetime.now(UTC)
    s = Store(web_dir(web_cfg), web_cfg.paths.results_dir, web_cfg.paths.out_dir)
    _write_season(web_dir(web_cfg), now - timedelta(days=400), live=False)
    season = s.season(SEASON)
    w = this_week(season, now, 3.0)
    assert w["next_slot"] is None and w["last_slot"] == "r01" and w["upcoming"] == []


def test_rankings_default_to_latest_week_and_link_teams(client):
    html = client.get("/rankings").text
    assert "2026 rankings · Week 1" in html
    assert 'href="/team/2026/Texas%20A%26M"' in html
    assert 'href="/team/2026/Hawai%27i"' in html
    assert "FCS:Big Sky" not in html  # pooled FCS entities are not ranked


def test_top25_filter():
    from cfbr.web.app import rankings_rows
    from cfbr.web.store import Season

    n = 30
    ratings = pl.DataFrame({
        "week": [1] * n, "team": [f"Team {i:02d}" for i in range(1, n + 1)], "fbs": [True] * n,
        "rank": list(range(1, n + 1)), "conference": ["SEC" if i % 2 else "ACC" for i in range(1, n + 1)],
        "rating": [30.0 - i for i in range(1, n + 1)],
    }, schema_overrides={"rank": pl.Int32})
    s = Season({"weeks": [1], "labels": {"1": "Week 1"}}, ratings, pl.DataFrame())
    rows = rankings_rows(s, 1, "top25")
    assert [r["rank"] for r in rows] == list(range(1, 26))
    assert len(rankings_rows(s, 1, "SEC")) == 15 and len(rankings_rows(s, 1, None)) == n


def test_top25_option_on_rankings_page(client):
    html = " ".join(client.get("/rankings?season=2026&week=1&conf=top25").text.split())
    assert '<option value="top25" selected>Top 25</option>' in html
    assert "2026 rankings · Week 1 · Top 25" in html
    assert client.get("/api/rankings?season=2026&week=1&conf=top25").status_code == 200


def test_conference_filter(client):
    html = client.get("/rankings?season=2026&week=1&conf=SEC").text
    assert "Hawai&#39;i" in html and "Texas A&amp;M" not in html.split("<tbody>")[1]


def test_team_names_are_escaped(client):
    for url in ("/rankings", "/games?slot=r01", "/team/2026/%3Cb%3EEvil%3C%2Fb%3E", "/week"):
        html = client.get(url).text
        assert "<b>Evil</b>" not in html
        assert "&lt;b&gt;Evil&lt;/b&gt;" in html


def test_team_page_takes_team_perspective(client):
    html = client.get("/team/2026/Ohio%20State").text
    # Ohio State won 24-17 at Hawai'i; the home-side prediction of -4 means Ohio State by 4.
    assert "<span class=\"up\">W</span> 24–17" in " ".join(html.split())
    assert "+4.0" in html


def test_upcoming_slot_is_default_in_live_season(client):
    html = client.get("/games").text
    assert "2026 · Week 2" in html
    assert "Toss-up" in html  # |pred| < 0.05
    assert "Final</th>" not in html  # no result columns before any game is final


def test_game_picks_link_to_the_matchup_tool(client):
    # Week 1's Texas A&M (home) vs <b>Evil</b>: predicted on the preseason ratings (as_of T0),
    # so the link pins week 0 and the real home team; the name is URL-encoded, never raw HTML.
    html = client.get("/games?season=2026&slot=r01").text
    assert ('href="/matchup?season=2026&amp;week=0&amp;home=Texas%20A%26M&amp;away=%3Cb%3EEvil%3C/b%3E"'
            in html)
    # Week 2: no ratings week has that as_of yet, so the link opens on the latest ratings;
    # the FCS game gets no link (FCS teams aren't in the tool).
    html = client.get("/games?season=2026&slot=r02").text
    assert 'href="/matchup?season=2026&amp;home=Ohio%20State&amp;away=Texas%20A%26M"' in html
    assert "Montana" in html and "&amp;away=Montana" not in html     # the FCS game is listed, unlinked
    table = html[html.index('id="games"'):html.index("</table>")]
    assert table.count('href="/matchup?') == 1                        # Ohio State–Texas A&M only


def test_neutral_site_pick_links_with_neutral(web_cfg):
    d = _write_season(web_dir(web_cfg), datetime.now(UTC))
    g = pl.read_parquet(d / "games.parquet")
    g.with_columns(neutral_site=pl.col("game_id") == 3).write_parquet(d / "games.parquet")
    (d / "meta.json").write_text((d / "meta.json").read_text())  # meta last, as web build writes it
    html = TestClient(create_app(web_cfg)).get("/games?season=2026&slot=r02").text
    assert "&amp;home=Ohio%20State&amp;away=Texas%20A%26M&amp;neutral=1" in html


def test_missing_things_404(client):
    for url in ("/team/2026/Nobody", "/rankings?season=1999", "/rankings?week=7", "/games?slot=r09"):
        assert client.get(url).status_code == 404, url
    r = client.get("/api/team/2026/Nobody")
    assert r.status_code == 404 and r.json()["detail"].startswith("No team")


def test_no_data_is_503(web_cfg):
    r = TestClient(create_app(web_cfg)).get("/rankings")
    assert r.status_code == 503 and "cfbr web build" in r.text


def test_lab_lists_backtest_runs_and_track_record_does_not(client):
    html = client.get("/lab").text
    assert "20260101T000000Z-test" in html
    assert f"{MODEL_NAME} MAE" in html
    acc = client.get("/accuracy").text
    assert "20260101T000000Z-test" not in acc and 'href="/lab' not in acc  # linked from nowhere


# -- polls ------------------------------------------------------------------------

@pytest.fixture
def poll_client(web_cfg):
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    return TestClient(create_app(web_cfg))


def test_rankings_show_poll_columns_and_note(poll_client):
    html = " ".join(poll_client.get("/rankings?season=2026&week=1").text.split())
    assert ">AP</th>" in html and ">Coaches</th>" in html
    assert ">CFP</th>" not in html  # no CFP rankings released yet
    assert '<span class="pr m-only">#3</span><a href="/team/2026/Texas%20A%26M">' in html
    assert "AP: week 2 poll; Coaches: week 1 poll (week 2 not released yet)." in html
    assert 'data-v="999"' in html  # unranked teams sort last


def test_team_page_poll_tiles_and_opponent_rank(poll_client):
    html = " ".join(poll_client.get("/team/2026/Ohio%20State").text.split())
    assert "AP poll</div><div class=\"tile-value\">#11" in html
    assert "Coaches #12" in html  # Coaches rank rides on the AP tile
    assert '<span class="pr">#3</span><a href="/team/2026/Texas%20A%26M">' in html  # upcoming opponent
    html = " ".join(poll_client.get("/team/2026/Hawai%27i").text.split())
    assert "AP poll</div><div class=\"tile-value\">Unranked" in html
    assert "Coaches unranked" in html  # a released poll without the team still says so


def test_games_show_ap_rank_before_names(poll_client):
    html = " ".join(poll_client.get("/games?season=2026&slot=r01").text.split())
    assert '<span class="pr">#3</span><a href="/team/2026/Texas%20A%26M">' in html


def test_pages_render_without_poll_columns(client):
    # The default fixture predates polls; the store fills the columns with nulls.
    html = client.get("/rankings").text
    assert ">AP</th>" in html and "poll-note" not in html
    assert client.get("/api/rankings").json()["teams"][0]["ap_rank"] is None


def test_poll_note_wording():
    class S:  # minimal stand-in for Season
        meta = {"polls": {"0": {"ap_rank": 1, "coaches_rank": 1, "cfp_rank": None},
                          "6": {"ap_rank": 6, "coaches_rank": 7, "cfp_rank": None},
                          "99": {"ap_rank": "final", "coaches_rank": "final", "cfp_rank": 16}}}
    assert poll_note(S, 0) == ["AP: preseason poll", "Coaches: preseason poll"]
    assert poll_note(S, 6) == ["AP: week 6 poll (week 7 not released yet)", "Coaches: week 7 poll"]
    assert poll_note(S, FINAL) == ["AP: final poll", "Coaches: final poll", "CFP: week 16 (last released)"]
    assert poll_note(S, 3) == []


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


def test_web_polls_match_rank_output(engine, dataset, tmp_path):
    from cfbr.report.rank import add_polls, build_rankings

    season, week = 2024, 12
    build_season(engine, season, tmp_path, now=datetime.now(UTC))
    web = pl.read_parquet(tmp_path / str(season) / "ratings.parquet").filter(
        (pl.col("week") == week) & pl.col("fbs"))
    ref, _ = add_polls(build_rankings(engine, season, week)[0], dataset.polls, season, week)
    m = ref.join(web, on="team", suffix="_web")
    for c in ("ap_rank", "coaches_rank", "cfp_rank"):
        assert m.select(pl.col(c).cast(pl.Int32).eq_missing(pl.col(f"{c}_web"))).to_series().all(), c
    assert web["ap_rank"].drop_nulls().len() > 20 and web["cfp_rank"].drop_nulls().len() > 20


def test_final_table_uses_final_postseason_poll(dataset):
    season = 2023
    ranks, used = timeline_polls(dataset.polls, season, FINAL)
    assert used["ap_rank"] == "final" and used["coaches_rank"] == "final"
    post = dataset.polls.filter((pl.col("season") == season) & (pl.col("season_type") == "postseason")
                                & (pl.col("poll") == "AP Top 25"))
    no1 = post.filter(pl.col("rank") == 1)["team"].item()
    assert ranks.filter(pl.col("ap_rank") == 1)["team"].item() == no1


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


# -- paper bets ---------------------------------------------------------------------

def _bet(gid, side, line, status="open", hp=None, ap=None, closing=None, week=7, provider="TestBook",
         best_odds=None):
    from cfbr.bets.ledger import clv, risk

    risked, won = risk(-110, 1.0)
    units = {"won": won, "lost": -risked, "push": 0.0}.get(status)
    home, away = ("Texas A&M", "Ohio State") if gid % 2 else ("Hawai'i", "<b>Evil</b>")
    return {
        "game_id": gid, "season": SEASON, "week": week, "placed_at": "2026-10-11T16:24:08+00:00",
        "kickoff_ts": f"2026-10-{10 + gid:02d}T23:00:00Z", "home_team": home, "away_team": away, "side": side,
        "team": home if side == "home" else away, "model_margin": 10.0 if side == "home" else -10.0, "line": line,
        "edge": (10.0 + line) if side == "home" else (-10.0 + line), "n_providers": 3, "best_line": line + 0.5,
        "best_provider": provider, "odds": -110, "stake": 1.0, "status": status, "home_points": hp,
        "away_points": ap, "closing_line": closing, "clv": None if closing is None else clv(side, line, closing),
        "units": units, "best_odds": best_odds,
    }


def _write_ledger(cfg):
    from cfbr.bets.ledger import LEDGER_SCHEMA

    rows = [_bet(1, "home", -3.5, best_odds=-105), _bet(2, "away", 4.0),
            _bet(3, "home", -7.0, "won", 31, 14, -8.0, week=6), _bet(4, "away", 3.0, "lost", 28, 10, 2.0, week=6),
            _bet(5, "home", -3.0, "push", 24, 21, -3.0, week=6)]
    d = cfg.paths.out_dir / str(SEASON)
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=LEDGER_SCHEMA).write_csv(d / "bets.csv")
    return d / "bets.csv"


def _write_retro(cfg):
    """Retro bets for both models; "RetroBook" and "Retro U" only exist in this file."""
    from cfbr.bets.retro import RETRO_SCHEMA

    rows = []
    for model in ("v1", "roster"):
        for i, (st, hp, ap, wk) in enumerate([("won", 31, 14, 1), ("lost", 10, 28, 1), ("won", 35, 7, 2)]):
            r = _bet(11 + i, "home", -7.0, st, hp, ap, -8.0, week=wk, provider="RetroBook")
            r |= {"home_team": "Retro U", "team": "Retro U", "model": model, "season_type": "regular"}
            rows.append(r)
    d = cfg.paths.out_dir / str(SEASON)
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=RETRO_SCHEMA).write_csv(d / "bets_retro.csv")


@pytest.fixture
def bets_client(web_cfg):
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_ledger(web_cfg)
    return TestClient(create_app(web_cfg))


def test_break_even():
    from cfbr.bets.ledger import break_even

    assert round(break_even(-110), 4) == 0.5238
    assert round(break_even(-105), 4) == 0.5122
    assert break_even(100) == 0.5
    assert round(break_even(105), 4) == 0.4878


def test_bets_guide(bets_client, web_cfg):
    r = bets_client.get("/bets/guide")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    html = " ".join(r.text.split())
    assert '<a href="/bets/guide" aria-current="page">Guide</a>' in html
    # Break-even table from the odds math, not hand-typed.
    for price, be in (("−120", "54.5%"), ("−110", "52.4%"), ("−105", "51.2%"), ("+105", "48.8%")):
        assert f'<td class="num">{price}</td>' in html and f'<td class="num">{be}</td>' in html, price
    # Edge tiers and holdout cover rates come from config and the backtest constants.
    assert "Small edge (3–6 pts)" in html and "50.4%" in html and "49.0%" in html and "48.4%" in html
    assert "at least <strong>3 points</strong>" in html and "Under 3, pass." in html
    assert "before 500 settled bets" in html and "$11 to win $10" in html
    assert 'id="rules"' in html and html.count('<li><strong>') >= 10
    # Definitions and made-up examples only: nothing from the ledger.
    assert "TestBook" not in html and "Texas A&amp;M" not in html

    assert "1 unit ($11 to win $10), where a unit is $10, 1.0% of a $1,000 bankroll" in html
    assert 'Lines fetched only to price a bet (bet365\'s) are marked "pricing only"' in html

    web_cfg.bets.threshold = 4.0
    web_cfg.bets.stake, web_cfg.bets.unit_value, web_cfg.bets.bankroll = 2.0, 25.0, 1000.0
    html = " ".join(TestClient(create_app(web_cfg)).get("/bets/guide").text.split())
    assert "at least <strong>4 points</strong>" in html and "Under 4, pass." in html
    assert "2 units ($55 to win $50), where a unit is $25, 2.5% of a $1,000 bankroll" in html


def test_bets_pages_link_the_guide(bets_client):
    for url in ("/bets", "/bets/shortlist"):
        html = bets_client.get(url).text
        assert '<a href="/bets/guide">Guide</a>' in html and 'Start with the <a href="/bets/guide">guide</a>' in html, url


def test_bets_page(bets_client):
    r = bets_client.get("/bets")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "private, no-store"
    html = " ".join(r.text.split())
    assert "2026 paper bets" in html and "not betting advice" in html
    assert "No conclusion yet." in html and "3 of the 500 settled bets" in html
    assert "1–1–1" in html  # won-lost-push
    assert "All bets, best line" in html
    # bet365 has a pricing-only feed now; the page explains the "(pricing only)" marker.
    assert "has no feed" not in html and '"(pricing only)" marks a line used only to price the bet' in html
    # On phones the Result column is hidden; the amount carries the result as a mark, not by color alone.
    assert '<span class="m-only">✓ </span>' in html and '<span class="m-only">✗ </span>' in html
    # The best-line book's own price shows when the feed has one, and only then.
    assert "−3 (−105) <span class=\"muted\">TestBook</span>" in html
    assert "−4.5 <span class=\"muted\">TestBook</span>" in html
    # Rules describe the current pricing rule: offshore books never set the best line.
    assert "Bovada" in html and "never sets the best line" in html
    assert "Only DraftKings, FanDuel, bet365 can set the best line" in html
    assert "bet365's lines come from a separate feed used only to price bets" in html
    # The selection rule and the close's CFBD fallback are both stated, not simplified away.
    assert "the number-and-price pair worth the most against the consensus" in html
    assert "at most 12 hours earlier), or CFBD's post-game consensus when none qualifies" in html
    # What-if panel: hypothetical framing, the live plan first and marked, all three plans.
    assert "What if: staking plans" in html and "not a separate record" in html
    assert "Flat 1% (live)" in html.replace('<span class="muted">', "").replace("</span>", "")
    assert "Flat 2%" in html and "All-in" in html
    # Lines are shown from the bet team's side: an away bet at home +4 is "−4".
    assert "Texas A&amp;M −3.5" in html and "&lt;b&gt;Evil&lt;/b&gt; −4" in html
    assert "<b>Evil</b>" not in html
    # Week 6 went 1-1-1 at $10 units: +$10 - $11 = -$1, leaving $999 of the $1,000 bankroll;
    # the open bets put $22 at risk.
    assert "Week by week" in html and "Week 6" in html and "−$1" in html and "$22 at risk" in html
    assert "$999" in html and "$1,000 bankroll" in html
    assert "By edge tier" in html and "Small edge (3–6 pts)" in html and "$11 / $10" in html
    # Tracked tests: rule, pre-set minimum and "no conclusion yet" framing are shown.
    assert "Tracked tests" in html and "T1 · Core filter" in html and "T2 · Second-look bets" in html
    assert "No conclusion yet." in html and "of the 50 per group set as the minimum" in html and "Decision rule:" in html


def test_bets_api_matches_ledger_summary(bets_client, web_cfg):
    from cfbr.bets.ledger import read_ledger, summarize

    body = bets_client.get("/api/bets").json()
    m = summarize(read_ledger(web_cfg.paths.out_dir / str(SEASON) / "bets.csv"), web_cfg.bets)
    assert body["summary"]["won"] == m["won"] == 1 and body["summary"]["open"] == 2
    assert body["summary"]["units"] == pytest.approx(m["units"])
    assert body["rules"]["threshold"] == web_cfg.bets.threshold
    assert body["summary_best_line"]["settled"] == 0  # synthetic ledger predates best-line grading
    assert len(body["bets"]) == 5
    assert body["summary"]["dollars"] == pytest.approx(m["units"] * 10)
    assert [w["label"] for w in body["weekly"]] == ["Week 6"] and body["weekly"][0]["cum_dollars"] == pytest.approx(-1)
    assert body["weekly"][0]["balance"] == pytest.approx(999) and body["summary"]["balance"] == pytest.approx(999)
    assert [t["name"] for t in body["tiers"]] == ["Small", "Medium", "Large"]
    assert [t["key"] for t in body["tests"]] == ["core_filter", "second_look", "shortlist"]
    assert [p["key"] for p in body["what_if"]] == ["flat_1", "flat_2", "all_in"]
    assert body["what_if"][0]["balance"] == pytest.approx(body["summary"]["balance"])
    assert sum(g["settled"] for g in body["tests"][0]["groups"]) == m["settled"]
    assert {b["tier"] for b in body["bets"]} <= {"Small", "Medium", "Large"}


def test_retro_bets_page_and_api(web_cfg):
    from cfbr.bets.retro import read_retro
    from cfbr.bets.ledger import summarize

    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_ledger(web_cfg)
    _write_retro(web_cfg)
    c = TestClient(create_app(web_cfg))
    assert "/bets/retro?season=2026" in c.get("/bets").text  # linked from the live page
    r = c.get("/bets/retro")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    html = " ".join(r.text.split())
    assert "Retroactive, not live." in html and "<strong>opening</strong> line" in html and "point-in-time" in html
    assert f"{MODEL_NAME} (published)" in html and "Experimental roster model" in html
    assert "v1 prior" not in html and "shadow" not in html  # internal names stay off the page
    # Each model: week 1 went 1-1 (+$10 - $11), week 2 won: bankroll $1,000 -> $999 -> $1,009.
    assert "Week 1" in html and "−$1" in html and "$1,009" in html and "Retro U −7" in html
    assert "3 in common, 3 on the same side" in html
    assert html.count("What if: staking plans") == 2  # one panel per model
    assert html.count("not a separate record") == 1  # explained once, above the first
    body = c.get("/api/bets/retro").json()
    assert body["retroactive"] is True and [m["model"] for m in body["models"]] == ["v1", "roster"]
    led = read_retro(web_cfg.paths.out_dir / str(SEASON) / "bets_retro.csv")
    m = summarize(led.filter(pl.col("model") == "v1"), web_cfg.bets)
    assert body["models"][0]["summary"]["won"] == m["won"] == 2 and len(body["bets"]) == 6
    # The live page and API never include retro bets.
    assert "RetroBook" not in c.get("/api/bets").text and "Retro U" not in c.get("/bets").text
    assert c.get("/bets/retro?season=2019").status_code == 404


def test_retro_empty_state(client):
    r = client.get("/bets/retro")
    assert r.status_code == 200 and "No retroactive paper bets yet" in r.text
    assert client.get("/api/bets/retro").status_code == 404


def test_bets_empty_state_and_unknown_season(client):
    r = client.get("/bets")
    assert r.status_code == 200 and "No paper bets recorded yet" in r.text
    assert client.get("/api/bets").status_code == 404
    assert client.get("/bets?season=2019").status_code == 404


def test_bet_rows_team_perspective(web_cfg):
    from cfbr.bets.ledger import LEDGER_SCHEMA
    from cfbr.web.bets import bet_rows

    rows = bet_rows(pl.DataFrame([_bet(2, "away", 4.0), _bet(4, "away", 3.0, "lost", 28, 10, 2.0)],
                                 schema=LEDGER_SCHEMA), web_cfg)
    a = {r["game_id"]: r for r in rows}
    assert a[2]["team_line"] == -4.0 and a[2]["model_team"] == 10.0 and a[2]["best"] == -4.5
    # Away team laid 3 and closed at -2: the bet got a worse number than the close.
    assert a[4]["close"] == -2.0 and a[4]["clv"] == -1.0
    # $100 units: an open bet risks $110 to win $100; a lost bet is down $110.
    assert (a[2]["risk_dollars"], a[2]["win_dollars"], a[2]["dollars"]) == pytest.approx((11.0, 10.0, None))
    assert a[4]["dollars"] == pytest.approx(-11.0) and a[4]["tier"] == "Medium"  # |edge| = 7


def test_bets_gate_blocks_only_the_betting_section(web_cfg):
    """The single access dependency gates every betting route and nothing else,
    and no other page or API response carries betting data."""
    from fastapi import HTTPException

    from cfbr.web.access import bets_access

    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_ledger(web_cfg)
    _write_retro(web_cfg)
    app = create_app(web_cfg)

    def deny():
        raise HTTPException(402, "Subscription required.")

    app.dependency_overrides[bets_access] = deny
    c = TestClient(app)
    assert c.get("/bets").status_code == 402
    r = c.get("/api/bets")
    assert r.status_code == 402 and r.json()["detail"] == "Subscription required."
    assert c.get("/bets/retro").status_code == 402 and c.get("/api/bets/retro").status_code == 402
    assert c.get("/bets/shortlist").status_code == 402 and c.get("/api/bets/shortlist").status_code == 402
    assert c.get("/bets/guide").status_code == 402
    assert c.get("/lab").status_code == 402  # edge-vs-close research sits with the bets
    for url in ("/rankings", "/games?slot=r01", "/games?slot=r02", "/accuracy", "/method", "/team/2026/Ohio%20State",
                "/api/rankings", "/api/games?slot=r02", "/api/team/2026/Ohio%20State", "/api/seasons"):
        r = c.get(url)
        assert r.status_code == 200, url
        assert "TestBook" not in r.text and "paper bet" not in r.text.lower(), url
        assert "RetroBook" not in r.text and "Retro U" not in r.text and "retroactive" not in r.text.lower(), url


# -- free viewers: no live-season model output anywhere -------------------------------

# Strings that only exist if model output for the synthetic live season leaked:
# ratings (+21.0, +16.0 ...), offense/defense, predicted margins and picks.
MODEL_STRINGS = ["+21.0", "+16.0", "+11.0", "+6.0", "+10.0", "+9.0", "by 10.0", "by 4.0", "by 30.0", "Toss-up",
                 f"{MODEL_NAME} err", f">{MODEL_NAME}</th>", ">Edge</th>", "Win %"]


def _write_finished(root, now):
    """A finished 2025 season (copy of the synthetic one) for free viewers to land on."""
    d = _write_season(root, now, live=False, polls=True)
    dst = root / "2025"
    dst.mkdir(parents=True, exist_ok=True)
    pl.read_parquet(d / "ratings.parquet").with_columns(season=pl.lit(2025, pl.Int32)).write_parquet(
        dst / "ratings.parquet")
    pl.read_parquet(d / "games.parquet").with_columns(season=pl.lit(2025)).write_parquet(dst / "games.parquet")
    meta = json.loads((d / "meta.json").read_text()) | {"season": 2025, "live": False}
    (dst / "meta.json").write_text(json.dumps(meta))


@pytest.fixture
def free_client(web_cfg):
    from cfbr.web.access import premium_viewer

    now = datetime.now(UTC)
    _write_finished(web_dir(web_cfg), now)
    _write_season(web_dir(web_cfg), now, live=True, polls=True)  # 2026, in progress
    _write_ledger(web_cfg)
    app = create_app(web_cfg)
    app.dependency_overrides[premium_viewer] = lambda: False
    return TestClient(app)


@pytest.mark.parametrize("url", [
    "/rankings?season=2026", "/rankings?season=2026&week=1", "/team/2026/Ohio%20State", "/team/2026/Texas%20A%26M",
    "/games?season=2026&slot=r01", "/games?season=2026&slot=r02", "/week?season=2026", "/week",
])
def test_free_viewers_see_no_live_model_output(free_client, url):
    r = free_client.get(url)
    assert r.status_code == 200, url
    html = " ".join(r.text.split())
    for leak in MODEL_STRINGS:
        assert leak not in html, (url, leak)
    assert "for subscribers while the season is in progress" in html
    assert r.headers["cache-control"] == "private, no-store"


def test_free_viewers_keep_public_live_data(free_client):
    html = " ".join(free_client.get("/games?season=2026&slot=r01").text.split())
    assert "31–10" not in html and "10–31" in html  # final score away–home
    assert "Texas A&amp;M −7" in html  # closing line
    assert '<span class="pr">#3</span>' in html  # AP rank
    html = " ".join(free_client.get("/team/2026/Ohio%20State").text.split())
    assert "0–1" in html and "AP poll" in html  # record and polls


def test_free_viewers_land_on_finished_season(free_client):
    html = " ".join(free_client.get("/rankings").text.split())
    assert "2025 rankings · Week 1" in html and "+21.0" in html  # finished season is fully free
    assert "2026 ratings and rankings are for subscribers" in html


def test_free_api_withholds_live_model_output(free_client):
    r = free_client.get("/api/rankings?season=2026")
    assert r.status_code == 402
    team = free_client.get("/api/team/2026/Ohio%20State").json()
    assert all(w[c] is None for w in team["weeks"] for c in ("rank", "rating", "offense", "defense", "o_ppa"))
    assert all(g["t_pred"] is None and g["t_win_prob"] is None for g in team["games"])
    for slot in ("r01", "r02"):
        games = free_client.get(f"/api/games?season=2026&slot={slot}").json()["games"]
        assert all(g[c] is None for g in games for c in ("pred_margin", "win_prob", "prior_pred", "massey_pred"))
        assert any(g["spread"] is not None for g in games)  # lines are public
    assert free_client.get("/api/rankings?season=2025").json()["teams"][0]["rating"] == 21.0
    assert free_client.get("/bets").status_code == 402 and free_client.get("/api/bets").status_code == 402


def test_accuracy_stays_free_and_aggregate(free_client):
    html = " ".join(free_client.get("/accuracy?season=2026").text.split())
    assert f"{MODEL_NAME} MAE" in html
    for leak in ("by 10.0", "by 4.0", "Toss-up", "+21.0"):
        assert leak not in html


def test_premium_viewers_see_live_model_output(web_cfg):
    now = datetime.now(UTC)
    _write_season(web_dir(web_cfg), now, live=True, polls=True)
    c = TestClient(create_app(web_cfg))  # default: everyone is premium
    html = " ".join(c.get("/games?season=2026&slot=r01").text.split())
    assert "by 10.0" in html and "for subscribers" not in html
    assert c.get("/games?season=2026&slot=r01").headers["cache-control"] == "private, no-store"



def test_bankroll_chart_is_formatted_as_dollars():
    from cfbr.web.bets import money_chart

    spec = money_chart([{"label": "Week 5", "balance": 986.0}, {"label": "Week 6", "balance": 951.0}], 1000.0)
    assert spec["format"] == "dollars" and spec["signed"] is False
    assert spec["x"] == ["Start", "Week 5", "Week 6"] and spec["series"][0]["values"] == [1000.0, 986.0, 951.0]


def test_templates_do_not_hot_reload(web_cfg):
    """Templates change only on restart, together with the code that renders them."""
    assert create_app(web_cfg).state.templates.env.auto_reload is False


# -- rosters: public facts, free even for a live season ----------------------------------

def _write_careers(season_dir):
    """Career rows for player 1 (two completed seasons at Hawai'i, then this season to date)."""
    base = {"season": SEASON, "team": "Texas A&M", "player_id": "1", "games": 12, "qb_starts": 12, "plays": 500,
            "pass_att": 450, "rush_car": 40, "rec": 0, "ppa_total": 120.0, "ppa_avg": 0.24, "pass_ppa_total": 110.0,
            "pass_ppa_avg": 0.244, "tackles": None, "tfl": None, "sacks": None, "pd": None, "hurries": None,
            "ints": None}
    rows = [base | {"career_season": 2024, "career_team": "Hawai'i", "source": "season stats"},
            base | {"career_season": 2025, "career_team": "Hawai'i", "source": "season stats", "pass_att": 734},
            base | {"career_season": 2026, "career_team": "Texas A&M", "source": "box scores (to date)", "games": 1,
                    "qb_starts": 1, "pass_att": 30, "ppa_total": None, "ppa_avg": None, "pass_ppa_total": None,
                    "pass_ppa_avg": None}]
    pl.DataFrame(rows, schema_overrides={"season": pl.Int32, "career_season": pl.Int32}).write_parquet(
        season_dir / "roster_careers.parquet")


def _write_roster(season_dir):
    """Synthetic roster files next to a season's ratings, then touch meta.json (the reload signal)."""
    from cfbr.report.roster_web import ROSTER_SUMMARY_LABELS

    players = pl.DataFrame({
        "season": [SEASON] * 4, "team": ["Texas A&M", "Texas A&M", "Texas A&M", "Ohio State"],
        "player_id": ["1", "2", "3", "4"], "name": ["Sam Starter", "Tom Transfer", "<i>Fresh</i> Man", "Oscar Buckeye"],
        "jersey": [7, 12, None, 1], "position": ["QB", "QB", "OL", "WR"], "class_year": [4, 5, 1, 2],
        "height": [74, 75, 78, None], "weight": [215, 220, 310, 190], "home_city": ["Austin", "Tulsa", None, "Columbus"],
        "home_state": ["TX", "OK", None, "OH"], "status": ["returner", "transfer_in", "new", "returner"],
        "from_team": [None, "Hawai'i", None, None], "prev_plays": [412, 288, None, 61],
        "prev_ppa_total": [130.5, 70.1, None, 40.0], "prev_ppa_avg": [0.317, 0.243, None, 0.656],
        "prev_pass_plays": [380, 260, None, 0], "prev_pass_ppa_total": [120.0, 60.0, None, 0.0],
        "prev_pass_ppa_avg": [0.316, 0.231, None, None], "prev_tackles": [None, None, None, 2.0],
        "prev_havoc": [None, None, None, 0.5],
        "career_seasons": [3, 2, 0, 1], "career_pass_att": [1184, 260, 0, 0], "career_qb_starts": [31, 4, 0, 0],
    }, schema_overrides={"season": pl.Int32, "jersey": pl.Int32, "class_year": pl.Int32, "height": pl.Int32,
                         "weight": pl.Int32, "prev_plays": pl.Int64, "prev_pass_plays": pl.Int64})
    departures = pl.DataFrame({
        "season": [SEASON] * 2, "team": ["Texas A&M"] * 2, "player_id": ["8", "9"], "name": ["Gone Pro", "Left Town"],
        "position": ["RB", "LB"], "to_team": [None, "Ohio State"], "prev_plays": [190, None],
        "prev_ppa_total": [22.0, None], "prev_tackles": [None, 61.0], "prev_havoc": [None, 7.5],
    }, schema_overrides={"season": pl.Int32, "prev_plays": pl.Int64})
    summary = pl.DataFrame([{"season": SEASON, "team": "Texas A&M", "roster_as_of": T0,
                             **({k: 0.0 for k in ROSTER_SUMMARY_LABELS} | {"off_ret_share": 0.62, "qb_returns": 1.0}),
                             "in_qb_name": "Tom Transfer", "in_qb_from": "Hawai'i", "in_qb_pass_att": 75,
                             "in_qb_pass_ppa_avg": 0.123, "in_qb_pass_ppa_total": 9.2}],
                           schema_overrides={"season": pl.Int32})
    players.write_parquet(season_dir / "roster_players.parquet")
    departures.write_parquet(season_dir / "roster_departures.parquet")
    summary.write_parquet(season_dir / "roster_summary.parquet")
    meta = season_dir / "meta.json"
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))


@pytest.fixture
def roster_client(web_cfg):
    _write_roster(_write_season(web_dir(web_cfg), datetime.now(UTC), live=True, polls=True))
    return TestClient(create_app(web_cfg))


def test_roster_page(roster_client):
    r = roster_client.get("/roster/2026/Texas%20A%26M")
    assert r.status_code == 200
    html = " ".join(r.text.split())
    assert 'Texas A&amp;M <span class="muted">2026</span>' in html and 'aria-current="page">Roster</a>' in html
    assert "Sam Starter" in html and "Returner" in html and "No prior stats" in html
    assert 'Transfer from <a href="/team/2025/Hawai%27i">Hawai&#39;i</a>' in html
    assert "&lt;i&gt;Fresh&lt;/i&gt; Man" in html and "<i>Fresh</i>" not in html  # names are escaped
    assert "6-2" in html and ">Sr<" in html and ">Gr<" in html  # height and class labels
    assert "Oscar Buckeye" not in html  # another team's player
    # Departures, with the destination linked when it's an FBS team this season.
    assert "Gone Pro" in html and "not on a roster" in html and '<a href="/team/2026/Ohio%20State">Ohio State</a>' in html
    # Summary: shares as percentages, QB back as Yes, no transfer QB as None.
    assert '<a href="/player/2026/1">Sam Starter</a>' in html  # names link to player pages
    assert "3 seasons · 31 starts" in html
    assert "62%" in html and "Starting QB back" in html and ">Yes<" in html
    # qb_in_ppa = 0 means no 100+-attempt transfer QB, which is not the same as no transfer QB.
    assert ">None with 100+ att.<" in html and ">None<" not in html
    # ...and the incoming QB below that cutoff is still shown.
    assert "Incoming transfer QB" in html and "from Hawai&#39;i · 75 pass att. · +0.12 PPA/att." in html


def test_roster_page_departure_reasons(web_cfg):
    season_dir = _write_season(web_dir(web_cfg), datetime.now(UTC), live=True, polls=True)
    _write_roster(season_dir)
    pl.DataFrame({
        "season": [SEASON] * 5, "team": ["Texas A&M"] * 5, "player_id": ["8", "9", "10", "11", "12"],
        "name": ["Gone Pro", "Left Town", "Big Tackle", "Portal Guy", "Old Man"],
        "position": ["RB", "LB", "OL", "WR", "S"], "prev_class_year": [3, 2, 4, 2, 4],
        "to_team": [None, "Ohio State", None, None, None],
        "reason": ["drafted", "transfer", "drafted", "portal_transfer", "unknown"],
        "draft_round": [1, None, 4, None, None], "draft_overall": [12, None, 120, None, None],
        "draft_nfl_team": ["Chicago", None, "Dallas", None, None],
        # Full name known for one pick only: the other falls back to the city.
        "draft_nfl_team_name": ["Chicago Bears", None, None, None, None],
        "in_portal": [False, True, False, True, False], "portal_destination": [None, "Ohio State", None, "Montana", None],
        "prev_plays": [190, None, None, 20, None], "prev_ppa_total": [22.0, None, None, 1.0, None],
        "prev_tackles": [None, 61.0, None, None, 30.0], "prev_havoc": [None, 7.5, None, None, 1.0],
    }, schema_overrides={"season": pl.Int32, "prev_class_year": pl.Int32, "draft_round": pl.Int32,
                         "draft_overall": pl.Int32, "prev_plays": pl.Int64}).write_parquet(
        season_dir / "roster_departures.parquet")
    client = TestClient(create_app(web_cfg))
    html = " ".join(client.get("/roster/2026/Texas%20A%26M").text.split())
    assert "Drafted</span> <span class=\"muted\">Rd 1 · #12 overall · Chicago Bears" in html
    assert "Big Tackle" in html and "Rd 4 · #120 overall · Dallas" in html
    assert "Montana <span class=\"muted\">(portal)</span>" in html
    assert "not on a roster · senior last season" in html
    assert '<option value="portal_transfer">' in html
    assert "2 drafted · 1 now on another roster" in html


def test_team_page_shows_roster_line(roster_client):
    html = " ".join(roster_client.get("/team/2026/Texas%20A%26M").text.split())
    assert '<a href="/roster/2026/Texas%20A%26M">Full roster' in html
    assert "(1 returners, 1 transfers in)" in html
    # The starting QB is named with where he is, not a bare Yes/No.
    assert "last season's starting QB Sam Starter (returns · 380 pass att. last season)" in html
    assert "Starting QB back" not in html
    # No roster rows for this team -> no roster line.
    assert "/roster/2026/Hawai%27i" not in roster_client.get("/team/2026/Hawai%27i").text


def test_roster_api(roster_client):
    body = roster_client.get("/api/roster/2026/Texas%20A%26M").json()
    assert [p["name"] for p in body["players"]] == ["Sam Starter", "Tom Transfer", "<i>Fresh</i> Man"]
    assert body["summary"]["off_ret_share"] == 0.62
    assert body["summary_labels"]["qb_returns"]["label"] == "Starting QB back"
    assert len(body["departures"]) == 2


def test_roster_404s(roster_client, client):
    assert roster_client.get("/roster/2026/Nobody").status_code == 404
    assert roster_client.get("/api/roster/2026/Nobody").status_code == 404


def test_roster_404_without_roster_files(client):
    assert client.get("/roster/2026/Texas%20A%26M").status_code == 404  # season built without roster files


def test_rosters_are_free_during_a_live_season(web_cfg):
    from cfbr.web.access import premium_viewer

    _write_roster(_write_season(web_dir(web_cfg), datetime.now(UTC), live=True, polls=True))
    app = create_app(web_cfg)
    app.dependency_overrides[premium_viewer] = lambda: False
    c = TestClient(app)
    assert "Sam Starter" in c.get("/roster/2026/Texas%20A%26M").text
    team = " ".join(c.get("/team/2026/Texas%20A%26M").text.split())
    assert "for subscribers" in team and "Full roster" in team  # ratings locked, roster card shown


def test_build_survives_roster_failure(engine, tmp_path, monkeypatch):
    import cfbr.web.build as build

    def boom(*a, **k):
        raise RuntimeError("roster data unavailable")

    monkeypatch.setattr(build, "write_roster_web", boom)
    out = build_season(engine, 2023, tmp_path, now=datetime.now(UTC))
    meta = json.loads((out / "meta.json").read_text())
    assert meta["roster"] is False and (out / "ratings.parquet").exists()


# -- season-simulation odds: premium while live, free once the season is over --------------

def _write_sim(cfg, season=SEASON, week=1):
    d = cfg.paths.out_dir / str(season) / f"w{week:02d}"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({
        "team": ["Texas A&M", "Ohio State", "Hawai'i"], "conference": ["SEC", "Big Ten", "Mountain West"],
        "rating": [21.0, 6.0, 16.0], "wins_now": [1, 0, 1], "losses_now": [0, 1, 0], "exp_wins": [10.4, 7.1, 8.8],
        "wins_p10": [9, 5, 7], "wins_p90": [12, 9, 10], "p_undefeated": [0.31, 0.0, 0.004],
        "p_bowl_eligible": [0.9991, 0.62, 0.9], "p_title_game": [0.6, 0.1, 0.5], "p_conf_title": [0.4321, 0.05, 0.3],
        "p_playoff": [0.8765, 0.02, 0.0], "p_bye": [0.4, 0.0, 0.0], "p_champion": [0.1234, 0.0, 0.0],
    }).write_csv(d / "season_sim.csv")
    (d / "season_sim.md").write_text(
        "# sim\n\n100,000 simulated seasons. Probabilities, not predictions.\n\n"
        "**Approximations:** pairings use win % <then> strength.\n\n| team | x |\n|---|---|\n| A | 1 |\n")


@pytest.fixture
def odds_client(web_cfg):
    _write_season(web_dir(web_cfg), datetime.now(UTC), live=True, polls=True)
    _write_sim(web_cfg)
    return TestClient(create_app(web_cfg))


def test_team_page_shows_its_projection(odds_client):
    html = " ".join(odds_client.get("/team/2026/Texas%20A%26M").text.split())
    assert "<h2>Projections <span" in html and "Expected wins</div><div class=\"tile-value\">10.4" in html
    # The link filters /odds to the team's conference in the ratings (Big Ten in this fixture).
    assert "/odds?season=2026&amp;week=1&amp;conf=Big%20Ten" in html


def test_prob_formatting():
    from cfbr.web.odds import prob

    assert [prob(x) for x in (0.0, 0.004, 0.5, 0.9991, 1.0, None)] == ["0%", "<1%", "50%", ">99%", "100%", "—"]


def test_odds_page_for_premium_viewers(odds_client):
    r = odds_client.get("/odds")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    html = " ".join(r.text.split())
    assert "2026 projections <span class=\"muted\">through week 1</span>" in html
    assert "Probabilities, not predictions." in html
    # Notes come from the .md (escaped, with **bold**), and stop before its table.
    assert "<strong>Approximations:</strong> pairings use win % &lt;then&gt; strength." in html and "| team |" not in html
    assert ">88%<" in html and ">12%<" in html and ">&lt;1%<" in html and ">&gt;99%<" in html
    assert "10.4 <span class=\"muted\">[9–12]</span>" in html
    # Conference title cards.
    assert "Mountain West" in html and ">43%<" in html
    sec = " ".join(odds_client.get("/odds?conf=SEC").text.split())
    assert 'id="odds"' in sec and sec.count('<td class="team"><a href="/team/2026/') >= 1 and "Big Ten</td>" not in sec


def test_odds_are_premium_while_live(web_cfg):
    from cfbr.web.access import premium_viewer

    _write_season(web_dir(web_cfg), datetime.now(UTC), live=True, polls=True)
    _write_sim(web_cfg)
    app = create_app(web_cfg)
    app.dependency_overrides[premium_viewer] = lambda: False
    c = TestClient(app)
    html = c.get("/odds").text
    assert "for subscribers while the season is in progress" in html and "88%" not in html and "Texas A&amp;M" not in html
    assert c.get("/api/odds").status_code == 402


def test_finished_season_odds_are_free(web_cfg):
    from cfbr.web.access import premium_viewer

    _write_season(web_dir(web_cfg), datetime.now(UTC), live=False, polls=True)
    _write_sim(web_cfg)
    app = create_app(web_cfg)
    app.dependency_overrides[premium_viewer] = lambda: False
    c = TestClient(app)
    assert ">88%<" in c.get("/odds").text
    body = c.get("/api/odds").json()
    assert body["week"] == 1 and len(body["teams"]) == 3 and "Probabilities, not predictions." in body["notes"]


def test_odds_404_for_missing_week(odds_client):
    assert odds_client.get("/odds?week=5").status_code == 404


def test_odds_404_without_simulations(client):
    assert client.get("/odds").status_code == 404  # no simulations written at all


def test_player_page_with_career(web_cfg):
    d = _write_season(web_dir(web_cfg), datetime.now(UTC), live=True, polls=True)
    _write_careers(d)
    _write_roster(d)
    c = TestClient(create_app(web_cfg))
    r = c.get("/player/2026/1")
    assert r.status_code == 200
    html = " ".join(r.text.split())
    assert "Sam Starter" in html and 'href="/roster/2026/Texas%20A%26M">← Texas A&amp;M 2026 roster</a>' in html
    assert ">1,184<" in html and ">31<" in html  # career pass attempts and starts (tiles, with separators)
    assert "2026 <span class=\"muted\">to date</span>" in html and "734" in html
    # Only column groups with data: passing yes, defense no.
    assert ">Pass att<" in html and ">Tkl<" not in html
    body = c.get("/api/player/2026/1").json()
    assert [row["career_season"] for row in body["career"]] == [2024, 2025, 2026]
    assert body["career_labels"]["qb_starts"]["label"] == "QB starts"


def test_player_page_without_career_file(roster_client):
    r = roster_client.get("/player/2026/3")
    assert r.status_code == 200 and "Career history isn't available" in r.text
    assert roster_client.get("/player/2026/nope").status_code == 404
    assert roster_client.get("/api/player/2026/nope").status_code == 404


def test_last_qb_names_the_starter_and_where_he_went():
    from cfbr.web.roster import last_qb

    players = pl.DataFrame({"name": ["Backup Bob"], "status": ["returner"], "prev_pass_plays": [40]})
    deps = pl.DataFrame({"name": ["Gone Pro"], "reason": ["drafted"], "draft_round": [1], "to_team": [None],
                         "portal_destination": [None], "in_portal": [False], "prev_pass_plays": [410]},
                        schema_overrides={"to_team": pl.Utf8, "portal_destination": pl.Utf8})
    assert last_qb(players, deps, {"qb_returns": 0.0}) == {
        "name": "Gone Pro", "detail": "drafted (Rd 1) · 410 pass att. last season"}
    moved = deps.with_columns(reason=pl.lit("transfer"), to_team=pl.lit("Ohio State"), draft_round=pl.lit(None, pl.Int32))
    assert last_qb(players, moved, {"qb_returns": 0.0})["detail"].startswith("now at Ohio State")
    # Returner with the most pass plays is the starter.
    assert last_qb(players, deps.clear(), {"qb_returns": 1.0}) == {
        "name": "Backup Bob", "detail": "returns · 40 pass att. last season"}
    # Older build: departures lack pass plays, so the feature decides and no name is guessed.
    old = deps.drop("prev_pass_plays")
    assert last_qb(players, old, {"qb_returns": 0.0})["name"] == "Not back"
    assert last_qb(players.clear(), old, None) == {"name": "—", "detail": None}


# -- review W1/W2: postponed games, build atomicity, memoization, cache headers -------------

def _games_df(rows):
    return pl.DataFrame(rows, schema_overrides={"home_points": pl.Int64, "away_points": pl.Int64,
                                                "margin": pl.Float64, "spread": pl.Float64})


def test_completed_weeks_moves_a_postponed_game_to_its_kickoff_week(caplog):
    """A week-2 game moved three weeks out (CFBD week number unchanged) must not hold week 2 open."""
    from types import SimpleNamespace

    from cfbr.web.build import completed_weeks, effective_week

    now = T0 + timedelta(days=10)
    rows = [_game(1, "r01", "Texas A&M", "Ohio State", T0, True, 31, 10),
            _game(2, "r02", "Hawai'i", "Ohio State", T0 + timedelta(days=7), True, 17, 24),
            _game(3, "r02", "Texas A&M", "Hawai'i", T0 + timedelta(days=21), False),  # moved to week 4's weekend
            _game(4, "r03", "Ohio State", "Texas A&M", T0 + timedelta(days=14), False),
            _game(5, "r04", "Hawai'i", "Texas A&M", T0 + timedelta(days=21), False)]
    eng = SimpleNamespace(games=_games_df(rows))
    with caplog.at_level("WARNING", logger="cfbr.web.build"):
        assert completed_weeks(eng, SEASON, now) == [1, 2]
    assert "not renumbered" in caplog.text
    eff = effective_week(eng.games)
    assert dict(zip(eff["game_id"].to_list(), eff["week"].to_list())) == {1: 1, 2: 2, 3: 4, 4: 3, 5: 4}
    # Once the makeup is played, week 4 completes with it.
    eng.games = eng.games.with_columns(completed=pl.col("game_id").is_in([3, 4, 5]) | pl.col("completed"))
    assert completed_weeks(eng, SEASON, now + timedelta(days=30)) == [1, 2, 3, 4]


def test_completed_play_weeks_moves_a_postponed_game(caplog):
    from cfbr.ingest.fetch import completed_play_weeks

    def raw(gid, week, start, completed):
        return {"id": gid, "seasonType": "regular", "week": week, "startDate": start.isoformat().replace("+00:00", "Z"),
                "completed": completed, "homeClassification": "fbs", "awayClassification": "fbs",
                "homeTeam": "H", "awayTeam": "A"}
    games = [raw(1, 1, T0, True), raw(2, 2, T0 + timedelta(days=7), True),
             raw(3, 2, T0 + timedelta(days=21), False), raw(4, 3, T0 + timedelta(days=14), True),
             raw(5, 4, T0 + timedelta(days=21), False)]
    with caplog.at_level("WARNING", logger="cfbr.ingest.fetch"):
        assert completed_play_weeks(games) == [("regular", 1), ("regular", 2), ("regular", 3)]
    assert "week 2" in caplog.text and "week 4" in caplog.text


def test_default_slot_skips_a_stale_game(web_cfg):
    """A postponed/cancelled game (kickoff long past, never completed) doesn't pin the default slot."""
    now = datetime.now(UTC)
    root = web_dir(web_cfg)
    d = _write_season(root, now)
    g = _games(now).with_columns(kickoff_ts=pl.when(pl.col("game_id") == 3).then(pl.lit(now - timedelta(days=10)))
                                 .otherwise(pl.col("kickoff_ts")))
    g = pl.concat([g, _games_df([_game(5, "r03", "Ohio State", "Hawai'i", now + timedelta(days=9), False)])])
    g.write_parquet(d / "games.parquet")
    meta = d / "meta.json"
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    s = Store(root, web_cfg.paths.results_dir, web_cfg.paths.out_dir).season(SEASON)
    assert s is not None
    assert default_slot(s, now) == "r02"  # game 4 (fresh, upcoming) still keeps r02 current
    s2 = Season(s.meta, s.ratings, s.games.filter(pl.col("game_id") != 4))
    assert default_slot(s2, now) == "r03"  # only the stale game left in r02: move on


def test_game_status_labels_postponed_and_cancelled():
    from cfbr.web.app import game_status

    now = datetime.now(UTC)
    stale = {"completed": False, "kickoff_ts": now - timedelta(days=10)}
    assert game_status(stale, now, live=True) == "postponed"
    assert game_status(stale, now, live=False) == "cancelled"
    assert game_status({"completed": False, "kickoff_ts": now + timedelta(days=1)}, now, True) == "upcoming"
    assert game_status({"completed": True, "kickoff_ts": now - timedelta(days=10)}, now, True) == "final"


def test_pages_show_postponed_games(web_cfg):
    now = datetime.now(UTC)
    d = _write_season(web_dir(web_cfg), now)
    g = _games(now).with_columns(kickoff_ts=pl.when(pl.col("game_id") == 3).then(pl.lit(now - timedelta(days=10)))
                                 .otherwise(pl.col("kickoff_ts")))
    g.write_parquet(d / "games.parquet")
    meta = d / "meta.json"
    st = meta.stat()
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    c = TestClient(create_app(web_cfg))
    assert "postponed" in c.get("/games?season=2026&slot=r02").text
    assert "postponed" in c.get("/team/2026/Ohio%20State").text


def test_sim_skips_stale_games_in_live_mode(engine):
    """A game still not completed days after kickoff is not a game left to simulate."""
    from cfbr.sim.season import build_inputs

    season = 2026
    g = engine.games.filter((pl.col("season") == season) & (pl.col("season_type") == "regular"))
    stale = g.filter(~pl.col("completed")).sort("kickoff_ts")
    if stale.height == 0:
        pytest.skip("no uncompleted 2026 games in the store")
    as_of = stale["kickoff_ts"][0] - timedelta(hours=1)
    far = stale["kickoff_ts"].max() + timedelta(days=30)
    n_live = len(build_inputs(engine, season, as_of, now=far).g_mean)  # everything is stale from `far`
    n_then = len(build_inputs(engine, season, as_of, now=as_of).g_mean)
    assert n_live == 0 < n_then


def test_store_refuses_a_season_mid_rebuild(web_cfg):
    """ratings.parquet newer than meta.json means a rebuild is in progress: keep the old copy,
    or with nothing cached report the season as not ready (503, not a 404 or mixed data)."""
    root = web_dir(web_cfg)
    d = _write_season(root, datetime.now(UTC))
    store = Store(root, web_cfg.paths.results_dir, web_cfg.paths.out_dir)
    assert store.season(SEASON).ratings["rating"].max() == 21.0
    # Rebuild in progress: new parquet, old meta.
    _ratings().with_columns(rating=pl.col("rating") + 100).write_parquet(d / "ratings.parquet")
    meta = d / "meta.json"
    st = meta.stat()
    os.utime(d / "ratings.parquet", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert store.season(SEASON).ratings["rating"].max() == 21.0  # warm cache: previous copy
    cold = Store(root, web_cfg.paths.results_dir, web_cfg.paths.out_dir)
    assert cold.season(SEASON) is None and cold.rebuilding(SEASON)
    c = TestClient(create_app(web_cfg, cold))
    assert c.get("/rankings?season=2026").status_code == 503
    assert c.get("/api/seasons").status_code == 200 and c.get("/api/seasons").json() == []
    # meta.json lands last: both stores pick up the new build.
    os.utime(meta, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    assert store.season(SEASON).ratings["rating"].max() == 121.0
    assert cold.season(SEASON).ratings["rating"].max() == 121.0 and not cold.rebuilding(SEASON)


def test_api_seasons_skips_a_vanished_meta(web_cfg, monkeypatch):
    now = datetime.now(UTC)
    _write_finished(web_dir(web_cfg), now)
    _write_season(web_dir(web_cfg), now)
    store = Store(web_dir(web_cfg), web_cfg.paths.results_dir, web_cfg.paths.out_dir)
    c = TestClient(create_app(web_cfg, store))
    assert [m["season"] for m in c.get("/api/seasons").json()] == [2025, 2026]
    real = store.season
    monkeypatch.setattr(store, "seasons", lambda: [2025, 2026, 2027])  # listed, then gone
    monkeypatch.setattr(store, "season", lambda y: None if y == 2027 else real(y))
    assert [m["season"] for m in c.get("/api/seasons").json()] == [2025, 2026]


def test_accuracy_log_and_runs_are_memoized(web_cfg, monkeypatch):
    now = datetime.now(UTC)
    _write_season(web_dir(web_cfg), now)
    out = web_cfg.paths.out_dir / str(SEASON)
    out.mkdir(parents=True, exist_ok=True)
    hdr = "run_at,season,through_week,n_games,mae,line_mae,massey_mae,ats_win_rate\n"
    (out / "accuracy.csv").write_text(hdr + "2026-10-05T15:00:00+00:00,2026,5,50,13.1,12.0,15.0,0.5\n")
    results = web_cfg.paths.results_dir
    run = results / "20260101T000000Z-a"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps(METRICS))
    store = Store(web_dir(web_cfg), results, web_cfg.paths.out_dir)

    reads = {"csv": 0, "json": 0}
    real_csv, real_json = pl.read_csv, json.loads
    monkeypatch.setattr(pl, "read_csv", lambda *a, **k: (reads.__setitem__("csv", reads["csv"] + 1), real_csv(*a, **k))[1])
    monkeypatch.setattr(json, "loads", lambda *a, **k: (reads.__setitem__("json", reads["json"] + 1), real_json(*a, **k))[1])
    for _ in range(3):
        assert store.accuracy_log(SEASON)["mae"][0] == 13.1
        assert [r.run_id for r in store.runs()] == ["20260101T000000Z-a"]
    assert reads["csv"] == 1 and reads["json"] == 1
    # A change is picked up: a new line in the log, a new run directory.
    (out / "accuracy.csv").write_text(hdr + "2026-10-05T15:00:00+00:00,2026,5,50,13.1,12.0,15.0,0.5\n"
                                      "2026-10-12T15:00:00+00:00,2026,6,60,12.9,12.0,15.0,0.5\n")
    st = (out / "accuracy.csv").stat()
    os.utime(out / "accuracy.csv", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert store.accuracy_log(SEASON).height == 2
    run2 = results / "20260102T000000Z-b"
    run2.mkdir()
    st = results.stat()
    os.utime(results, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert [r.run_id for r in store.runs()] == ["20260101T000000Z-a"]  # no metrics.json yet: pending
    (run2 / "metrics.json").write_text(json.dumps(METRICS | {"seasons": [2022, 2025]}))
    assert [r.run_id for r in store.runs()] == ["20260102T000000Z-b", "20260101T000000Z-a"]
    assert reads["json"] == 2
    # /accuracy computes each season's summary once per loaded Season.
    from cfbr.web import app as app_mod

    calls = {"n": 0}
    real_sum = app_mod.summarize
    monkeypatch.setattr(app_mod, "summarize", lambda df: (calls.__setitem__("n", calls["n"] + 1), real_sum(df))[1])
    c = TestClient(create_app(web_cfg, store))
    c.get("/accuracy")
    c.get("/accuracy")
    assert calls["n"] == 1


def test_finished_season_pages_are_publicly_cacheable(web_cfg):
    now = datetime.now(UTC)
    _write_finished(web_dir(web_cfg), now)
    _write_season(web_dir(web_cfg), now, live=True, polls=True)
    c = TestClient(create_app(web_cfg))
    for url in ("/rankings?season=2025", "/team/2025/Ohio%20State", "/games?season=2025&slot=r01",
                "/api/rankings?season=2025", "/api/games?season=2025&slot=r01", "/matchup?season=2025",
                "/accuracy?season=2025"):
        r = c.get(url)
        assert r.status_code == 200 and r.headers["cache-control"] == "public, max-age=3600", url
    for url in ("/rankings?season=2026", "/team/2026/Ohio%20State", "/api/games?season=2026&slot=r01",
                "/matchup?season=2026"):
        assert c.get(url).headers["cache-control"] == "private, no-store", url
    assert "cache-control" not in c.get("/accuracy?season=2026").headers
    # The free landing page carries a live-season notice: viewer-specific, so never shared.
    from cfbr.web.access import premium_viewer

    app = create_app(web_cfg)
    app.dependency_overrides[premium_viewer] = lambda: False
    r = TestClient(app).get("/rankings")
    assert "2026 ratings and rankings are for subscribers" in r.text
    assert r.headers["cache-control"] == "private, no-store"


# -- review W3: visitor features ---------------------------------------------------------

def test_explain_tiles_and_resume_on_team_page(poll_client):
    html = " ".join(poll_client.get("/team/2026/Texas%20A%26M").text.split())
    assert "Explain the rating" in html and "Offense: points per play" in html and "percentile" in html
    assert "Schedule and résumé" in html and "Best win" in html and "Record vs. top 25" in html


def test_explain_rating_percentiles():
    from cfbr.web.app import explain_rating, resume
    from cfbr.web.store import Season

    r = _ratings().with_columns(o_ppa=pl.col("rating") / 100)  # ranks 1..4 -> distinct values
    s = Season({"season": SEASON, "live": False, "weeks": [0, 1], "labels": {"0": "Preseason", "1": "Week 1"}},
               _ensure(r, ["ap_rank", "coaches_rank", "cfp_rank"]),
               _ensure(_games(T0), [f"{x}_{c}" for x in ("home", "away") for c in ("ap_rank", "coaches_rank", "cfp_rank")]))
    e = {x["label"]: x for x in explain_rating(s, "Texas A&M", 1)}
    assert e["Offense: points per play"]["pct"] == 1.0 and "100th percentile" in e["Offense: points per play"]["sub"]
    e = {x["label"]: x for x in explain_rating(s, "Ohio State", 1)}
    assert e["Offense: points per play"]["pct"] == 0.25
    res = resume(s, "Texas A&M", 1)
    assert res["best_win"]["opponent"] == "<b>Evil</b>" and res["worst_loss"] is None
    assert 1 <= res["sos_rank"] <= 4 and res["top25_w"] == 1


def test_rankings_vs_ap_column(poll_client):
    html = " ".join(poll_client.get("/rankings?season=2026&week=1").text.split())
    assert "vs AP" in html
    # Texas A&M: model #1, AP #3 -> model has the team higher than the poll: vs_ap = -2.
    api = poll_client.get("/api/rankings?season=2026&week=1").json()["teams"]
    assert {t["team"]: t["vs_ap"] for t in api}["Texas A&M"] == -2
    # On the page the arrow points the way the model moves the team: up two places, in green.
    assert '<span class="up">▲ 2</span>' in html


def test_margin_filter_never_reads_as_a_line():
    from cfbr.web.app import margin

    assert margin(9.04) == "wins by 9.0" and margin(-3.5) == "loses by 3.5"
    assert margin(0.02) == "toss-up" and margin(None) == "—" and margin(float("nan")) == "—"


def test_delta_shows_no_arrow_for_a_rounded_zero(client):
    tmpl = client.app.state.templates.env.from_string('{% from "macros.html" import delta %}{{ delta(x) }}')
    assert tmpl.render(x=0.02) == '<span class="muted">0</span>'
    assert "▲ 0.1" in tmpl.render(x=0.06) and "▼ 1.0" in tmpl.render(x=-1.0)


def test_poll_line_names_matching_polls_together():
    from cfbr.web.app import poll_line

    assert poll_line(["AP: week 7 poll", "Coaches: week 7 poll"]) == "AP and Coaches: week 7 polls."
    assert poll_line(["AP: week 7 poll", "Coaches: week 7 poll", "CFP: week 7 poll"]) == "AP, Coaches and CFP: week 7 polls."
    assert poll_line(["AP: final poll", "Coaches: final poll"]) == "AP and Coaches: final polls."
    assert poll_line(["AP: week 2 poll", "Coaches: week 1 poll (week 2 not released yet)"]) == \
        "AP: week 2 poll; Coaches: week 1 poll (week 2 not released yet)."
    assert poll_line([]) == ""


def _odds_event(eid, home, away, kickoff, books):
    """An Odds API event with spreads markets: books = [(title, home_point), ...]."""
    return {"id": eid, "home_team": home, "away_team": away,
            "commence_time": kickoff.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "bookmakers": [{"title": b, "markets": [{"key": "spreads", "outcomes": [
                {"name": home, "point": pt, "price": -110}, {"name": away, "point": -pt, "price": -110}]}]}
                           for b, pt in books]}


def test_apply_lines_overlays_only_unplayed_games_the_snapshot_carries():
    from cfbr.web.lines import apply_lines

    snap = datetime(2026, 10, 14, 21, tzinfo=UTC)
    games = pl.DataFrame({
        "game_id": [1, 2, 3, 4],
        "completed": [True, False, False, False],
        "kickoff_ts": [snap - timedelta(days=3), snap + timedelta(days=3), snap + timedelta(days=3), snap - timedelta(hours=1)],
        "cfbd_spread": [-7.0, -3.0, -10.0, -1.0],
    })
    fresh = (snap, pl.DataFrame({"game_id": [1, 2, 4], "spread": [-8.0, -4.5, -2.0]}))
    out = apply_lines(games, fresh, "2026-10-11T20:00:11+00:00").sort("game_id").to_dicts()
    # 1: final -> keeps CFBD's close. 2: unplayed and in the snapshot -> Odds API line.
    # 3: unplayed, not in the snapshot -> CFBD. 4: kicked off before the snapshot -> CFBD (no live lines).
    assert [(r["spread"], r["line_source"]) for r in out] == [(-7.0, "cfbd"), (-4.5, "odds"), (-10.0, "cfbd"), (-1.0, "cfbd")]
    assert out[1]["line_as_of"] == snap.isoformat() and out[2]["line_as_of"] == "2026-10-11T20:00:11+00:00"
    assert apply_lines(games, None, None)["spread"].to_list() == [-7.0, -3.0, -10.0, -1.0]
    # During a build kickoff_ts is local time (America/Chicago), not UTC; same result.
    local = games.with_columns(pl.col("kickoff_ts").dt.convert_time_zone("America/Chicago"))
    assert apply_lines(local, fresh, None).sort("game_id")["line_source"].to_list() == ["cfbd", "odds", "cfbd", "cfbd"]


def test_refresh_lines_patches_the_site_and_never_sticks(web_cfg):
    from cfbr.web.lines import refresh_lines

    d = _write_season(web_dir(web_cfg), datetime.now(UTC))
    games = pl.read_parquet(d / "games.parquet")
    # As built today: CFBD lines kept in cfbd_spread.
    games.with_columns(cfbd_spread=pl.col("spread")).write_parquet(d / "games.parquet")
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps(meta | {"lines_as_of": "2026-10-11T20:00:11+00:00",
                                                    "cfbd_lines_as_of": "2026-10-11T20:00:11+00:00"}))
    web_cfg.paths.raw_dir = web_cfg.paths.out_dir.parent / "raw"
    web_cfg.paths.db_path = web_cfg.paths.out_dir.parent / "db" / "cfb.duckdb"
    web_cfg.paths.db_path.parent.mkdir(parents=True)
    assert refresh_lines(web_cfg, SEASON) is None  # no snapshot yet: nothing changes

    # Game 3 (Texas A&M @ Ohio State, unplayed, CFBD −3) moves to −6.5 in a later Odds API snapshot.
    kick = games.filter(pl.col("game_id") == 3)["kickoff_ts"][0]
    snaps = web_cfg.paths.raw_dir / "odds" / str(SEASON) / "snapshots"
    snaps.mkdir(parents=True)
    ts = datetime.now(UTC).replace(microsecond=0)
    (snaps / f"{ts:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(
        [_odds_event("e3", "Ohio State", "Texas A&M", kick, [("DraftKings", -6.0), ("FanDuel", -7.0)])]))
    before = (d / "meta.json").stat().st_mtime_ns
    assert refresh_lines(web_cfg, SEASON) == 1
    after = pl.read_parquet(d / "games.parquet").sort("game_id")
    assert after.filter(pl.col("game_id") == 3)["spread"][0] == -6.5  # median of −6 and −7
    assert after.filter(pl.col("game_id") == 3)["line_source"][0] == "odds"
    assert after.filter(pl.col("game_id") == 1)["spread"][0] == -7.0  # final: CFBD close untouched
    # Predictions untouched; meta rewritten last with the snapshot time.
    assert after["pred_margin"].to_list() == games.sort("game_id")["pred_margin"].to_list()
    meta = json.loads((d / "meta.json").read_text())
    assert meta["lines_as_of"] == ts.isoformat() and (d / "meta.json").stat().st_mtime_ns > before
    # Each book's line for the refreshed game, written before meta.json.
    books = pl.read_parquet(d / "book_lines.parquet")
    assert books.select("game_id", "book", "home_spread").rows() == [(3, "DraftKings", -6.0), (3, "FanDuel", -7.0)]
    assert set(books["as_of"]) == {ts.isoformat()}
    assert (d / "book_lines.parquet").stat().st_mtime_ns <= (d / "meta.json").stat().st_mtime_ns
    html = " ".join(TestClient(create_app(web_cfg)).get("/games?season=2026&slot=r02").text.split())
    assert "Ohio State −6.5" in html

    # A newer snapshot without the game: it falls back to CFBD instead of keeping the stale odds line.
    (snaps / f"{ts + timedelta(minutes=5):%Y%m%dT%H%M%SZ}.json").write_text("[]")
    assert refresh_lines(web_cfg, SEASON) == 0
    g3 = pl.read_parquet(d / "games.parquet").filter(pl.col("game_id") == 3)
    assert g3["spread"][0] == -3.0 and g3["line_source"][0] == "cfbd"
    assert pl.read_parquet(d / "book_lines.parquet").height == 0  # no stale book lines either


def test_refresh_lines_skips_while_the_weekly_lock_is_held(web_cfg):
    import fcntl

    from cfbr.web.lines import refresh_lines

    d = _write_season(web_dir(web_cfg), datetime.now(UTC))
    pl.read_parquet(d / "games.parquet").with_columns(cfbd_spread=pl.col("spread")).write_parquet(d / "games.parquet")
    web_cfg.paths.raw_dir = web_cfg.paths.out_dir.parent / "raw"
    web_cfg.paths.db_path = web_cfg.paths.out_dir.parent / "db" / "cfb.duckdb"
    web_cfg.paths.db_path.parent.mkdir(parents=True)
    snaps = web_cfg.paths.raw_dir / "odds" / str(SEASON) / "snapshots"
    snaps.mkdir(parents=True)
    (snaps / "20261014T210000Z.json").write_text("[]")
    with open(web_cfg.paths.db_path.parent / ".weekly.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = (d / "games.parquet").read_bytes()
        assert refresh_lines(web_cfg, SEASON) is None
        assert (d / "games.parquet").read_bytes() == before


def test_an_older_odds_snapshot_never_rolls_lines_back():
    from cfbr.web.lines import newer_than

    sat = datetime(2026, 10, 17, 23, tzinfo=UTC)
    fresh = (sat, pl.DataFrame({"game_id": [1], "spread": [-3.0]}))
    # Sunday's CFBD fetch succeeded but the Odds API fetch failed: Saturday's odds snapshot is older.
    assert newer_than(fresh, "2026-10-18T14:00:00+00:00") is None
    assert newer_than(fresh, "2026-10-17T14:00:00+00:00") is fresh  # older CFBD: odds apply
    assert newer_than(fresh, None) is fresh and newer_than(None, "2026-10-18T14:00:00+00:00") is None


def test_games_mark_lines_older_than_the_snapshot(web_cfg):
    d = _write_season(web_dir(web_cfg), datetime.now(UTC))
    games = pl.read_parquet(d / "games.parquet")
    games.with_columns(line_source=pl.lit("cfbd"), line_as_of=pl.lit("2026-10-11T20:00:11+00:00")).write_parquet(d / "games.parquet")
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps(meta | {"lines_as_of": "2026-10-14T21:00:00+00:00",
                                                    "cfbd_lines_as_of": "2026-10-11T20:00:11+00:00"}))
    html = " ".join(TestClient(create_app(web_cfg)).get("/games?season=2026&slot=r02").text.split())
    assert "Lines marked * are older (Oct 11, 3:00 PM CT)" in html and 'title="Older line: as of' in html
    # Snapshots seconds apart (the same run) aren't "older": no marker, no note.
    (d / "meta.json").write_text(json.dumps(meta | {"lines_as_of": "2026-10-11T20:00:55+00:00",
                                                    "cfbd_lines_as_of": "2026-10-11T20:00:11+00:00"}))
    html = TestClient(create_app(web_cfg)).get("/games?season=2026&slot=r02").text
    assert "Older line" not in html and "Lines marked *" not in html


def test_unplayed_lines_say_when_they_were_fetched(web_cfg, tmp_path):
    from cfbr.web.build import lines_as_of

    raw = tmp_path / "raw"
    assert lines_as_of(raw, 2026) is None
    (raw / "lines" / "2026" / "snapshots").mkdir(parents=True)
    for stem in ("20261011T162408Z", "20261011T200011Z"):
        (raw / "lines" / "2026" / "snapshots" / f"{stem}.json").write_text("[]")
    assert lines_as_of(raw, 2026) == "2026-10-11T20:00:11+00:00"  # the newest snapshot

    d = _write_season(web_dir(web_cfg), datetime.now(UTC))
    meta = json.loads((d / "meta.json").read_text())
    c = TestClient(create_app(web_cfg))
    # Week 2 has unplayed games; no snapshot recorded -> no claim about freshness.
    assert "lines-as-of" not in c.get("/games?season=2026&slot=r02").text
    (d / "meta.json").write_text(json.dumps(meta | {"lines_as_of": "2026-10-11T20:00:11+00:00"}))
    c = TestClient(create_app(web_cfg))
    html = c.get("/games?season=2026&slot=r02").text
    assert "Vegas lines for games not yet played are as of Oct 11, 3:00 PM CT" in html
    # A week that's all final doesn't need it.
    assert "lines-as-of" not in c.get("/games?season=2026&slot=r01").text
    # Nor does a finished season, even with leftover uncompleted rows (cancelled games).
    (d / "meta.json").write_text(json.dumps(meta | {"lines_as_of": "2026-10-11T20:00:11+00:00", "live": False}))
    assert "lines-as-of" not in TestClient(create_app(web_cfg)).get("/games?season=2026&slot=r02").text


def test_more_columns_note_lists_edge_cents_and_cover_when_shown(web_cfg):
    from cfbr.bets.ledger import read_ledger

    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    path = _write_ledger(web_cfg)
    c = TestClient(create_app(web_cfg))
    html = " ".join(c.get("/bets").text.split())
    assert '"More columns" adds Edge now, Tier,' in html  # neither column in this ledger
    read_ledger(path).with_columns(edge_cents=pl.lit(12.0), cover_prob=pl.lit(0.51)).write_csv(path)
    html = " ".join(c.get("/bets").text.split())
    assert '"More columns" adds Edge now, Edge ¢, Cover %, Tier,' in html


def test_games_and_bets_label_their_line_sources(client, bets_client):
    games = client.get("/games?season=2026&slot=r01").text
    assert "for games not yet played, the closing line once final" in games and ">Edge</th>" in games
    team = client.get("/team/2026/Texas%20A%26M").text
    assert "for games not yet played, the closing line once final" in team
    bets = bets_client.get("/bets").text
    assert ">Bet (line at bet)</th>" in bets and "wins by" in bets


def test_bad_week_error_hides_slot_ids(client):
    r = client.get("/games?season=2026&slot=r6")
    assert r.status_code == 404 and "No such week in 2026" in r.text and "r6" not in r.text


def test_malformed_params_get_the_error_page_not_json(client):
    r = client.get("/rankings?week=abc")
    assert r.status_code == 400 and "text/html" in r.headers["content-type"]
    assert "Bad link" in r.text and "invalid week" in r.text
    # The API keeps FastAPI's validation detail.
    r = client.get("/api/rankings?week=abc")
    assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "week"


def test_games_sort_and_conference_filter(client):
    html = client.get("/games?season=2026&slot=r01").text
    assert '<table class="data sortable" id="games">' in html and 'data-sort="num" title="Points between' in html
    assert "<option >SEC</option>" in html and '<option value="top25"' in html
    sec = client.get("/games?season=2026&slot=r01&conf=SEC").text
    assert "<option selected>SEC</option>" in sec
    # A conference with no games this week is ignored rather than emptying the table.
    assert client.get("/games?season=2026&slot=r01&conf=Nope").status_code == 200
    # Kickoff cells drop the time zone; the header carries it.
    assert "Kickoff (CT)" in html and " PM CT</td>" not in html and " AM CT</td>" not in html


def test_bets_show_the_line_now(bets_client):
    from cfbr.web.bets import with_line_now

    html = bets_client.get("/bets").text
    assert ">Line now</th>" in html and ">Best available</th>" in html  # Best available no longer phone-hidden
    s = type("S", (), {"games": pl.DataFrame({"game_id": [7], "spread": [-1.0]}), "meta": {}})()
    row = {"game_id": "7", "side": "away", "model_team": 4.0}
    out = with_line_now([row], s, 3.0)[0]
    # Home −1 is away +1; model has the away side winning by 4 -> 5 points off, still past the threshold.
    assert out["line_now"] == 1.0 and out["edge_now"] == 5.0 and not out["below_threshold"]
    assert with_line_now([row | {"side": "home", "model_team": 3.0}], s, 3.0)[0]["below_threshold"]


def test_flagged_games_are_highlighted_and_filterable(client):
    html = " ".join(client.get("/games?season=2026&slot=r01").text.split())
    # Game 1: model Texas A&M by 10, line Texas A&M -7: 3-point edge -> flagged at the bet threshold.
    assert 'class="flagged"' in html and "Upset alerts" not in html
    assert "(flagged: 3+ off the line)" in html  # a text marker, not just the row tint
    assert "Flagged <span class=\"muted\">1</span>" in html
    # "Flagged" is defined once, under the chips, with the season's public ATS as a caution.
    assert html.count("picks the underdog, or is 3+ points off the") == 1
    assert "against the spread this season" in html and "break-even at −110 is 52.4%" in html
    only = " ".join(client.get("/games?season=2026&slot=r01&show=flagged").text.split())
    assert only.count('class="flagged"') == only.count("<tr class") == 1
    # An unknown filter falls back to all games.
    assert client.get("/games?season=2026&slot=r01&show=bogus").status_code == 200
    r = game_row({"home_team": "H", "away_team": "A", "pred_margin": -2.0, "win_prob": 0.42, "spread": -3.0,
                  "margin": None, "completed": False})
    assert r["alert"] == "A to win outright (58%)" and r["flag"] == "upset"
    r = game_row({"home_team": "H", "away_team": "A", "pred_margin": 4.0, "win_prob": 0.6, "spread": -3.0,
                  "margin": None, "completed": False})
    assert r["alert"] is None and r["flag"] is None


def test_flags_withheld_from_free_viewers(free_client):
    # A flag reveals the model's side, so free viewers get neither highlighting nor the filter.
    html = free_client.get("/games?season=2026&slot=r01").text
    assert 'class="flagged"' not in html and "show=flagged" not in html
    html = free_client.get("/games?season=2026&slot=r01&show=flagged").text
    assert 'class="flagged"' not in html


def test_matchup_page(web_cfg):
    now = datetime.now(UTC)
    d = _write_season(web_dir(web_cfg), now, live=False)
    meta = json.loads((d / "meta.json").read_text())
    meta["calibration"] = {"b_home": 2.5, "b_ppa": 30.0, "b_sr": 40.0, "sigma": 16.0, "stage2": None}
    (d / "meta.json").write_text(json.dumps(meta))
    c = TestClient(create_app(web_cfg))
    html = c.get("/matchup?season=2026").text
    assert "Head to head" in html and 'id="h2h-home"' in html and '"b_home": 2.5' in html
    assert "cdn" not in html.lower()  # no external assets
    # A shared link preselects both teams and the neutral box; unknown names fall back.
    html = c.get("/matchup?season=2026&home=Ohio%20State&away=Texas%20A%26M&neutral=1").text
    assert '<option value="Ohio State" selected>' in html and '<option value="Texas A&amp;M" selected>' in html
    assert 'id="h2h-neutral" checked' in html
    # The season/week form carries the picks, so changing the week keeps the matchup.
    form = html[html.index('<form class="filters"'):html.index("</form>")]
    assert 'name="home" id="h2h-q-home" value="Ohio State"' in form
    assert 'name="away" id="h2h-q-away" value="Texas A&amp;M"' in form
    assert 'name="neutral" id="h2h-q-neutral" value="1">' in form  # enabled, so it submits
    html = c.get("/matchup?season=2026&home=Nobody&away=Nobody").text
    assert html.count(" selected>") >= 2  # still two different teams picked
    # Matchup lives under Games in the nav, as the second tab.
    assert 'href="/games" aria-current="page">Games</a>' in html and 'aria-current="page">Any matchup</a>' in html
    # Free viewers on a live season see no ratings in the page.
    from cfbr.web.access import premium_viewer

    meta["live"] = True
    (d / "meta.json").write_text(json.dumps(meta))
    app = create_app(web_cfg)
    app.dependency_overrides[premium_viewer] = lambda: False
    r = TestClient(app).get("/matchup?season=2026")
    assert r.status_code == 200 and "for subscribers" in r.text and "h2h-data" not in r.text
    for leak in MODEL_STRINGS:
        assert leak not in r.text

def test_model_name_is_site_text_only(client):
    html = client.get("/games?season=2026&slot=r01").text
    assert f">{MODEL_NAME}</th>" in html and f">{MODEL_NAME} err</th>" in html
    assert f"called <strong>{MODEL_NAME}</strong>" in client.get("/method").text
    # The API keeps its field names; the name is a display label.
    api = client.get("/api/games?season=2026&slot=r01").text
    assert "pred_margin" in api and MODEL_NAME not in api


def test_footer_shows_build_time(client):
    assert "data built" in client.get("/rankings").text


def test_method_page_holds_shared_definitions(client, web_cfg):
    r = client.get("/method")
    assert r.status_code == 200 and r.headers["cache-control"] == "public, max-age=3600"
    html = " ".join(r.text.split())
    assert "never model inputs" in html and "Tuning seasons</strong> (2015–2021)" in html
    # Pooling follows the config the site runs on (tuned.toml pools by conference).
    pooled = "by conference" if web_cfg.prep.fcs_pooling == "conference" else "into a single rated team"
    assert f"FCS opponents are pooled {pooled}" in html
    for anchor in ("rating", "point-in-time", "lines", "polls", "accuracy", "projections", "rosters"):
        assert f'id="{anchor}"' in html
    # Every page links to it from the footer instead of repeating the caveats.
    rankings = client.get("/rankings").text
    assert '<a href="/method">' in rankings and "never model inputs" not in rankings


def test_bets_hide_columns_the_ledger_never_filled(bets_client):
    # The synthetic ledger predates Edge ¢ and Cover %, so those columns are left out, not shown as dashes.
    html = bets_client.get("/bets").text
    assert ">Edge ¢</th>" not in html and ">Cover %</th>" not in html


def _write_real(cfg):
    from cfbr.bets.real import REAL_SCHEMA, write_real

    rows = [{"game_id": 1, "season": SEASON, "week": 1, "kickoff_ts": "2026-09-05T16:00:00+00:00",
             "home_team": "Texas A&M", "away_team": "Ohio State", "side": "away", "team": "Ohio State",
             "line": -1.5, "odds": -110, "risk_dollars": 10.0, "book": "DraftKings",
             "placed_at": "2026-09-01T04:06:17+00:00", "recorded_at": "2026-09-01T14:00:00+00:00", "status": "open"}]
    write_real(pl.DataFrame(rows, schema=REAL_SCHEMA), cfg.paths.out_dir / str(SEASON) / "bets_real.csv")


def test_real_bets_are_owner_only(web_cfg, monkeypatch):
    from fastapi import HTTPException

    from cfbr.web.access import OWNER_COOKIE, bets_access, owner_token

    monkeypatch.setenv("OWNER_KEY", "s3cret-key")
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_ledger(web_cfg)
    before = TestClient(create_app(web_cfg)).get("/api/bets").json()
    _write_real(web_cfg)
    app = create_app(web_cfg)
    c = TestClient(app, follow_redirects=False)

    # Premium (everyone today) but not the owner: no section.
    html = c.get("/bets").text
    assert 'id="mine"' not in html and "Ohio State +1.5" not in html
    # The key travels only in a POST body: a key in the URL does nothing (URLs land in the
    # server log and browser history).
    form = c.get("/owner")
    assert form.status_code == 200 and 'method="post"' in form.text and 'type="password"' in form.text
    assert c.get("/owner?key=s3cret-key").status_code == 200 and OWNER_COOKIE not in c.cookies
    # A wrong key and a forged cookie both get nothing.
    r = c.post("/owner", content="key=nope", headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 403 and "didn't match" in r.text and OWNER_COOKIE not in c.cookies
    c.cookies.set(OWNER_COOKIE, owner_token("nope"))
    assert 'id="mine"' not in c.get("/bets").text
    c.cookies.clear()

    # The right key sets an HttpOnly cookie holding an HMAC, never the key itself.
    r = c.post("/owner", content="key=s3cret-key", headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 303 and r.headers["location"] == "/bets#mine"
    set_cookie = r.headers["set-cookie"]
    assert "HttpOnly" in set_cookie and "samesite=lax" in set_cookie.lower() and "s3cret-key" not in set_cookie
    assert owner_token("s3cret-key") in set_cookie
    html = " ".join(c.get("/bets").text.split())
    assert 'id="mine">My real bets' in html and "Ohio State +1.5" in html and "$10 / $9" in html
    assert "these are real wagers" in html and "Your bets" not in html
    # Below the paper record: after its rules and open bets.
    assert html.index('id="how"') < html.index('id="mine"')
    assert c.get("/bets").headers["cache-control"] == "private, no-store"

    # Never in the API, never changes the paper record, never on other pages, even for the owner.
    after = c.get("/api/bets").json()
    assert after["summary"] == before["summary"] and len(after["bets"]) == len(before["bets"])
    assert '"risk_dollars"' not in c.get("/api/bets").text
    for url in ("/rankings", "/games?slot=r01", "/team/2026/Ohio%20State", "/method"):
        assert "My real bets" not in c.get(url).text, url

    # The betting gate still applies to the owner too.
    app.dependency_overrides[bets_access] = lambda: (_ for _ in ()).throw(HTTPException(402, "no"))
    assert c.get("/bets").status_code == 402
    app.dependency_overrides.clear()

    # Logging out clears it.
    r = c.get("/owner/logout")
    assert r.status_code == 303 and OWNER_COOKIE in r.headers["set-cookie"]
    c.cookies.clear()
    assert 'id="mine"' not in c.get("/bets").text


def test_owner_sees_a_season_with_only_real_bets(web_cfg, monkeypatch):
    from cfbr.web.access import OWNER_COOKIE, owner_token

    monkeypatch.setenv("OWNER_KEY", "s3cret-key")
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_real(web_cfg)  # no bets.csv at all
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    # Others: the betting page has no season, and the API has no paper bets.
    assert 'id="mine"' not in c.get("/bets").text
    assert c.get("/bets?season=2026").status_code == 404 and c.get("/api/bets").status_code == 404
    c.cookies.set(OWNER_COOKIE, owner_token("s3cret-key"))
    html = " ".join(c.get("/bets").text.split())
    assert 'id="mine">My real bets' in html and "Ohio State +1.5" in html and "No paper bets recorded yet" in html
    assert c.get("/bets?season=2026").status_code == 200
    assert c.get("/api/bets").status_code == 404  # the API stays paper-only, owner or not


def test_no_owner_key_means_no_owner(web_cfg, monkeypatch):
    from cfbr.web import access
    from cfbr.web import app as app_mod
    from cfbr.web.access import OWNER_COOKIE, owner_token

    # As if neither the environment nor .keys has one (javelin's .keys may).
    monkeypatch.delenv("OWNER_KEY", raising=False)
    monkeypatch.setattr(access, "owner_key", lambda: None)
    monkeypatch.setattr(app_mod, "owner_key", lambda: None)
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_ledger(web_cfg)
    _write_real(web_cfg)
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    assert c.get("/owner").status_code == 404
    assert c.post("/owner", content="key=anything",
                  headers={"content-type": "application/x-www-form-urlencoded"}).status_code == 404
    c.cookies.set(OWNER_COOKIE, owner_token(""))
    assert 'id="mine"' not in c.get("/bets").text


def test_single_book_bets_are_explained_only_when_present(web_cfg):
    from cfbr.bets.ledger import read_ledger

    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    path = _write_ledger(web_cfg)  # every fixture bet has 3 books
    c = TestClient(create_app(web_cfg))
    assert "only one book" not in c.get("/bets").text
    led = read_ledger(path)
    led.with_columns(n_providers=pl.when(pl.col("game_id").is_in([1, 2])).then(1)
                     .otherwise(pl.col("n_providers")).cast(pl.Int32)).write_csv(path)
    html = " ".join(c.get("/bets").text.split())
    assert "2 bets were recorded when only one book's line was available (Books = 1)" in html


def test_open_bets_hide_secondary_columns_by_default(bets_client):
    html = bets_client.get("/bets").text
    table = html[html.index('id="open-bets"'):]
    table = table[:table.index("</table>")]
    assert 'data-target="open-bets"' in html  # the "More columns" toggle
    # Kept by default: what a bettor acts on.
    for th in (">Game</th>", ">Bet (line at bet)</th>", ">Edge</th>", ">Best available</th>"):
        assert th in table and f'extra">{th[1:]}' not in table, th
    # Behind "More columns".
    for th in ("Tier", "Risk / to win", "Books", "Edge now"):
        assert f'extra">{th}</th>' in table, th


def test_bets_lead_with_open_bets(bets_client):
    html = bets_client.get("/bets").text
    # Open bets come before the rules and the results.
    assert html.index("<h2>Open bets</h2>") < html.index('id="how"') < html.index("<h2>Results</h2>")
    # With settled bets, results show in full rather than collapsed.
    assert "no settled bets yet" not in html


def test_track_record_is_a_ratings_tab(client):
    from cfbr.web import MODEL_NAME

    rankings = client.get("/rankings?season=2026&week=1").text
    accuracy = client.get("/accuracy?season=2026").text
    for html, here in ((rankings, "Rankings"), (accuracy, "Track record")):
        tabs = html[html.index('<nav class="tabs" aria-label="Rankings">'):]
        tabs = tabs[:tabs.index("</nav>")]
        assert ">Rankings</a>" in tabs and ">Track record</a>" in tabs
        assert f'aria-current="page">{here}</a>' in tabs
    # Rankings names the model and points at its track record.
    assert f"{MODEL_NAME} ratings: each team's expected margin" in rankings
    assert f'<a href="/accuracy?season=2026">How accurate is {MODEL_NAME}?' in rankings


def test_games_sort_and_filter_fbs_conferences(web_cfg):
    d = _write_season(web_dir(web_cfg), datetime.now(UTC))
    # Week 2's FCS opponent (Montana, from the FCS:Big Sky pool) plays in a real FCS league.
    games = pl.read_parquet(d / "games.parquet")
    games.with_columns(away_conference=pl.when(pl.col("away_team") == "Montana").then(pl.lit("Big Sky"))
                       .otherwise(pl.col("away_conference"))).write_parquet(d / "games.parquet")
    (d / "meta.json").write_text((d / "meta.json").read_text())  # meta last, as the build does
    client = TestClient(create_app(web_cfg))
    html = client.get("/games?season=2026&slot=r02").text
    assert "Montana" in html
    # CFBrR margin and the Vegas line sort too.
    assert html.count('<th data-sort="num"') >= 5 and '<th data-sort="num" title="Median spread' in html
    # Only FBS teams' conferences are offered.
    conf = html[html.index('name="conf"'):html.index("</select>", html.index('name="conf"'))]
    assert "Big Sky" not in conf and "<option >SEC</option>" in conf
    assert "Big Sky" in pl.read_parquet(d / "games.parquet")["away_conference"].to_list()  # the check is live


def test_flagged_views_render_biggest_edge_first(web_cfg):
    d = _write_season(web_dir(web_cfg), datetime.now(UTC))
    # Two more week-2 games, both flagged, kicking off EARLIER than the existing ones and in
    # ascending-edge order, so kickoff order and edge order disagree: only the route's sort
    # can put the bigger edge first.
    games = pl.read_parquet(d / "games.parquet")
    base = games.filter(pl.col("game_id") == 3).to_dicts()[0]
    early = base["kickoff_ts"] - timedelta(days=1)
    extra = [
        # Edge 4: model Small U by 10, line −6 (both favor the home team: an "edge" flag).
        base | {"game_id": 10, "home_team": "Small U", "away_team": "Tiny Tech", "home": "Small U", "away": "Tiny Tech",
                "kickoff_ts": early, "as_of": early, "pred_margin": 10.0, "win_prob": 0.75, "spread": -6.0},
        # Edge 9: model Big State by 6, line Big State +3 (model picks the underdog: an "upset" flag).
        base | {"game_id": 11, "home_team": "Big State", "away_team": "Giant A&M", "home": "Big State", "away": "Giant A&M",
                "kickoff_ts": early + timedelta(hours=1), "as_of": early, "pred_margin": 6.0, "win_prob": 0.65, "spread": 3.0},
    ]
    pl.concat([games, pl.DataFrame(extra, schema=games.schema)]).write_parquet(d / "games.parquet")
    (d / "meta.json").write_text((d / "meta.json").read_text())  # meta last, as the build does
    c = TestClient(create_app(web_cfg))
    every = c.get("/games?season=2026&slot=r02").text
    assert every.index("Small U") < every.index("Big State")  # the plain view stays in kickoff order
    flagged = c.get("/games?season=2026&slot=r02&show=flagged").text
    assert flagged.index("Big State") < flagged.index("Small U")  # edge 9 before edge 4
    upsets = c.get("/games?season=2026&slot=r02&show=upsets").text
    assert "Big State" in upsets and "Small U" not in upsets


def test_books_tooltip_reads_from_the_bet_team_s_side():
    from cfbr.web.bets import books_from_side

    books = "DraftKings +5.5 (-110/-105); Bovada +6 (-110/-110) (offshore); bet365 +5 (pricing only); odd entry"
    # Away bet: the home spread flips and the away price shows.
    assert books_from_side(books, home=False) == \
        "DraftKings −5.5 (−105); Bovada −6 (−110) (offshore); bet365 −5 (pricing only); odd entry"
    assert books_from_side(books, home=True).startswith("DraftKings +5.5 (−110); Bovada +6 (−110)")
    assert books_from_side(None, True) is None


def test_column_definitions_are_visible_not_just_tooltips(client):
    html = client.get("/rankings?season=2026&week=1").text
    legend = html[html.index('<details class="legend">'):]
    legend = legend[:legend.index("</details>")]
    # Same text in the tooltip and the legend.
    assert 'title="How much of the rating still comes from the preseason prior">Prior</th>' in html
    assert "<dt>Prior</dt><dd>How much of the rating still comes from the preseason prior</dd>" in legend
    for url in ("/games?season=2026&slot=r01", "/team/2026/Ohio%20State"):
        assert '<details class="legend">' in client.get(url).text, url


def test_win_probability_never_rounds_to_certain():
    from cfbr.web.odds import prob

    assert prob(0.996) == ">99%" and prob(0.004) == "<1%" and prob(0.5) == "50%"


def test_independents_get_no_conference_title_odds(odds_client, web_cfg):
    import polars as pl

    d = web_cfg.paths.out_dir / "2026" / "w01"
    sim = pl.read_csv(d / "season_sim.csv")
    sim.with_columns(conference=pl.when(pl.col("team") == "Ohio State").then(pl.lit("FBS Independents"))
                     .otherwise(pl.col("conference"))).write_csv(d / "season_sim.csv")
    html = " ".join(odds_client.get("/odds?season=2026&week=1").text.split())
    row = html[html.index(">Ohio State</a>"):]
    row = row[:row.index("</tr>")]
    assert row.count(">—</td>") == 2  # title game and conference title


def test_nav_and_skip_link(client):
    html = client.get("/rankings").text
    nav = html[html.index('<nav aria-label="Sections">'):html.index("</nav>")]
    assert [s.split("</a>")[0].split(">")[-1] for s in nav.split("<a ")[1:]] == \
        ["This week", "Rankings", "Games", "Projections", "Bets"]
    assert '<a class="skip" href="#main">' in html and '<main id="main"' in html
    # Old routes still work under the new labels.
    for url, label in (("/odds", None), ("/accuracy", "Rankings"), ("/matchup", "Games"), ("/week", "This week")):
        if label:
            assert f'aria-current="page">{label}</a>' in client.get(url).text, url


def test_build_exports_calibration(engine, tmp_path):
    build_season(engine, 2023, tmp_path, now=datetime.now(UTC))
    cal = json.loads((tmp_path / "2023" / "meta.json").read_text())["calibration"]
    assert cal["b_home"] == pytest.approx(engine.calibration(2023).b_home) and cal["sigma"] > 0


# -- shortlist book choice ---------------------------------------------------------------


def test_choose_books_validates_against_the_selectable_list():
    from cfbr.config import Bets
    from cfbr.web.access import book_cookie_value, choose_books

    b = Bets()
    assert choose_books(["draftkings", "Bovada", "nope", " FanDuel "], b) == {"DraftKings", "FanDuel"}
    assert choose_books([], b) == choose_books(["Bovada"], b) == frozenset(b.pricing_books)  # offshore is not selectable
    assert "Bovada" not in b.selectable_books and set(b.pricing_books) <= set(b.selectable_books)
    v = book_cookie_value({"Bally Bet", "DraftKings"})
    assert v == "ballybet.draftkings" and choose_books(v.split("."), b) == {"Bally Bet", "DraftKings"}


def _write_shortlist_ledger(cfg):
    from cfbr.bets.ledger import LEDGER_SCHEMA

    first, later = "2026-10-11T14:05:00+00:00", "2026-10-11T20:05:00+00:00"
    rows = [
        _bet(1, "home", -3.5) | {"fbs_game": True, "placed_at": first,
                                 "books": "DraftKings -3.5 (-110/-110); BetMGM -3 (-115/-105)"},
        _bet(2, "away", 4.0) | {"fbs_game": True, "placed_at": first, "books": "BetMGM +3.5 (-110/-110)"},
        _bet(3, "home", -3.0) | {"fbs_game": True, "placed_at": later, "books": "DraftKings -3 (-110/-110)"},
        _bet(5, "home", -7.0, "won", 31, 14, -8.0, week=6) | {"fbs_game": True, "placed_at": "2026-10-04T14:00:00+00:00",
                                                              "books": "FanDuel -7 (-110/-110)"},
    ]
    d = cfg.paths.out_dir / str(SEASON)
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=LEDGER_SCHEMA).write_csv(d / "bets.csv")


@pytest.fixture
def shortlist_client(web_cfg):
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_shortlist_ledger(web_cfg)
    return TestClient(create_app(web_cfg), follow_redirects=False)


def test_shortlist_defaults_to_the_reference_books(shortlist_client):
    r = shortlist_client.get("/bets/shortlist")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    html = " ".join(r.text.split())
    assert "Shortlist" in html and "Texas A&amp;M" in html       # game 1 (home, DK at the line)
    assert "−3.5 or better at DraftKings" in html
    assert "Hawai" not in html.split('id="settled"')[0]          # game 2: only BetMGM has its number
    body = shortlist_client.get("/api/bets/shortlist").json()
    assert [p["game_id"] for p in body["picks"]] == [1, 5] and body["books"] == ["DraftKings", "FanDuel", "bet365"]
    assert body["summary"]["won"] == 1


def test_shortlist_follows_the_viewers_books(shortlist_client):
    c = shortlist_client
    r = c.post("/bets/shortlist/books", content="books=BetMGM&books=Bovada",
               headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 303 and r.headers["location"] == "/bets/shortlist"
    # The season being viewed survives a save; raw input never reaches the Location header.
    r2 = c.post("/bets/shortlist/books", content="books=BetMGM&season=2026",
                headers={"content-type": "application/x-www-form-urlencoded"})
    assert r2.headers["location"] == "/bets/shortlist?season=2026"
    r3 = c.post("/bets/shortlist/books", content="books=BetMGM&season=2026%0d%0aX:1",
                headers={"content-type": "application/x-www-form-urlencoded"})
    assert r3.headers["location"] == "/bets/shortlist"
    assert 'name="season" value="2026"' in c.get("/bets/shortlist?season=2026").text
    assert "cfbr_books=betmgm" in r.headers["set-cookie"] and "HttpOnly" in r.headers["set-cookie"]
    body = c.get("/api/bets/shortlist").json()
    assert body["books"] == ["BetMGM"] and [p["game_id"] for p in body["picks"]] == [1, 2]
    assert [p["bet_at_book"] for p in body["picks"]] == ["BetMGM", "BetMGM"]
    # An explicit ?books= wins over the cookie, even when empty (the reference set); junk falls back to the reference set.
    assert c.get("/api/bets/shortlist?books=FanDuel").json()["books"] == ["FanDuel"]
    assert c.get("/api/bets/shortlist?books=").json()["books"] == ["DraftKings", "FanDuel", "bet365"]
    c.cookies.set("cfbr_books", "nope.bovada")
    assert c.get("/api/bets/shortlist").json()["books"] == ["DraftKings", "FanDuel", "bet365"]
    # Submitting no books clears the choice.
    r = c.post("/bets/shortlist/books", content="", headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 303 and ("Max-Age=0" in r.headers["set-cookie"] or 'cfbr_books=""' in r.headers["set-cookie"])


def test_shortlist_is_gated_and_linked(web_cfg):
    from cfbr.web.access import bets_access

    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_shortlist_ledger(web_cfg)
    app = create_app(web_cfg)
    c = TestClient(app)
    assert 'href="/bets/shortlist?season=2026"' in c.get("/bets").text
    app.dependency_overrides[bets_access] = lambda: (_ for _ in ()).throw(HTTPException(402, "no"))
    assert c.get("/bets/shortlist").status_code == 402
    assert c.get("/api/bets/shortlist").status_code == 402
    assert c.post("/bets/shortlist/books", content="books=FanDuel").status_code == 402


def _shortlist_html(web_cfg, spreads, books=None):
    d = _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True, spreads=spreads)
    if books is not None:
        from cfbr.web.lines import BOOK_LINES_SCHEMA

        pl.DataFrame([{"game_id": g, "book": b, "home_spread": h, "home_price": hp, "away_price": ap,
                       "as_of": "2026-10-14T14:00:00+00:00"} for g, b, h, hp, ap in books],
                     schema=BOOK_LINES_SCHEMA).write_parquet(d / "book_lines.parquet")
        (d / "meta.json").write_text((d / "meta.json").read_text())  # meta last, as the build writes it
    _write_shortlist_ledger(web_cfg)
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    c.cookies.set("cfbr_books", "betmgm")
    return " ".join(c.get("/bets/shortlist").text.split())


def test_keys_crossed():
    from cfbr.web.bets import keys_crossed

    assert keys_crossed(-11.5, -14.5) == [14]     # favorite: a 14-point win no longer covers
    assert keys_crossed(8.5, 4.5) == [7]          # underdog: a 7-point loss no longer covers
    assert keys_crossed(-11.5, -12.5) == []
    assert keys_crossed(-2.5, -3.5) == [3] and keys_crossed(3.5, 2.5) == [3]
    assert keys_crossed(-3.0, -3.5) == [3]        # a push becomes a loss
    assert keys_crossed(-6.5, -10.5) == [7, 10]
    assert keys_crossed(-7.0, -7.0) == []


def test_shortlist_shows_how_the_picks_book_moved(web_cfg):
    # Game 1: home bet, paper line −3.5, BetMGM −3 at capture. Game 2: away bet, paper line −4,
    # BetMGM −3.5 for the bettor at capture. The model has each bet team by 10.

    # Without per-book lines (older web data): the consensus against the paper line.
    html = _shortlist_html(web_cfg, {1: -3.5, 2: 4.0})
    assert "Texas A&amp;M" in html and "or better at BetMGM" in html
    assert "Consensus now −3.5 (BetMGM not in the newest odds): unchanged" in html
    assert html.count("unchanged") == 2 and "edge now 6.5" in html and "would pass" not in html

    # BetMGM moved game 1 against the bettor (across 3) and game 2 toward them (also across 3).
    html = _shortlist_html(web_cfg, {1: -5.0, 2: 3.0},
                           [(1, "BetMGM", -4.5, -105, -115), (2, "BetMGM", 2.5, -110, -110), (2, "DraftKings", 2.0, -110, -110)])
    assert 'class="down">BetMGM now −4.5 (−105): 1.5 pts worse, crossed 3</span> <span class="muted">· edge now 5.5</span>' in html
    assert 'class="up">BetMGM now −2.5 (−110): 1 pt better, crossed 3</span> <span class="muted">· edge now 7.5</span>' in html
    assert "Consensus now" not in html

    # Far enough that the model would no longer bet it.
    html = _shortlist_html(web_cfg, {1: -7.5, 2: 4.0}, [(1, "BetMGM", -7.5, -110, -110)])
    assert "BetMGM now −7.5 (−110): 4.5 pts worse, crossed 3, 7" in html
    assert "edge now 2.5, under 3: the model would pass" in html
    # Game 2's book has no line now: it falls back to the consensus.
    assert "Consensus now −4 (BetMGM not in the newest odds): unchanged" in html


def test_shortlist_with_no_picks_at_the_viewers_books(shortlist_client):
    shortlist_client.cookies.set("cfbr_books", "betrivers")
    r = shortlist_client.get("/bets/shortlist")
    assert r.status_code == 200
    assert "No open picks at your books" in " ".join(r.text.split())
    body = shortlist_client.get("/api/bets/shortlist?books=BetRivers").json()
    assert body["picks"] == []
    assert body["summary"]["settled"] == 0 and body["summary"]["won"] == 0


@pytest.mark.parametrize("content", ["[]", '{"seasons": 1}', "{not json"])
def test_shortlist_survives_a_malformed_pricing_table(web_cfg, content):
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_shortlist_ledger(web_cfg)
    (web_cfg.paths.out_dir / "pricing_table.json").write_text(content)
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    assert c.get("/bets/shortlist").status_code == 200


def test_shortlist_ignores_a_pricing_table_that_includes_the_season(shortlist_client, web_cfg, monkeypatch):
    from cfbr.bets.pricing import PricingTable
    from cfbr.web import bets as web_bets

    pmf = {m: {3: 0.2, 7: 0.1, -3: 0.1, 10: 0.6} for m in range(36)}
    seen = []
    real = web_bets.shortlist
    monkeypatch.setattr(web_bets, "shortlist", lambda ledger, books, cfg, table=None: seen.append(table) or real(ledger, books, cfg, table))
    for seasons, expect_table in (([2025, SEASON], False), ([2024, 2025], True), ([], True)):
        PricingTable(seasons=seasons, n_games=1000, window=2.0, min_games=300, pmf=pmf).save(
            web_cfg.paths.out_dir / "pricing_table.json")
        seen.clear()
        r = shortlist_client.get("/api/bets/shortlist?books=DraftKings,BetMGM")
        assert r.status_code == 200 and shortlist_client.get("/bets/shortlist").status_code == 200
        assert [t is not None for t in seen] == [expect_table, expect_table], seasons
    # Without a usable table the book is picked by points: BetMGM -3 beats DraftKings -3.5 for the home side.
    PricingTable(seasons=[2025, SEASON], n_games=1000, window=2.0, min_games=300, pmf=pmf).save(
        web_cfg.paths.out_dir / "pricing_table.json")
    picks = shortlist_client.get("/api/bets/shortlist?books=DraftKings,BetMGM").json()["picks"]
    assert [p["bet_at_book"] for p in picks][0] == "BetMGM"


def test_shortlist_book_picker_summarizes_the_choice_and_offers_the_default(shortlist_client):
    c = shortlist_client
    html = " ".join(c.get("/bets/shortlist").text.split())
    assert "Your books: <strong>DraftKings, FanDuel, bet365</strong>" in html and "(default)" in html
    assert 'form="books-default"' not in html                  # already on the default: no reset offered
    c.post("/bets/shortlist/books", content="books=BetMGM&season=2026",
           headers={"content-type": "application/x-www-form-urlencoded"})
    html = " ".join(c.get("/bets/shortlist?season=2026").text.split())
    assert "Your books: <strong>BetMGM</strong>" in html and "(default)" not in html
    assert 'id="books-default"' in html and 'form="books-default"' in html
    # The reset form posts no books (clearing the cookie) and keeps the season.
    reset = html.split('id="books-default"')[1].split("</form>")[0]
    assert 'name="books"' not in reset and 'name="season" value="2026"' in reset


def test_shortlist_book_picker_names_the_empty_reference_set(shortlist_client, web_cfg):
    # pricing_books = [] means any non-offshore book (config.py); the picker must say so.
    cfg = web_cfg.model_copy(update={"bets": web_cfg.bets.model_copy(update={"pricing_books": []})})
    c = TestClient(create_app(cfg))
    html = " ".join(c.get("/bets/shortlist").text.split())
    assert "Your books: <strong>any non-offshore book</strong>" in html and "(default)" in html
    c.cookies.set("cfbr_books", "betmgm")
    html = " ".join(c.get("/bets/shortlist").text.split())
    assert "Use default (any non-offshore book)" in html


def test_shortlist_names_the_open_week_and_how_old_line_now_is(shortlist_client, web_cfg):
    from cfbr.web.app import iso_ts

    html = " ".join(shortlist_client.get("/bets/shortlist").text.split())
    assert "<h2>Week 7 picks</h2>" in html and "<h2>Open picks</h2>" not in html
    assert "<strong>Line now</strong> is the consensus on Games." in html   # no lines_as_of in meta yet
    d = web_dir(web_cfg) / str(SEASON)
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps(meta | {"lines_as_of": "2026-10-14T21:00:00+00:00"}))
    html = " ".join(shortlist_client.get("/bets/shortlist").text.split())
    assert f"is the consensus on Games, as of {iso_ts('2026-10-14T21:00:00+00:00')}." in html
    # No open picks at the viewer's books: the heading falls back and the note goes with the table.
    shortlist_client.cookies.set("cfbr_books", "betrivers")
    html = " ".join(shortlist_client.get("/bets/shortlist").text.split())
    assert "<h2>Open picks</h2>" in html and "consensus on Games" not in html


def test_line_now_stars_a_line_older_than_the_page(shortlist_client, web_cfg):
    # Games keeps CFBD's line when the newest odds snapshot has none for a game; /bets and the
    # shortlist must not present that line as current "as of" the newer snapshot.
    from cfbr.web.app import iso_ts

    d = web_dir(web_cfg) / str(SEASON)
    games = pl.read_parquet(d / "games.parquet")
    new, old = "2026-10-14T21:00:00+00:00", "2026-10-11T20:00:11+00:00"
    games.with_columns(line_source=pl.when(pl.col("game_id") == 1).then(pl.lit("cfbd")).otherwise(pl.lit("odds")),
                       line_as_of=pl.when(pl.col("game_id") == 1).then(pl.lit(old)).otherwise(pl.lit(new))
                       ).write_parquet(d / "games.parquet")
    meta = json.loads((d / "meta.json").read_text())
    (d / "meta.json").write_text(json.dumps(meta | {"lines_as_of": new}))
    star = f'<sup class="muted" title="Older line: as of {iso_ts(old)}">*</sup>'
    for url in ("/bets/shortlist", "/bets"):
        html = " ".join(shortlist_client.get(url).text.split())
        assert html.count(star) == 1, url                       # game 1 only; the odds-sourced lines aren't starred
        assert "Lines marked * are older" in html, url
    # The same line from the newest snapshot gets no star and no caveat (rewriting meta.json
    # last is what tells the store to reload, as web build and refresh_lines do).
    games.with_columns(line_source=pl.lit("odds"), line_as_of=pl.lit(new)).write_parquet(d / "games.parquet")
    (d / "meta.json").write_text(json.dumps(meta | {"lines_as_of": new, "rebuilt": 1}))
    html = " ".join(shortlist_client.get("/bets/shortlist").text.split())
    assert "Older line" not in html and "Lines marked * are older" not in html


def test_owner_records_a_real_bet_from_the_site(web_cfg, monkeypatch):
    from cfbr.bets.real import read_real
    from cfbr.web.access import OWNER_COOKIE, owner_token

    monkeypatch.setenv("OWNER_KEY", "s3cret-key")
    now = datetime.now(UTC)
    _write_season(web_dir(web_cfg), now, polls=True)
    _write_ledger(web_cfg)
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    before = c.get("/api/bets").json()
    hdr = {"content-type": "application/x-www-form-urlencoded"}
    path = web_cfg.paths.out_dir / str(SEASON) / "bets_real.csv"

    # Not the owner: no form, no "Record" links, and the route doesn't exist.
    html = c.get("/bets").text
    assert 'id="record"' not in html and "record-link" not in html
    r = c.post("/bets/real", content=f"season={SEASON}&game=3&side=away&line=3&odds=-110&stake=10", headers=hdr)
    assert r.status_code == 404 and not path.exists()

    c.cookies.set(OWNER_COOKIE, owner_token("s3cret-key"))
    # The owner gets the form even with no real bets yet: upcoming games only (game 3 kicks off
    # in two days; games 1 and 2 were weeks ago), one line per game, away @ home, with the
    # consensus as a hint. The team is a separate choice, named once a game is picked.
    html = " ".join(c.get("/bets").text.split())
    assert 'id="record"' in html and 'action="/bets/real"' in html and "No real bets recorded yet" in html
    assert ('<option value="3" data-away="Texas A&amp;M" data-home="Ohio State">Texas A&amp;M @ Ohio State'
            in html and "· Ohio State −3" in html)
    assert '<option value="1"' not in html and html.count('name="side"') == 2
    assert '<span data-side="away">Away team</span>' in html and '<span data-side="home">Home team</span>' in html
    # An open paper bet links to the form with its side, best line, price and book filled in.
    assert 'href="/bets?season=2026&amp;game=1&amp;side=home&amp;line=-3&amp;odds=-105&amp;book=TestBook#record"' in html
    pre = " ".join(c.get("/bets?season=2026&game=3&side=away&line=3.5&odds=-105&book=FanDuel#record").text.split())
    assert 'data-home="Ohio State" selected>' in pre and 'name="line" inputmode="decimal" placeholder="+3.5" value="3.5"' in pre
    # A prefilled game names its teams on the buttons, with the link's side chosen.
    assert 'value="away" required checked> <span data-side="away">Texas A&amp;M</span>' in pre
    assert '<span data-side="home">Ohio State</span>' in pre
    assert 'name="odds" inputmode="numeric" required placeholder="-110 or +158" value="-105"' in pre and 'value="FanDuel"' in pre

    # Recording: through the ledger (home-perspective line, dollars risked), then back to the section.
    r = c.post("/bets/real", content=f"season={SEASON}&game=3&side=away&line=%2B3.5&odds=-105&stake=25&book=FanDuel", headers=hdr)
    assert r.status_code == 303 and r.headers["location"] == f"/bets?season={SEASON}&saved=1#mine"
    led = read_real(path)
    assert led.height == 1
    row = led.to_dicts()[0]
    assert row["game_id"] == 3 and row["side"] == "away" and row["team"] == "Texas A&M" and row["line"] == -3.5
    assert row["odds"] == -105 and row["risk_dollars"] == 25.0 and row["book"] == "FanDuel" and row["status"] == "open"
    assert row["week"] == 2 and abs(datetime.fromisoformat(row["placed_at"]) - now) < timedelta(minutes=5)
    html = " ".join(c.get(f"/bets?season={SEASON}&saved=1").text.split())
    # The confirmation names the bet, so a wrong team is caught right away.
    assert "Recorded: Texas A&amp;M +3.5 −105." in html and "$25 / $23" in html

    # Refusals come back on the page with what was typed, and write nothing.
    for body, msg in [(f"season={SEASON}&game=3&side=away&line=3.5&odds=-105&stake=25&book=FanDuel", "recorded a moment ago"),
                      (f"season={SEASON}&game=&line=3&odds=-110&stake=10", "pick a game"),
                      (f"season={SEASON}&game=3&line=3&odds=-110&stake=10", "pick the team you bet on"),
                      (f"season={SEASON}&game=3&side=home&line=abc&odds=-110&stake=10", "line must be a number"),
                      (f"season={SEASON}&game=3&side=home&line=-3&odds=-50&stake=10", "not an American price"),
                      (f"season={SEASON}&game=3&side=home&line=-3&odds=-110&stake=0", "stake must be a positive"),
                      (f"season={SEASON}&game=3&side=home&line=-3&odds=-110&stake=10&placed=2099-01-01T00:00", "in the future"),
                      (f"season={SEASON}&game=999&side=home&line=-3&odds=-110&stake=10", "no game 999")]:
        r = c.post("/bets/real", content=body, headers=hdr)
        assert r.status_code == 422 and msg in r.text, body
        # What was typed comes back: the game and, when given, the side.
        assert "Not recorded:" in r.text, body
        if "game=3&" in body:
            assert 'data-home="Ohio State" selected>' in r.text, body
        if "side=home" in body:
            assert 'value="home" required checked>' in r.text, body
    assert read_real(path).height == 1
    # A placed time is read as Central and must be before kickoff.
    local = (now - timedelta(hours=1)).astimezone(ZoneInfo("America/Chicago")).strftime("%Y-%m-%dT%H:%M")
    r = c.post("/bets/real", content=f"season={SEASON}&game=3&side=home&line=-3&odds=-110&stake=10&placed={local}", headers=hdr)
    assert r.status_code == 303
    rows = read_real(path).sort("placed_at").to_dicts()
    assert len(rows) == 2 and abs(datetime.fromisoformat(rows[0]["placed_at"]) - (now - timedelta(hours=1))) < timedelta(minutes=2)

    # The same terms at another book is a separate wager, not a double-submit.
    r = c.post("/bets/real", content=f"season={SEASON}&game=3&side=away&line=3.5&odds=-105&stake=25&book=DraftKings", headers=hdr)
    assert r.status_code == 303 and read_real(path).height == 3
    # Malformed numbers are errors, never reinterpreted ("3+5" is not 35, "10+0" is not $100).
    for body in (f"season={SEASON}&game=3&side=home&line=3%2B5&odds=-110&stake=10",
                 f"season={SEASON}&game=3&side=home&line=-3&odds=-110&stake=10%2B0"):
        assert c.post("/bets/real", content=body, headers=hdr).status_code == 422, body
    assert read_real(path).height == 3
    # A Record link for a game already final still opens the form on that game (a backdated
    # bet with a pre-kickoff placed time); without a placed time it is refused at kickoff.
    pre = c.get(f"/bets?season={SEASON}&game=1&side=home&line=-3&odds=-105&book=TestBook").text
    assert '<option value="1" data-away="&lt;b&gt;Evil&lt;/b&gt;" data-home="Texas A&amp;M" selected>' in pre
    assert 'value="home" required checked>' in pre
    r = c.post("/bets/real", content=f"season={SEASON}&game=1&side=home&line=-3&odds=-105&stake=10", headers=hdr)
    assert r.status_code == 422 and "after kickoff" in r.text
    # A typed placed time survives a validation error, so fixing another field doesn't turn it into "now".
    r = c.post("/bets/real", content=f"season={SEASON}&game=3&side=home&line=abc&odds=-110&stake=10&placed={local}", headers=hdr)
    assert r.status_code == 422 and f'name="placed" value="{local}"' in r.text

    # The paper record and API are untouched.
    assert c.get("/api/bets").json()["summary"] == before["summary"]
    assert '"risk_dollars"' not in c.get("/api/bets").text


def test_owner_records_a_live_moneyline_on_a_game_that_kicked_off(web_cfg, monkeypatch):
    from cfbr.bets.real import read_real
    from cfbr.web.access import OWNER_COOKIE, owner_token

    monkeypatch.setenv("OWNER_KEY", "s3cret-key")
    # Games 1 and 2 kicked off two hours ago (game 1 already final); game 3 is two days out.
    now = datetime.now(UTC)
    _write_season(web_dir(web_cfg), now, polls=True)
    d = web_dir(web_cfg) / str(SEASON)
    games = pl.read_parquet(d / "games.parquet")
    games.with_columns(kickoff_ts=pl.when(pl.col("game_id").is_in([1, 2])).then(pl.lit(now - timedelta(hours=2)))
                       .otherwise(pl.col("kickoff_ts")).cast(games.schema["kickoff_ts"])).write_parquet(d / "games.parquet")
    (d / "meta.json").write_text((d / "meta.json").read_text())  # meta last, or the store sees a rebuild in progress
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    c.cookies.set(OWNER_COOKIE, owner_token("s3cret-key"))
    hdr = {"content-type": "application/x-www-form-urlencoded"}
    path = web_cfg.paths.out_dir / str(SEASON) / "bets_real.csv"

    # Started games are on the list for a live bet, marked as such; the form has the new fields.
    html = " ".join(c.get("/bets").text.split())
    assert 'data-home="Hawai&#39;i">Ohio State @ Hawai&#39;i' in html and "· started" in html
    assert '<option value="1"' in html  # final, but kicked off within the last 3 days
    assert 'name="market"' in html and 'value="moneyline"' in html and 'name="live"' in html

    # A live moneyline: no line needed, placed time defaults to now (after kickoff).
    r = c.post("/bets/real", content=f"season={SEASON}&game=2&side=away&market=moneyline&live=on&odds=%2B158&stake=10&book=DraftKings",
               headers=hdr)
    assert r.status_code == 303, r.text
    row = read_real(path).to_dicts()[0]
    assert row["market"] == "moneyline" and row["live"] is True and row["line"] is None
    assert row["team"] == "Ohio State" and row["odds"] == 158 and row["status"] == "open"
    html = " ".join(c.get(f"/bets?season={SEASON}").text.split())
    assert 'Ohio State ML <span class="badge">live</span>' in html and "$10 / $15.80" in html
    assert "Recorded: Ohio State ML +158 (live)." in " ".join(c.get(f"/bets?season={SEASON}&saved=1").text.split())

    # Refused, and nothing written: a double-submit; a bet after kickoff not marked live; a live
    # bet on a game not yet started; an unknown bet type; a spread with no line.
    local = (now - timedelta(minutes=30)).astimezone(ZoneInfo("America/Chicago")).strftime("%Y-%m-%dT%H:%M")
    for body, msg in [(f"season={SEASON}&game=2&side=away&market=moneyline&live=on&odds=158&stake=10&book=DraftKings", "recorded a moment ago"),
                      (f"season={SEASON}&game=2&side=away&market=moneyline&odds=158&stake=10", "mark it live"),
                      (f"season={SEASON}&game=3&side=away&market=moneyline&live=on&odds=158&stake=10", "so not live"),
                      (f"season={SEASON}&game=2&side=away&market=total&odds=158&stake=10", "pick a bet type"),
                      (f"season={SEASON}&game=2&side=away&market=spread&live=on&line=&odds=-110&stake=10&placed={local}",
                       "line must be a number")]:
        r = c.post("/bets/real", content=body, headers=hdr)
        assert r.status_code == 422 and msg in r.text, body
    assert read_real(path).height == 1
    # An error keeps the bet type and live box as typed.
    r = c.post("/bets/real", content=f"season={SEASON}&game=3&side=away&market=moneyline&live=on&odds=158&stake=10", headers=hdr)
    assert '<option value="moneyline" selected>' in r.text and 'name="live" checked' in r.text

    # A live spread with a typed in-game time records too.
    r = c.post("/bets/real", content=f"season={SEASON}&game=2&side=home&market=spread&live=on&line=%2B6.5&odds=-115&stake=10&placed={local}",
               headers=hdr)
    assert r.status_code == 303
    assert read_real(path).filter(pl.col("market") == "spread")["line"].to_list() == [6.5]


def test_owner_voids_a_mistaken_real_bet(web_cfg, monkeypatch):
    from cfbr.bets.real import read_real
    from cfbr.web.access import OWNER_COOKIE, owner_token

    monkeypatch.setenv("OWNER_KEY", "s3cret-key")
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_real(web_cfg)  # one open bet: Ohio State +1.5 on game 1
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    hdr = {"content-type": "application/x-www-form-urlencoded"}
    path = web_cfg.paths.out_dir / str(SEASON) / "bets_real.csv"
    row = read_real(path).to_dicts()[0]
    body = f"season={SEASON}&game_id={row['game_id']}&recorded_at={quote(row['recorded_at'], safe='')}"

    # Not the owner: no button, and the route doesn't exist.
    assert 'action="/bets/real/void"' not in c.get("/bets").text
    assert c.post("/bets/real/void", content=body, headers=hdr).status_code == 404
    assert read_real(path)["status"][0] == "open"

    c.cookies.set(OWNER_COOKIE, owner_token("s3cret-key"))
    assert 'action="/bets/real/void"' in c.get("/bets").text
    r = c.post("/bets/real/void", content=body, headers=hdr)
    assert r.status_code == 303 and r.headers["location"] == f"/bets?season={SEASON}&voided=1#mine"
    led = read_real(path)
    # Kept in the file, marked void and when; out of the record; no button on it any more.
    assert led.height == 1 and led["status"][0] == "void" and led["voided_at"][0] is not None
    html = " ".join(c.get(f"/bets?season={SEASON}&voided=1").text.split())
    assert "Voided." in html and '<tr class="voided">' in html and 'action="/bets/real/void"' not in html
    assert "0–0–0" in html
    # Only an open bet can be voided, once; an unknown bet is refused too.
    assert c.post("/bets/real/void", content=body, headers=hdr).status_code == 409
    assert c.post("/bets/real/void", content=f"season={SEASON}&game_id=1&recorded_at=nope",
                  headers=hdr).status_code == 409
    assert '"risk_dollars"' not in c.get("/api/bets").text


def test_owner_reaches_a_season_with_no_bets_yet(web_cfg, monkeypatch):
    from cfbr.web.access import OWNER_COOKIE, owner_token

    monkeypatch.setenv("OWNER_KEY", "s3cret-key")
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)  # no bets.csv, no bets_real.csv
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    assert c.get(f"/bets?season={SEASON}").status_code == 404
    c.cookies.set(OWNER_COOKIE, owner_token("s3cret-key"))
    html = c.get("/bets").text
    assert 'id="record"' in html and '<option value="3"' in html
    assert c.get(f"/bets?season={SEASON}").status_code == 200
    assert c.get("/api/bets").status_code == 404


def test_voided_real_bets_are_hidden_behind_a_toggle(web_cfg, monkeypatch):
    """A voided row is a wrong-team entry recorded again a moment later, not a wager: it stays
    in the ledger and the markup, but the table hides it until "Show N voided entries"."""
    from cfbr.bets.real import REAL_SCHEMA, read_real, write_real
    from cfbr.web.access import OWNER_COOKIE, owner_token

    monkeypatch.setenv("OWNER_KEY", "s3cret-key")
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_real(web_cfg)
    path = web_cfg.paths.out_dir / str(SEASON) / "bets_real.csv"
    c = TestClient(create_app(web_cfg), follow_redirects=False)
    c.cookies.set(OWNER_COOKIE, owner_token("s3cret-key"))

    # No voids: no toggle.
    html = " ".join(c.get(f"/bets?season={SEASON}").text.split())
    assert '<table class="data" id="real-bets">' in html and "voided entr" not in html

    def void(team, recorded_at):
        mistake = read_real(path).to_dicts()[0] | {"team": team, "side": "home", "recorded_at": recorded_at,
                                                    "status": "void", "voided_at": "2026-09-01T14:01:00+00:00"}
        write_real(pl.concat([read_real(path), pl.DataFrame([mistake], schema=REAL_SCHEMA)]), path)

    void("Texas A&M", "2026-09-01T13:59:00+00:00")
    html = " ".join(c.get(f"/bets?season={SEASON}").text.split())
    assert '<tr class="voided">' in html  # still in the markup, for the toggle to reveal
    assert ('<button type="button" class="link" data-toggle-class="show-void" data-target="real-bets" '
            'data-label-on="Hide voided entries" data-label-off="Show 1 voided entry" aria-pressed="false">'
            'Show 1 voided entry</button>') in html
    void("Texas A&M", "2026-09-01T13:58:00+00:00")
    html = " ".join(c.get(f"/bets?season={SEASON}").text.split())
    assert html.count('<tr class="voided">') == 2 and 'data-label-off="Show 2 voided entries"' in html
    # The stylesheet does the hiding; the button only flips the class.
    import cfbr.web
    from pathlib import Path

    css = (Path(cfbr.web.__file__).parent / "static" / "style.css").read_text()
    assert "#real-bets:not(.show-void) tr.voided { display: none; }" in css
