"""Web app: data build agrees with `cfbr rank`; server pages, API, escaping, reload."""

import json
import math
import os
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from fastapi.testclient import TestClient

from cfbr.config import load_config
from cfbr.web.app import create_app, default_slot, game_row, poll_note
from cfbr.web.build import FINAL, build_season, timeline_polls, web_dir
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


AP = {"Texas A&M": 3, "Ohio State": 11}


def _write_season(root, now, live=True, polls=False):
    """Synthetic season; ``polls=False`` mimics web data built before poll columns existed."""
    d = root / str(SEASON)
    d.mkdir(parents=True, exist_ok=True)
    ratings, games = _ratings(), _games(now)
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
    assert "2026 ratings · Week 1 · Top 25" in html
    assert client.get("/api/rankings?season=2026&week=1&conf=top25").status_code == 200


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
    # Ohio State won 24-17 at Hawai'i; the home-side prediction of -4 means Ohio State by 4.
    assert "<span class=\"up\">W</span> 24–17" in " ".join(html.split())
    assert "+4.0" in html


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
    assert "AP: week 2 poll · Coaches: week 1 poll (week 2 not released yet)" in html
    assert 'data-v="999"' in html  # unranked teams sort last


def test_team_page_poll_tiles_and_opponent_rank(poll_client):
    html = " ".join(poll_client.get("/team/2026/Ohio%20State").text.split())
    assert "AP poll</div><div class=\"tile-value\">#11" in html
    assert "Coaches poll</div><div class=\"tile-value\">#12" in html
    assert '<span class="pr">#3</span><a href="/team/2026/Texas%20A%26M">' in html  # upcoming opponent
    html = " ".join(poll_client.get("/team/2026/Hawai%27i").text.split())
    assert "AP poll</div><div class=\"tile-value\">Unranked" in html


def test_games_show_ap_rank_before_names(poll_client):
    html = " ".join(poll_client.get("/games?season=2026&slot=r01").text.split())
    assert '<span class="pr">#3</span><a href="/team/2026/Texas%20A%26M">' in html


def test_pages_render_without_poll_columns(client):
    # The default fixture predates polls; the store fills the columns with nulls.
    html = client.get("/rankings").text
    assert ">AP</th>" in html and "Polls for comparison" not in html
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


def test_bets_page(bets_client):
    r = bets_client.get("/bets")
    assert r.status_code == 200
    assert r.headers["cache-control"] == "private, no-store"
    html = " ".join(r.text.split())
    assert "2026 paper bets" in html and "not betting advice" in html
    assert "No conclusion yet." in html and "3 of the 500 settled bets" in html
    assert "1–1–1" in html  # won-lost-push
    assert "All bets, best line" in html
    # On phones the Result column is hidden; the amount carries the result as a mark, not by color alone.
    assert '<span class="m-only">✓ </span>' in html and '<span class="m-only">✗ </span>' in html
    # The best-line book's own price shows when the feed has one, and only then.
    assert "−3 (−105) <span class=\"muted\">TestBook</span>" in html
    assert "−4.5 <span class=\"muted\">TestBook</span>" in html
    # Rules describe the current pricing rule: offshore books never set the best line.
    assert "Bovada" in html and "never sets the best line" in html
    assert "Only DraftKings, FanDuel, bet365 can set the best line" in html and "bet365 has no feed" in html
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
    assert [t["key"] for t in body["tests"]] == ["core_filter", "second_look"]
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
    assert "Published model (v1 prior)" in html and "Roster-prior shadow model" in html
    # Each model: week 1 went 1-1 (+$10 - $11), week 2 won: bankroll $1,000 -> $999 -> $1,009.
    assert "Week 1" in html and "−$1" in html and "$1,009" in html and "Retro U −7" in html
    assert "3 in common, 3 on the same side" in html
    assert html.count("What if: staking plans") == 2  # one panel per model
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
    for url in ("/rankings", "/games?slot=r01", "/games?slot=r02", "/accuracy", "/team/2026/Ohio%20State",
                "/api/rankings", "/api/games?slot=r02", "/api/team/2026/Ohio%20State", "/api/seasons"):
        r = c.get(url)
        assert r.status_code == 200, url
        assert "TestBook" not in r.text and "paper bet" not in r.text.lower(), url
        assert "RetroBook" not in r.text and "Retro U" not in r.text and "retroactive" not in r.text.lower(), url


# -- free viewers: no live-season model output anywhere -------------------------------

# Strings that only exist if model output for the synthetic live season leaked:
# ratings (+21.0, +16.0 ...), offense/defense, predicted margins and picks.
MODEL_STRINGS = ["+21.0", "+16.0", "+11.0", "+6.0", "+10.0", "+9.0", "by 10.0", "by 4.0", "by 30.0", "Toss-up",
                 "Model err", "Model vs Vegas", "Win %"]


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
    "/games?season=2026&slot=r01", "/games?season=2026&slot=r02",
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
    assert "2025 ratings · Week 1" in html and "+21.0" in html  # finished season is fully free
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
    assert "Model MAE" in html
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
    assert 'Texas A&amp;M <span class="muted">2026 roster</span>' in html
    assert "Sam Starter" in html and "Returner" in html and "No prior stats" in html
    assert 'Transfer from <a href="/team/2025/Hawai%27i">Hawai&#39;i</a>' in html
    assert "&lt;i&gt;Fresh&lt;/i&gt; Man" in html and "<i>Fresh</i>" not in html  # names are escaped
    assert "6-2" in html and ">Sr<" in html and ">Gr<" in html  # height and class labels
    assert "Oscar Buckeye" not in html  # another team's player
    # Departures, with the destination linked when it's an FBS team this season.
    assert "Gone Pro" in html and "not on a roster" in html and '<a href="/team/2026/Ohio%20State">Ohio State</a>' in html
    # Summary: shares as percentages, QB back as Yes, no transfer QB as None.
    assert "62%" in html and "Starting QB back" in html and ">Yes<" in html
    # qb_in_ppa = 0 means no 100+-attempt transfer QB, which is not the same as no transfer QB.
    assert ">None with 100+ att.<" in html and ">None<" not in html
    # ...and the incoming QB below that cutoff is still shown.
    assert "Incoming transfer QB" in html and "from Hawai&#39;i · 75 pass att. · +0.12 PPA/att." in html


def test_team_page_shows_roster_card(roster_client):
    html = " ".join(roster_client.get("/team/2026/Texas%20A%26M").text.split())
    assert '<a href="/roster/2026/Texas%20A%26M">Full roster' in html
    assert "1 returners · 1 transfers in" in html
    # No roster rows for this team -> no card.
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
