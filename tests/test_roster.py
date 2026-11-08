import polars as pl
import pytest

from cfbr.model.prior import fit_prior_models, predict_priors
from cfbr.model.roster import ROSTER_FEATURES, roster_features


def _data(with_defense=True):
    # Season 2025 production, season 2026 rosters.
    ppa = pl.DataFrame({
        "season": [2025] * 5, "player_id": ["qbA", "rbA", "wrA", "qbB", "rbFCS"],
        "team": ["A", "A", "A", "B", "F"], "position": ["QB", "RB", "WR", "QB", "RB"],
        "total_ppa": [80.0, 20.0, 10.0, 60.0, 15.0], "plays": [400.0, 150.0, 50.0, 350.0, 100.0],
        "pass_ppa": [75.0, None, None, 58.0, None], "pass_plays": [380.0, None, None, 330.0, None],
    }).with_columns(avg_ppa=pl.col("total_ppa") / pl.col("plays"), avg_pass_ppa=pl.col("pass_ppa") / pl.col("pass_plays"))
    roster = pl.DataFrame({
        "season": [2026] * 4, "player_id": ["rbA", "wrA", "qbB", "rbFCS"],
        "team": ["A", "B", "A", "A"], "position": ["RB", "WR", "QB", "RB"], "class_year": [3, 4, 4, 2],
    })
    d = pl.DataFrame({
        "season": [2025] * 3, "player_id": ["lbA", "cbA", "lbB"], "team": ["A", "A", "B"],
        "tackles": [100.0, 50.0, 80.0] if with_defense else [0.0, 0.0, 0.0],
        "tfl": [10.0, 2.0, 8.0], "sacks": [5.0, 0.0, 4.0], "pd": [2.0, 10.0, 1.0], "hurries": [3.0, 0.0, 2.0],
        "ints": [1.0, 3.0, 0.0],
    })
    roster = pl.concat([roster, pl.DataFrame({"season": [2026, 2026], "player_id": ["lbA", "lbB"], "team": ["A", "A"],
                                              "position": ["LB", "LB"], "class_year": [3, 4]})])
    status = pl.DataFrame({"season": [2025] * 3, "team": ["A", "B", "F"], "classification": ["fbs", "fbs", "fcs"],
                           "conference": ["X", "X", "Y"]})
    return roster, ppa, d, status


def test_returners_departures_and_transfers():
    roster, ppa, d, status = _data()
    f = {r["team"]: r for r in roster_features(2026, ["A", "B"], roster, ppa, d, status).iter_rows(named=True)}
    a = f["A"]
    assert a["off_ret_share"] == pytest.approx(150 / 600)        # only rbA returned
    assert a["off_out"] == pytest.approx(80 + 10)                 # qbA left, wrA moved to B
    assert a["off_in_fbs"] == pytest.approx(60.0)                 # qbB from FBS team B
    assert a["off_in_fcs"] == pytest.approx(15.0)                 # rbFCS from FCS team F
    assert a["qb_returns"] == 0.0 and a["qb_in_ppa"] == pytest.approx(58.0)
    assert a["def_ret_tackles"] == pytest.approx(100 / 150)       # lbA returned, cbA gone
    assert a["def_ret_havoc"] == pytest.approx((10 + 2 + 1) / (10 + 2 + 1 + 2 + 10 + 3))
    assert a["def_in_tackles"] == pytest.approx(80.0)             # lbB transferred in from B
    assert f["B"]["off_in_fbs"] == pytest.approx(10.0)            # wrA
    assert f["B"]["qb_returns"] == 0.0                            # qbB left B


def test_defense_features_null_without_defensive_stats():
    roster, ppa, d, status = _data(with_defense=False)
    f = roster_features(2026, ["A"], roster, ppa, d, status)
    assert f["def_ret_tackles"].is_null().all() and f["off_ret_share"].is_not_null().all()


def test_roster_prior_falls_back_to_v1_when_features_missing():
    import numpy as np

    rng = np.random.default_rng(0)
    n = 120
    base = {"team": [f"T{i}" for i in range(n)], "new_program": [False] * n,
            "talent": rng.normal(700, 50, n), "recruit4": rng.normal(180, 30, n),
            "percent_ppa": rng.uniform(0.2, 0.8, n), "usage": rng.uniform(0.2, 0.8, n)}
    for c in ("o_ppa", "d_ppa", "o_sr", "d_sr"):
        base[f"prev_{c}"] = rng.normal(0, 0.1, n)
        base[c] = 0.5 * base[f"prev_{c}"] + rng.normal(0, 0.03, n)
    for c in ROSTER_FEATURES:
        base[c] = rng.normal(0, 1, n)
    train = pl.DataFrame(base)
    models = fit_prior_models(train, "roster")
    assert all(len(v) == 2 for v in models.values())          # roster model + v1 fallback
    feats = train.head(4).with_columns(def_ret_tackles=pl.lit(None, pl.Float64))
    out = predict_priors(models, feats)
    assert out.select(["o_ppa", "d_ppa", "o_sr", "d_sr"]).null_count().sum_horizontal().item() == 0


def test_roster_web_tables_status_departures_and_labels():
    from types import SimpleNamespace

    from cfbr.report.roster_web import ROSTER_SUMMARY_LABELS, roster_tables

    roster, ppa, d, status = _data()
    roster = roster.with_columns(name=pl.col("player_id"), jersey=pl.lit(None, pl.Int32), height=pl.lit(74, pl.Int32),
                                 weight=pl.lit(210, pl.Int32), home_city=pl.lit("X"), home_state=pl.lit("MO"))
    ds = SimpleNamespace(roster=roster, player_ppa=ppa, player_defense=d, team_status=status,
                         fbs_teams=lambda s: ["A", "B"])
    t = roster_tables(ds, 2026)
    p = {r["player_id"]: r for r in t["roster_players"].iter_rows(named=True)}
    assert p["rbA"]["status"] == "returner" and p["rbA"]["from_team"] is None
    assert p["qbB"]["status"] == "transfer_in" and p["qbB"]["from_team"] == "B"
    assert p["qbB"]["prev_ppa_avg"] == pytest.approx(60 / 350)
    assert p["qbB"]["prev_pass_ppa_avg"] == pytest.approx(58 / 330)
    assert p["rbFCS"]["from_team"] == "F"
    assert set(t["roster_players"]["team"]) <= {"A", "B"}          # FBS teams only
    dep = {r["player_id"]: r for r in t["roster_departures"].iter_rows(named=True)}
    assert dep["qbA"]["team"] == "A" and dep["qbA"]["to_team"] is None
    assert dep["qbA"]["prev_pass_plays"] > 0  # the web tile names the departed starter from this
    assert dep["wrA"]["to_team"] == "B" and "rbA" not in dep
    assert set(ROSTER_SUMMARY_LABELS) == set(ROSTER_FEATURES)
    assert t["roster_summary"].columns[:3] == ["season", "team", "roster_as_of"]
    empty = SimpleNamespace(roster=roster.head(0), player_ppa=ppa, player_defense=d, team_status=status,
                            fbs_teams=lambda s: [])
    assert roster_tables(empty, 2026) == {}
    # Incoming-QB display fields: qbB (330 attempts at B) is A's incoming transfer QB.
    a = t["roster_summary"].filter(pl.col("team") == "A").row(0, named=True)
    assert a["in_qb_name"] == "qbB" and a["in_qb_from"] == "B" and a["in_qb_pass_att"] == 330


def test_roster_web_status_uses_last_seasons_roster_when_no_stats():
    """Offensive linemen and backups rarely record a stat: last season's roster says
    whether they returned or transferred. Production still wins where it exists."""
    from types import SimpleNamespace

    from cfbr.report.roster_web import roster_tables

    roster, ppa, d, status = _data()
    prev = pl.DataFrame({"season": [2025] * 4, "player_id": ["olA", "olB", "rbA", "qbB"], "team": ["A", "B", "B", "B"],
                         "position": ["OL", "OL", "RB", "QB"], "class_year": [2, 3, 2, 3]})
    cur = pl.DataFrame({"season": [2026] * 3, "player_id": ["olA", "olB", "frA"], "team": ["A", "A", "A"],
                        "position": ["OL", "OL", "OL"], "class_year": [3, 4, 1]})
    roster = (pl.concat([roster, cur, prev], how="diagonal_relaxed")
              .with_columns(name=pl.col("player_id"), jersey=pl.lit(None, pl.Int32), height=pl.lit(None, pl.Int32),
                            weight=pl.lit(None, pl.Int32), home_city=pl.lit(None, pl.Utf8),
                            home_state=pl.lit(None, pl.Utf8)))
    ds = SimpleNamespace(roster=roster, player_ppa=ppa, player_defense=d, team_status=status,
                         fbs_teams=lambda s: ["A", "B"])
    p = {r["player_id"]: r for r in roster_tables(ds, 2026)["roster_players"].iter_rows(named=True)}
    assert p["olA"]["status"] == "returner" and p["olA"]["from_team"] is None
    assert p["olB"]["status"] == "transfer_in" and p["olB"]["from_team"] == "B"
    assert p["frA"]["status"] == "new" and p["frA"]["from_team"] is None
    # rbA is on B's 2025 roster here but produced for A: production wins.
    assert p["rbA"]["status"] == "returner"


def test_roster_web_prefers_latest_snapshot_but_model_keeps_fixed_one():
    from types import SimpleNamespace

    from cfbr.report.roster_web import roster_tables

    roster, ppa, d, status = _data()
    roster = roster.with_columns(name=pl.col("player_id"), jersey=pl.lit(None, pl.Int32), height=pl.lit(None, pl.Int32),
                                 weight=pl.lit(None, pl.Int32), home_city=pl.lit(None, pl.Utf8),
                                 home_state=pl.lit(None, pl.Utf8))
    # Weekly snapshot: rbA has left A since the fixed snapshot.
    latest = roster.filter(pl.col("player_id") != "rbA").with_columns(as_of=pl.lit(None, pl.Datetime("us", "UTC")))
    ds = SimpleNamespace(roster=roster, roster_latest=latest, player_ppa=ppa, player_defense=d, team_status=status,
                         fbs_teams=lambda s: ["A", "B"])
    t = roster_tables(ds, 2026)
    assert "rbA" not in set(t["roster_players"]["player_id"])
    assert "rbA" in set(t["roster_departures"]["player_id"])
    # The model's input (fixed snapshot) is untouched.
    assert "rbA" in set(ds.roster["player_id"])


def test_build_roster_latest_picks_newest_weekly_file(tmp_path):
    import json
    import os

    from cfbr.ingest.store import build_roster, build_roster_latest

    d = tmp_path / "roster" / "2026"
    d.mkdir(parents=True)
    rec = lambda pid: {"id": pid, "firstName": "A", "lastName": pid, "team": "T", "year": 2, "position": "QB"}  # noqa: E731
    (d / "season.json").write_text(json.dumps([rec("1")]))
    (d / "w06.json").write_text(json.dumps([rec("1"), rec("2")]))
    (d / "w07.json").write_text(json.dumps([rec("1"), rec("2"), rec("3")]))
    os.utime(d / "w07.json", (1_800_000_000, 1_800_000_000))
    assert build_roster(tmp_path)["player_id"].to_list() == ["1"]           # model: fixed first snapshot
    latest = build_roster_latest(tmp_path)
    assert sorted(latest["player_id"].to_list()) == ["1", "2", "3"]         # display: newest weekly
    assert latest["as_of"][0].timestamp() == 1_800_000_000


def test_build_roster_latest_repairs_broken_class_years(tmp_path):
    """CFBD's 2026 w08 roster listed every FBS player as a freshman. The display roster keeps
    the newest snapshot's players and other fields, but takes a broken team's class years
    from the newest earlier snapshot where that team's looked right."""
    import json

    from cfbr.ingest.store import BROKEN_CLASS_MIN_KNOWN, build_roster, build_roster_latest

    d = tmp_path / "roster" / "2026"
    d.mkdir(parents=True)
    n = BROKEN_CLASS_MIN_KNOWN
    real = [1, 2, 3, 4] * (n // 4)                          # a normal class spread, n players

    def snap(name, teams, weight=200):
        recs = [{"id": f"{t}{i}", "firstName": "A", "lastName": str(i), "team": t, "year": y, "position": "QB",
                 "weight": weight} for t, years in teams.items() for i, y in enumerate(years)]
        (d / name).write_text(json.dumps(recs))

    def classes(team):
        df = build_roster_latest(tmp_path).filter(pl.col("team") == team)
        return df.sort(pl.col("player_id").str.slice(1).cast(pl.Int32))["class_year"].to_list()
    snap("season.json", {"A": real, "B": real, "S": [2, 3]})
    snap("w06.json", {"A": [*real, 2026], "B": real, "S": [2, 3]})          # 2026 = unknown class (null)
    snap("w07.json", {"A": [1] * (n + 2), "B": real, "S": [1, 1]}, weight=210)  # A broken; A21 new
    assert classes("A") == [*real, None, None]              # from w06; A20 unknown there, A21 new
    assert classes("B") == real                             # healthy team left alone
    assert classes("S") == [1, 1]                           # too few to judge: taken as given
    assert build_roster_latest(tmp_path)["weight"].unique().to_list() == [210]  # other fields from w07
    assert build_roster(tmp_path).filter(pl.col("team") == "A")["class_year"].to_list() == real  # model untouched

    snap("w08.json", {"A": real, "B": real})                # a later good snapshot is used as-is
    assert classes("A") == real


def test_career_history_and_qb_experience():
    from types import SimpleNamespace

    from cfbr.report.roster_web import CAREER_LABELS, roster_tables

    roster, ppa, d, status = _data()
    roster = roster.with_columns(name=pl.col("player_id"), jersey=pl.lit(None, pl.Int32), height=pl.lit(None, pl.Int32),
                                 weight=pl.lit(None, pl.Int32), home_city=pl.lit(None, pl.Utf8),
                                 home_state=pl.lit(None, pl.Utf8))
    ppa = ppa.with_columns(rush_car=pl.lit(10.0), rec=pl.lit(0.0))
    older = ppa.filter(pl.col("player_id") == "qbB").with_columns(season=pl.lit(2024, pl.Int64), pass_plays=pl.lit(200.0))
    ppa = pl.concat([ppa, older.cast(ppa.schema)])

    def gp_row(season, game, team, pid, att):
        return {"season": season, "season_type": "regular", "week": 1, "game_id": game, "team": team,
                "player_id": pid, "name": pid, "pass_cmp": 0.0, "pass_att": float(att), "rush_att": 0.0, "rec": 0.0,
                "tackles": 0.0, "tfl": 0.0, "pd": 0.0, "ints": 0.0}
    gp = pl.DataFrame([gp_row(2025, 1, "B", "qbB", 30), gp_row(2025, 1, "B", "rbB", 2),   # qbB starts
                       gp_row(2025, 2, "B", "qbB", 3), gp_row(2025, 2, "B", "qbB2", 25),  # backup starts
                       gp_row(2026, 3, "A", "qbB", 28)])                                  # this season, at A
    ds = SimpleNamespace(roster=roster, player_ppa=ppa, player_defense=d, team_status=status, game_players=gp,
                         fbs_teams=lambda s: ["A", "B"])
    t = roster_tables(ds, 2026)
    c = t["roster_careers"].filter(pl.col("player_id") == "qbB").sort("career_season")
    assert c["career_season"].to_list() == [2024, 2025, 2026]
    assert c["source"].to_list() == ["season stats", "season stats", "box scores (to date)"]
    assert c["career_team"].to_list() == ["B", "B", "A"] and set(c["team"]) == {"A"}
    assert c.filter(pl.col("career_season") == 2025)["qb_starts"].item() == 1.0        # led B in game 1 only
    assert c.filter(pl.col("career_season") == 2026)["pass_att"].item() == 28.0
    assert t["roster_careers"].columns[4:] == [k for k in CAREER_LABELS if k != "career_season"]
    q = t["roster_players"].filter(pl.col("player_id") == "qbB").row(0, named=True)
    assert q["career_seasons"] == 2 and q["career_pass_att"] == 530.0 and q["career_qb_starts"] == 1.0


def test_departure_reasons_from_draft_and_portal():
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from cfbr.report.roster_web import DEPARTURE_REASONS, roster_tables

    roster, ppa, d, status = _data()
    prev = pl.DataFrame({"season": [2025] * 4, "player_id": ["qbA", "cbA", "olA", "wrA"], "team": ["A"] * 4,
                         "position": ["QB", "CB", "OL", "WR"], "class_year": [3, 4, 4, 2]})
    roster = pl.concat([roster, prev], how="diagonal_relaxed").with_columns(
        name=pl.col("player_id").replace({"cbA": "Cee Bee Jr.", "wrA": "Dub Arr"}),
        jersey=pl.lit(None, pl.Int32), height=pl.lit(None, pl.Int32), weight=pl.lit(None, pl.Int32),
        home_city=pl.lit(None, pl.Utf8), home_state=pl.lit(None, pl.Utf8))
    draft = pl.DataFrame({"season": [2026, 2026, 2025], "player_id": ["qbA", "olA", "lbA"],
                          "college_team": ["A", "A", "A"], "nfl_team": ["Los Angeles", "Chicago", "Dallas"],
                          "nfl_team_name": ["Los Angeles Chargers", "Chicago Bears", "Dallas Cowboys"],
                          "round": [1, 3, 1], "overall": [5, 80, 2]})
    t0 = datetime(2026, 1, 5, tzinfo=UTC)
    portal = pl.DataFrame({"season": [2026] * 3, "name": ["Cee Bee", "Dub Arr", "Nobody Here"],
                           "origin": ["A", "A", "A"], "destination": ["Montana", "B", None],
                           "transfer_date": [t0] * 3})
    ds = SimpleNamespace(roster=roster, player_ppa=ppa, player_defense=d, team_status=status, draft_picks=draft,
                         portal=portal, fbs_teams=lambda s: ["A", "B"])
    dep = {r["player_id"]: r for r in roster_tables(ds, 2026)["roster_departures"].iter_rows(named=True)}
    assert dep["qbA"]["reason"] == "drafted" and dep["qbA"]["draft_overall"] == 5
    assert dep["qbA"]["draft_nfl_team"] == "Los Angeles" and dep["qbA"]["prev_class_year"] == 3
    assert dep["qbA"]["draft_nfl_team_name"] == "Los Angeles Chargers"
    assert dep["cbA"]["draft_nfl_team_name"] is None
    assert dep["olA"]["reason"] == "drafted" and dep["olA"]["team"] == "A"   # no stats, still listed
    assert dep["olA"]["prev_plays"] is None
    assert "lbA" not in dep                                                 # returner; 2025 draft is irrelevant
    assert dep["cbA"]["reason"] == "portal_transfer" and dep["cbA"]["portal_destination"] == "Montana"  # "Jr." ignored
    assert dep["wrA"]["reason"] == "transfer" and dep["wrA"]["in_portal"] is True   # roster beats portal
    assert dep["qbB"]["in_portal"] is False                                 # B had no portal entries
    assert {r["reason"] for r in dep.values()} <= set(DEPARTURE_REASONS)


def test_build_draft_and_portal_tables(tmp_path):
    import json

    from cfbr.ingest.store import build_draft_picks, build_portal

    (tmp_path / "draft_picks" / "2026").mkdir(parents=True)
    (tmp_path / "draft_picks" / "2026" / "season.json").write_text(json.dumps([
        {"collegeAthleteId": 4837248, "collegeTeam": "Indiana", "nflTeam": "Las Vegas", "nflTeamId": 13, "round": 1,
         "pick": 1, "overall": 1, "name": "Fernando Mendoza", "position": "Quarterback"},
        {"collegeAthleteId": None, "collegeTeam": "Nowhere", "nflTeam": "X", "nflTeamId": 99, "round": 7, "pick": 1,
         "overall": 220, "name": "No Id", "position": "Punter"}]))
    (tmp_path / "player_portal" / "2026").mkdir(parents=True)
    (tmp_path / "player_portal" / "2026" / "season.json").write_text(json.dumps([
        {"season": 2026, "firstName": "Tray", "lastName": "Taylor", "position": "WR", "origin": "Troy",
         "destination": None, "transferDate": "2026-01-17T05:39:00.000Z", "eligibility": "Immediate"}]))
    dp = build_draft_picks(tmp_path)
    assert dp["player_id"].to_list() == ["4837248", None] and dp["season"].to_list() == [2026, 2026]
    assert dp["nfl_team_name"].to_list() == ["Las Vegas Raiders", None]   # unknown franchise id -> null
    po = build_portal(tmp_path).row(0, named=True)
    assert po["name"] == "Tray Taylor" and po["destination"] is None and po["transfer_date"].day == 17
    assert build_draft_picks(tmp_path / "empty").height == 0 and build_portal(tmp_path / "empty").height == 0


def test_roster_regression_trains_only_on_seasons_at_or_after_the_cutoff():
    import numpy as np
    import polars as pl
    from sklearn.linear_model import LinearRegression
    from cfbr.model.prior import COMPONENTS, OFFENSE_RETURNING, fit_prior_models
    from cfbr.model.roster import DEF_FEATURES, OFF_FEATURES

    rng = np.random.default_rng(0)
    n = 400
    seasons = np.repeat([2017, 2019, 2021, 2023], n // 4)
    cols = {"season": seasons, "team": [f"T{i}" for i in range(n)], "new_program": [False] * n,
            "talent": rng.normal(size=n), "recruit4": rng.normal(size=n)}
    for c in COMPONENTS:
        cols[f"prev_{c}"] = rng.normal(size=n)
    for f in OFFENSE_RETURNING + OFF_FEATURES + DEF_FEATURES:
        cols[f] = rng.normal(size=n)
    # the target depends on an offense roster feature only in the portal era, so the two fits differ
    cols["o_ppa"] = cols["prev_o_ppa"] + np.where(seasons >= 2021, 2.0, 0.0) * cols[OFF_FEATURES[0]]
    for c in ("d_ppa", "o_sr", "d_sr"):
        cols[c] = cols[f"prev_{c}"]
    train = pl.DataFrame(cols)
    all_rows = fit_prior_models(train, "roster")["o_ppa"][0][1]
    feats, portal = fit_prior_models(train, "roster", roster_min_season=2021)["o_ppa"][0]
    sub = train.filter(pl.col("season") >= 2021)
    manual = LinearRegression().fit(sub.select(feats).to_numpy(), sub["o_ppa"].to_numpy())
    assert np.allclose(portal.coef_, manual.coef_)
    assert not np.allclose(portal.coef_, all_rows.coef_)


def test_roster_as_of_is_utc_whatever_the_session_time_zone(tmp_path):
    """DuckDB returns TIMESTAMPTZ columns in the session's (machine's) time zone; roster_as_of must still
    hand back UTC so roster_tables can stamp roster_as_of as Datetime(UTC) on a non-UTC host."""
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from cfbr.report.roster_web import roster_as_of, roster_tables

    fetched = datetime(2026, 10, 11, 17, 3, 43, tzinfo=UTC)
    latest = pl.DataFrame({"season": [2026], "as_of": [fetched]}).with_columns(
        pl.col("as_of").dt.convert_time_zone("America/Chicago"))
    roster, ppa, d, status = _data()
    roster = roster.with_columns(name=pl.col("player_id"), jersey=pl.lit(None, pl.Int32), height=pl.lit(74, pl.Int32),
                                 weight=pl.lit(210, pl.Int32), home_city=pl.lit("X"), home_state=pl.lit("MO"))
    ds = SimpleNamespace(roster=roster, player_ppa=ppa, player_defense=d, team_status=status, roster_latest=latest,
                         fbs_teams=lambda s: ["A", "B"])
    cfg = SimpleNamespace(paths=SimpleNamespace(raw_dir=tmp_path))
    as_of = roster_as_of(cfg, 2026, ds)
    assert as_of == fetched and as_of.utcoffset().total_seconds() == 0
    summary = roster_tables(ds, 2026, as_of)["roster_summary"]
    assert summary["roster_as_of"].dtype == pl.Datetime("us", "UTC") and summary["roster_as_of"][0] == fetched


def test_build_player_stats_and_box_score_kicking(tmp_path):
    import json

    from cfbr.ingest.store import PLAYER_STATS_COLS, build_game_players, build_player_stats

    (tmp_path / "stats_player_season" / "2025").mkdir(parents=True)
    rows = [("qb", "A", "passing", "YDS", "3169"), ("qb", "A", "passing", "TD", "25"),
            ("qb", "A", "passing", "COMPLETIONS", "234"), ("qb", "A", "rushing", "YDS", "-12"),
            ("k", "A", "kicking", "FGM", "20"), ("k", "A", "kicking", "LONG", "47"),
            ("k", "A", "kicking", "PCT", "87.0"),                      # not kept
            ("lb", "A", "defensive", "SACKS", "1.5"), ("lb", "A", "defensive", "TD", "1"),
            ("lb", "A", "defensive", "TOT", "bad")]                    # unparseable -> skipped
    (tmp_path / "stats_player_season" / "2025" / "season.json").write_text(json.dumps(
        [{"playerId": p, "team": t, "category": c, "statType": s, "stat": v} for p, t, c, s, v in rows]))
    built = build_player_stats(tmp_path)
    assert built.columns == ["season", "player_id", "team", *PLAYER_STATS_COLS]
    st = {r["player_id"]: r for r in built.iter_rows(named=True)}
    assert (st["qb"]["pass_yds"], st["qb"]["pass_td"], st["qb"]["pass_cmp"], st["qb"]["rush_yds"]) == (3169, 25, 234, -12)
    assert st["qb"]["fgm"] is None                                     # no kicking row: unknown, not 0
    assert (st["k"]["fgm"], st["k"]["fg_long"]) == (20, 47)
    assert (st["lb"]["sacks"], st["lb"]["def_td"], st["lb"]["tackles"]) == (1.5, 1, None)
    assert build_player_stats(tmp_path / "empty").height == 0

    (tmp_path / "games_players" / "2026").mkdir(parents=True)

    def ath(pid, stat):
        return {"id": pid, "name": pid, "stat": stat}

    def game(gid, fg, xp, long_, sacks):
        return {"id": gid, "teams": [{"team": "A", "categories": [
            {"name": "kicking", "types": [{"name": "FG", "athletes": [ath("k", fg), ath("-1", "9/9")]},
                                          {"name": "XP", "athletes": [ath("k", xp)]},
                                          {"name": "LONG", "athletes": [ath("k", long_)]}]},
            {"name": "defensive", "types": [{"name": "SACKS", "athletes": [ath("lb", sacks)]}]}]}]}
    (tmp_path / "games_players" / "2026" / "regular_w01.json").write_text(json.dumps([game(1, "2/3", "4/4", "47", "1")]))
    (tmp_path / "games_players" / "2026" / "regular_w02.json").write_text(json.dumps([game(2, "1/1", "3/4", "38", "0.5")]))
    gp = build_game_players(tmp_path)
    k = gp.filter(pl.col("player_id") == "k")
    assert k.select("fgm", "fga", "xpm", "xpa").sum().row(0) == (3, 4, 7, 8)   # "made/att" split per game
    assert sorted(k["fg_long"].to_list()) == [38, 47]                           # one long per game
    assert gp.filter(pl.col("player_id") == "lb")["sacks"].sum() == 1.5
    assert "-1" not in gp["player_id"].to_list()                                # team totals dropped


def test_build_player_ppa_drops_cfbd_duplicate_rows(tmp_path):
    import json

    from cfbr.ingest.store import build_player_ppa

    def ppa(pid, team, conference, total):
        return {"season": 2025, "id": pid, "name": pid, "position": "RB", "team": team, "conference": conference,
                "averagePPA": {"all": total / 100}, "totalPPA": {"all": total}}
    (tmp_path / "ppa_players_season" / "2025").mkdir(parents=True)
    (tmp_path / "ppa_players_season" / "2025" / "season.json").write_text(json.dumps([
        ppa("rb", "Troy", "SBC", 40.0), ppa("rb", "Troy", "", 40.0),     # CFBD repeats a player with conference ""
        ppa("tr", "A", "X", 10.0), ppa("tr", "B", "Y", 5.0)]))         # a mid-season transfer keeps both teams
    built = build_player_ppa(tmp_path)
    assert sorted(built.select("player_id", "team").rows()) == [("rb", "Troy"), ("tr", "A"), ("tr", "B")]
    assert built.filter(pl.col("player_id") == "rb")["total_ppa"].sum() == 40.0


def test_roster_web_box_score_stats():
    from types import SimpleNamespace

    from cfbr.ingest.store import PLAYER_STATS_COLS
    from cfbr.report.roster_web import PREV_STATS, roster_tables

    roster, ppa, d, status = _data()
    cur = pl.DataFrame({"season": [2026] * 2, "player_id": ["kA", "pA"], "team": ["A", "A"], "position": ["PK", "P"],
                        "class_year": [3, 2]})
    roster = (pl.concat([roster, cur], how="diagonal_relaxed")
              .with_columns(name=pl.col("player_id"), jersey=pl.lit(None, pl.Int32), height=pl.lit(None, pl.Int32),
                            weight=pl.lit(None, pl.Int32), home_city=pl.lit(None, pl.Utf8),
                            home_state=pl.lit(None, pl.Utf8)))

    def st_row(pid, team, **vals):
        return {"season": 2025, "player_id": pid, "team": team, **{c: vals.get(c) for c in PLAYER_STATS_COLS}}
    stats = pl.DataFrame([st_row("qbB", "B", pass_att=300.0, pass_yds=2500.0, pass_td=20.0),
                          st_row("kA", "A", fgm=15.0, fga=18.0, fg_long=51.0),
                          st_row("pA", "A", punts=0.0)],                 # all zeros: not production
                         schema={"season": pl.Int32, "player_id": pl.Utf8, "team": pl.Utf8,
                                 **{c: pl.Float64 for c in PLAYER_STATS_COLS}})
    ds = SimpleNamespace(roster=roster, player_ppa=ppa, player_defense=d, team_status=status, player_stats=stats,
                         fbs_teams=lambda s: ["A", "B"])
    t = roster_tables(ds, 2026)
    p = t["roster_players"]
    assert p.height == p["player_id"].n_unique()                       # one row per player
    assert all(f"prev_{c}" in p.columns for c in PREV_STATS) and "prev_tackles" in p.columns
    rows = {r["player_id"]: r for r in p.iter_rows(named=True)}
    assert (rows["qbB"]["prev_pass_yds"], rows["qbB"]["prev_pass_td"]) == (2500.0, 20.0)
    assert rows["kA"]["prev_fg_long"] == 51.0
    assert rows["pA"]["prev_punts"] is None
    c = t["roster_careers"].filter(pl.col("player_id") == "kA")
    assert c.select("career_season", "career_team", "fgm", "fga").row(0) == (2025, "A", 15.0, 18.0)
    assert t["roster_careers"].filter(pl.col("player_id") == "pA").height == 0
    dep = t["roster_departures"]
    assert dep.height == dep.select("player_id", "team").n_unique()

    # A store built before player_stats existed: same columns, box-score stats all null.
    old = roster_tables(SimpleNamespace(**{k: v for k, v in vars(ds).items() if k != "player_stats"}), 2026)
    op = old["roster_players"]
    assert op.columns == p.columns and op["prev_pass_yds"].null_count() == op.height
