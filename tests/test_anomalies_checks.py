import duckdb
import pytest

from cfbr.anomalies import checks
from cfbr.config import Anomalies

CFG = Anomalies()


def _db(path):
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE games (game_id BIGINT, season INTEGER, week INTEGER, season_type VARCHAR,
        completed BOOLEAN, home_team VARCHAR, away_team VARCHAR, home_class VARCHAR, away_class VARCHAR,
        home_points INTEGER, away_points INTEGER)""")
    con.execute("""CREATE TABLE plays_raw (play_id VARCHAR, game_id BIGINT, season INTEGER, week INTEGER,
        season_type VARCHAR, offense VARCHAR, defense VARCHAR, home VARCHAR, offense_score INTEGER,
        defense_score INTEGER, period INTEGER, clock_minutes INTEGER, clock_seconds INTEGER, down INTEGER,
        distance INTEGER, yards_to_goal INTEGER, play_type VARCHAR, play_text VARCHAR)""")
    con.execute("CREATE TABLE team_status (season INTEGER, team VARCHAR, classification VARCHAR, conference VARCHAR)")
    con.execute("""CREATE TABLE model_plays (season INTEGER, week INTEGER, season_type VARCHAR, game_id BIGINT,
        offense_team VARCHAR, offense_fcs BOOLEAN, ppa DOUBLE)""")
    return con


def _game(con, gid, home="Rice", away="UTSA", hp=21, ap=14, week=9, cls=("fbs", "fbs"), completed=True):
    con.execute("INSERT INTO games VALUES (?, 2026, ?, 'regular', ?, ?, ?, ?, ?, ?, ?)",
                [gid, week, completed, home, away, cls[0], cls[1], hp, ap])


def _plays(con, gid, n, home="Rice", away="UTSA", final=(21, 14), week=9, text="run for 3"):
    """n distinct plays; the last one carries the final score from the home offense's view."""
    rows = []
    for i in range(n):
        hs, as_ = final if i == n - 1 else (0, 0)
        rows.append([f"{gid}{i:04d}", gid, 2026, week, "regular", home, away, home, hs, as_,
                     1 + i // 50, 14 - (i % 15), i % 60, 1 + i % 4, 10, 75 - i % 50, "Rush", f"{text} #{i}"])
    con.executemany("INSERT INTO plays_raw VALUES (" + ", ".join(["?"] * 18) + ")", rows)


def test_a1_flags_completed_fbs_game_without_plays(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    _game(con, 1)
    _plays(con, 1, 170)
    _game(con, 2, home="Army", away="Navy")                         # no plays -> fires
    _game(con, 3, home="Fordham", away="Colgate", cls=("fcs", "fcs"))  # FCS-only: ignored
    _game(con, 4, home="Iowa", away="Ohio", completed=False)        # not completed: ignored
    out = checks.a1_no_pbp(con, 2026, 9, CFG)
    assert [(a.check, a.key, a.investigate, a.alert) for a in out] == [("A1", "2", False, True)]
    assert out[0].message == "Navy @ Army: completed, no play-by-play"


def test_a2_score_gap_over_threshold_only(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    _game(con, 1, hp=21, ap=14)
    _plays(con, 1, 170, final=(21, 14))   # matches
    _game(con, 2, home="Army", away="Navy", hp=28, ap=14)
    _plays(con, 2, 170, home="Army", away="Navy", final=(20, 14))  # gap 8: not > 8
    _game(con, 3, home="Iowa", away="Ohio", hp=30, ap=10)
    _plays(con, 3, 170, home="Iowa", away="Ohio", final=(14, 7))   # gap 19
    out = checks.a2_score_gap(con, 2026, 9, CFG)
    assert [(a.key, a.value, a.investigate) for a in out] == [("3", 19.0, True)]
    assert out[0].message == "Ohio @ Iowa: PBP max score 7-14 vs final 10-30 (gap 19)"


def test_a3_play_count_outside_range(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    for gid, n in [(1, 170), (2, 105), (3, 260), (4, 110)]:
        _game(con, gid, home=f"H{gid}", away=f"A{gid}")
        _plays(con, gid, n, home=f"H{gid}", away=f"A{gid}")
    out = checks.a3_play_count(con, 2026, 9, CFG)
    assert [(a.key, a.value) for a in out] == [("2", 105.0), ("3", 260.0)]
    assert out[0].message == "A2 @ H2: 105 plays (expected 110–250)" and out[0].investigate


def test_a4_counts_duplicate_rows_but_ignores_timeouts(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    _game(con, 1)
    _plays(con, 1, 170)
    dup = ["Rice", "UTSA", "Rice", 0, 0, 2, 7, 30, 2, 5, 40, "Pass Reception", "pass for 12"]
    con.executemany("INSERT INTO plays_raw VALUES (?, 1, 2026, 9, 'regular', " + ", ".join(["?"] * 13) + ")",
                    [[f"d{i}", *dup] for i in range(5)])            # 4 extra rows: > 3
    to = ["Rice", "UTSA", "Rice", 0, 0, 3, 5, 0, None, None, None, "Timeout", "Timeout RICE"]
    con.executemany("INSERT INTO plays_raw VALUES (?, 1, 2026, 9, 'regular', " + ", ".join(["?"] * 13) + ")",
                    [[f"t{i}", *to] for i in range(6)])             # timeouts never count
    _game(con, 2, home="Army", away="Navy")
    _plays(con, 2, 170, home="Army", away="Navy")
    con.executemany("INSERT INTO plays_raw VALUES (?, 2, 2026, 9, 'regular', " + ", ".join(["?"] * 13) + ")",
                    [[f"e{i}", "Army", "Navy", "Army", 0, 0, 1, 9, 0, 1, 10, 60, "Rush", "run for 2"] for i in range(4)])
    out = checks.a4_duplicates(con, 2026, 9, CFG)                   # game 2 has 3 extra: not > 3
    assert [(a.key, a.value) for a in out] == [("1", 4.0)]
    assert out[0].message == "UTSA @ Rice: 4 duplicate play rows"


def test_a5_flags_status_change_only_in_first_week(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    con.executemany("INSERT INTO team_status VALUES (?, ?, ?, NULL)", [
        (2025, "Sacramento State", "fcs"), (2026, "Sacramento State", "fbs"),   # moved up: fires
        (2025, "Rice", "fbs"), (2026, "Rice", "fbs"),                           # unchanged
        (2026, "Newco", "fbs"),                                                 # no row last season: fires
        (2025, "Delaware", "fcs"), (2026, "Delaware", "fbs")])                  # first game in week 1: not this week
    _game(con, 1, home="Sacramento State", away="Rice", week=9)
    _game(con, 2, home="Newco", away="Rice", week=10, completed=False)
    _game(con, 3, home="Newco", away="Rice", week=9)
    _game(con, 4, home="Delaware", away="Rice", week=1)
    _game(con, 5, home="Delaware", away="Rice", week=9)
    out = checks.a5_fbs_status(con, 2026, 9, CFG)
    assert sorted(a.key for a in out) == ["Newco", "Sacramento State"]
    msg = {a.key: a.message for a in out}
    assert msg["Sacramento State"] == "Sacramento State: fcs last season → fbs this season (check FBS status and pooling)"
    assert msg["Newco"] == "Newco: no row last season → fbs this season (check FBS status and pooling)"
    assert all(not a.investigate and a.alert for a in out)
