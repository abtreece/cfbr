"""Normalize cached CFBD JSON into DuckDB tables.

Tables are rebuilt from the raw cache on every load. That is cheap, and it
keeps the store a pure function of the cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import polars as pl

log = logging.getLogger(__name__)


def _read(path: Path) -> list:
    return json.loads(path.read_text())


def _season_files(raw: Path, slug: str) -> list[tuple[int, Path]]:
    d = raw / slug
    if not d.exists():
        return []
    return sorted((int(p.parent.name), p) for p in d.glob("*/season.json"))


def _games_file(raw: Path, slug: str, year: int) -> Path | None:
    d = raw / slug / str(year)
    if (d / "both.json").exists():
        return d / "both.json"
    weekly = sorted(d.glob("both_w*.json"))
    return weekly[-1] if weekly else None


def _years(raw: Path, slug: str) -> list[int]:
    d = raw / slug
    return sorted(int(p.name) for p in d.iterdir() if p.is_dir()) if d.exists() else []


def build_teams(raw: Path) -> pl.DataFrame:
    rows = []
    for year, p in _season_files(raw, "teams_fbs"):
        for t in _read(p):
            rows.append(
                {"season": year, "team_id": t["id"], "team": t["school"],
                 "conference": t.get("conference"), "classification": t.get("classification") or "fbs"}
            )
    return pl.DataFrame(rows, schema={"season": pl.Int32, "team_id": pl.Int64, "team": pl.Utf8,
                                      "conference": pl.Utf8, "classification": pl.Utf8})


GAME_SCHEMA = {
    "game_id": pl.Int64, "season": pl.Int32, "week": pl.Int32, "season_type": pl.Utf8,
    "kickoff_ts": pl.Utf8, "start_time_tbd": pl.Boolean, "completed": pl.Boolean,
    "neutral_site": pl.Boolean, "conference_game": pl.Boolean,
    "home_id": pl.Int64, "home_team": pl.Utf8, "home_conference": pl.Utf8, "home_class": pl.Utf8,
    "home_points": pl.Int32,
    "away_id": pl.Int64, "away_team": pl.Utf8, "away_conference": pl.Utf8, "away_class": pl.Utf8,
    "away_points": pl.Int32,
}


def build_games(raw: Path) -> pl.DataFrame:
    rows = []
    for year in _years(raw, "games"):
        p = _games_file(raw, "games", year)
        if p is None:
            continue
        for g in _read(p):
            if g["seasonType"] not in ("regular", "postseason"):
                continue
            rows.append({
                "game_id": g["id"], "season": g["season"], "week": g["week"], "season_type": g["seasonType"],
                "kickoff_ts": g["startDate"], "start_time_tbd": bool(g.get("startTimeTBD")),
                "completed": bool(g.get("completed")), "neutral_site": bool(g.get("neutralSite")),
                "conference_game": bool(g.get("conferenceGame")),
                "home_id": g.get("homeId"), "home_team": g["homeTeam"], "home_conference": g.get("homeConference"),
                "home_class": g.get("homeClassification"), "home_points": g.get("homePoints"),
                "away_id": g.get("awayId"), "away_team": g["awayTeam"], "away_conference": g.get("awayConference"),
                "away_class": g.get("awayClassification"), "away_points": g.get("awayPoints"),
            })
    df = pl.DataFrame(rows, schema=GAME_SCHEMA)
    return df.with_columns(pl.col("kickoff_ts").str.to_datetime(time_zone="UTC"))


def build_lines(raw: Path) -> pl.DataFrame:
    rows = []
    for year in _years(raw, "lines"):
        p = _games_file(raw, "lines", year)
        if p is None:
            continue
        for g in _read(p):
            for ln in g.get("lines") or []:
                rows.append({
                    "game_id": g["id"], "season": g["season"], "week": g["week"],
                    "season_type": g["seasonType"], "provider": ln.get("provider"),
                    "spread": ln.get("spread"), "spread_open": ln.get("spreadOpen"),
                    "over_under": ln.get("overUnder"),
                })
    return pl.DataFrame(rows, schema={"game_id": pl.Int64, "season": pl.Int32, "week": pl.Int32,
                                      "season_type": pl.Utf8, "provider": pl.Utf8, "spread": pl.Float64,
                                      "spread_open": pl.Float64, "over_under": pl.Float64})


def build_returning(raw: Path) -> pl.DataFrame:
    rows = []
    for year, p in _season_files(raw, "player_returning"):
        for r in _read(p):
            rows.append({
                "season": year, "team": r["team"], "conference": r.get("conference"),
                "total_ppa": r.get("totalPPA"), "percent_ppa": r.get("percentPPA"),
                "percent_passing_ppa": r.get("percentPassingPPA"),
                "percent_rushing_ppa": r.get("percentRushingPPA"),
                "percent_receiving_ppa": r.get("percentReceivingPPA"),
                "usage": r.get("usage"), "passing_usage": r.get("passingUsage"),
                "rushing_usage": r.get("rushingUsage"), "receiving_usage": r.get("receivingUsage"),
            })
    schema = {"season": pl.Int32, "team": pl.Utf8, "conference": pl.Utf8}
    schema |= {k: pl.Float64 for k in ["total_ppa", "percent_ppa", "percent_passing_ppa", "percent_rushing_ppa",
                                       "percent_receiving_ppa", "usage", "passing_usage", "rushing_usage",
                                       "receiving_usage"]}
    return pl.DataFrame(rows, schema=schema)


def build_talent(raw: Path) -> pl.DataFrame:
    rows = [{"season": y, "team": r["team"], "talent": float(r["talent"])}
            for y, p in _season_files(raw, "talent") for r in _read(p)]
    return pl.DataFrame(rows, schema={"season": pl.Int32, "team": pl.Utf8, "talent": pl.Float64})


def build_recruiting(raw: Path) -> pl.DataFrame:
    rows = [{"season": y, "team": r["team"], "rank": r.get("rank"), "points": r.get("points")}
            for y, p in _season_files(raw, "recruiting_teams") for r in _read(p)]
    return pl.DataFrame(rows, schema={"season": pl.Int32, "team": pl.Utf8, "rank": pl.Int32,
                                      "points": pl.Float64})


POLL_SCHEMA = {
    "season": pl.Int32, "season_type": pl.Utf8, "week": pl.Int32, "poll": pl.Utf8, "rank": pl.Int32,
    "team": pl.Utf8, "conference": pl.Utf8, "first_place_votes": pl.Int32, "points": pl.Int32,
}


def build_polls(raw: Path) -> pl.DataFrame:
    rows = []
    for year in _years(raw, "rankings"):
        p = _games_file(raw, "rankings", year)
        if p is None:
            continue
        for w in _read(p):
            for poll in w.get("polls") or []:
                for r in poll.get("ranks") or []:
                    rows.append({
                        "season": w["season"], "season_type": w["seasonType"], "week": int(w["week"]),
                        "poll": poll["poll"], "rank": r.get("rank"), "team": r.get("school"),
                        "conference": r.get("conference"), "first_place_votes": r.get("firstPlaceVotes"),
                        "points": r.get("points"),
                    })
    return pl.DataFrame(rows, schema=POLL_SCHEMA)


def build_calendar(raw: Path) -> pl.DataFrame:
    rows = [{"season": y, "week": w["week"], "season_type": w["seasonType"], "start_date": w.get("startDate"),
             "end_date": w.get("endDate"), "first_game_start": w.get("firstGameStart"),
             "last_game_start": w.get("lastGameStart")}
            for y, p in _season_files(raw, "calendar") for w in _read(p)]
    df = pl.DataFrame(rows, schema={"season": pl.Int32, "week": pl.Int32, "season_type": pl.Utf8,
                                    "start_date": pl.Utf8, "end_date": pl.Utf8, "first_game_start": pl.Utf8,
                                    "last_game_start": pl.Utf8})
    return df.with_columns(pl.col(c).str.to_datetime(time_zone="UTC")
                           for c in ["start_date", "end_date", "first_game_start", "last_game_start"])


PLAY_SCHEMA = {
    "play_id": pl.Utf8, "game_id": pl.Int64, "season": pl.Int32, "week": pl.Int32, "season_type": pl.Utf8,
    "drive_number": pl.Int32, "play_number": pl.Int32, "offense": pl.Utf8, "defense": pl.Utf8,
    "home": pl.Utf8, "away": pl.Utf8, "offense_score": pl.Int32, "defense_score": pl.Int32,
    "period": pl.Int32, "clock_minutes": pl.Int32, "clock_seconds": pl.Int32,
    "yards_to_goal": pl.Int32, "down": pl.Int32, "distance": pl.Int32, "yards_gained": pl.Int32,
    "scoring": pl.Boolean, "play_type": pl.Utf8, "play_text": pl.Utf8, "ppa": pl.Float64,
}


def _plays_frame(year: int, season_type: str, week: int, plays: list) -> pl.DataFrame:
    rows = []
    for x in plays:
        clock = x.get("clock") or {}
        rows.append({
            "play_id": str(x["id"]), "game_id": x["gameId"], "season": year, "week": week,
            "season_type": season_type, "drive_number": x.get("driveNumber"), "play_number": x.get("playNumber"),
            "offense": x.get("offense"), "defense": x.get("defense"), "home": x.get("home"), "away": x.get("away"),
            "offense_score": x.get("offenseScore"), "defense_score": x.get("defenseScore"),
            "period": x.get("period"), "clock_minutes": clock.get("minutes"), "clock_seconds": clock.get("seconds"),
            "yards_to_goal": x.get("yardsToGoal"), "down": x.get("down"), "distance": x.get("distance"),
            "yards_gained": x.get("yardsGained"), "scoring": x.get("scoring"), "play_type": x.get("playType"),
            "play_text": x.get("playText"), "ppa": x.get("ppa"),
        })
    return pl.DataFrame(rows, schema=PLAY_SCHEMA)


def load_plays(con: duckdb.DuckDBPyConnection, raw: Path) -> int:
    con.execute("DROP TABLE IF EXISTS plays_raw")
    created = False
    n = 0
    for p in sorted((raw / "plays").glob("*/*.json")):
        st, _, wk = p.stem.rpartition("_w")
        df = _plays_frame(int(p.parent.name), st, int(wk), _read(p))
        if df.is_empty():
            continue
        if not created:
            con.execute("CREATE TABLE plays_raw AS SELECT * FROM df")
            created = True
        else:
            con.execute("INSERT INTO plays_raw SELECT * FROM df")
        n += df.height
    if not created:
        empty = pl.DataFrame(schema=PLAY_SCHEMA)
        con.execute("CREATE TABLE plays_raw AS SELECT * FROM empty")
    return n


def build_store(raw: Path, db_path: Path) -> dict[str, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        counts = {}
        for name, builder in [
            ("teams_season", build_teams), ("games", build_games), ("lines", build_lines),
            ("returning_production", build_returning), ("talent", build_talent), ("recruiting", build_recruiting),
            ("calendar", build_calendar), ("polls", build_polls),
        ]:
            df = builder(raw)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")
            counts[name] = df.height
        counts["plays_raw"] = load_plays(con, raw)
        _build_team_status(con)
        return counts
    finally:
        con.close()


def _build_team_status(con: duckdb.DuckDBPyConnection) -> None:
    """Per team-season FBS/FCS status: FBS if listed by /teams/fbs, else the
    classification CFBD attached to the team in that season's games."""
    con.execute("""
        CREATE OR REPLACE TABLE team_status AS
        WITH from_games AS (
            SELECT season, home_team AS team, home_class AS cls, home_conference AS conf FROM games
            UNION ALL
            SELECT season, away_team, away_class, away_conference FROM games
        ),
        g AS (
            SELECT season, team, mode(cls) AS cls, mode(conf) AS conf
            FROM from_games GROUP BY season, team
        )
        SELECT g.season, g.team,
               CASE WHEN t.team IS NOT NULL THEN 'fbs' ELSE coalesce(g.cls, 'unknown') END AS classification,
               coalesce(t.conference, g.conf) AS conference
        FROM g LEFT JOIN teams_season t ON t.season = g.season AND t.team = g.team
    """)
