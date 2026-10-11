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


ROSTER_SCHEMA = {"season": pl.Int32, "player_id": pl.Utf8, "team": pl.Utf8, "position": pl.Utf8,
                 "class_year": pl.Int32, "name": pl.Utf8, "jersey": pl.Int32, "height": pl.Int32, "weight": pl.Int32,
                 "home_city": pl.Utf8, "home_state": pl.Utf8}


def _roster_rows(year: int, records: list) -> list[dict]:
    return [{"season": year, "player_id": str(r["id"]), "team": r.get("team"), "position": r.get("position"),
             "class_year": r.get("year") if isinstance(r.get("year"), int) and r.get("year") <= 6 else None,
             "name": " ".join(x for x in (r.get("firstName"), r.get("lastName")) if x) or None,
             "jersey": r.get("jersey"), "height": r.get("height"), "weight": r.get("weight"),
             "home_city": r.get("homeCity") or None, "home_state": r.get("homeState") or None}
            for r in records if r.get("id") is not None]


def build_roster_latest(raw: Path) -> pl.DataFrame:
    """Newest roster snapshot per season (weekly wNN.json if any, else season.json),
    with ``as_of`` = when that file was fetched. For display only; the model uses
    ``roster`` (the first, fixed snapshot)."""
    from datetime import UTC, datetime

    frames = []
    d = raw / "roster"
    for yd in sorted(p for p in d.iterdir() if p.is_dir()) if d.exists() else []:
        weekly = sorted(yd.glob("w*.json"))
        f = weekly[-1] if weekly else (yd / "season.json")
        if not f.exists():
            continue
        df = pl.DataFrame(_roster_rows(int(yd.name), _read(f)), schema=ROSTER_SCHEMA)
        frames.append(df.with_columns(as_of=pl.lit(datetime.fromtimestamp(f.stat().st_mtime, UTC))))
    if not frames:
        return pl.DataFrame(schema={**ROSTER_SCHEMA, "as_of": pl.Datetime("us", "UTC")})
    return pl.concat(frames)


def build_roster(raw: Path) -> pl.DataFrame:
    """First (fixed) roster snapshot per season; what the model uses."""
    rows = [row for y, p in _season_files(raw, "roster") for row in _roster_rows(y, _read(p))]
    return pl.DataFrame(rows, schema=ROSTER_SCHEMA)


DRAFT_SCHEMA = {"season": pl.Int32, "player_id": pl.Utf8, "name": pl.Utf8, "college_team": pl.Utf8,
                "position": pl.Utf8, "nfl_team": pl.Utf8, "round": pl.Int32, "pick": pl.Int32, "overall": pl.Int32}


def build_draft_picks(raw: Path) -> pl.DataFrame:
    """NFL draft picks; ``season`` is the draft year (the college season the player
    did not return for). ``player_id`` is the CFBD college athlete id."""
    rows = [{"season": y, "player_id": str(r["collegeAthleteId"]) if r.get("collegeAthleteId") is not None else None,
             "name": r.get("name"), "college_team": r.get("collegeTeam"), "position": r.get("position"),
             "nfl_team": r.get("nflTeam"), "round": r.get("round"), "pick": r.get("pick"), "overall": r.get("overall")}
            for y, p in _season_files(raw, "draft_picks") for r in _read(p)]
    return pl.DataFrame(rows, schema=DRAFT_SCHEMA)


PORTAL_SCHEMA = {"season": pl.Int32, "name": pl.Utf8, "position": pl.Utf8, "origin": pl.Utf8,
                 "destination": pl.Utf8, "transfer_date": pl.Datetime("us", "UTC"), "eligibility": pl.Utf8}


def build_portal(raw: Path) -> pl.DataFrame:
    """Transfer portal entries for the season the player would play in. No player
    id: CFBD's portal records carry only name, position and schools."""
    rows = [{"season": y, "name": " ".join(x for x in (r.get("firstName"), r.get("lastName")) if x) or None,
             "position": r.get("position"), "origin": r.get("origin"), "destination": r.get("destination"),
             "transfer_date": r.get("transferDate"), "eligibility": r.get("eligibility")}
            for y, p in _season_files(raw, "player_portal") for r in _read(p)]
    df = pl.DataFrame(rows, schema={**PORTAL_SCHEMA, "transfer_date": pl.Utf8})
    return df.with_columns(transfer_date=pl.col("transfer_date").str.to_datetime(time_zone="UTC", strict=False)
                           .cast(pl.Datetime("us", "UTC")))


COUNT_STATS = {("passing", "ATT"): "pass_att", ("rushing", "CAR"): "rush_car", ("receiving", "REC"): "rec"}


def _season_counts(raw: Path) -> dict[tuple[int, str], dict[str, float]]:
    """(season, player_id) -> pass attempts, carries, receptions from CFBD season stats."""
    out: dict[tuple[int, str], dict[str, float]] = {}
    for y, p in _season_files(raw, "stats_player_season"):
        for r in _read(p):
            col = COUNT_STATS.get((r.get("category"), r.get("statType")))
            if col is None:
                continue
            try:
                v = float(r.get("stat"))
            except (TypeError, ValueError):
                continue
            d = out.setdefault((y, str(r["playerId"])), {})
            d[col] = d.get(col, 0.0) + v
    return out


def build_player_ppa(raw: Path) -> pl.DataFrame:
    """Season PPA per player (offense).

    CFBD's season PPA has no play count. ``plays`` = pass attempts + carries +
    receptions and ``pass_plays`` = pass attempts, both actual counts from CFBD
    season stats (null when the player has none). Averages come straight from
    CFBD's averagePPA. (Earlier versions back-calculated plays as total/average,
    which blows up when the rounded average is near zero.)
    """
    counts = _season_counts(raw)
    rows = []
    for y, p in _season_files(raw, "ppa_players_season"):
        for r in _read(p):
            avg, tot = r.get("averagePPA") or {}, r.get("totalPPA") or {}
            c = counts.get((y, str(r["id"])))
            plays = (c.get("pass_att", 0.0) + c.get("rush_car", 0.0) + c.get("rec", 0.0)) if c else None
            rows.append({
                "season": y, "player_id": str(r["id"]), "team": r.get("team"), "position": r.get("position"),
                "total_ppa": tot.get("all"), "avg_ppa": avg.get("all"), "plays": plays or None,
                "pass_ppa": tot.get("pass"), "avg_pass_ppa": avg.get("pass"),
                "pass_plays": (c.get("pass_att") or None) if c else None,
                "rush_car": c.get("rush_car") if c else None, "rec": c.get("rec") if c else None,
            })
    return pl.DataFrame(rows, schema={"season": pl.Int32, "player_id": pl.Utf8, "team": pl.Utf8, "position": pl.Utf8,
                                      "total_ppa": pl.Float64, "avg_ppa": pl.Float64, "plays": pl.Float64,
                                      "pass_ppa": pl.Float64, "avg_pass_ppa": pl.Float64, "pass_plays": pl.Float64,
                                      "rush_car": pl.Float64, "rec": pl.Float64})


DEF_STATS = {("defensive", "TOT"): "tackles", ("defensive", "TFL"): "tfl", ("defensive", "SACKS"): "sacks",
             ("defensive", "PD"): "pd", ("defensive", "QB HUR"): "hurries", ("interceptions", "INT"): "ints"}


def build_player_defense(raw: Path) -> pl.DataFrame:
    rows = []
    for y, p in _season_files(raw, "stats_player_season"):
        for r in _read(p):
            col = DEF_STATS.get((r.get("category"), r.get("statType")))
            if col is None:
                continue
            try:
                v = float(r.get("stat"))
            except (TypeError, ValueError):
                continue
            rows.append({"season": y, "player_id": str(r["playerId"]), "team": r.get("team"), "stat": col, "value": v})
    long = pl.DataFrame(rows, schema={"season": pl.Int32, "player_id": pl.Utf8, "team": pl.Utf8, "stat": pl.Utf8,
                                      "value": pl.Float64})
    cols = list(dict.fromkeys(DEF_STATS.values()))
    if long.is_empty():
        return pl.DataFrame(schema={"season": pl.Int32, "player_id": pl.Utf8, "team": pl.Utf8,
                                    **{c: pl.Float64 for c in cols}})
    wide = long.pivot(on="stat", index=["season", "player_id", "team"], values="value", aggregate_function="sum")
    for c in cols:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(0.0).alias(c))
    return wide.select("season", "player_id", "team", *[pl.col(c).fill_null(0.0) for c in cols])


GAME_PLAYER_STATS = {
    ("passing", "C/ATT"): None,  # parsed into pass_cmp / pass_att
    ("rushing", "CAR"): "rush_att", ("receiving", "REC"): "rec", ("defensive", "TOT"): "tackles",
    ("defensive", "TFL"): "tfl", ("defensive", "PD"): "pd", ("interceptions", "INT"): "ints",
}
GAME_PLAYER_SCHEMA = {
    "season": pl.Int32, "season_type": pl.Utf8, "week": pl.Int32, "game_id": pl.Int64, "team": pl.Utf8,
    "player_id": pl.Utf8, "name": pl.Utf8, "pass_cmp": pl.Float64, "pass_att": pl.Float64, "rush_att": pl.Float64,
    "rec": pl.Float64, "tackles": pl.Float64, "tfl": pl.Float64, "pd": pl.Float64, "ints": pl.Float64,
}


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def build_game_players(raw: Path) -> pl.DataFrame:
    """One row per player per game with any box-score stat (= appeared in the game)."""
    rows: dict[tuple, dict] = {}
    d = raw / "games_players"
    if not d.exists():
        return pl.DataFrame(schema=GAME_PLAYER_SCHEMA)
    for p in sorted(d.glob("*/*.json")):
        year = int(p.parent.name)
        st, _, wk = p.stem.rpartition("_w")
        for g in _read(p):
            for t in g.get("teams") or []:
                for cat in t.get("categories") or []:
                    for ty in cat.get("types") or []:
                        key_stat = (cat.get("name"), ty.get("name"))
                        if key_stat not in GAME_PLAYER_STATS:
                            continue
                        for a in ty.get("athletes") or []:
                            if a.get("id") is None or str(a["id"]).startswith("-"):
                                continue  # team totals use negative ids
                            k = (g["id"], t["team"], str(a["id"]))
                            r = rows.setdefault(k, {
                                "season": year, "season_type": st, "week": int(wk), "game_id": g["id"],
                                "team": t["team"], "player_id": str(a["id"]), "name": a.get("name"),
                                **{c: 0.0 for c in ("pass_cmp", "pass_att", "rush_att", "rec", "tackles", "tfl", "pd", "ints")},
                            })
                            if key_stat == ("passing", "C/ATT"):
                                cmp_, _, att = str(a.get("stat", "")).partition("/")
                                r["pass_cmp"] += _num(cmp_)
                                r["pass_att"] += _num(att)
                            else:
                                r[GAME_PLAYER_STATS[key_stat]] += _num(a.get("stat"))
    return pl.DataFrame(list(rows.values()), schema=GAME_PLAYER_SCHEMA)


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
            ("calendar", build_calendar), ("polls", build_polls), ("roster", build_roster),
            ("roster_latest", build_roster_latest), ("draft_picks", build_draft_picks), ("portal", build_portal),
            ("player_ppa", build_player_ppa), ("player_defense", build_player_defense),
            ("game_players", build_game_players),
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
