"""Deterministic anomaly checks. Each takes a read-only DuckDB connection (ingest checks) or a
`rank` output file (model checks) and returns Anomaly records; none writes anything.

Baselines behind the thresholds (2022–2026, completed games with an FBS side) are in the spec:
docs/superpowers/specs/2026-11-08-anomaly-investigation-design.md.
"""

from __future__ import annotations

import duckdb

from cfbr.anomalies.records import Anomaly
from cfbr.config import Anomalies

# Rows excluded from the duplicate check: administrative rows legitimately repeat.
DROPPED_FOR_DUPES = ("Timeout", "End Period", "End of Half", "End of Game", "End of Regulation")

_WEEK_GAMES = """
    SELECT * FROM games
    WHERE season = ? AND week = ? AND season_type = 'regular' AND completed
      AND (home_class = 'fbs' OR away_class = 'fbs')
"""


def _label(home: str, away: str) -> str:
    return f"{away} @ {home}"


def a1_no_pbp(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    rows = con.execute(f"""
        SELECT g.game_id, g.home_team, g.away_team FROM ({_WEEK_GAMES}) g
        WHERE NOT EXISTS (SELECT 1 FROM plays_raw p WHERE p.game_id = g.game_id)
        ORDER BY g.game_id""", [season, week]).fetchall()
    return [Anomaly("A1", season, week, str(gid), f"{_label(h, a)}: completed, no play-by-play",
                    investigate=False, game_id=gid) for gid, h, a in rows]


def a2_score_gap(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    rows = con.execute(f"""
        WITH m AS (
            SELECT game_id,
                   max(CASE WHEN offense = home THEN offense_score ELSE defense_score END) AS hs,
                   max(CASE WHEN offense = home THEN defense_score ELSE offense_score END) AS as_
            FROM plays_raw WHERE season = ? AND week = ? AND season_type = 'regular' GROUP BY game_id),
        d AS (
            SELECT g.game_id, g.home_team, g.away_team, g.home_points, g.away_points, m.hs, m.as_,
                   abs(m.hs - g.home_points) + abs(m.as_ - g.away_points) AS gap
            FROM ({_WEEK_GAMES}) g JOIN m USING (game_id))
        SELECT * FROM d WHERE gap > ? ORDER BY game_id""",
        [season, week, season, week, cfg.a2_max_score_gap]).fetchall()
    return [Anomaly("A2", season, week, str(gid),
                    f"{_label(h, a)}: PBP max score {as_}-{hs} vs final {ap}-{hp} (gap {gap})",
                    value=float(gap), threshold=float(cfg.a2_max_score_gap), investigate=True, game_id=gid)
            for gid, h, a, hp, ap, hs, as_, gap in rows]


def a3_play_count(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    lo, hi = cfg.a3_min_plays, cfg.a3_max_plays
    rows = con.execute(f"""
        WITH c AS (SELECT game_id, count(*) AS n FROM plays_raw
                   WHERE season = ? AND week = ? AND season_type = 'regular' GROUP BY game_id)
        SELECT g.game_id, g.home_team, g.away_team, c.n FROM ({_WEEK_GAMES}) g JOIN c USING (game_id)
        WHERE c.n < ? OR c.n > ? ORDER BY g.game_id""", [season, week, season, week, lo, hi]).fetchall()
    return [Anomaly("A3", season, week, str(gid), f"{_label(h, a)}: {n} plays (expected {lo}–{hi})",
                    value=float(n), investigate=True, game_id=gid) for gid, h, a, n in rows]


def a4_duplicates(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    excluded = ", ".join(f"'{t}'" for t in DROPPED_FOR_DUPES)
    rows = con.execute(f"""
        WITH dup AS (
            SELECT game_id, sum(c - 1) AS extra FROM (
                SELECT game_id, period, clock_minutes, clock_seconds, down, distance, yards_to_goal,
                       play_text, count(*) AS c
                FROM plays_raw
                WHERE season = ? AND week = ? AND season_type = 'regular' AND play_text IS NOT NULL
                  AND play_type NOT IN ({excluded})
                GROUP BY ALL HAVING count(*) > 1)
            GROUP BY game_id)
        SELECT g.game_id, g.home_team, g.away_team, dup.extra FROM ({_WEEK_GAMES}) g JOIN dup USING (game_id)
        WHERE dup.extra > ? ORDER BY g.game_id""",
        [season, week, season, week, cfg.a4_max_duplicates]).fetchall()
    return [Anomaly("A4", season, week, str(gid), f"{_label(h, a)}: {extra} duplicate play rows",
                    value=float(extra), threshold=float(cfg.a4_max_duplicates), investigate=True, game_id=gid)
            for gid, h, a, extra in rows]


def a5_fbs_status(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    rows = con.execute("""
        WITH first AS (
            SELECT team, min(week) AS first_week FROM (
                SELECT home_team AS team, week FROM games WHERE season = ? AND season_type = 'regular'
                UNION ALL
                SELECT away_team, week FROM games WHERE season = ? AND season_type = 'regular')
            GROUP BY team)
        SELECT c.team, c.classification, p.classification
        FROM team_status c JOIN first f ON f.team = c.team
        LEFT JOIN team_status p ON p.team = c.team AND p.season = c.season - 1
        WHERE c.season = ? AND f.first_week = ?
          AND (c.classification = 'fbs') <> (coalesce(p.classification, '') = 'fbs')
        ORDER BY c.team""", [season, season, season, week]).fetchall()
    return [Anomaly("A5", season, week, team,
                    f"{team}: {prev if prev else 'no row'} last season → {cur} this season "
                    "(check FBS status and pooling)", investigate=False, team=team)
            for team, cur, prev in rows]
