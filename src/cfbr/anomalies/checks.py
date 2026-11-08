"""Deterministic anomaly checks. Each takes a read-only DuckDB connection (ingest checks) or a
`rank` output file (model checks) and returns Anomaly records; none writes anything.

Baselines behind the thresholds (2022–2026, completed games with an FBS side) are in the spec:
docs/superpowers/specs/2026-11-08-anomaly-investigation-design.md.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import duckdb
import polars as pl

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
        SELECT * FROM d WHERE gap > ? OR hs IS NULL OR as_ IS NULL ORDER BY game_id""",
        [season, week, season, week, cfg.a2_max_score_gap]).fetchall()
    out = []
    for gid, h, a, hp, ap, hs, as_, gap in rows:
        if gap is None:
            out.append(Anomaly("A2", season, week, str(gid),
                               f"{_label(h, a)}: play-by-play has no scores (final {ap}-{hp})",
                               value=None, threshold=float(cfg.a2_max_score_gap), investigate=True, game_id=gid))
        else:
            out.append(Anomaly("A2", season, week, str(gid),
                               f"{_label(h, a)}: PBP max score {as_}-{hs} vs final {ap}-{hp} (gap {gap})",
                               value=float(gap), threshold=float(cfg.a2_max_score_gap), investigate=True,
                               game_id=gid))
    return out


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


BUCKETS = ("1-3", "4-7", "8+")


def week_bucket(week: int) -> str:
    """Early-season ratings move more (the prior is fading), so B1 uses a threshold per bucket."""
    if week <= 3:
        return "1-3"
    return "4-7" if week <= 7 else "8+"


def load_thresholds(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def b1_rating_moves(rankings_csv: Path, season: int, week: int, thresholds: dict) -> list[Anomaly]:
    bucket = week_bucket(week)
    thr = float(thresholds["b1"][bucket])
    df = pl.read_csv(rankings_csv).filter(pl.col("rating_change").is_not_null()
                                          & (pl.col("rating_change").abs() > thr))
    return [Anomaly("B1", season, week, team,
                    f"{team}: rating {chg:+.1f} this week (threshold {thr:.1f}, weeks {bucket})",
                    value=float(chg), threshold=thr, investigate=True, team=team)
            for team, chg in df.select("team", "rating_change").iter_rows()]


def b2_ppa_outliers(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    lo_q, hi_q = cfg.b2_quantiles
    rows = con.execute("""
        WITH tg AS (
            SELECT game_id, week, offense_team AS team, avg(ppa) AS ppa, count(*) AS n
            FROM model_plays
            WHERE season = ? AND season_type = 'regular' AND NOT offense_fcs AND ppa IS NOT NULL
            GROUP BY game_id, week, offense_team),
        q AS (SELECT quantile_cont(ppa, ?) AS lo, quantile_cont(ppa, ?) AS hi FROM tg)
        SELECT tg.game_id, tg.team, tg.ppa, tg.n, q.lo, q.hi FROM tg, q
        WHERE tg.week = ? AND (tg.ppa < q.lo OR tg.ppa > q.hi)
        ORDER BY tg.game_id, tg.team""", [season, lo_q, hi_q, week]).fetchall()
    return [Anomaly("B2", season, week, f"{gid}:{team}",
                    f"{team} in game {gid}: offense {ppa:+.2f} PPA/play over {n} plays "
                    f"(season range {lo:+.2f}..{hi:+.2f})",
                    value=float(ppa), alert=False, game_id=gid, team=team)
            for gid, team, ppa, n, lo, hi in rows]


def b3_line_gaps(predictions_csv: Path, season: int, week: int, thresholds: dict) -> list[Anomaly]:
    """Model home margin vs the current consensus line (home margin = -spread). Compared only;
    nothing here feeds the model or the bets."""
    thr = float(thresholds["b3"]["max_abs_gap"])
    df = (pl.read_csv(predictions_csv)
          .filter(pl.col("closing_spread").is_not_null())
          .with_columns(vegas=-pl.col("closing_spread"))
          .with_columns(gap=(pl.col("pred_margin") - pl.col("vegas")).abs())
          .filter(pl.col("gap") > thr))
    return [Anomaly("B3", season, week, str(gid),
                    f"{_label(h, a)}: model {pred:+.1f} vs line {vegas:+.1f} home margin "
                    f"(gap {gap:.1f}, threshold {thr:.1f})",
                    value=float(gap), threshold=thr, investigate=True, game_id=int(gid))
            for gid, h, a, pred, vegas, gap in df.select("game_id", "home_team", "away_team", "pred_margin",
                                                         "vegas", "gap").iter_rows()]


DB_CHECKS = (("A1", a1_no_pbp), ("A2", a2_score_gap), ("A3", a3_play_count), ("A4", a4_duplicates),
             ("A5", a5_fbs_status), ("B2", b2_ppa_outliers))


def _guarded(found: list[Anomaly], check: str, season: int, week: int, fn) -> None:
    """A check that crashes becomes an alerted ERR item; the other checks still run."""
    try:
        found.extend(fn())
    except Exception as e:  # noqa: BLE001 - advisory step, must never stop the run
        found.append(Anomaly("ERR", season, week, check, f"check {check} failed: {type(e).__name__}: {e}"[:300]))


def run_db_checks(db_path: Path, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    found: list[Anomaly] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        for check, fn in DB_CHECKS:
            _guarded(found, check, season, week, lambda fn=fn: fn(con, season, week, cfg))
    return found


def run_checks(db_path: Path, week_dir: Path, thresholds_file: Path, season: int, week: int,
               cfg: Anomalies) -> list[Anomaly]:
    found = run_db_checks(db_path, season, week, cfg)
    _guarded(found, "B1", season, week,
             lambda: b1_rating_moves(week_dir / "rankings.csv", season, week, load_thresholds(thresholds_file)))
    _guarded(found, "B3", season, week,
             lambda: b3_line_gaps(week_dir / "predictions.csv", season, week, load_thresholds(thresholds_file)))
    return found
