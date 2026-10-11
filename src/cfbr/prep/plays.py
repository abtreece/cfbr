"""Build the scrimmage-play tables used by the rating model.

``scrimmage_plays`` keeps every scrimmage play with the columns needed to
re-apply the configurable filters (garbage time, FCS pooling) in memory during
tuning. ``model_plays`` is that table with the configured filters applied.
"""

from __future__ import annotations

import logging

import duckdb
import polars as pl

from cfbr.config import Config, GarbageTime

log = logging.getLogger(__name__)

# CFBD playType -> kept as a scrimmage play. Everything else is dropped.
# The full mapping, including dropped types, is recorded in DECISIONS.md.
RUSH_TYPES = {"Rush", "Rushing Touchdown"}
PASS_TYPES = {
    "Pass Reception", "Pass Completion", "Pass Incompletion", "Passing Touchdown", "Pass",
    "Sack", "Pass Interception", "Pass Interception Return", "Interception",
    "Interception Return Touchdown",
}
FUMBLE_TYPES = {"Fumble", "Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown"}
OTHER_SCRIMMAGE = {"Safety"}
SCRIMMAGE_TYPES = RUSH_TYPES | PASS_TYPES | FUMBLE_TYPES | OTHER_SCRIMMAGE

OFFENSIVE_TD_TYPES = {"Rushing Touchdown", "Passing Touchdown"}
TURNOVER_TYPES = {
    "Pass Interception", "Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Fumble Recovery (Opponent)", "Fumble Return Touchdown", "Safety",
}

FCS = "FCS"


def success_expr(
    down: str = "down", distance: str = "distance", gained: str = "yards_gained", play_type: str = "play_type"
) -> pl.Expr:
    """Success: 1st down gains >= 50% of distance, 2nd >= 70%, 3rd/4th >= 100%.

    Offensive touchdowns always succeed; turnovers and safeties never do.
    """
    need = (
        pl.when(pl.col(down) == 1).then(0.5 * pl.col(distance))
        .when(pl.col(down) == 2).then(0.7 * pl.col(distance))
        .otherwise(pl.col(distance).cast(pl.Float64))
    )
    return (
        pl.when(pl.col(play_type).is_in(list(TURNOVER_TYPES))).then(False)
        .when(pl.col(play_type).is_in(list(OFFENSIVE_TD_TYPES))).then(True)
        .otherwise(pl.col(gained) >= need)
    )


# Rows whose scores are unreliable: CFBD sometimes swaps offense/defense or
# carries stale scores on administrative and kickoff rows.
SCORE_UNRELIABLE_TYPES = {
    "Timeout", "End Period", "End of Half", "End of Game", "End of Regulation", "Kickoff",
    "Kickoff Return (Offense)", "Penalty", "Uncategorized", "placeholder",
}


def presnap_scores(df: pl.DataFrame) -> pl.DataFrame:
    """CFBD offense/defense scores are *after* the play. The score at the snap
    is the post-play score of the last reliable row before it in the game."""
    reliable = ~pl.col("play_type").is_in(list(SCORE_UNRELIABLE_TYPES))
    df = df.with_columns(
        home_after=pl.when(reliable).then(
            pl.when(pl.col("offense") == pl.col("home")).then(pl.col("offense_score"))
            .otherwise(pl.col("defense_score"))),
        away_after=pl.when(reliable).then(
            pl.when(pl.col("offense") == pl.col("home")).then(pl.col("defense_score"))
            .otherwise(pl.col("offense_score"))),
    ).sort(["game_id", "play_seq"])
    df = df.with_columns(
        home_before=pl.col("home_after").shift(1).forward_fill().over("game_id").fill_null(0),
        away_before=pl.col("away_after").shift(1).forward_fill().over("game_id").fill_null(0),
    )
    return df.with_columns(margin_at_snap=(pl.col("home_before") - pl.col("away_before")).abs())


def garbage_time_mask(gt: GarbageTime) -> pl.Expr:
    """True for plays to keep. No filter in Q1 or overtime; drop when the
    absolute margin at the snap exceeds the quarter's threshold."""
    if not gt.enabled:
        return pl.lit(True)
    m, q = pl.col("margin_at_snap"), pl.col("period")
    return (
        pl.when(q == 2).then(m <= gt.q2)
        .when(q == 3).then(m <= gt.q3)
        .when(q == 4).then(m <= gt.q4)
        .otherwise(True)
    )


def pool_fcs(df: pl.DataFrame, strategy: str) -> pl.DataFrame:
    """Replace non-FBS teams with pooled entities in ``offense``/``defense``."""
    def pooled(team: str, fcs: str, conf: str) -> pl.Expr:
        if strategy == "single":
            label = pl.lit(FCS)
        elif strategy == "conference":
            label = pl.lit(FCS + ":") + pl.col(conf).fill_null("Unknown")
        else:
            raise ValueError(f"unknown fcs_pooling strategy {strategy!r}")
        return pl.when(pl.col(fcs)).then(label).otherwise(pl.col(team))

    return df.with_columns(
        offense=pooled("offense_team", "offense_fcs", "offense_conference"),
        defense=pooled("defense_team", "defense_fcs", "defense_conference"),
    )


def apply_filters(scrimmage: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """scrimmage_plays -> model_plays under a given config."""
    out = scrimmage.filter(garbage_time_mask(cfg.prep.garbage_time))
    return pool_fcs(out, cfg.prep.fcs_pooling)


SCRIMMAGE_SQL = """
SELECT p.play_id, p.game_id, p.season, p.week, p.season_type, g.kickoff_ts,
       p.drive_number, p.play_number, p.offense, p.defense, p.home, p.away,
       p.offense_score, p.defense_score, p.period, p.down, p.distance, p.yards_gained,
       p.play_type, p.ppa,
       g.home_team, g.neutral_site,
       coalesce(so.classification, 'unknown') <> 'fbs' AS offense_fcs,
       coalesce(sd.classification, 'unknown') <> 'fbs' AS defense_fcs,
       so.conference AS offense_conference, sd.conference AS defense_conference
FROM plays_raw p
JOIN games g ON g.game_id = p.game_id
LEFT JOIN team_status so ON so.season = p.season AND so.team = p.offense
LEFT JOIN team_status sd ON sd.season = p.season AND sd.team = p.defense
"""


def build_scrimmage(con: duckdb.DuckDBPyConnection) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (scrimmage_plays, per-season null-PPA report)."""
    raw = con.execute(SCRIMMAGE_SQL).pl()
    # Play order within a game: CFBD play ids are the ESPN game id followed by
    # a sequence number, so numeric order is snap order.
    raw = raw.with_columns(play_seq=pl.col("play_id").cast(pl.Decimal(38, 0)))
    raw = presnap_scores(raw)

    df = raw.filter(
        pl.col("play_type").is_in(list(SCRIMMAGE_TYPES))
        & pl.col("down").is_between(1, 4)
        & (pl.col("distance") > 0)
        # Need at least one FBS side; FCS-vs-FCS plays carry no FBS information.
        & ~(pl.col("offense_fcs") & pl.col("defense_fcs"))
    )

    null_report = (
        df.group_by("season")
        .agg(plays=pl.len(), null_ppa=pl.col("ppa").is_null().sum())
        .with_columns(null_share=pl.col("null_ppa") / pl.col("plays"))
        .sort("season")
    )
    for r in null_report.iter_rows(named=True):
        log.info("season %d: %d scrimmage plays, %.2f%% null PPA dropped", r["season"], r["plays"],
                 100 * r["null_share"])

    df = df.filter(pl.col("ppa").is_not_null()).with_columns(success=success_expr())
    df = df.rename({"offense": "offense_team", "defense": "defense_team"}).select(
        "season", "week", "season_type", "game_id", "play_id", "kickoff_ts",
        "offense_team", "defense_team", "offense_fcs", "defense_fcs",
        "offense_conference", "defense_conference", "home_team", "neutral_site",
        "period", "down", "distance", "yards_gained", "margin_at_snap", "play_type", "ppa", "success",
    )
    return df, null_report


def build_prep(cfg: Config) -> dict[str, int]:
    con = duckdb.connect(str(cfg.paths.db_path))
    try:
        scrimmage, null_report = build_scrimmage(con)
        model = apply_filters(scrimmage, cfg)
        con.execute("CREATE OR REPLACE TABLE scrimmage_plays AS SELECT * FROM scrimmage")
        con.execute("CREATE OR REPLACE TABLE model_plays AS SELECT * FROM model")
        con.execute("CREATE OR REPLACE TABLE prep_null_ppa AS SELECT * FROM null_report")
        return {"scrimmage_plays": scrimmage.height, "model_plays": model.height}
    finally:
        con.close()
