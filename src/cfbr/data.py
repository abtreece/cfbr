"""Load modeling inputs from DuckDB into memory once per run."""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import polars as pl

from cfbr.config import Config
from cfbr.ingest.store import POLL_SCHEMA
from cfbr.prep.plays import FCS, apply_filters

POSTSEASON_BUCKET_DAYS = 7

GAMES_SQL = """
SELECT g.game_id, g.season, g.week, g.season_type, g.kickoff_ts, g.completed, g.neutral_site,
       g.home_team, g.away_team, g.home_points, g.away_points,
       coalesce(hs.classification, 'unknown') <> 'fbs' AS home_fcs,
       coalesce(aws.classification, 'unknown') <> 'fbs' AS away_fcs,
       hs.conference AS home_conference, aws.conference AS away_conference
FROM games g
LEFT JOIN team_status hs ON hs.season = g.season AND hs.team = g.home_team
LEFT JOIN team_status aws ON aws.season = g.season AND aws.team = g.away_team
"""


def assign_slots(games: pl.DataFrame) -> pl.DataFrame:
    """Prediction slot per game: ``r{week}`` in the regular season and
    ``p{n}`` for 7-day postseason buckets (so CFP games after the first bowl
    week can use earlier bowl results). Adds ``slot`` and ``slot_order``."""
    reg = games.filter(pl.col("season_type") == "regular").with_columns(
        slot=pl.format("r{}", pl.col("week").cast(pl.Utf8).str.zfill(2)),
        slot_order=pl.col("week"),
    )
    post = games.filter(pl.col("season_type") == "postseason")
    if post.height:
        first = post.group_by("season").agg(first_post=pl.col("kickoff_ts").min())
        post = post.join(first, on="season").with_columns(
            bucket=((pl.col("kickoff_ts") - pl.col("first_post")).dt.total_seconds()
                    // (POSTSEASON_BUCKET_DAYS * 86400)).cast(pl.Int32) + 1
        ).with_columns(
            slot=pl.format("p{}", pl.col("bucket")),
            slot_order=100 + pl.col("bucket"),
        ).drop("first_post", "bucket")
    else:
        post = post.with_columns(slot=pl.lit(None, pl.Utf8), slot_order=pl.lit(None, pl.Int32))
    return pl.concat([reg, post.select(reg.columns)])


@dataclass
class Dataset:
    scrimmage: pl.DataFrame       # all scrimmage plays, unfiltered
    games: pl.DataFrame           # FBS-involved games with slots and pooled names
    team_status: pl.DataFrame
    returning: pl.DataFrame
    talent: pl.DataFrame
    recruiting: pl.DataFrame
    lines: pl.DataFrame
    polls: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=POLL_SCHEMA))
    roster: pl.DataFrame | None = None
    player_ppa: pl.DataFrame | None = None
    player_defense: pl.DataFrame | None = None
    roster_latest: pl.DataFrame | None = None

    @classmethod
    def load(cls, cfg: Config) -> "Dataset":
        con = duckdb.connect(str(cfg.paths.db_path), read_only=True)
        try:
            scrimmage = con.execute("SELECT * FROM scrimmage_plays").pl()
            games = con.execute(GAMES_SQL).pl()
            team_status = con.execute("SELECT * FROM team_status").pl()
            returning = con.execute("SELECT * FROM returning_production").pl()
            talent = con.execute("SELECT * FROM talent").pl()
            recruiting = con.execute("SELECT * FROM recruiting").pl()
            lines = con.execute("SELECT * FROM lines").pl()
            has_polls = con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'polls'").fetchone()[0]
            polls = con.execute("SELECT * FROM polls").pl() if has_polls else pl.DataFrame(schema=POLL_SCHEMA)
            tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
            player = {t: (con.execute(f"SELECT * FROM {t}").pl() if t in tables else None)
                      for t in ("roster", "player_ppa", "player_defense", "roster_latest")}
        finally:
            con.close()
        games = games.filter(~(pl.col("home_fcs") & pl.col("away_fcs")))
        return cls(scrimmage, assign_slots(games), team_status, returning, talent, recruiting, lines, polls,
                   player["roster"], player["player_ppa"], player["player_defense"], player["roster_latest"])

    def model_plays(self, cfg: Config) -> pl.DataFrame:
        return apply_filters(self.scrimmage, cfg)

    def pooled_games(self, cfg: Config) -> pl.DataFrame:
        def pooled(team: str, fcs: str, conf: str) -> pl.Expr:
            label = (pl.lit(FCS) if cfg.prep.fcs_pooling == "single"
                     else pl.lit(FCS + ":") + pl.col(conf).fill_null("Unknown"))
            return pl.when(pl.col(fcs)).then(label).otherwise(pl.col(team))

        return self.games.with_columns(
            home=pooled("home_team", "home_fcs", "home_conference"),
            away=pooled("away_team", "away_fcs", "away_conference"),
            margin=(pl.col("home_points") - pl.col("away_points")).cast(pl.Float64),
        )

    def fbs_teams(self, season: int) -> list[str]:
        return sorted(self.team_status.filter((pl.col("season") == season) & (pl.col("classification") == "fbs"))
                      ["team"].to_list())


def slot_as_of(games: pl.DataFrame) -> pl.DataFrame:
    """As-of time per (season, slot): the earliest kickoff among that slot's games."""
    return games.group_by("season", "slot", "slot_order").agg(as_of=pl.col("kickoff_ts").min()).sort(
        "season", "slot_order")


def training_slice(df: pl.DataFrame, season: int, as_of, ts_col: str = "kickoff_ts") -> pl.DataFrame:
    """The leakage guard: rows from ``season`` with kickoff strictly before ``as_of``."""
    return df.filter((pl.col("season") == season) & (pl.col(ts_col) < as_of))

