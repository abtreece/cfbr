"""Roster data for the web view: who's on each roster, where they came from,
who left, and the team-level roster summary.

Linking is by CFBD player id between the season's roster and the previous
season's player production (at any school). Rosters are imperfect: players
without box-score production look like "new", and CFBD's roster for a season
may be a later snapshot. Facts and model inputs only; no predictions.

No API calls: everything comes from the Dataset already loaded. A season with
no roster rows writes nothing.

Columns (null = unknown):
  roster_players     season, team (CFBD school name, FBS only), player_id, name, jersey,
                     position, class_year (1-5), height (inches), weight (lbs), home_city,
                     home_state, status ("returner" | "transfer_in" | "new"), from_team
                     (transfer_in only), prev_plays, prev_ppa_total, prev_ppa_avg (per play),
                     prev_pass_plays, prev_pass_ppa_total, prev_pass_ppa_avg, prev_tackles,
                     prev_havoc (TFL + passes defended + interceptions) -- all "prev_" values
                     are last season's, at whatever school the player was on
  roster_departures  season, team (the team they left), player_id, name, position, to_team
                     (this season's roster they appear on, else null), prev_plays,
                     prev_ppa_total, prev_tackles, prev_havoc
  roster_summary     season, team, roster_as_of (UTC time the roster was fetched), and the
                     features in ROSTER_SUMMARY_LABELS
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.model.roster import ROSTER_FEATURES, roster_features

# feature -> (label, definition with unit). Null when the prior season has no
# data (e.g. no defensive player stats before 2016, programs new to FBS).
ROSTER_SUMMARY_LABELS: dict[str, tuple[str, str]] = {
    "off_ret_share": ("Returning offense", "Share (0–1) of last season's offensive plays made by players back on this roster"),
    "off_in_fbs": ("Transfer offense in (FBS)", "Total PPA (expected points added) last season by incoming transfers from FBS programs"),
    "off_in_fcs": ("Transfer offense in (non-FBS)", "Total PPA last season by incoming transfers from non-FBS programs"),
    "off_out": ("Offense lost", "Total PPA last season by players from this team who are not on this roster"),
    "qb_returns": ("Starting QB back", "1 if last season's starting QB (most pass plays) is on this roster, else 0"),
    "qb_in_ppa": ("Transfer QB", "Passing PPA total last season of the best incoming transfer QB (100+ pass plays); 0 if none"),
    "def_ret_tackles": ("Returning tackles", "Share (0–1) of last season's tackles made by players back on this roster"),
    "def_ret_havoc": ("Returning havoc", "Share (0–1) of last season's TFL + passes defended + interceptions by returning players"),
    "def_in_tackles": ("Transfer tackles in", "Count of tackles last season by incoming transfers"),
    "def_in_havoc": ("Transfer havoc in", "Count of TFL + passes defended + interceptions last season by incoming transfers"),
}
assert set(ROSTER_SUMMARY_LABELS) == set(ROSTER_FEATURES)

PLAYER_COLS = ["player_id", "name", "jersey", "position", "class_year", "height", "weight", "home_city", "home_state"]


def _prev_production(ds: Dataset, season: int) -> pl.DataFrame:
    off = ds.player_ppa.filter(pl.col("season") == season - 1).select(
        "player_id", prev_team="team", prev_plays="plays", prev_ppa_total="total_ppa",
        prev_pass_plays="pass_plays", prev_pass_ppa_total="pass_ppa")
    de = ds.player_defense.filter(pl.col("season") == season - 1).select(
        "player_id", prev_team_d="team", prev_tackles="tackles",
        prev_havoc=pl.col("tfl") + pl.col("pd") + pl.col("ints"))
    both = off.join(de, on="player_id", how="full", coalesce=True)
    avg = lambda tot, n: pl.when(pl.col(n) > 0).then(pl.col(tot) / pl.col(n))  # noqa: E731
    return both.with_columns(
        prev_team=pl.coalesce("prev_team", "prev_team_d"),
        prev_ppa_avg=avg("prev_ppa_total", "prev_plays"),
        prev_pass_ppa_avg=avg("prev_pass_ppa_total", "prev_pass_plays"),
    ).drop("prev_team_d")


def roster_as_of(cfg: Config, season: int) -> datetime | None:
    p = cfg.paths.raw_dir / "roster" / str(season) / "season.json"
    return datetime.fromtimestamp(p.stat().st_mtime, UTC) if p.exists() else None


def roster_tables(ds: Dataset, season: int, as_of: datetime | None = None) -> dict[str, pl.DataFrame]:
    if ds.roster is None or ds.player_ppa is None or ds.player_defense is None:
        return {}
    if any(c not in ds.roster.columns for c in PLAYER_COLS):
        return {}  # store built before the roster table gained names etc.; rebuild with `cfbr ingest`
    ros = ds.roster.filter(pl.col("season") == season)
    if ros.height == 0:
        return {}
    teams = ds.fbs_teams(season)
    fbs = list(set(teams))
    ros = ros.unique("player_id", keep="first").select("season", "team", *PLAYER_COLS)
    prev = _prev_production(ds, season)
    players = (ros.join(prev, on="player_id", how="left")
               .with_columns(status=pl.when(pl.col("prev_team").is_null()).then(pl.lit("new"))
                             .when(pl.col("prev_team") == pl.col("team")).then(pl.lit("returner"))
                             .otherwise(pl.lit("transfer_in")))
               .with_columns(from_team=pl.when(pl.col("status") == "transfer_in").then(pl.col("prev_team")))
               .select("season", "team", *PLAYER_COLS, "status", "from_team", "prev_plays", "prev_ppa_total",
                       "prev_ppa_avg", "prev_pass_plays", "prev_pass_ppa_total", "prev_pass_ppa_avg",
                       "prev_tackles", "prev_havoc")
               .filter(pl.col("team").is_in(fbs)).sort("team", "position", "name"))

    prev_names = (ds.roster.filter(pl.col("season") == season - 1).unique("player_id", keep="first")
                  .select("player_id", "name", "position"))
    departures = (prev.join(ros.select("player_id", to_team="team"), on="player_id", how="left")
                  .filter(pl.col("to_team").is_null() | (pl.col("to_team") != pl.col("prev_team")))
                  .join(prev_names, on="player_id", how="left")
                  .select(pl.lit(season, pl.Int32).alias("season"), pl.col("prev_team").alias("team"), "player_id",
                          "name", "position", "to_team", "prev_plays", "prev_ppa_total", "prev_tackles", "prev_havoc")
                  .filter(pl.col("team").is_in(fbs)).sort("team", "name"))

    summary = roster_features(season, teams, ds.roster, ds.player_ppa, ds.player_defense, ds.team_status)
    summary = summary.with_columns(season=pl.lit(season, pl.Int32),
                                   roster_as_of=pl.lit(as_of, pl.Datetime("us", "UTC")))
    summary = summary.select("season", "team", "roster_as_of", *ROSTER_FEATURES).sort("team")
    return {"roster_players": players, "roster_departures": departures, "roster_summary": summary}


def write_roster_web(cfg: Config, ds: Dataset, season: int, out_dir: Path | None = None) -> dict[str, Path]:
    """Write the three roster parquet files to out/web/{season}/ (temp file +
    rename). Returns {} and writes nothing when the season has no roster."""
    tables = roster_tables(ds, season, roster_as_of(cfg, season))
    if not tables:
        return {}
    out = out_dir or (cfg.paths.out_dir / "web" / str(season))
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in tables.items():
        p = out / f"{name}.parquet"
        tmp = out / f".{name}.parquet.tmp"
        df.write_parquet(tmp)
        tmp.replace(p)
        paths[name] = p
    return paths
