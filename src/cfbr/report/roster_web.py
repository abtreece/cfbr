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
  roster_departures  season, team (the team they left), player_id, name, position,
                     prev_class_year (last season's, 1-5), to_team (this season's roster
                     they appear on, else null), reason (DEPARTURE_REASONS), draft_round,
                     draft_overall (NFL draft held before this season; null if not
                     drafted), draft_nfl_team (city only, as CFBD gives it),
                     draft_nfl_team_name (full franchise name, e.g. "Los Angeles
                     Rams"; null if unknown), in_portal (entered the transfer portal
                     for this season; null before 2021, when CFBD's portal data
                     starts), portal_destination (school listed in the portal, which
                     may be non-FBS), prev_plays, prev_ppa_total, prev_tackles,
                     prev_havoc. Players with no stats last season appear only if
                     drafted. Portal records are linked by name + school (no id).
  roster_careers     one row per player per season of history, for players on this season's
                     roster: season (display season), team (current team), player_id,
                     career_season, career_team, source ("season stats" for completed
                     seasons, "box scores (to date)" for the current season), games,
                     qb_starts (games he led his team in pass attempts, 5+), plays (pass
                     att + carries + receptions), pass_att, rush_car, rec, ppa_total,
                     ppa_avg, pass_ppa_total, pass_ppa_avg, tackles, tfl, sacks, pd,
                     hurries, ints. PPA, sacks and hurries are null for to-date rows;
                     defense is null before 2016; games/qb_starts only count games
                     involving an FBS team. Labels: CAREER_LABELS
  roster_summary     season, team, roster_as_of (UTC time the roster snapshot was fetched),
                     the features in ROSTER_SUMMARY_LABELS, and the incoming-QB display fields
                     in ROSTER_SUMMARY_EXTRA_LABELS

The web view uses the newest roster snapshot (refreshed weekly in-season). The
model's roster prior uses the season's first, fixed snapshot instead, so the
summary features here can differ slightly from the model's inputs.
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
    "qb_in_ppa": ("Proven transfer QB", "Passing PPA total last season of the best incoming transfer QB with 100+ pass attempts; 0 if none (see 'Incoming transfer QB' for any passing history)"),
    "def_ret_tackles": ("Returning tackles", "Share (0–1) of last season's tackles made by players back on this roster"),
    "def_ret_havoc": ("Returning havoc", "Share (0–1) of last season's TFL + passes defended + interceptions by returning players"),
    "def_in_tackles": ("Transfer tackles in", "Count of tackles last season by incoming transfers"),
    "def_in_havoc": ("Transfer havoc in", "Count of TFL + passes defended + interceptions last season by incoming transfers"),
}
assert set(ROSTER_SUMMARY_LABELS) == set(ROSTER_FEATURES)

# Display-only fields: the incoming transfer QB with the most pass attempts last
# season (any number), null if none. Not used by the model.
ROSTER_SUMMARY_EXTRA_LABELS: dict[str, tuple[str, str]] = {
    "in_qb_name": ("Incoming transfer QB", "Incoming transfer QB with the most pass attempts last season (any number)"),
    "in_qb_from": ("From", "School he played for last season"),
    "in_qb_pass_att": ("Pass attempts", "His pass attempts last season (count)"),
    "in_qb_pass_ppa_avg": ("PPA per pass", "His passing PPA per attempt last season"),
    "in_qb_pass_ppa_total": ("Passing PPA", "His total passing PPA last season"),
}

# Extra roster_players columns: experience before this season.
PLAYER_EXTRA_LABELS: dict[str, tuple[str, str]] = {
    "career_seasons": ("Seasons played", "Prior seasons with any recorded stats (count)"),
    "career_pass_att": ("Career pass attempts", "Pass attempts in prior seasons, all schools (count)"),
    "career_qb_starts": ("Career QB starts", "Prior-season games he led his team in pass attempts (5+), FBS-involved games only"),
}

CAREER_LABELS: dict[str, tuple[str, str]] = {
    "career_season": ("Season", "Season the row describes"),
    "career_team": ("Team", "School he played for that season"),
    "source": ("Source", "'season stats' (completed season) or 'box scores (to date)' (current season)"),
    "games": ("G", "Games with a recorded stat (FBS-involved games only)"),
    "qb_starts": ("QB starts", "Games he led his team in pass attempts (5+)"),
    "plays": ("Plays", "Pass attempts + carries + receptions"),
    "pass_att": ("Pass att", "Pass attempts"),
    "rush_car": ("Car", "Carries"),
    "rec": ("Rec", "Receptions"),
    "ppa_total": ("PPA", "Total expected points added (offense)"),
    "ppa_avg": ("PPA/play", "Expected points added per play"),
    "pass_ppa_total": ("Pass PPA", "Total passing PPA"),
    "pass_ppa_avg": ("Pass PPA/att", "Passing PPA per attempt"),
    "tackles": ("Tkl", "Total tackles"),
    "tfl": ("TFL", "Tackles for loss"),
    "sacks": ("Sacks", "Sacks"),
    "pd": ("PD", "Passes defended"),
    "hurries": ("Hur", "QB hurries"),
    "ints": ("INT", "Interceptions"),
}

# roster_departures.reason, first match wins in this order.
DEPARTURE_REASONS: dict[str, str] = {
    "drafted": "Selected in the NFL draft held before this season",
    "transfer": "On another FBS roster this season",
    "portal_transfer": "Entered the transfer portal with a new school listed, but is not on an FBS roster here (often a non-FBS school)",
    "portal_no_school": "Entered the transfer portal; no new school listed",
    "unknown": "No record: graduation, undrafted, left football, or missing from CFBD's rosters",
}

PLAYER_COLS = ["player_id", "name", "jersey", "position", "class_year", "height", "weight", "home_city", "home_state"]


def _prev_production(ds: Dataset, season: int) -> pl.DataFrame:
    off = ds.player_ppa.filter(pl.col("season") == season - 1).select(
        "player_id", prev_team="team", prev_plays="plays", prev_ppa_total="total_ppa", prev_ppa_avg="avg_ppa",
        prev_pass_plays="pass_plays", prev_pass_ppa_total="pass_ppa", prev_pass_ppa_avg="avg_pass_ppa")
    de = ds.player_defense.filter(pl.col("season") == season - 1).select(
        "player_id", prev_team_d="team", prev_tackles="tackles",
        prev_havoc=pl.col("tfl") + pl.col("pd") + pl.col("ints"))
    both = off.join(de, on="player_id", how="full", coalesce=True)
    return both.with_columns(prev_team=pl.coalesce("prev_team", "prev_team_d")).drop("prev_team_d")


def _name_key(col: str) -> pl.Expr:
    """Loose name match key: lowercase letters only, suffixes (Jr., III, ...) dropped."""
    return (pl.col(col).str.to_lowercase().str.replace_all(r"\b(jr|sr|ii|iii|iv|v)\b", "")
            .str.replace_all(r"[^a-z]", ""))


def _departures(ds: Dataset, season: int, prev: pl.DataFrame, ros: pl.DataFrame) -> pl.DataFrame:
    """Last season's players not on their team's roster now, with why they left
    when CFBD records it: the NFL draft (linked by player id) or the transfer portal
    (linked by name + school, as portal records carry no id)."""
    prev_ros = (ds.roster.filter(pl.col("season") == season - 1).unique("player_id", keep="first")
                .select("player_id", "name", "position", prev_class_year="class_year", roster_team="team"))
    draft = getattr(ds, "draft_picks", None)
    if draft is not None and "nfl_team_name" not in draft.columns:  # store built before the column existed
        draft = draft.with_columns(nfl_team_name=pl.lit(None, pl.Utf8))
    draft = (draft.filter((pl.col("season") == season) & pl.col("player_id").is_not_null())
             .unique("player_id", keep="first")
             .select("player_id", draft_college_team="college_team", draft_round="round", draft_overall="overall",
                     draft_nfl_team="nfl_team", draft_nfl_team_name="nfl_team_name")
             if draft is not None else pl.DataFrame(schema={"player_id": pl.Utf8, "draft_college_team": pl.Utf8,
                                                           "draft_round": pl.Int32, "draft_overall": pl.Int32,
                                                           "draft_nfl_team": pl.Utf8,
                                                           "draft_nfl_team_name": pl.Utf8}))
    # Drafted players with no stats last season (mostly offensive linemen) are
    # departures too; place them on last season's roster team.
    no_stats = (draft.join(prev.select("player_id"), on="player_id", how="anti")
                .join(prev_ros.select("player_id", "roster_team"), on="player_id", how="left")
                .select("player_id", prev_team=pl.coalesce("roster_team", "draft_college_team")))
    base = pl.concat([prev, no_stats], how="diagonal_relaxed")
    out = (base.join(ros.select("player_id", to_team="team"), on="player_id", how="left")
           .filter(pl.col("to_team").is_null() | (pl.col("to_team") != pl.col("prev_team")))
           .join(prev_ros.drop("roster_team"), on="player_id", how="left")
           .join(draft.drop("draft_college_team"), on="player_id", how="left")
           .with_columns(drafted=pl.col("draft_overall").is_not_null(), _key=_name_key("name")))
    portal = getattr(ds, "portal", None)
    if portal is not None and portal.filter(pl.col("season") == season).height:
        entries = (portal.filter(pl.col("season") == season)
                   .with_columns(_key=_name_key("name"))
                   .sort("transfer_date", nulls_last=False)
                   .group_by("_key", "origin", maintain_order=True).last()  # latest entry if listed twice
                   .select("_key", prev_team="origin", portal_destination="destination", in_portal=pl.lit(True)))
        out = out.join(entries, on=["_key", "prev_team"], how="left").with_columns(
            in_portal=pl.col("in_portal").fill_null(False))
    else:
        out = out.with_columns(in_portal=pl.lit(None, pl.Boolean), portal_destination=pl.lit(None, pl.Utf8))
    reason = (pl.when(pl.col("drafted")).then(pl.lit("drafted"))
              .when(pl.col("to_team").is_not_null()).then(pl.lit("transfer"))
              .when(pl.col("portal_destination").is_not_null()).then(pl.lit("portal_transfer"))
              .when(pl.col("in_portal")).then(pl.lit("portal_no_school"))
              .otherwise(pl.lit("unknown")))
    return out.select(pl.lit(season, pl.Int32).alias("season"), pl.col("prev_team").alias("team"), "player_id",
                      "name", "position", "prev_class_year", "to_team", reason.alias("reason"), "draft_round",
                      "draft_overall", "draft_nfl_team", "draft_nfl_team_name", "in_portal", "portal_destination", "prev_plays",
                      "prev_ppa_total", "prev_tackles", "prev_havoc")


def _display_roster(ds: Dataset, season: int) -> pl.DataFrame:
    """Newest roster snapshot for the season, falling back to the fixed one."""
    latest = getattr(ds, "roster_latest", None)
    if latest is not None and all(c in latest.columns for c in PLAYER_COLS):
        r = latest.filter(pl.col("season") == season)
        if r.height:
            return r
    return ds.roster.filter(pl.col("season") == season)


def roster_as_of(cfg: Config, season: int, ds: Dataset | None = None) -> datetime | None:
    latest = getattr(ds, "roster_latest", None) if ds is not None else None
    if latest is not None and "as_of" in latest.columns:
        r = latest.filter(pl.col("season") == season)
        if r.height:
            return r["as_of"][0]
    p = cfg.paths.raw_dir / "roster" / str(season) / "season.json"
    return datetime.fromtimestamp(p.stat().st_mtime, UTC) if p.exists() else None


def roster_tables(ds: Dataset, season: int, as_of: datetime | None = None) -> dict[str, pl.DataFrame]:
    if ds.roster is None or ds.player_ppa is None or ds.player_defense is None:
        return {}
    if any(c not in ds.roster.columns for c in PLAYER_COLS):
        return {}  # store built before the roster table gained names etc.; rebuild with `cfbr ingest`
    ros = _display_roster(ds, season)
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

    departures = _departures(ds, season, prev, ros).filter(pl.col("team").is_in(fbs)).sort("team", "name")

    # Summary from the displayed snapshot (same as the player list). Previous
    # seasons' rosters come from the fixed table.
    display_roster = pl.concat([ds.roster.filter(pl.col("season") != season), ros.select(ds.roster.columns)])
    summary = roster_features(season, teams, display_roster, ds.player_ppa, ds.player_defense, ds.team_status)
    in_qb = (players.filter((pl.col("status") == "transfer_in") & (pl.col("position") == "QB")
                            & (pl.col("prev_pass_plays") > 0))
             .sort("prev_pass_plays", descending=True).group_by("team", maintain_order=True).first()
             .select("team", in_qb_name="name", in_qb_from="from_team", in_qb_pass_att="prev_pass_plays",
                     in_qb_pass_ppa_avg="prev_pass_ppa_avg", in_qb_pass_ppa_total="prev_pass_ppa_total"))
    summary = (summary.join(in_qb, on="team", how="left")
               .with_columns(season=pl.lit(season, pl.Int32), roster_as_of=pl.lit(as_of, pl.Datetime("us", "UTC")))
               .select("season", "team", "roster_as_of", *ROSTER_FEATURES, *ROSTER_SUMMARY_EXTRA_LABELS).sort("team"))
    careers = career_tables(ds, season, players)
    players = (players.join(experience(careers, season), on="player_id", how="left")
               .with_columns(career_seasons=pl.col("career_seasons").fill_null(0),
                             career_pass_att=pl.col("career_pass_att").fill_null(0.0),
                             career_qb_starts=pl.col("career_qb_starts").fill_null(0.0)))
    return {"roster_players": players, "roster_departures": departures, "roster_summary": summary,
            "roster_careers": careers}


def _qb_starts(gp: pl.DataFrame) -> pl.DataFrame:
    """Per (season, player_id): games played and games he led his team in pass attempts (5+)."""
    lead = gp.with_columns(top=pl.col("pass_att").max().over("game_id", "team"))
    return lead.group_by("season", "player_id").agg(
        games=pl.col("game_id").n_unique().cast(pl.Float64),
        qb_starts=((pl.col("pass_att") == pl.col("top")) & (pl.col("pass_att") >= 5)).sum().cast(pl.Float64))


def career_tables(ds: Dataset, season: int, players: pl.DataFrame) -> pl.DataFrame:
    """Season-by-season history (all schools) for players on this season's roster."""
    ids = players.select("player_id", "team")
    ppa = ds.player_ppa.with_columns([pl.lit(None, pl.Float64).alias(c) for c in ("rush_car", "rec")
                                      if c not in ds.player_ppa.columns])  # stores built before these existed
    off = ppa.filter(pl.col("season") < season).select(
        "season", "player_id", career_team="team", plays="plays", pass_att="pass_plays", rush_car="rush_car",
        rec="rec", ppa_total="total_ppa", ppa_avg="avg_ppa", pass_ppa_total="pass_ppa", pass_ppa_avg="avg_pass_ppa")
    de = (ds.player_defense
          .filter((pl.col("season") < season) & ((pl.col("tackles") + pl.col("pd") + pl.col("ints")) > 0))
          .select("season", "player_id", "tackles", "tfl", "sacks", "pd", "hurries", "ints", career_team_d="team"))
    hist = (off.join(de, on=["season", "player_id"], how="full", coalesce=True)
            .with_columns(career_team=pl.coalesce("career_team", "career_team_d"), source=pl.lit("season stats"))
            .drop("career_team_d"))
    gp = getattr(ds, "game_players", None)
    if gp is not None and gp.height:
        starts = _qb_starts(gp)
        hist = hist.join(starts, on=["season", "player_id"], how="left")
        cur = gp.filter(pl.col("season") == season)
        if cur.height:
            to_date = (cur.group_by("season", "player_id").agg(
                career_team=pl.col("team").last(), pass_att=pl.col("pass_att").sum(), rush_car=pl.col("rush_att").sum(),
                rec=pl.col("rec").sum(), tackles=pl.col("tackles").sum(), tfl=pl.col("tfl").sum(),
                pd=pl.col("pd").sum(), ints=pl.col("ints").sum())
                .with_columns(plays=pl.col("pass_att") + pl.col("rush_car") + pl.col("rec"),
                              source=pl.lit("box scores (to date)"))
                .join(starts, on=["season", "player_id"], how="left"))
            hist = pl.concat([hist, to_date], how="diagonal_relaxed")
    else:
        hist = hist.with_columns(games=pl.lit(None, pl.Float64), qb_starts=pl.lit(None, pl.Float64))
    cols = [c for c in CAREER_LABELS if c != "career_season"]
    out = (hist.rename({"season": "career_season"}).join(ids, on="player_id", how="inner")
           .with_columns(season=pl.lit(season, pl.Int32)))
    for c in cols:
        if c not in out.columns:
            out = out.with_columns(pl.lit(None, pl.Float64).alias(c))
    return out.select("season", "team", "player_id", "career_season", *cols).sort("team", "player_id", "career_season")


def experience(careers: pl.DataFrame, season: int) -> pl.DataFrame:
    """Experience before ``season`` per player, from his career rows."""
    prior = careers.filter(pl.col("career_season") < season)
    return prior.group_by("player_id").agg(
        career_seasons=pl.col("career_season").n_unique().cast(pl.Int32),
        career_pass_att=pl.col("pass_att").fill_null(0).sum(),
        career_qb_starts=pl.col("qb_starts").fill_null(0).sum())


def write_roster_web(cfg: Config, ds: Dataset, season: int, out_dir: Path | None = None) -> dict[str, Path]:
    """Write the roster parquet files (players, departures, summary, careers) to out/web/{season}/ (temp file +
    rename). Returns {} and writes nothing when the season has no roster."""
    tables = roster_tables(ds, season, roster_as_of(cfg, season, ds))
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
