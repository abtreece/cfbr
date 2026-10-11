"""Roster-based preseason features (roadmap #1).

For team T in season s, compare T's season-s roster (published preseason)
with player production from season s-1, linked by CFBD player id:

  returner   on T in s-1 and on T's roster in s
  departed   on T in s-1, not on T's roster in s (graduation, NFL, transfer out)
  incoming   on T's roster in s, produced for another team in s-1 (transfer in)

Only season s-1 production and the season-s roster are used, so every feature
is known before week 1 of season s. Incoming transfers are split by whether
they came from an FBS or a non-FBS program.
"""

from __future__ import annotations

import polars as pl

OFF_FEATURES = ["off_ret_share", "off_in_fbs", "off_in_fcs", "off_out", "qb_returns", "qb_in_ppa"]
DEF_FEATURES = ["def_ret_tackles", "def_ret_havoc", "def_in_tackles", "def_in_havoc"]
ROSTER_FEATURES = OFF_FEATURES + DEF_FEATURES


def _havoc(df: pl.DataFrame) -> pl.Expr:
    # Sacks are already counted inside TFL in CFBD's stats, so they're not added again.
    return pl.col("tfl") + pl.col("pd") + pl.col("ints")


def roster_features(season: int, teams: list[str], roster: pl.DataFrame, player_ppa: pl.DataFrame,
                    player_def: pl.DataFrame, team_status: pl.DataFrame) -> pl.DataFrame:
    """One row per team in ``teams``. Defensive features are null when the
    prior season has no defensive player stats (CFBD lacks them before 2016)."""
    ros = roster.filter(pl.col("season") == season).select("player_id", pl.col("team").alias("t_new")).unique("player_id")
    fbs_prev = set(team_status.filter((pl.col("season") == season - 1) & (pl.col("classification") == "fbs"))["team"])
    out = pl.DataFrame({"team": teams})

    # Offense: season s-1 PPA by player.
    off = player_ppa.filter((pl.col("season") == season - 1) & pl.col("plays").is_not_null())
    if off.height:
        off = off.join(ros, on="player_id", how="left")
        team_plays = off.group_by("team").agg(team_plays=pl.col("plays").sum())
        ret = (off.filter(pl.col("t_new") == pl.col("team")).group_by("team")
               .agg(ret_plays=pl.col("plays").sum()))
        gone = (off.filter(pl.col("t_new").is_null() | (pl.col("t_new") != pl.col("team"))).group_by("team")
                .agg(off_out=pl.col("total_ppa").sum()))
        inc = (off.filter(pl.col("t_new").is_not_null() & (pl.col("t_new") != pl.col("team")))
               .with_columns(from_fbs=pl.col("team").is_in(list(fbs_prev)))
               .group_by("t_new").agg(off_in_fbs=pl.col("total_ppa").filter(pl.col("from_fbs")).sum(),
                                      off_in_fcs=pl.col("total_ppa").filter(~pl.col("from_fbs")).sum())
               .rename({"t_new": "team"}))
        # The team's s-1 starting QB = most pass plays for that team.
        qb = (off.filter(pl.col("pass_plays").is_not_null()).sort("pass_plays", descending=True)
              .group_by("team").first().select("team", qb_returns=(pl.col("t_new") == pl.col("team")).fill_null(False).cast(pl.Float64)))
        qb_in = (off.filter(pl.col("t_new").is_not_null() & (pl.col("t_new") != pl.col("team"))
                            & pl.col("pass_plays").is_not_null() & (pl.col("pass_plays") >= 100))
                 .group_by("t_new").agg(qb_in_ppa=pl.col("pass_ppa").max()).rename({"t_new": "team"}))
        out = (out.join(team_plays, on="team", how="left").join(ret, on="team", how="left")
               .join(gone, on="team", how="left").join(inc, on="team", how="left")
               .join(qb, on="team", how="left").join(qb_in, on="team", how="left")
               .with_columns(off_ret_share=pl.col("ret_plays").fill_null(0.0) / pl.col("team_plays"))
               .drop("team_plays", "ret_plays"))
        out = out.with_columns([pl.col(c).fill_null(0.0) for c in ("off_in_fbs", "off_in_fcs", "off_out", "qb_in_ppa")])
    else:
        out = out.with_columns([pl.lit(None, pl.Float64).alias(c) for c in OFF_FEATURES])

    # Defense: season s-1 tackles and havoc plays by player.
    de = player_def.filter(pl.col("season") == season - 1)
    if de.height and de["tackles"].sum() > 0:
        de = de.join(ros, on="player_id", how="left").with_columns(havoc=_havoc(de))
        tot = de.group_by("team").agg(t_tk=pl.col("tackles").sum(), t_hv=pl.col("havoc").sum())
        ret = (de.filter(pl.col("t_new") == pl.col("team")).group_by("team")
               .agg(r_tk=pl.col("tackles").sum(), r_hv=pl.col("havoc").sum()))
        inc = (de.filter(pl.col("t_new").is_not_null() & (pl.col("t_new") != pl.col("team")))
               .group_by("t_new").agg(def_in_tackles=pl.col("tackles").sum(), def_in_havoc=pl.col("havoc").sum())
               .rename({"t_new": "team"}))
        out = (out.join(tot, on="team", how="left").join(ret, on="team", how="left").join(inc, on="team", how="left")
               .with_columns(def_ret_tackles=pl.col("r_tk").fill_null(0.0) / pl.col("t_tk"),
                             def_ret_havoc=pl.col("r_hv").fill_null(0.0) / pl.col("t_hv"),
                             def_in_tackles=pl.col("def_in_tackles").fill_null(0.0),
                             def_in_havoc=pl.col("def_in_havoc").fill_null(0.0))
               .drop("t_tk", "t_hv", "r_tk", "r_hv"))
    else:
        out = out.with_columns([pl.lit(None, pl.Float64).alias(c) for c in DEF_FEATURES])
    return out.select("team", *ROSTER_FEATURES)
