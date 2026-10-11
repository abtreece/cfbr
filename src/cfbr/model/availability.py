"""Availability features derived from CFBD box scores (who played), point-in-time.

For team T's k-th game of a season, using only T's games 1..k-1:

  qb_out        T's primary QB (most pass attempts so far, >= 20) did not lead T
                in pass attempts in game k-1 (a backup started last time out)
  skill_missing share of T's season touches (carries + receptions) so far by
                players who appeared in 2+ of games 1..k-1 but not in game k-1
  def_missing   the same share for tackles
  has_info      1 if T has at least one earlier game this season

"Played" means the player has a box-score stat, so players without stats
(offensive linemen, most special teamers) are invisible. Gamesmanship can't
touch these features: they come from what happened on the field, not reports.
"""

from __future__ import annotations

import numpy as np
import polars as pl

FEATURES = ["qb_out", "skill_missing", "def_missing"]
MIN_PRIMARY_ATT = 20
MIN_LAST_ATT = 5


def _team_features(gp: pl.DataFrame) -> list[dict]:
    """``gp``: one team-season's player-game rows, with a ``k`` game index (0-based, kickoff order)."""
    out = []
    games = sorted(gp["k"].unique().to_list())
    by_k = {k: gp.filter(pl.col("k") == k) for k in games}
    gid = {k: by_k[k]["game_id"][0] for k in games}
    for k in games:
        row = {"game_id": gid[k], "qb_out": 0.0, "skill_missing": 0.0, "def_missing": 0.0, "has_info": 0.0}
        prior = [j for j in games if j < k]
        if prior:
            row["has_info"] = 1.0
            hist = pl.concat([by_k[j] for j in prior])
            last = by_k[prior[-1]]
            last_ids = set(last["player_id"])
            # QB
            cum = hist.group_by("player_id").agg(att=pl.col("pass_att").sum()).sort("att", descending=True)
            lq = last.sort("pass_att", descending=True)
            if cum.height and cum["att"][0] >= MIN_PRIMARY_ATT and lq.height and lq["pass_att"][0] >= MIN_LAST_ATT:
                row["qb_out"] = float(lq["player_id"][0] != cum["player_id"][0])
            # regulars who missed the last game
            app = hist.group_by("player_id").agg(
                n=pl.len(), touches=(pl.col("rush_att") + pl.col("rec")).sum(), tackles=pl.col("tackles").sum())
            missing = app.filter((pl.col("n") >= 2) & ~pl.col("player_id").is_in(list(last_ids)))
            tt, tk = app["touches"].sum(), app["tackles"].sum()
            row["skill_missing"] = float(missing["touches"].sum() / tt) if tt else 0.0
            row["def_missing"] = float(missing["tackles"].sum() / tk) if tk else 0.0
        out.append(row)
    return out


def availability_features(game_players: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    """Per (game_id, team) features. ``games`` needs game_id and kickoff_ts."""
    gp = game_players.join(games.select("game_id", "kickoff_ts"), on="game_id", how="inner")
    order = (gp.select("season", "team", "game_id", "kickoff_ts").unique()
             .sort("kickoff_ts").with_columns(k=pl.int_range(pl.len()).over("season", "team")))
    gp = gp.join(order.select("game_id", "team", "k"), on=["game_id", "team"])
    rows = []
    for (season, team), g in gp.group_by("season", "team"):
        for r in _team_features(g):
            rows.append({"season": season, "team": team, **r})
    return pl.DataFrame(rows, schema={"season": pl.Int32, "team": pl.Utf8, "game_id": pl.Int64, "qb_out": pl.Float64,
                                      "skill_missing": pl.Float64, "def_missing": pl.Float64, "has_info": pl.Float64})


def attach(preds: pl.DataFrame, feats: pl.DataFrame) -> pl.DataFrame:
    """Add home-minus-away feature differences (d_qb_out, ...) to prediction rows."""
    f = feats.select("game_id", "team", *FEATURES)
    out = (preds.join(f.rename({c: f"{c}_h" for c in FEATURES}), left_on=["game_id", "home_team"],
                      right_on=["game_id", "team"], how="left")
           .join(f.rename({c: f"{c}_a" for c in FEATURES}), left_on=["game_id", "away_team"],
                 right_on=["game_id", "team"], how="left"))
    return out.with_columns([(pl.col(f"{c}_h").fill_null(0.0) - pl.col(f"{c}_a").fill_null(0.0)).alias(f"d_{c}")
                             for c in FEATURES])


def fit_adjustment(rows: pl.DataFrame) -> np.ndarray:
    """OLS (no intercept) of the frozen model's residual on the feature differences."""
    X = rows.select([f"d_{c}" for c in FEATURES]).to_numpy()
    y = (rows["margin"] - rows["pred_margin"]).to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def apply_adjustment(rows: pl.DataFrame, coef: np.ndarray) -> pl.Series:
    X = rows.select([f"d_{c}" for c in FEATURES]).to_numpy()
    return rows["pred_margin"] + pl.Series(X @ coef)
