"""Team-specific home field from the model's own residuals, point-in-time
(DECISIONS "Team-specific home field").

For a game in slot w of season S, team T's estimate

  delta_T = mean residual in T's home games - mean residual in T's away games

over T's completed games kicked off before slot w's as_of, in seasons
max(FIRST, S - LOOKBACK)..S, neutral sites left out; 0 below MIN_GAMES home or
away games, and for pooled FCS sides. A team's rating is fit on home and away
games together and soaks up about half the gap, hence the game feature

  x = delta_home * (home - 0.5) + 0.5 * delta_away     (home = 0 at a neutral site)

and adjusted margin = pred + c * x, c by OLS with no intercept.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cfbr.prep.plays import FCS

FIRST = 2015
LOOKBACK = 4
MIN_GAMES = 10


def team_games(preds: pl.DataFrame) -> pl.DataFrame:
    """One row per (completed, non-neutral game, FBS team): residual from that team's side."""
    d = preds.filter(pl.col("completed") & pl.col("margin").is_not_null() & pl.col("pred_margin").is_not_null()
                     & ~pl.col("neutral_site"))
    resid = pl.col("margin") - pl.col("pred_margin")
    cols = ["season", "kickoff_ts"]
    out = pl.concat([d.select(*cols, team=pl.col("home"), is_home=pl.lit(True), resid=resid),
                     d.select(*cols, team=pl.col("away"), is_home=pl.lit(False), resid=-resid)])
    return out.filter(~pl.col("team").str.starts_with(FCS))


def _gaps(h: pl.DataFrame) -> pl.DataFrame:
    """delta per team over the team-game rows ``h`` (teams under the minimum omitted)."""
    agg = h.group_by("team").agg(
        n_home=pl.col("is_home").sum(), n_away=(~pl.col("is_home")).sum(),
        home=pl.col("resid").filter(pl.col("is_home")).mean(), away=pl.col("resid").filter(~pl.col("is_home")).mean())
    return (agg.filter((pl.col("n_home") >= MIN_GAMES) & (pl.col("n_away") >= MIN_GAMES))
            .select("team", delta=pl.col("home") - pl.col("away")))


def team_gaps(tg: pl.DataFrame, season: int, as_of) -> pl.DataFrame:
    """delta per team from games before ``as_of`` in the lookback window."""
    return _gaps(tg.filter(pl.col("season").is_between(max(FIRST, season - LOOKBACK), season)
                           & (pl.col("kickoff_ts") < as_of)))


def attach(preds: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """Add delta_h, delta_a and the game feature x to prediction rows (each slot sees only its past)."""
    parts = []
    for (season, as_of), g in preds.group_by("season", "as_of"):
        gaps = team_gaps(tg, season, as_of)
        parts.append(g.join(gaps.rename({"team": "home", "delta": "delta_h"}), on="home", how="left")
                     .join(gaps.rename({"team": "away", "delta": "delta_a"}), on="away", how="left"))
    out = pl.concat(parts).with_columns(pl.col("delta_h", "delta_a").fill_null(0.0))
    home = (~pl.col("neutral_site")).cast(pl.Float64)
    return out.with_columns(x=pl.col("delta_h") * (home - 0.5) + 0.5 * pl.col("delta_a")).sort("kickoff_ts", "game_id")


def fit_c(rows: pl.DataFrame) -> float:
    """OLS (no intercept) of the model's residual on x."""
    x = rows["x"].to_numpy()
    y = (rows["margin"] - rows["pred_margin"]).to_numpy()
    return float(x @ y / (x @ x)) if x @ x > 0 else 0.0


def reliability(tg: pl.DataFrame, block_a: tuple[int, int], block_b: tuple[int, int]) -> dict:
    """Correlation of each team's home-minus-away residual gap across two disjoint season blocks."""
    a, b = (_gaps(tg.filter(pl.col("season").is_between(*blk))) for blk in (block_a, block_b))
    j = a.join(b, on="team", suffix="_b")
    r = float(np.corrcoef(j["delta"].to_numpy(), j["delta_b"].to_numpy())[0, 1]) if j.height > 2 else float("nan")
    return {"n_teams": j.height, "r": r}
