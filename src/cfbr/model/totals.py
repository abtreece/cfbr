"""Point totals (and team points), point-in-time, per the rule in DECISIONS.md.

For each prediction slot (same as-of times as the margin model):
  - pace: ridge on team-game offensive play counts, plays = mu + Poff[team] + Pdef[opp],
    shrunk toward half of last season's final pace effects (4-game prior);
  - league points per team-game so far this season (last season's if none yet);
  - efficiency matchups from the margin model's point-in-time ratings.
Team points = OLS on [1, Δppa, Δsr, expected plays, home, league ppg], fit on
training seasons under the hybrid scheme. Total = home + away points.

Baseline: points-for / points-against per game so far, shrunk 50% toward the
league average with a 4-game prior.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cfbr.model.engine import Engine
from cfbr.model.schemes import training_seasons

PACE_PRIOR_GAMES = 4.0
PACE_PRIOR_SHRINK = 0.5
BASELINE_PRIOR_GAMES = 4.0
FEATURES = ["d_ppa", "d_sr", "exp_plays", "home", "lg_ppg"]


def team_game_plays(eng: Engine, season: int) -> pl.DataFrame:
    """Offensive scrimmage plays per (game, offense team), with kickoff and opponent."""
    p = eng.season_plays(season)
    return (p.group_by("game_id", "offense", "defense").agg(plays=pl.len(), kickoff_ts=pl.col("kickoff_ts").first())
            .rename({"offense": "team", "defense": "opp"}))


def fit_pace(rows: pl.DataFrame, teams: list[str], prior_off: np.ndarray, prior_def: np.ndarray,
             mu0: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Ridge: plays - (prior_off[team] + prior_def[opp]) = mu + Poff + Pdef, 4-game penalty."""
    n = len(teams)
    if rows.height == 0:
        return mu0, prior_off.copy(), prior_def.copy()
    idx = {t: i for i, t in enumerate(teams)}
    ti = rows["team"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
    oi = rows["opp"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
    y = rows["plays"].to_numpy().astype(float) - (prior_off[ti] + prior_def[oi])
    m = len(y)
    X = np.zeros((m, 2 * n + 1))
    X[np.arange(m), ti] = 1.0
    X[np.arange(m), n + oi] = 1.0
    X[:, 2 * n] = 1.0
    A = X.T @ X
    A[np.arange(2 * n), np.arange(2 * n)] += PACE_PRIOR_GAMES
    A[2 * n, 2 * n] += 1e-6
    coef = np.linalg.solve(A, X.T @ y)
    return float(coef[2 * n]), prior_off + coef[:n], prior_def + coef[n:2 * n]


class Totals:
    def __init__(self, eng: Engine):
        self.eng = eng
        self._rows: dict[int, pl.DataFrame] = {}
        self._final_pace: dict[int, tuple[list[str], np.ndarray, np.ndarray, float]] = {}

    def final_pace(self, season: int) -> tuple[dict[str, float], dict[str, float], float]:
        """Full-season pace effects (no prior), used as next season's prior."""
        if season not in self._final_pace:
            tg = team_game_plays(self.eng, season)
            teams = sorted(set(tg["team"]) | set(tg["opp"]))
            z = np.zeros(len(teams))
            mu, po, pd_ = fit_pace(tg, teams, z, z, float(tg["plays"].mean()) if tg.height else 68.0)
            self._final_pace[season] = (teams, po, pd_, mu)
        teams, po, pd_, mu = self._final_pace[season]
        return dict(zip(teams, po)), dict(zip(teams, pd_)), mu

    def season_rows(self, season: int) -> pl.DataFrame:
        """One row per team-game with point-in-time features, target and baseline."""
        if season in self._rows:
            return self._rows[season]
        eng = self.eng
        games = eng.season_games_with_snapshots(season)
        tg = team_game_plays(eng, season)
        prev_off, prev_def, prev_mu = self.final_pace(season - 1)
        teams = eng.entities(season)
        p_off = np.array([PACE_PRIOR_SHRINK * prev_off.get(t, 0.0) for t in teams])
        p_def = np.array([PACE_PRIOR_SHRINK * prev_def.get(t, 0.0) for t in teams])
        idx = {t: i for i, t in enumerate(teams)}
        allg = eng.games.filter(pl.col("season") == season)
        prev_g = eng.games.filter((pl.col("season") == season - 1) & pl.col("completed") & pl.col("margin").is_not_null())
        prev_ppg = float(((prev_g["home_points"] + prev_g["away_points"]) / 2).mean()) if prev_g.height else 28.0
        out = []
        for r in eng.season_slots(season).iter_rows(named=True):
            as_of = r["as_of"]
            mu, po, pd_ = fit_pace(tg.filter(pl.col("kickoff_ts") < as_of), teams, p_off, p_def, prev_mu)
            done = allg.filter((pl.col("kickoff_ts") < as_of) & pl.col("completed") & pl.col("margin").is_not_null())
            lg = float(((done["home_points"] + done["away_points"]) / 2).mean()) if done.height else prev_ppg
            # baseline: points for / against so far
            pf = pl.concat([done.select(team="home", pf="home_points", pa="away_points"),
                            done.select(team="away", pf="away_points", pa="home_points")])
            agg = pf.group_by("team").agg(n=pl.len(), pf=pl.col("pf").sum(), pa=pl.col("pa").sum())
            pfh = {t: (f + BASELINE_PRIOR_GAMES * lg) / (k + BASELINE_PRIOR_GAMES) for t, k, f, _ in agg.iter_rows()}
            pah = {t: (a + BASELINE_PRIOR_GAMES * lg) / (k + BASELINE_PRIOR_GAMES) for t, k, _, a in agg.iter_rows()}
            sl = games.filter(pl.col("slot") == r["slot"])
            for g in sl.iter_rows(named=True):
                for side, opp_side, home in (("h", "a", 1.0), ("a", "h", -1.0)):
                    team, opp = (g["home"], g["away"]) if side == "h" else (g["away"], g["home"])
                    h = 0.0 if g["neutral_site"] else home
                    pts = g["home_points"] if side == "h" else g["away_points"]
                    out.append({
                        "game_id": g["game_id"], "season": season, "slot": r["slot"], "week": g["week"],
                        "season_type": g["season_type"], "team": team, "is_home": side == "h",
                        "d_ppa": g[f"o_ppa_{side}"] - g[f"d_ppa_{opp_side}"],
                        "d_sr": g[f"o_sr_{side}"] - g[f"d_sr_{opp_side}"],
                        "exp_plays": mu + po[idx[team]] + pd_[idx[opp]], "home": h, "lg_ppg": lg,
                        "points": float(pts) if pts is not None and g["completed"] else None,
                        "base_points": 0.5 * (pfh.get(team, lg) + pah.get(opp, lg)),
                    })
        df = pl.DataFrame(out, schema={"game_id": pl.Int64, "season": pl.Int32, "slot": pl.Utf8, "week": pl.Int32,
                                       "season_type": pl.Utf8, "team": pl.Utf8, "is_home": pl.Boolean,
                                       "d_ppa": pl.Float64, "d_sr": pl.Float64, "exp_plays": pl.Float64,
                                       "home": pl.Float64, "lg_ppg": pl.Float64, "points": pl.Float64,
                                       "base_points": pl.Float64})
        self._rows[season] = df
        return df

    def fit(self, seasons: list[int]) -> np.ndarray:
        rows = pl.concat([self.season_rows(s) for s in seasons]).filter(pl.col("points").is_not_null())
        X = np.column_stack([np.ones(rows.height), rows.select(FEATURES).to_numpy()])
        coef, *_ = np.linalg.lstsq(X, rows["points"].to_numpy(), rcond=None)
        return coef

    def predict_season(self, season: int, available: list[int]) -> pl.DataFrame:
        """Per game: model total/points and baseline total (plus actual where final)."""
        coef = self.fit(training_seasons(season, self.eng.cfg, available))
        r = self.season_rows(season)
        X = np.column_stack([np.ones(r.height), r.select(FEATURES).to_numpy()])
        r = r.with_columns(pred_points=pl.Series(X @ coef))
        h = r.filter(pl.col("is_home")).select("game_id", "season", "week", "season_type", home_pred="pred_points",
                                               home_base="base_points", home_pts="points")
        a = r.filter(~pl.col("is_home")).select("game_id", away_pred="pred_points", away_base="base_points",
                                                away_pts="points")
        return h.join(a, on="game_id").with_columns(
            pred_total=pl.col("home_pred") + pl.col("away_pred"),
            base_total=pl.col("home_base") + pl.col("away_base"),
            total=pl.col("home_pts") + pl.col("away_pts"),
        )
