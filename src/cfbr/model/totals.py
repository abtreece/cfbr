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


# -- Totals v2 (shadow): direct points ratings ------------------------------------
# Settings fixed in DECISIONS.md before the shadow window; not tuned.
V2_HALF_LIFE_WEEKS = 3.7
V2_PRIOR_GAMES = 4.0
V2_PRIOR_SHRINK = 0.5
V2_MIN_GAMES_FOR_ENV = 20


def _points_rows(games: pl.DataFrame) -> pl.DataFrame:
    """Two rows per completed game: team, opp, points scored, home indicator, kickoff."""
    g = games.filter(pl.col("completed") & pl.col("margin").is_not_null())
    h = g.select("kickoff_ts", team="home", opp="away", points=pl.col("home_points").cast(pl.Float64),
                 home=pl.when(pl.col("neutral_site")).then(0.0).otherwise(1.0))
    a = g.select("kickoff_ts", team="away", opp="home", points=pl.col("away_points").cast(pl.Float64),
                 home=pl.when(pl.col("neutral_site")).then(0.0).otherwise(-1.0))
    return pl.concat([h, a])


def fit_points(rows: pl.DataFrame, teams: list[str], prior_off: np.ndarray, prior_def: np.ndarray,
               weights: np.ndarray | None, mu_fallback: float) -> tuple[float, float, np.ndarray, np.ndarray]:
    """points - (prior_off[team] + prior_def[opp]) = mu + Off + Def + H*home; ridge on Off/Def only.
    With fewer than V2_MIN_GAMES_FOR_ENV games, mu is fixed at ``mu_fallback`` and H at 0."""
    n = len(teams)
    if rows.height == 0:
        return mu_fallback, 0.0, prior_off.copy(), prior_def.copy()
    idx = {t: i for i, t in enumerate(teams)}
    ti = rows["team"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
    oi = rows["opp"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
    w = np.ones(rows.height) if weights is None else weights
    free_env = rows.height >= 2 * V2_MIN_GAMES_FOR_ENV
    y = rows["points"].to_numpy() - (prior_off[ti] + prior_def[oi]) - (0.0 if free_env else mu_fallback)
    m = len(y)
    k = 2 * n + (2 if free_env else 0)
    X = np.zeros((m, k))
    X[np.arange(m), ti] = 1.0
    X[np.arange(m), n + oi] = 1.0
    if free_env:
        X[:, 2 * n] = 1.0
        X[:, 2 * n + 1] = rows["home"].to_numpy()
    Xw = X * w[:, None]
    A = X.T @ Xw
    A[np.arange(2 * n), np.arange(2 * n)] += V2_PRIOR_GAMES
    if free_env:
        A[2 * n, 2 * n] += 1e-6
        A[2 * n + 1, 2 * n + 1] += 1e-6
    coef = np.linalg.solve(A, Xw.T @ y)
    mu = float(coef[2 * n]) if free_env else mu_fallback
    hh = float(coef[2 * n + 1]) if free_env else 0.0
    return mu, hh, prior_off + coef[:n], prior_def + coef[n:2 * n]


class TotalsV2:
    def __init__(self, eng: Engine, base: Totals | None = None):
        self.eng = eng
        self.base = base or Totals(eng)
        self._final: dict[int, tuple[dict, dict, float]] = {}

    def final_effects(self, season: int) -> tuple[dict[str, float], dict[str, float], float]:
        if season not in self._final:
            g = self.eng.games.filter(pl.col("season") == season)
            rows = _points_rows(g)
            teams = sorted(set(rows["team"]) | set(rows["opp"])) if rows.height else []
            z = np.zeros(len(teams))
            mu0 = float(rows["points"].mean()) if rows.height else 28.0
            mu, _h, po, pd_ = fit_points(rows, teams, z, z, None, mu0)
            self._final[season] = (dict(zip(teams, po)), dict(zip(teams, pd_)), mu)
        return self._final[season]

    def predict_season(self, season: int) -> pl.DataFrame:
        """Per game: v2 total, baseline total, actual total (if final)."""
        from cfbr.model.ridge import recency_weights
        from cfbr.model.engine import _ns

        eng = self.eng
        teams = eng.entities(season)
        idx = {t: i for i, t in enumerate(teams)}
        prev_off, prev_def, prev_mu = self.final_effects(season - 1)
        p_off = np.array([V2_PRIOR_SHRINK * prev_off.get(t, 0.0) for t in teams])
        p_def = np.array([V2_PRIOR_SHRINK * prev_def.get(t, 0.0) for t in teams])
        allg = eng.games.filter(pl.col("season") == season)
        out = []
        for r in eng.season_slots(season).iter_rows(named=True):
            as_of = r["as_of"]
            rows = _points_rows(allg.filter(pl.col("kickoff_ts") < as_of))
            w = None
            if rows.height:
                kns = rows["kickoff_ts"].dt.cast_time_unit("ns").dt.epoch("ns").to_numpy()
                w = recency_weights(kns, _ns(as_of), V2_HALF_LIFE_WEEKS)
            mu, hh, po, pd_ = fit_points(rows, teams, p_off, p_def, w, prev_mu)
            for g in allg.filter(pl.col("slot") == r["slot"]).iter_rows(named=True):
                hi, ai = idx[g["home"]], idx[g["away"]]
                hsgn = 0.0 if g["neutral_site"] else 1.0
                ph = mu + po[hi] + pd_[ai] + hh * hsgn
                pa = mu + po[ai] + pd_[hi] - hh * hsgn
                out.append({"game_id": g["game_id"], "v2_total": ph + pa})
        v2 = pl.DataFrame(out, schema={"game_id": pl.Int64, "v2_total": pl.Float64})
        base = self.base.season_rows(season)
        bt = base.group_by("game_id").agg(base_total=pl.col("base_points").sum())
        games = allg.select("game_id", "season", "week", "season_type", "kickoff_ts", "completed", "home_team",
                            "away_team", total=(pl.col("home_points") + pl.col("away_points")).cast(pl.Float64))
        return games.join(v2, on="game_id", how="left").join(bt, on="game_id", how="left").sort("kickoff_ts")
