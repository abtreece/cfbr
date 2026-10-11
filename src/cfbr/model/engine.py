"""Point-in-time rating engine.

For a config, computes (all lazily, cached per instance):
  final(s)      end-of-season ratings with no prior (prior features/targets)
  priors(s)     preseason prior for every entity in season s
  snapshot(s,t) ratings using only season-s plays with kickoff < t
  calibration(s) PPA/SR -> points map, fit on as-of ratings of training seasons
  predict(s)    per-game predictions for every slot of season s
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import norm

from cfbr.config import Config
from cfbr.data import Dataset, slot_as_of, training_slice
from cfbr.model.calibrate import Calibration, fit_calibration, fit_stage2, predict_margin, stage2_design
from cfbr.model.massey import fit_massey
from cfbr.model.prior import (COMPONENTS, fit_prior_models, pooled_priors, predict_priors,
                              season_features)
from cfbr.model.ridge import fit_efficiency_arrays, home_indicator, recency_weights
from cfbr.model.roster import roster_features
from cfbr.model.schemes import training_seasons

log = logging.getLogger(__name__)

RATING_COLS = ["o_ppa", "d_ppa", "o_sr", "d_sr"]


@dataclass
class SeasonArrays:
    """A season's model plays encoded for fast point-in-time fits, sorted by
    kickoff so 'plays before t' is a prefix found by binary search."""
    teams: list[str]
    center: np.ndarray       # FBS mask over teams
    kickoff_ns: np.ndarray   # sorted ascending
    off: np.ndarray
    de: np.ndarray
    home: np.ndarray
    ppa: np.ndarray
    success: np.ndarray

    def n_before(self, as_of) -> int:
        """Number of plays with kickoff strictly before ``as_of`` (the leakage guard)."""
        if as_of is None:
            return len(self.kickoff_ns)
        return int(np.searchsorted(self.kickoff_ns, _ns(as_of), side="left"))


def _ns(ts) -> int:
    return int(pl.Series([ts]).dt.cast_time_unit("ns").dt.epoch("ns")[0])


class Engine:
    def __init__(self, ds: Dataset, cfg: Config, last_completed: int):
        self.ds = ds
        self.cfg = cfg
        self.first = cfg.ingest.first_season          # features only (no prior)
        self.first_eval = cfg.ingest.first_season + 1  # first season with a prior
        self.last_completed = last_completed
        self.plays = ds.model_plays(cfg)
        self.games = ds.pooled_games(cfg)
        self.slots = slot_as_of(self.games)
        self._final: dict[int, pl.DataFrame] = {}
        self._priors: dict[int, pl.DataFrame] = {}
        self._snaps: dict[int, dict[str, pl.DataFrame]] = {}
        self._cal: dict[int, Calibration] = {}
        self._stage2: dict[int, np.ndarray] = {}
        self._season_plays: dict[int, pl.DataFrame] = {}
        self._arrays: dict[int, SeasonArrays] = {}
        self._games_snap: dict[int, pl.DataFrame] = {}
        self._fbs: dict[int, list[str]] = {}

    # -- entities -----------------------------------------------------------
    def season_plays(self, season: int) -> pl.DataFrame:
        if season not in self._season_plays:
            self._season_plays[season] = self.plays.filter(pl.col("season") == season).sort("kickoff_ts")
        return self._season_plays[season]

    def fbs(self, season: int) -> list[str]:
        if season not in self._fbs:
            self._fbs[season] = self.ds.fbs_teams(season)
        return self._fbs[season]

    def entities(self, season: int) -> list[str]:
        return self.arrays(season).teams

    def arrays(self, season: int) -> SeasonArrays:
        if season not in self._arrays:
            p = self.season_plays(season)
            g = self.games.filter(pl.col("season") == season)
            ents = set(self.fbs(season)) | set(p["offense"]) | set(p["defense"]) | set(g["home"]) | set(g["away"])
            teams = sorted(e for e in ents if e is not None)
            idx = {t: i for i, t in enumerate(teams)}
            self._arrays[season] = SeasonArrays(
                teams=teams,
                center=np.isin(np.array(teams, dtype=object), self.fbs(season)),
                kickoff_ns=p["kickoff_ts"].dt.cast_time_unit("ns").dt.epoch("ns").to_numpy(),
                off=p["offense"].replace_strict(idx, return_dtype=pl.Int64).to_numpy(),
                de=p["defense"].replace_strict(idx, return_dtype=pl.Int64).to_numpy(),
                home=home_indicator(p["offense"], p["defense"], p["home_team"], p["neutral_site"]),
                ppa=p["ppa"].to_numpy().astype(float),
                success=p["success"].cast(pl.Float64).to_numpy(),
            )
        return self._arrays[season]

    # -- fits ---------------------------------------------------------------
    def _fit(self, season: int, prior: pl.DataFrame | None, as_of=None) -> pl.DataFrame:
        """Fit both components on season plays with kickoff < as_of (all plays when None)."""
        a = self.arrays(season)
        k = a.n_before(as_of)
        teams = a.teams
        if prior is not None:
            pr = pl.DataFrame({"team": teams}).join(prior, on="team", how="left").fill_null(0.0)
            p = {c: pr[c].to_numpy() for c in RATING_COLS}
        else:
            p = {c: None for c in RATING_COLS}
        w = None
        if as_of is not None:
            w = recency_weights(a.kickoff_ns[:k], _ns(as_of), self.cfg.model.half_life_weeks)
        off, de, h = a.off[:k], a.de[:k], a.home[:k]
        m = self.cfg.model
        f_ppa = fit_efficiency_arrays(off, de, h, a.ppa[:k], teams, m.alpha_ppa, p["o_ppa"], p["d_ppa"], w, a.center)
        f_sr = fit_efficiency_arrays(off, de, h, a.success[:k], teams, m.alpha_sr, p["o_sr"], p["d_sr"], w, a.center)
        n = len(teams)
        w_off = np.bincount(off, weights=w, minlength=n).astype(float)
        w_def = np.bincount(de, weights=w, minlength=n).astype(float)
        return pl.DataFrame({
            "team": teams, "o_ppa": f_ppa.o, "d_ppa": f_ppa.d, "o_sr": f_sr.o, "d_sr": f_sr.d,
            "n_off": f_ppa.n_off, "n_def": f_ppa.n_def, "w_off": w_off, "w_def": w_def,
        })

    def final(self, season: int) -> pl.DataFrame:
        """No-prior full-season ratings, used as prior features and targets."""
        if season not in self._final:
            self._final[season] = self._fit(season, prior=None)
        return self._final[season]

    def prior_available(self) -> list[int]:
        return list(range(self.first_eval, self.last_completed + 1))

    def priors(self, season: int) -> pl.DataFrame:
        if season in self._priors:
            return self._priors[season]
        train_seasons = training_seasons(season, self.cfg, self.prior_available())
        if not train_seasons:
            raise RuntimeError(f"no training seasons for prior of {season}")
        finals = {s: self.final(s) for s in range(min(train_seasons) - 1, max(train_seasons) + 1)}
        finals[season - 1] = self.final(season - 1)
        mode = self.cfg.prior.features
        rows = []
        for s in train_seasons:
            f = self._prior_features(s, finals, mode)
            rows.append(f.join(finals[s].select("team", *COMPONENTS), on="team", how="left"))
        models = fit_prior_models(pl.concat(rows, how="diagonal_relaxed"), mode)
        feats = self._prior_features(season, finals, mode)
        fbs_prior = predict_priors(models, feats).select("team", *COMPONENTS)
        pooled = [e for e in self.entities(season) if e not in set(self.fbs(season))]
        out = pl.concat([fbs_prior, pooled_priors(season, pooled, finals).select("team", *COMPONENTS)])
        self._priors[season] = out
        return out

    def _prior_features(self, season: int, finals: dict[int, pl.DataFrame], mode: str) -> pl.DataFrame:
        f = season_features(season, self.fbs(season), finals, self.ds.returning, self.ds.talent, self.ds.recruiting)
        if mode == "roster":
            if self.ds.roster is None or self.ds.player_ppa is None or self.ds.player_defense is None:
                raise RuntimeError("prior.features = 'roster' needs roster/player tables (run cfbr ingest --backfill)")
            r = roster_features(season, self.fbs(season), self.ds.roster, self.ds.player_ppa,
                                self.ds.player_defense, self.ds.team_status)
            f = f.join(r, on="team", how="left")
        return f

    def snapshot(self, season: int, as_of) -> pl.DataFrame:
        return self._fit(season, self.priors(season), as_of)

    def season_slots(self, season: int) -> pl.DataFrame:
        return self.slots.filter(pl.col("season") == season)

    def snapshots(self, season: int) -> dict[str, pl.DataFrame]:
        if season not in self._snaps:
            self._snaps[season] = {
                r["slot"]: self.snapshot(season, r["as_of"]) for r in self.season_slots(season).iter_rows(named=True)
            }
        return self._snaps[season]

    # -- games -> features ----------------------------------------------------
    def _attach(self, games: pl.DataFrame, ratings: pl.DataFrame) -> pl.DataFrame:
        r = ratings.select("team", *RATING_COLS)
        return (games.join(r.rename({c: f"{c}_h" for c in RATING_COLS}), left_on="home", right_on="team", how="left")
                .join(r.rename({c: f"{c}_a" for c in RATING_COLS}), left_on="away", right_on="team", how="left")
                .with_columns([pl.col(f"{c}_{s}").fill_null(0.0) for c in RATING_COLS for s in "ha"]))

    def season_games_with_snapshots(self, season: int) -> pl.DataFrame:
        if season not in self._games_snap:
            self._games_snap[season] = self._season_games_with_snapshots(season)
        return self._games_snap[season]

    def _season_games_with_snapshots(self, season: int) -> pl.DataFrame:
        snaps = self.snapshots(season)
        g = self.games.filter(pl.col("season") == season)
        parts = []
        for slot, snap in snaps.items():
            share = snap.select("team", prior_share=prior_share_expr(self.cfg))
            part = self._attach(g.filter(pl.col("slot") == slot), snap)
            part = (part.join(share.rename({"prior_share": "prior_share_h"}), left_on="home", right_on="team", how="left")
                    .join(share.rename({"prior_share": "prior_share_a"}), left_on="away", right_on="team", how="left"))
            parts.append(part)
        return pl.concat(parts) if parts else g

    def calibration(self, season: int) -> Calibration:
        if season not in self._cal:
            train = training_seasons(season, self.cfg, self.prior_available())
            rows = pl.concat([self.season_games_with_snapshots(s) for s in train])
            rows = rows.filter(pl.col("completed") & pl.col("margin").is_not_null())
            self._cal[season] = fit_calibration(rows)
        return self._cal[season]

    def base_predictions(self, season: int) -> tuple[pl.DataFrame, np.ndarray]:
        """Season games with snapshots, and each game's base calibrated margin (that season's calibration)."""
        g = self.season_games_with_snapshots(season)
        return g, predict_margin(self.calibration(season), g)

    def stage2(self, season: int) -> np.ndarray:
        """Calibration v2 coefficients for ``season``: OLS of margin − pred on the stage-2 terms over the
        same training seasons as the base calibration, each game's pred from its own slot snapshot and
        its own season's calibration (out of sample for the base fit)."""
        if season not in self._stage2:
            rows, preds = [], []
            for s in training_seasons(season, self.cfg, self.prior_available()):
                g, p = self.base_predictions(s)
                keep = (g["completed"] & g["margin"].is_not_null()).fill_null(False).to_numpy()
                rows.append(g.filter(pl.Series(keep)))
                preds.append(p[keep])
            self._stage2[season] = fit_stage2(pl.concat(rows, how="diagonal_relaxed"), np.concatenate(preds))
        return self._stage2[season]

    def game_margin(self, season: int, g: pl.DataFrame) -> np.ndarray:
        """Predicted home margin for rows of ``g`` (with snapshot ratings and prior shares attached)."""
        pred = predict_margin(self.calibration(season), g)
        if self.cfg.model.calibration_v2:
            pred = pred + stage2_design(g, pred) @ self.stage2(season)
        return pred

    # -- predictions ------------------------------------------------------------
    def massey_predictions(self, season: int) -> pl.DataFrame:
        g = self.games.filter(pl.col("season") == season)
        teams = self.entities(season)
        out = []
        for r in self.season_slots(season).iter_rows(named=True):
            past = training_slice(g, season, r["as_of"]).filter(pl.col("completed") & pl.col("margin").is_not_null())
            fit = fit_massey(past, teams, self.cfg.baseline.alpha, self.cfg.baseline.margin_cap, self.fbs(season))
            rat = fit.lookup()
            sl = g.filter(pl.col("slot") == r["slot"])
            pred = [rat.get(h, 0.0) - rat.get(a, 0.0) + (0.0 if n else fit.home)
                    for h, a, n in sl.select("home", "away", "neutral_site").iter_rows()]
            out.append(sl.select("game_id").with_columns(massey_pred=pl.Series(pred, dtype=pl.Float64)))
        return pl.concat(out) if out else pl.DataFrame(schema={"game_id": pl.Int64, "massey_pred": pl.Float64})

    def predict(self, season: int) -> pl.DataFrame:
        cal = self.calibration(season)
        g = self.season_games_with_snapshots(season)
        g = g.with_columns(pred_margin=pl.Series(self.game_margin(season, g)))
        g = g.with_columns(win_prob=pl.Series(norm.cdf(g["pred_margin"].to_numpy() / cal.sigma)))
        prior_g = self._attach(self.games.filter(pl.col("season") == season), self.priors(season))
        prior_g = prior_g.select("game_id", prior_pred=pl.Series(predict_margin(cal, prior_g)))
        as_of = self.season_slots(season).select("slot", "as_of")
        return (g.join(prior_g, on="game_id", how="left")
                .join(self.massey_predictions(season), on="game_id", how="left")
                .join(as_of, on="slot", how="left")
                .with_columns(sigma=pl.lit(cal.sigma))
                .select("game_id", "season", "week", "season_type", "slot", "slot_order", "as_of", "kickoff_ts",
                        "completed", "neutral_site", "home", "away", "home_team", "away_team", "margin",
                        "pred_margin", "win_prob", "prior_pred", "massey_pred", "sigma",
                        "prior_share_h", "prior_share_a"))

    def slot_as_of(self, season: int, slot: str):
        r = self.slots.filter((pl.col("season") == season) & (pl.col("slot") == slot))
        return r["as_of"].item() if r.height else None

    def ratings(self, season: int, week: int | None = None, as_of=None) -> pl.DataFrame:
        """Ratings in points available before regular-season ``week`` kicks off
        (or as of an explicit timestamp), for all FBS teams plus pooled FCS."""
        if as_of is None:
            as_of = self.slot_as_of(season, f"r{week:02d}")
            if as_of is None:
                raise ValueError(f"no games for season {season} week {week}")
        snap = self.snapshot(season, as_of)
        pts = team_ratings_points(snap, self.calibration(season), self.cfg)
        fbs = set(self.fbs(season))
        return (pts.with_columns(fbs=pl.col("team").is_in(list(fbs)))
                .join(snap.select("team", *RATING_COLS), on="team")
                .sort("rating", descending=True))


def team_ratings_points(snap: pl.DataFrame, cal: Calibration, cfg: Config) -> pl.DataFrame:
    """Ratings in points plus prior share for reporting."""
    rating, off, de = cal.rating(snap["o_ppa"], snap["d_ppa"], snap["o_sr"], snap["d_sr"])
    return snap.select("team", prior_share=prior_share_expr(cfg)).with_columns(
        rating=pl.Series(np.asarray(rating)), offense=pl.Series(np.asarray(off)),
        defense=pl.Series(np.asarray(de)),
    ).select("team", "rating", "offense", "defense", "prior_share")


def prior_share_expr(cfg: Config) -> pl.Expr:
    """Approximate share of a rating still coming from the prior: the ridge
    shrinkage alpha / (alpha + effective plays), where effective plays are the
    recency-weighted counts the fit actually used. Averaged over offense and
    defense and over the PPA and success-rate fits (each with its own alpha)."""
    terms = [a / (a + pl.col(c)) for a in (cfg.model.alpha_ppa, cfg.model.alpha_sr) for c in ("w_off", "w_def")]
    return sum(terms) / len(terms)
