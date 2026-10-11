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
import numpy as np
import polars as pl
from scipy.stats import norm

from cfbr.config import Config
from cfbr.data import Dataset, slot_as_of, training_slice
from cfbr.model.calibrate import Calibration, fit_calibration, predict_margin
from cfbr.model.massey import fit_massey
from cfbr.model.prior import (COMPONENTS, fit_prior_models, pooled_priors, predict_priors,
                              season_features)
from cfbr.model.ridge import fit_efficiency, recency_weights
from cfbr.model.schemes import training_seasons

log = logging.getLogger(__name__)

RATING_COLS = ["o_ppa", "d_ppa", "o_sr", "d_sr"]


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
        self._season_plays: dict[int, pl.DataFrame] = {}

    # -- entities -----------------------------------------------------------
    def season_plays(self, season: int) -> pl.DataFrame:
        if season not in self._season_plays:
            self._season_plays[season] = self.plays.filter(pl.col("season") == season)
        return self._season_plays[season]

    def fbs(self, season: int) -> list[str]:
        return self.ds.fbs_teams(season)

    def entities(self, season: int) -> list[str]:
        p = self.season_plays(season)
        g = self.games.filter(pl.col("season") == season)
        ents = set(self.fbs(season)) | set(p["offense"]) | set(p["defense"]) | set(g["home"]) | set(g["away"])
        return sorted(e for e in ents if e is not None)

    # -- fits ---------------------------------------------------------------
    def _fit(self, plays: pl.DataFrame, season: int, prior: pl.DataFrame | None, as_of=None) -> pl.DataFrame:
        teams = self.entities(season)
        if prior is not None:
            pr = pl.DataFrame({"team": teams}).join(prior, on="team", how="left").fill_null(0.0)
            p = {c: pr[c].to_numpy() for c in RATING_COLS}
        else:
            p = {c: None for c in RATING_COLS}
        w = recency_weights(plays["kickoff_ts"], as_of, self.cfg.model.half_life_weeks) if as_of is not None else None
        fbs = self.fbs(season)
        f_ppa = fit_efficiency(plays, "ppa", self.cfg.model.alpha_ppa, teams, p["o_ppa"], p["d_ppa"], w, fbs)
        f_sr = fit_efficiency(plays, "success", self.cfg.model.alpha_sr, teams, p["o_sr"], p["d_sr"], w, fbs)
        return pl.DataFrame({
            "team": teams, "o_ppa": f_ppa.o, "d_ppa": f_ppa.d, "o_sr": f_sr.o, "d_sr": f_sr.d,
            "n_off": f_ppa.n_off, "n_def": f_ppa.n_def,
        })

    def final(self, season: int) -> pl.DataFrame:
        """No-prior full-season ratings, used as prior features and targets."""
        if season not in self._final:
            self._final[season] = self._fit(self.season_plays(season), season, prior=None)
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
        rows = []
        for s in train_seasons:
            f = season_features(s, self.fbs(s), finals, self.ds.returning, self.ds.talent, self.ds.recruiting)
            rows.append(f.join(finals[s].select("team", *COMPONENTS), on="team", how="left"))
        models = fit_prior_models(pl.concat(rows, how="diagonal_relaxed"))
        feats = season_features(season, self.fbs(season), finals, self.ds.returning, self.ds.talent,
                                self.ds.recruiting)
        fbs_prior = predict_priors(models, feats).select("team", *COMPONENTS)
        pooled = [e for e in self.entities(season) if e not in set(self.fbs(season))]
        out = pl.concat([fbs_prior, pooled_priors(season, pooled, finals).select("team", *COMPONENTS)])
        self._priors[season] = out
        return out

    def snapshot(self, season: int, as_of) -> pl.DataFrame:
        plays = training_slice(self.season_plays(season), season, as_of)
        return self._fit(plays, season, self.priors(season), as_of)

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
        snaps = self.snapshots(season)
        g = self.games.filter(pl.col("season") == season)
        a = self.cfg.model.alpha_ppa
        parts = []
        for slot, snap in snaps.items():
            share = snap.select("team", prior_share=0.5 * (a / (a + pl.col("n_off")) + a / (a + pl.col("n_def"))))
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
        g = g.with_columns(pred_margin=pl.Series(predict_margin(cal, g)))
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
    a = cfg.model.alpha_ppa
    share = 0.5 * (a / (a + snap["n_off"]) + a / (a + snap["n_def"]))
    return snap.select("team").with_columns(
        rating=pl.Series(np.asarray(rating)), offense=pl.Series(np.asarray(off)),
        defense=pl.Series(np.asarray(de)), prior_share=pl.Series(np.asarray(share)),
    )
