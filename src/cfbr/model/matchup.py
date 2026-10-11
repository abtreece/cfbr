"""Matchup (rush/pass split) ratings, point-in-time (rule: DECISIONS.md "Matchup").

Per prediction slot, the v1 ridge (same alphas, recency weights and priors) is
fit separately on rush plays and on pass plays, for PPA and success rate.
An offense's matchup expectation weights its rush and pass matchups by its
rush share so far: r·(O_rush − D_rush_opp) + (1 − r)·(O_pass − D_pass_opp).
Margin = OLS (no intercept) on home, Δ matchup PPA, Δ matchup SR, fit under
the hybrid scheme like v1's calibration.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cfbr.model.engine import RATING_COLS, Engine, _ns
from cfbr.model.ridge import fit_efficiency_arrays, recency_weights
from cfbr.model.schemes import training_seasons
from cfbr.prep.plays import PASS_TYPES, RUSH_TYPES

RUSH_SHARE_PRIOR_PLAYS = 100.0


class Matchup:
    def __init__(self, eng: Engine):
        self.eng = eng
        self._masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._rows: dict[int, pl.DataFrame] = {}

    def masks(self, season: int) -> tuple[np.ndarray, np.ndarray]:
        """Rush / pass masks aligned with Engine.arrays(season) (same kickoff-sorted order)."""
        if season not in self._masks:
            pt = self.eng.season_plays(season)["play_type"]
            self._masks[season] = (pt.is_in(list(RUSH_TYPES)).to_numpy(), pt.is_in(list(PASS_TYPES)).to_numpy())
        return self._masks[season]

    def slot_ratings(self, season: int, as_of) -> dict[str, np.ndarray]:
        eng, m = self.eng, self.eng.cfg.model
        a = eng.arrays(season)
        k = a.n_before(as_of)
        rush, pas = (x[:k] for x in self.masks(season))
        pr = pl.DataFrame({"team": a.teams}).join(eng.priors(season), on="team", how="left").fill_null(0.0)
        p = {c: pr[c].to_numpy() for c in RATING_COLS}
        w_all = recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)
        out = {}
        for split, mask in (("rush", rush), ("pass", pas)):
            sel = np.flatnonzero(mask)
            w = None if w_all is None else w_all[sel]
            off, de, h = a.off[:k][sel], a.de[:k][sel], a.home[:k][sel]
            for tgt, y, alpha, po, pd_ in (("ppa", a.ppa[:k][sel], m.alpha_ppa, p["o_ppa"], p["d_ppa"]),
                                            ("sr", a.success[:k][sel], m.alpha_sr, p["o_sr"], p["d_sr"])):
                f = fit_efficiency_arrays(off, de, h, y, a.teams, alpha, po, pd_, w, a.center)
                out[f"o_{split}_{tgt}"], out[f"d_{split}_{tgt}"] = f.o, f.d
        n = len(a.teams)
        r_n = np.bincount(a.off[:k][rush], minlength=n).astype(float)
        p_n = np.bincount(a.off[:k][pas], minlength=n).astype(float)
        tot = r_n.sum() + p_n.sum()
        league = r_n.sum() / tot if tot > 0 else 0.5
        out["rush_share"] = (r_n + RUSH_SHARE_PRIOR_PLAYS * league) / (r_n + p_n + RUSH_SHARE_PRIOR_PLAYS)
        out["teams"] = np.array(a.teams, dtype=object)
        return out

    def season_rows(self, season: int) -> pl.DataFrame:
        """Per game: Δ matchup PPA / SR (home minus away) and v1's prediction."""
        if season in self._rows:
            return self._rows[season]
        eng = self.eng
        v1 = eng.predict(season).select("game_id", "home", "away", "neutral_site", "margin", "completed",
                                        "pred_margin", "season", "week", "season_type", "slot")
        out = []
        for r in eng.season_slots(season).iter_rows(named=True):
            R = self.slot_ratings(season, r["as_of"])
            idx = {t: i for i, t in enumerate(R["teams"])}
            for g in v1.filter(pl.col("slot") == r["slot"]).iter_rows(named=True):
                hi, ai = idx[g["home"]], idx[g["away"]]
                row = {"game_id": g["game_id"]}
                for tgt in ("ppa", "sr"):
                    def exp_off(o, d):
                        rs = R["rush_share"][o]
                        return (rs * (R[f"o_rush_{tgt}"][o] - R[f"d_rush_{tgt}"][d])
                                + (1 - rs) * (R[f"o_pass_{tgt}"][o] - R[f"d_pass_{tgt}"][d]))
                    row[f"d_{tgt}"] = exp_off(hi, ai) - exp_off(ai, hi)
                out.append(row)
        feats = pl.DataFrame(out, schema={"game_id": pl.Int64, "d_ppa": pl.Float64, "d_sr": pl.Float64})
        df = v1.join(feats, on="game_id", how="left").with_columns(
            home_ind=(~pl.col("neutral_site")).cast(pl.Float64))
        self._rows[season] = df
        return df

    def predict_season(self, season: int, available: list[int]) -> pl.DataFrame:
        train = pl.concat([self.season_rows(s) for s in training_seasons(season, self.eng.cfg, available)])
        train = train.filter(pl.col("completed") & pl.col("margin").is_not_null())
        X = train.select("home_ind", "d_ppa", "d_sr").to_numpy()
        coef, *_ = np.linalg.lstsq(X, train["margin"].to_numpy(), rcond=None)
        r = self.season_rows(season)
        return r.with_columns(matchup_pred=pl.Series(r.select("home_ind", "d_ppa", "d_sr").to_numpy() @ coef))
