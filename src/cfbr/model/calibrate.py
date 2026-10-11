"""Map efficiency ratings to points.

    margin_home = b0 * home + b1 * (net_ppa_h - net_ppa_a) + b2 * (net_sr_h - net_sr_a) + e

where net = O + D (higher D is a better defense in the efficiency model).
A team's Rating = b1 * net_ppa + b2 * net_sr: expected margin vs an average
FBS team on a neutral field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class Calibration:
    b_home: float
    b_ppa: float
    b_sr: float
    sigma: float
    n_games: int

    def rating(self, o_ppa, d_ppa, o_sr, d_sr):
        off = self.b_ppa * o_ppa + self.b_sr * o_sr
        de = self.b_ppa * d_ppa + self.b_sr * d_sr
        return off + de, off, de


def design(games: pl.DataFrame) -> np.ndarray:
    """``games`` has home/away rating columns suffixed _h/_a and neutral_site."""
    home = (~games["neutral_site"]).cast(pl.Float64).to_numpy()
    d_ppa = ((games["o_ppa_h"] + games["d_ppa_h"]) - (games["o_ppa_a"] + games["d_ppa_a"])).to_numpy()
    d_sr = ((games["o_sr_h"] + games["d_sr_h"]) - (games["o_sr_a"] + games["d_sr_a"])).to_numpy()
    return np.column_stack([home, d_ppa, d_sr])


def fit_calibration(games: pl.DataFrame) -> Calibration:
    X = design(games)
    y = games["margin"].to_numpy()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sigma = float(np.sqrt(resid @ resid / max(len(y) - X.shape[1], 1)))
    return Calibration(float(coef[0]), float(coef[1]), float(coef[2]), sigma, len(y))


def predict_margin(cal: Calibration, games: pl.DataFrame) -> np.ndarray:
    return design(games) @ np.array([cal.b_home, cal.b_ppa, cal.b_sr])


# -- calibration v2 (DECISIONS "M2+M3 calibration v2") ------------------------------
# A second stage on the calibrated margin: pred' = pred + c . [pred, pred*ps, home*ps, fcs].
# Prior-heavy predictions (early season) are under-dispersed and data-heavy ones slightly
# over-dispersed; games against a pooled FCS side are under-predicted. Game predictions
# only: published ratings keep the base coefficients.
STAGE2_TERMS = ["pred", "pred_ps", "home_ps", "fcs"]


def stage2_design(games: pl.DataFrame, pred: np.ndarray) -> np.ndarray:
    """``games`` needs neutral_site, prior_share_h/_a and home_fcs/away_fcs."""
    ps = ((games["prior_share_h"].fill_null(1.0) + games["prior_share_a"].fill_null(1.0)) / 2).to_numpy()
    home = (~games["neutral_site"]).cast(pl.Float64).to_numpy()
    fcs = (games["away_fcs"].cast(pl.Float64) - games["home_fcs"].cast(pl.Float64)).fill_null(0.0).to_numpy()
    return np.column_stack([pred, pred * ps, home * ps, fcs])


def fit_stage2(games: pl.DataFrame, pred: np.ndarray) -> np.ndarray:
    y = games["margin"].to_numpy() - pred
    coef, *_ = np.linalg.lstsq(stage2_design(games, pred), y, rcond=None)
    return coef
