"""Which past seasons may train a cross-season model (prior, calibration)
when predicting a target season.

hybrid (default): leave-one-season-out inside the tuning window; strict
expanding window (only earlier seasons) everywhere else, so holdout and live
seasons never see their own or later outcomes.
"""

from __future__ import annotations

from collections.abc import Iterable

from cfbr.config import Config


def training_seasons(target: int, cfg: Config, available: Iterable[int]) -> list[int]:
    avail = sorted(set(available))
    scheme = cfg.prior.fit_scheme
    t0, t1 = cfg.seasons.tuning
    if scheme == "hybrid":
        if t0 <= target <= t1:
            return [s for s in avail if t0 <= s <= t1 and s != target]
        return [s for s in avail if s < target]
    if scheme == "expanding":
        return [s for s in avail if s < target]
    if scheme == "loso":
        return [s for s in avail if s != target]
    raise ValueError(f"unknown fit_scheme {scheme!r}")
