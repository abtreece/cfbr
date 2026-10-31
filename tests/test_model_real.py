import time

import polars as pl
import pytest

from cfbr.model.engine import Engine


def test_week_ratings_cover_fbs_and_fcs_and_are_fast(cfg, dataset):
    eng = Engine(dataset, cfg, last_completed=2025)
    eng.calibration(2019)
    t0 = time.perf_counter()
    tbl = eng.ratings(2019, 8)
    assert time.perf_counter() - t0 < 10
    assert set(dataset.fbs_teams(2019)) <= set(tbl["team"])
    assert "FCS" in set(tbl["team"])
    assert tbl["rating"].is_finite().all()
    # Average FBS team is ~0 by construction.
    assert abs(tbl.filter(pl.col("fbs"))["rating"].mean()) < 0.5


def test_prior_share_uses_recency_weighted_plays(cfg, dataset):
    """With recency weighting on, the prior share must reflect the down-weighted
    plays the fit actually used, not raw play counts."""
    from cfbr.backtest.tune import apply_params

    tuned = apply_params(cfg, {"half_life_weeks": 3.0})
    eng = Engine(dataset, tuned, last_completed=2025)
    eng.calibration(2019)
    week1 = eng.ratings(2019, 1).filter(pl.col("fbs"))
    assert (week1["prior_share"] - 1.0).abs().max() < 1e-9  # no plays yet: all prior
    snap = eng.snapshot(2019, eng.slot_as_of(2019, "r08"))
    assert (snap["w_off"] < snap["n_off"]).filter(snap["n_off"] > 0).all()
    tbl = eng.ratings(2019, 8).filter(pl.col("fbs"))
    a = (tuned.model.alpha_ppa, tuned.model.alpha_sr)
    unweighted = snap.select("team", raw=sum(x / (x + pl.col(c)) for x in a for c in ("n_off", "n_def")) / 4)
    joined = tbl.join(unweighted, on="team")
    assert (joined["prior_share"] > joined["raw"]).all()


def test_portal_variant_is_v1_before_the_cutoff(cfg, dataset):
    if dataset.roster is None or dataset.player_ppa is None or dataset.player_defense is None:
        pytest.skip("roster tables not in the local store")
    portal = cfg.model_copy(deep=True)
    portal.prior.features = "roster"
    portal.prior.roster_min_season = 2021
    v1 = Engine(dataset, cfg, last_completed=2025)
    pv = Engine(dataset, portal, last_completed=2025)
    a = v1.priors(2019).sort("team")
    b = pv.priors(2019).sort("team")
    assert a.equals(b)
    assert v1.priors(2021).sort("team").equals(pv.priors(2021).sort("team"))  # 2021 falls back through the row-count guard
    c = pv.priors(2025).sort("team")
    assert not a.select("team").join(c, on="team").is_empty()  # it still produces priors after the cutoff
    assert not v1.priors(2025).sort("team").equals(c)
