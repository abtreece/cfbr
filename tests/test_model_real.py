import time

import polars as pl

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
