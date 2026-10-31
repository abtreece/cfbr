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


def test_stage2_v3_coefficients_are_cached_per_design(cfg, dataset):
    from cfbr.backtest.tune import apply_params

    tuned = apply_params(cfg, {"alpha_ppa": 303.298, "alpha_sr": 90.7544, "half_life_weeks": 3.6978,
                               "fcs_pooling": "conference", "gt_enabled": False, "margin_cap": 42})
    eng = Engine(dataset, tuned, last_completed=2025)
    v2 = eng.stage2(2019, fcs_slope=False)
    v3 = eng.stage2(2019, fcs_slope=True)
    assert v2.shape == (4,) and v3.shape == (5,)
    assert eng.stage2(2019) is v2                      # default follows cfg.model.calibration_v3 == False
    assert eng.stage2(2019, fcs_slope=True) is v3
    assert v3[4] < 0                                  # FCS games: the model is too extreme, so the slope term shrinks


def test_v3_engine_matches_the_shadow_column(cfg, dataset):
    """Flipping calibration_v3 on makes predict() equal the shadow comparison's v3 column."""
    from cfbr.backtest.tune import apply_params
    from cfbr.model.calibrate import stage2_terms
    from cfbr.report.shadow import calibration_v3_comparison

    params = {"alpha_ppa": 303.298, "alpha_sr": 90.7544, "half_life_weeks": 3.6978,
              "fcs_pooling": "conference", "gt_enabled": False, "margin_cap": 42}
    eng_plain = Engine(dataset, apply_params(cfg, params), last_completed=2025)
    tuned_v3 = apply_params(cfg, params)
    tuned_v3.model.calibration_v3 = True
    eng = Engine(dataset, tuned_v3, last_completed=2025)
    got = eng.predict(2026).select("game_id", "pred_margin")
    want = calibration_v3_comparison(eng_plain, 2026).select("game_id", "pred_v3")
    joined = got.join(want, on="game_id")
    assert joined.height == want.height > 0
    assert (joined["pred_margin"] - joined["pred_v3"]).abs().max() < 1e-9
    assert len(stage2_terms(True)) == eng.stage2(2026).shape[0]
