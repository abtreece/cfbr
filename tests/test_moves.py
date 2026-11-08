"""What-moved breakdown (spec 2026-11-08-what-moved-design.md). Real-data tests skip without the DB."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cfbr.model.engine import RATING_COLS, Engine, _ns
from cfbr.model.ridge import fit_efficiency_arrays, fit_fcs_stage2, play_weights, recency_weights


@pytest.fixture(params=["default", "two_stage"])
def model_cfg(request, cfg, cfg_two_stage):
    return cfg if request.param == "default" else cfg_two_stage


def test_unmasked_fit_is_unchanged_and_carries_home_coefficients(model_cfg, dataset):
    eng = Engine(dataset, model_cfg, last_completed=2025)
    as_of = eng.slot_as_of(2018, "r07")
    k = eng.arrays(2018).n_before(as_of)
    plain = eng._fit(2018, eng.priors(2018), as_of)
    full_mask = eng._fit(2018, eng.priors(2018), as_of, mask=np.ones(k, bool))
    assert_frame_equal(plain, full_mask)
    assert {"home_ppa", "home_sr"} <= set(plain.columns)
    assert plain["home_ppa"].n_unique() == 1 and plain["home_sr"].n_unique() == 1


def test_masked_fit_equals_explicit_fit_on_remaining_plays(model_cfg, dataset):
    eng = Engine(dataset, model_cfg, last_completed=2025)
    m, a = model_cfg.model, eng.arrays(2018)
    as_of = eng.slot_as_of(2018, "r07")
    k = a.n_before(as_of)
    keep = np.ones(k, bool)
    keep[k // 2:k // 2 + 500] = False
    got = eng._fit(2018, eng.priors(2018), as_of, mask=keep)
    pr = pl.DataFrame({"team": a.teams}).join(eng.priors(2018), on="team", how="left").fill_null(0.0)
    p = {c: pr[c].to_numpy() for c in RATING_COLS}
    w = play_weights(a, k, _ns(as_of), m)
    w = None if w is None else w[keep]
    off, de, h = a.off[:k][keep], a.de[:k][keep], a.home[:k][keep]
    ref = fit_efficiency_arrays(off, de, h, a.ppa[:k][keep], a.teams, m.alpha_ppa, p["o_ppa"], p["d_ppa"], w, a.center)
    if m.fcs_two_stage:
        rec = recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)
        ref = fit_fcs_stage2(off, de, h, a.ppa[:k][keep], ~a.center, m.alpha_ppa, ref, p["o_ppa"], p["d_ppa"],
                             None if rec is None else rec[keep])
    assert np.abs(got["o_ppa"].to_numpy() - ref.o).max() < 1e-9
    assert np.abs(got["d_ppa"].to_numpy() - ref.d).max() < 1e-9
    assert got["home_ppa"][0] == pytest.approx(ref.home, abs=1e-12)
