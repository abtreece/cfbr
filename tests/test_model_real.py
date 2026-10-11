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
