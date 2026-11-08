import pytest

from cfbr.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def cfg_two_stage(cfg):
    """default.toml with the FCS two-stage fit on and recency on, so the dataset tests exercise it."""
    c = cfg.model_copy(deep=True)
    c.model.fcs_two_stage = True
    c.model.fcs_weight = 0.3
    c.model.half_life_weeks = 3.7
    return c


@pytest.fixture(scope="session")
def dataset(cfg):
    """Real data from the local DuckDB store; tests using it skip when absent."""
    if not cfg.paths.db_path.exists():
        pytest.skip("no local DuckDB store (run cfbr ingest --backfill && cfbr prep)")
    from cfbr.data import Dataset

    return Dataset.load(cfg)
