import pytest

from cfbr.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def dataset(cfg):
    """Real data from the local DuckDB store; tests using it skip when absent."""
    if not cfg.paths.db_path.exists():
        pytest.skip("no local DuckDB store (run cfbr ingest --backfill && cfbr prep)")
    from cfbr.data import Dataset

    return Dataset.load(cfg)
