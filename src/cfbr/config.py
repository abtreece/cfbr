"""Typed configuration loaded from TOML."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.toml"


class Paths(BaseModel):
    raw_dir: Path = Path("data/raw")
    db_path: Path = Path("data/cfb.duckdb")
    results_dir: Path = Path("results")
    out_dir: Path = Path("out")

    def resolve(self, root: Path) -> "Paths":
        return Paths(**{k: (v if v.is_absolute() else root / v) for k, v in self})


class Ingest(BaseModel):
    base_url: str = "https://apinext.collegefootballdata.com"
    first_season: int = 2014
    recruiting_lookback: int = 3
    monthly_quota: int = 1000
    quota_reserve: int = 50


class GarbageTime(BaseModel):
    enabled: bool = True
    q2: int = 38
    q3: int = 28
    q4: int = 22


class Prep(BaseModel):
    garbage_time: GarbageTime = GarbageTime()
    fcs_pooling: str = "single"


class Model(BaseModel):
    alpha_ppa: float = 300.0
    alpha_sr: float = 300.0
    half_life_weeks: float = 0.0
    prior_k: float = 0.0


class Baseline(BaseModel):
    margin_cap: int = 28
    alpha: float = 10.0


class Prior(BaseModel):
    fit_scheme: str = "hybrid"


class Seasons(BaseModel):
    tuning: tuple[int, int] = (2015, 2021)
    holdout: tuple[int, int] = (2022, 2025)


class Backtest(BaseModel):
    spread_consensus: str = "median"


class Bets(BaseModel):
    """Paper-betting tracker. Rules are fixed before results come in; see DECISIONS.md."""
    enabled: bool = True
    # Bet when |model home margin - market home margin| >= threshold (points).
    threshold: float = 3.0
    # American odds assumed for every spread bet (CFBD lines carry no spread prices).
    odds: int = -110
    # Units risked per bet (flat staking).
    stake: float = 1.0
    # Minimum settled bets before the summary draws any conclusion.
    min_sample: int = 500


class Config(BaseModel):
    paths: Paths = Paths()
    ingest: Ingest = Ingest()
    prep: Prep = Prep()
    model: Model = Model()
    baseline: Baseline = Baseline()
    prior: Prior = Prior()
    seasons: Seasons = Seasons()
    backtest: Backtest = Backtest()
    bets: Bets = Bets()
    source: str | None = Field(default=None, exclude=True)


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "rb") as f:
        data = tomllib.load(f)
    cfg = Config(**data)
    cfg.paths = cfg.paths.resolve(REPO_ROOT)
    cfg.source = str(path)
    return cfg


def api_key() -> str:
    """CFBD key from the environment, falling back to the gitignored .keys file."""
    key = os.environ.get("CFBD_API_KEY")
    if key:
        return key
    keys_file = REPO_ROOT / ".keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "CFBD_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    raise RuntimeError("CFBD_API_KEY not set (env var or .keys file)")
