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
    # Second-stage game calibration (DECISIONS "M2+M3 calibration v2"); shadow only in 2026.
    calibration_v2: bool = False


class Baseline(BaseModel):
    margin_cap: int = 28
    alpha: float = 10.0


class Prior(BaseModel):
    fit_scheme: str = "hybrid"
    # "v1": prev ratings + CFBD returning production + talent + recruiting.
    # "roster": v1 plus player-level roster features (model/roster.py).
    features: str = "v1"


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
    # Offshore books: their lines feed the consensus (market data) but a bet is
    # never priced at one, so they never supply the best-line number.
    offshore_books: list[str] = ["Bovada", "BetOnline", "LowVig", "MyBookie", "BetUS", "Bookmaker", "5Dimes",
                                 "SportsBetting", "Heritage"]
    # Books a bet may be priced at (the best line). Every book, offshore included, still
    # counts toward the consensus line. Empty = any non-offshore book.
    pricing_books: list[str] = ["DraftKings", "FanDuel", "bet365"]
    # Dollar reporting only: a starting bankroll and the dollar value of one unit
    # (1% of the bankroll, so a 1-unit bet at -110 risks $11 to win $10).
    bankroll: float = 1000.0
    unit_value: float = 10.0
    # Reporting tiers by edge size: name -> minimum |edge| in points. Every bet is
    # still a flat stake; the tier only groups bets by how far the model is from the market.
    tiers: dict[str, float] = {"Small": 3.0, "Medium": 6.0, "Large": 10.0}


class Odds(BaseModel):
    """The Odds API snapshots for the paper-bet tracker (US sportsbook spreads with prices)."""
    enabled: bool = True
    # Each region costs one credit per call: "us" = DraftKings, FanDuel, BetRivers, BetMGM...
    # plus offshore books; "us2" = theScore Bet, Hard Rock Bet, Bally Bet, betPARX...
    regions: list[str] = ["us", "us2"]
    monthly_quota: int = 500
    quota_reserve: int = 50
    # Combine an odds snapshot with the CFBD line snapshot only when taken this close together.
    max_snapshot_gap_hours: float = 6.0
    # Closing line: consensus from the last snapshot before kickoff, if no older than this.
    closing_max_age_hours: float = 12.0
    # `cfbr odds-snapshot` only spends credits when a game kicks off within this many hours.
    snapshot_window_hours: float = 6.0


class OddsPapi(BaseModel):
    """OddsPapi v4: bet365 lines for pricing paper bets (not in the consensus). Key: ODDSPAPI_KEY."""
    enabled: bool = True
    # OddsPapi bookmaker slugs fetched as extra pricing books.
    books: list[str] = ["bet365"]
    # Free tier: 250 requests a month; each bet's game costs one request, plus one fixture list per run.
    monthly_quota: int = 250
    quota_reserve: int = 20


class Alerts(BaseModel):
    """Push alerts from unattended runs via ntfy; the topic is NTFY_TOPIC in the env or .keys."""
    enabled: bool = True
    server: str = "https://ntfy.sh"


class Backup(BaseModel):
    """Nightly rsync of data that can't be rebuilt (`cfbr backup`; see cfbr/backup.py)."""
    target: str = "~/cfbr-backup"
    include_db: bool = False  # data/cfb.duckdb is rebuilt from data/raw, so optional
    keep_previous_days: int = 30
    max_age_days: float = 8.0  # the health check flags an older backup


class Sim(BaseModel):
    """Monte Carlo season simulation (conference titles, playoff odds)."""
    enabled: bool = True
    n_sims: int = 100_000
    seed: int = 0
    # Per-team strength offset drawn once per simulated season (points); game noise is
    # sqrt(sigma^2 - 2 tau^2) so each game's spread stays at the calibrated sigma.
    tau: float = 6.0
    # 2026 CFP: 12 teams; auto bids for these champions, the top Group of Six champion
    # ("champion") or team ("team"), Notre Dame if ranked in the top N.
    power_conferences: list[str] = ["SEC", "Big Ten", "ACC", "Big 12"]
    group_of_six_bid: str = "champion"
    notre_dame_top: int = 12
    field_size: int = 12
    byes: int = 4
    # Committee-ranking proxy: score = strength - losses_weight * losses + bonuses.
    losses_weight: float = 6.5
    champion_bonus: float = 0.0
    power_bonus: float = 10.0


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
    odds: Odds = Odds()
    oddspapi: OddsPapi = OddsPapi()
    alerts: Alerts = Alerts()
    backup: Backup = Backup()
    sim: Sim = Sim()
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
