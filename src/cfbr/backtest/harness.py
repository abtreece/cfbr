"""Walk-forward backtest."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import consensus_spread, opening_spread, report
from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.ingest.fetch import current_season
from cfbr.model.engine import Engine

log = logging.getLogger(__name__)


def parse_seasons(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def run_backtest(cfg: Config, seasons: list[int], ds: Dataset | None = None) -> tuple[pl.DataFrame, dict]:
    ds = ds or Dataset.load(cfg)
    eng = Engine(ds, cfg, last_completed=current_season() - 1)
    t0 = time.perf_counter()
    preds = []
    for s in seasons:
        p = eng.predict(s)
        preds.append(p)
        log.info("season %d predicted: %d games (%.1fs elapsed)", s, p.height, time.perf_counter() - t0)
    df = pl.concat(preds)
    df = df.join(consensus_spread(ds.lines, cfg.backtest.spread_consensus), on="game_id", how="left")
    df = df.join(opening_spread(ds.lines, cfg.backtest.spread_consensus), on="game_id", how="left")
    metrics = report(df.filter(pl.col("completed")))
    metrics["calibration"] = {str(s): vars(eng.calibration(s)) for s in seasons}
    return df, metrics


def write_results(cfg: Config, seasons: list[int], preds: pl.DataFrame, metrics: dict, tag: str = "") -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + (f"-{tag}" if tag else "")
    out = cfg.paths.results_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    preds.write_parquet(out / "predictions.parquet")
    (out / "metrics.json").write_text(json.dumps({"seasons": seasons, **metrics}, indent=2, default=float))
    resolved = cfg.model_dump(mode="json", exclude={"paths"})
    (out / "config.resolved.json").write_text(json.dumps(resolved, indent=2))
    if cfg.source:
        (out / "config.source.toml").write_text(Path(cfg.source).read_text())
    return out
