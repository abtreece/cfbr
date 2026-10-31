"""Walk-forward backtest."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import consensus_spread, diagnostics, opening_spread, report
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


OPENER_COLUMNS = ["spread_open", "n_providers_open"]
HEADLINE_KEYS = ("overall", "by_season", "by_bucket", "by_season_bucket")


def refresh_lines(cfg: Config, run_dir: Path, lines: pl.DataFrame) -> dict:
    """Add the opening-line comparison to an existing run, additively. The opener columns are
    re-joined from the current lines table and written next to the untouched stored ``spread``; the
    function refuses before writing anything if ``report()`` on the stored close no longer reproduces
    the run's headline keys. Only ``diagnostics.ats_by_edge_open`` and ``diagnostics.opener`` are
    written into metrics.json; every other key stays byte-identical. The opener block's close is the
    books-only consensus from the current table (the stored close where the table has none), so the
    opener and the close are like-for-like even for a run that predates the consensus-rule fix. The
    model is not re-run."""
    method = cfg.backtest.spread_consensus
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    preds = preds.drop([c for c in OPENER_COLUMNS if c in preds.columns])
    preds = preds.join(opening_spread(lines, method), on="game_id", how="left", maintain_order="left")
    old = json.loads((run_dir / "metrics.json").read_text())
    new = report(preds.filter(pl.col("completed")))
    for key in HEADLINE_KEYS:
        if json.dumps(old.get(key), sort_keys=True, default=float) != json.dumps(new.get(key), sort_keys=True, default=float):
            raise RuntimeError(f"{key} would change: the lines table differs from the one this run was scored on")
    current_close = consensus_spread(lines, method).select("game_id", spread_now="spread")
    like_for_like = (preds.join(current_close, on="game_id", how="left", maintain_order="left")
                     .with_columns(spread=pl.coalesce("spread_now", "spread")).drop("spread_now"))
    diag = diagnostics(like_for_like.filter(pl.col("completed")), close_basis="books-only consensus from the current lines table")
    merged = json.loads(json.dumps(old, default=float))
    merged.setdefault("diagnostics", {})
    for key in ("ats_by_edge_open", "opener"):
        if key in diag:
            merged["diagnostics"][key] = diag[key]
    tmp = run_dir / "predictions.parquet.tmp"
    preds.write_parquet(tmp)
    tmp.replace(run_dir / "predictions.parquet")
    tmp = run_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(merged, indent=2, default=float))
    tmp.replace(run_dir / "metrics.json")
    return merged
