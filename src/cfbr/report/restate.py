"""Restate a live season under the published config (DECISIONS "FCS games, two-stage fit").

Moves the existing weekly outputs aside (never deletes them), then rebuilds rankings, the accuracy
line and the season simulation for every week that had been published. The paper-bet ledger and
other season-level files are never touched."""

from __future__ import annotations

import logging
from pathlib import Path

from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.model.engine import Engine
from cfbr.report.rank import append_accuracy, run_rank, season_accuracy

log = logging.getLogger(__name__)


def archive_outputs(season_dir: Path, through_week: int, name: str = "_v1") -> list[int]:
    """Move out/{season}/wNN/ (NN <= through_week) and accuracy.csv into season_dir/name/.
    Returns the archived week numbers, i.e. the weeks to rebuild. Refuses if the archive exists."""
    dest = season_dir / name
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; refusing to overwrite an archive")
    weeks = sorted(int(p.name[1:]) for p in season_dir.glob("w[0-9][0-9]")
                   if p.is_dir() and int(p.name[1:]) <= through_week)
    dest.mkdir(parents=True)
    for w in weeks:
        (season_dir / f"w{w:02d}").rename(dest / f"w{w:02d}")
    acc = season_dir / "accuracy.csv"
    if acc.exists():
        acc.rename(dest / "accuracy.csv")
    return weeks


def restate_season(cfg: Config, ds: Dataset, season: int, weeks: list[int]) -> Engine:
    eng = Engine(ds, cfg, last_completed=season - 1)
    acc_path = cfg.paths.out_dir / str(season) / "accuracy.csv"
    for w in weeks:
        log.info("restating %d week %d", season, w)
        run_rank(cfg, season, w, last_completed=season - 1, ds=ds)
        append_accuracy(acc_path, season, w, season_accuracy(eng, season, w))
        if cfg.sim.enabled:
            from cfbr.sim.report import run_sim

            run_sim(eng, season, w)
    return eng
