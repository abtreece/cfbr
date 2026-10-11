"""Read-only access to the files the web app shows, reloaded when they change.

``cfbr weekly`` rewrites ``out/web/{season}/`` while the server is running;
each season's ``meta.json`` is written last, so its mtime is the reload signal.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Season:
    meta: dict
    ratings: pl.DataFrame
    games: pl.DataFrame

    @property
    def weeks(self) -> list[int]:
        return self.meta["weeks"]

    def label(self, week: int) -> str:
        return self.meta["labels"][str(week)]


@dataclass(frozen=True)
class Run:
    run_id: str
    metrics: dict


class Store:
    def __init__(self, web_root: Path, results_dir: Path, out_dir: Path):
        self.web_root = web_root
        self.results_dir = results_dir
        self.out_dir = out_dir
        self._lock = threading.Lock()
        self._seasons: dict[int, tuple[float, Season]] = {}
        self._runs: dict[str, tuple[float, Run]] = {}

    def seasons(self) -> list[int]:
        if not self.web_root.is_dir():
            return []
        return sorted(int(p.parent.name) for p in self.web_root.glob("*/meta.json") if p.parent.name.isdigit())

    def season(self, season: int) -> Season | None:
        meta_path = self.web_root / str(season) / "meta.json"
        try:
            mtime = meta_path.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            hit = self._seasons.get(season)
            if hit and hit[0] == mtime:
                return hit[1]
            d = meta_path.parent
            s = Season(json.loads(meta_path.read_text()), pl.read_parquet(d / "ratings.parquet"),
                       pl.read_parquet(d / "games.parquet"))
            self._seasons[season] = (mtime, s)
            return s

    def latest(self) -> int | None:
        seasons = self.seasons()
        return seasons[-1] if seasons else None

    def runs(self) -> list[Run]:
        """Backtest runs under results/, newest first."""
        out = []
        if not self.results_dir.is_dir():
            return out
        for path in sorted(self.results_dir.glob("*/metrics.json"), reverse=True):
            mtime = path.stat().st_mtime
            with self._lock:
                hit = self._runs.get(path.parent.name)
                if not hit or hit[0] != mtime:
                    hit = (mtime, Run(path.parent.name, json.loads(path.read_text())))
                    self._runs[path.parent.name] = hit
            out.append(hit[1])
        return out

    def accuracy_log(self, season: int) -> pl.DataFrame | None:
        """Season-to-date lines appended by ``cfbr weekly``."""
        path = self.out_dir / str(season) / "accuracy.csv"
        return pl.read_csv(path) if path.exists() else None
