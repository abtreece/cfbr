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

POLL_COLS = ("ap_rank", "coaches_rank", "cfp_rank")


def _ensure(df: pl.DataFrame, cols) -> pl.DataFrame:
    """Add missing (all-null) poll columns, for web data built before polls existed."""
    return df.with_columns([pl.lit(None, pl.Int32).alias(c) for c in cols if c not in df.columns])


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
class Roster:
    """A season's roster facts (see cfbr.report.roster_web): public, not model output."""
    players: pl.DataFrame
    departures: pl.DataFrame
    summary: pl.DataFrame
    careers: pl.DataFrame | None = None  # season-by-season history; optional (older builds lack it)


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
        self._bets: dict[int, tuple[float, pl.DataFrame]] = {}
        self._retro: dict[int, tuple[float, pl.DataFrame]] = {}
        self._rosters: dict[int, tuple[float, Roster | None]] = {}

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
            s = Season(json.loads(meta_path.read_text()),
                       _ensure(pl.read_parquet(d / "ratings.parquet"), POLL_COLS),
                       _ensure(pl.read_parquet(d / "games.parquet"),
                               [f"{side}_{c}" for side in ("home", "away") for c in POLL_COLS]))
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

    def bet_seasons(self) -> list[int]:
        """Seasons with a paper-bet ledger (``out/{season}/bets.csv``)."""
        if not self.out_dir.is_dir():
            return []
        return sorted(int(p.parent.name) for p in self.out_dir.glob("*/bets.csv") if p.parent.name.isdigit())

    def bets(self, season: int) -> pl.DataFrame | None:
        """The season's paper-bet ledger, reloaded when ``cfbr weekly`` rewrites it."""
        from cfbr.bets.ledger import read_ledger

        return self._cached_csv(self._bets, season, self.out_dir / str(season) / "bets.csv", read_ledger)

    def retro_seasons(self) -> list[int]:
        """Seasons with retroactive paper bets (``out/{season}/bets_retro.csv``)."""
        if not self.out_dir.is_dir():
            return []
        return sorted(int(p.parent.name) for p in self.out_dir.glob("*/bets_retro.csv") if p.parent.name.isdigit())

    def retro_bets(self, season: int) -> pl.DataFrame | None:
        """Retroactive paper bets from ``cfbr bets-retro``, reloaded when rewritten."""
        from cfbr.bets.retro import read_retro

        return self._cached_csv(self._retro, season, self.out_dir / str(season) / "bets_retro.csv", read_retro)

    def _cached_csv(self, cache: dict, season: int, path: Path, reader) -> pl.DataFrame | None:
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            hit = cache.get(season)
            if not hit or hit[0] != mtime:
                try:
                    hit = (mtime, reader(path))
                except Exception:  # noqa: BLE001 - any parse failure means "try again later"
                    # The CSV is rewritten in place; a read mid-write can fail. Serve the
                    # previous copy (if any) and retry on the next request.
                    return hit[1] if hit else None
                cache[season] = hit
            return hit[1]

    def roster(self, season: int) -> Roster | None:
        """The season's roster files, written by the web build before meta.json, so
        meta.json's mtime is the reload signal here too. None if the build had none."""
        meta_path = self.web_root / str(season) / "meta.json"
        try:
            mtime = meta_path.stat().st_mtime
        except FileNotFoundError:
            return None
        with self._lock:
            hit = self._rosters.get(season)
            if hit and hit[0] == mtime:
                return hit[1]
            d = meta_path.parent
            files = [d / f"{n}.parquet" for n in ("roster_players", "roster_departures", "roster_summary")]
            r = None
            if all(f.exists() for f in files):
                careers = d / "roster_careers.parquet"
                r = Roster(*(pl.read_parquet(f) for f in files),
                           careers=pl.read_parquet(careers) if careers.exists() else None)
            self._rosters[season] = (mtime, r)
            return r

    def accuracy_log(self, season: int) -> pl.DataFrame | None:
        """Season-to-date lines appended by ``cfbr weekly``."""
        path = self.out_dir / str(season) / "accuracy.csv"
        return pl.read_csv(path) if path.exists() else None
