"""Read-only access to the files the web app shows, reloaded when they change.

``cfbr weekly`` rewrites ``out/web/{season}/`` while the server is running;
each season's ``meta.json`` is written last, so its mtime is the reload signal. A
parquet newer than its ``meta.json`` means a rebuild is in progress: the previous copy
is kept (or, with nothing cached, the season is reported not ready) rather than pairing
new ratings with old games.
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
        self._sims: dict[tuple[int, int], tuple[float, tuple[pl.DataFrame, str]]] = {}
        self._accuracy: dict[int, tuple[float, pl.DataFrame]] = {}
        self._run_list: tuple[float, list[str], list[str]] | None = None  # (results mtime, run ids, pending dirs)

    def seasons(self) -> list[int]:
        if not self.web_root.is_dir():
            return []
        return sorted(int(p.parent.name) for p in self.web_root.glob("*/meta.json") if p.parent.name.isdigit())

    def season(self, season: int) -> Season | None:
        """The season's web data, or None when there is none (or it is mid-rebuild with
        nothing cached yet; see :meth:`rebuilding`)."""
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
            files = [d / "ratings.parquet", d / "games.parquet"]
            try:
                meta = json.loads(meta_path.read_text())
                ratings, games = (pl.read_parquet(f) for f in files)
                consistent = (all(f.stat().st_mtime <= mtime for f in files)
                              and meta_path.stat().st_mtime == mtime)
            except (OSError, ValueError, pl.exceptions.PolarsError):
                consistent = False  # a file vanished or was half-written: a rebuild in progress
            if not consistent:
                return hit[1] if hit else None
            s = Season(meta, _ensure(ratings, POLL_COLS),
                       _ensure(games, [f"{side}_{c}" for side in ("home", "away") for c in POLL_COLS]))
            self._seasons[season] = (mtime, s)
            return s

    def rebuilding(self, season: int) -> bool:
        """True when the season's files exist but :meth:`season` can't serve a consistent set yet."""
        return (self.web_root / str(season) / "meta.json").exists() and self.season(season) is None

    def latest(self) -> int | None:
        seasons = self.seasons()
        return seasons[-1] if seasons else None

    def runs(self) -> list[Run]:
        """Backtest runs under results/, newest first. The directory is rescanned only when
        its mtime changes (a run added or removed); a run directory without metrics.json
        yet (in progress) is re-checked until the file lands. Each metrics.json is read once."""
        try:
            dir_mtime = self.results_dir.stat().st_mtime
        except FileNotFoundError:
            return []
        with self._lock:
            listing = self._run_list
            if listing is None or listing[0] != dir_mtime:
                dirs = sorted((p for p in self.results_dir.iterdir() if p.is_dir()), reverse=True)
                ids = [p.name for p in dirs if (p / "metrics.json").exists()]
                pending = [p.name for p in dirs if not (p / "metrics.json").exists()]
                listing = self._run_list = (dir_mtime, ids, pending)
            elif listing[2]:
                landed = [n for n in listing[2] if (self.results_dir / n / "metrics.json").exists()]
                if landed:
                    ids = sorted(set(listing[1]) | set(landed), reverse=True)
                    listing = self._run_list = (dir_mtime, ids, [n for n in listing[2] if n not in landed])
            out = []
            for run_id in listing[1]:
                hit = self._runs.get(run_id)
                if hit is None:
                    path = self.results_dir / run_id / "metrics.json"
                    try:
                        hit = (path.stat().st_mtime, Run(run_id, json.loads(path.read_text())))
                    except (OSError, ValueError):
                        continue  # removed or half-written; picked up on the next rescan
                    self._runs[run_id] = hit
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

    def sim_weeks(self, season: int) -> list[int]:
        """Weeks with a season simulation (``out/{season}/wNN/season_sim.csv``, written by weekly)."""
        d = self.out_dir / str(season)
        if not d.is_dir():
            return []
        return sorted(int(p.parent.name[1:]) for p in d.glob("w[0-9][0-9]/season_sim.csv"))

    def season_sim(self, season: int, week: int) -> tuple[pl.DataFrame, str] | None:
        """(table, notes markdown) for a week's simulation, reloaded when the CSV changes.
        The notes are the .md's text above its table, so the page says exactly what the
        simulation's own write-up says."""
        path = self.out_dir / str(season) / f"w{week:02d}" / "season_sim.csv"
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        key = (season, week)
        with self._lock:
            hit = self._sims.get(key)
            if hit and hit[0] == mtime:
                return hit[1]
            try:
                df = pl.read_csv(path)
            except Exception:  # noqa: BLE001 - mid-write; serve the previous copy if any
                return hit[1] if hit else None
            md = path.with_suffix(".md")
            notes = []
            try:
                text = md.read_text() if md.exists() else ""
            except OSError:  # being rewritten; the notes are optional
                text = ""
            for line in text.splitlines():
                if line.startswith("|"):
                    break
                if line.strip() and not line.startswith("#"):
                    notes.append(line.strip())
            value = (df, "\n".join(notes))
            self._sims[key] = (mtime, value)
            return value

    def accuracy_log(self, season: int) -> pl.DataFrame | None:
        """Season-to-date lines appended by ``cfbr weekly``, reloaded when the file changes."""
        return self._cached_csv(self._accuracy, season, self.out_dir / str(season) / "accuracy.csv", pl.read_csv)
