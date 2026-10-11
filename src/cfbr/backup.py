"""Nightly copy of the data that can't be rebuilt: the CFBD raw cache (can't be
refetched under the free-tier quota), the paper-bet ledgers with their hash chain and
history, and the news archive.

``rsync -a --delete`` into ``[backup].target``. Files deleted or changed since the last
run are moved to ``target/_previous/{date}/`` instead of being lost, and those folders
are pruned after ``keep_previous_days``, so an accidental deletion can be undone for a
while but a deliberate purge (DECISIONS: delete unreliable data) still takes effect.
Each successful run writes ``target/LAST_BACKUP.json``; the health check reads it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cfbr.config import REPO_ROOT, Config

STAMP = "LAST_BACKUP.json"
PREVIOUS = "_previous"


def target(cfg: Config) -> Path:
    return Path(cfg.backup.target).expanduser()


def sources(cfg: Config, root: Path = REPO_ROOT) -> list[Path]:
    """Paths to copy, relative to ``root`` (only those that exist)."""
    out_dir = cfg.paths.out_dir
    rel = [cfg.paths.raw_dir, root / "data" / "news"]
    if cfg.backup.include_db:
        rel.append(cfg.paths.db_path)
    for season in sorted(p for p in out_dir.glob("*") if p.is_dir() and p.name.isdigit()):
        rel += sorted(season.glob("bets*.csv")) + [season / "ledger_chain.jsonl", season / "ledger_history"]
    paths = [p if p.is_absolute() else root / p for p in rel]
    return [p.relative_to(root) for p in paths if p.exists()]


def run_backup(cfg: Config, root: Path = REPO_ROOT, now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    dest = target(cfg)
    dest.mkdir(parents=True, exist_ok=True)
    src = sources(cfg, root)
    prev = dest / PREVIOUS / now.strftime("%Y%m%d")
    # --relative keeps data/raw/... paths under dest; --delete only touches the listed trees.
    # A trailing slash on directories makes rsync follow a symlinked dir (a worktree's data/raw).
    args = [f"{p}/" if (root / p).is_dir() else str(p) for p in src]
    cmd = ["rsync", "-a", "--relative", "--delete", "--backup", f"--backup-dir={prev}",
           "--exclude", f"/{PREVIOUS}/", *args, str(dest) + "/"]
    subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True)
    pruned = prune_previous(dest, cfg.backup.keep_previous_days, now)
    stamp = {"finished_at": now.isoformat(timespec="seconds"), "paths": [str(p) for p in src],
             "pruned": pruned}
    (dest / STAMP).write_text(json.dumps(stamp, indent=2))
    return stamp


def prune_previous(dest: Path, keep_days: int, now: datetime) -> list[str]:
    cutoff = (now - timedelta(days=keep_days)).strftime("%Y%m%d")
    gone = []
    for d in sorted((dest / PREVIOUS).glob("*")):
        if d.is_dir() and d.name.isdigit() and d.name < cutoff:
            shutil.rmtree(d)
            gone.append(d.name)
    return gone


def last_backup(cfg: Config) -> datetime | None:
    f = target(cfg) / STAMP
    if not f.exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(f.read_text())["finished_at"])
    except (ValueError, KeyError, json.JSONDecodeError):
        return None


def chain_head(out_dir: Path, season: int) -> str | None:
    """Short summary of the ledger hash chain's newest entry, e.g. for an off-machine timestamp."""
    chain = out_dir / str(season) / "ledger_chain.jsonl"
    if not chain.exists():
        return None
    lines = [ln for ln in chain.read_text().splitlines() if ln.strip()]
    if not lines:
        return None
    e = json.loads(lines[-1])
    return f"ledger chain {season}: {len(lines)} entries, head {e['chain'][:16]} ({e['ts']})"
