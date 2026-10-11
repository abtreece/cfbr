"""Ledger history and tamper evidence.

After every weekly run the ledger is copied to ``out/{season}/ledger_history/`` (when
it changed) and its SHA-256 is appended to ``ledger_chain.jsonl``, each entry chained
to the previous one. ``verify`` re-checks the chain and enforces the rule that a
recorded bet's terms never change: game metadata may be filled in later, results get
added at settlement, but side, line, prices and placement time are fixed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

TERMS = ["placed_at", "kickoff_ts", "side", "team", "model_margin", "line", "edge", "best_line", "best_provider",
         "odds", "stake", "best_odds", "recorded_at", "edge_cents"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(out_dir: Path, season: int) -> tuple[Path, Path, Path]:
    d = out_dir / str(season)
    return d / "bets.csv", d / "ledger_history", d / "ledger_chain.jsonl"


def _entries(chain: Path) -> list[dict]:
    return [json.loads(x) for x in chain.read_text().splitlines() if x.strip()] if chain.exists() else []


def record(out_dir: Path, season: int, now: datetime | None = None) -> dict | None:
    """Snapshot the ledger if it changed since the last entry. Returns the new entry."""
    ledger, hist, chain = _paths(out_dir, season)
    if not ledger.exists():
        return None
    sha = _sha(ledger)
    entries = _entries(chain)
    if entries and entries[-1]["sha256"] == sha:
        return None
    now = now or datetime.now(UTC)
    hist.mkdir(parents=True, exist_ok=True)
    name = f"bets-{now:%Y%m%dT%H%M%SZ}.csv"
    shutil.copy2(ledger, hist / name)
    prev = entries[-1]["chain"] if entries else ""
    df = pl.read_csv(ledger, infer_schema_length=0)
    entry = {"ts": now.isoformat(timespec="seconds"), "file": name, "sha256": sha, "prev": prev,
             "chain": hashlib.sha256((prev + sha).encode()).hexdigest(), "bets": df.height,
             "settled": int((df["status"] != "open").sum()) if "status" in df.columns else 0}
    with open(chain, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def verify(out_dir: Path, season: int) -> list[str]:
    ledger, hist, chain = _paths(out_dir, season)
    entries = _entries(chain)
    if not entries:
        return []
    problems, prev = [], ""
    for e in entries:
        if e["prev"] != prev or e["chain"] != hashlib.sha256((prev + e["sha256"]).encode()).hexdigest():
            problems.append(f"ledger hash chain broken at {e['ts']}")
            break
        f = hist / e["file"]
        if not f.exists() or _sha(f) != e["sha256"]:
            problems.append(f"ledger history file {e['file']} missing or altered")
        prev = e["chain"]
    if ledger.exists() and _sha(ledger) != entries[-1]["sha256"]:
        problems.append("bets.csv changed outside the weekly run (not in the hash chain)")
    # Terms of every recorded bet must match across all versions.
    first: dict[int, tuple] = {}
    for e in entries:
        f = hist / e["file"]
        if not f.exists():
            continue
        df = pl.read_csv(f, infer_schema_length=0)
        cols = [c for c in TERMS if c in df.columns]
        seen = set()
        for r in df.select("game_id", *cols).iter_rows(named=True):
            gid = r["game_id"]
            seen.add(gid)
            terms = tuple((c, r[c]) for c in cols)
            if gid in first and dict(first[gid]) != {c: v for c, v in terms if c in dict(first[gid])}:
                problems.append(f"recorded bet {gid} changed terms in {e['file']}")
            first.setdefault(gid, terms)
        gone = set(first) - seen
        if gone:
            problems.append(f"{len(gone)} recorded bets missing from {e['file']}")
    return sorted(set(problems))
