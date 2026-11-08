"""Anomaly records and the per-week anomalies.json document.

Ids are a stable hash of (check, season, week, key), so a same-week re-run of `cfbr weekly`
recognizes items it has already seen: their status and verdict are kept, new ids are added,
and ids that no longer fire are marked resolved.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

SEVERITY = {"A1": 0, "A5": 1, "A2": 2, "A3": 3, "A4": 4, "B1": 5, "B3": 6, "B2": 7, "ERR": 8}


def anomaly_id(check: str, season: int, week: int, key: str) -> str:
    return hashlib.sha1(f"{check}|{season}|{week}|{key}".encode()).hexdigest()[:12]


@dataclass
class Anomaly:
    check: str  # "A1".."A5", "B1".."B3", or "ERR" for a check that crashed
    season: int
    week: int
    key: str  # game id or team; unique within (check, season, week)
    message: str
    value: float | None = None
    threshold: float | None = None
    investigate: bool = False
    alert: bool = True
    game_id: int | None = None
    team: str | None = None

    @property
    def id(self) -> str:
        return anomaly_id(self.check, self.season, self.week, self.key)


def initial_status(a: Anomaly) -> str:
    if a.investigate:
        return "pending"
    return "plain" if a.alert else "context"


def merge(existing: dict | None, fresh: list[Anomaly], season: int, week: int,
          now: datetime) -> tuple[dict, list[dict]]:
    """New document plus the rows added by this run (the ones to alert on)."""
    stamp = now.isoformat()
    old = {row["id"]: row for row in (existing or {}).get("items", [])}
    items: list[dict] = []
    added: list[dict] = []
    seen: set[str] = set()
    for a in fresh:
        seen.add(a.id)
        prev = old.get(a.id)
        if prev is None:
            row = {"id": a.id, **asdict(a), "updated_at": stamp,
                   "status": initial_status(a), "first_seen_at": stamp, "verdict": None}
            added.append(row)
        else:
            status = prev.get("resolved_from", initial_status(a)) if prev["status"] == "resolved" else prev["status"]
            base = {k: v for k, v in prev.items() if k != "resolved_from"}
            row = {**base, "id": a.id, **asdict(a), "updated_at": stamp, "status": status}
        items.append(row)
    for row_id, prev in old.items():
        if row_id in seen:
            continue
        if prev["status"] == "resolved":
            items.append(prev)
        else:
            items.append({**prev, "status": "resolved", "resolved_from": prev["status"], "updated_at": stamp})
    items.sort(key=lambda r: (SEVERITY.get(r["check"], 99), str(r["key"])))
    return {"season": season, "week": week, "generated_at": stamp, "items": items}, added


def read_doc(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def write_doc(path: Path, doc: dict) -> None:
    """Atomic: write a temp file next to the target, then rename over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def render_md(doc: dict) -> str:
    lines = [f"# Data anomalies — {doc['season']} week {doc['week']}", "",
             f"Generated {doc['generated_at']}.", ""]
    if not doc["items"]:
        return "\n".join(lines + ["No anomalies.", ""])
    lines += ["| Status | Check | Detail |", "|---|---|---|"]
    for row in doc["items"]:
        detail = " ".join(row["message"].split()).replace("|", "\\|")
        verdict = row.get("verdict")
        if verdict:
            summary = " ".join(str(verdict.get("summary", "")).split())
            detail += f" — **{verdict.get('verdict')}**: {summary}".replace("|", "\\|")
        lines.append(f"| {row['status']} | {row['check']} | {detail} |")
    return "\n".join(lines + [""])
