"""Orchestration: the advisory step inside `cfbr weekly`, a history mode for the shakedown,
and the alert text. Nothing here raises into the weekly run's exit code (cli._anomaly_step
catches everything)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from cfbr.anomalies.checks import run_checks, run_db_checks
from cfbr.anomalies.records import Anomaly, merge, read_doc, render_md, write_doc
from cfbr.config import REPO_ROOT, Config


def thresholds_path(cfg: Config) -> Path:
    p = Path(cfg.anomalies.thresholds)
    return p if p.is_absolute() else REPO_ROOT / p


def week_dir(cfg: Config, season: int, week: int) -> Path:
    return cfg.paths.out_dir / str(season) / f"w{week:02d}"


def latest_week(cfg: Config, season: int) -> int:
    weeks = [int(m.group(1)) for d in (cfg.paths.out_dir / str(season)).glob("w*")
             if d.is_dir() and (m := re.fullmatch(r"w(\d\d)", d.name))]
    if not weeks:
        raise FileNotFoundError(f"no out/{season}/wNN directories")
    return max(weeks)


def run_step(cfg: Config, season: int, week: int, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(UTC)
    wd = week_dir(cfg, season, week)
    found = run_checks(cfg.paths.db_path, wd, thresholds_path(cfg), season, week, cfg.anomalies)
    path = wd / "anomalies.json"
    doc, added = merge(read_doc(path), found, season, week, now)
    write_doc(path, doc)
    (wd / "anomalies.md").write_text(render_md(doc))
    return [row for row in added if row["alert"]]


def run_history(cfg: Config, season: int) -> list[Anomaly]:
    with duckdb.connect(str(cfg.paths.db_path), read_only=True) as con:
        weeks = [w for (w,) in con.execute(
            "SELECT DISTINCT week FROM games WHERE season = ? AND season_type = 'regular' AND completed "
            "ORDER BY week", [season]).fetchall()]
    found: list[Anomaly] = []
    for week in weeks:
        found.extend(run_db_checks(cfg.paths.db_path, season, week, cfg.anomalies))
    return found


def alert_text(rows: list[dict], limit: int = 15) -> str:
    lines = [f"• {r['check']} {r['message']}{' (to investigate)' if r['investigate'] else ''}" for r in rows[:limit]]
    if len(rows) > limit:
        lines.append(f"… and {len(rows) - limit} more (see anomalies.md)")
    return "\n".join(lines)
