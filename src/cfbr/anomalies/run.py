"""Orchestration: the advisory step inside `cfbr weekly`, a history mode for the shakedown,
and the alert text. Nothing here raises into the weekly run's exit code (cli._anomaly_step
catches everything)."""

from __future__ import annotations

import fcntl
import json
import re
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from cfbr.anomalies.checks import run_checks, run_db_checks
from cfbr.anomalies.records import SEVERITY, Anomaly, merge, read_doc, render_md, update_doc
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
    added: list[dict] = []

    def apply(existing: dict | None) -> dict:
        doc, new = merge(existing, found, season, week, now)
        added.extend(new)
        return doc

    doc = update_doc(path, apply)
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


TRANSIENT_STATUSES = {408, 409, 429}


def plan_targets(doc: dict, force_ids: tuple[str, ...], max_items: int) -> tuple[list[dict], list[dict]]:
    if force_ids:
        rows = [r for r in doc["items"] if r["id"] in force_ids]
    else:
        rows = [r for r in doc["items"] if r["status"] == "pending"]
    rows.sort(key=lambda r: (SEVERITY.get(r["check"], 99), str(r["key"])))
    return rows[:max_items], rows[max_items:]


def classify_error(e: BaseException) -> str:
    """transient: retry on a later run; auth: stop, the key is wrong; budget: credits or spend
    limit exhausted; item: this request is bad (record it, move on). anthropic.APIConnectionError
    and APITimeoutError carry no status_code; their class names contain Connection/Timeout."""
    status = getattr(e, "status_code", None)
    kind = getattr(e, "type", None)
    if status is None:
        name = type(e).__name__
        transient = isinstance(e, (ConnectionError, TimeoutError)) or "Connection" in name or "Timeout" in name
        return "transient" if transient else "item"
    if status in (401, 403):
        return "auth"
    if status == 402 or kind == "billing_error" or type(e).__name__ == "EnforcedSpendLimitReachedError":
        return "budget"
    if status == 400 and "credit balance" in str(e).lower():
        return "budget"
    if status in TRANSIENT_STATUSES or status >= 500:
        return "transient"
    return "item"


def try_lock(path: Path):
    fh = open(path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    return fh


def _patch(doc_path: Path, ids: list[str], patch: dict) -> None:
    def apply(doc: dict | None) -> dict:
        for row in (doc or {"items": []})["items"]:
            if row["id"] in ids:
                status = patch.get("status")
                row.update({k: v for k, v in patch.items() if k != "status"})
                if status:
                    if row["status"] == "resolved":
                        row["resolved_from"] = status
                    else:
                        row["status"] = status
        return doc
    update_doc(doc_path, apply)


def _context(doc: dict, row: dict) -> list[dict]:
    def related(r: dict) -> bool:
        return r["check"] == "B2" and ((row.get("team") and r.get("team") == row["team"])
                                       or (row.get("game_id") and r.get("game_id") == row["game_id"]))
    return [r for r in doc["items"] if related(r)]


def _label(doc: dict) -> str:
    return doc.get("title") or f"w{int(doc['week']):02d}"


def _ledger_total(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd"] for line in path.read_text().splitlines() if line.strip())


def investigate_doc(cfg: Config, doc_path: Path, client, *, now: datetime | None = None,
                    force_ids: tuple[str, ...] = (), clock=time.monotonic) -> dict:
    """One run at a time: a run can last hours and two would investigate the same items and
    replace the snapshot under each other's query children."""
    lock_path = cfg.investigate.snapshot.parent / ".investigate.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = try_lock(lock_path)
    if lock is None:
        return {"status": "busy", "items": [], "skipped_volume": 0, "skipped_budget": 0, "errors": 0,
                "stopped": None, "cost_usd": 0.0, "label": ""}
    try:
        return _investigate_locked(cfg, doc_path, client, now=now, force_ids=force_ids, clock=clock)
    finally:
        lock.close()


def _investigate_locked(cfg: Config, doc_path: Path, client, *, now: datetime | None,
                        force_ids: tuple[str, ...], clock) -> dict:
    from cfbr.anomalies import ledger
    from cfbr.anomalies.agent import investigate
    from cfbr.anomalies.prompt import build_system, schema_text
    from cfbr.anomalies.tools import ToolSession

    now = now or datetime.now(UTC)
    icfg = cfg.investigate
    doc = read_doc(doc_path)
    summary = {"status": "nothing", "items": [], "skipped_volume": 0, "skipped_budget": 0, "errors": 0,
               "stopped": None, "cost_usd": 0.0, "label": _label(doc) if doc else ""}
    if doc is None:
        return {**summary, "status": "no_doc"}
    todo, overflow = plan_targets(doc, force_ids, icfg.max_items)
    if not todo:
        return summary
    if ledger.month_to_date(icfg.ledger, now) >= icfg.monthly_budget_usd:
        _patch(doc_path, [r["id"] for r in todo + overflow], {"status": "skipped_budget"})
        return {**summary, "status": "budget", "skipped_budget": len(todo) + len(overflow)}
    lock = try_lock(cfg.paths.db_path.parent / ".weekly.lock")
    if lock is None:
        return {**summary, "status": "locked"}
    try:
        if cfg.paths.db_path.with_name(cfg.paths.db_path.name + ".wal").exists():
            return {**summary, "status": "locked"}
        fd, tmp = tempfile.mkstemp(dir=icfg.snapshot.parent, suffix=".tmp")
        os.close(fd)
        try:
            shutil.copyfile(cfg.paths.db_path, tmp)
            os.replace(tmp, icfg.snapshot)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    finally:
        lock.close()
    if overflow:
        _patch(doc_path, [r["id"] for r in overflow], {"status": "skipped_volume"})
        summary["skipped_volume"] = len(overflow)
    system = build_system(schema_text(icfg.snapshot))
    summary["status"] = "done"
    ledger_before = _ledger_total(icfg.ledger)
    for i, row in enumerate(todo):
        if ledger.month_to_date(icfg.ledger, now) >= icfg.monthly_budget_usd:
            rest = [r["id"] for r in todo[i:]]
            _patch(doc_path, rest, {"status": "skipped_budget"})
            summary["skipped_budget"] += len(rest)
            summary["stopped"] = "budget"
            break
        session = ToolSession(icfg.snapshot, cfg.paths.out_dir, int(row["season"]), int(row["week"]), icfg)
        outcome = None
        try:
            try:
                outcome = investigate(client, row, _context(doc, row), session, system, icfg, icfg.ledger, clock)
            finally:
                transcript = icfg.transcripts / str(row["season"]) / f"w{int(row['week']):02d}" / f"{row['id']}.json"
                transcript.parent.mkdir(parents=True, exist_ok=True)
                transcript.write_text(json.dumps({"row": row, "queries": session.log, "outcome": outcome},
                                                 indent=2, default=str) + "\n")
                session.close()
        except Exception as e:  # noqa: BLE001 - only API/agent failures land here; doc writes below are not item errors
            kind = classify_error(e)
            error = f"{type(e).__name__}: {e}"[:300]
            if kind == "auth":
                summary["stopped"] = "auth"
                summary["auth_error"] = error
                break
            if kind == "budget":
                rest = [r["id"] for r in todo[i:]]
                _patch(doc_path, rest, {"status": "skipped_budget", "last_error": error})
                summary["skipped_budget"] += len(rest)
                summary["stopped"] = "budget"
                break
            attempts = int(row.get("attempts", 0)) + 1
            if kind == "transient" and attempts < icfg.max_attempts:
                _patch(doc_path, [row["id"]], {"attempts": attempts, "last_error": error})
            else:
                _patch(doc_path, [row["id"]], {"status": "error", "attempts": attempts, "last_error": error})
                summary["errors"] += 1
            continue
        _patch(doc_path, [row["id"]], {"status": "investigated", "verdict": outcome,
                                       "investigated_at": now.isoformat()})
        summary["items"].append({"id": row["id"], "check": row["check"], "message": row["message"],
                                 **{k: outcome[k] for k in ("verdict", "confidence", "summary",
                                                            "model_impact", "reason")}})
    doc_path.with_suffix(".md").write_text(render_md(read_doc(doc_path)))
    summary["cost_usd"] = round(_ledger_total(icfg.ledger) - ledger_before, 2)
    return summary


def investigate_alert(summary: dict) -> tuple[str, str, str] | None:
    status = summary.get("status")
    if status == "budget":
        return ("cfbr investigate: monthly budget reached",
                f"{summary['skipped_budget']} item(s) marked skipped_budget; raise [investigate].monthly_budget_usd "
                "or wait for next month.", "default")
    if status != "done":
        return None
    if summary.get("stopped") == "auth":
        body = [f"Anthropic key rejected: {summary.get('auth_error', '')}"]
        body += [f"• {i['check']} {i['message']}: {i['verdict']}/{i['confidence']} — {i['summary']}; {i['model_impact']}"
                 for i in summary["items"]]
        return "cfbr investigate: Anthropic key rejected", "\n".join(body), "high"
    items = summary["items"]
    counts = {v: sum(1 for i in items if i["verdict"] == v) for v in ("data_bug", "real", "inconclusive")}
    title = (f"cfbr anomalies {summary['label']} — {counts['data_bug']} data bug, {counts['real']} real, "
             f"{counts['inconclusive']} inconclusive (${summary['cost_usd']:.2f})")
    lines = [f"• {i['check']} {i['message']}: {i['verdict']}/{i['confidence']} — {i['summary']}; {i['model_impact']}"
             for i in items]
    for key, text in (("skipped_volume", "not investigated (volume)"), ("skipped_budget", "skipped (budget)"),
                      ("errors", "failed after retries")):
        if summary.get(key):
            lines.append(f"{summary[key]} {text}")
    if summary.get("stopped") == "auth":
        lines.append(f"Stopped: Anthropic key rejected ({summary.get('auth_error', '')})")
    material = any(i["verdict"] == "data_bug" and i["model_impact"] == "material" for i in items)
    priority = "high" if material or summary.get("stopped") == "auth" else "default"
    if not lines:
        return None
    return title, "\n".join(lines), priority
