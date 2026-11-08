# Data Anomaly Checks (PR 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add advisory, deterministic data-anomaly checks (A1–A5, B1–B3) that run inside `cfbr weekly` after ranking, write `out/{season}/wNN/anomalies.{json,md}`, and push new items to ntfy — without ever changing the weekly exit code.

**Architecture:** A new package `src/cfbr/anomalies/` with three layers: `records.py` (the `Anomaly` record, stable ids, the JSON document with re-run merge, markdown), `checks.py` (pure check functions over a read-only DuckDB connection and the week's `rank` output), and `run.py` (orchestration: one guarded step for `weekly`, a history mode, alert text). `baselines.py` computes the B1/B3 thresholds once from history into a committed `config/anomaly_thresholds.toml`. Spec: `docs/superpowers/specs/2026-11-08-anomaly-investigation-design.md`.

**Tech Stack:** Python 3.12, uv, duckdb, polars, typer, pydantic, pytest. No new dependencies.

## Global Constraints

- No new dependencies in this PR; no `anthropic` import anywhere.
- Advisory only: the anomaly step never raises out of `_weekly_run` and never changes an exit code (0/3/4/5/6 keep their meanings).
- Checks open DuckDB with `read_only=True`.
- No module under `src/cfbr/model/`, `src/cfbr/bets/`, `src/cfbr/report/` or `src/cfbr/web/` imports `cfbr.anomalies` (tested). `cfbr.anomalies` may import from them.
- Betting lines are never model inputs: B3 reads `closing_spread` from `predictions.csv` only to compare; nothing reads `anomalies.*` back into the pipeline.
- Thresholds (config `[anomalies]`): A2 gap > 8; A3 plays < 110 or > 250; A4 duplicate rows > 3; B2 quantiles 0.0025 / 0.9975; B1/B3 quantile 0.995. Week buckets for B1: "1-3", "4-7", "8+".
- Severity order (sorting, and later investigation order): A1, A5, A2, A3, A4, B1, B3, B2, ERR.
- Statuses written by this PR: `pending` (investigate items), `plain` (alerted, not investigated), `context` (B2: stored, never alerted), `resolved` (no longer fires on a same-week re-run).
- Don't run `cfbr weekly`, `bets`, `odds-snapshot`, `backup` or `check` locally (production-only, see CLAUDE.md). No API calls anywhere in this PR.
- Python files use 4-space indentation (existing repo style). Commit messages: conventional commits, no Co-Authored-By line.
- Work on branch `feat/anomaly-checks` (Task 0 renames the current `docs/anomaly-investigation-spec` branch; the spec and this plan ship in the same PR).

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/cfbr/config.py` | modify | `Anomalies` pydantic section; `Config.anomalies` |
| `config/default.toml` | modify | `[anomalies]` block with comments |
| `src/cfbr/anomalies/__init__.py` | create | package docstring |
| `src/cfbr/anomalies/records.py` | create | `Anomaly`, `anomaly_id`, `initial_status`, `merge`, `read_doc`, `write_doc`, `render_md` |
| `src/cfbr/anomalies/checks.py` | create | `week_bucket`, A1–A5, B1–B3, `load_thresholds`, `run_db_checks`, `run_checks` |
| `src/cfbr/anomalies/baselines.py` | create | `regular_weeks`, `b1_baseline`, `b3_baseline`, `write_thresholds`, `compute_baselines` |
| `src/cfbr/anomalies/run.py` | create | `thresholds_path`, `week_dir`, `latest_week`, `run_step`, `run_history`, `alert_text` |
| `config/anomaly_thresholds.toml` | create (generated) | frozen B1/B3 thresholds |
| `src/cfbr/cli.py` | modify | `_anomaly_step` in `_weekly_run`; `anomalies` command |
| `tests/test_anomalies_records.py` | create | ids, merge, statuses, atomic write, markdown |
| `tests/test_anomalies_checks.py` | create | each check fires / doesn't on synthetic DuckDB and CSVs; guard |
| `tests/test_anomalies_run.py` | create | `run_step` end to end, no re-alert, resolve; `_anomaly_step` never raises; thresholds round trip; isolation; real-data firing bands |
| `README.md`, `DECISIONS.md` | modify | command row, cron/ops note, decision lines |

---

### Task 0: Branch

- [ ] **Step 1: Rename the branch**

Run: `git branch -m feat/anomaly-checks`
Then: `git branch --show-current`
Expected: `feat/anomaly-checks`

---

### Task 1: Config section and anomaly records

**Files:**
- Modify: `src/cfbr/config.py` (add class before `class Config`, add field to `Config`)
- Modify: `config/default.toml` (append block after `[sim]` section)
- Create: `src/cfbr/anomalies/__init__.py`
- Create: `src/cfbr/anomalies/records.py`
- Test: `tests/test_anomalies_records.py`

**Interfaces:**
- Produces:
  - `cfbr.config.Anomalies` with fields `enabled: bool`, `a2_max_score_gap: int`, `a3_min_plays: int`, `a3_max_plays: int`, `a4_max_duplicates: int`, `b2_quantiles: tuple[float, float]`, `baseline_quantile: float`, `thresholds: Path`; `Config.anomalies: Anomalies`.
  - `Anomaly` dataclass (fields below), property `Anomaly.id -> str`.
  - `anomaly_id(check: str, season: int, week: int, key: str) -> str` (12 hex chars).
  - `initial_status(a: Anomaly) -> str`.
  - `merge(existing: dict | None, fresh: list[Anomaly], season: int, week: int, now: datetime) -> tuple[dict, list[dict]]` — returns `(document, newly_added_rows)`.
  - `read_doc(path: Path) -> dict | None`, `write_doc(path: Path, doc: dict) -> None`, `render_md(doc: dict) -> str`.
  - `SEVERITY: dict[str, int]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_anomalies_records.py`:

```python
import json
from datetime import UTC, datetime, timedelta

from cfbr.anomalies.records import Anomaly, anomaly_id, initial_status, merge, read_doc, render_md, write_doc
from cfbr.config import Config

T0 = datetime(2026, 11, 8, 15, 0, tzinfo=UTC)


def _a(check="A2", key="401", investigate=True, alert=True, msg="gap"):
    return Anomaly(check=check, season=2026, week=9, key=key, message=msg, investigate=investigate, alert=alert)


def test_config_has_anomaly_defaults():
    a = Config().anomalies
    assert a.enabled and a.a2_max_score_gap == 8 and (a.a3_min_plays, a.a3_max_plays) == (110, 250)
    assert a.a4_max_duplicates == 3 and a.b2_quantiles == (0.0025, 0.9975) and a.baseline_quantile == 0.995


def test_id_is_stable_and_key_sensitive():
    assert _a().id == anomaly_id("A2", 2026, 9, "401") == _a(msg="different text").id
    assert len(_a().id) == 12
    assert _a(key="402").id != _a().id
    assert _a(check="A3").id != _a().id


def test_initial_status():
    assert initial_status(_a(investigate=True)) == "pending"
    assert initial_status(_a(investigate=False)) == "plain"
    assert initial_status(_a(investigate=False, alert=False)) == "context"


def test_merge_first_run_adds_everything_sorted_by_severity():
    doc, added = merge(None, [_a(check="B1", key="Iowa"), _a(check="A1", key="400", investigate=False)], 2026, 9, T0)
    assert [r["check"] for r in doc["items"]] == ["A1", "B1"]
    assert {r["check"] for r in added} == {"A1", "B1"}
    assert doc["season"] == 2026 and doc["week"] == 9
    row = doc["items"][1]
    assert row["status"] == "pending" and row["verdict"] is None and row["first_seen_at"] == T0.isoformat()


def test_merge_rerun_keeps_status_and_verdict_and_adds_only_new():
    doc, _ = merge(None, [_a(key="401")], 2026, 9, T0)
    doc["items"][0]["status"] = "investigated"
    doc["items"][0]["verdict"] = {"verdict": "real"}
    t1 = T0 + timedelta(hours=6)
    doc2, added = merge(doc, [_a(key="401"), _a(key="402")], 2026, 9, t1)
    by_key = {r["key"]: r for r in doc2["items"]}
    assert by_key["401"]["status"] == "investigated" and by_key["401"]["verdict"] == {"verdict": "real"}
    assert by_key["401"]["first_seen_at"] == T0.isoformat() and by_key["401"]["updated_at"] == t1.isoformat()
    assert [r["key"] for r in added] == ["402"]


def test_merge_marks_vanished_items_resolved_and_restores_on_refire():
    doc, _ = merge(None, [_a(key="401")], 2026, 9, T0)
    doc2, added = merge(doc, [], 2026, 9, T0)
    assert doc2["items"][0]["status"] == "resolved" and doc2["items"][0]["resolved_from"] == "pending"
    assert added == []
    doc3, added3 = merge(doc2, [_a(key="401")], 2026, 9, T0)
    assert doc3["items"][0]["status"] == "pending" and added3 == []


def test_write_doc_is_atomic_and_round_trips(tmp_path):
    p = tmp_path / "w09" / "anomalies.json"
    doc, _ = merge(None, [_a()], 2026, 9, T0)
    write_doc(p, doc)
    assert read_doc(p) == doc
    assert not list(p.parent.glob("*.tmp"))
    assert read_doc(tmp_path / "missing.json") is None
    assert json.loads(p.read_text())["items"][0]["id"] == _a().id


def test_render_md_lists_items_and_says_when_empty():
    doc, _ = merge(None, [_a(msg="Rice @ UTSA: 38 duplicate play rows", check="A4")], 2026, 9, T0)
    md = render_md(doc)
    assert "2026 week 9" in md and "Rice @ UTSA: 38 duplicate play rows" in md and "| pending | A4 |" in md
    empty, _ = merge(None, [], 2026, 9, T0)
    assert "No anomalies." in render_md(empty)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_anomalies_records.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'cfbr.anomalies'`.

- [ ] **Step 3: Add the config section**

In `src/cfbr/config.py`, insert before `class Config(BaseModel):`:

```python
class Anomalies(BaseModel):
    """Advisory data checks run by `cfbr weekly` after ranking (cfbr/anomalies). They write
    out/{season}/wNN/anomalies.{json,md} and push new items to ntfy; they never block publishing."""
    enabled: bool = True
    a2_max_score_gap: int = 8  # |PBP max score - final|, home + away
    a3_min_plays: int = 110
    a3_max_plays: int = 250
    a4_max_duplicates: int = 3  # duplicate play rows in one game
    b2_quantiles: tuple[float, float] = (0.0025, 0.9975)  # single-game offensive PPA/play
    baseline_quantile: float = 0.995  # B1/B3 thresholds, computed by `cfbr anomalies --baselines`
    thresholds: Path = Path("config/anomaly_thresholds.toml")
```

In `class Config(BaseModel):`, add after `sim: Sim = Sim()`:

```python
    anomalies: Anomalies = Anomalies()
```

Append to `config/default.toml`:

```toml

[anomalies]
# Advisory data checks after `cfbr weekly` ranks (src/cfbr/anomalies). They write
# out/{season}/wNN/anomalies.{json,md} and push new items to ntfy; they never block publishing.
# A1 no play-by-play, A2 PBP score vs final, A3 play count, A4 duplicate plays, A5 FBS status
# change, B1 rating move, B2 single-game PPA (context only), B3 model vs current line.
enabled = true
a2_max_score_gap = 8
a3_min_plays = 110
a3_max_plays = 250
a4_max_duplicates = 3
b2_quantiles = [0.0025, 0.9975]
# B1/B3 thresholds are this quantile of history, frozen in `thresholds` by `cfbr anomalies --baselines`.
baseline_quantile = 0.995
thresholds = "config/anomaly_thresholds.toml"
```

- [ ] **Step 4: Create the package and records module**

`src/cfbr/anomalies/__init__.py`:

```python
"""Advisory data-anomaly checks run after `cfbr weekly` ranks (see docs/superpowers/specs/
2026-11-08-anomaly-investigation-design.md). Nothing in the model, bets, report or web
packages imports this package: its output is for people, never an input."""
```

`src/cfbr/anomalies/records.py`:

```python
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
        row = {"id": a.id, **asdict(a), "updated_at": stamp}
        prev = old.get(a.id)
        if prev is None:
            row.update(status=initial_status(a), first_seen_at=stamp, verdict=None)
            added.append(row)
        else:
            status = prev.get("resolved_from", initial_status(a)) if prev["status"] == "resolved" else prev["status"]
            row.update(status=status, first_seen_at=prev["first_seen_at"], verdict=prev.get("verdict"))
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
        detail = row["message"].replace("|", "\\|")
        verdict = row.get("verdict")
        if verdict:
            detail += f" — **{verdict.get('verdict')}**: {verdict.get('summary', '')}".replace("|", "\\|")
        lines.append(f"| {row['status']} | {row['check']} | {detail} |")
    return "\n".join(lines + [""])
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_anomalies_records.py -v`
Expected: 8 passed.

Also run: `uv run pytest tests/test_report.py tests/test_health.py -q` (config change sanity)
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/cfbr/config.py config/default.toml src/cfbr/anomalies/__init__.py src/cfbr/anomalies/records.py tests/test_anomalies_records.py
git commit -m "feat(anomalies): config section and anomaly records with stable ids"
```

---

### Task 2: Ingest checks A1–A5

**Files:**
- Create: `src/cfbr/anomalies/checks.py`
- Test: `tests/test_anomalies_checks.py`

**Interfaces:**
- Consumes: `Anomaly` (Task 1), `cfbr.config.Anomalies`.
- Produces (all `(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]`): `a1_no_pbp`, `a2_score_gap`, `a3_play_count`, `a4_duplicates`, `a5_fbs_status`. Constants `DROPPED_FOR_DUPES: tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_anomalies_checks.py`:

```python
import duckdb
import pytest

from cfbr.anomalies import checks
from cfbr.config import Anomalies

CFG = Anomalies()


def _db(path):
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE games (game_id BIGINT, season INTEGER, week INTEGER, season_type VARCHAR,
        completed BOOLEAN, home_team VARCHAR, away_team VARCHAR, home_class VARCHAR, away_class VARCHAR,
        home_points INTEGER, away_points INTEGER)""")
    con.execute("""CREATE TABLE plays_raw (play_id VARCHAR, game_id BIGINT, season INTEGER, week INTEGER,
        season_type VARCHAR, offense VARCHAR, defense VARCHAR, home VARCHAR, offense_score INTEGER,
        defense_score INTEGER, period INTEGER, clock_minutes INTEGER, clock_seconds INTEGER, down INTEGER,
        distance INTEGER, yards_to_goal INTEGER, play_type VARCHAR, play_text VARCHAR)""")
    con.execute("CREATE TABLE team_status (season INTEGER, team VARCHAR, classification VARCHAR, conference VARCHAR)")
    con.execute("""CREATE TABLE model_plays (season INTEGER, week INTEGER, season_type VARCHAR, game_id BIGINT,
        offense_team VARCHAR, offense_fcs BOOLEAN, ppa DOUBLE)""")
    return con


def _game(con, gid, home="Rice", away="UTSA", hp=21, ap=14, week=9, cls=("fbs", "fbs"), completed=True):
    con.execute("INSERT INTO games VALUES (?, 2026, ?, 'regular', ?, ?, ?, ?, ?, ?, ?)",
                [gid, week, completed, home, away, cls[0], cls[1], hp, ap])


def _plays(con, gid, n, home="Rice", away="UTSA", final=(21, 14), week=9, text="run for 3"):
    """n distinct plays; the last one carries the final score from the home offense's view."""
    rows = []
    for i in range(n):
        hs, as_ = final if i == n - 1 else (0, 0)
        rows.append([f"{gid}{i:04d}", gid, 2026, week, "regular", home, away, home, hs, as_,
                     1 + i // 50, 14 - (i % 15), i % 60, 1 + i % 4, 10, 75 - i % 50, "Rush", f"{text} #{i}"])
    con.executemany("INSERT INTO plays_raw VALUES (" + ", ".join(["?"] * 18) + ")", rows)


def test_a1_flags_completed_fbs_game_without_plays(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    _game(con, 1)
    _plays(con, 1, 170)
    _game(con, 2, home="Army", away="Navy")                         # no plays -> fires
    _game(con, 3, home="Fordham", away="Colgate", cls=("fcs", "fcs"))  # FCS-only: ignored
    _game(con, 4, home="Iowa", away="Ohio", completed=False)        # not completed: ignored
    out = checks.a1_no_pbp(con, 2026, 9, CFG)
    assert [(a.check, a.key, a.investigate, a.alert) for a in out] == [("A1", "2", False, True)]
    assert out[0].message == "Navy @ Army: completed, no play-by-play"


def test_a2_score_gap_over_threshold_only(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    _game(con, 1, hp=21, ap=14)
    _plays(con, 1, 170, final=(21, 14))   # matches
    _game(con, 2, home="Army", away="Navy", hp=28, ap=14)
    _plays(con, 2, 170, home="Army", away="Navy", final=(20, 14))  # gap 8: not > 8
    _game(con, 3, home="Iowa", away="Ohio", hp=30, ap=10)
    _plays(con, 3, 170, home="Iowa", away="Ohio", final=(14, 7))   # gap 19
    out = checks.a2_score_gap(con, 2026, 9, CFG)
    assert [(a.key, a.value, a.investigate) for a in out] == [("3", 19.0, True)]
    assert out[0].message == "Ohio @ Iowa: PBP max score 7-14 vs final 10-30 (gap 19)"


def test_a3_play_count_outside_range(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    for gid, n in [(1, 170), (2, 105), (3, 260), (4, 110)]:
        _game(con, gid, home=f"H{gid}", away=f"A{gid}")
        _plays(con, gid, n, home=f"H{gid}", away=f"A{gid}")
    out = checks.a3_play_count(con, 2026, 9, CFG)
    assert [(a.key, a.value) for a in out] == [("2", 105.0), ("3", 260.0)]
    assert out[0].message == "A2 @ H2: 105 plays (expected 110–250)" and out[0].investigate


def test_a4_counts_duplicate_rows_but_ignores_timeouts(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    _game(con, 1)
    _plays(con, 1, 170)
    dup = ["Rice", "UTSA", "Rice", 0, 0, 2, 7, 30, 2, 5, 40, "Pass Reception", "pass for 12"]
    con.executemany("INSERT INTO plays_raw VALUES (?, 1, 2026, 9, 'regular', " + ", ".join(["?"] * 13) + ")",
                    [[f"d{i}", *dup] for i in range(5)])            # 4 extra rows: > 3
    to = ["Rice", "UTSA", "Rice", 0, 0, 3, 5, 0, None, None, None, "Timeout", "Timeout RICE"]
    con.executemany("INSERT INTO plays_raw VALUES (?, 1, 2026, 9, 'regular', " + ", ".join(["?"] * 13) + ")",
                    [[f"t{i}", *to] for i in range(6)])             # timeouts never count
    _game(con, 2, home="Army", away="Navy")
    _plays(con, 2, 170, home="Army", away="Navy")
    con.executemany("INSERT INTO plays_raw VALUES (?, 2, 2026, 9, 'regular', " + ", ".join(["?"] * 13) + ")",
                    [[f"e{i}", "Army", "Navy", "Army", 0, 0, 1, 9, 0, 1, 10, 60, "Rush", "run for 2"] for i in range(4)])
    out = checks.a4_duplicates(con, 2026, 9, CFG)                   # game 2 has 3 extra: not > 3
    assert [(a.key, a.value) for a in out] == [("1", 4.0)]
    assert out[0].message == "UTSA @ Rice: 4 duplicate play rows"


def test_a5_flags_status_change_only_in_first_week(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    con.executemany("INSERT INTO team_status VALUES (?, ?, ?, NULL)", [
        (2025, "Sacramento State", "fcs"), (2026, "Sacramento State", "fbs"),   # moved up: fires
        (2025, "Rice", "fbs"), (2026, "Rice", "fbs"),                           # unchanged
        (2026, "Newco", "fbs"),                                                 # no row last season: fires
        (2025, "Delaware", "fcs"), (2026, "Delaware", "fbs")])                  # first game in week 1: not this week
    _game(con, 1, home="Sacramento State", away="Rice", week=9)
    _game(con, 2, home="Newco", away="Rice", week=10, completed=False)
    _game(con, 3, home="Newco", away="Rice", week=9)
    _game(con, 4, home="Delaware", away="Rice", week=1)
    _game(con, 5, home="Delaware", away="Rice", week=9)
    out = checks.a5_fbs_status(con, 2026, 9, CFG)
    assert sorted(a.key for a in out) == ["Newco", "Sacramento State"]
    msg = {a.key: a.message for a in out}
    assert msg["Sacramento State"] == "Sacramento State: fcs last season → fbs this season (check FBS status and pooling)"
    assert msg["Newco"] == "Newco: no row last season → fbs this season (check FBS status and pooling)"
    assert all(not a.investigate and a.alert for a in out)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_anomalies_checks.py -v`
Expected: FAIL — `ImportError: cannot import name 'checks' from 'cfbr.anomalies'`.

- [ ] **Step 3: Implement the ingest checks**

`src/cfbr/anomalies/checks.py`:

```python
"""Deterministic anomaly checks. Each takes a read-only DuckDB connection (ingest checks) or a
`rank` output file (model checks) and returns Anomaly records; none writes anything.

Baselines behind the thresholds (2022–2026, completed games with an FBS side) are in the spec:
docs/superpowers/specs/2026-11-08-anomaly-investigation-design.md.
"""

from __future__ import annotations

import duckdb

from cfbr.anomalies.records import Anomaly
from cfbr.config import Anomalies

# Rows excluded from the duplicate check: administrative rows legitimately repeat.
DROPPED_FOR_DUPES = ("Timeout", "End Period", "End of Half", "End of Game", "End of Regulation")

_WEEK_GAMES = """
    SELECT * FROM games
    WHERE season = ? AND week = ? AND season_type = 'regular' AND completed
      AND (home_class = 'fbs' OR away_class = 'fbs')
"""


def _label(home: str, away: str) -> str:
    return f"{away} @ {home}"


def a1_no_pbp(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    rows = con.execute(f"""
        SELECT g.game_id, g.home_team, g.away_team FROM ({_WEEK_GAMES}) g
        WHERE NOT EXISTS (SELECT 1 FROM plays_raw p WHERE p.game_id = g.game_id)
        ORDER BY g.game_id""", [season, week]).fetchall()
    return [Anomaly("A1", season, week, str(gid), f"{_label(h, a)}: completed, no play-by-play",
                    investigate=False, game_id=gid) for gid, h, a in rows]


def a2_score_gap(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    rows = con.execute(f"""
        WITH m AS (
            SELECT game_id,
                   max(CASE WHEN offense = home THEN offense_score ELSE defense_score END) AS hs,
                   max(CASE WHEN offense = home THEN defense_score ELSE offense_score END) AS as_
            FROM plays_raw WHERE season = ? AND week = ? AND season_type = 'regular' GROUP BY game_id),
        d AS (
            SELECT g.game_id, g.home_team, g.away_team, g.home_points, g.away_points, m.hs, m.as_,
                   abs(m.hs - g.home_points) + abs(m.as_ - g.away_points) AS gap
            FROM ({_WEEK_GAMES}) g JOIN m USING (game_id))
        SELECT * FROM d WHERE gap > ? ORDER BY game_id""",
        [season, week, season, week, cfg.a2_max_score_gap]).fetchall()
    return [Anomaly("A2", season, week, str(gid),
                    f"{_label(h, a)}: PBP max score {as_}-{hs} vs final {ap}-{hp} (gap {gap})",
                    value=float(gap), threshold=float(cfg.a2_max_score_gap), investigate=True, game_id=gid)
            for gid, h, a, hp, ap, hs, as_, gap in rows]


def a3_play_count(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    lo, hi = cfg.a3_min_plays, cfg.a3_max_plays
    rows = con.execute(f"""
        WITH c AS (SELECT game_id, count(*) AS n FROM plays_raw
                   WHERE season = ? AND week = ? AND season_type = 'regular' GROUP BY game_id)
        SELECT g.game_id, g.home_team, g.away_team, c.n FROM ({_WEEK_GAMES}) g JOIN c USING (game_id)
        WHERE c.n < ? OR c.n > ? ORDER BY g.game_id""", [season, week, season, week, lo, hi]).fetchall()
    return [Anomaly("A3", season, week, str(gid), f"{_label(h, a)}: {n} plays (expected {lo}–{hi})",
                    value=float(n), investigate=True, game_id=gid) for gid, h, a, n in rows]


def a4_duplicates(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    excluded = ", ".join(f"'{t}'" for t in DROPPED_FOR_DUPES)
    rows = con.execute(f"""
        WITH dup AS (
            SELECT game_id, sum(c - 1) AS extra FROM (
                SELECT game_id, period, clock_minutes, clock_seconds, down, distance, yards_to_goal,
                       play_text, count(*) AS c
                FROM plays_raw
                WHERE season = ? AND week = ? AND season_type = 'regular' AND play_text IS NOT NULL
                  AND play_type NOT IN ({excluded})
                GROUP BY ALL HAVING count(*) > 1)
            GROUP BY game_id)
        SELECT g.game_id, g.home_team, g.away_team, dup.extra FROM ({_WEEK_GAMES}) g JOIN dup USING (game_id)
        WHERE dup.extra > ? ORDER BY g.game_id""",
        [season, week, season, week, cfg.a4_max_duplicates]).fetchall()
    return [Anomaly("A4", season, week, str(gid), f"{_label(h, a)}: {extra} duplicate play rows",
                    value=float(extra), threshold=float(cfg.a4_max_duplicates), investigate=True, game_id=gid)
            for gid, h, a, extra in rows]


def a5_fbs_status(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    rows = con.execute("""
        WITH first AS (
            SELECT team, min(week) AS first_week FROM (
                SELECT home_team AS team, week FROM games WHERE season = ? AND season_type = 'regular'
                UNION ALL
                SELECT away_team, week FROM games WHERE season = ? AND season_type = 'regular')
            GROUP BY team)
        SELECT c.team, c.classification, p.classification
        FROM team_status c JOIN first f ON f.team = c.team
        LEFT JOIN team_status p ON p.team = c.team AND p.season = c.season - 1
        WHERE c.season = ? AND f.first_week = ?
          AND (c.classification = 'fbs') <> (coalesce(p.classification, '') = 'fbs')
        ORDER BY c.team""", [season, season, season, week]).fetchall()
    return [Anomaly("A5", season, week, team,
                    f"{team}: {prev if prev else 'no row'} last season → {cur} this season "
                    "(check FBS status and pooling)", investigate=False, team=team)
            for team, cur, prev in rows]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_anomalies_checks.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/anomalies/checks.py tests/test_anomalies_checks.py
git commit -m "feat(anomalies): ingest integrity checks A1-A5"
```

---

### Task 3: Model-output checks B1–B3 and the guarded runner

**Files:**
- Modify: `src/cfbr/anomalies/checks.py` (append)
- Test: `tests/test_anomalies_checks.py` (append)

**Interfaces:**
- Consumes: Task 2 checks; `Anomaly`.
- Produces:
  - `week_bucket(week: int) -> str` returning `"1-3" | "4-7" | "8+"`; `BUCKETS: tuple[str, ...]`.
  - `load_thresholds(path: Path) -> dict` (parsed TOML; keys `b1` with bucket floats, `b3` with `max_abs_gap`).
  - `b1_rating_moves(rankings_csv: Path, season: int, week: int, thresholds: dict) -> list[Anomaly]`.
  - `b2_ppa_outliers(con, season: int, week: int, cfg: Anomalies) -> list[Anomaly]`.
  - `b3_line_gaps(predictions_csv: Path, season: int, week: int, thresholds: dict) -> list[Anomaly]`.
  - `run_db_checks(db_path: Path, season: int, week: int, cfg: Anomalies) -> list[Anomaly]` (A1–A5, B2).
  - `run_checks(db_path: Path, week_dir: Path, thresholds_file: Path, season: int, week: int, cfg: Anomalies) -> list[Anomaly]` (all eight). A check that raises becomes `Anomaly("ERR", season, week, <check id>, "check <id> failed: <Type>: <msg>")`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anomalies_checks.py`:

```python
import polars as pl

THRESH = {"b1": {"1-3": 9.0, "4-7": 6.0, "8+": 4.0}, "b3": {"max_abs_gap": 18.0}}


def test_week_bucket():
    assert [checks.week_bucket(w) for w in (1, 3, 4, 7, 8, 15)] == ["1-3", "1-3", "4-7", "4-7", "8+", "8+"]


def test_b1_uses_the_week_bucket_threshold(tmp_path):
    p = tmp_path / "rankings.csv"
    pl.DataFrame({"rank": [1, 2, 3], "team": ["Iowa", "Rice", "Ohio"],
                  "rating_change": [-4.5, 3.9, None]}).write_csv(p)
    out = checks.b1_rating_moves(p, 2026, 9, THRESH)                     # 8+: threshold 4.0
    assert [(a.key, a.value, a.threshold, a.investigate) for a in out] == [("Iowa", -4.5, 4.0, True)]
    assert out[0].message == "Iowa: rating -4.5 this week (threshold 4.0, weeks 8+)"
    assert checks.b1_rating_moves(p, 2026, 5, THRESH) == []              # 4-7: threshold 6.0


def test_b2_flags_single_game_ppa_outliers_as_context(tmp_path):
    con = _db(tmp_path / "x.duckdb")
    rows = []
    for g in range(200):                                                  # season-to-date spread
        rows += [[2026, 1 + g % 9, "regular", g, f"T{g}", False, 0.1 + (g % 10) * 0.01]] * 10
    rows += [[2026, 9, "regular", 999, "Iowa", False, 1.5]] * 10          # extreme, this week
    rows += [[2026, 9, "regular", 998, "Fcsu", True, 2.0]] * 10           # FCS offense: ignored
    con.executemany("INSERT INTO model_plays VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    out = checks.b2_ppa_outliers(con, 2026, 9, CFG)
    assert [(a.check, a.key, a.alert, a.investigate) for a in out] == [("B2", "999:Iowa", False, False)]
    assert out[0].message.startswith("Iowa in game 999: offense +1.50 PPA/play over 10 plays")


def test_b3_compares_model_to_current_line(tmp_path):
    p = tmp_path / "predictions.csv"
    pl.DataFrame({"game_id": [1, 2, 3], "home_team": ["Army", "Iowa", "Rice"], "away_team": ["Navy", "Ohio", "UTSA"],
                  "pred_margin": [10.0, 5.0, 3.0], "closing_spread": [10.0, -4.0, None]}).write_csv(p)
    out = checks.b3_line_gaps(p, 2026, 9, THRESH)                        # gap 20 > 18; 1; no line
    assert [(a.key, a.value, a.investigate) for a in out] == [("1", 20.0, True)]
    assert out[0].message == "Navy @ Army: model +10.0 vs line -10.0 home margin (gap 20.0, threshold 18.0)"


def test_run_checks_turns_a_crashing_check_into_err(tmp_path):
    db = tmp_path / "x.duckdb"
    _db(db).close()
    week_dir = tmp_path / "w09"
    week_dir.mkdir()                                                      # no rankings/predictions files
    out = checks.run_checks(db, week_dir, tmp_path / "missing.toml", 2026, 9, CFG)
    errs = {a.key: a for a in out if a.check == "ERR"}
    assert set(errs) == {"B1", "B3"}
    assert errs["B1"].message.startswith("check B1 failed: FileNotFoundError")
    assert errs["B1"].alert and not errs["B1"].investigate
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_anomalies_checks.py -v`
Expected: the 5 Task 2 tests pass; the 5 new ones FAIL with `AttributeError: module 'cfbr.anomalies.checks' has no attribute 'week_bucket'` (and similar).

- [ ] **Step 3: Implement**

At the top of `src/cfbr/anomalies/checks.py`, extend the imports to:

```python
import tomllib
from pathlib import Path

import duckdb
import polars as pl

from cfbr.anomalies.records import Anomaly
from cfbr.config import Anomalies
```

Append to `src/cfbr/anomalies/checks.py`:

```python
BUCKETS = ("1-3", "4-7", "8+")


def week_bucket(week: int) -> str:
    """Early-season ratings move more (the prior is fading), so B1 uses a threshold per bucket."""
    if week <= 3:
        return "1-3"
    return "4-7" if week <= 7 else "8+"


def load_thresholds(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def b1_rating_moves(rankings_csv: Path, season: int, week: int, thresholds: dict) -> list[Anomaly]:
    bucket = week_bucket(week)
    thr = float(thresholds["b1"][bucket])
    df = pl.read_csv(rankings_csv).filter(pl.col("rating_change").is_not_null()
                                          & (pl.col("rating_change").abs() > thr))
    return [Anomaly("B1", season, week, team,
                    f"{team}: rating {chg:+.1f} this week (threshold {thr:.1f}, weeks {bucket})",
                    value=float(chg), threshold=thr, investigate=True, team=team)
            for team, chg in df.select("team", "rating_change").iter_rows()]


def b2_ppa_outliers(con: duckdb.DuckDBPyConnection, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    lo_q, hi_q = cfg.b2_quantiles
    rows = con.execute("""
        WITH tg AS (
            SELECT game_id, week, offense_team AS team, avg(ppa) AS ppa, count(*) AS n
            FROM model_plays
            WHERE season = ? AND season_type = 'regular' AND NOT offense_fcs AND ppa IS NOT NULL
            GROUP BY game_id, week, offense_team),
        q AS (SELECT quantile_cont(ppa, ?) AS lo, quantile_cont(ppa, ?) AS hi FROM tg)
        SELECT tg.game_id, tg.team, tg.ppa, tg.n, q.lo, q.hi FROM tg, q
        WHERE tg.week = ? AND (tg.ppa < q.lo OR tg.ppa > q.hi)
        ORDER BY tg.game_id, tg.team""", [season, lo_q, hi_q, week]).fetchall()
    return [Anomaly("B2", season, week, f"{gid}:{team}",
                    f"{team} in game {gid}: offense {ppa:+.2f} PPA/play over {n} plays "
                    f"(season range {lo:+.2f}..{hi:+.2f})",
                    value=float(ppa), alert=False, game_id=gid, team=team)
            for gid, team, ppa, n, lo, hi in rows]


def b3_line_gaps(predictions_csv: Path, season: int, week: int, thresholds: dict) -> list[Anomaly]:
    """Model home margin vs the current consensus line (home margin = -spread). Compared only;
    nothing here feeds the model or the bets."""
    thr = float(thresholds["b3"]["max_abs_gap"])
    df = (pl.read_csv(predictions_csv)
          .filter(pl.col("closing_spread").is_not_null())
          .with_columns(vegas=-pl.col("closing_spread"))
          .with_columns(gap=(pl.col("pred_margin") - pl.col("vegas")).abs())
          .filter(pl.col("gap") > thr))
    return [Anomaly("B3", season, week, str(gid),
                    f"{_label(h, a)}: model {pred:+.1f} vs line {vegas:+.1f} home margin "
                    f"(gap {gap:.1f}, threshold {thr:.1f})",
                    value=float(gap), threshold=thr, investigate=True, game_id=int(gid))
            for gid, h, a, pred, vegas, gap in df.select("game_id", "home_team", "away_team", "pred_margin",
                                                         "vegas", "gap").iter_rows()]


DB_CHECKS = (("A1", a1_no_pbp), ("A2", a2_score_gap), ("A3", a3_play_count), ("A4", a4_duplicates),
             ("A5", a5_fbs_status), ("B2", b2_ppa_outliers))


def _guarded(found: list[Anomaly], check: str, season: int, week: int, fn) -> None:
    """A check that crashes becomes an alerted ERR item; the other checks still run."""
    try:
        found.extend(fn())
    except Exception as e:  # noqa: BLE001 - advisory step, must never stop the run
        found.append(Anomaly("ERR", season, week, check, f"check {check} failed: {type(e).__name__}: {e}"[:300]))


def run_db_checks(db_path: Path, season: int, week: int, cfg: Anomalies) -> list[Anomaly]:
    found: list[Anomaly] = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        for check, fn in DB_CHECKS:
            _guarded(found, check, season, week, lambda fn=fn: fn(con, season, week, cfg))
    return found


def run_checks(db_path: Path, week_dir: Path, thresholds_file: Path, season: int, week: int,
               cfg: Anomalies) -> list[Anomaly]:
    found = run_db_checks(db_path, season, week, cfg)
    _guarded(found, "B1", season, week,
             lambda: b1_rating_moves(week_dir / "rankings.csv", season, week, load_thresholds(thresholds_file)))
    _guarded(found, "B3", season, week,
             lambda: b3_line_gaps(week_dir / "predictions.csv", season, week, load_thresholds(thresholds_file)))
    return found
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_anomalies_checks.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/anomalies/checks.py tests/test_anomalies_checks.py
git commit -m "feat(anomalies): model output checks B1-B3 and guarded runner"
```

---

### Task 4: Baselines and the frozen thresholds file

**Files:**
- Create: `src/cfbr/anomalies/baselines.py`
- Create (generated): `config/anomaly_thresholds.toml`
- Test: `tests/test_anomalies_run.py` (create; this task adds the round-trip test)

**Interfaces:**
- Consumes: `week_bucket`, `BUCKETS`, `load_thresholds` (Task 3); `cfbr.report.rank.build_rankings(eng, season, week) -> (DataFrame, as_of)`; `Engine.predict(season)` (columns include `game_id`, `slot`, `pred_margin`); `Engine.games` (has `season`, `completed`, `slot`); `Engine.ds.lines`; `cfbr.backtest.metrics.opening_spread(lines, method) -> DataFrame[game_id, spread_open, n_providers_open]`; `cfbr.ingest.fetch.current_season()`.
- Produces:
  - `regular_weeks(eng, season: int) -> list[int]`
  - `b1_baseline(eng, seasons: list[int], q: float) -> tuple[dict[str, float], dict[str, int]]`
  - `b3_baseline(eng, seasons: list[int], q: float) -> tuple[float, int]`
  - `write_thresholds(path: Path, b1: dict[str, float], b1_n: dict[str, int], b3: float, b3_n: int, meta: dict) -> None` where `meta` has `generated_at: str`, `config: str`, `quantile: float`, `b1_seasons: list[int]`, `b3_seasons: list[int]`.
  - `compute_baselines(cfg: Config, path: Path) -> Path`.

- [ ] **Step 1: Write the failing test**

`tests/test_anomalies_run.py`:

```python
from cfbr.anomalies.baselines import write_thresholds
from cfbr.anomalies.checks import load_thresholds


def test_thresholds_round_trip(tmp_path):
    p = tmp_path / "t.toml"
    meta = {"generated_at": "2026-11-08T00:00:00+00:00", "config": "config/tuned.toml", "quantile": 0.995,
            "b1_seasons": [2015, 2016], "b3_seasons": [2021]}
    write_thresholds(p, {"1-3": 9.123, "4-7": 6.0, "8+": 4.5}, {"1-3": 300, "4-7": 400, "8+": 500}, 17.456, 812, meta)
    t = load_thresholds(p)
    assert t["b1"] == {"seasons": [2015, 2016], "1-3": 9.12, "4-7": 6.0, "8+": 4.5}
    assert t["b1_n"]["8+"] == 500
    assert t["b3"] == {"seasons": [2021], "max_abs_gap": 17.46, "n": 812}
    assert t["quantile"] == 0.995 and t["config"] == "config/tuned.toml"
    assert p.read_text().startswith("# Generated by `cfbr anomalies --baselines`")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_anomalies_run.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cfbr.anomalies.baselines'`.

- [ ] **Step 3: Implement**

`src/cfbr/anomalies/baselines.py`:

```python
"""B1/B3 thresholds from history, frozen to config/anomaly_thresholds.toml.

B1: |week-over-week rating change| of FBS teams, per week bucket, from the same
`build_rankings` that writes rankings.csv. B3: |model margin - opening line| from point-in-time
predictions (openers exist from 2021); Sunday's lines are early-week numbers, so the opener is
the right comparison, not the close. These are thresholds for a data check, not model settings:
using the 2022–2025 holdout seasons here does not touch the holdout rule. No API calls.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from cfbr.anomalies.checks import BUCKETS, week_bucket
from cfbr.config import Config

log = logging.getLogger(__name__)

OPENERS_FROM = 2021


def regular_weeks(eng, season: int) -> list[int]:
    g = eng.games.filter((pl.col("season") == season) & pl.col("completed") & pl.col("slot").str.starts_with("r"))
    return sorted(int(s[1:]) for s in g["slot"].unique().to_list())


def b1_baseline(eng, seasons: list[int], q: float) -> tuple[dict[str, float], dict[str, int]]:
    from cfbr.report.rank import build_rankings

    parts = []
    for season in seasons:
        for week in regular_weeks(eng, season):
            r, _ = build_rankings(eng, season, week)
            parts.append(r.select(move=pl.col("rating_change").abs()).with_columns(bucket=pl.lit(week_bucket(week))))
        log.info("B1 baseline: season %d done", season)
    df = pl.concat(parts).drop_nulls("move")
    values = {b: float(df.filter(pl.col("bucket") == b)["move"].quantile(q, "linear")) for b in BUCKETS}
    counts = {b: df.filter(pl.col("bucket") == b).height for b in BUCKETS}
    return values, counts


def b3_baseline(eng, seasons: list[int], q: float) -> tuple[float, int]:
    from cfbr.backtest.metrics import opening_spread

    preds = pl.concat([eng.predict(s) for s in seasons]).filter(pl.col("slot").str.starts_with("r"))
    opens = opening_spread(eng.ds.lines, eng.cfg.backtest.spread_consensus)
    gaps = (preds.join(opens, on="game_id", how="inner")
            .filter(pl.col("spread_open").is_not_null())
            .select(gap=(pl.col("pred_margin") + pl.col("spread_open")).abs()))
    return float(gaps["gap"].quantile(q, "linear")), gaps.height


def write_thresholds(path: Path, b1: dict[str, float], b1_n: dict[str, int], b3: float, b3_n: int,
                     meta: dict) -> None:
    lines = ["# Generated by `cfbr anomalies --baselines`; do not edit by hand.",
             f'generated_at = "{meta["generated_at"]}"',
             f'config = "{meta["config"]}"',
             f"quantile = {meta['quantile']}",
             "",
             "# |week-over-week rating change| of FBS teams, by regular-season week bucket",
             "[b1]",
             f"seasons = {meta['b1_seasons']}",
             *[f'"{b}" = {b1[b]:.2f}' for b in BUCKETS],
             "",
             "[b1_n]",
             *[f'"{b}" = {b1_n[b]}' for b in BUCKETS],
             "",
             "# |model home margin - opening line home margin|, regular season",
             "[b3]",
             f"seasons = {meta['b3_seasons']}",
             f"max_abs_gap = {b3:.2f}",
             f"n = {b3_n}",
             ""]
    path.write_text("\n".join(lines))


def compute_baselines(cfg: Config, path: Path) -> Path:
    from cfbr.data import Dataset
    from cfbr.ingest.fetch import current_season
    from cfbr.model.engine import Engine

    last = current_season() - 1
    eng = Engine(Dataset.load(cfg), cfg, last_completed=last)
    q = cfg.anomalies.baseline_quantile
    b1_seasons = list(range(cfg.seasons.tuning[0], last + 1))
    b3_seasons = list(range(OPENERS_FROM, last + 1))
    b1, b1_n = b1_baseline(eng, b1_seasons, q)
    b3, b3_n = b3_baseline(eng, b3_seasons, q)
    write_thresholds(path, b1, b1_n, b3, b3_n, {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": Path(cfg.source).name if getattr(cfg, "source", None) else "default",
        "quantile": q, "b1_seasons": b1_seasons, "b3_seasons": b3_seasons})
    return path
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_anomalies_run.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit the module**

```bash
git add src/cfbr/anomalies/baselines.py tests/test_anomalies_run.py
git commit -m "feat(anomalies): compute B1/B3 thresholds from history"
```

(The thresholds file itself is generated in Task 5, Step 6, once the CLI exists.)

---

### Task 5: `run_step`, the weekly hook, and the `cfbr anomalies` command

**Files:**
- Create: `src/cfbr/anomalies/run.py`
- Modify: `src/cfbr/cli.py` — add `_anomaly_step` (after `_bets_step`, ~line 415); call it in `_weekly_run` between the `log.info("season-to-date: ...")` statement and `bets_failed = _bets_step(...)` (~line 513); add the `anomalies` command after the `check` command (~line 394).
- Create (generated): `config/anomaly_thresholds.toml`
- Test: `tests/test_anomalies_run.py` (append)

**Interfaces:**
- Consumes: `run_checks`, `run_db_checks` (Task 3); `merge`, `read_doc`, `write_doc`, `render_md` (Task 1); `compute_baselines` (Task 4); `cfbr.alerts.send(cfg, title, message, priority)`; `cfbr.config.REPO_ROOT`.
- Produces:
  - `thresholds_path(cfg: Config) -> Path` (absolute; relative paths resolve against `REPO_ROOT`).
  - `week_dir(cfg: Config, season: int, week: int) -> Path` = `out_dir/{season}/w{week:02d}`.
  - `latest_week(cfg: Config, season: int) -> int` (largest `wNN` directory; `FileNotFoundError` if none).
  - `run_step(cfg: Config, season: int, week: int, now: datetime | None = None) -> list[dict]` — runs all checks, merges, writes `anomalies.json` and `anomalies.md`, returns newly added rows whose `alert` is true.
  - `run_history(cfg: Config, season: int) -> list[Anomaly]` — database checks for every completed regular-season week.
  - `alert_text(rows: list[dict], limit: int = 15) -> str`.
  - `cli._anomaly_step(cfg, season: int, week: int) -> None` (never raises).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anomalies_run.py`:

```python
import json
from datetime import UTC, datetime

import duckdb
import polars as pl
import pytest

from cfbr.anomalies import run
from cfbr.config import Config, Paths


def _env(tmp_path):
    cfg = Config(paths=Paths(raw_dir=tmp_path / "raw", db_path=tmp_path / "x.duckdb", results_dir=tmp_path / "res",
                             out_dir=tmp_path / "out"))
    cfg.anomalies.thresholds = tmp_path / "t.toml"
    write_thresholds(cfg.anomalies.thresholds, {"1-3": 9.0, "4-7": 6.0, "8+": 4.0},
                     {"1-3": 1, "4-7": 1, "8+": 1}, 18.0, 1,
                     {"generated_at": "x", "config": "x", "quantile": 0.995, "b1_seasons": [2025], "b3_seasons": [2025]})
    con = duckdb.connect(str(cfg.paths.db_path))
    con.execute("""CREATE TABLE games (game_id BIGINT, season INTEGER, week INTEGER, season_type VARCHAR,
        completed BOOLEAN, home_team VARCHAR, away_team VARCHAR, home_class VARCHAR, away_class VARCHAR,
        home_points INTEGER, away_points INTEGER)""")
    con.execute("""CREATE TABLE plays_raw (play_id VARCHAR, game_id BIGINT, season INTEGER, week INTEGER,
        season_type VARCHAR, offense VARCHAR, defense VARCHAR, home VARCHAR, offense_score INTEGER,
        defense_score INTEGER, period INTEGER, clock_minutes INTEGER, clock_seconds INTEGER, down INTEGER,
        distance INTEGER, yards_to_goal INTEGER, play_type VARCHAR, play_text VARCHAR)""")
    con.execute("CREATE TABLE team_status (season INTEGER, team VARCHAR, classification VARCHAR, conference VARCHAR)")
    con.execute("""CREATE TABLE model_plays (season INTEGER, week INTEGER, season_type VARCHAR, game_id BIGINT,
        offense_team VARCHAR, offense_fcs BOOLEAN, ppa DOUBLE)""")
    con.execute("INSERT INTO games VALUES (7, 2026, 9, 'regular', true, 'Army', 'Navy', 'fbs', 'fbs', 21, 14)")
    con.close()                                                     # game 7 has no plays: A1
    wd = cfg.paths.out_dir / "2026" / "w09"
    wd.mkdir(parents=True)
    pl.DataFrame({"rank": [1], "team": ["Iowa"], "rating_change": [-5.0]}).write_csv(wd / "rankings.csv")
    pl.DataFrame({"game_id": [8], "home_team": ["Rice"], "away_team": ["UTSA"], "pred_margin": [1.0],
                  "closing_spread": [-2.0]}).write_csv(wd / "predictions.csv")
    return cfg, wd


T = datetime(2026, 11, 8, 15, tzinfo=UTC)


def test_run_step_writes_files_and_returns_new_alert_rows(tmp_path):
    cfg, wd = _env(tmp_path)
    added = run.run_step(cfg, 2026, 9, now=T)
    assert sorted(r["check"] for r in added) == ["A1", "B1"]
    doc = json.loads((wd / "anomalies.json").read_text())
    assert {r["check"]: r["status"] for r in doc["items"]} == {"A1": "plain", "B1": "pending"}
    assert "Navy @ Army: completed, no play-by-play" in (wd / "anomalies.md").read_text()


def test_rerun_does_not_realert_and_resolves_fixed_items(tmp_path):
    cfg, wd = _env(tmp_path)
    run.run_step(cfg, 2026, 9, now=T)
    assert run.run_step(cfg, 2026, 9, now=T) == []
    pl.DataFrame({"rank": [1], "team": ["Iowa"], "rating_change": [-1.0]}).write_csv(wd / "rankings.csv")
    run.run_step(cfg, 2026, 9, now=T)
    doc = json.loads((wd / "anomalies.json").read_text())
    assert {r["check"]: r["status"] for r in doc["items"]}["B1"] == "resolved"


def test_alert_text_marks_investigate_items_and_truncates():
    rows = [{"check": "A1", "message": "no pbp", "investigate": False},
            {"check": "B1", "message": "Iowa -5.0", "investigate": True}] * 10
    text = run.alert_text(rows, limit=3)
    assert text.splitlines()[:2] == ["• A1 no pbp", "• B1 Iowa -5.0 (to investigate)"]
    assert text.splitlines()[-1] == "… and 17 more (see anomalies.md)"


def test_latest_week(tmp_path):
    cfg, _ = _env(tmp_path)
    (cfg.paths.out_dir / "2026" / "w10").mkdir()
    (cfg.paths.out_dir / "2026" / "wXX").mkdir()
    assert run.latest_week(cfg, 2026) == 10


def test_weekly_anomaly_step_never_raises_and_alerts(monkeypatch, tmp_path):
    from cfbr import cli

    sent = []
    monkeypatch.setattr("cfbr.alerts.send", lambda cfg, title, msg, priority="high": sent.append((title, priority)) or True)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr("cfbr.anomalies.run.run_step", boom)
    cli._anomaly_step(Config(), 2026, 9)                             # must not raise
    assert sent == [("cfbr anomalies: checks failed", "default")]

    sent.clear()
    monkeypatch.setattr("cfbr.anomalies.run.run_step",
                        lambda *a, **k: [{"check": "A1", "message": "x", "investigate": False}])
    cli._anomaly_step(Config(), 2026, 9)
    assert sent == [("cfbr anomalies w09: 1 new", "default")]

    sent.clear()
    monkeypatch.setattr("cfbr.anomalies.run.run_step", lambda *a, **k: [])
    cli._anomaly_step(Config(), 2026, 9)
    assert sent == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_anomalies_run.py -v`
Expected: round-trip test passes; the 5 new tests FAIL (`ImportError: cannot import name 'run'`).

- [ ] **Step 3: Implement `run.py`**

`src/cfbr/anomalies/run.py`:

```python
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
```

- [ ] **Step 4: Add the weekly hook in `src/cfbr/cli.py`**

After `_bets_step` (ends ~line 414), add:

```python
def _anomaly_step(cfg, season: int, week: int) -> None:
    """Advisory data checks (cfbr/anomalies): write out/{season}/wNN/anomalies.{json,md} and push
    items new in this run. Never raises and never changes the exit code."""
    from cfbr.alerts import send

    try:
        from cfbr.anomalies.run import alert_text, run_step

        added = run_step(cfg, season, week)
    except Exception as e:  # noqa: BLE001 - advisory step
        log.exception("anomaly checks failed; continuing")
        send(cfg, "cfbr anomalies: checks failed", f"{type(e).__name__}: {e}"[:300], priority="default")
        return
    if added:
        send(cfg, f"cfbr anomalies w{week:02d}: {len(added)} new", alert_text(added), priority="default")
```

In `_weekly_run`, immediately before `bets_failed = _bets_step(cfg, ds, season, week)`, add:

```python
    if cfg.anomalies.enabled:
        _anomaly_step(cfg, season, week)
```

Note: `_anomaly_step` imports `send` and `run_step` at call time from their modules, so the test's `monkeypatch.setattr("cfbr.alerts.send", ...)` and `monkeypatch.setattr("cfbr.anomalies.run.run_step", ...)` take effect.

- [ ] **Step 5: Add the `cfbr anomalies` command in `src/cfbr/cli.py`**

After the `check` command (~line 394), add:

```python
@app.command()
def anomalies(
    season: Annotated[int, typer.Option()],
    week: Annotated[Optional[int], typer.Option(help="Regular-season week (default: latest out/{season}/wNN)")] = None,
    history: Annotated[bool, typer.Option(help="Database checks over every completed week; "
                                               "writes results/anomalies_history_{season}.json")] = False,
    baselines: Annotated[bool, typer.Option(help="Recompute the B1/B3 thresholds file from history "
                                                 "(no API calls; several minutes)")] = False,
    config: ConfigOpt = None,
) -> None:
    """Advisory data-anomaly checks. Without flags: run all checks for one week, update
    anomalies.{json,md} and print them (no alert is sent; `cfbr weekly` sends those)."""
    import json
    from collections import Counter
    from dataclasses import asdict

    from cfbr.anomalies import run as arun
    from cfbr.anomalies.records import read_doc, render_md

    cfg = _default_config(config)
    if baselines:
        from cfbr.anomalies.baselines import compute_baselines

        typer.echo(f"wrote {compute_baselines(cfg, arun.thresholds_path(cfg))}")
        return
    if history:
        items = arun.run_history(cfg, season)
        out = cfg.paths.results_dir / f"anomalies_history_{season}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([{"id": a.id, **asdict(a)} for a in items], indent=2) + "\n")
        counts = Counter(a.check for a in items)
        typer.echo(" ".join(f"{c}={counts[c]}" for c in sorted(counts)) or "no anomalies")
        typer.echo(f"wrote {out}")
        return
    week = week or arun.latest_week(cfg, season)
    added = arun.run_step(cfg, season, week)
    typer.echo(render_md(read_doc(arun.week_dir(cfg, season, week) / "anomalies.json")))
    typer.echo(f"{len(added)} new alertable item(s)")
```

Confirm `Optional` and `Annotated` are already imported at the top of `cli.py` (`grep -n "^from typing" src/cfbr/cli.py`); they are used by existing commands.

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_anomalies_run.py tests/test_anomalies_checks.py tests/test_anomalies_records.py -v`
Expected: all pass (6 + 10 + 8 = 24).

- [ ] **Step 7: Generate the thresholds file from real data**

Run: `uv run cfbr anomalies --season 2026 --baselines`
Expected: `wrote /…/config/anomaly_thresholds.toml` after several minutes (one Engine, ~11 seasons of weekly rankings plus 5 seasons of predictions; no API calls).

Then: `cat config/anomaly_thresholds.toml`
Sanity-check before committing: `b1."1-3"` > `b1."4-7"` > `b1."8+"` (early weeks move more), every bucket between 2 and 20 points, `b3.max_abs_gap` between 12 and 30, `b1_n` buckets each > 1,000, `b3.n` > 2,000. If any value is outside these ranges, stop and report it instead of committing (it means the baseline is computing the wrong thing).

- [ ] **Step 8: Try one live week (read-only on local data)**

Run: `uv run cfbr anomalies --season 2026 --week 8`
Expected: a markdown table (or "No anomalies.") and `N new alertable item(s)`; `out/2026/w08/anomalies.{json,md}` exist. Re-run the same command: expected `0 new alertable item(s)`.

Run: `uv run cfbr anomalies --season 2025 --history`
Expected: a counts line — A2 and A4 non-zero; A1=0 — and `wrote …/results/anomalies_history_2025.json`.

- [ ] **Step 9: Commit**

```bash
git add src/cfbr/anomalies/run.py src/cfbr/cli.py config/anomaly_thresholds.toml tests/test_anomalies_run.py
git commit -m "feat(anomalies): weekly hook, cfbr anomalies command and frozen thresholds"
```

---

### Task 6: Isolation and real-data firing bands

**Files:**
- Test: `tests/test_anomalies_run.py` (append)

**Interfaces:**
- Consumes: `run.run_history` (Task 5); `cfg` fixture from `tests/conftest.py`.

- [ ] **Step 1: Write the tests**

Append to `tests/test_anomalies_run.py`:

```python
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "cfbr"


def test_pipeline_packages_never_import_anomalies():
    offenders = []
    for pkg in ("model", "bets", "report", "web"):
        for f in (SRC / pkg).rglob("*.py"):
            text = f.read_text()
            if "cfbr.anomalies" in text or "from cfbr import anomalies" in text:
                offenders.append(str(f.relative_to(SRC)))
    assert offenders == []


# Measured 2026-11-08 on the dev store (regular season, FBS side): A2 19/11/5/7, A3 10/3/3/4,
# A4 72/13/12/24 for 2022–2025, A1 0. Bands are wide on purpose: they catch a broken check or a
# schema change, not normal drift.
BANDS = {"A1": (0, 0), "A2": (2, 30), "A3": (1, 15), "A4": (5, 100)}


@pytest.mark.parametrize("season", [2022, 2023, 2024, 2025])
def test_history_firing_rates_stay_in_measured_bands(cfg, season):
    if not cfg.paths.db_path.exists():
        pytest.skip("no local DuckDB store")
    counts = Counter(a.check for a in run.run_history(cfg, season))
    assert "ERR" not in counts
    for check, (lo, hi) in BANDS.items():
        assert lo <= counts[check] <= hi, (season, check, counts[check])
```

- [ ] **Step 2: Run them**

Run: `uv run pytest tests/test_anomalies_run.py -v -k "import or history"`
Expected: 5 passed (1 isolation + 4 seasons), or the 4 history tests skipped where no store exists.

If a band fails, print the count (it's in the assertion message) and compare with the measured numbers in the comment. A count outside the band means the SQL differs from the measured query; fix the SQL, don't widen the band.

- [ ] **Step 3: Commit**

```bash
git add tests/test_anomalies_run.py
git commit -m "test(anomalies): import isolation and real-data firing bands"
```

---

### Task 7: Docs, full suite, PR

**Files:**
- Modify: `README.md` (command table near line 33–41; weekly section near line 143)
- Modify: `DECISIONS.md` (append lines)
- Modify: `CLAUDE.md` (one line under `cfbr weekly` in "Pipeline and data flow")

- [ ] **Step 1: README**

In the command table, after the `cfbr check` row, add:

```markdown
| `cfbr anomalies --season 2026` | Advisory data checks for the latest week (A1–A5 ingest integrity, B1–B3 model output): writes `out/{season}/wNN/anomalies.{json,md}`; `weekly` runs them and pushes new items. `--history` runs the database checks over a season; `--baselines` recomputes `config/anomaly_thresholds.toml` (no API calls) |
```

In the weekly section (the paragraph starting "`weekly` takes a lock"), append this sentence:

```markdown
After ranking, `weekly` also runs the advisory data checks (`cfbr anomalies`) and pushes a `default`-priority alert listing anomalies first seen in this run; a same-week re-run doesn't repeat them, and a failure there never changes the exit code.
```

- [ ] **Step 2: DECISIONS**

Append to `DECISIONS.md`:

```markdown
- Data anomaly checks (user request, 2026-11-08; spec docs/superpowers/specs/2026-11-08-anomaly-investigation-design.md): `cfbr weekly` runs eight deterministic checks after ranking — A1 completed FBS game without play-by-play, A2 PBP max score vs final > 8, A3 plays < 110 or > 250, A4 > 3 duplicate play rows, A5 FBS status change from last season (first game of the season), B1 |rating change| above the week bucket's p99.5, B2 single-game offensive PPA outside p0.25–p99.75 (stored as context, never alerted), B3 |model − current consensus line| above the p99.5 of |model − opener| — and writes out/{season}/wNN/anomalies.{json,md}. Advisory only: nothing blocks publishing and no exit code changes; nothing in model/bets/report/web imports the package (tested). Thresholds come from measured 2022–2026 rates (in the spec); A2 uses the PBP maximum score because the last play by numeric play_id misses the final in 5–10% of games. A5 replaced "team name missing from team_status", which can never fire because team_status is derived from games — Ops (anomalies)
- Anomaly thresholds B1/B3 are frozen in config/anomaly_thresholds.toml by `cfbr anomalies --baselines` (the production config, 2015–2025 for B1, 2021–2025 for B3 because openers start in 2021). They are thresholds for a data check, not model settings, so using the 2022–2025 holdout seasons does not touch the holdout rule. B3 is calibrated on the opener because Sunday's lines are early-week numbers. Recompute after a model change that alters rating dynamics (e.g. the FCS two-stage fit) — Ops (anomalies)
```

- [ ] **Step 3: CLAUDE.md**

In "Pipeline and data flow", in the `cfbr weekly` paragraph, change "rank → accuracy line → paper bets" to "rank → accuracy line → anomaly checks (advisory; `anomalies/`) → paper bets".

- [ ] **Step 4: Full test suite**

Run: `uv run pytest -q`
Expected: all pass (dataset-dependent tests run locally since the dev store exists). Record the summary line for the PR description.

- [ ] **Step 5: Commit docs**

```bash
git add README.md DECISIONS.md CLAUDE.md
git commit -m "docs: anomaly checks in README, DECISIONS and CLAUDE.md"
```

- [ ] **Step 6: Push and open the PR**

```bash
git push -u origin feat/anomaly-checks
gh pr create --title "feat: advisory data anomaly checks after weekly" --body-file /tmp/pr-body.md
```

PR body (write to the scratchpad and pass with `--body-file`): summary of the eight checks, advisory-only guarantee, thresholds file and how it was generated (paste the generated values), `cfbr anomalies --season 2026 --week 8` output, the `--history` counts for 2025, the full-suite summary line, and a note that PR 2 (LLM investigation) and the shakedown follow; include the spec and plan paths. End with the attribution line from the session reminder.

Do not merge or deploy: deploying is a stop-and-ask step (CLAUDE.md, "Where things run").
