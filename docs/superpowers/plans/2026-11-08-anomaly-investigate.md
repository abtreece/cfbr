# Anomaly Investigation Agent (PR 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `cfbr investigate`, which runs a Claude agent over each `pending` anomaly from PR 1, decides whether it is a data bug or real football, writes a grounded verdict back into `anomalies.json`, and pushes one ntfy summary — plus `cfbr anomalies --sample-patterns` to build the historical shakedown set.

**Architecture:** New modules in `src/cfbr/anomalies/`: `ledger.py` (spend log and monthly guard), `tools.py` (sandboxed DuckDB `query` and whitelisted `read_output`, one result-id space for grounding), `prompt.py` (frozen system prompt, tool schemas, per-item message), `agent.py` (one investigation: a manual tool loop over `client.beta.messages.stream`, caps, refusal handling, deterministic grounding check), `shakedown.py` (pattern sampling). `run.py` gains the orchestration (`investigate_doc`) and `records.py` gains a locked read-modify-write so `cfbr weekly` and `cfbr investigate` never lose each other's writes. Spec: `docs/superpowers/specs/2026-11-08-anomaly-investigation-design.md`.

**Tech Stack:** Python 3.12, uv, duckdb 1.5.6, polars, typer, pydantic, pytest; new dependency `anthropic` 2.6.0 (brings `httpx2`).

## Global Constraints

- Advisory only. Nothing in `src/cfbr/model/`, `bets/`, `report/` or `web/` imports `cfbr.anomalies` (existing AST test). Agent output is written only to `anomalies.{json,md}`, transcripts under `data/investigations/` and `data/anthropic_calls.jsonl`.
- The feature is off unless `ANTHROPIC_API_KEY` is set (environment, else `.keys`). No key → `cfbr investigate` logs one line and exits 0. Never create or edit a `.keys` file.
- **No test makes a network call.** Every agent test uses the fake client from `tests/anomaly_fakes.py`. Never run `cfbr investigate` without `--dry-run` during implementation (that would spend real credits).
- DuckDB for the agent: `duckdb.connect(path, read_only=True, config={"enable_external_access": False, "lock_configuration": True})` on a **snapshot copy** (`data/investigations/snapshot.duckdb`). The weekly lock (`data/.weekly.lock`) is held only while copying.
- API call shape (verified against anthropic 2.6.0): `client.beta.messages.stream(model=..., max_tokens=..., system=<str>, cache_control={"type": "ephemeral"}, tools=TOOLS, messages=..., output_config={"effort": ..., "task_budget": {"type": "tokens", "total": ...}}, betas=["task-budgets-2026-03-13", "server-side-fallback-2026-07-01"], fallbacks="default")`, used as a context manager with `.get_final_message()`. Never send `thinking`, `temperature` or a forced `tool_choice` (Opus 5.5 rejects them). Append each response's full `content` to the history unchanged.
- Settings (config `[investigate]`): model `claude-opus-5-5`, effort `xhigh`, max_tokens 32000, task_budget_tokens 60000 (0 disables), max_turns 40, max_items 12, max_attempts 3, wall_clock_minutes 15, query_timeout_seconds 30, query_max_rows 200, query_max_chars 20000, monthly_budget_usd 50.
- Verdict fields: `verdict` ∈ {data_bug, real, inconclusive}; `confidence` ∈ {low, medium, high}; `summary` ≤ 280 chars; `evidence[]` of {claim, query_id, excerpt ≤ 500 chars}; `affected` {game_ids, teams}; `model_impact` ∈ {none, small, material, unknown}; `suggested_action` ≤ 200 chars. Grounding: every evidence `query_id` names a successful tool result in this session and its whitespace-normalized `excerpt` is a substring of that result; `data_bug`/`real` need ≥ 1 evidence item. Failure → `inconclusive`, reason `ungrounded`.
- Item statuses added by this PR: `investigated`, `error`, `skipped_budget`, `skipped_volume`. Only `pending` items (or ids passed with `--force`) are investigated.
- Don't run `cfbr weekly`, `bets`, `odds-snapshot`, `backup` or `check` locally. Python uses 4-space indentation. Conventional commits, **no Co-Authored-By line, no generated-by line**. Never `git add -A`; `data/`, `out/`, `results/` in the worktree are symlinks to dev data.

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `pyproject.toml`, `uv.lock` | modify | add `anthropic` |
| `src/cfbr/config.py` | modify | `Investigate` section; `_keys_file_value`; `anthropic_api_key()` |
| `config/default.toml` | modify | `[investigate]` block |
| `src/cfbr/anomalies/records.py` | modify | unique temp names, `update_doc` (sidecar flock), optional `title` in `render_md` |
| `src/cfbr/anomalies/ledger.py` | create | prices, per-call cost, append-only ledger, month-to-date |
| `src/cfbr/anomalies/tools.py` | create | `ToolSession` (query + read_output, result ids, caps, timeout) |
| `src/cfbr/anomalies/prompt.py` | create | `TOOLS`, `VERDICT_SCHEMA`, `schema_text`, `build_system`, `item_message` |
| `src/cfbr/anomalies/agent.py` | create | `investigate` loop, `grounded`, `inconclusive` |
| `src/cfbr/anomalies/run.py` | modify | `run_step` via `update_doc`; `investigate_doc`, `plan_targets`, `classify_error`, `try_lock`, `investigate_alert` |
| `src/cfbr/anomalies/shakedown.py` | create | `pattern_group`, `sample_patterns`, `shakedown_doc` |
| `src/cfbr/cli.py` | modify | `investigate` command; `anomalies --sample-patterns --seasons --per-group` |
| `tests/anomaly_fakes.py` | create | fake Anthropic client and message builders |
| `tests/test_anomalies_records.py` | modify | update_doc concurrency, temp names, title |
| `tests/test_anomalies_ledger.py` | create | cost math, month boundary |
| `tests/test_anomalies_tools.py` | create | sandbox, caps, ids, timeout, whitelist |
| `tests/test_anomalies_agent.py` | create | loop behaviour with the fake client |
| `tests/test_anomalies_investigate.py` | create | orchestration, statuses, errors, lock, concurrency, sampling, CLI |
| `README.md`, `DECISIONS.md`, `CLAUDE.md` | modify | command rows, cron line, decisions |

---

### Task 1: Dependency, config, key helper and locked document updates

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Modify: `src/cfbr/config.py` (add `Investigate` before `class Config`; field on `Config`; refactor `api_key`; add `anthropic_api_key`)
- Modify: `config/default.toml` (append `[investigate]`)
- Modify: `src/cfbr/anomalies/records.py`, `src/cfbr/anomalies/run.py` (`run_step` only)
- Test: `tests/test_anomalies_records.py` (append), `tests/test_anomalies_investigate.py` (create)

**Interfaces:**
- Produces: `cfbr.config.Investigate` (fields listed in Global Constraints plus `ledger: Path = Path("data/anthropic_calls.jsonl")`, `snapshot: Path = Path("data/investigations/snapshot.duckdb")`, `transcripts: Path = Path("data/investigations")`); `Config.investigate`; `cfbr.config.anthropic_api_key() -> str | None`; `records.update_doc(path: Path, fn: Callable[[dict | None], dict]) -> dict`; `render_md` uses `doc["title"]` when present.

- [ ] **Step 1: Add the dependency**

Run: `uv add anthropic`
Then: `grep -n anthropic pyproject.toml`
Expected: a line like `"anthropic>=2.6.0",` in `[project].dependencies`, and `uv.lock` changed.

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_anomalies_records.py` (merge new imports into the file's top import block):

```python
import threading

from cfbr.anomalies.records import update_doc


def test_update_doc_serializes_concurrent_writers(tmp_path):
    p = tmp_path / "w09" / "anomalies.json"

    def bump(doc):
        doc = doc or {"n": 0}
        doc["n"] += 1
        return doc

    threads = [threading.Thread(target=lambda: [update_doc(p, bump) for _ in range(25)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert read_doc(p)["n"] == 100
    assert not list(p.parent.glob("*.tmp"))


def test_render_md_uses_title_when_present():
    doc, _ = merge(None, [_a()], 2026, 9, T0)
    doc["title"] = "Shakedown 2022–2025"
    assert render_md(doc).startswith("# Data anomalies — Shakedown 2022–2025")
```

Create `tests/test_anomalies_investigate.py`:

```python
from cfbr.config import Config, anthropic_api_key


def test_investigate_config_defaults():
    i = Config().investigate
    assert (i.model, i.effort, i.max_turns, i.max_items, i.max_attempts) == ("claude-opus-5-5", "xhigh", 40, 12, 3)
    assert (i.max_tokens, i.task_budget_tokens, i.monthly_budget_usd) == (32000, 60000, 50.0)
    assert (i.wall_clock_minutes, i.query_timeout_seconds, i.query_max_rows, i.query_max_chars) == (15, 30, 200, 20000)


def test_anthropic_key_env_then_keys_file_then_none(monkeypatch, tmp_path):
    monkeypatch.setattr("cfbr.config.REPO_ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert anthropic_api_key() is None
    (tmp_path / ".keys").write_text("CFBD_API_KEY=abc\nANTHROPIC_API_KEY='sk-file'\n")
    assert anthropic_api_key() == "sk-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert anthropic_api_key() == "sk-env"
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_anomalies_records.py tests/test_anomalies_investigate.py -q`
Expected: collection errors — `cannot import name 'update_doc'` and `cannot import name 'anthropic_api_key'`.

- [ ] **Step 4: Config**

In `src/cfbr/config.py`, insert before `class Config(BaseModel):`:

```python
class Investigate(BaseModel):
    """LLM investigation of pending anomalies (`cfbr investigate`, cfbr/anomalies/agent.py).
    Off unless ANTHROPIC_API_KEY is set in the environment or .keys."""
    model: str = "claude-opus-5-5"
    effort: str = "xhigh"
    max_tokens: int = 32000
    task_budget_tokens: int = 60000  # advisory per-investigation budget (beta); 0 disables
    max_turns: int = 40
    max_items: int = 12  # per run; pending items beyond this are marked skipped_volume
    max_attempts: int = 3  # transient API failures before an item becomes "error"
    wall_clock_minutes: float = 15
    query_timeout_seconds: float = 30
    query_max_rows: int = 200
    query_max_chars: int = 20000
    monthly_budget_usd: float = 50.0  # local guard; the account's monthly credits hard-stop as the backstop
    ledger: Path = Path("data/anthropic_calls.jsonl")
    snapshot: Path = Path("data/investigations/snapshot.duckdb")
    transcripts: Path = Path("data/investigations")
```

In `class Config`, after `anomalies: Anomalies = Anomalies()`, add `investigate: Investigate = Investigate()`.

Replace the existing `api_key` function with:

```python
def _keys_file_value(name: str) -> str | None:
    keys_file = REPO_ROOT / ".keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def api_key() -> str:
    """CFBD key from the environment, falling back to the gitignored .keys file."""
    key = os.environ.get("CFBD_API_KEY") or _keys_file_value("CFBD_API_KEY")
    if key:
        return key
    raise RuntimeError("CFBD_API_KEY not set (env var or .keys file)")


def anthropic_api_key() -> str | None:
    """Anthropic key for `cfbr investigate`; None means investigation is off."""
    return os.environ.get("ANTHROPIC_API_KEY") or _keys_file_value("ANTHROPIC_API_KEY")
```

Append to `config/default.toml`:

```toml

[investigate]
# LLM investigation of pending anomalies (`cfbr investigate`). Off unless ANTHROPIC_API_KEY is set
# (environment or .keys). Spend is logged to `ledger`; runs stop at monthly_budget_usd.
model = "claude-opus-5-5"
effort = "xhigh"
max_tokens = 32000
task_budget_tokens = 60000
max_turns = 40
max_items = 12
max_attempts = 3
wall_clock_minutes = 15
query_timeout_seconds = 30
query_max_rows = 200
query_max_chars = 20000
monthly_budget_usd = 50.0
ledger = "data/anthropic_calls.jsonl"
snapshot = "data/investigations/snapshot.duckdb"
transcripts = "data/investigations"
```

- [ ] **Step 5: Records**

In `src/cfbr/anomalies/records.py`, add imports `fcntl`, `tempfile`, `from collections.abc import Callable`, `from contextlib import contextmanager`. Replace `write_doc` and add `update_doc`:

```python
def write_doc(path: Path, doc: dict) -> None:
    """Atomic: write a uniquely named temp file next to the target, then rename over it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(doc, indent=2, default=str) + "\n")
    os.replace(tmp, path)


@contextmanager
def _doc_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield


def update_doc(path: Path, fn: Callable[[dict | None], dict]) -> dict:
    """Read-modify-write under a sidecar lock, so `cfbr weekly` and `cfbr investigate` never
    lose each other's writes to the same anomalies.json."""
    with _doc_lock(path):
        doc = fn(read_doc(path))
        write_doc(path, doc)
        return doc
```

In `render_md`, change the first line of `lines` to use `doc.get("title") or f"{doc['season']} week {doc['week']}"`:

```python
    title = doc.get("title") or f"{doc['season']} week {doc['week']}"
    lines = [f"# Data anomalies — {title}", "", f"Generated {doc['generated_at']}.", ""]
```

In `src/cfbr/anomalies/run.py`, change the records import to `from cfbr.anomalies.records import Anomaly, merge, read_doc, render_md, update_doc` (keep `read_doc`; Task 6 uses it) and replace the body of `run_step` after `path = wd / "anomalies.json"`:

```python
    added: list[dict] = []

    def apply(existing: dict | None) -> dict:
        doc, new = merge(existing, found, season, week, now)
        added.extend(new)
        return doc

    doc = update_doc(path, apply)
    (wd / "anomalies.md").write_text(render_md(doc))
    return [row for row in added if row["alert"]]
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_anomalies_records.py tests/test_anomalies_investigate.py tests/test_anomalies_run.py -q`
Expected: all pass. Then `uv run pytest -q` — all pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/cfbr/config.py config/default.toml src/cfbr/anomalies/records.py src/cfbr/anomalies/run.py tests/test_anomalies_records.py tests/test_anomalies_investigate.py
git commit -m "feat(anomalies): investigate config, Anthropic key helper and locked document updates"
```

---

### Task 2: Spend ledger

**Files:**
- Create: `src/cfbr/anomalies/ledger.py`
- Test: `tests/test_anomalies_ledger.py`

**Interfaces:**
- Produces: `PRICES: dict[str, tuple[float, float, float, float]]`, `FALLBACK_PRICE`, `call_cost(model: str, usage) -> float`, `record(path: Path, *, model: str, usage, purpose: str, request_id: str | None, now: datetime | None = None) -> float`, `month_to_date(path: Path, now: datetime | None = None) -> float`. `usage` is any object with `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` attributes (missing or None count as 0).

- [ ] **Step 1: Write the failing tests**

`tests/test_anomalies_ledger.py`:

```python
import json
from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest

from cfbr.anomalies import ledger


def _u(i=0, o=0, cw=0, cr=0):
    return NS(input_tokens=i, output_tokens=o, cache_creation_input_tokens=cw, cache_read_input_tokens=cr)


def test_call_cost_opus_5_5():
    # 1M of each: 4 + 20 + 5 + 0.20
    assert ledger.call_cost("claude-opus-5-5", _u(10**6, 10**6, 10**6, 10**6)) == pytest.approx(29.2)


def test_unknown_model_is_priced_conservatively_and_none_counts_as_zero():
    u = NS(input_tokens=10**6, output_tokens=None)
    assert ledger.call_cost("some-fallback-model", u) == pytest.approx(ledger.FALLBACK_PRICE[0])


def test_record_appends_and_month_to_date_sums_only_this_month(tmp_path):
    p = tmp_path / "calls.jsonl"
    ledger.record(p, model="claude-opus-5-5", usage=_u(o=10**6), purpose="a1", request_id="r1",
                  now=datetime(2026, 10, 31, 23, 0, tzinfo=UTC))
    c = ledger.record(p, model="claude-opus-5-5", usage=_u(i=10**6), purpose="a2", request_id="r2",
                      now=datetime(2026, 11, 1, 1, 0, tzinfo=UTC))
    assert c == pytest.approx(4.0)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert [r["purpose"] for r in rows] == ["a1", "a2"] and rows[1]["request_id"] == "r2"
    assert ledger.month_to_date(p, datetime(2026, 11, 8, tzinfo=UTC)) == pytest.approx(4.0)
    assert ledger.month_to_date(p, datetime(2026, 10, 15, tzinfo=UTC)) == pytest.approx(20.0)
    assert ledger.month_to_date(tmp_path / "missing.jsonl") == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_anomalies_ledger.py -q`
Expected: FAIL — `cannot import name 'ledger'`.

- [ ] **Step 3: Implement**

`src/cfbr/anomalies/ledger.py`:

```python
"""Anthropic API spend ledger: one JSON line per call in data/anthropic_calls.jsonl, like the
CFBD quota ledger. `cfbr investigate` refuses to start once month-to-date spend (UTC calendar
month) reaches [investigate].monthly_budget_usd. It is a runaway guard; the account's monthly
credits, which hard-stop when used up, are the backstop."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

# USD per million tokens: input, output, cache write (5-minute), cache read.
PRICES = {"claude-opus-5-5": (4.0, 20.0, 5.0, 0.20)}
# Any other model (e.g. one a refusal fallback routed to) is priced at the top tier, so spend
# is over-counted rather than under-counted.
FALLBACK_PRICE = (10.0, 50.0, 12.5, 1.0)

_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _tokens(usage) -> list[int]:
    return [int(getattr(usage, f, None) or 0) for f in _FIELDS]


def call_cost(model: str, usage) -> float:
    prices = PRICES.get(model, FALLBACK_PRICE)
    return sum(n * p for n, p in zip(_tokens(usage), prices)) / 1e6


def record(path: Path, *, model: str, usage, purpose: str, request_id: str | None,
           now: datetime | None = None) -> float:
    cost = call_cost(model, usage)
    row = {"ts": (now or datetime.now(UTC)).isoformat(), "model": model, "purpose": purpose,
           "request_id": request_id, **dict(zip(_FIELDS, _tokens(usage))), "cost_usd": round(cost, 6)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return cost


def month_to_date(path: Path, now: datetime | None = None) -> float:
    if not path.exists():
        return 0.0
    month = (now or datetime.now(UTC)).strftime("%Y-%m")
    total = 0.0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["ts"].startswith(month):
            total += row["cost_usd"]
    return total
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_anomalies_ledger.py -q` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/anomalies/ledger.py tests/test_anomalies_ledger.py
git commit -m "feat(anomalies): Anthropic spend ledger and monthly guard"
```

---

### Task 3: Sandboxed agent tools

**Files:**
- Create: `src/cfbr/anomalies/tools.py`
- Test: `tests/test_anomalies_tools.py`

**Interfaces:**
- Consumes: `cfbr.config.Investigate` (query_timeout_seconds, query_max_rows, query_max_chars).
- Produces: `OUTPUT_FILES: dict[str, str]` (name → description; keys `rankings.csv`, `rankings_prev.csv`, `predictions.csv`, `accuracy.csv`); `class ToolSession(db_path: Path, out_dir: Path, season: int, week: int, cfg: Investigate)` with `query(sql: str) -> tuple[str, bool]`, `read_output(name: str) -> tuple[str, bool]` (text, is_error), attributes `results: dict[int, str]` (successful results by id) and `log: list[dict]` (every call: `{"id", "tool", "input", "ok", "text"}`), and `close()`. Results from both tools share one id sequence starting at 1; each result's text starts with `result_id {n}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_anomalies_tools.py`:

```python
import duckdb
import polars as pl
import pytest

from cfbr.anomalies.tools import OUTPUT_FILES, ToolSession
from cfbr.config import Investigate


@pytest.fixture
def env(tmp_path):
    db = tmp_path / "snap.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE games AS SELECT range AS game_id, 'T' || range AS home_team FROM range(500)")
    out = tmp_path / "out"
    (out / "2026" / "w09").mkdir(parents=True)
    (out / "2026" / "w08").mkdir(parents=True)
    pl.DataFrame({"team": ["Iowa"], "rating_change": [-5.0]}).write_csv(out / "2026" / "w09" / "rankings.csv")
    pl.DataFrame({"team": ["Iowa"], "rating_change": [1.0]}).write_csv(out / "2026" / "w08" / "rankings.csv")
    s = ToolSession(db, out, 2026, 9, Investigate(query_max_rows=200, query_max_chars=20000))
    yield s
    s.close()


def test_query_returns_table_with_result_id_and_keeps_it(env):
    text, err = env.query("SELECT game_id, home_team FROM games WHERE game_id < 2 ORDER BY 1")
    assert not err
    assert text.splitlines()[:4] == ["result_id 1 (2 rows)", "game_id\thome_team", "0\tT0", "1\tT1"]
    assert env.results[1] == text and env.log[0]["ok"] and env.log[0]["tool"] == "query"


def test_query_caps_rows_and_chars(env):
    text, _ = env.query("SELECT * FROM games")
    assert "result_id 1 (200 rows)" in text and "more than 200 rows" in text
    small = ToolSession(env.db_path, env.out_dir, 2026, 9, Investigate(query_max_chars=50))
    text2, _ = small.query("SELECT * FROM games LIMIT 100")
    assert text2.endswith("… truncated") and len(text2) < 200
    small.close()


@pytest.mark.parametrize("sql", ["SELECT * FROM read_csv('/etc/hosts')", "ATTACH '/tmp/x.db'",
                                 "COPY (SELECT 1) TO '/tmp/y.csv'", "CREATE TABLE t AS SELECT 1",
                                 "SET enable_external_access = true", "INSTALL httpfs"])
def test_sandbox_refuses_writes_files_and_config(env, sql):
    text, err = env.query(sql)
    assert err and text.startswith("result_id 1: error:")
    assert 1 not in env.results and env.log[0]["ok"] is False


def test_query_timeout_interrupts(env):
    slow = ToolSession(env.db_path, env.out_dir, 2026, 9, Investigate(query_timeout_seconds=0.2))
    text, err = slow.query("SELECT count(*) FROM range(10000000000) a")
    assert err and "timed out after 0.2s" in text
    slow.close()


def test_read_output_whitelist_and_shared_ids(env):
    env.query("SELECT 1 AS x")
    text, err = env.read_output("rankings_prev.csv")
    assert not err and text.startswith("result_id 2 (rankings_prev.csv)") and "Iowa,1.0" in text
    assert env.results[2] == text
    bad, err2 = env.read_output("../../.keys")
    assert err2 and "allowed:" in bad and 3 not in env.results
    missing, err3 = env.read_output("predictions.csv")
    assert err3 and "not available" in missing
    assert set(OUTPUT_FILES) == {"rankings.csv", "rankings_prev.csv", "predictions.csv", "accuracy.csv"}
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_anomalies_tools.py -q` → collection error (no module `cfbr.anomalies.tools`).

- [ ] **Step 3: Implement**

`src/cfbr/anomalies/tools.py`:

```python
"""Tools for the investigation agent.

`query` runs SQL on a snapshot of cfb.duckdb opened read-only, with external access disabled
and configuration locked: no writes, no file reads (read_csv on .keys), no ATTACH, COPY or
INSTALL, and no SET to undo that. `read_output` serves a fixed set of `rank` output files by
name, never by path. Both tools number their results in one sequence; a verdict's evidence cites
those ids and the grounding check (agent.grounded) looks them up in `results`.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb

from cfbr.config import Investigate

OUTPUT_FILES = {
    "rankings.csv": "this week's ratings: rank, team, rating, offense_rating, defense_rating, rating_change, prior_share, ...",
    "rankings_prev.csv": "the previous week's rankings.csv",
    "predictions.csv": "next week's games: pred_margin (model home margin) and closing_spread (current consensus, home perspective)",
    "accuracy.csv": "season-to-date accuracy line, one row per week",
}


def _cell(v) -> str:
    return "NULL" if v is None else str(v)


class ToolSession:
    def __init__(self, db_path: Path, out_dir: Path, season: int, week: int, cfg: Investigate):
        self.db_path, self.out_dir, self.season, self.week, self.cfg = db_path, out_dir, season, week, cfg
        self.con = duckdb.connect(str(db_path), read_only=True,
                                  config={"enable_external_access": False, "lock_configuration": True})
        self.results: dict[int, str] = {}
        self.log: list[dict] = []

    def _next_id(self) -> int:
        return len(self.log) + 1

    def _cap(self, text: str) -> str:
        limit = self.cfg.query_max_chars
        return text if len(text) <= limit else text[:limit] + "\n… truncated"

    def _done(self, rid: int, tool: str, inp: str, text: str, ok: bool) -> tuple[str, bool]:
        self.log.append({"id": rid, "tool": tool, "input": inp, "ok": ok, "text": text})
        if ok:
            self.results[rid] = text
        return text, not ok

    def query(self, sql: str) -> tuple[str, bool]:
        rid = self._next_id()
        timed_out = threading.Event()

        def stop() -> None:
            timed_out.set()
            self.con.interrupt()

        timer = threading.Timer(self.cfg.query_timeout_seconds, stop)
        timer.start()
        try:
            cur = self.con.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(self.cfg.query_max_rows + 1)
        except Exception as e:  # noqa: BLE001 - every failure goes back to the model as a tool error
            why = f"timed out after {self.cfg.query_timeout_seconds:g}s" if timed_out.is_set() else f"{type(e).__name__}: {e}"
            return self._done(rid, "query", sql, f"result_id {rid}: error: {why}"[:2000], False)
        finally:
            timer.cancel()
        shown = rows[: self.cfg.query_max_rows]
        body = "\n".join(["\t".join(cols)] + ["\t".join(_cell(v) for v in r) for r in shown])
        if len(rows) > self.cfg.query_max_rows:
            body += f"\n… more than {self.cfg.query_max_rows} rows; aggregate or add LIMIT"
        return self._done(rid, "query", sql, self._cap(f"result_id {rid} ({len(shown)} rows)\n{body}"), True)

    def read_output(self, name: str) -> tuple[str, bool]:
        rid = self._next_id()
        season_dir = self.out_dir / str(self.season)
        paths = {"rankings.csv": season_dir / f"w{self.week:02d}" / "rankings.csv",
                 "rankings_prev.csv": season_dir / f"w{self.week - 1:02d}" / "rankings.csv",
                 "predictions.csv": season_dir / f"w{self.week:02d}" / "predictions.csv",
                 "accuracy.csv": season_dir / "accuracy.csv"}
        if name not in paths:
            return self._done(rid, "read_output", name,
                              f"result_id {rid}: error: unknown file {name!r}; allowed: {', '.join(paths)}", False)
        path = paths[name]
        if not path.exists():
            return self._done(rid, "read_output", name,
                              f"result_id {rid}: error: {name} is not available for {self.season} week {self.week}", False)
        return self._done(rid, "read_output", name, self._cap(f"result_id {rid} ({name})\n{path.read_text()}"), True)

    def close(self) -> None:
        self.con.close()
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_anomalies_tools.py -q` → 10 passed (6 parametrized sandbox cases). If the timeout test's SQL finishes too fast on the machine, raise the range bound rather than the timeout; do not remove the test.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/anomalies/tools.py tests/test_anomalies_tools.py
git commit -m "feat(anomalies): sandboxed query and output tools for the investigation agent"
```

---

### Task 4: Prompt and tool schemas

**Files:**
- Create: `src/cfbr/anomalies/prompt.py`
- Test: `tests/test_anomalies_agent.py` (create; prompt tests now, agent tests in Task 5)

**Interfaces:**
- Consumes: `OUTPUT_FILES` (Task 3).
- Produces: `VERDICT_SCHEMA: dict`, `TOOLS: list[dict]` (names `query`, `read_output`, `submit_verdict`, all `strict: True`), `SCHEMA_TABLES: tuple[str, ...]`, `schema_text(db_path: Path) -> str`, `build_system(schema: str) -> str`, `item_message(row: dict, context: list[dict]) -> str`, `ITEM_FIELDS: tuple[str, ...]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_anomalies_agent.py`:

```python
import json

import duckdb

from cfbr.anomalies.prompt import TOOLS, VERDICT_SCHEMA, build_system, item_message, schema_text


def _strict_ok(schema):
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for sub in schema["properties"].values():
            _strict_ok(sub)
    if schema.get("type") == "array":
        _strict_ok(schema["items"])


def test_tools_are_strict_and_complete():
    assert [t["name"] for t in TOOLS] == ["query", "read_output", "submit_verdict"]
    for t in TOOLS:
        assert t["strict"] is True
        _strict_ok(t["input_schema"])
    assert VERDICT_SCHEMA["properties"]["verdict"]["enum"] == ["data_bug", "real", "inconclusive"]
    assert TOOLS[1]["input_schema"]["properties"]["name"]["enum"] == [
        "rankings.csv", "rankings_prev.csv", "predictions.csv", "accuracy.csv"]


def test_system_prompt_carries_schema_and_domain_notes(tmp_path):
    db = tmp_path / "s.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE games (game_id BIGINT, home_team VARCHAR)")
    schema = schema_text(db)
    assert schema == "games(game_id BIGINT, home_team VARCHAR)"
    system = build_system(schema)
    for needle in ["games(game_id BIGINT, home_team VARCHAR)", "play_id::DECIMAL(38,0)", "after the play",
                   "2025 week 8", "submit_verdict", "result_id", "model_plays"]:
        assert needle in system, needle


def test_item_message_has_the_record_and_context_but_no_status():
    row = {"id": "abc", "check": "B1", "season": 2026, "week": 9, "key": "Iowa", "message": "Iowa: rating -6.2",
           "value": -6.2, "threshold": 6.26, "game_id": None, "team": "Iowa", "status": "pending", "verdict": None}
    ctx = [{"check": "B2", "message": "Iowa in game 7: offense -0.60 PPA/play", "team": "Iowa", "game_id": 7}]
    msg = item_message(row, ctx)
    payload = json.loads(msg.split("\n", 1)[1].rsplit("\n\n", 1)[0])
    assert payload["anomaly"]["check"] == "B1" and "status" not in payload["anomaly"]
    assert payload["context"] == [{"check": "B2", "message": "Iowa in game 7: offense -0.60 PPA/play"}]
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_anomalies_agent.py -q` → collection error.

- [ ] **Step 3: Implement**

`src/cfbr/anomalies/prompt.py`:

```python
"""Frozen system prompt, tool schemas and the per-item message for the investigation agent.
The system prompt is identical for every item in a run (schema built once from the snapshot),
so it caches across a run's investigations. Domain notes come from DECISIONS.md and CLAUDE.md."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from cfbr.anomalies.tools import OUTPUT_FILES

SCHEMA_TABLES = ("games", "plays_raw", "model_plays", "team_status", "lines", "teams_season", "calendar")
ITEM_FIELDS = ("check", "season", "week", "key", "message", "value", "threshold", "game_id", "team")

VERDICT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["verdict", "confidence", "summary", "evidence", "affected", "model_impact", "suggested_action"],
    "properties": {
        "verdict": {"type": "string", "enum": ["data_bug", "real", "inconclusive"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string", "description": "At most 280 characters, readable on a phone."},
        "evidence": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["claim", "query_id", "excerpt"],
            "properties": {
                "claim": {"type": "string"},
                "query_id": {"type": "integer", "description": "The result_id of a query or read_output result."},
                "excerpt": {"type": "string", "description": "Copied verbatim from that result, at most 500 characters."}}}},
        "affected": {"type": "object", "additionalProperties": False, "required": ["game_ids", "teams"],
                     "properties": {"game_ids": {"type": "array", "items": {"type": "integer"}},
                                    "teams": {"type": "array", "items": {"type": "string"}}}},
        "model_impact": {"type": "string", "enum": ["none", "small", "material", "unknown"]},
        "suggested_action": {"type": "string", "description": "At most 200 characters; advisory, never executed."},
    },
}

TOOLS = [
    {"name": "query", "strict": True,
     "description": "Run read-only DuckDB SQL on the cfbr store (a snapshot taken at the start of this run). "
                    "Returns a tab-separated table headed by its result_id. At most 200 rows and 20,000 characters "
                    "are shown: aggregate or use LIMIT.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["sql"],
                      "properties": {"sql": {"type": "string"}}}},
    {"name": "read_output", "strict": True,
     "description": "Read one of this anomaly week's `cfbr rank` output files. "
                    + "; ".join(f"{k}: {v}" for k, v in OUTPUT_FILES.items()),
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["name"],
                      "properties": {"name": {"type": "string", "enum": list(OUTPUT_FILES)}}}},
    {"name": "submit_verdict", "strict": True,
     "description": "Finish the investigation. Call it exactly once, as your last action.",
     "input_schema": VERDICT_SCHEMA},
]

SYSTEM_TEMPLATE = """You investigate data anomalies flagged by cfbr, a college football rating engine. A deterministic \
check flagged the anomaly in the user message. Decide whether it is a data bug (the source data misrepresents what \
happened) or real (the data is internally consistent and reflects what happened on the field, or a legitimate model \
reaction). Your verdict is advisory: it goes to the operator's phone and to a report. It never changes ratings, bets \
or data.

## Store schema
{schema}

## Data facts
- plays_raw has one row per CFBD play. play_id is VARCHAR: order plays within a game numerically with \
play_id::DECIMAL(38,0), never as text. Late-game order is unreliable: the last play by id misses the final score in \
5-10% of games, so compare maximum scores, not the last row.
- offense_score and defense_score are the score after the play, from the offense's view. Home score = offense_score \
when offense = home, else defense_score.
- games has one row per game: home_points/away_points are finals, home_class/away_class are 'fbs', 'fcs', 'ii', \
'iii' or NULL, plus completed and neutral_site.
- model_plays holds the plays the model fits on, after filtering. Kickoffs, punts, field goals, penalty-only rows, \
timeouts, end-of-period rows, two-point tries and uncategorized rows are dropped. Plays where both offense and \
defense are non-FBS are dropped. Non-FBS teams are pooled into FCS entities (offense_fcs/defense_fcs), and plays with \
a pooled FCS side get reduced weight in the fit.
- Penalty-only rows have yards_gained signed from the offense's view through 2025 week 8 and unsigned after.
- CFBD has listed the same play twice: hundreds of games a season in 2022-2025, rare in 2026.
- team_status is derived from games plus CFBD's FBS team list, per season.
- lines holds the latest spread per game and provider, home perspective (negative = home favored); the consensus is \
the median across providers. Lines are never model inputs.
- A team's rating is its expected point margin against an average FBS team on a neutral field: an opponent-adjusted \
ridge fit on play efficiency (PPA and success rate) with recency weighting, shrunk toward a preseason prior.

## Method
- Start with targeted queries: the game's row, play counts by period or drive, duplicate rows, the plays behind a \
score. Use LIMIT and aggregates. Stop as soon as the evidence settles the question.
- Evidence must come from your tool results. Each evidence item cites the result_id as query_id and an excerpt \
copied verbatim from that result (at most 500 characters). Football knowledge from memory (injuries, weather, real \
results) is not evidence; if it matters, say so in the summary as unverified.
- verdict: data_bug, real, or inconclusive when the evidence does not settle it. data_bug and real need at least one \
evidence item.
- confidence: high only when a result shows it directly.
- model_impact: none if the affected rows never reach model_plays or no FBS team is involved; small if they change \
under about 5% of a team's model plays this season; material if more; unknown otherwise. Check with a query.
- Finish by calling submit_verdict exactly once.
"""


def schema_text(db_path: Path) -> str:
    lines = []
    with duckdb.connect(str(db_path), read_only=True) as con:
        tables = {t for (t,) in con.execute("SHOW TABLES").fetchall()}
        for table in SCHEMA_TABLES:
            if table in tables:
                cols = con.execute(f"DESCRIBE {table}").fetchall()
                lines.append(f"{table}({', '.join(f'{c[0]} {c[1]}' for c in cols)})")
    return "\n".join(lines)


def build_system(schema: str) -> str:
    return SYSTEM_TEMPLATE.replace("{schema}", schema)


def item_message(row: dict, context: list[dict]) -> str:
    payload = {"anomaly": {k: row.get(k) for k in ITEM_FIELDS},
               "context": [{"check": c["check"], "message": c["message"]} for c in context]}
    return ("Investigate this anomaly. Context rows are related single-game efficiency outliers, for information.\n"
            + json.dumps(payload, indent=2) + "\n\nFinish with submit_verdict.")
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_anomalies_agent.py -q` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/anomalies/prompt.py tests/test_anomalies_agent.py
git commit -m "feat(anomalies): investigation prompt and strict tool schemas"
```

---

### Task 5: The investigation loop and grounding check

**Files:**
- Create: `src/cfbr/anomalies/agent.py`, `tests/anomaly_fakes.py`
- Test: `tests/test_anomalies_agent.py` (append)

**Interfaces:**
- Consumes: `ToolSession` (Task 3), `TOOLS`, `item_message` (Task 4), `ledger.record` (Task 2), `Investigate`.
- Produces: `BETAS: list[str]`; `inconclusive(reason: str, summary: str) -> dict`; `grounded(verdict: dict, results: dict[int, str]) -> bool`; `investigate(client, row: dict, context: list[dict], session: ToolSession, system: str, cfg: Investigate, ledger_path: Path, clock=time.monotonic) -> dict` returning the verdict fields plus `reason` (None | "budget" | "refused" | "ungrounded" | "no_verdict") and `meta` ({"turns", "queries", "cost_usd", "request_ids", "served_by", "duration_s", "refusal_category"}); an ungrounded result also carries `original_verdict`. API exceptions propagate to the caller.
- Produces (tests): `tests/anomaly_fakes.py` with `usage()`, `tool_use(name, inp, id_)`, `text_block(t)`, `message(content, stop="tool_use", model="claude-opus-5-5", details=None)`, `FakeClient(script)` (script items: a message, an Exception to raise, or a callable `kw -> message`; `calls` records each request with a copy of `messages`), `FakeAPIError(status_code, type_=None)`.

- [ ] **Step 1: Write the fakes**

`tests/anomaly_fakes.py`:

```python
"""A fake Anthropic client for agent tests: no network, scripted responses."""

from types import SimpleNamespace as NS


def usage(i=1000, o=200):
    return NS(input_tokens=i, output_tokens=o, cache_creation_input_tokens=0, cache_read_input_tokens=0)


def tool_use(name, inp, id_):
    return NS(type="tool_use", name=name, input=inp, id=id_)


def text_block(t):
    return NS(type="text", text=t)


def message(content, stop="tool_use", model="claude-opus-5-5", details=None):
    return NS(content=content, stop_reason=stop, model=model, usage=usage(), _request_id="req_1", stop_details=details)


class _Stream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Stream(item(kw) if callable(item) else item)


class FakeAPIError(Exception):
    def __init__(self, status_code, type_=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.type = type_
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_anomalies_agent.py` (merge imports at the top):

```python
import pytest

from cfbr.anomalies.agent import BETAS, grounded, investigate
from cfbr.anomalies.tools import ToolSession
from cfbr.config import Investigate
from tests.anomaly_fakes import FakeClient, message, text_block, tool_use

ROW = {"id": "abc123", "check": "A4", "season": 2026, "week": 9, "key": "1", "message": "UTSA @ Rice: 4 duplicate play rows",
       "value": 4.0, "threshold": 3.0, "game_id": 1, "team": None, "status": "pending", "verdict": None}


def _verdict(**over):
    v = {"verdict": "data_bug", "confidence": "high", "summary": "Drive 7 is listed twice.",
         "evidence": [{"claim": "four extra rows", "query_id": 1, "excerpt": "extra\n4"}],
         "affected": {"game_ids": [1], "teams": ["Rice"]}, "model_impact": "small",
         "suggested_action": "Re-fetch the game's plays."}
    v.update(over)
    return v


@pytest.fixture
def session(tmp_path):
    db = tmp_path / "s.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE t AS SELECT 4 AS extra")
    s = ToolSession(db, tmp_path / "out", 2026, 9, Investigate())
    yield s
    s.close()


def _run(client, session, tmp_path, cfg=None, clock=None):
    kw = {"clock": clock} if clock else {}
    return investigate(client, ROW, [], session, "SYSTEM", cfg or Investigate(), tmp_path / "calls.jsonl", **kw)


def test_grounded_verdict_is_kept_and_request_shape_is_right(session, tmp_path):
    client = FakeClient([message([tool_use("query", {"sql": "SELECT extra FROM t"}, "tu1")]),
                         message([tool_use("submit_verdict", _verdict(), "tu2")])])
    out = _run(client, session, tmp_path)
    assert out["verdict"] == "data_bug" and out["reason"] is None
    assert out["meta"]["turns"] == 2 and out["meta"]["queries"] == 1 and out["meta"]["cost_usd"] > 0
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["betas"] == BETAS and first["fallbacks"] == "default"
    assert first["output_config"] == {"effort": "xhigh", "task_budget": {"type": "tokens", "total": 60000}}
    assert first["cache_control"] == {"type": "ephemeral"} and "thinking" not in first and "tool_choice" not in first
    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and second[2]["content"][0]["tool_use_id"] == "tu1"
    assert second[2]["content"][0]["is_error"] is False and "result_id 1" in second[2]["content"][0]["content"]
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 2


def test_ungrounded_excerpt_downgrades_to_inconclusive(session, tmp_path):
    bad = _verdict(evidence=[{"claim": "x", "query_id": 1, "excerpt": "38 duplicate rows"}])
    client = FakeClient([message([tool_use("query", {"sql": "SELECT extra FROM t"}, "tu1")]),
                         message([tool_use("submit_verdict", bad, "tu2")])])
    out = _run(client, session, tmp_path)
    assert (out["verdict"], out["confidence"], out["reason"]) == ("inconclusive", "low", "ungrounded")
    assert out["original_verdict"]["verdict"] == "data_bug"


def test_grounded_rules():
    results = {1: "result_id 1 (1 rows)\nextra\n4"}
    assert grounded(_verdict(), results)
    assert not grounded(_verdict(evidence=[]), results)                       # data_bug needs evidence
    assert grounded(_verdict(verdict="inconclusive", evidence=[]), results)
    assert not grounded(_verdict(evidence=[{"claim": "c", "query_id": 9, "excerpt": "4"}]), results)
    assert not grounded(_verdict(evidence=[{"claim": "c", "query_id": 1, "excerpt": "  "}]), results)
    assert grounded(_verdict(evidence=[{"claim": "c", "query_id": 1, "excerpt": "extra   4"}]), results)


def test_turn_cap_gives_inconclusive_budget(session, tmp_path):
    client = FakeClient([message([tool_use("query", {"sql": "SELECT 1"}, f"tu{i}")]) for i in range(3)])
    out = _run(client, session, tmp_path, cfg=Investigate(max_turns=3))
    assert (out["verdict"], out["reason"]) == ("inconclusive", "budget") and len(client.calls) == 3


def test_wall_clock_gives_inconclusive_budget(session, tmp_path):
    ticks = iter([0.0, 0.0, 10_000.0, 10_000.0])
    client = FakeClient([message([tool_use("query", {"sql": "SELECT 1"}, "tu1")])])
    out = _run(client, session, tmp_path, clock=lambda: next(ticks))
    assert out["reason"] == "budget" and len(client.calls) == 1


def test_refusal_is_recorded(session, tmp_path):
    from types import SimpleNamespace as NS
    client = FakeClient([message([], stop="refusal", details=NS(category="cyber"))])
    out = _run(client, session, tmp_path)
    assert (out["verdict"], out["reason"]) == ("inconclusive", "refused")
    assert out["meta"]["refusal_category"] == "cyber"


def test_text_only_turn_gets_a_nudge_then_verdict(session, tmp_path):
    client = FakeClient([message([text_block("Thinking it over.")], stop="end_turn"),
                         message([tool_use("submit_verdict", _verdict(verdict="inconclusive", evidence=[]), "tu1")])])
    out = _run(client, session, tmp_path)
    assert out["verdict"] == "inconclusive" and out["reason"] is None
    assert client.calls[1]["messages"][-1] == {"role": "user", "content": "Finish by calling submit_verdict."}


def test_query_errors_go_back_as_tool_errors_and_long_fields_are_clipped(session, tmp_path):
    long = _verdict(verdict="inconclusive", evidence=[], summary="s" * 400, suggested_action="a" * 300)
    client = FakeClient([message([tool_use("query", {"sql": "SELECT nope FROM t"}, "tu1")]),
                         message([tool_use("submit_verdict", long, "tu2")])])
    out = _run(client, session, tmp_path)
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True
    assert len(out["summary"]) == 280 and len(out["suggested_action"]) == 200


def test_api_errors_propagate(session, tmp_path):
    from tests.anomaly_fakes import FakeAPIError
    with pytest.raises(FakeAPIError):
        _run(FakeClient([FakeAPIError(529, "overloaded_error")]), session, tmp_path)
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/test_anomalies_agent.py -q` → collection error (no `cfbr.anomalies.agent`).

- [ ] **Step 4: Implement**

`src/cfbr/anomalies/agent.py`:

```python
"""One investigation: a manual tool loop over the Messages API (beta), capped by turns and
wall clock, ending in a submit_verdict call that is checked against the session's own tool
results before it is trusted. API exceptions propagate; run.investigate_doc classifies them."""

from __future__ import annotations

import time
from pathlib import Path

from cfbr.anomalies import ledger
from cfbr.anomalies.prompt import TOOLS, item_message
from cfbr.anomalies.tools import ToolSession
from cfbr.config import Investigate

BETAS = ["task-budgets-2026-03-13", "server-side-fallback-2026-07-01"]
NUDGE = "Finish by calling submit_verdict."
REASON_TEXT = {"budget": "Stopped at the turn or time limit before a verdict.",
               "refused": "The model declined this request.",
               "no_verdict": "No verdict was submitted."}


def inconclusive(reason: str, summary: str) -> dict:
    return {"verdict": "inconclusive", "confidence": "low", "summary": summary[:280], "evidence": [],
            "affected": {"game_ids": [], "teams": []}, "model_impact": "unknown", "suggested_action": ""}


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def grounded(verdict: dict, results: dict[int, str]) -> bool:
    evidence = verdict.get("evidence") or []
    if verdict.get("verdict") in ("data_bug", "real") and not evidence:
        return False
    for item in evidence:
        source = results.get(item.get("query_id"))
        excerpt = _norm(item.get("excerpt", ""))
        if source is None or not excerpt or excerpt not in _norm(source):
            return False
    return True


def _clip(verdict: dict) -> dict:
    v = dict(verdict)
    v["summary"] = str(v.get("summary", ""))[:280]
    v["suggested_action"] = str(v.get("suggested_action", ""))[:200]
    v["evidence"] = [{**e, "excerpt": str(e.get("excerpt", ""))[:500]} for e in v.get("evidence") or []]
    return v


def _run_tool(session: ToolSession, block) -> tuple[str, bool]:
    if block.name == "query":
        return session.query(str(block.input.get("sql", "")))
    if block.name == "read_output":
        return session.read_output(str(block.input.get("name", "")))
    return f"unknown tool {block.name!r}", True


def investigate(client, row: dict, context: list[dict], session: ToolSession, system: str, cfg: Investigate,
                ledger_path: Path, clock=time.monotonic) -> dict:
    start = clock()
    messages: list[dict] = [{"role": "user", "content": item_message(row, context)}]
    output_config: dict = {"effort": cfg.effort}
    if cfg.task_budget_tokens:
        output_config["task_budget"] = {"type": "tokens", "total": cfg.task_budget_tokens}
    meta = {"turns": 0, "cost_usd": 0.0, "request_ids": [], "served_by": [], "refusal_category": None}
    verdict, reason = None, None
    for _ in range(cfg.max_turns):
        if clock() - start > cfg.wall_clock_minutes * 60:
            reason = "budget"
            break
        with client.beta.messages.stream(model=cfg.model, max_tokens=cfg.max_tokens, system=system,
                                         cache_control={"type": "ephemeral"}, tools=TOOLS, messages=messages,
                                         output_config=output_config, betas=BETAS, fallbacks="default") as stream:
            msg = stream.get_final_message()
        meta["turns"] += 1
        request_id = getattr(msg, "_request_id", None)
        meta["cost_usd"] += ledger.record(ledger_path, model=msg.model, usage=msg.usage, purpose=row["id"],
                                          request_id=request_id)
        meta["request_ids"].append(request_id)
        if msg.model not in meta["served_by"]:
            meta["served_by"].append(msg.model)
        messages.append({"role": "assistant", "content": msg.content})
        if msg.stop_reason == "refusal":
            reason = "refused"
            meta["refusal_category"] = getattr(getattr(msg, "stop_details", None), "category", None)
            break
        uses = [b for b in msg.content if b.type == "tool_use"]
        submit = next((b for b in uses if b.name == "submit_verdict"), None)
        if submit is not None:
            verdict = dict(submit.input)
            break
        if not uses:
            messages.append({"role": "user", "content": NUDGE})
            continue
        results = []
        for block in uses:
            text, is_error = _run_tool(session, block)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": is_error})
        messages.append({"role": "user", "content": results})
    else:
        reason = "budget"

    if verdict is None:
        out = inconclusive(reason or "no_verdict", REASON_TEXT[reason or "no_verdict"])
        reason = reason or "no_verdict"
    else:
        clipped = _clip(verdict)
        if grounded(clipped, session.results):
            out = clipped
        else:
            reason = "ungrounded"
            out = inconclusive(reason, clipped["summary"])
            out["original_verdict"] = clipped
    out["reason"] = reason
    meta["queries"] = len(session.log)
    meta["cost_usd"] = round(meta["cost_usd"], 4)
    meta["duration_s"] = round(clock() - start, 1)
    out["meta"] = meta
    return out
```

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_anomalies_agent.py -q` → 12 passed.

- [ ] **Step 6: Commit**

```bash
git add src/cfbr/anomalies/agent.py tests/anomaly_fakes.py tests/test_anomalies_agent.py
git commit -m "feat(anomalies): investigation loop with caps, refusal handling and grounding check"
```

---

### Task 6: Orchestration (`investigate_doc`)

**Files:**
- Modify: `src/cfbr/anomalies/run.py` (append)
- Test: `tests/test_anomalies_investigate.py` (append)

**Interfaces:**
- Consumes: `investigate` (Task 5), `ToolSession` (Task 3), `build_system`, `schema_text` (Task 4), `ledger.month_to_date` (Task 2), `update_doc`, `read_doc`, `render_md`, `SEVERITY` (records).
- Produces:
  - `plan_targets(doc: dict, force_ids: tuple[str, ...], max_items: int) -> tuple[list[dict], list[dict]]` — (to investigate, overflow).
  - `classify_error(e: BaseException) -> str` — "transient" | "auth" | "budget" | "item".
  - `try_lock(path: Path)` — an open locked file, or None if held.
  - `investigate_doc(cfg: Config, doc_path: Path, client, *, now: datetime | None = None, force_ids: tuple[str, ...] = (), clock=time.monotonic) -> dict` — summary `{"status": "no_doc"|"nothing"|"budget"|"locked"|"done", "items": [{"id","check","message","verdict","confidence","summary","model_impact","reason"}], "skipped_volume": int, "skipped_budget": int, "errors": int, "stopped": None|"auth"|"budget", "cost_usd": float, "label": str}`.
  - `investigate_alert(summary: dict) -> tuple[str, str, str] | None` — (title, body, priority) or None when nothing should be sent.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anomalies_investigate.py` (merge imports at the top):

```python
import fcntl
import json
from datetime import UTC, datetime

import duckdb
import pytest

from cfbr.anomalies.records import Anomaly, merge, read_doc, update_doc, write_doc
from cfbr.anomalies.run import classify_error, investigate_alert, investigate_doc, plan_targets
from cfbr.config import Paths
from tests.anomaly_fakes import FakeAPIError, FakeClient, message, tool_use

NOW = datetime(2026, 11, 15, 16, tzinfo=UTC)
GOOD = {"verdict": "real", "confidence": "medium", "summary": "Consistent PBP.",
        "evidence": [{"claim": "170 plays", "query_id": 1, "excerpt": "170"}],
        "affected": {"game_ids": [1], "teams": []}, "model_impact": "none", "suggested_action": ""}


def _ok_script():
    return [message([tool_use("query", {"sql": "SELECT 170 AS n"}, "q")]),
            message([tool_use("submit_verdict", GOOD, "v")])]


@pytest.fixture
def env(tmp_path):
    cfg = Config(paths=Paths(raw_dir=tmp_path / "raw", db_path=tmp_path / "data" / "cfb.duckdb",
                             results_dir=tmp_path / "res", out_dir=tmp_path / "out"))
    cfg.investigate.ledger = tmp_path / "data" / "calls.jsonl"
    cfg.investigate.snapshot = tmp_path / "data" / "inv" / "snapshot.duckdb"
    cfg.investigate.transcripts = tmp_path / "data" / "inv"
    cfg.paths.db_path.parent.mkdir(parents=True)
    with duckdb.connect(str(cfg.paths.db_path)) as con:
        con.execute("CREATE TABLE games (game_id BIGINT, home_team VARCHAR)")
    items = [Anomaly("A2", 2026, 9, "1", "Navy @ Army: gap 19", value=19.0, investigate=True, game_id=1),
             Anomaly("A3", 2026, 9, "2", "Ohio @ Iowa: 105 plays", value=105.0, investigate=True, game_id=2),
             Anomaly("A1", 2026, 9, "3", "no pbp", investigate=False, game_id=3)]
    doc, _ = merge(None, items, 2026, 9, NOW)
    path = cfg.paths.out_dir / "2026" / "w09" / "anomalies.json"
    write_doc(path, doc)
    return cfg, path


def _status(path):
    return {r["check"]: r["status"] for r in read_doc(path)["items"]}


def test_plan_targets_orders_by_severity_and_splits_overflow():
    doc = {"items": [{"id": "b", "check": "B1", "key": "x", "status": "pending"},
                     {"id": "a", "check": "A2", "key": "y", "status": "pending"},
                     {"id": "c", "check": "A1", "key": "z", "status": "plain"},
                     {"id": "d", "check": "A3", "key": "w", "status": "investigated"}]}
    todo, over = plan_targets(doc, (), 1)
    assert [r["id"] for r in todo] == ["a"] and [r["id"] for r in over] == ["b"]
    todo2, over2 = plan_targets(doc, ("d",), 12)
    assert [r["id"] for r in todo2] == ["d"] and over2 == []


def test_investigates_pending_items_and_writes_verdicts(env):
    cfg, path = env
    summary = investigate_doc(cfg, path, FakeClient(_ok_script() + _ok_script()), now=NOW)
    assert summary["status"] == "done" and len(summary["items"]) == 2 and summary["cost_usd"] > 0
    assert _status(path) == {"A2": "investigated", "A3": "investigated", "A1": "plain"}
    row = next(r for r in read_doc(path)["items"] if r["check"] == "A2")
    assert row["verdict"]["verdict"] == "real" and row["investigated_at"] == NOW.isoformat()
    assert "real" in path.with_suffix(".md").read_text()
    transcript = cfg.investigate.transcripts / "2026" / "w09" / f"{row['id']}.json"
    assert json.loads(transcript.read_text())["queries"][0]["input"] == "SELECT 170 AS n"


def test_volume_overflow_is_marked_and_reported(env):
    cfg, path = env
    cfg.investigate.max_items = 1
    summary = investigate_doc(cfg, path, FakeClient(_ok_script()), now=NOW)
    assert _status(path)["A3"] == "skipped_volume" and summary["skipped_volume"] == 1


def test_budget_guard_marks_items_and_stops_before_any_call(env):
    cfg, path = env
    cfg.investigate.ledger.write_text(json.dumps({"ts": NOW.isoformat(), "cost_usd": 60.0}) + "\n")
    client = FakeClient([])
    summary = investigate_doc(cfg, path, client, now=NOW)
    assert summary["status"] == "budget" and client.calls == []
    assert _status(path)["A2"] == "skipped_budget"


def test_lock_held_skips_quietly(env):
    cfg, path = env
    with open(cfg.paths.db_path.parent / ".weekly.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = investigate_doc(cfg, path, FakeClient([]), now=NOW)
    assert summary["status"] == "locked" and _status(path)["A2"] == "pending"


def test_transient_errors_count_attempts_then_error(env):
    cfg, path = env
    cfg.investigate.max_attempts = 2
    investigate_doc(cfg, path, FakeClient([FakeAPIError(529, "overloaded_error"), FakeAPIError(500)]), now=NOW)
    row = next(r for r in read_doc(path)["items"] if r["check"] == "A2")
    assert row["status"] == "pending" and row["attempts"] == 1
    investigate_doc(cfg, path, FakeClient([FakeAPIError(529), FakeAPIError(529)]), now=NOW)
    assert _status(path) == {"A2": "error", "A3": "error", "A1": "plain"}


def test_auth_error_stops_the_run_and_leaves_items_pending(env):
    cfg, path = env
    summary = investigate_doc(cfg, path, FakeClient([FakeAPIError(401, "authentication_error")]), now=NOW)
    assert summary["stopped"] == "auth" and _status(path)["A2"] == "pending"


def test_billing_error_marks_remaining_skipped_budget(env):
    cfg, path = env
    summary = investigate_doc(cfg, path, FakeClient([FakeAPIError(402, "billing_error")]), now=NOW)
    assert summary["stopped"] == "budget" and _status(path) == {"A2": "skipped_budget", "A3": "skipped_budget", "A1": "plain"}


def test_concurrent_weekly_rerun_is_not_lost(env):
    cfg, path = env

    def rerun_then_query(kw):
        update_doc(path, lambda d: {**d, "items": d["items"] + [{"id": "new", "check": "B1", "key": "Iowa",
                                                               "status": "pending", "message": "m"}]})
        return message([tool_use("query", {"sql": "SELECT 170 AS n"}, "q")])

    investigate_doc(cfg, path, FakeClient([rerun_then_query, message([tool_use("submit_verdict", GOOD, "v")])]
                                          + _ok_script()), now=NOW)
    ids = {r["id"] for r in read_doc(path)["items"]}
    assert "new" in ids and _status(path)["A2"] == "investigated"


def test_classify_error():
    assert classify_error(FakeAPIError(429, "rate_limit_error")) == "transient"
    assert classify_error(FakeAPIError(503)) == "transient"
    assert classify_error(FakeAPIError(403, "permission_error")) == "auth"
    assert classify_error(FakeAPIError(402, "billing_error")) == "budget"
    assert classify_error(FakeAPIError(400, "invalid_request_error")) == "item"
    assert classify_error(ConnectionError("reset")) == "transient"


def test_investigate_alert_text():
    s = {"status": "done", "label": "w09", "cost_usd": 0.84, "skipped_volume": 0, "skipped_budget": 0,
         "errors": 0, "stopped": None,
         "items": [{"check": "A4", "message": "Rice–UTSA dup", "verdict": "data_bug", "confidence": "high",
                    "summary": "drive 14 duplicated", "model_impact": "material", "reason": None},
                   {"check": "B1", "message": "Iowa −6.2", "verdict": "real", "confidence": "medium",
                    "summary": "lost QB", "model_impact": "small", "reason": None}]}
    title, body, priority = investigate_alert(s)
    assert title == "cfbr anomalies w09 — 1 data bug, 1 real, 0 inconclusive ($0.84)"
    assert body.splitlines()[0] == "• A4 Rice–UTSA dup: data_bug/high — drive 14 duplicated; material"
    assert priority == "high"
    assert investigate_alert({"status": "nothing"}) is None and investigate_alert({"status": "locked"}) is None
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_anomalies_investigate.py -q` → ImportError on `classify_error`.

- [ ] **Step 3: Implement**

Append to `src/cfbr/anomalies/run.py` (add imports `fcntl`, `json`, `shutil`, `time` at the top, and `from cfbr.anomalies.records import SEVERITY, update_doc`):

```python
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
                if status and row["status"] != "resolved":
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


def investigate_doc(cfg: Config, doc_path: Path, client, *, now: datetime | None = None,
                    force_ids: tuple[str, ...] = (), clock=time.monotonic) -> dict:
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
        icfg.snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg.paths.db_path, icfg.snapshot)
    finally:
        lock.close()
    if overflow:
        _patch(doc_path, [r["id"] for r in overflow], {"status": "skipped_volume"})
        summary["skipped_volume"] = len(overflow)
    system = build_system(schema_text(icfg.snapshot))
    summary["status"] = "done"
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
            outcome = investigate(client, row, _context(doc, row), session, system, icfg, icfg.ledger, clock)
            _patch(doc_path, [row["id"]], {"status": "investigated", "verdict": outcome,
                                           "investigated_at": now.isoformat()})
            summary["cost_usd"] += outcome["meta"]["cost_usd"]
            summary["items"].append({"id": row["id"], "check": row["check"], "message": row["message"],
                                     **{k: outcome[k] for k in ("verdict", "confidence", "summary",
                                                                "model_impact", "reason")}})
        except Exception as e:  # noqa: BLE001 - classified below; one item never stops the others unless auth/budget
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
        finally:
            transcript = icfg.transcripts / str(row["season"]) / f"w{int(row['week']):02d}" / f"{row['id']}.json"
            transcript.parent.mkdir(parents=True, exist_ok=True)
            transcript.write_text(json.dumps({"row": row, "queries": session.log, "outcome": outcome},
                                             indent=2, default=str) + "\n")
            session.close()
    doc_path.with_suffix(".md").write_text(render_md(read_doc(doc_path)))
    summary["cost_usd"] = round(summary["cost_usd"], 2)
    return summary


def investigate_alert(summary: dict) -> tuple[str, str, str] | None:
    status = summary.get("status")
    if status == "budget":
        return ("cfbr investigate: monthly budget reached",
                f"{summary['skipped_budget']} item(s) marked skipped_budget; raise [investigate].monthly_budget_usd "
                "or wait for next month.", "default")
    if status != "done":
        return None
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
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_anomalies_investigate.py -q` → all pass. Then `uv run pytest -q`.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/anomalies/run.py tests/test_anomalies_investigate.py
git commit -m "feat(anomalies): investigate pending anomalies with snapshot, budget guard and error handling"
```

---

### Task 7: Shakedown sampling and CLI

**Files:**
- Create: `src/cfbr/anomalies/shakedown.py`
- Modify: `src/cfbr/cli.py` (new `investigate` command after `anomalies`; new options on `anomalies`)
- Test: `tests/test_anomalies_investigate.py` (append)

**Interfaces:**
- Consumes: `investigate_doc`, `investigate_alert`, `plan_targets`, `latest_week`, `week_dir`, `run_history` (run.py); `anthropic_api_key` (config); `cfbr.backtest.harness.parse_seasons(spec: str) -> list[int]`; `cfbr.ingest.fetch.current_season()`; `cfbr.alerts.send`.
- Produces: `pattern_group(row: dict, a3_min: int) -> str | None`; `sample_patterns(rows: list[dict], per_group: int, a3_min: int) -> list[dict]` (each row gains `"group"`); `shakedown_doc(rows: list[dict], seasons: list[int], now: datetime) -> dict`; CLI `cfbr investigate [--season] [--week] [--doc PATH] [--force ID ...] [--dry-run]`; `cfbr anomalies --sample-patterns --seasons 2022-2025 [--per-group 3]` (no `--season` needed for it: make `season` optional and require it only for the other modes).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_anomalies_investigate.py`:

```python
from typer.testing import CliRunner

from cfbr.anomalies.shakedown import pattern_group, sample_patterns, shakedown_doc


def _h(check, value, key):
    return {"id": f"{check}{key}", "check": check, "season": 2023, "week": 5, "key": key, "message": "m",
            "value": value, "threshold": None, "investigate": True, "alert": True, "game_id": int(key), "team": None}


def test_pattern_groups():
    assert pattern_group(_h("A2", None, "1"), 110) == "A2 no scores"
    assert [pattern_group(_h("A2", v, "1"), 110) for v in (9, 17, 31)] == ["A2 gap 9-16", "A2 gap 17-30", "A2 gap >30"]
    assert [pattern_group(_h("A3", v, "1"), 110) for v in (90, 300)] == ["A3 low", "A3 high"]
    assert [pattern_group(_h("A4", v, "1"), 110) for v in (4, 11, 41)] == ["A4 4-10", "A4 11-40", "A4 >40"]
    assert pattern_group(_h("A1", None, "1"), 110) is None and pattern_group(_h("B2", 1.5, "1"), 110) is None


def test_sample_patterns_is_deterministic_and_capped():
    rows = [_h("A4", 5, str(k)) for k in range(10)] + [_h("A2", 20, "99")]
    picked = sample_patterns(rows, 3, 110)
    assert [r["group"] for r in picked].count("A4 4-10") == 3 and len(picked) == 4
    assert picked == sample_patterns(list(reversed(rows)), 3, 110)


def test_shakedown_doc_is_pending_and_titled():
    doc = shakedown_doc(sample_patterns([_h("A3", 90, "1")], 3, 110), [2022, 2023], NOW)
    assert doc["title"] == "shakedown 2022–2023" and doc["items"][0]["status"] == "pending"
    assert doc["items"][0]["verdict"] is None and doc["items"][0]["group"] == "A3 low"


def test_cli_investigate_without_key_exits_zero(env, monkeypatch):
    from cfbr import cli
    cfg, path = env
    monkeypatch.setattr(cli, "_default_config", lambda c: cfg)
    monkeypatch.setattr("cfbr.config.anthropic_api_key", lambda: None)
    result = CliRunner().invoke(cli.app, ["investigate", "--doc", str(path)])
    assert result.exit_code == 0 and "investigation is off" in result.output
    assert _status(path)["A2"] == "pending"


def test_cli_investigate_dry_run_lists_targets_without_a_key(env, monkeypatch):
    from cfbr import cli
    cfg, path = env
    monkeypatch.setattr(cli, "_default_config", lambda c: cfg)
    monkeypatch.setattr("cfbr.config.anthropic_api_key", lambda: None)
    result = CliRunner().invoke(cli.app, ["investigate", "--doc", str(path), "--dry-run"])
    assert result.exit_code == 0
    assert "A2 Navy @ Army: gap 19" in result.output and "A3 Ohio @ Iowa: 105 plays" in result.output
    assert "would investigate 2" in result.output
```

- [ ] **Step 2: Run to verify failure** — ImportError on `cfbr.anomalies.shakedown`.

- [ ] **Step 3: Implement sampling**

`src/cfbr/anomalies/shakedown.py`:

```python
"""Historical shakedown set: sample up to N fired items per check and shape from
`cfbr anomalies --history` output, so the investigation agent can be judged on 2022–2025
before it runs on Sundays. Only checks the agent investigates (A2, A3, A4) are sampled;
B1/B3 need weekly rank output that history seasons don't have. Sampling is deterministic
(rows ordered by their hash id)."""

from __future__ import annotations

from datetime import datetime


def pattern_group(row: dict, a3_min: int) -> str | None:
    check, value = row["check"], row.get("value")
    if check == "A2":
        if value is None:
            return "A2 no scores"
        return "A2 gap 9-16" if value <= 16 else "A2 gap 17-30" if value <= 30 else "A2 gap >30"
    if check == "A3":
        return "A3 low" if value < a3_min else "A3 high"
    if check == "A4":
        return "A4 4-10" if value <= 10 else "A4 11-40" if value <= 40 else "A4 >40"
    return None


def sample_patterns(rows: list[dict], per_group: int, a3_min: int) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        group = pattern_group(row, a3_min)
        if group is not None and len(groups.setdefault(group, [])) < per_group:
            groups[group].append({**row, "group": group})
    return [row for group in sorted(groups) for row in groups[group]]


def shakedown_doc(rows: list[dict], seasons: list[int], now: datetime) -> dict:
    stamp = now.isoformat()
    items = [{**row, "status": "pending", "verdict": None, "first_seen_at": stamp, "updated_at": stamp}
             for row in rows]
    return {"title": f"shakedown {seasons[0]}–{seasons[-1]}", "season": seasons[0], "week": 0,
            "generated_at": stamp, "items": items}
```

- [ ] **Step 4: Implement the CLI**

In `src/cfbr/cli.py`, change the `anomalies` command: make `season` `Annotated[Optional[int], typer.Option()] = None`, and add options:

```python
    sample_patterns: Annotated[bool, typer.Option("--sample-patterns", help="Build the shakedown set from "
                                                  "--history output for --seasons (no API calls)")] = False,
    seasons: Annotated[Optional[str], typer.Option(help="Season range for --sample-patterns, e.g. 2022-2025")] = None,
    per_group: Annotated[int, typer.Option(help="Items per check/shape group for --sample-patterns")] = 3,
```

At the top of the body (after `cfg = _default_config(config)`), add:

```python
    if sample_patterns:
        from datetime import UTC, datetime

        from cfbr.anomalies.records import render_md, write_doc
        from cfbr.anomalies.shakedown import sample_patterns as sample, shakedown_doc
        from cfbr.backtest.harness import parse_seasons

        if not seasons:
            raise typer.BadParameter("--sample-patterns needs --seasons, e.g. 2022-2025")
        years = parse_seasons(seasons)
        out = cfg.paths.results_dir / "anomaly_shakedown" / "anomalies.json"
        if out.exists():
            typer.echo(f"{out} exists (it may hold paid verdicts); delete it to resample")
            raise typer.Exit(code=1)
        rows = []
        for year in years:
            hist = cfg.paths.results_dir / f"anomalies_history_{year}.json"
            if not hist.exists():
                items = arun.run_history(cfg, year)
                hist.parent.mkdir(parents=True, exist_ok=True)
                hist.write_text(json.dumps([{"id": a.id, **asdict(a)} for a in items], indent=2) + "\n")
            rows += json.loads(hist.read_text())
        picked = sample(rows, per_group, cfg.anomalies.a3_min_plays)
        doc = shakedown_doc(picked, years, datetime.now(UTC))
        write_doc(out, doc)
        out.with_suffix(".md").write_text(render_md(doc))
        counts = Counter(r["group"] for r in picked)
        typer.echo("\n".join(f"{g}: {n}" for g, n in sorted(counts.items())))
        typer.echo(f"wrote {out} ({len(picked)} items)")
        return
    if season is None:
        raise typer.BadParameter("--season is required")
```

(Move the existing `import json`, `Counter`, `asdict`, `arun` imports above this block if they are below it.)

Add the `investigate` command after `anomalies`:

```python
@app.command()
def investigate(
    season: Annotated[Optional[int], typer.Option(help="Default: the current season")] = None,
    week: Annotated[Optional[int], typer.Option(help="Default: the latest out/{season}/wNN")] = None,
    doc: Annotated[Optional[Path], typer.Option(help="Investigate this anomalies.json instead (e.g. the shakedown set)")] = None,
    force: Annotated[Optional[list[str]], typer.Option(help="Re-investigate this anomaly id (repeatable)")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="List what would be investigated; no API calls")] = False,
    config: ConfigOpt = None,
) -> None:
    """Investigate pending anomalies with Claude (advisory). Cron runs it hourly on Sundays; it exits
    quietly when there is nothing to do, the weekly lock is held, or no ANTHROPIC_API_KEY is configured."""
    from cfbr import config as cfbr_config
    from cfbr.alerts import send
    from cfbr.anomalies import run as arun
    from cfbr.anomalies.records import read_doc
    from cfbr.ingest.fetch import current_season

    cfg = _default_config(config)
    if doc is None:
        season = season or current_season()
        try:
            week = week or arun.latest_week(cfg, season)
        except FileNotFoundError:
            typer.echo(f"no weekly output for {season}; nothing to investigate")
            return
        doc = arun.week_dir(cfg, season, week) / "anomalies.json"
    force_ids = tuple(force or ())
    if dry_run:
        current = read_doc(doc)
        if current is None:
            typer.echo(f"{doc} does not exist")
            return
        todo, overflow = arun.plan_targets(current, force_ids, cfg.investigate.max_items)
        for row in todo:
            typer.echo(f"{row['id']}  {row['check']} {row['message']}")
        typer.echo(f"would investigate {len(todo)}; {len(overflow)} over the per-run limit")
        return
    key = cfbr_config.anthropic_api_key()
    if not key:
        typer.echo("no ANTHROPIC_API_KEY: investigation is off")
        return
    import anthropic

    try:
        summary = arun.investigate_doc(cfg, doc, anthropic.Anthropic(api_key=key), force_ids=force_ids)
    except Exception as e:
        send(cfg, "cfbr investigate crashed", f"{type(e).__name__}: {e}"[:300])
        raise
    alert = arun.investigate_alert(summary)
    if alert:
        title, body, priority = alert
        send(cfg, title, body, priority=priority)
    typer.echo(f"{summary['status']}: {len(summary['items'])} investigated, ${summary['cost_usd']:.2f}")
```

Check that `Path` is imported in `cli.py` (`grep -n "^from pathlib" src/cfbr/cli.py`); add `from pathlib import Path` if not.

- [ ] **Step 5: Run tests** — `uv run pytest tests/test_anomalies_investigate.py -q` → all pass; then `uv run pytest -q`.

- [ ] **Step 6: Exercise the dry paths on real local data (no API calls)**

Run: `uv run cfbr anomalies --sample-patterns --seasons 2022-2025`
Expected: group counts (each ≤ 3; total roughly 25–30) and `wrote …/results/anomaly_shakedown/anomalies.json`.

Run: `uv run cfbr investigate --doc results/anomaly_shakedown/anomalies.json --dry-run`
Expected: up to 12 lines `id  check message`, then `would investigate 12; N over the per-run limit`.

Run: `env -u ANTHROPIC_API_KEY uv run cfbr investigate --season 2026 --week 8`
Expected: `no ANTHROPIC_API_KEY: investigation is off` and no change to `out/2026/w08/anomalies.json`.

Do **not** run `cfbr investigate` with a key. Record the outputs in the report.

- [ ] **Step 7: Commit**

```bash
git add src/cfbr/anomalies/shakedown.py src/cfbr/cli.py tests/test_anomalies_investigate.py
git commit -m "feat(anomalies): cfbr investigate command and shakedown sampling"
```

---

### Task 8: Docs, full suite, PR

**Files:**
- Modify: `README.md`, `DECISIONS.md`, `CLAUDE.md`

- [ ] **Step 1: README**

In the command table, after the `cfbr anomalies` row, add:

```markdown
| `cfbr investigate` | Advisory: a Claude agent investigates each pending anomaly (data bug or real?) on a read-only snapshot of the store and writes a grounded verdict into `anomalies.{json,md}`; one ntfy summary per run. Off unless `ANTHROPIC_API_KEY` is set (env or `.keys`); spend logged to `data/anthropic_calls.jsonl`, stops at `[investigate].monthly_budget_usd`. `--dry-run` lists targets, `--doc` runs on another file (the shakedown set), `--force ID` re-runs one item |
```

Extend the `cfbr anomalies` row's text with: "`--sample-patterns --seasons 2022-2025` builds the shakedown set in `results/anomaly_shakedown/` (no API calls)".

After the weekly cron block, add a short section:

````markdown
### Anomaly investigation (Sunday afternoons)

`cfbr investigate` runs hourly from 10 AM to 6 PM Central on Sundays, after the 9 AM `weekly`. It does nothing unless the latest week's `anomalies.json` has pending items; it holds the weekly lock only while copying `data/cfb.duckdb` to `data/investigations/snapshot.duckdb`, so a long run never blocks the 3 PM `weekly`. Transcripts go to `data/investigations/{season}/wNN/`. More than `[investigate].max_items` pending items in one run are marked `skipped_volume` (a sign of a systemic break, not 12 separate problems).

```cron
20 * * * 0,1  [ "$(TZ=America/Chicago date +\%u)" = 7 ] && [ "$(TZ=America/Chicago date +\%H)" -ge 10 ] && [ "$(TZ=America/Chicago date +\%H)" -le 18 ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr investigate >> data/investigate.log 2>&1
```
````

- [ ] **Step 2: DECISIONS** — append:

```markdown
- Anomaly investigation agent (user request, 2026-11-08; spec docs/superpowers/specs/2026-11-08-anomaly-investigation-design.md): `cfbr investigate` runs claude-opus-5-5 at effort xhigh over each pending anomaly with two tools — read-only DuckDB SQL on a snapshot copy (read_only, enable_external_access=false, lock_configuration=true: no writes, file reads, ATTACH, COPY, INSTALL or SET) and a fixed whitelist of the week's rank outputs by name — and a strict submit_verdict. A verdict is kept only if every evidence item cites a tool result from the same session and quotes it verbatim (whitespace-normalized); otherwise it is recorded as inconclusive/ungrounded. Advisory only: verdicts go to anomalies.{json,md}, transcripts and one ntfy message, never to ratings or bets. Off without ANTHROPIC_API_KEY. Limits: 40 turns, 15 minutes, 12 items per run (the rest skipped_volume), 3 attempts on transient errors, local guard $50/month on top of the account's monthly credits (which expire and hard-stop) — Ops (anomalies)
- The anthropic SDK (2.x, which brings httpx2) is used for the Claude API instead of raw httpx: the repo rule "HTTP through httpx directly" exists to avoid the cfbd client, and the SDK's typed errors, streaming helper and beta parameters (task budgets, refusal fallbacks) would otherwise be hand-rolled — Ops (anomalies)
- read_output exposes rankings.csv, rankings_prev.csv, predictions.csv and accuracy.csv (the spec said "ratings.csv", which doesn't exist). anomalies.json itself is not exposed, so the agent can't anchor on other items' verdicts — Ops (anomalies)
```

- [ ] **Step 3: CLAUDE.md** — in "Pipeline and data flow", after the `cfbr weekly` paragraph, add one sentence: "`cfbr investigate` (cron, Sunday afternoons) runs an advisory Claude agent over pending anomalies; it is off without `ANTHROPIC_API_KEY`, and its output never feeds the model or bets." Also add `cfbr investigate` (with a key) to the list of commands not to run locally in "Where things run".

- [ ] **Step 4: Full suite** — `uv run pytest -q`; record the summary line.

- [ ] **Step 5: Commit docs**

```bash
git add README.md DECISIONS.md CLAUDE.md
git commit -m "docs: anomaly investigation agent in README, DECISIONS and CLAUDE.md"
```

- [ ] **Step 6: PR** — done by the controller after the final review (push, `gh pr create`). Not part of this task.
