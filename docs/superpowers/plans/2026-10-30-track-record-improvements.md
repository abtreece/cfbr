# Track-Record Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the paper-bet record the right comparator (the opening line), pre-register and gate one model fix (a pooled-FCS slope term), set up the portal-era roster-prior variant for a forward-looking test, and pre-register a CLV-based verdict so the ledger can conclude before 500 bets.

**Architecture:** Four independent PRs, one per improvement, each on its own branch/worktree and each deployable alone. Every model-side change follows the repo's pattern: a DECISIONS.md rule is committed *before* any code is fit or any result is looked at, the result is recorded in a second DECISIONS.md line that cites the rule's commit hash, and nothing enters the published model until its gate passes. The web server keeps reading files only.

**Tech Stack:** Python 3.12, polars, numpy, DuckDB, typer, FastAPI + Jinja, pytest, `uv run`.

## Global Constraints

- Python `>=3.12,<3.13`; run everything with `uv run ...`. No new dependencies.
- 2-space indentation does **not** apply here: the repo is standard 4-space Python. Match surrounding code.
- `config/tuned.toml` is frozen: never edit it. New knobs get defaults in `src/cfbr/config.py` (and a line in `config/default.toml` when they're user-facing) so `tuned.toml` keeps loading.
- Point-in-time invariant: a prediction for slot *w* uses only games that kicked off before that slot. None of these tasks add a new data path into predictions; Parts B and C only change how already-available features are combined. `tests/test_leakage.py` must still pass.
- Betting lines are never model inputs. Part A adds the opener to *evaluation* only.
- Holdout seasons 2022–2025 are never re-tuned on. Part A re-joins lines to the frozen holdout run and refuses if any headline metric changes; Part B looks at 2022–2025 only if its primary gate passes, and only as a sanity check.
- Never run `weekly`, `bets`, `odds-snapshot`, `backup` or `check` locally. `backtest`, `backtest-lines`, `calibration-gate`, `shadow`, `rank` and `web build` make no API calls and are fine locally.
- Deploying (`deploy/deploy.sh`) is a stop-and-ask step. Editing CI is not needed anywhere in this plan.
- Conventional-commit style with the repo's phase/feature prefix, e.g. `Backtest: ...`, `Paper bets: ...`, `Model: ...`, `Web app: ...`, `Docs: ...`. No Co-Authored-By lines.
- Every non-obvious choice gets a one-line DECISIONS.md entry: `decision — reason — Area`.
- Tests that use the `dataset` fixture skip without `data/cfb.duckdb`; the local checkout has it (synced from javelin 2026-10-30).

**Prerequisite for Part A, step A3:** the local `results/` holds `20261011T144444Z-tuned-tuning` and `20261011T145524Z-tuned-holdout` (rsynced from javelin). Production's copies are the ones that matter for the site; see Task A8.

---

## Part A — Opener comparator (PR 1, branch `feat/opener-comparator`)

What exists: `Dataset.lines` already carries `spread_open` (CFBD `spreadOpen`, books only, from 2021). `backtest/metrics.py:consensus_spread(lines, method, col)` already accepts `col="spread_open"`. The backtest joins only the closing consensus (`harness.py:40`). The /bets tier table compares the live record to `BACKTEST_TIER_COVER` (2015–21 vs close) and `bets.md` adds `HOLDOUT_TIER_COVER` (2022–25 vs close), both hardcoded in `bets/ledger.py:66-74`. Live bets are recorded Sunday against early lines, so the close is the wrong comparator.

### Task A1: Opening consensus helper + opener diagnostics in `metrics.py`

**Files:**
- Modify: `src/cfbr/backtest/metrics.py` (after `consensus_spread`, ~line 60; and `diagnostics`, lines 183–224)
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Produces: `opening_spread(lines: pl.DataFrame, method: str = "median") -> pl.DataFrame` with columns `game_id, spread_open, n_providers_open`.
- Produces: `diagnostics(preds)` gains keys `ats_by_edge_open` (same shape as `ats_by_edge`) and `opener` (dict, or `None`) when `preds` has a `spread_open` column.
- Produces: module constant `OPENER_MIN_EDGE = 3.0`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report.py`:

```python
def test_opener_diagnostics_from_synthetic_frame():
    import polars as pl
    from cfbr.backtest.metrics import diagnostics, opening_spread

    lines = pl.DataFrame({
        "game_id": [1, 1, 2, 3], "season": [2024] * 4, "week": [5] * 4, "season_type": ["regular"] * 4,
        "provider": ["Bovada", "consensus", "DraftKings", "Bovada"],
        "spread": [-7.0, -7.5, 3.0, -1.0], "spread_open": [-4.0, None, 3.5, None], "over_under": [50.0] * 4})
    op = opening_spread(lines)
    assert op.columns == ["game_id", "spread_open", "n_providers_open"]
    assert op.sort("game_id")["spread_open"].to_list() == [-4.0, 3.5]      # game 3 has no opener

    preds = pl.DataFrame({
        "game_id": [1, 2, 3], "season": [2024] * 3, "week": [5] * 3, "season_type": ["regular"] * 3,
        "margin": [10.0, -3.0, 2.0], "pred_margin": [9.0, -1.0, 0.0], "win_prob": [0.7, 0.45, 0.5],
        "prior_share_h": [0.4] * 3, "prior_share_a": [0.4] * 3,
        "spread": [-7.0, 3.0, -1.0], "spread_open": [-4.0, 3.5, None]})
    d = diagnostics(preds)
    assert "ats_by_edge_open" in d and "opener" in d
    o = d["opener"]
    assert o["n"] == 2 and o["seasons"] == [2024]
    # game 1: edge vs opener = 9 - 4 = +5 -> a bet; close moved from +4 to +7 toward the model: CLV +3
    # game 2: edge vs opener = -1 - (-3.5) = +2.5 -> no bet
    assert o["n_bets"] == 1 and o["clv_mean"] == 3.0 and o["beat_close"] == 1.0
    # move_slope = sum(edge*move)/sum(edge^2) over both games: (5*3 + 2.5*0.5) / (25 + 6.25)
    assert abs(o["move_slope"] - (15 + 1.25) / 31.25) < 1e-9
    buckets = {r["bucket"]: r for r in d["ats_by_edge_open"]}
    assert buckets["3-6"]["n"] == 1 and buckets["3-6"]["wins"] == 1
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_report.py::test_opener_diagnostics_from_synthetic_frame -v`
Expected: FAIL with `ImportError: cannot import name 'opening_spread'`.

- [ ] **Step 3: Implement**

In `src/cfbr/backtest/metrics.py`, directly after `consensus_spread` add:

```python
def opening_spread(lines: pl.DataFrame, method: str = "median") -> pl.DataFrame:
    """Home-perspective opening spread per game: ``consensus_spread`` applied to CFBD's
    ``spreadOpen``. Only sportsbooks carry an opener (the aggregators' column is null), so this
    is a books-only median. Columns: game_id, spread_open, n_providers_open."""
    return (consensus_spread(lines, method, "spread_open")
            .select("game_id", "spread_open", n_providers_open="n_providers"))
```

Below `DIAG_EDGE_BUCKETS` add:

```python
# Would-be bet threshold for the opener block: the paper-bet rule's Small tier (config
# [bets.tiers]); diagnostics take no config, so it is repeated here.
OPENER_MIN_EDGE = 3.0
```

Above `diagnostics` add two helpers, and replace the inline edge loop inside `diagnostics` with a call to the first:

```python
def _ats_by_edge(y: np.ndarray, line: np.ndarray, pred: np.ndarray) -> list[dict]:
    """Model-side cover rate against ``line`` by |pred − line| bucket, pushes excluded, Wilson CIs."""
    edge = pred - line
    result = np.sign(y - line)
    pick = np.sign(edge)
    ok = (result != 0) & (pick != 0)
    ae = np.abs(edge)
    rows = []
    for label, lo, hi in (("0-3", 0, 3), ("3-6", 3, 6), ("6-10", 6, 10), ("10+", 10, np.inf)):
        m = ok & (ae >= lo) & (ae < hi)
        n = int(m.sum())
        wins = int(np.sum(result[m] == pick[m]))
        ci = wilson_ci(wins, n)
        rows.append({"bucket": label, "n": n, "wins": wins, "win_rate": wins / n if n else math.nan,
                     "ci95_low": ci[0], "ci95_high": ci[1]})
    return rows


def _opener(d: pl.DataFrame) -> dict | None:
    """The model against the opening line on games with both an opener and a close: each line's
    MAE, the would-be bets (|pred − opener| ≥ OPENER_MIN_EDGE) with their closing-line value
    (points the close moved toward the bet side), the share that beat the close, and how far the
    close moves toward the model per point of disagreement (OLS of close − open on pred − open,
    no intercept)."""
    o = d.filter(pl.col("spread_open").is_not_null() & pl.col("spread").is_not_null())
    if o.height < 2:
        return None
    y = o["margin"].to_numpy()
    mo = -o["spread_open"].to_numpy()
    mc = -o["spread"].to_numpy()
    pred = o["pred_margin"].to_numpy()
    edge = pred - mo
    move = mc - mo
    bet = np.abs(edge) >= OPENER_MIN_EDGE
    clv = (move * np.sign(edge))[bet]
    ss = float((edge * edge).sum())
    return {"n": int(o.height), "seasons": sorted(int(s) for s in o["season"].unique()),
            "mae_open": float(np.mean(np.abs(y - mo))), "mae_close": float(np.mean(np.abs(y - mc))),
            "mae_model": float(np.mean(np.abs(y - pred))),
            "n_bets": int(bet.sum()),
            "clv_mean": float(clv.mean()) if clv.size else math.nan,
            "clv_sd": float(clv.std(ddof=1)) if clv.size > 1 else math.nan,
            "beat_close": float(np.mean(clv > 0)) if clv.size else math.nan,
            "push_close": float(np.mean(clv == 0)) if clv.size else math.nan,
            "move_slope": float((edge * move).sum() / ss) if ss > 0 else math.nan}
```

In `diagnostics`, replace everything from `edge = pred - line` through `res["ats_by_edge"] = rows` with:

```python
        res["ats_by_edge"] = _ats_by_edge(y, line, pred)
    if "spread_open" in d.columns:
        op = d.filter(pl.col("spread_open").is_not_null())
        if op.height >= 2:
            res["ats_by_edge_open"] = _ats_by_edge(op["margin"].to_numpy(), -op["spread_open"].to_numpy(),
                                                   op["pred_margin"].to_numpy())
        res["opener"] = _opener(d)
```

(The `res["ats_by_edge"]` line stays inside the existing `if lined.height >= 3:` block; the new `if "spread_open"` block is at the function's top indentation, before `return res`.)

- [ ] **Step 4: Run the test and the existing metrics tests**

Run: `uv run pytest tests/test_report.py -v`
Expected: all PASS, including the new test.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/backtest/metrics.py tests/test_report.py
git commit -m "Backtest: opening-line consensus and opener diagnostics (ATS by edge, CLV, move slope)"
```

### Task A2: Join the opener in the harness

**Files:**
- Modify: `src/cfbr/backtest/harness.py:13,40`
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Produces: `run_backtest` output frame gains `spread_open`, `n_providers_open`; `metrics["diagnostics"]` gains the Task A1 keys for any run with openers.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report.py`:

```python
def test_backtest_frame_carries_opener_columns(cfg, dataset):
    from cfbr.backtest.harness import run_backtest

    df, metrics = run_backtest(cfg, [2025], dataset)
    assert {"spread", "spread_open", "n_providers_open"} <= set(df.columns)
    assert df["spread_open"].is_not_null().sum() > 800        # 2025 has an opener on every lined game
    assert "ats_by_edge_open" in metrics["diagnostics"] and metrics["diagnostics"]["opener"]["seasons"] == [2025]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_report.py::test_backtest_frame_carries_opener_columns -v`
Expected: FAIL on the `spread_open` column assertion. (This test runs a one-season backtest, roughly a minute.)

- [ ] **Step 3: Implement**

In `src/cfbr/backtest/harness.py` change the import to `from cfbr.backtest.metrics import consensus_spread, opening_spread, report` and after line 40 (`df = df.join(consensus_spread(...), on="game_id", how="left")`) add:

```python
    df = df.join(opening_spread(ds.lines, cfg.backtest.spread_consensus), on="game_id", how="left")
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_report.py::test_backtest_frame_carries_opener_columns -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/backtest/harness.py tests/test_report.py
git commit -m "Backtest: join the opening consensus to every run's predictions"
```

### Task A3: `cfbr backtest-lines` — refresh a frozen run's line diagnostics without re-running the model

**Files:**
- Modify: `src/cfbr/backtest/harness.py` (append)
- Modify: `src/cfbr/cli.py` (new command after `holdout`, ~line 186)
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Produces: `refresh_lines(cfg: Config, run_dir: Path, lines: pl.DataFrame) -> dict` (the merged metrics). Raises `RuntimeError` if any of `overall`, `by_season`, `by_bucket`, `by_season_bucket` would change.

> **Divergence (2026-10-30, user-approved during execution):** the first version re-joined the *closing* consensus too, and both frozen runs refused: the current lines table reproduces a different close on about 8% of their games (holdout line MAE 12.02598 → 12.02701), so the raw cache was refreshed after 2026-10-11. `refresh_lines` now keeps the run's stored `spread`, `n_providers` and `source` and adds only `spread_open` / `n_providers_open` from the current table. Openers do not change after the fact, so the opener block's CLV uses the run's own close. The headline-metric refusal stays as a safety net. The second test became: the stored close survives a lines table whose close differs; plus a tamper test that still exercises the refusal.
- Produces: CLI `cfbr backtest-lines --run <run_id>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_report.py`:

```python
def _mini_run(tmp_path, lines_spread):
    """A fake results run: two completed games whose stored close came from ``lines_spread``."""
    import json
    import polars as pl
    from cfbr.backtest.metrics import report

    preds = pl.DataFrame({
        "game_id": [1, 2], "season": [2024, 2024], "week": [5, 5], "season_type": ["regular"] * 2,
        "slot": ["r05"] * 2, "completed": [True, True], "neutral_site": [False, False],
        "margin": [10.0, -3.0], "pred_margin": [9.0, -1.0], "win_prob": [0.7, 0.45],
        "prior_share_h": [0.4] * 2, "prior_share_a": [0.4] * 2,
        "spread": lines_spread, "n_providers": [1, 1], "source": ["books"] * 2})
    run = tmp_path / "20260101T000000Z-mini"
    run.mkdir()
    preds.write_parquet(run / "predictions.parquet")
    m = {"seasons": [2024], **report(preds), "calibration": {"2024": {"b_home": 3.0}}}
    (run / "metrics.json").write_text(json.dumps(m, default=float))
    return run


def _mini_lines(spread):
    import polars as pl
    return pl.DataFrame({"game_id": [1, 2], "season": [2024] * 2, "week": [5] * 2, "season_type": ["regular"] * 2,
                         "provider": ["Bovada"] * 2, "spread": spread, "spread_open": [-4.0, 3.5],
                         "over_under": [50.0] * 2})


def test_refresh_lines_adds_opener_and_keeps_headline(cfg, tmp_path):
    import json
    import polars as pl
    from cfbr.backtest.harness import refresh_lines

    run = _mini_run(tmp_path, [-7.0, 3.0])
    before = json.loads((run / "metrics.json").read_text())
    merged = refresh_lines(cfg, run, _mini_lines([-7.0, 3.0]))
    after = json.loads((run / "metrics.json").read_text())
    assert after["overall"] == before["overall"] and after["calibration"] == before["calibration"]
    assert after["seasons"] == [2024]
    assert merged["diagnostics"]["opener"]["n_bets"] == 1
    assert "spread_open" in pl.read_parquet(run / "predictions.parquet").columns


def test_refresh_lines_refuses_when_the_close_changed(cfg, tmp_path):
    import json
    import pytest
    from cfbr.backtest.harness import refresh_lines

    run = _mini_run(tmp_path, [-7.0, 3.0])
    before = (run / "metrics.json").read_text()
    with pytest.raises(RuntimeError, match="overall would change"):
        refresh_lines(cfg, run, _mini_lines([-6.0, 3.0]))
    assert (run / "metrics.json").read_text() == before        # nothing written
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest tests/test_report.py -k refresh_lines -v`
Expected: FAIL with `ImportError: cannot import name 'refresh_lines'`.

- [ ] **Step 3: Implement `refresh_lines`**

Append to `src/cfbr/backtest/harness.py`:

```python
LINE_COLUMNS = ["spread", "n_providers", "source", "spread_open", "n_providers_open"]
HEADLINE_KEYS = ("overall", "by_season", "by_bucket", "by_season_bucket")


def refresh_lines(cfg: Config, run_dir: Path, lines: pl.DataFrame) -> dict:
    """Re-join the lines table to an existing run's predictions and recompute the line-derived
    parts of its ``metrics.json``. The model is not re-run. Refuses if any headline metric would
    change, so a frozen run (the tuned holdout) only ever gains diagnostics such as the opener
    block; ``seasons`` and ``calibration`` are kept from the existing file."""
    preds = pl.read_parquet(run_dir / "predictions.parquet")
    preds = preds.drop([c for c in LINE_COLUMNS if c in preds.columns])
    preds = (preds.join(consensus_spread(lines, cfg.backtest.spread_consensus), on="game_id", how="left")
                  .join(opening_spread(lines, cfg.backtest.spread_consensus), on="game_id", how="left"))
    old = json.loads((run_dir / "metrics.json").read_text())
    new = report(preds.filter(pl.col("completed")))
    for key in HEADLINE_KEYS:
        if json.dumps(old.get(key), sort_keys=True, default=float) != json.dumps(new.get(key), sort_keys=True, default=float):
            raise RuntimeError(f"{key} would change: the lines table differs from the one this run was scored on")
    merged = {**old, **new}
    tmp = run_dir / "predictions.parquet.tmp"
    preds.write_parquet(tmp)
    tmp.replace(run_dir / "predictions.parquet")
    tmp = run_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(merged, indent=2, default=float))
    tmp.replace(run_dir / "metrics.json")
    return merged
```

- [ ] **Step 4: Add the CLI command**

In `src/cfbr/cli.py`, after the `holdout` command (its body ends around line 186), add:

```python
@app.command("backtest-lines")
def backtest_lines(
    run: Annotated[str, typer.Option(help="results/ run id, e.g. 20261011T145524Z-tuned-holdout")],
    config: ConfigOpt = None,
) -> None:
    """Re-join the lines table (closing and opening consensus) to an existing backtest run and
    recompute its line-derived diagnostics. The model is not re-run; refuses if a headline metric
    would change."""
    from cfbr.backtest.harness import refresh_lines

    cfg = _default_config(config)
    run_dir = cfg.paths.results_dir / run
    if not (run_dir / "predictions.parquet").exists():
        raise typer.BadParameter(f"no predictions.parquet under {run_dir}")
    metrics = refresh_lines(cfg, run_dir, Dataset.load(cfg).lines)
    op = metrics["diagnostics"].get("opener")
    if op:
        typer.echo(f"{run}: {op['n']} games with an opener ({op['seasons'][0]}–{op['seasons'][-1]}); "
                   f"{op['n_bets']} would-be bets, avg CLV {op['clv_mean']:+.2f} pts, "
                   f"beat close {100 * op['beat_close']:.1f}%, move slope {op['move_slope']:.3f}")
    else:
        typer.echo(f"{run}: no games with an opening line")
```

(`Dataset`, `Annotated`, `typer`, `ConfigOpt` and `_default_config` are already imported/defined in `cli.py`.)

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_report.py -k refresh_lines -v`
Expected: both PASS.

- [ ] **Step 6: Refresh the two frozen runs locally and record the numbers**

Run (each loads the dataset, ~30 s):
```bash
uv run cfbr backtest-lines --run 20261011T144444Z-tuned-tuning
uv run cfbr backtest-lines --run 20261011T145524Z-tuned-holdout
```
Expected: both print a line like `20261011T145524Z-tuned-holdout: 3462 games with an opener (2022–2025); ~1980 would-be bets, avg CLV +0.49 pts, beat close ~53.9%, move slope ~0.077`, and neither raises. If one raises `RuntimeError: overall would change`, stop: the local lines table no longer matches the run (the raw cache changed); report it rather than forcing.

Then read the holdout tier numbers:
```bash
uv run python -c "import json; d=json.load(open('results/20261011T145524Z-tuned-holdout/metrics.json'))['diagnostics']; print(d['ats_by_edge_open']); print(d['opener'])"
```
Write down `(wins, n)` for the `3-6`, `6-10` and `10+` buckets. They are expected to be close to `(534, 1039)`, `(345, 649)`, `(149, 294)` (the ad-hoc analysis of 2026-10-30); Task A4 uses the exact values printed here.

- [ ] **Step 7: Commit**

```bash
git add src/cfbr/backtest/harness.py src/cfbr/cli.py tests/test_report.py
git commit -m "Backtest: cfbr backtest-lines refreshes a run's line diagnostics without re-running the model"
```

### Task A4: Opener reference tier table in `bets/ledger.py` + markdown

**Files:**
- Modify: `src/cfbr/bets/ledger.py:66-74` (constants), `by_tier` (~419), `sample_size_note` (~459), `summary_markdown` tier block (~588-606)
- Test: `tests/test_bets.py` (append)

**Interfaces:**
- Produces: `OPENER_TIER_COVER: dict[tuple[float, float | None], tuple[int, int]]`; `by_tier(...)` rows gain key `"opener"` (same shape as `"backtest"`/`"holdout"`); `_cover_range(table: dict) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bets.py`:

```python
def test_opener_tier_reference_and_markdown():
    from cfbr.bets.ledger import OPENER_TIER_COVER, _cover_range

    t = {x["name"]: x for x in by_tier(read_ledger(Path("/nonexistent")), CFG)}
    assert all(x["opener"] is not None and 0.49 < x["opener"]["rate"] < 0.56 for x in t.values())
    md = summary_markdown(read_ledger(Path("/nonexistent")), CFG, 2026)
    assert "holdout 2022–25 vs close [95% CI]" in md and "holdout 2022–25 vs opener [95% CI]" in md
    assert "against the opening line" in md and "48.5–50.5%" not in md
    assert _cover_range(OPENER_TIER_COVER).endswith("%") and "–" in _cover_range(OPENER_TIER_COVER)


def test_opener_tier_constants_match_the_holdout_run(cfg):
    """Guards the hardcoded reference against the run it quotes; skips without results/."""
    import json
    from cfbr.bets.ledger import OPENER_TIER_COVER

    path = cfg.paths.results_dir / "20261011T145524Z-tuned-holdout" / "metrics.json"
    if not path.exists():
        pytest.skip("no holdout run locally")
    d = json.loads(path.read_text())["diagnostics"]
    if "ats_by_edge_open" not in d:
        pytest.skip("run not refreshed with cfbr backtest-lines")
    got = {r["bucket"]: (r["wins"], r["n"]) for r in d["ats_by_edge_open"]}
    assert OPENER_TIER_COVER[(3.0, 6.0)] == got["3-6"]
    assert OPENER_TIER_COVER[(6.0, 10.0)] == got["6-10"]
    assert OPENER_TIER_COVER[(10.0, None)] == got["10+"]
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest tests/test_bets.py -k opener -v`
Expected: FAIL with `ImportError: cannot import name 'OPENER_TIER_COVER'`.

- [ ] **Step 3: Implement**

In `src/cfbr/bets/ledger.py`, after `HOLDOUT_TIER_COVER` (line 74) add, substituting the exact `(wins, n)` values from Task A3 step 6:

```python
# The same, model side vs the OPENING consensus (CFBD spreadOpen, sportsbooks only), over the
# 2022-2025 holdout (results/20261011T145524Z-tuned-holdout, diagnostics.ats_by_edge_open after
# `cfbr backtest-lines`). Openers exist in CFBD from 2021, so there is no tuning-season reference.
# Live paper bets are recorded early in the week, so this is the closer comparator; it is still
# below break-even in every tier.
OPENER_TIER_COVER = {(3.0, 6.0): (534, 1039), (6.0, 10.0): (345, 649), (10.0, None): (149, 294)}


def _cover_range(table: dict) -> str:
    """'lo–hi%' of the tier cover rates in a reference table."""
    rates = [100 * w / n for w, n in table.values()]
    return f"{min(rates):.1f}–{max(rates):.1f}%"
```

In `by_tier`, extend the appended dict with `"opener": tier_reference(t, OPENER_TIER_COVER)`.

In `sample_size_note`, replace the sentence `"The 2022–2025 backtest (never tuned on) covered 48.5–50.5% in every edge tier."` with:

```python
            f"evidence by itself. The 2022–2025 backtest (never tuned on) covered {_cover_range(HOLDOUT_TIER_COVER)} "
            f"against the closing line and {_cover_range(OPENER_TIER_COVER)} against the opening line, by edge tier; "
            "bets here are recorded early in the week, so the opener is the closer comparator."
```

In `summary_markdown`'s "By edge tier" block, change the prose and header to:

```python
    lines += ["## By edge tier", "",
              "Every bet is the same flat stake, so the payout per bet is identical; tiers group bets by how far the "
              "model is from the market. Backtest = how often that tier covered in the walk-forward backtest (pushes "
              "excluded): 2015–2021 are the seasons the model was tuned on, 2022–2025 the holdout it never saw, shown "
              "against the closing line and against the opening line (CFBD openers, 2021 on). Bets here are recorded "
              "early in the week, so the opener is the closer comparator. No tier has cleared break-even with "
              "confidence against either line.", "",
              "| tier | bets | open | settled | W-L-P | win % [95% CI] | units | $ | ROI | backtest 2015–21 vs close [95% CI] "
              "| holdout 2022–25 vs close [95% CI] | holdout 2022–25 vs opener [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
```

and the row to `bt_txt = f"{cover_txt(t['backtest'])} | {cover_txt(t['holdout'])} | {cover_txt(t['opener'])}"`.

- [ ] **Step 4: Run the bets tests**

Run: `uv run pytest tests/test_bets.py -v`
Expected: all PASS (the existing `test_coin_roi_chance_and_holdout_tiers` still finds `"holdout 2022–25"`).

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/bets/ledger.py tests/test_bets.py
git commit -m "Paper bets: holdout cover vs the opening line as a tier reference in bets.md"
```

### Task A5: Web — /bets, /bets/guide and /lab show the opener

**Files:**
- Modify: `src/cfbr/web/templates/bets.html:90,140-157`
- Modify: `src/cfbr/web/templates/bets_guide.html:85-90`
- Modify: `src/cfbr/web/bets.py:575` (guide context)
- Modify: `src/cfbr/web/templates/lab.html:52-77`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `t.holdout`, `t.opener` from `by_tier`; `d.ats_by_edge_open`, `d.opener` from metrics.

- [ ] **Step 1: Write the failing tests**

In `tests/test_web.py`, extend the `METRICS["diagnostics"]` dict (lines ~105-112) with two keys:

```python
    "ats_by_edge_open": [{"bucket": "3-6", "n": 7, "wins": 4, "win_rate": 0.571, "ci95_low": 0.25, "ci95_high": 0.84}],
    "opener": {"n": 9, "seasons": [2024], "mae_open": 12.2, "mae_close": 12.0, "mae_model": 13.0, "n_bets": 7,
               "clv_mean": 0.49, "clv_sd": 2.0, "beat_close": 0.54, "push_close": 0.1, "move_slope": 0.077},
```

Append tests:

```python
def test_lab_shows_opener_diagnostics(client):
    html = " ".join(client.get("/lab").text.split())
    assert "vs opener" in html and "+0.49" in html and "0.077" in html


def test_bets_page_and_guide_show_opener_reference(bets_client):
    from cfbr.bets.ledger import OPENER_TIER_COVER

    small = OPENER_TIER_COVER[(3.0, 6.0)]
    rate = f"{100 * small[0] / small[1]:.1f}%"
    html = " ".join(bets_client.get("/bets").text.split())
    assert "vs opener" in html and rate in html
    guide = " ".join(bets_client.get("/bets/guide").text.split())
    assert "opening line" in guide and rate in guide
    api = bets_client.get("/api/bets").json()
    assert api["tiers"][0]["opener"]["n"] == small[1]
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `uv run pytest tests/test_web.py -k "opener" -v`
Expected: FAIL on the `"vs opener" in html` assertions.

- [ ] **Step 3: Implement the templates**

`bets.html` tier table header (line ~143): replace the single `<th class="num">Backtest<span class="ci"> [95% CI]</span></th>` with

```html
    <th class="num opt">Backtest vs close<span class="ci"> [95% CI]</span></th><th class="num">Backtest vs opener<span class="ci"> [95% CI]</span></th>
```

and the matching `<td>` (line ~151) with

```html
      <td class="num opt">{% if t.holdout %}{{ t.holdout.rate|pct }}<span class="ci"> [{{ t.holdout.ci95[0]|pct }}, {{ t.holdout.ci95[1]|pct }}]</span>{% else %}—{% endif %}</td>
      <td class="num">{% if t.opener %}{{ t.opener.rate|pct }}<span class="ci"> [{{ t.opener.ci95[0]|pct }}, {{ t.opener.ci95[1]|pct }}]</span>{% else %}—{% endif %}</td>
```

Replace the prose at `bets.html:90` ("none cleared break-even with confidence in the 2015–2021 backtest.") with: `none cleared break-even with confidence in the 2022–2025 holdout, against the closing line or the opening line. Bets are recorded early in the week, so the opener is the closer comparator.`

`bets_guide.html:85-90`: next to the "Holdout cover rate, 2022–2025" column add a column "Cover vs opening line, 2022–2025" rendering `tier.opener.rate|pct`, `tier.opener.ci95[0]|pct`, `tier.opener.ci95[1]|pct`, `tier.opener.n`, guarded by `{% if tier.opener %}`. In `src/cfbr/web/bets.py:575` change the comprehension to:

```python
tiers=[t | {"holdout": tier_reference(t, HOLDOUT_TIER_COVER), "opener": tier_reference(t, OPENER_TIER_COVER)} for t in tiers(cfg.bets)]
```

and add `OPENER_TIER_COVER` to the import on line 21.

`lab.html`, after the `{% if d.ats_by_edge %} ... {% endif %}` block, add:

```html
  {% if d.ats_by_edge_open %}<div class="table-wrap"><table class="data">
    <thead><tr><th>|Edge| vs opener</th><th class="num">Games</th><th class="num">ATS [95% CI]</th></tr></thead>
    <tbody>{% for r in d.ats_by_edge_open %}<tr><td>{{ r.bucket }}</td><td class="num">{{ r.n }}</td>
      <td class="num">{% if r.n %}{{ r.win_rate|pct(1) }} [{{ r.ci95_low|pct(1) }}, {{ r.ci95_high|pct(1) }}]{% else %}—{% endif %}</td></tr>{% endfor %}</tbody>
  </table></div>{% endif %}
  {% if d.opener %}<p class="note">Opener ({{ d.opener.seasons[0] }}–{{ d.opener.seasons[-1] }}, {{ d.opener.n }} games): MAE opener {{ d.opener.mae_open|num }}, close {{ d.opener.mae_close|num }}, model {{ d.opener.mae_model|num }}.
    {{ d.opener.n_bets }} would-be bets at ≥ 3 pts vs the opener: avg CLV {{ "%+.2f"|format(d.opener.clv_mean) }} pts, beat the close {{ d.opener.beat_close|pct(1) }}; the close moves {{ d.opener.move_slope|num(3) }} of the way toward the model per point of disagreement.</p>{% endif %}
```

- [ ] **Step 4: Run the web tests**

Run: `uv run pytest tests/test_web.py -v`
Expected: all PASS. If `test_bets_guide` asserts the holdout literals `"50.4%"`, `"49.0%"`, `"48.4%"`, they still hold (the holdout column is unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/web/templates/bets.html src/cfbr/web/templates/bets_guide.html src/cfbr/web/templates/lab.html src/cfbr/web/bets.py tests/test_web.py
git commit -m "Web app: show backtest cover vs the opening line on /bets, the guide and /lab"
```

### Task A6: DECISIONS.md entries + README command table

**Files:**
- Modify: `DECISIONS.md` (append two lines)
- Modify: `README.md` (command table: add `backtest-lines`)

- [ ] **Step 1: Append to DECISIONS.md**

```markdown
- Opening-line comparator (2026-10-30): the backtest joins CFBD's `spreadOpen` (sportsbooks only, same median rule; openers exist from 2021) and reports per run ATS by |edge| vs the opener and an opener block (would-be bets at ≥ 3 pts, their closing-line value, the share beating the close, and the OLS slope of close − open on model − open). `cfbr backtest-lines --run` adds these to an existing run without re-running the model and refuses if any headline metric would change; both frozen tuned runs were refreshed this way. On /bets the tier comparator is the 2022–2025 holdout vs the opener next to vs the close (the 2015–2021 vs-close reference stays in bets.md and /api/bets). Live bets are recorded early in the week, so the close was the wrong comparator — Bets
- Opening-line result (2026-10-30, holdout 2022–2025, read from the refreshed run): the model covers <X>% [<lo>, <hi>] vs the opener at ≥ 3 pts (n=<n>), avg CLV +<c> pts, beats the close <b>% of the time, and the close moves <s> of the way from the opener toward the model. Real but small: about a point of CLV is worth ~2.3% of cover probability, under the 2.4% the vig costs — Bets
```

Fill `<X>`, `<lo>`, `<hi>`, `<n>`, `<c>`, `<b>`, `<s>` from Task A3 step 6's printout (the `3-99` figure is the sum of the three bet tiers' wins and n; compute its Wilson CI with `from cfbr.backtest.metrics import wilson_ci`).

- [ ] **Step 2: README**

Add a row to the command table: `cfbr backtest-lines --run ID` — "re-join lines (close + opener) to an existing run's predictions; refuses if headline metrics change".

- [ ] **Step 3: Commit**

```bash
git add DECISIONS.md README.md
git commit -m "Docs: record the opening-line comparator and its holdout result"
```

### Task A7: Full test run, PR

- [ ] Run: `uv run pytest -q` — Expected: all pass (or skip where the dataset isn't needed).
- [ ] Open the PR (`gh pr create`) titled `feat: compare the paper-bet record to the opening line` with a body summarising A1–A6 and the holdout numbers. CI must pass before merge; squash-merge.

### Task A8: After merge — production refresh (stop and ask before each command)

The site reads `results/*/metrics.json` on javelin. After deploying (`deploy/deploy.sh`, a stop-and-ask step), the production runs must be refreshed too:

```bash
ssh javelin 'cd ~/cfb-ratings && ~/.local/bin/uv run cfbr backtest-lines --run 20261011T144444Z-tuned-tuning && ~/.local/bin/uv run cfbr backtest-lines --run 20261011T145524Z-tuned-holdout'
```

This makes no API calls and writes only under `results/`. `Store.runs()` reads each metrics.json once per process and `refresh_lines` rewrites files inside the run directory, so restart the service afterwards: `ssh javelin 'systemctl --user restart cfbr-web'` (check the unit name in deploy/deploy.sh). Then verify /lab contains "vs opener". `/bets` needs no data because its reference is the constant. Verify: `curl -s https://<site>/bets | grep -c "vs opener"` returns ≥ 1.

---

## Part B — Calibration v3: pooled-FCS slope term (PR 2, branch `feat/calibration-v3-fcs-slope`)

What exists: stage-2 calibration v2 (`calibrate.py:stage2_design`, four terms `pred, pred·ps, home·ps, fcs`) is fit under the hybrid scheme in `Engine.stage2` and applied in `Engine.game_margin` only when `cfg.model.calibration_v2` is true (it is false everywhere; v2 is a 2026 shadow with its own window). On FBS-vs-pooled-FCS games the frozen backtests show margin ≈ 10 + 0.71·pred from the FBS side in 2015–2021: the model is too extreme on those games, and v2's constant `fcs` term can only shift, not shrink. Ad-hoc LOSO on 2015–2021 (2026-10-30, not yet a gated result): a slope+intercept recalibration on FCS games gains +0.058 [+0.017, +0.098] alone and about +0.044 on top of a two-term prior-share slope. The gate below decides against the *real* v2.

**Sequencing rule:** Task B1 (the DECISIONS rule) is committed before any code. Tasks B2–B4 build the term and the gate command. Task B5 runs the gate once. Task B6 (shadow + weekly hook) is built **only if the gate passes**; otherwise B6 is skipped and B7 records the failure and keeps `backtest/calib_gate.py` as the record of the test (the `model/home_field.py` precedent).

### Task B1: Pre-register the rule

**Files:**
- Modify: `DECISIONS.md` (append one line)

- [ ] **Step 1: Append**

```markdown
- Calibration v3 (pooled-FCS slope) — evaluation rule fixed 2026-10-30, before building or fitting anything. Motivation, from the frozen tuned backtests' own predictions (2015–2021; the 2022–2025 FCS slope was seen once as a descriptive statistic, 0.70, never as a gain): on FBS-vs-pooled-FCS games the slope of margin on prediction is 0.71 with an FBS-side intercept of about +10, i.e. the model is too extreme on pooled-FCS sides, and v2's constant fcs term cannot fix a slope. Change: one extra stage-2 term, pred·|fcs| (the base prediction on games with exactly one pooled-FCS side, 0 elsewhere), on top of v2's four terms, all five fit jointly by the same OLS under the same hybrid scheme; nothing else changes. Primary test: 2015–2021 paired MAE gain (v2 − v3) ≥ +0.03 with 95% CI lower bound > 0 (the M4 threshold), computed by `cfbr calibration-gate` from one engine's base predictions. Only if that passes: 2022–2025 sanity (`--sanity`; gain ≥ 0, expanding window, reported not gated), then a 2026 shadow `calibration_v3_comparison` whose window is regular-season weeks ≥ 10 and the postseason (no week-10 game had kicked off when this line was committed), with the usual adoption rule (window gain v3 − v2 > 0, 95% CI lower bound > −0.10), decided once before 2027 week 1 together with v2: v3 can be adopted only if v2 is. The v2 shadow file and window are not touched. Reported, not gated: the fitted pred_fcs coefficient per season, FCS-game MAE and slope under base/v2/v3. If the primary fails, 2022–2025 is not looked at, no shadow is run, and `backtest/calib_gate.py` stays as the record — Model
```

- [ ] **Step 2: Commit alone (its hash is cited by the result line)**

```bash
git add DECISIONS.md
git commit -m "Model: pre-register the calibration v3 (pooled-FCS slope) gate"
git rev-parse --short HEAD
```

Write the short hash down for Task B7.

### Task B2: The fifth stage-2 term in `calibrate.py`

**Files:**
- Modify: `src/cfbr/model/calibrate.py:46-72`
- Test: `tests/test_report.py` (append; `test_calibration_v2_stage2_and_shadow_gain` at lines 141-169 is the pattern)

**Interfaces:**
- Produces: `STAGE2_TERMS_V3 = STAGE2_TERMS + ["pred_fcs"]`, `stage2_terms(fcs_slope: bool = False) -> list[str]`, `stage2_design(games, pred, fcs_slope: bool = False)`, `fit_stage2(games, pred, fcs_slope: bool = False)`.

- [ ] **Step 1: Write the failing test**

```python
def test_stage2_v3_adds_a_pred_times_fcs_column():
    import numpy as np
    import polars as pl
    from cfbr.model.calibrate import STAGE2_TERMS, STAGE2_TERMS_V3, fit_stage2, stage2_design, stage2_terms

    g = pl.DataFrame({"neutral_site": [False, False, True], "prior_share_h": [0.5, 0.2, 1.0],
                      "prior_share_a": [0.5, 0.4, 1.0], "home_fcs": [False, False, True], "away_fcs": [False, True, False],
                      "margin": [10.0, 30.0, -20.0]})
    pred = np.array([8.0, 20.0, -15.0])
    X2 = stage2_design(g, pred)
    X3 = stage2_design(g, pred, fcs_slope=True)
    assert X2.shape == (3, 4) and X3.shape == (3, 5)
    assert np.allclose(X3[:, :4], X2)
    assert X3[:, 4].tolist() == [0.0, 20.0, -15.0]          # pred on the two FCS games only
    assert stage2_terms() == STAGE2_TERMS and stage2_terms(True) == STAGE2_TERMS_V3 == STAGE2_TERMS + ["pred_fcs"]
    assert fit_stage2(g, pred, fcs_slope=True).shape == (5,)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_report.py::test_stage2_v3_adds_a_pred_times_fcs_column -v` → FAIL (`ImportError: STAGE2_TERMS_V3`).

- [ ] **Step 3: Implement** — replace the v2 block at the bottom of `calibrate.py` with:

```python
# -- calibration v2 (DECISIONS "M2+M3 calibration v2") and v3 (DECISIONS "Calibration v3") -------
STAGE2_TERMS = ["pred", "pred_ps", "home_ps", "fcs"]
STAGE2_TERMS_V3 = STAGE2_TERMS + ["pred_fcs"]


def stage2_terms(fcs_slope: bool = False) -> list[str]:
    return STAGE2_TERMS_V3 if fcs_slope else STAGE2_TERMS


def stage2_design(games: pl.DataFrame, pred: np.ndarray, fcs_slope: bool = False) -> np.ndarray:
    """``games`` needs neutral_site, prior_share_h/_a and home_fcs/away_fcs. With ``fcs_slope``
    (v3) a fifth column, the base prediction on games with exactly one pooled-FCS side and 0
    elsewhere, lets the fit shrink the model's margin on those games; v2's constant ``fcs``
    term can only shift it."""
    ps = ((games["prior_share_h"].fill_null(1.0) + games["prior_share_a"].fill_null(1.0)) / 2).to_numpy()
    home = (~games["neutral_site"]).cast(pl.Float64).to_numpy()
    fcs = (games["away_fcs"].cast(pl.Float64) - games["home_fcs"].cast(pl.Float64)).fill_null(0.0).to_numpy()
    cols = [pred, pred * ps, home * ps, fcs]
    if fcs_slope:
        cols.append(pred * np.abs(fcs))
    return np.column_stack(cols)


def fit_stage2(games: pl.DataFrame, pred: np.ndarray, fcs_slope: bool = False) -> np.ndarray:
    y = games["margin"].to_numpy() - pred
    coef, *_ = np.linalg.lstsq(stage2_design(games, pred, fcs_slope), y, rcond=None)
    return coef
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_report.py -v` → all PASS (the v2 test still passes: defaults unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/model/calibrate.py tests/test_report.py
git commit -m "Model: calibration v3 design adds a pred·|fcs| slope term to stage 2"
```

### Task B3: Config flag, engine cache per design, export

**Files:**
- Modify: `src/cfbr/config.py:48-54`
- Modify: `src/cfbr/model/engine.py:82,254-273`
- Modify: `src/cfbr/report/shadow.py:187-195` (`calibration_comparison`: pin v2)
- Modify: `src/cfbr/web/build.py:262-266`
- Test: `tests/test_model_real.py` (append)

**Interfaces:**
- Produces: `Model.calibration_v3: bool = False`; `Engine.stage2(season, fcs_slope: bool | None = None)` cached by `(season, fcs_slope)`; `Engine.game_margin` applies v3 when `calibration_v3`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_model_real.py`:

```python
def test_stage2_v3_coefficients_are_cached_per_design(cfg, dataset):
    from cfbr.backtest.tune import apply_params
    from cfbr.model.engine import Engine

    tuned = apply_params(cfg, {"alpha_ppa": 303.298, "alpha_sr": 90.7544, "half_life_weeks": 3.6978,
                               "fcs_pooling": "conference", "gt_enabled": False, "margin_cap": 42, "massey_alpha": 1.0})
    eng = Engine(dataset, tuned, last_completed=2025)
    v2 = eng.stage2(2019, fcs_slope=False)
    v3 = eng.stage2(2019, fcs_slope=True)
    assert v2.shape == (4,) and v3.shape == (5,)
    assert eng.stage2(2019) is v2                      # default follows cfg.model.calibration_v3 == False
    assert v3[4] < 0                                  # FCS games: the model is too extreme, so the slope term shrinks
```

(If `apply_params` does not accept all those keys, copy the exact call from `tests/test_model_real.py:21-37` and keep its parameter dict.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_model_real.py::test_stage2_v3_coefficients_are_cached_per_design -v` → FAIL (`TypeError: stage2() got an unexpected keyword argument 'fcs_slope'`).

- [ ] **Step 3: Implement**

`config.py`, in `class Model` after `calibration_v2`:

```python
    # Stage-2 with a fifth term, pred on FBS-vs-FCS games (DECISIONS "Calibration v3"). Implies the
    # v2 terms; shadow only in 2026, never published without passing its gate.
    calibration_v3: bool = False
```

`engine.py` line 82: `self._stage2: dict[tuple[int, bool], np.ndarray] = {}`. Replace `stage2` and `game_margin` (lines 254–273) with:

```python
    def stage2(self, season: int, fcs_slope: bool | None = None) -> np.ndarray:
        """Stage-2 coefficients for ``season`` under the training scheme. ``fcs_slope`` picks
        the v3 design (five terms); None means the config's calibration_v3."""
        if fcs_slope is None:
            fcs_slope = self.cfg.model.calibration_v3
        key = (season, fcs_slope)
        if key not in self._stage2:
            rows, preds = [], []
            for s in training_seasons(season, self.cfg, self.prior_available()):
                g, p = self.base_predictions(s)
                keep = (g["completed"] & g["margin"].is_not_null()).fill_null(False).to_numpy()
                rows.append(g.filter(pl.Series(keep)))
                preds.append(p[keep])
            self._stage2[key] = fit_stage2(pl.concat(rows, how="diagonal_relaxed"), np.concatenate(preds), fcs_slope)
        return self._stage2[key]

    def game_margin(self, season: int, g: pl.DataFrame) -> np.ndarray:
        """Calibrated margin for games ``g`` (home perspective); stage 2 applies when v2 or v3 is on."""
        pred = predict_margin(self.calibration(season), g)
        if self.cfg.model.calibration_v2 or self.cfg.model.calibration_v3:
            v3 = self.cfg.model.calibration_v3
            pred = pred + stage2_design(g, pred, v3) @ self.stage2(season, v3)
        return pred
```

`report/shadow.py:calibration_comparison` line `v2 = base + stage2_design(g, base) @ eng.stage2(season)` → `eng.stage2(season, fcs_slope=False)` (the v2 shadow must stay v2 whatever the config says).

`web/build.py:265-266`: import `stage2_terms` instead of `STAGE2_TERMS` and change the export to

```python
                        "stage2": (dict(zip(stage2_terms(eng.cfg.model.calibration_v3), (float(c) for c in eng.stage2(season))))
                                   if (eng.cfg.model.calibration_v2 or eng.cfg.model.calibration_v3) else None)},
```

(`matchup.html:76-78` applies only `pred`, `pred_ps`, `home_ps`; `fcs` and `pred_fcs` are 0 for the FBS-vs-FBS matchups the page prices, so no JS change.)

- [ ] **Step 4: Run** — `uv run pytest tests/test_model_real.py tests/test_report.py tests/test_leakage.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/config.py src/cfbr/model/engine.py src/cfbr/report/shadow.py src/cfbr/web/build.py tests/test_model_real.py
git commit -m "Model: calibration_v3 flag; engine caches stage-2 coefficients per design"
```

### Task B4: The gate module and `cfbr calibration-gate`

**Files:**
- Create: `src/cfbr/backtest/calib_gate.py`
- Modify: `src/cfbr/cli.py` (new command after `backtest-lines`)
- Test: `tests/test_calib_gate.py` (create)

**Interfaces:**
- Produces: `gate_frame(eng, seasons) -> pl.DataFrame` (columns `game_id, season, week, season_type, completed, neutral_site, home_fcs, away_fcs, margin, pred_base, pred_v2, pred_v3`), `paired_gain(err_a, err_b) -> dict(n, gain, ci95)`, `gate_report(df, eng, seasons) -> dict`, `gate_markdown(report) -> str`, `run_gate(cfg, ds, seasons, tag="calib-v3-gate") -> Path`.

- [ ] **Step 1: Write the failing tests** — `tests/test_calib_gate.py`:

```python
import numpy as np
import polars as pl

from cfbr.backtest.calib_gate import GATE_MIN_GAIN, gate_markdown, paired_gain


def test_paired_gain_direction_and_ci():
    a = np.array([10.0, 12.0, 8.0, 11.0])
    b = np.array([9.0, 11.0, 7.0, 10.0])        # b is one point better everywhere
    g = paired_gain(a, b)
    assert g["n"] == 4 and g["gain"] == 1.0 and g["ci95"] == (1.0, 1.0)


def test_gate_markdown_states_pass_or_fail():
    rep = {"seasons": [2015, 2021], "n_games": 100, "n_fcs_games": 10,
           "primary": {"n": 100, "gain": 0.05, "ci95": (0.01, 0.09), "threshold": GATE_MIN_GAIN, "passes": True},
           "base_vs_v2": {"n": 100, "gain": 0.04, "ci95": (0.0, 0.08)},
           "base_vs_v3": {"n": 100, "gain": 0.09, "ci95": (0.03, 0.15)},
           "mae": {"base": 13.4, "v2": 13.36, "v3": 13.31},
           "fcs_games": {k: {"mae": 14.0, "slope": s} for k, s in (("base", 0.71), ("v2", 0.75), ("v3", 0.98))},
           "pred_fcs_by_season": {"2015": -0.3, "2016": -0.28}}
    md = gate_markdown(rep)
    assert "**passes**" in md and "+0.0500 [+0.0100, +0.0900]" in md and "| v3 | 13.310 | 14.000 | 0.980 |" in md
    rep["primary"]["passes"] = False
    assert "**fails**" in gate_markdown(rep)
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_calib_gate.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Create `src/cfbr/backtest/calib_gate.py`**

```python
"""Calibration v3 gate (DECISIONS "Calibration v3"): does a pooled-FCS slope term improve on v2?

Paired by game, leave-one-season-out on the tuning seasons under the hybrid scheme, from one
engine's base predictions (stage 2 is refit per design; the ratings are shared). This module
stays as the record of the test whether or not it passes (cf. model/home_field.py)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.ingest.fetch import current_season
from cfbr.model.calibrate import stage2_design, stage2_terms
from cfbr.model.engine import Engine

GATE_MIN_GAIN = 0.03  # points of MAE; the M4 ceiling threshold (DECISIONS)


def gate_frame(eng: Engine, seasons: list[int]) -> pl.DataFrame:
    parts = []
    for s in seasons:
        g, base = eng.base_predictions(s)
        v2 = base + stage2_design(g, base) @ eng.stage2(s, fcs_slope=False)
        v3 = base + stage2_design(g, base, fcs_slope=True) @ eng.stage2(s, fcs_slope=True)
        parts.append(g.select("game_id", "season", "week", "season_type", "completed", "neutral_site",
                              "home_fcs", "away_fcs", "margin")
                      .with_columns(pred_base=pl.Series(base), pred_v2=pl.Series(v2), pred_v3=pl.Series(v3)))
    return pl.concat(parts).filter(pl.col("completed") & pl.col("margin").is_not_null())


def paired_gain(err_a: np.ndarray, err_b: np.ndarray) -> dict:
    """Mean of err_a − err_b (positive favours b) with a normal 95% CI."""
    d = err_a - err_b
    n = int(d.size)
    if n < 2:
        return {"n": n, "gain": math.nan, "ci95": (math.nan, math.nan)}
    m = float(d.mean())
    half = 1.96 * float(d.std(ddof=1)) / math.sqrt(n)
    return {"n": n, "gain": m, "ci95": (m - half, m + half)}


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.cov(x, y, ddof=0)[0, 1] / np.var(x)) if x.size >= 3 and np.var(x) > 0 else math.nan


def gate_report(df: pl.DataFrame, eng: Engine, seasons: list[int]) -> dict:
    y = df["margin"].to_numpy()
    err = {k: np.abs(y - df[f"pred_{k}"].to_numpy()) for k in ("base", "v2", "v3")}
    primary = paired_gain(err["v2"], err["v3"])
    fcs = (df["home_fcs"] | df["away_fcs"]).fill_null(False).to_numpy()
    fcs_games = {k: {"mae": float(err[k][fcs].mean()) if fcs.any() else math.nan,
                     "slope": _slope(df[f"pred_{k}"].to_numpy()[fcs], y[fcs])} for k in err}
    coefs = {str(s): dict(zip(stage2_terms(True), (float(c) for c in eng.stage2(s, fcs_slope=True)))) for s in seasons}
    return {"seasons": seasons, "n_games": int(df.height), "n_fcs_games": int(fcs.sum()),
            "primary": {**primary, "threshold": GATE_MIN_GAIN,
                        "passes": bool(primary["gain"] >= GATE_MIN_GAIN and primary["ci95"][0] > 0)},
            "base_vs_v2": paired_gain(err["base"], err["v2"]), "base_vs_v3": paired_gain(err["base"], err["v3"]),
            "mae": {k: float(v.mean()) for k, v in err.items()}, "fcs_games": fcs_games,
            "pred_fcs_by_season": {s: c["pred_fcs"] for s, c in coefs.items()}}


def gate_markdown(r: dict) -> str:
    p = r["primary"]

    def g(x: dict) -> str:
        return f"{x['gain']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"

    lines = [f"# Calibration v3 gate: pooled-FCS slope, {r['seasons'][0]}–{r['seasons'][-1]}", "",
             f"Paired MAE gain v2 → v3 over {r['n_games']:,} games: **{g(p)}**. Rule: gain ≥ +{p['threshold']:.2f} "
             f"and 95% CI lower bound > 0 → **{'passes' if p['passes'] else 'fails'}**.", "",
             "| model | MAE all | MAE FCS games | slope on FCS games |", "|---|---|---|---|"]
    for k in ("base", "v2", "v3"):
        lines.append(f"| {k} | {r['mae'][k]:.3f} | {r['fcs_games'][k]['mae']:.3f} | {r['fcs_games'][k]['slope']:.3f} |")
    lines += ["", f"Reference: base → v2 {g(r['base_vs_v2'])}; base → v3 {g(r['base_vs_v3'])}. "
                  f"FCS games: {r['n_fcs_games']:,}.", "",
              "pred_fcs coefficient by season: " + ", ".join(f"{s}: {c:+.3f}" for s, c in r["pred_fcs_by_season"].items())]
    return "\n".join(lines) + "\n"


def run_gate(cfg: Config, ds: Dataset, seasons: list[int], tag: str = "calib-v3-gate") -> Path:
    eng = Engine(ds, cfg, last_completed=current_season() - 1)
    df = gate_frame(eng, seasons)
    rep = gate_report(df, eng, seasons)
    out = cfg.paths.results_dir / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{tag}")
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "frame.parquet")
    (out / "gate.json").write_text(json.dumps(rep, indent=2, default=float))
    (out / "gate.md").write_text(gate_markdown(rep))
    return out
```

- [ ] **Step 4: CLI** — in `cli.py` after `backtest-lines`:

```python
@app.command("calibration-gate")
def calibration_gate(
    seasons: Annotated[str, typer.Option(help="e.g. 2015-2021 (tuning seasons unless --sanity)")] = "2015-2021",
    sanity: Annotated[bool, typer.Option("--sanity", help="Allow holdout seasons (only after the primary passed)")] = False,
    config: ConfigOpt = None,
) -> None:
    """Calibration v3 gate (DECISIONS "Calibration v3"): paired MAE gain of the pooled-FCS slope term
    over calibration v2 under the training scheme. Writes results/<ts>-calib-v3-gate/ (or -sanity)."""
    from cfbr.backtest.calib_gate import run_gate

    cfg = _default_config(config)
    season_list = parse_seasons(seasons)
    t0, t1 = cfg.seasons.tuning
    if not sanity and any(s < t0 or s > t1 for s in season_list):
        raise typer.BadParameter(f"the gate runs on tuning seasons {t0}–{t1}; pass --sanity for the holdout check")
    out = run_gate(cfg, Dataset.load(cfg), season_list, "calib-v3-sanity" if sanity else "calib-v3-gate")
    p = json.loads((out / "gate.json").read_text())["primary"]
    typer.echo(f"v2 → v3 paired MAE gain {p['gain']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] over {p['n']} games: "
               f"{'PASSES' if p['passes'] else 'FAILS'} (rule: ≥ +{p['threshold']:.2f}, CI low > 0). {out}")
```

(`json` and `parse_seasons` are already imported in `cli.py`; check the top of the file and add `import json` if not.)

- [ ] **Step 5: Run** — `uv run pytest tests/test_calib_gate.py -v` → PASS. Also `uv run cfbr calibration-gate --help` prints the option list.

- [ ] **Step 6: Commit**

```bash
git add src/cfbr/backtest/calib_gate.py src/cfbr/cli.py tests/test_calib_gate.py
git commit -m "Model: cfbr calibration-gate, the pre-registered v3 test on tuning seasons"
```

### Task B5: Run the gate once (tuning seasons), then the sanity check only if it passes

- [ ] **Step 1:** `uv run cfbr calibration-gate --seasons 2015-2021` (several minutes: it fits 2015–2021 ratings and snapshots once, stage 2 twice). Expected output ends with `PASSES` or `FAILS` and the results path. Record `gate.md` verbatim for Task B7.
- [ ] **Step 2 (only if PASSES):** `uv run cfbr calibration-gate --seasons 2022-2025 --sanity`. Record its gain. If it does **not** pass, do not run this and skip Task B6.

### Task B6: 2026 shadow + weekly hook (ONLY IF THE GATE PASSED)

**Files:**
- Modify: `src/cfbr/report/shadow.py` (append after the v2 shadow block)
- Modify: `src/cfbr/cli.py:418-427` (weekly) and `cfbr shadow` (~line 701)
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Produces: `V3_WINDOW_FIRST_WEEK = 10`, `in_window_v3() -> pl.Expr`, `calibration_v3_comparison(eng, season) -> pl.DataFrame` (columns of `calibration_comparison` with `pred_v2`, `pred_v3` instead of `pred_base`, `pred_v2`), `paired_mae_gain(df, a, b) -> dict`, `calibration_v3_markdown(df, season) -> str`, `run_calibration_v3_shadow(eng, season) -> Path` writing `out/{season}/calibration_v3_comparison.{csv,md}`.

- [ ] **Step 1: Write the failing test**

```python
def test_calibration_v3_shadow_window_and_gain():
    import polars as pl
    from datetime import UTC, datetime
    from cfbr.report.shadow import calibration_v3_markdown, in_window_v3, paired_mae_gain

    df = pl.DataFrame({
        "game_id": [1, 2, 3], "season": [2026] * 3, "week": [9, 10, 1], "season_type": ["regular", "regular", "postseason"],
        "kickoff_ts": [datetime(2026, 11, 1, tzinfo=UTC)] * 3, "completed": [True, True, True],
        "home_team": ["A"] * 3, "away_team": ["B"] * 3, "margin": [10.0, 3.0, -7.0],
        "pred_v2": [6.0, 0.0, -1.0], "pred_v3": [8.0, 1.0, -4.0]}).with_columns(in_window=in_window_v3())
    assert df["in_window"].to_list() == [False, True, True]          # week 9 is outside; week 10 and bowls inside
    g = paired_mae_gain(df.filter(pl.col("in_window")), "pred_v2", "pred_v3")
    assert g["n"] == 2 and g["gain"] == 2.0                           # errors 3→2 and 6→3
    md = calibration_v3_markdown(df, 2026)
    assert "weeks 10+" in md and "v3" in md and "adopt v3" in md
```

- [ ] **Step 2: Run to verify failure** → FAIL (`ImportError: in_window_v3`).

- [ ] **Step 3: Implement** — append to `shadow.py`:

```python
# -- Calibration v3 shadow (rule: DECISIONS.md "Calibration v3") ---------------------------
V3_WINDOW_FIRST_WEEK = 10  # rule committed 2026-10-30, before any week-10 kickoff


def in_window_v3() -> pl.Expr:
    return (pl.col("season") == WINDOW_SEASON) & (
        (pl.col("season_type") == "postseason")
        | ((pl.col("season_type") == "regular") & (pl.col("week") >= V3_WINDOW_FIRST_WEEK)))


def paired_mae_gain(df: pl.DataFrame, a: str, b: str) -> dict:
    """Mean of |margin − a| − |margin − b| over completed games (positive favours b), normal 95% CI."""
    d = df.filter(pl.col("completed") & pl.col("margin").is_not_null())
    n = d.height
    if n < 2:
        return {"n": n, "gain": math.nan, "ci95": (math.nan, math.nan), "mae_a": math.nan, "mae_b": math.nan}
    ea = (d["margin"] - d[a]).abs().to_numpy()
    eb = (d["margin"] - d[b]).abs().to_numpy()
    x = ea - eb
    half = 1.96 * float(x.std(ddof=1)) / math.sqrt(n)
    return {"n": n, "gain": float(x.mean()), "ci95": (float(x.mean() - half), float(x.mean() + half)),
            "mae_a": float(ea.mean()), "mae_b": float(eb.mean())}


def calibration_v3_comparison(eng: Engine, season: int) -> pl.DataFrame:
    from cfbr.model.calibrate import stage2_design

    g, base = eng.base_predictions(season)
    v2 = base + stage2_design(g, base) @ eng.stage2(season, fcs_slope=False)
    v3 = base + stage2_design(g, base, fcs_slope=True) @ eng.stage2(season, fcs_slope=True)
    return (g.select("game_id", "season", "week", "season_type", "kickoff_ts", "completed", "home_team",
                     "away_team", "margin")
            .with_columns(pred_v2=pl.Series(v2), pred_v3=pl.Series(v3), in_window=in_window_v3())
            .sort("kickoff_ts", "game_id"))


def calibration_v3_markdown(df: pl.DataFrame, season: int) -> str:
    w = paired_mae_gain(df.filter(pl.col("in_window")), "pred_v2", "pred_v3")
    o = paired_mae_gain(df.filter(~pl.col("in_window")), "pred_v2", "pred_v3")

    def row(label: str, g: dict) -> str:
        if g["n"] == 0:
            return f"| {label} | 0 | – | – | – |"
        return (f"| {label} | {g['n']} | {g['mae_a']:.2f} | {g['mae_b']:.2f} | "
                f"{g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")

    verdict = ("**Currently meets** the adoption rule" if w["n"] and w["gain"] > 0 and w["ci95"][0] > GAIN_CI_FLOOR
               else "**Does not currently meet** the adoption rule")
    return "\n".join([
        f"# {season} calibration shadow: v2 vs v3 (pooled-FCS slope)", "",
        "v3 adds one stage-2 term, the base prediction on FBS-vs-pooled-FCS games, to calibration v2. Nothing here "
        "is published. Gain = v2 error − v3 error (positive favors v3). Rule (DECISIONS.md): adopt v3 before 2027 "
        f"week 1 only if v2 is adopted and the window gain is > 0 with a 95% CI lower bound > {GAIN_CI_FLOOR:+.2f}.", "",
        f"Window so far: gain {w['gain']:+.3f} (95% CI lower bound {w['ci95'][0]:+.3f}). {verdict}. "
        "The decision is made once, before 2027 week 1." if w["n"] else "Window so far: no completed games.", "",
        "| games | n | v2 MAE | v3 MAE | gain [95% CI] |", "|---|---|---|---|---|",
        row(f"**window (weeks {V3_WINDOW_FIRST_WEEK}+ & postseason)**", w),
        row(f"weeks 1–{V3_WINDOW_FIRST_WEEK - 1} (outside the window)", o), ""])


def run_calibration_v3_shadow(eng: Engine, season: int) -> Path:
    df = calibration_v3_comparison(eng, season)
    out = eng.cfg.paths.out_dir / str(season)
    out.mkdir(parents=True, exist_ok=True)
    csv, md = out / "calibration_v3_comparison.csv", out / "calibration_v3_comparison.md"
    tmp = csv.with_suffix(".csv.tmp")
    df.write_csv(tmp)
    tmp.replace(csv)
    tmp = md.with_suffix(".md.tmp")
    tmp.write_text(calibration_v3_markdown(df, season))
    tmp.replace(md)
    w = paired_mae_gain(df.filter(pl.col("in_window")), "pred_v2", "pred_v3")
    log.info("calibration v3 shadow: %d window games, gain %s", w["n"],
             "n/a" if w["n"] < 2 else f"{w['gain']:+.3f} [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}]")
    return csv
```

(`GAIN_CI_FLOOR`, `WINDOW_SEASON`, `Engine`, `Path`, `math`, `log` already exist in `shadow.py`.) Check `eng.cfg.paths.out_dir` is how the v2 shadow finds `out/`; if `run_calibration_shadow` uses a different expression, copy it.

In `cli.py` weekly, after the calibration v2 shadow `try` block (line 427) add:

```python
    try:
        from cfbr.report.shadow import run_calibration_v3_shadow

        if eng.cfg.model.calibration_v3:
            log.info("calibration v3 is the published model; no v3 shadow")
        else:
            run_calibration_v3_shadow(eng, season)
    except Exception:
        log.exception("calibration v3 shadow comparison failed; continuing")
        shadow_failed = True
```

In the `shadow` command (line ~701), after `cal = run_calibration_shadow(...)` add `run_calibration_v3_shadow(Engine(ds, cfg, last_completed=last), season)` reusing the same `Engine` instance (assign it to a variable first so the two shadows share fits).

- [ ] **Step 4: Run** — `uv run pytest tests/test_report.py -v` → PASS. Then `uv run cfbr shadow --season 2026` writes `out/2026/calibration_v3_comparison.md`; open it and confirm the window table is present (zero or few games until week 10 is played).

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/report/shadow.py src/cfbr/cli.py tests/test_report.py
git commit -m "Model: calibration v3 shadow (weeks 10+ and postseason), run by weekly and cfbr shadow"
```

### Task B7: Record the result; README; PR

- [ ] Append the result line to `DECISIONS.md`, citing Task B1's hash, in the project's style. Pass case:

```markdown
- Calibration v3 gate result (2026-10-<dd>), judged by the rule fixed in <hash>: passes. 2015–2021 paired MAE gain (v2 − v3) +<g> [+<lo>, +<hi>] over <n> games; FCS-game slope base <s0> → v2 <s2> → v3 <s3>, FCS-game MAE <m2> → <m3>; pred_fcs by season <min>..<max>. 2022–2025 sanity +<g22> [<lo>, <hi>]. Shadow `calibration_v3_comparison` runs from this deploy; window weeks 10+ and postseason; decided with v2 before 2027 week 1 — Model
```

Fail case:

```markdown
- Calibration v3 gate result (2026-10-<dd>), judged by the rule fixed in <hash>: NOT adopted; 2022–2025 not looked at, no shadow. 2015–2021 paired MAE gain (v2 − v3) +<g> [<lo>, <hi>] over <n> games (primary fails: <reason>). FCS-game slope base <s0> → v2 <s2> → v3 <s3>. `backtest/calib_gate.py` and `calibration_v3` stay as the record of the test — Model
```

- [ ] README command table: add `cfbr calibration-gate [--seasons] [--sanity]`.
- [ ] `uv run pytest -q` → all pass. Commit `Docs: record the calibration v3 gate result`.
- [ ] PR titled `feat: calibration v3 gate (pooled-FCS slope) and shadow` (or `test: ...` if it failed and only the record ships). Squash-merge. Deploy is stop-and-ask; if the gate passed, deploy **before the first week-10 kickoff (Thu 2026-11-05)** so the shadow's first write precedes its window (the predictions are point-in-time regardless, but the record is cleaner).

---

## Part C — Roster prior, portal-era variant (PR 3, branch `feat/roster-prior-portal-era`)

What exists: the roster prior (`prior.features = "roster"`) was NOT adopted (DECISIONS line 72): primary 2026 +0.34 ± 0.22 passed, but the secondary (2015–2021 weeks 1–3) was −0.07 ± 0.06. Line 72 says the pattern "suggests the features help in the transfer-portal era and not before it. Any portal-era variant must be judged by a new rule fixed in advance, on data not yet seen." Line 73 fixes the current 2026 roster shadow window (weeks 7+) and says "No other variants enter this window." So the variant gets its own file and a window that starts after everything seen so far.

**Variant:** `prior.roster_min_season = 2021`: the roster regression is trained only on seasons ≥ 2021 and used only for target seasons ≥ 2021; earlier targets are v1 exactly. Under the hybrid scheme this leaves 2015–2021 unchanged by construction (for target 2021 the LOSO pool has no 2021 rows, so it falls back to v1).

### Task C1: Pre-register the rule

- [ ] Append to `DECISIONS.md` and commit alone (record the hash):

```markdown
- Roster prior, portal-era variant — evaluation rule fixed 2026-10-30, before building. Variant: `prior.roster_min_season = 2021` with `prior.features = "roster"`: the roster regression is trained on seasons ≥ 2021 and used for target seasons ≥ 2021 only; earlier targets are v1 exactly, so 2015–2021 is unchanged by construction. Why 2021: the one-time transfer rule and NIL start then, and the 2026-10-11 result showed the roster features helping only from 2024 on. Data already seen and therefore not usable as a gate: 2015–2021 (tuning), 2022–2025 (viewed as the plain roster prior's sanity, +0.08), 2026 weeks 1–6 (built the plain roster rule) and 2026 weeks 7+ (the plain roster shadow window; no other variant enters it). Window for this variant: the 2026 postseason plus all of 2027 (regular season and postseason), written to `out/{season}/prior_portal_comparison.{csv,md}` from the same point-in-time predictions as the prior shadow, with three columns: the model published that season, the plain roster prior, and the portal variant. Primary: paired MAE gain over the window of portal vs the published model, > 0 with 95% CI lower bound > −0.10; the variant also has to beat the plain roster prior over the window (gain > 0). Decided once, before 2028 week 1. Reported, not gated: 2022–2025 expanding-window gain (designed after seeing related numbers, so it is context only) and weeks 1–4 vs 5+ splits. Interaction with the plain roster decision (before 2027 week 1): whichever prior is published in 2027 is the "published" column — Roster
```

```bash
git add DECISIONS.md && git commit -m "Roster: pre-register the portal-era roster-prior variant" && git rev-parse --short HEAD
```

### Task C2: `roster_min_season` in config, prior fit and engine

**Files:**
- Modify: `src/cfbr/config.py:62-66`, `config/default.toml:48-52`
- Modify: `src/cfbr/model/prior.py:80-100`
- Modify: `src/cfbr/model/engine.py:156-171`
- Test: `tests/test_roster.py` (append), `tests/test_model_real.py` (append)

**Interfaces:**
- Produces: `Prior.roster_min_season: int = 0`; `fit_prior_models(train, mode="v1", roster_min_season: int = 0)`; `Engine.priors(season)` uses v1 for `season < roster_min_season` when mode is roster.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_roster.py`:

```python
def test_roster_regression_trains_only_on_seasons_at_or_after_the_cutoff():
    import numpy as np
    import polars as pl
    from sklearn.linear_model import LinearRegression
    from cfbr.model.prior import COMPONENTS, OFFENSE_RETURNING, fit_prior_models
    from cfbr.model.roster import DEF_FEATURES, OFF_FEATURES

    rng = np.random.default_rng(0)
    n = 400
    seasons = np.repeat([2017, 2019, 2021, 2023], n // 4)
    cols = {"season": seasons, "team": [f"T{i}" for i in range(n)], "new_program": [False] * n,
            "talent": rng.normal(size=n), "recruit4": rng.normal(size=n)}
    for c in COMPONENTS:
        cols[f"prev_{c}"] = rng.normal(size=n)
    for f in OFFENSE_RETURNING + OFF_FEATURES + DEF_FEATURES:
        cols[f] = rng.normal(size=n)
    # the target depends on an offense roster feature only in the portal era, so the two fits differ
    cols["o_ppa"] = cols["prev_o_ppa"] + np.where(seasons >= 2021, 2.0, 0.0) * cols[OFF_FEATURES[0]]
    for c in ("d_ppa", "o_sr", "d_sr"):
        cols[c] = cols[f"prev_{c}"]
    train = pl.DataFrame(cols)
    all_rows = fit_prior_models(train, "roster")["o_ppa"][0][1]
    portal = fit_prior_models(train, "roster", roster_min_season=2021)["o_ppa"][0][1]
    feats = fit_prior_models(train, "roster", roster_min_season=2021)["o_ppa"][0][0]
    sub = train.filter(pl.col("season") >= 2021)
    manual = LinearRegression().fit(sub.select(feats).to_numpy(), sub["o_ppa"].to_numpy())
    assert np.allclose(portal.coef_, manual.coef_)
    assert not np.allclose(portal.coef_, all_rows.coef_)
```

Append to `tests/test_model_real.py`:

```python
def test_portal_variant_is_v1_before_the_cutoff(cfg, dataset):
    import polars as pl
    from cfbr.model.engine import Engine

    if dataset.roster is None or dataset.player_ppa is None or dataset.player_defense is None:
        pytest.skip("roster tables not in the local store")
    portal = cfg.model_copy(deep=True)
    portal.prior.features = "roster"
    portal.prior.roster_min_season = 2021
    a = Engine(dataset, cfg, last_completed=2025).priors(2019).sort("team")
    b = Engine(dataset, portal, last_completed=2025).priors(2019).sort("team")
    assert a.equals(b)
    c = Engine(dataset, portal, last_completed=2025).priors(2025).sort("team")
    assert not a.select("team").join(c, on="team").is_empty()      # it still produces priors after the cutoff
```

(`import pytest` at the top of `test_model_real.py` if missing.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_roster.py tests/test_model_real.py -k "portal or cutoff" -v` → FAIL (`TypeError: unexpected keyword 'roster_min_season'` / `ValidationError`).

- [ ] **Step 3: Implement**

`config.py`, `class Prior`:

```python
    # Roster regression trained on, and used for, target seasons >= this (0 = every season with
    # roster data). DECISIONS "Roster prior, portal-era variant": shadow only.
    roster_min_season: int = 0
```

`config/default.toml` under `[prior]`: `roster_min_season = 0`.

`prior.py:fit_prior_models`:

```python
def fit_prior_models(train: pl.DataFrame, mode: str = "v1", roster_min_season: int = 0) -> PriorModels:
    """``train`` = stacked season_features rows joined with targets (``o_ppa`` ...).

    Returns, per component, models in priority order. In roster mode the first
    model adds the roster features (fit on rows that have them, from season
    ``roster_min_season`` on) and the v1 model is the fallback for rows that don't
    (e.g. no defensive player stats before 2016).
    """
    rows = train.filter(~pl.col("new_program"))
    models: PriorModels = {}
    for comp in COMPONENTS:
        models[comp] = []
        modes = ["roster", "v1"] if mode == "roster" else ["v1"]
        for m in modes:
            feats = _features_for(comp, m)
            if any(f not in rows.columns for f in feats):
                continue
            pool = rows.filter(pl.col("season") >= roster_min_season) if m == "roster" else rows
            r = pool.drop_nulls([comp, *feats])
            if r.height < 2 * len(feats) + 10:
                continue
            models[comp].append((feats, LinearRegression().fit(r.select(feats).to_numpy(), r[comp].to_numpy())))
    return models
```

`engine.py:priors`, replace `mode = self.cfg.prior.features` with:

```python
        mode = self.cfg.prior.features
        if mode == "roster" and season < self.cfg.prior.roster_min_season:
            mode = "v1"  # portal-era variant: v1 before the cutoff season
```

and the fit call with `models = fit_prior_models(pl.concat(rows, how="diagonal_relaxed"), mode, self.cfg.prior.roster_min_season)`.

- [ ] **Step 4: Run** — `uv run pytest tests/test_roster.py tests/test_model_real.py tests/test_leakage.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/config.py config/default.toml src/cfbr/model/prior.py src/cfbr/model/engine.py tests/test_roster.py tests/test_model_real.py
git commit -m "Roster: prior.roster_min_season trains and applies the roster regression from a cutoff season"
```

### Task C3: The portal shadow file

**Files:**
- Modify: `src/cfbr/report/shadow.py` (append after `run_shadow`)
- Modify: `src/cfbr/cli.py` weekly (after `run_shadow`, lines 403-410) and `cfbr shadow` (~701)
- Test: `tests/test_report.py` (append)

**Interfaces:**
- Produces: `PORTAL_MIN_SEASON = 2021`, `in_window_portal() -> pl.Expr`, `portal_comparison(cfg, ds, season, last_completed, published: pl.DataFrame) -> pl.DataFrame` (columns of `comparison()` renamed `pred_published`, `pred_roster`, plus `pred_portal`, `in_window`), `portal_markdown(df, season) -> str`, `run_portal_shadow(cfg, ds, season, last_completed) -> Path` which reads `out/{season}/prior_comparison.csv` (written by `run_shadow` moments earlier) and writes `out/{season}/prior_portal_comparison.{csv,md}`.

- [ ] **Step 1: Write the failing test**

```python
def test_portal_shadow_window_and_markdown():
    import polars as pl
    from datetime import UTC, datetime
    from cfbr.report.shadow import in_window_portal, paired_mae_gain, portal_markdown

    df = pl.DataFrame({
        "game_id": [1, 2, 3], "season": [2026, 2026, 2027], "week": [9, 1, 1], "season_type": ["regular", "postseason", "regular"],
        "kickoff_ts": [datetime(2026, 11, 1, tzinfo=UTC)] * 3, "completed": [True] * 3, "home_team": ["A"] * 3,
        "away_team": ["B"] * 3, "margin": [10.0, 3.0, -7.0],
        "pred_published": [6.0, 0.0, -1.0], "pred_roster": [7.0, 1.0, -2.0], "pred_portal": [8.0, 2.0, -4.0]}
    ).with_columns(in_window=in_window_portal())
    assert df["in_window"].to_list() == [False, True, True]
    assert paired_mae_gain(df.filter(pl.col("in_window")), "pred_published", "pred_portal")["gain"] == 2.5
    md = portal_markdown(df, 2026)
    assert "portal" in md and "published" in md and "before 2028 week 1" in md
```

(`paired_mae_gain` comes from Part B. If Part C is executed before Part B merges, copy that function into this PR; the merge will dedupe.)

- [ ] **Step 2: Run to verify failure** → FAIL (`ImportError: in_window_portal`).

- [ ] **Step 3: Implement** — append to `shadow.py`:

```python
# -- Portal-era roster prior shadow (rule: DECISIONS.md "Roster prior, portal-era variant") --------
PORTAL_MIN_SEASON = 2021


def in_window_portal() -> pl.Expr:
    """2026 postseason and every 2027 game: nothing earlier is unseen."""
    return (((pl.col("season") == 2026) & (pl.col("season_type") == "postseason"))
            | (pl.col("season") >= 2027))


def portal_comparison(cfg: Config, ds: Dataset, season: int, last_completed: int, published: pl.DataFrame) -> pl.DataFrame:
    """``published`` is ``comparison()``'s frame (pred_v1, pred_roster) for the same season and
    as-of, so only the portal engine is built here."""
    portal_cfg = cfg.model_copy(deep=True)
    portal_cfg.prior.features = "roster"
    portal_cfg.prior.roster_min_season = PORTAL_MIN_SEASON
    portal = Engine(ds, portal_cfg, last_completed=last_completed).predict(season)
    return (published.rename({"pred_v1": "pred_published"}).drop("in_window")
            .join(portal.select("game_id", pred_portal="pred_margin"), on="game_id", how="left")
            .with_columns(in_window=in_window_portal())
            .sort("kickoff_ts", "game_id"))


def portal_markdown(df: pl.DataFrame, season: int) -> str:
    w = df.filter(pl.col("in_window"))
    vs_pub = paired_mae_gain(w, "pred_published", "pred_portal")
    vs_roster = paired_mae_gain(w, "pred_roster", "pred_portal")
    early = paired_mae_gain(w.filter((pl.col("season_type") == "regular") & (pl.col("week") <= 4)), "pred_published", "pred_portal")
    late = paired_mae_gain(w.filter(~((pl.col("season_type") == "regular") & (pl.col("week") <= 4))), "pred_published", "pred_portal")

    def row(label: str, g: dict) -> str:
        if g["n"] == 0:
            return f"| {label} | 0 | – | – | – |"
        return (f"| {label} | {g['n']} | {g['mae_a']:.2f} | {g['mae_b']:.2f} | "
                f"{g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}] |")

    meets = (vs_pub["n"] > 1 and vs_pub["gain"] > 0 and vs_pub["ci95"][0] > GAIN_CI_FLOOR and vs_roster["gain"] > 0)
    return "\n".join([
        f"# {season} prior shadow: portal-era roster variant", "",
        f"Roster regression trained on and used from {PORTAL_MIN_SEASON} (v1 before). Point-in-time predictions; nothing "
        "here is published. Gain = published error − portal error (positive favors portal). Rule (DECISIONS.md): adopt "
        f"for 2028 if the window gain vs the published model is > 0 with a 95% CI lower bound > {GAIN_CI_FLOOR:+.2f} and "
        "the variant also beats the plain roster prior over the window. Window: 2026 postseason + all of 2027. "
        "Decided once, before 2028 week 1.", "",
        (f"Window so far: {'**currently meets**' if meets else '**does not currently meet**'} the rule." if vs_pub["n"]
         else "Window so far: no completed games."), "",
        "| comparison (window) | n | MAE a | MAE b | gain [95% CI] |", "|---|---|---|---|---|",
        row("published → portal", vs_pub), row("plain roster → portal", vs_roster),
        row("published → portal, weeks 1–4", early), row("published → portal, week 5+ and bowls", late), ""])


def run_portal_shadow(cfg: Config, ds: Dataset, season: int, last_completed: int) -> Path:
    out = cfg.paths.out_dir / str(season)
    src = out / "prior_comparison.csv"
    if not src.exists():
        raise RuntimeError(f"{src} missing; run_shadow writes it first")
    published = pl.read_csv(src, try_parse_dates=True)
    df = portal_comparison(cfg, ds, season, last_completed, published)
    csv, md = out / "prior_portal_comparison.csv", out / "prior_portal_comparison.md"
    tmp = csv.with_suffix(".csv.tmp")
    df.write_csv(tmp)
    tmp.replace(csv)
    tmp = md.with_suffix(".md.tmp")
    tmp.write_text(portal_markdown(df, season))
    tmp.replace(md)
    w = paired_mae_gain(df.filter(pl.col("in_window")), "pred_published", "pred_portal")
    log.info("portal prior shadow: %d window games, gain %s", w["n"],
             "n/a" if w["n"] < 2 else f"{w['gain']:+.3f} [{w['ci95'][0]:+.3f}, {w['ci95'][1]:+.3f}]")
    return csv
```

(`Config`, `Dataset`, `Engine` are already imported in `shadow.py`; `kickoff_ts` round-trips through CSV with `try_parse_dates=True`. If `comparison()`'s CSV has a column the join renames differently, adapt the `rename` to match `comparison()`'s actual output names `pred_v1`/`pred_roster`.)

`cli.py` weekly, inside the existing prior-shadow `try` after `run_shadow(...)`:

```python
        from cfbr.report.shadow import run_portal_shadow

        run_portal_shadow(cfg, ds, season, last_completed=season - 1)
```

and in the `shadow` command after `path = run_shadow(...)`: `run_portal_shadow(cfg, ds, season, last_completed=last)`.

- [ ] **Step 4: Run** — `uv run pytest tests/test_report.py -v` → PASS. `uv run cfbr shadow --season 2026` → `out/2026/prior_portal_comparison.md` exists and says "no completed games" in the window (bowls haven't been played). This builds three engines; expect a few minutes.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/report/shadow.py src/cfbr/cli.py tests/test_report.py
git commit -m "Roster: portal-era prior shadow (2026 postseason + 2027), run by weekly and cfbr shadow"
```

### Task C4: Context number, docs, PR

- [ ] Reported, not gated: run `uv run cfbr backtest --seasons 2022-2025 --config config/tuned.toml --tag v1-ctx` and a second run with a temporary config copy (`cp config/tuned.toml /tmp/portal.toml`, append `features = "roster"` and `roster_min_season = 2021` under `[prior]`) tagged `portal-ctx`; compute the paired MAE gain between the two `predictions.parquet` files with `paired_mae_gain` after joining on `game_id`. This is the "designed after seeing related numbers" context figure the rule allows as a report. Delete `/tmp/portal.toml` afterwards; do not commit any config copy.
- [ ] Append a result line to `DECISIONS.md` citing Task C1's hash: the shadow is live from this deploy, the 2022–2025 context gain, and that the first window games are the 2026 bowls.
- [ ] README: mention `prior_portal_comparison.*` next to the other shadow outputs; config table gets `prior.roster_min_season`.
- [ ] `uv run pytest -q` → all pass. PR titled `feat: portal-era roster-prior shadow (2026 postseason + 2027)`. Squash-merge; deploy is stop-and-ask.

---

## Part D — CLV verdict for 2027 (PR 4, branch `feat/clv-verdict`)

What exists: the ledger records `closing_line`, `closing_source` ("snapshot" = pre-kickoff multi-book consensus, "cfbd" = post-game fallback), `odds` (−110 flat) and the `PricingTable` (`bets/pricing.py`) with `expected_value(side, number, odds, center)` = units won per unit risked when `center` is the true line. `summarize()` reports avg CLV in points and cents. The pre-set 500-settled-bet minimum applies to the win-rate record. The decision log already acknowledges CLV is the only metric the sample can answer. The verdict below is the pre-registered way to turn it into a decision.

**Metric:** closing-implied ROI per bet = `table.expected_value(side, line, odds, closing_line)` for settled bets with a snapshot close. A bettor who always took the close itself earns about −4.5% (the vig); a positive mean means the model's recorded lines beat the close by more than the vig.

### Task D1: Pre-register the rule

- [ ] Append to `DECISIONS.md` and commit alone:

```markdown
- Paper-bet verdict by closing-line value — rule fixed 2026-10-30, after 57 bets had settled (24-33, avg CLV +0.90 pts; the closing-implied ROI below had not been computed) and before any bet is excluded: the verdict uses every settled bet with a pre-kickoff snapshot close, the 57 included. Metric: closing-implied ROI = units won per unit risked at the recorded number and price if the snapshot close were the true line (`PricingTable.expected_value`, key numbers included); a bettor taking the close itself earns about −4.5%, the vig. Verdict, recomputed every weekly run and shown on /bets and in bets.md: at ≥ 300 such bets, "edge over the market" if the mean's 95% CI lower bound is > 0; "no edge" if its upper bound is < 0; otherwise undecided and it continues. 300 because with a per-bet SD near 0.05 the CI half-width is about ±0.6% ROI, enough to separate 0 from the vig. Decision point: the first weekly run after the 2027 regular season at which n ≥ 300 (else it carries to 2028). The win-rate record and the 500-settled-bet minimum stay as the secondary record; the bet rule itself (every ≥ 3-pt edge) does not change before the verdict — Bets
```

```bash
git add DECISIONS.md && git commit -m "Paper bets: pre-register the closing-line-value verdict" && git rev-parse --short HEAD
```

### Task D2: `bets/verdict.py` + config

**Files:**
- Create: `src/cfbr/bets/verdict.py`
- Modify: `src/cfbr/config.py` `class Bets` (after `min_sample`, ~line 88), `config/default.toml` `[bets]`
- Test: `tests/test_verdict.py` (create)

**Interfaces:**
- Produces: `Bets.verdict_min_bets: int = 300`; `closing_implied_roi(ledger, table) -> pl.Series` (nulls where not applicable); `verdict(ledger, cfg: Bets, table: PricingTable | None) -> dict` with keys `status` (`"unavailable" | "too few" | "edge" | "no edge" | "undecided"`), `n`, `min_bets`, `mean`, `sd`, `ci_low`, `ci_high`, `at_close`, `decision_point`.

- [ ] **Step 1: Write the failing tests** — `tests/test_verdict.py`:

```python
import math

import polars as pl

from cfbr.bets.ledger import LEDGER_SCHEMA
from cfbr.bets.verdict import closing_implied_roi, verdict
from cfbr.config import Bets


class FakeTable:
    """expected_value from the point gap only: +2% per point the number beats the center, minus 4.5% vig."""

    def expected_value(self, side, number, odds, center):
        gap = (number - center) if side == "home" else (center - number)
        return -0.045 + 0.02 * gap


def _row(i, status, source, side, line, close):
    return {**{k: None for k in LEDGER_SCHEMA}, "game_id": i, "season": 2026, "week": 7, "side": side, "team": "T",
            "line": line, "odds": -110, "stake": 1.0, "status": status, "closing_source": source, "closing_line": close}


def _ledger(rows):
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def test_closing_implied_roi_only_for_settled_snapshot_closes():
    led = _ledger([_row(1, "won", "snapshot", "home", -3.0, -5.0),     # home at -3, close -5: 2 pts of CLV
                   _row(2, "lost", "cfbd", "home", -3.0, -5.0),         # fallback close: excluded
                   _row(3, "open", "snapshot", "away", 3.0, 3.0)])       # not settled: excluded
    r = closing_implied_roi(led, FakeTable())
    assert r.to_list()[0] == -0.045 + 0.04 and r.to_list()[1] is None and r.to_list()[2] is None


def test_verdict_statuses():
    cfg = Bets(pricing_books=[], verdict_min_bets=3)
    assert verdict(_ledger([]), cfg, None)["status"] == "unavailable"
    few = _ledger([_row(1, "won", "snapshot", "home", -3.0, -5.0)])
    assert verdict(few, cfg, FakeTable())["status"] == "too few"
    edge = _ledger([_row(i, "won", "snapshot", "home", -3.0, -7.0) for i in range(4)])          # +3.5% each
    v = verdict(edge, cfg, FakeTable())
    assert v["status"] == "edge" and math.isclose(v["mean"], 0.035) and v["n"] == 4 and math.isclose(v["at_close"], -0.045)
    none = _ledger([_row(i, "lost", "snapshot", "home", -3.0, -3.0) for i in range(4)])          # -4.5% each
    assert verdict(none, cfg, FakeTable())["status"] == "no edge"
    mixed = _ledger([_row(1, "won", "snapshot", "home", -3.0, -7.0), _row(2, "lost", "snapshot", "home", -3.0, -3.0),
                     _row(3, "won", "snapshot", "home", -3.0, -5.0), _row(4, "lost", "snapshot", "home", -3.0, -3.5)])
    assert verdict(mixed, cfg, FakeTable())["status"] == "undecided"
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_verdict.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`config.py`, `class Bets` after `min_sample`:

```python
    # Bets with a snapshot close before the closing-line-value verdict is read (DECISIONS
    # "Paper-bet verdict by closing-line value").
    verdict_min_bets: int = 300
```

`config/default.toml` `[bets]`: `verdict_min_bets = 300`.

`src/cfbr/bets/verdict.py`:

```python
"""Verdict by closing-line value (DECISIONS "Paper-bet verdict by closing-line value", 2026-10-30).

Closing-implied ROI of a settled bet = units won per unit risked at the recorded number and price if
the pre-kickoff consensus close were the true line (PricingTable.expected_value, key numbers
included). Its mean is what the ledger can measure precisely long before the win rate means
anything; a bettor taking the close itself earns about the vig, −4.5% at −110."""

from __future__ import annotations

import math

import polars as pl

from cfbr.bets.ledger import SETTLED
from cfbr.bets.pricing import PricingTable
from cfbr.config import Bets

DECISION_POINT = "the first weekly run after the 2027 regular season with enough bets (else 2028)"
_COLS = ("status", "closing_source", "closing_line", "side", "line", "odds")


def _eligible(r: dict) -> bool:
    return r["status"] in SETTLED and r["closing_source"] == "snapshot" and r["closing_line"] is not None


def closing_implied_roi(ledger: pl.DataFrame, table: PricingTable) -> pl.Series:
    """Per-bet closing-implied ROI for settled bets with a snapshot close; null elsewhere."""
    vals = [table.expected_value(r["side"], r["line"], int(r["odds"]), r["closing_line"]) if _eligible(r) else None
            for r in ledger.select(*_COLS).iter_rows(named=True)]
    return pl.Series("close_roi", vals, dtype=pl.Float64)


def verdict(ledger: pl.DataFrame, cfg: Bets, table: PricingTable | None) -> dict:
    out = {"status": "unavailable", "n": 0, "min_bets": cfg.verdict_min_bets, "decision_point": DECISION_POINT,
           "mean": math.nan, "sd": math.nan, "ci_low": math.nan, "ci_high": math.nan, "at_close": math.nan}
    if table is None:
        return out
    rows = [r for r in ledger.select(*_COLS).iter_rows(named=True) if _eligible(r)]
    r = closing_implied_roi(ledger, table).drop_nulls()
    n = r.len()
    out["n"] = n
    if n:
        out["mean"] = float(r.mean())
        out["at_close"] = float(sum(table.expected_value(x["side"], x["closing_line"], int(x["odds"]), x["closing_line"])
                                    for x in rows) / n)
    if n > 1:
        out["sd"] = float(r.std())
        half = 1.96 * out["sd"] / math.sqrt(n)
        out["ci_low"], out["ci_high"] = out["mean"] - half, out["mean"] + half
    if n < cfg.verdict_min_bets:
        out["status"] = "too few"
    elif out["ci_low"] > 0:
        out["status"] = "edge"
    elif out["ci_high"] < 0:
        out["status"] = "no edge"
    else:
        out["status"] = "undecided"
    return out
```

- [ ] **Step 4: Run** — `uv run pytest tests/test_verdict.py -v` → PASS. (In `test_verdict_statuses`, the "edge" ledger has four identical values, so sd = 0 and the CI collapses to the mean: status "edge" because 0.035 > 0. The "no edge" ledger likewise collapses to −0.045 < 0.)

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/bets/verdict.py src/cfbr/config.py config/default.toml tests/test_verdict.py
git commit -m "Paper bets: closing-implied ROI and the pre-registered CLV verdict"
```

### Task D3: bets.md section, /bets section, /api/bets key

**Files:**
- Modify: `src/cfbr/bets/ledger.py:summary_markdown` (insert a section after the headline table, before "## Week by week")
- Modify: `src/cfbr/web/bets.py:333-360` (`bets_ctx`), `:500-520` (`/api/bets`)
- Modify: `src/cfbr/web/templates/bets.html` (new section after the headline table)
- Test: `tests/test_bets.py`, `tests/test_web.py`

**Interfaces:**
- Consumes: `verdict(ledger, cfg.bets, table)`; `season_pricing_table(season)` (web/bets.py:533, same enclosing function as `bets_ctx`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bets.py`:

```python
def test_summary_markdown_has_the_verdict_section():
    md = summary_markdown(read_ledger(Path("/nonexistent")), CFG, 2026)
    assert "## Verdict by closing-line value" in md and "Pricing table not available" in md
```

Append to `tests/test_web.py`:

```python
def test_bets_page_and_api_carry_the_verdict(bets_client):
    html = " ".join(bets_client.get("/bets").text.split())
    assert "Verdict by closing-line value" in html
    api = bets_client.get("/api/bets").json()
    assert api["verdict"]["status"] in ("unavailable", "too few") and api["verdict"]["min_bets"] == 300
```

- [ ] **Step 2: Run to verify failure** → both FAIL on the missing text/key.

- [ ] **Step 3: Implement**

`ledger.py`, add `from cfbr.bets.verdict import verdict` **inside** `summary_markdown` (verdict.py imports `SETTLED` from ledger, so a top-level import would be circular), and insert before the `"## Week by week"` block:

```python
    from cfbr.bets.verdict import verdict

    v = verdict(ledger, cfg, table)
    lines += ["## Verdict by closing-line value", "",
              "Pre-registered 2026-10-30 (DECISIONS.md) after 57 bets had settled, none excluded. Closing-implied ROI = "
              "units won per unit risked at the recorded number and price if the pre-kickoff snapshot close were the "
              "true line (points-to-cents table, key numbers included). A bettor taking the close itself earns about "
              f"{_fmt(v['at_close'], pct=True)}, the vig. At ≥ {v['min_bets']} bets with a snapshot close: **edge over "
              "the market** if the 95% CI lower bound is above 0, **no edge** if the upper bound is below 0, otherwise "
              f"undecided and it continues. Decision: {v['decision_point']}.", ""]
    if v["status"] == "unavailable":
        lines += ["Pricing table not available; no verdict computed.", ""]
    else:
        lines += [f"Bets with a snapshot close: {v['n']} of {v['min_bets']}. Mean closing-implied ROI "
                  f"{_fmt(v['mean'], pct=True)} [{_fmt(v['ci_low'], pct=True)}, {_fmt(v['ci_high'], pct=True)}] "
                  f"(per-bet SD {_fmt(v['sd'], pct=True)}). Status: **{v['status']}**.", ""]
```

`web/bets.py`: in `bets_ctx` add `"verdict": verdict(ledger, cfg.bets, season_pricing_table(season)),` next to `"tiers"`; in `/api/bets` add `"verdict": clean_summary(verdict(ledger, cfg.bets, season_pricing_table(season))),`; import `from cfbr.bets.verdict import verdict` at the top. (`season_pricing_table` is defined later in the same enclosing function; closures resolve at call time, so the reference is fine. If `bets_ctx` turns out to be module-level, pass the table in from each route the way `tiers` is.)

`bets.html`: after the headline results table add

```html
<h2>Verdict by closing-line value</h2>
<p class="note">Pre-registered Oct 30, 2026. Closing-implied ROI is what each bet would return if the pre-kickoff close were the true line; taking the close itself returns about {{ verdict.at_close|pct if verdict.at_close == verdict.at_close else "−4.5%" }}, the vig.
  At {{ verdict.min_bets }} bets: <strong>edge</strong> if the 95% interval is above 0, <strong>no edge</strong> if below, else undecided. Decision: {{ verdict.decision_point }}.</p>
{% if verdict.status == "unavailable" %}<p class="muted">Pricing table not available.</p>
{% else %}<p>{{ verdict.n }} of {{ verdict.min_bets }} bets with a snapshot close. Mean closing-implied ROI
  {% if verdict.n %}<strong>{{ verdict.mean|pct }}</strong>{% if verdict.n > 1 %} [{{ verdict.ci_low|pct }}, {{ verdict.ci_high|pct }}]{% endif %}{% else %}—{% endif %}.
  Status: <strong>{{ verdict.status }}</strong>.</p>{% endif %}
```

(`pct` is the existing Jinja filter; the `x == x` test guards NaN. Check `_fmt`'s `pct=True` output for negatives renders as `-4.5%`, matching.)

- [ ] **Step 4: Run** — `uv run pytest tests/test_bets.py tests/test_web.py -v` → PASS. Also `tests/test_web.py::test_bets_api_matches_ledger_summary` still passes (it compares `summary`, not the new key).

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/bets/ledger.py src/cfbr/web/bets.py src/cfbr/web/templates/bets.html tests/test_bets.py tests/test_web.py
git commit -m "Paper bets: show the closing-line-value verdict in bets.md, /bets and /api/bets"
```

### Task D4: Docs, PR

- [ ] README: in the Bets section add one sentence on the verdict and `bets.verdict_min_bets` to the config table.
- [ ] `uv run pytest -q` → all pass. PR titled `feat: pre-registered closing-line-value verdict for paper bets`. Squash-merge; deploy is stop-and-ask. After deploy, the next weekly run writes the section with the real 57+ bets; nothing else is needed.

---

## Order and dependencies

1. **Part A** first; it is additive and gives every later discussion the right comparator.
2. **Part B** next. B1 is a one-line commit that must land before any fitting; B5 decides whether B6 exists. If the gate passes, deploy before Thu 2026-11-05.
3. **Part C** and **Part D** are independent of A and B and of each other. Both reuse `paired_mae_gain` (C) from B6; if C merges first, carry the function in C and let the later merge dedupe.
4. Every PR: `uv run pytest -q` green, CI green, squash-merge, then stop and ask before `deploy/deploy.sh`.

## Self-review notes

- Spec coverage: item 1 → Part A; item 2 → Part B; item 4 → Part C; item 5 → Part D. The week-1 slope note (item 3 of the review) is deliberately not a task: its incremental MAE over v2 was ~+0.007 and the user did not select it.
- Numbers in `OPENER_TIER_COVER` are placeholders *by design*: Task A3 step 6 produces the authoritative values and `test_opener_tier_constants_match_the_holdout_run` fails if they drift.
- `paired_mae_gain` is defined once (Part B, Task B6) and reused in Part C; signatures match (`df, a, b`). `paired_gain` in `calib_gate.py` takes arrays, not frames; the two are intentionally different helpers for different inputs.
- Nothing here touches `config/tuned.toml`, the lines → model boundary, `schemes.py`, or the holdout's headline numbers.
