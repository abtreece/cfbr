# FCS Two-Stage Fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** FBS-vs-FCS games barely move FBS ratings (`fcs_weight`, tuned), and FCS groups are rated separately against the FCS average. If the pre-registered three-gate rule passes, adopt it mid-2026 and restate the season.

**Architecture:** Stage 1 is the existing ridge, with each play's weight multiplied by `fcs_weight` when either side is a pooled FCS entity; the FBS ratings come from it. Stage 2 is a small ridge over the FCS plays only, with Stage 1's FBS ratings, home term and intercept as fixed offsets, plus a shared FCS level per side and stat and per-group deviations shrunk toward the prior deviation. It produces the FCS ratings. One helper computes per-play weights for every fit.

**Tech Stack:** Python 3.12, numpy, scipy.sparse, polars, DuckDB, Optuna (TPE), typer, pytest, Jinja.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-11-08-fcs-two-stage-design.md`. Rule: last line of `DECISIONS.md` ("Model (FCS)"), committed in `55fb749`.
- `model.fcs_two_stage = false` must reproduce today's fit **bit for bit**. Verify with a v1 backtest against the baseline run `results/*-v1-baseline-timing` (MAE 13.382342531147511 over 5,844 games).
- Tuning: 256 Optuna TPE trials, seed 0. Fixed: `fcs_two_stage = true`, `fcs_pooling = "conference"`, garbage time off, recency on. Searched: `fcs_weight` U[0, 1], `alpha_ppa` and `alpha_sr` log 20–3000, `half_life_weeks` 2–26. Massey settings are not re-tuned.
- Gates, all on paired MAE gain v1 − v2 with a 95% CI: (1) 2015–2021 gain > 0; (2) 2022–2025 gain > 0 and CI low > −0.10; (3) 2026 regular season through week 10, gain > 0 and CI low > −0.10. If any gate fails, nothing published changes and there is no re-tuning.
- `config/tuned.toml` is not modified unless all three gates pass.
- Point-in-time: Stage 2 uses only plays with kickoff < as-of (the same prefix `k` as Stage 1).
- Betting lines are never model inputs. The paper-bet ledger is append-only and never rewritten.
- Local work uses the production DB copy symlinked at `data/cfb.duckdb`. No API calls. Don't run `weekly`, `bets`, `odds-snapshot`, `backup` or `check` locally.
- User style: 2-space indent in shell, 4-space Python (existing code), single-purpose commits, conventional commit prefix, **no Co-Authored-By lines**. Separate Bash calls instead of `&&` chains.
- Deploying and the javelin restatement are stop-and-ask steps.

## File map

| File | Change |
|---|---|
| `src/cfbr/config.py`, `config/default.toml` | `model.fcs_two_stage`, `model.fcs_weight`; `bets.verdict_split_at` (Task 6) |
| `src/cfbr/model/ridge.py` | `play_weights()`, `fit_fcs_stage2()` |
| `src/cfbr/model/engine.py` | `_fit` uses `play_weights`, then Stage 2 when enabled |
| `src/cfbr/model/uncertainty.py`, `src/cfbr/model/matchup.py` | use `play_weights` |
| `src/cfbr/backtest/tune.py`, `src/cfbr/cli.py` (`tune`) | `fcs-two-stage` search space, `patch_frozen`, FCS report |
| `src/cfbr/backtest/fcs_gate.py` (new), `src/cfbr/cli.py` (`fcs-gate`) | the three gates |
| `src/cfbr/report/restate.py` (new), `src/cfbr/cli.py` (`restate`, shadow helper) | archive and re-rank a live season |
| `src/cfbr/bets/verdict.py`, `src/cfbr/bets/ledger.py`, `src/cfbr/web/templates/bets.html` | before/after context split |
| `src/cfbr/web/templates/method.html` | FCS sentence and restatement note |
| `tests/test_fcs_two_stage.py` (new), `tests/conftest.py`, `tests/test_leakage.py`, `tests/test_web.py`, `tests/test_verdict.py`, `tests/test_fcs_gate.py` (new), `tests/test_tune.py` (new), `tests/test_restate.py` (new) | tests |

---

### Task 1: Config keys and the shared play-weight helper

**Files:**
- Modify: `src/cfbr/config.py:48-56` (class `Model`)
- Modify: `config/default.toml` (`[model]` section)
- Modify: `src/cfbr/model/ridge.py` (add `play_weights` after `recency_weights`)
- Modify: `src/cfbr/model/engine.py:26,122-146` (import; `_fit`)
- Modify: `src/cfbr/model/uncertainty.py:82`, `src/cfbr/model/matchup.py:44`
- Test: `tests/test_fcs_two_stage.py` (new)

**Interfaces:**
- Produces: `Model.fcs_two_stage: bool = False`, `Model.fcs_weight: float = 1.0` (0 ≤ x ≤ 1).
- Produces: `play_weights(a, k: int, as_of_ns: int | None, m: Model) -> np.ndarray | None` in `cfbr.model.ridge`. `a` is any object with `kickoff_ns`, `off`, `de`, `center` arrays (`engine.SeasonArrays`). Returns `None` when no weighting applies (same contract as `recency_weights`).
- Produces (test helper, reused by Task 2): `_season(thin_plays=40)` in `tests/test_fcs_two_stage.py`, returning a synthetic `SeasonArrays` with FBS teams `T0..T19` and FCS groups `F0, F1, F2` (indices 20, 21, 22; F2 is thin).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fcs_two_stage.py`:

```python
"""FCS two-stage fit (DECISIONS "FCS games, two-stage fit"): synthetic checks, no DB."""

import numpy as np
import pytest

from cfbr.config import Model, load_config
from cfbr.model.engine import SeasonArrays
from cfbr.model.ridge import fit_efficiency_arrays, play_weights, recency_weights

WEEK_NS = 7 * 86400 * 10**9
N_FBS = 20
THIN = 22  # index of the thin FCS group F2


def _season(thin_plays=40, seed=11):
    """20 FBS teams (true O/D ~ N(0, 0.1)); FCS groups F0, F1 (8 games each) and F2 (one short game),
    all with true O = D = -0.4. Plays sorted by kickoff like Engine.arrays."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(N_FBS)] + ["F0", "F1", "F2"]
    center = np.array([t.startswith("T") for t in teams])
    o_true = np.r_[rng.normal(0, 0.1, N_FBS), [-0.4] * 3]
    d_true = np.r_[rng.normal(0, 0.1, N_FBS), [-0.4] * 3]
    rows = []  # (offense, defense, home indicator, week, plays)
    for week in range(10):
        for a, b in rng.permutation(N_FBS).reshape(-1, 2):
            rows += [(a, b, 1.0, week, 60), (b, a, -1.0, week, 60)]
    for g, n_games, per_side in ((20, 8, 60), (21, 8, 60), (THIN, 1, thin_plays // 2)):
        for week in range(n_games):
            fbs = int(rng.integers(N_FBS))
            rows += [(fbs, g, 1.0, week, per_side), (g, fbs, -1.0, week, per_side)]
    off, de, h, wk = [], [], [], []
    for o, d, hh, w, c in rows:
        off += [o] * c
        de += [d] * c
        h += [hh] * c
        wk += [w] * c
    off, de, h, wk = np.array(off), np.array(de), np.array(h, dtype=float), np.array(wk)
    ppa = 0.05 + o_true[off] - d_true[de] + 0.02 * h + rng.normal(0, 1.2, off.size)
    order = np.argsort(wk, kind="stable")
    return SeasonArrays(teams=teams, center=center, kickoff_ns=(wk[order] * WEEK_NS).astype(np.int64),
                        off=off[order], de=de[order], home=h[order], ppa=ppa[order],
                        success=(ppa[order] > 0).astype(float))


def _stage1(a, w, alpha=300.0, keep=None):
    sel = np.ones(a.off.size, dtype=bool) if keep is None else keep
    return fit_efficiency_arrays(a.off[sel], a.de[sel], a.home[sel], a.ppa[sel], a.teams, alpha, None, None,
                                 None if w is None else w[sel], a.center)


def test_config_defaults_leave_behaviour_unchanged():
    for path in (None, "config/tuned.toml"):
        m = load_config(path).model
        assert m.fcs_two_stage is False and m.fcs_weight == 1.0


def test_fcs_weight_must_be_a_fraction():
    with pytest.raises(ValueError):
        Model(fcs_weight=1.5)


@pytest.mark.parametrize("two_stage,weight", [(False, 0.2), (True, 1.0)])
def test_play_weights_reduce_to_recency_exactly(two_stage, weight):
    a = _season()
    k = a.off.size
    as_of = int(a.kickoff_ns.max()) + WEEK_NS
    m = Model(half_life_weeks=3.7, fcs_two_stage=two_stage, fcs_weight=weight)
    np.testing.assert_array_equal(play_weights(a, k, as_of, m), recency_weights(a.kickoff_ns[:k], as_of, 3.7))
    assert play_weights(a, k, None, m) is None


def test_play_weights_scale_fcs_plays_only():
    a = _season()
    k = a.off.size
    w = play_weights(a, k, None, Model(fcs_two_stage=True, fcs_weight=0.25))
    fcs = ~a.center[a.off] | ~a.center[a.de]
    assert set(np.unique(w[fcs])) == {0.25}
    assert set(np.unique(w[~fcs])) == {1.0}


def test_play_weights_respect_the_prefix():
    a = _season()
    k = a.off.size // 2
    w = play_weights(a, k, int(a.kickoff_ns[k]), Model(half_life_weeks=3.7, fcs_two_stage=True, fcs_weight=0.3))
    assert w.size == k


def test_fcs_weight_zero_isolates_fbs_ratings():
    a = _season()
    k = a.off.size
    drop_thin_game = ~((a.off == THIN) | (a.de == THIN))
    w0 = play_weights(a, k, None, Model(fcs_two_stage=True, fcs_weight=0.0))
    full, part = _stage1(a, w0), _stage1(a, w0, keep=drop_thin_game)
    np.testing.assert_allclose(full.o[a.center], part.o[a.center], atol=1e-12)
    np.testing.assert_allclose(full.d[a.center], part.d[a.center], atol=1e-12)
    # Control: at full weight the same game does move FBS ratings.
    w1 = play_weights(a, k, None, Model(fcs_two_stage=True, fcs_weight=1.0))
    moved = np.abs(_stage1(a, w1).o[a.center] - _stage1(a, w1, keep=drop_thin_game).o[a.center]).max()
    assert moved > 1e-4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fcs_two_stage.py -v`
Expected: collection error, `ImportError: cannot import name 'play_weights'`.

- [ ] **Step 3: Add the config keys**

In `src/cfbr/config.py`, class `Model`, after `calibration_v3`:

```python
    # FCS games, two-stage fit (DECISIONS "FCS games, two-stage fit"): False = one fit for everything.
    fcs_two_stage: bool = False
    # Stage-1 weight of a play with a pooled FCS side; only used when fcs_two_stage is on.
    fcs_weight: float = Field(1.0, ge=0.0, le=1.0)
```

In `config/default.toml`, `[model]`, after `prior_k = 0.0`:

```toml
# FCS games, two-stage fit (DECISIONS "FCS games, two-stage fit"): false = one fit for everything.
fcs_two_stage = false
# Stage-1 weight of a play with a pooled FCS side (only used when fcs_two_stage is on).
fcs_weight = 1.0
```

- [ ] **Step 4: Add `play_weights` to `src/cfbr/model/ridge.py`** (after `recency_weights`)

```python
def play_weights(a, k: int, as_of_ns: int | None, m) -> np.ndarray | None:
    """Per-play weights for the first ``k`` plays of season arrays ``a`` (Engine.SeasonArrays):
    recency decay at ``as_of_ns`` (none when it is None), times ``m.fcs_weight`` for plays with a
    pooled FCS side when ``m.fcs_two_stage`` is on. ``None`` when nothing applies, so the fit is
    unweighted. Every fit that should match the published one calls this (engine, uncertainty, matchup)."""
    w = None if as_of_ns is None else recency_weights(a.kickoff_ns[:k], as_of_ns, m.half_life_weeks)
    if m.fcs_two_stage and m.fcs_weight != 1.0:
        fcs = ~a.center[a.off[:k]] | ~a.center[a.de[:k]]
        factor = np.where(fcs, m.fcs_weight, 1.0)
        w = factor if w is None else w * factor
    return w
```

- [ ] **Step 5: Use it in `Engine._fit`, `uncertainty.py` and `matchup.py`**

`src/cfbr/model/engine.py` import line 26 becomes:

```python
from cfbr.model.ridge import fit_efficiency_arrays, home_indicator, play_weights, recency_weights
```

In `_fit`, replace

```python
        w = None
        if as_of is not None:
            w = recency_weights(a.kickoff_ns[:k], _ns(as_of), self.cfg.model.half_life_weeks)
```

with

```python
        w = play_weights(a, k, None if as_of is None else _ns(as_of), self.cfg.model)
```

`src/cfbr/model/uncertainty.py:82`: replace `w = recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)` with `w = play_weights(a, k, _ns(as_of), m)`, and add `play_weights` to that file's `cfbr.model.ridge` import (drop `recency_weights` from it if nothing else in the file uses it; check with `grep -n recency_weights src/cfbr/model/uncertainty.py`).

`src/cfbr/model/matchup.py:44`: replace `w_all = recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)` with `w_all = play_weights(a, k, _ns(as_of), m)`, and update its import the same way.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fcs_two_stage.py -v`
Expected: all PASS.

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: all pass (the defaults route through `play_weights` and return the same arrays as before).

- [ ] **Step 8: Commit**

```bash
git add src/cfbr/config.py config/default.toml src/cfbr/model/ridge.py src/cfbr/model/engine.py src/cfbr/model/uncertainty.py src/cfbr/model/matchup.py tests/test_fcs_two_stage.py
git commit -m "feat: FCS two-stage: config keys and shared play-weight helper"
```

---

### Task 2: Stage 2 FCS fit

**Files:**
- Modify: `src/cfbr/model/ridge.py` (add `fit_fcs_stage2`; add `from dataclasses import dataclass, replace`)
- Test: `tests/test_fcs_two_stage.py`

**Interfaces:**
- Consumes: `EffFit` (o, d, home, intercept, n_off, n_def), `_season`, `_stage1` and `THIN` from Task 1's test file.
- Produces: `fit_fcs_stage2(off, de, h, y, fcs, alpha, stage1: EffFit, prior_o=None, prior_d=None, weights=None) -> EffFit`. `fcs` is a boolean mask over teams (True = pooled FCS entity). Returns a copy of `stage1` whose `o`/`d` FCS entries are replaced; FBS entries, `home` and `intercept` are unchanged. `stage1` is never mutated.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fcs_two_stage.py`; add `fit_fcs_stage2` to the ridge import at the top)

```python
def test_stage2_thin_group_shrinks_to_the_fcs_level_not_an_average_fbs_team():
    a = _season(thin_plays=40)
    s1 = _stage1(a, None)
    s2 = fit_fcs_stage2(a.off, a.de, a.home, a.ppa, ~a.center, 300.0, s1)
    assert s1.o[THIN] > -0.15   # single fit, no prior: pulled toward 0 (an average FBS team)
    assert s2.o[THIN] < -0.30   # stage 2: pulled toward the FCS level (true -0.4)
    assert s2.d[THIN] < -0.30


def test_stage2_keeps_stage1_fbs_ratings_and_does_not_mutate_stage1():
    a = _season()
    s1 = _stage1(a, None)
    o0, d0 = s1.o.copy(), s1.d.copy()
    s2 = fit_fcs_stage2(a.off, a.de, a.home, a.ppa, ~a.center, 300.0, s1)
    np.testing.assert_array_equal(s2.o[a.center], o0[a.center])
    np.testing.assert_array_equal(s2.d[a.center], d0[a.center])
    np.testing.assert_array_equal(s1.o, o0)
    np.testing.assert_array_equal(s1.d, d0)
    assert (s2.home, s2.intercept) == (s1.home, s1.intercept)


def test_stage2_without_fcs_plays_returns_the_prior():
    a = _season()
    fbs_only = a.center[a.off] & a.center[a.de]
    s1 = _stage1(a, None, keep=fbs_only)
    prior_o = np.r_[np.zeros(N_FBS), [-0.3, -0.5, -0.1]]
    prior_d = np.r_[np.zeros(N_FBS), [-0.2, -0.6, 0.0]]
    s2 = fit_fcs_stage2(a.off[fbs_only], a.de[fbs_only], a.home[fbs_only], a.ppa[fbs_only], ~a.center, 300.0,
                        s1, prior_o, prior_d)
    np.testing.assert_allclose(s2.o[~a.center], prior_o[~a.center], atol=1e-12)
    np.testing.assert_allclose(s2.d[~a.center], prior_d[~a.center], atol=1e-12)


def test_stage2_ignores_zero_weight_plays():
    a = _season()
    s1 = _stage1(a, None)
    w = np.where(a.off == THIN, 0.0, 1.0) * np.where(a.de == THIN, 0.0, 1.0)
    keep = w > 0
    s2_w = fit_fcs_stage2(a.off, a.de, a.home, a.ppa, ~a.center, 300.0, s1, weights=w)
    s2_k = fit_fcs_stage2(a.off[keep], a.de[keep], a.home[keep], a.ppa[keep], ~a.center, 300.0, s1)
    np.testing.assert_allclose(s2_w.o, s2_k.o, atol=1e-12)
    np.testing.assert_allclose(s2_w.d, s2_k.d, atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fcs_two_stage.py -v -k stage2`
Expected: `ImportError: cannot import name 'fit_fcs_stage2'`.

- [ ] **Step 3: Implement `fit_fcs_stage2`** in `src/cfbr/model/ridge.py` (after `fit_efficiency_arrays`; change the import to `from dataclasses import dataclass, replace`)

```python
FCS_LEVEL_PENALTY = 1e-6  # x alpha: keeps the shared FCS levels defined before any FCS play


def fit_fcs_stage2(
    off: np.ndarray,
    de: np.ndarray,
    h: np.ndarray,
    y: np.ndarray,
    fcs: np.ndarray,
    alpha: float,
    stage1: EffFit,
    prior_o: np.ndarray | None = None,
    prior_d: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> EffFit:
    """Stage 2 of the FCS two-stage fit (DECISIONS "FCS games, two-stage fit").

    Rates pooled FCS entities (``fcs`` mask over teams) from plays with an FCS side, holding
    Stage 1's FBS ratings, home coefficient and intercept fixed:
        FCS offense:  y = c + H h + O[g] - D1[fbs]
        FCS defense:  y = c + H h + O1[fbs] - D[g]
    with O[g] = prior_o[g] + dL_o + dev_o[g] (same for D). dL (shared by all FCS groups) has a
    negligible penalty; dev has ``alpha``. With no prior (final()), groups shrink toward the FCS
    average, not toward an average FBS team. Returns a copy of ``stage1`` with FCS rows replaced."""
    n = len(stage1.teams)
    groups = np.flatnonzero(fcs)
    o, d = stage1.o.copy(), stage1.d.copy()
    if groups.size == 0:
        return replace(stage1, o=o, d=d)
    p_o = np.zeros(n) if prior_o is None else np.asarray(prior_o, dtype=float)
    p_d = np.zeros(n) if prior_d is None else np.asarray(prior_d, dtype=float)
    g = groups.size
    pos = np.full(n, -1)
    pos[groups] = np.arange(g)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    so = np.flatnonzero(fcs[off] & ~fcs[de])   # FCS on offense
    sd = np.flatnonzero(fcs[de] & ~fcs[off])   # FCS on defense
    c, hh = stage1.intercept, stage1.home
    target = np.concatenate([
        y[so] - (c + hh * h[so] - stage1.d[de[so]]) - p_o[off[so]],
        y[sd] - (c + hh * h[sd] + stage1.o[off[sd]]) + p_d[de[sd]],
    ])
    # Columns: [dL_o, dL_d, dev_o (g), dev_d (g)]
    X = np.zeros((so.size + sd.size, 2 + 2 * g))
    X[:so.size, 0] = 1.0
    X[np.arange(so.size), 2 + pos[off[so]]] = 1.0
    X[so.size:, 1] = -1.0
    X[so.size + np.arange(sd.size), 2 + g + pos[de[sd]]] = -1.0
    ww = np.concatenate([w[so], w[sd]])
    A = X.T @ (X * ww[:, None])
    b = X.T @ (ww * target)
    A[np.diag_indices_from(A)] += np.r_[[FCS_LEVEL_PENALTY * alpha] * 2, np.full(2 * g, alpha)]
    coef = np.linalg.solve(A, b)
    o[groups] = p_o[groups] + coef[0] + coef[2:2 + g]
    d[groups] = p_d[groups] + coef[1] + coef[2 + g:]
    return replace(stage1, o=o, d=d)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fcs_two_stage.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/model/ridge.py tests/test_fcs_two_stage.py
git commit -m "feat: FCS two-stage: stage-2 fit for pooled FCS entities"
```

---

### Task 3: Wire two-stage into the engine; leakage and site-match tests under both configs

**Files:**
- Modify: `src/cfbr/model/engine.py` (`_fit`; import `fit_fcs_stage2`)
- Modify: `tests/conftest.py` (add `cfg_two_stage`)
- Modify: `tests/test_leakage.py` (canary and prefix tests parametrized over both configs)
- Modify: `tests/test_web.py:460-481` (site-matches-rank test under both configs)

**Interfaces:**
- Consumes: `play_weights` (Task 1), `fit_fcs_stage2` (Task 2).
- Produces: `Engine.snapshot()` / `Engine.final()` return two-stage ratings when `cfg.model.fcs_two_stage`. Same columns as before (`team, o_ppa, d_ppa, o_sr, d_sr, n_off, n_def, w_off, w_def`). Fixture `cfg_two_stage` (session scope).

- [ ] **Step 1: Add the fixture** to `tests/conftest.py` after `cfg`:

```python
@pytest.fixture(scope="session")
def cfg_two_stage(cfg):
    """default.toml with the FCS two-stage fit on and recency on, so the dataset tests exercise it."""
    c = cfg.model_copy(deep=True)
    c.model.fcs_two_stage = True
    c.model.fcs_weight = 0.3
    c.model.half_life_weeks = 3.7
    return c
```

- [ ] **Step 2: Parametrize the leakage tests.** In `tests/test_leakage.py`, add after the imports:

```python
import numpy as np

from cfbr.model.engine import _ns
from cfbr.model.ridge import fit_efficiency, fit_fcs_stage2, home_indicator


@pytest.fixture(params=["default", "two_stage"])
def model_cfg(request, cfg, cfg_two_stage):
    return cfg if request.param == "default" else cfg_two_stage
```

Change `test_canary_future_game_does_not_change_earlier_ratings(cfg, dataset)` to take `model_cfg` and use it for both engines:

```python
def test_canary_future_game_does_not_change_earlier_ratings(model_cfg, dataset):
    clean = Engine(dataset, model_cfg, last_completed=2025)
    dirty = Engine(_with_canary(dataset, clean), model_cfg, last_completed=2025)
```

(the rest of the body is unchanged). Replace `test_engine_prefix_fit_matches_explicit_slice` entirely with:

```python
def test_engine_prefix_fit_matches_explicit_slice(model_cfg, dataset):
    """The engine's prefix fit equals an independent fit on training_slice, with the weights rebuilt
    from the plays frame (recency x FCS factor) and, under two-stage, Stage 2 on the same slice."""
    m = model_cfg.model
    eng = Engine(dataset, model_cfg, last_completed=2025)
    season = 2018
    teams, fbs = eng.entities(season), eng.fbs(season)
    fcs_mask = ~np.isin(np.array(teams, dtype=object), fbs)
    idx = {t: i for i, t in enumerate(teams)}
    for slot in ("r01", "r05", "r12", "p1"):
        as_of = eng.slot_as_of(season, slot)
        snap = eng.snapshot(season, as_of)
        plays = training_slice(eng.plays, season, as_of)
        pr = pl.DataFrame({"team": teams}).join(eng.priors(season), on="team", how="left").fill_null(0.0)
        rec = None
        if m.half_life_weeks > 0 and plays.height:
            kick = plays["kickoff_ts"].dt.cast_time_unit("ns").dt.epoch("ns").to_numpy()
            rec = np.power(0.5, (_ns(as_of) - kick) / (7 * 86400 * 1e9) / m.half_life_weeks)
        w = rec
        if m.fcs_two_stage:
            is_fcs = ~(plays["offense"].is_in(fbs) & plays["defense"].is_in(fbs)).to_numpy()
            factor = np.where(is_fcs, m.fcs_weight, 1.0)
            w = factor if w is None else w * factor
        ref = fit_efficiency(plays, "ppa", m.alpha_ppa, teams, pr["o_ppa"].to_numpy(), pr["d_ppa"].to_numpy(),
                             weights=w, center_on=fbs)
        if m.fcs_two_stage:
            off = plays["offense"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
            de = plays["defense"].replace_strict(idx, return_dtype=pl.Int64).to_numpy()
            h = home_indicator(plays["offense"], plays["defense"], plays["home_team"], plays["neutral_site"])
            ref = fit_fcs_stage2(off, de, h, plays["ppa"].cast(pl.Float64).to_numpy(), fcs_mask, m.alpha_ppa, ref,
                                 pr["o_ppa"].to_numpy(), pr["d_ppa"].to_numpy(), rec)
        assert abs(snap["o_ppa"].to_numpy() - ref.o).max() < 1e-9
        assert abs(snap["d_ppa"].to_numpy() - ref.d).max() < 1e-9
```

- [ ] **Step 3: Parametrize the site-match test.** In `tests/test_web.py`, after the `engine` fixture (line ~465) add:

```python
@pytest.fixture(scope="module")
def engine_two_stage(cfg_two_stage, dataset):
    from cfbr.ingest.fetch import current_season
    from cfbr.model.engine import Engine

    return Engine(dataset, cfg_two_stage, last_completed=current_season() - 1)
```

and change the test header to:

```python
@pytest.mark.parametrize("engine_name", ["engine", "engine_two_stage"])
def test_web_ratings_match_rank_output(engine_name, request, tmp_path):
    from cfbr.report.rank import build_rankings

    engine = request.getfixturevalue(engine_name)
```

(the rest of the body is unchanged).

- [ ] **Step 4: Run the new parametrizations to verify they fail**

Run: `uv run pytest tests/test_leakage.py -v -k "two_stage"`
Expected: `test_engine_prefix_fit_matches_explicit_slice[two_stage]` FAILS (the engine has no Stage 2 yet, so FCS rows differ).

- [ ] **Step 5: Implement Stage 2 in `Engine._fit`.** Change the ridge import to include `fit_fcs_stage2`. In `_fit`, after the two `fit_efficiency_arrays` calls, insert:

```python
        if m.fcs_two_stage:
            rec = None if as_of is None else recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)
            fcs = ~a.center
            f_ppa = fit_fcs_stage2(off, de, h, a.ppa[:k], fcs, m.alpha_ppa, f_ppa, p["o_ppa"], p["d_ppa"], rec)
            f_sr = fit_fcs_stage2(off, de, h, a.success[:k], fcs, m.alpha_sr, f_sr, p["o_sr"], p["d_sr"], rec)
```

(`m = self.cfg.model` is already defined above the fits; move that line above the new `play_weights` call if it isn't. `w_off`/`w_def` keep using `w`, the Stage 1 weights, which is what the prior share should reflect.)

- [ ] **Step 6: Run the targeted tests to verify they pass**

Run: `uv run pytest tests/test_leakage.py tests/test_fcs_two_stage.py -v`
Run: `uv run pytest tests/test_web.py -v -k test_web_ratings_match_rank_output`
Expected: all PASS, both parametrizations.

- [ ] **Step 7: Prove v1 is bit-for-bit unchanged**

Run: `uv run cfbr backtest --seasons 2015-2021 --config config/tuned.toml --tag v1-after-two-stage`
Expected: `games 5844  MAE 13.38 ...`, the same as the baseline line.

Run:
```bash
uv run python -c "import glob, polars as pl; a=pl.read_parquet(glob.glob('results/*-v1-baseline-timing/predictions.parquet')[0]).sort('game_id'); b=pl.read_parquet(sorted(glob.glob('results/*-v1-after-two-stage/predictions.parquet'))[-1]).sort('game_id'); print((a['pred_margin']==b['pred_margin']).all(), a.height, b.height)"
```
Expected: `True 5844 5844` for the 2015–2021 rows. (The baseline run covers 2015–2021; both runs use the same seasons.)

- [ ] **Step 8: Full suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/cfbr/model/engine.py tests/conftest.py tests/test_leakage.py tests/test_web.py
git commit -m "feat: FCS two-stage: engine wiring; leakage and site-match tests under two-stage"
```

---

### Task 4: Tuning space, frozen-config patch, run the search

**Files:**
- Modify: `src/cfbr/backtest/tune.py` (`apply_params`, new `suggest_fcs_two_stage`, `SPACES`, `run_search(space=)`, `sensitivity_grids`, `sensitivity(space=)`, `dict_to_toml`, `patch_frozen`, `write_fcs_report`)
- Modify: `src/cfbr/cli.py:131-169` (`tune --space`)
- Test: `tests/test_tune.py` (new)
- Output (committed): `config/tuned_v2.toml`

**Interfaces:**
- Consumes: `Model.fcs_two_stage`, `Model.fcs_weight` (Task 1).
- Produces: `suggest_fcs_two_stage(trial) -> dict`; `run_search(config_path, seasons, n_trials, n_jobs, seed=0, space="phase6")`; `sensitivity(config_path, best, seasons, n_jobs, space="phase6")`; `patch_frozen(base_path: Path, best: dict, path: Path) -> Config`; CLI `cfbr tune --space fcs-two-stage --config config/tuned.toml`, which writes `config/tuned_v2.toml` and `results/tuning-fcs-two-stage/`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_tune.py`:

```python
import optuna

from cfbr.backtest.tune import apply_params, patch_frozen, sensitivity_grids, suggest_fcs_two_stage
from cfbr.config import REPO_ROOT, load_config


def test_apply_params_sets_fcs_keys():
    cfg = apply_params(load_config(), {"fcs_two_stage": True, "fcs_weight": 0.2})
    assert cfg.model.fcs_two_stage is True and cfg.model.fcs_weight == 0.2


def test_fcs_space_fixes_the_rule_settings():
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(25):
        p = suggest_fcs_two_stage(study.ask())
        assert p["fcs_two_stage"] is True and p["fcs_pooling"] == "conference" and p["gt_enabled"] is False
        assert 0.0 <= p["fcs_weight"] <= 1.0
        assert 20 <= p["alpha_ppa"] <= 3000 and 20 <= p["alpha_sr"] <= 3000
        assert 2 <= p["half_life_weeks"] <= 26


def test_fcs_sensitivity_grid_never_turns_recency_off_or_varies_fixed_settings():
    g = sensitivity_grids({"alpha_ppa": 300.0, "alpha_sr": 90.0, "half_life_weeks": 3.7, "fcs_weight": 0.1},
                          "fcs-two-stage")
    assert set(g) == {"fcs_weight", "alpha_ppa", "alpha_sr", "half_life_weeks"}
    assert min(g["half_life_weeks"]) > 0 and 0.0 in g["fcs_weight"] and 1.0 in g["fcs_weight"]


def test_patch_frozen_changes_only_the_named_keys(tmp_path):
    base = REPO_ROOT / "config" / "tuned.toml"
    out = tmp_path / "t.toml"
    best = {"fcs_two_stage": True, "fcs_weight": 0.12, "alpha_ppa": 250.0, "alpha_sr": 80.0,
            "half_life_weeks": 4.2, "fcs_pooling": "conference", "gt_enabled": False}
    cfg, ref = patch_frozen(base, best, out), load_config(base)
    assert cfg.model.fcs_two_stage is True and cfg.model.fcs_weight == 0.12
    assert (cfg.model.alpha_ppa, cfg.model.alpha_sr, cfg.model.half_life_weeks) == (250.0, 80.0, 4.2)
    assert cfg.baseline == ref.baseline and cfg.ingest == ref.ingest and cfg.seasons == ref.seasons
    assert cfg.prior == ref.prior and cfg.backtest == ref.backtest and cfg.bets == ref.bets
    assert "[bets]" not in out.read_text()  # unlisted sections stay on pydantic defaults, as in tuned.toml
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_tune.py -v`
Expected: `ImportError` (`patch_frozen`, `sensitivity_grids`, `suggest_fcs_two_stage` don't exist).

- [ ] **Step 3: Implement in `src/cfbr/backtest/tune.py`.** Add `import tomllib` at the top.

In `apply_params`, before the final `else: raise KeyError(k)`:

```python
        elif k == "fcs_weight":
            m.fcs_weight = float(v)
        elif k == "fcs_two_stage":
            m.fcs_two_stage = bool(v)
```

After `suggest`:

```python
def suggest_fcs_two_stage(trial: optuna.Trial) -> dict:
    """Search space fixed by DECISIONS "FCS games, two-stage fit" (2026-11-08)."""
    return {
        "fcs_two_stage": True, "fcs_pooling": "conference", "gt_enabled": False,
        "fcs_weight": trial.suggest_float("fcs_weight", 0.0, 1.0),
        "alpha_ppa": trial.suggest_float("alpha_ppa", 20, 3000, log=True),
        "alpha_sr": trial.suggest_float("alpha_sr", 20, 3000, log=True),
        "half_life_weeks": trial.suggest_float("half_life_weeks", 2, 26),
    }


SPACES = {"phase6": suggest, "fcs-two-stage": suggest_fcs_two_stage}
```

`run_search` gets `space: str = "phase6"` as its last parameter, and its line `params = [suggest(t) for t in batch]` becomes `params = [SPACES[space](t) for t in batch]`.

Replace the `grids` construction at the top of `sensitivity` with a new function, and give `sensitivity` a `space` parameter:

```python
def sensitivity_grids(best: dict, space: str = "phase6") -> dict[str, list]:
    scale = (0.25, 0.5, 0.75, 1, 1.5, 2, 4)
    if space == "fcs-two-stage":
        return {
            "fcs_weight": [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
            "alpha_ppa": [best["alpha_ppa"] * f for f in scale],
            "alpha_sr": [best["alpha_sr"] * f for f in scale],
            "half_life_weeks": [2.0, 3.0, 3.7, 5.0, 8.0, 13.0, 26.0],
        }
    grids: dict[str, list] = {
        "alpha_ppa": [best["alpha_ppa"] * f for f in scale],
        "alpha_sr": [best["alpha_sr"] * f for f in scale],
        "half_life_weeks": [0.0, 3, 6, 10, 16, 26],
        "fcs_pooling": ["single", "conference"],
        "gt_enabled": [True, False],
    }
    if best.get("gt_enabled"):
        for q, span in (("q2", 12), ("q3", 10), ("q4", 8)):
            c = best[q]
            grids[q] = sorted({max(7, c + d) for d in range(-span, span + 1, max(1, span // 4))})
    return grids


def sensitivity(config_path: str | None, best: dict, seasons: list[int], n_jobs: int,
                space: str = "phase6") -> dict[str, list[dict]]:
    """One-at-a-time sweeps around the best params."""
    grids = sensitivity_grids(best, space)
    jobs = []
    # ... the rest of the existing body, unchanged from `for name, values in grids.items():`
```

Split `to_toml` so a raw TOML dict can be written, then add `patch_frozen`:

```python
def to_toml(cfg: Config) -> str:
    """Serialize the tunable sections (paths and ingest come from defaults)."""
    return dict_to_toml(cfg.model_dump(mode="json", exclude={"paths", "source"}))


def dict_to_toml(d: dict) -> str:
    def fmt(v):
        # ... the existing body of fmt, unchanged
    lines = ["# Frozen by `cfbr tune`. Do not edit after holdout evaluation.", ""]

    def section(name: str, body: dict):
        # ... the existing body of section, unchanged
    for name, body in d.items():
        section(name, body)
    return "\n".join(lines)


def patch_frozen(base_path: Path, best: dict, path: Path) -> Config:
    """Copy a frozen config file, changing only the keys in ``best``. Sections the base file doesn't
    list stay unlisted, so they keep following pydantic defaults exactly as the base config does."""
    with open(base_path, "rb") as f:
        d = tomllib.load(f)
    model, prep = d.setdefault("model", {}), d.setdefault("prep", {})
    gt = prep.setdefault("garbage_time", {})
    for k, v in best.items():
        if k in ("alpha_ppa", "alpha_sr", "half_life_weeks", "fcs_weight"):
            model[k] = float(v)
        elif k == "fcs_two_stage":
            model[k] = bool(v)
        elif k == "fcs_pooling":
            prep[k] = v
        elif k == "gt_enabled":
            gt["enabled"] = bool(v)
        else:
            raise KeyError(k)
    path.write_text(dict_to_toml(d))
    return load_config(path)


def write_fcs_report(path: Path, best: dict, metrics: dict, v1_metrics: dict, sens: dict, n_trials: int,
                     run_dir: Path) -> None:
    o, v = metrics["overall"], v1_metrics["overall"]
    lines = [
        "# FCS two-stage tuning report", "",
        f"Tuning seasons {', '.join(metrics['by_season'])}; objective = overall MAE of predicted margin. "
        f"{n_trials} Optuna TPE trials (seed 0), search space fixed by DECISIONS \"FCS games, two-stage fit\". "
        "Comparator: the published v1 config (`config/tuned.toml`). This is in-sample; the gates decide.", "",
        "## Best config", "", "| parameter | value |", "|---|---|",
        *[f"| {k} | {_fmtv(val) if not isinstance(val, float) or val >= 10 else f'{val:.3f}'} |"
          for k, val in best.items()], "",
        f"Tuning-season MAE: **{o['mae']:.4f}** v2 vs {v['mae']:.4f} v1.", "",
        "## Sensitivity (MAE on tuning seasons, one parameter varied at a time)", "",
        "| parameter | value: MAE | range |", "|---|---|---|", *describe_sensitivity(sens, o["mae"]), "",
        "## Tuning-season metrics (v2)", "", *_metric_rows(metrics), "",
        f"Run: `{run_dir}`", "",
    ]
    path.write_text("\n".join(lines))
```

- [ ] **Step 4: Add `--space` to the `tune` CLI.** In `src/cfbr/cli.py`, add the parameter `space: Annotated[str, typer.Option(help="phase6 (writes tuned.toml) or fcs-two-stage (writes tuned_v2.toml)")] = "phase6",` to `tune`. Then, right after `cpath = str(config) if config else None`, insert:

```python
    if space == "fcs-two-stage":
        if config is None:
            raise typer.BadParameter("--space fcs-two-stage needs --config config/tuned.toml (the published v1 config)")
        out_dir = cfg.paths.results_dir / "tuning-fcs-two-stage"
        out_dir.mkdir(parents=True, exist_ok=True)
        results = T.run_search(cpath, seasons, trials, jobs, space=space)
        T.summarize_trials(results).write_parquet(out_dir / "trials.parquet")
        best = min(results, key=lambda r: r["mae"])["params"]
        sens = T.sensitivity(cpath, best, seasons, jobs, space=space)
        (out_dir / "search.json").write_text(json.dumps({"best": best, "sensitivity": sens}, indent=2, default=float))
        tuned_path = REPO_ROOT / "config" / "tuned_v2.toml"
        tuned = T.patch_frozen(Path(config), best, tuned_path)
        preds, metrics = run_backtest(tuned, seasons)
        run_dir = write_results(tuned, seasons, preds, metrics, "tuned-v2-tuning")
        T.write_fcs_report(out_dir / "tuning_report.md", best, metrics, run_backtest(cfg, seasons)[1], sens, trials,
                           run_dir)
        typer.echo(f"frozen: {tuned_path}\nreport: {out_dir / 'tuning_report.md'}")
        return
    if space != "phase6":
        raise typer.BadParameter(f"unknown --space {space!r}")
```

(The existing `phase6` path below it stays byte-identical. Check that `Path` is imported in `cli.py`; it is used by `_default_config`'s type hints.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tune.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit the code**

```bash
git add src/cfbr/backtest/tune.py src/cfbr/cli.py tests/test_tune.py
git commit -m "feat: FCS two-stage: tuning space and frozen-config patch"
```

- [ ] **Step 7: Smoke-test the search** (4 trials, 2 jobs; this output is discarded)

Run: `uv run python -c "from cfbr.backtest.tune import run_search; r = run_search('config/tuned.toml', [2019], 4, 2, space='fcs-two-stage'); print([round(x['mae'], 3) for x in r])"`
Expected: four MAE values around 13–14, no exceptions.

- [ ] **Step 8: Run the real search** in the background (10 cores; use 8 jobs)

Run (background): `uv run cfbr tune --space fcs-two-stage --config config/tuned.toml --trials 256 --jobs 8`
Expected: ends with `frozen: .../config/tuned_v2.toml`. Read `results/tuning-fcs-two-stage/tuning_report.md` and record the best `fcs_weight`, `alpha_ppa`, `alpha_sr` and `half_life_weeks`, and the v2 vs v1 tuning MAE.

- [ ] **Step 9: Check `tuned_v2.toml`**

Run: `git diff --no-index config/tuned.toml config/tuned_v2.toml`
Expected: only `[model]` changes (the alphas, `half_life_weeks`, and the new `fcs_two_stage = true`, `fcs_weight = …`) and possibly float re-rounding. No new sections.

- [ ] **Step 10: Commit the frozen candidate**

```bash
git add config/tuned_v2.toml
git commit -m "feat: FCS two-stage: tuned candidate config (tuned_v2.toml)"
```

---

### Task 5: The three gates

**Files:**
- Create: `src/cfbr/backtest/fcs_gate.py`
- Modify: `src/cfbr/cli.py` (new `fcs-gate` command after `calibration-gate`)
- Test: `tests/test_fcs_gate.py` (new)
- Modify: `DECISIONS.md` (result line)

**Interfaces:**
- Consumes: `run_backtest(cfg, seasons, ds) -> (preds, metrics)`, whose preds include `pred_margin`, `spread`, `home`, `away` and `completed`; `paired_gain` and `_slope` from `cfbr.backtest.calib_gate`; `build_rankings(eng, season, week)` from `cfbr.report.rank`.
- Produces: `GATES`, `gate_frame`, `gate_report(df, gate) -> dict`, `ats(df, col) -> dict`, `latest_gate(results_dir, gate) -> dict | None`, `run_fcs_gate(v1, v2, ds, gate) -> Path`. CLI: `cfbr fcs-gate --gate N [--v2 config/tuned_v2.toml]`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_fcs_gate.py`:

```python
import json

import polars as pl

from cfbr.backtest.fcs_gate import ats, gate_report, latest_gate


def _frame(p1, p2, fcs=None, spread=None):
    n = len(p1)
    return pl.DataFrame({"margin": [0.0] * n, "pred_v1": p1, "pred_v2": p2,
                         "fcs_game": fcs or [False] * n, "spread": spread or [None] * n},
                        schema_overrides={"spread": pl.Float64})


def test_gate1_needs_only_a_positive_gain_but_gates_2_and_3_need_the_ci_floor():
    noisy = _frame([2.0, 1.0] * 5, [1.0, 1.8] * 5)   # mean gain +0.1, wide CI
    assert gate_report(noisy, 1)["primary"]["passes"] is True
    assert gate_report(noisy, 2)["primary"]["passes"] is False
    assert gate_report(noisy, 3)["primary"]["passes"] is False
    tight = _frame([2.0] * 10, [1.5, 1.49] * 5)
    assert gate_report(tight, 2)["primary"]["passes"] is True


def test_negative_gain_fails_every_gate():
    worse = _frame([1.0] * 10, [2.0] * 10)
    assert not any(gate_report(worse, g)["primary"]["passes"] for g in (1, 2, 3))


def test_split_by_fcs_game():
    df = _frame([2.0, 2.0, 1.0], [1.0, 1.0, 1.0], fcs=[True, True, False])
    r = gate_report(df, 1)
    assert r["split"]["fbs_fcs"]["n"] == 2 and r["split"]["fbs_fbs"]["n"] == 1
    assert r["split"]["fbs_fcs"]["mae_v1"] == 2.0 and r["split"]["fbs_fcs"]["mae_v2"] == 1.0


def test_ats_skips_pushes_and_unlined_games():
    # Vegas home margin = -spread = 3. Margins 7 (cover), 3 (push), 0 (no cover), unlined.
    df = pl.DataFrame({"margin": [7.0, 3.0, 0.0, 5.0], "p": [5.0, 5.0, 5.0, 5.0],
                       "spread": [-3.0, -3.0, -3.0, None]})
    assert ats(df, "p") == {"n": 2, "wins": 1, "rate": 0.5}


def test_latest_gate_reads_the_newest_run(tmp_path):
    for ts, ok in (("20261108T000000Z", False), ("20261109T000000Z", True)):
        d = tmp_path / f"{ts}-fcs-gate-1"
        d.mkdir()
        (d / "gate.json").write_text(json.dumps({"primary": {"passes": ok}}))
    assert latest_gate(tmp_path, 1)["primary"]["passes"] is True
    assert latest_gate(tmp_path, 2) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_fcs_gate.py -v`
Expected: `ModuleNotFoundError: No module named 'cfbr.backtest.fcs_gate'`.

- [ ] **Step 3: Implement `src/cfbr/backtest/fcs_gate.py`**

```python
"""FCS two-stage gates (DECISIONS "FCS games, two-stage fit", rule committed in 55fb749).

v1 = the published config, v2 = the tuned two-stage candidate, paired by game, each game predicted
point-in-time from its own slot. This module stays as the record of the test whether or not it passes."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from cfbr.backtest.calib_gate import _slope, paired_gain
from cfbr.backtest.harness import run_backtest
from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.model.engine import Engine
from cfbr.report.rank import build_rankings

GATES = {
    1: {"seasons": list(range(2015, 2022)), "max_week": None, "role": "implementation check (in-sample)",
        "ci_floor": None},
    2: {"seasons": list(range(2022, 2026)), "max_week": None, "role": "out of sample", "ci_floor": -0.10},
    3: {"seasons": [2026], "max_week": 10, "role": "2026 regular season through week 10", "ci_floor": -0.10},
}


def gate_frame(v1: Config, v2: Config, ds: Dataset, seasons: list[int], max_week: int | None) -> pl.DataFrame:
    p1, _ = run_backtest(v1, seasons, ds)
    p2, _ = run_backtest(v2, seasons, ds)
    df = (p1.filter(pl.col("completed") & pl.col("margin").is_not_null())
          .select("game_id", "season", "week", "season_type", "home", "away", "home_team", "away_team", "margin",
                  "spread", pred_v1="pred_margin")
          .join(p2.select("game_id", pred_v2="pred_margin"), on="game_id", how="inner"))
    if max_week is not None:
        df = df.filter((pl.col("season_type") == "regular") & (pl.col("week") <= max_week))
    return df.with_columns(fcs_game=pl.col("home").str.starts_with("FCS") | pl.col("away").str.starts_with("FCS"))


def ats(df: pl.DataFrame, col: str) -> dict:
    """Against the closing line (Vegas home margin = -spread); pushes and unlined games skipped."""
    d = df.filter(pl.col("spread").is_not_null())
    line = -d["spread"].to_numpy()
    result, pick = np.sign(d["margin"].to_numpy() - line), np.sign(d[col].to_numpy() - line)
    ok = (result != 0) & (pick != 0)
    n, wins = int(ok.sum()), int((result[ok] == pick[ok]).sum())
    return {"n": n, "wins": wins, "rate": wins / n if n else math.nan}


def gate_report(df: pl.DataFrame, gate: int) -> dict:
    spec = GATES[gate]
    y = df["margin"].to_numpy()
    p1, p2 = df["pred_v1"].to_numpy(), df["pred_v2"].to_numpy()
    e1, e2 = np.abs(y - p1), np.abs(y - p2)
    fcs = df["fcs_game"].to_numpy()
    g = paired_gain(e1, e2)  # mean(e1 - e2): positive favours v2
    passes = bool(g["gain"] > 0 and (spec["ci_floor"] is None or g["ci95"][0] > spec["ci_floor"]))
    split = {}
    for name, mask in (("fbs_fbs", ~fcs), ("fbs_fcs", fcs)):
        split[name] = {**paired_gain(e1[mask], e2[mask]),
                       "mae_v1": float(e1[mask].mean()) if mask.any() else math.nan,
                       "mae_v2": float(e2[mask].mean()) if mask.any() else math.nan}
    return {"gate": gate, **spec, "n_games": int(y.size), "primary": {**g, "passes": passes},
            "mae": {"v1": float(e1.mean()), "v2": float(e2.mean())}, "split": split,
            "ats": {"v1": ats(df, "pred_v1"), "v2": ats(df, "pred_v2")},
            "fcs_slope": {"v1": _slope(p1[fcs], y[fcs]), "v2": _slope(p2[fcs], y[fcs])}}


def top25(cfg: Config, ds: Dataset, season: int, week: int) -> pl.DataFrame:
    rankings, _ = build_rankings(Engine(ds, cfg, last_completed=season - 1), season, week)
    return rankings.select("rank", "team", "rating").head(25)


def gate_markdown(r: dict, v1_model: dict, v2_model: dict, side_by_side: pl.DataFrame | None) -> str:
    def gain(x: dict) -> str:
        return f"{x['gain']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] (n={x['n']})"

    floor = "" if r["ci_floor"] is None else f" and 95% CI lower bound > {r['ci_floor']:+.2f}"
    lines = [f"# FCS two-stage gate {r['gate']}: {r['role']}", "",
             f"Seasons {r['seasons'][0]}–{r['seasons'][-1]}. Paired MAE gain v1 − v2: **{gain(r['primary'])}**. "
             f"Rule: gain > 0{floor} → **{'PASSES' if r['primary']['passes'] else 'FAILS'}**.", "",
             "| slice | MAE v1 | MAE v2 | gain |", "|---|---|---|---|",
             f"| all | {r['mae']['v1']:.3f} | {r['mae']['v2']:.3f} | {gain(r['primary'])} |"]
    for name in ("fbs_fbs", "fbs_fcs"):
        s = r["split"][name]
        lines.append(f"| {name} | {s['mae_v1']:.3f} | {s['mae_v2']:.3f} | {gain(s)} |")
    lines += ["", f"ATS vs close: v1 {r['ats']['v1']['wins']}/{r['ats']['v1']['n']} = {r['ats']['v1']['rate']:.3f}; "
                  f"v2 {r['ats']['v2']['wins']}/{r['ats']['v2']['n']} = {r['ats']['v2']['rate']:.3f}.",
              f"Slope of margin on prediction, FBS–FCS games: v1 {r['fcs_slope']['v1']:.3f}, "
              f"v2 {r['fcs_slope']['v2']:.3f}.", "",
              "| setting | v1 | v2 |", "|---|---|---|"]
    for k in ("alpha_ppa", "alpha_sr", "half_life_weeks", "fcs_two_stage", "fcs_weight"):
        lines.append(f"| {k} | {v1_model[k]} | {v2_model[k]} |")
    if side_by_side is not None:
        lines += ["", "## 2026 after week 10, top 25", "", "| rank | v1 | rating | v2 | rating |",
                  "|---|---|---|---|---|"]
        for row in side_by_side.iter_rows(named=True):
            lines.append(f"| {row['rank']} | {row['team']} | {row['rating']:.2f} | {row['team_v2']} | "
                         f"{row['rating_v2']:.2f} |")
    return "\n".join(lines) + "\n"


def latest_gate(results_dir: Path, gate: int) -> dict | None:
    runs = sorted(results_dir.glob(f"*-fcs-gate-{gate}"))
    return json.loads((runs[-1] / "gate.json").read_text()) if runs else None


def run_fcs_gate(v1: Config, v2: Config, ds: Dataset, gate: int) -> Path:
    spec = GATES[gate]
    df = gate_frame(v1, v2, ds, spec["seasons"], spec["max_week"])
    rep = gate_report(df, gate)
    rep["models"] = {"v1": v1.model.model_dump(), "v2": v2.model.model_dump()}
    side = None
    if gate == 3:
        side = top25(v1, ds, 2026, 10).join(
            top25(v2, ds, 2026, 10).rename({"team": "team_v2", "rating": "rating_v2"}), on="rank")
    out = v1.paths.results_dir / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-fcs-gate-{gate}")
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "frame.parquet")
    (out / "gate.json").write_text(json.dumps(rep, indent=2, default=float))
    (out / "gate.md").write_text(gate_markdown(rep, rep["models"]["v1"], rep["models"]["v2"], side))
    return out
```

- [ ] **Step 4: Add the CLI** in `src/cfbr/cli.py`, after `calibration_gate`:

```python
@app.command("fcs-gate")
def fcs_gate(
    gate: Annotated[int, typer.Option(help="1 = 2015–2021, 2 = 2022–2025, 3 = 2026 through week 10")],
    v2: Annotated[Optional[Path], typer.Option(help="Candidate config (default config/tuned_v2.toml)")] = None,
    config: ConfigOpt = None,
) -> None:
    """FCS two-stage gate (DECISIONS "FCS games, two-stage fit"): paired MAE gain of the tuned two-stage
    candidate over the published config. Gates run in order; each needs the previous one passed.
    Writes results/<ts>-fcs-gate-N/."""
    import json

    from cfbr.backtest.fcs_gate import GATES, latest_gate, run_fcs_gate
    from cfbr.config import REPO_ROOT
    from cfbr.data import Dataset

    if gate not in GATES:
        raise typer.BadParameter(f"--gate must be one of {sorted(GATES)}")
    v1_cfg = load_config(config or REPO_ROOT / "config" / "tuned.toml")
    v2_cfg = load_config(v2 or REPO_ROOT / "config" / "tuned_v2.toml")
    if gate > 1:
        prev = latest_gate(v1_cfg.paths.results_dir, gate - 1)
        if not prev or not prev["primary"]["passes"]:
            raise typer.BadParameter(f"gate {gate - 1} has not passed; gates run in order")
    out = run_fcs_gate(v1_cfg, v2_cfg, Dataset.load(v1_cfg), gate)
    p = json.loads((out / "gate.json").read_text())["primary"]
    typer.echo(f"gate {gate}: v1 − v2 paired MAE gain {p['gain']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] "
               f"over {p['n']} games: {'PASSES' if p['passes'] else 'FAILS'}. {out}")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fcs_gate.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/cfbr/backtest/fcs_gate.py src/cfbr/cli.py tests/test_fcs_gate.py
git commit -m "feat: FCS two-stage: gate command (three pre-registered gates)"
```

- [ ] **Step 7: Run gate 1**

Run: `uv run cfbr fcs-gate --gate 1`
Expected: one line ending `PASSES` or `FAILS`. Read the run's `gate.md`. **If it FAILS, skip to Step 11.**

- [ ] **Step 8: Run gate 2**

Run: `uv run cfbr fcs-gate --gate 2`
Read `gate.md`. **If it FAILS, skip to Step 11.**

- [ ] **Step 9: Run gate 3**

Run: `uv run cfbr fcs-gate --gate 3`
Read `gate.md`, including the week-10 top 25 side by side.

- [ ] **Step 10: Wake Forest / Merrimack check under v2** (reported, not gated). Write the scratch script `/private/tmp/claude-501/-Users-abtreece-Projects-cfbr/cbe622d2-8c10-41ae-8e4d-3a378152fc0e/scratchpad/wake_check.py`:

```python
import dataclasses
import sys

import polars as pl

from cfbr.config import load_config
from cfbr.data import Dataset
from cfbr.model.engine import Engine
from cfbr.report.rank import build_rankings

for path in sys.argv[1:]:
    cfg = load_config(path)
    ds = Dataset.load(cfg)
    gid = ds.games.filter((pl.col("season") == 2026) & (pl.col("home_team") == "Wake Forest")
                          & (pl.col("away_team") == "Merrimack"))["game_id"].item()
    without = dataclasses.replace(ds, scrimmage=ds.scrimmage.filter(pl.col("game_id") != gid))
    for label, d in (("with", ds), ("without", without)):
        r, _ = build_rankings(Engine(d, cfg, last_completed=2025), 2026, 10)
        w = r.filter(pl.col("team") == "Wake Forest")
        print(f"{path} {label:8s} Merrimack: Wake {w['rating'].item():.2f} (#{w['rank'].item()})")
```

Run: `uv run python /private/tmp/claude-501/-Users-abtreece-Projects-cfbr/cbe622d2-8c10-41ae-8e4d-3a378152fc0e/scratchpad/wake_check.py config/tuned.toml config/tuned_v2.toml`
Expected v1 lines: 23.22 (#3) with, 15.69 (#13) without (the numbers from this morning's diagnosis). The v2 gap should be much smaller.

- [ ] **Step 11: Record the result in `DECISIONS.md`** (one line, appended), in the house format:

```
- FCS games, two-stage fit result (<YYYY-MM-DD>), judged by the rule fixed in 55fb749: <ADOPTED | NOT adopted>. Tuned (results/tuning-fcs-two-stage): fcs_weight <x>, alpha_ppa <x>, alpha_sr <x>, half_life_weeks <x> (v1 303.3 / 90.75 / 3.70). Gate 1 2015–2021 gain <g> [<lo>, <hi>] over <n> games (<pass/fail>); gate 2 2022–2025 <…>; gate 3 2026 through week 10 <…>. FBS–FCS vs FBS–FBS split, ATS and pooled-FCS slope per gate in results/<run ids>. Wake Forest after week 10: v1 23.22 with Merrimack / 15.69 without; v2 <x> / <x> — Model (FCS)
```

Fill every `<…>` from the gate outputs. For gates not run because an earlier one failed, write "not run (gate N failed)".

```bash
git add DECISIONS.md
git commit -m "docs: FCS two-stage fit — gate results"
```

- [ ] **Step 12: Checkpoint — report to the user** with the three gate lines, the tuned values, the FBS–FCS split and the Wake check. **If any gate failed:** stop here. The code ships with `fcs_two_stage` off, `tuned_v2.toml` stays as the record, and Task 7 opens the PR without Task 6.

---

### Task 6: Adoption — config swap, Method page, verdict split, restate command (only if all three gates passed)

**Files:**
- Rename: `config/tuned.toml` → `config/tuned_v1.toml`; `config/tuned_v2.toml` → `config/tuned.toml`
- Modify: `config/tuned.toml` (add a `[bets]` section with `verdict_split_at`)
- Modify: `src/cfbr/config.py` (class `Bets`: `verdict_split_at`)
- Modify: `src/cfbr/bets/verdict.py`, `src/cfbr/bets/ledger.py:~593-612`, `src/cfbr/web/templates/bets.html:~123`
- Modify: `src/cfbr/web/templates/method.html:14`
- Create: `src/cfbr/report/restate.py`
- Modify: `src/cfbr/cli.py` (extract `_run_shadows`, add `restate`)
- Test: `tests/test_fcs_two_stage.py`, `tests/test_verdict.py`, `tests/test_web.py`, `tests/test_restate.py` (new)
- Modify: `DECISIONS.md` (adoption and superseding-shadow line)

**Interfaces:**
- Produces: `Bets.verdict_split_at: str | None = None` (ISO-8601 with offset); `verdict()` result gets a `"split"` key, either `None` or `{"at": str, "before": {"n", "mean"}, "after": {"n", "mean"}}`.
- Produces: `archive_outputs(season_dir: Path, through_week: int, name: str = "_v1") -> list[int]`; `restate_season(cfg, season: int, weeks: list[int]) -> None`; CLI `cfbr restate --season S --through-week N`.

- [ ] **Step 1: Swap the configs**

```bash
git mv config/tuned.toml config/tuned_v1.toml
git mv config/tuned_v2.toml config/tuned.toml
```

Append to `config/tuned.toml`:

```toml
[bets]
# Paper bets recorded at or after this time used the FCS two-stage model (DECISIONS "FCS games, two-stage fit").
verdict_split_at = "2026-11-09T00:00:00+00:00"
```

(The last v1 weekly run is 2026-11-08 3pm CT = 21:00 UTC. If the deploy slips past the 2026-11-15 9am CT cron, change this to the deploy time before merging.)

- [ ] **Step 2: Update the config tests.** In `tests/test_fcs_two_stage.py`, replace `test_config_defaults_leave_behaviour_unchanged` with:

```python
def test_config_files():
    for path in (None, "config/tuned_v1.toml"):
        m = load_config(path).model
        assert m.fcs_two_stage is False and m.fcs_weight == 1.0
    v1, pub = load_config("config/tuned_v1.toml"), load_config("config/tuned.toml")
    assert pub.model.fcs_two_stage is True and 0.0 <= pub.model.fcs_weight <= 1.0
    assert (v1.model.alpha_ppa, v1.model.alpha_sr, v1.model.half_life_weeks) == (303.298, 90.7544, 3.6978)
    assert pub.baseline == v1.baseline and pub.bets.verdict_min_bets == v1.bets.verdict_min_bets
```

- [ ] **Step 3: Write the failing verdict and Method tests**

Append to `tests/test_verdict.py`:

```python
def test_verdict_split_is_context_only():
    rows = [_row(i, "won", "snapshot", "home", -3.0, -7.0) for i in range(3)]          # +3.5% each
    rows += [_row(i + 10, "lost", "snapshot", "home", -3.0, -3.0) for i in range(2)]   # -4.5% each
    for r, ts in zip(rows, ["2026-11-01T15:00:00+00:00"] * 3 + ["2026-11-15T15:00:00+00:00"] * 2):
        r["recorded_at"] = ts
    led = _ledger(rows)
    plain = verdict(led, Bets(pricing_books=[], verdict_min_bets=3), FakeTable())
    split = verdict(led, Bets(pricing_books=[], verdict_min_bets=3, verdict_split_at="2026-11-09T00:00:00+00:00"),
                    FakeTable())
    assert plain["split"] is None
    assert {k: split[k] for k in ("n", "mean", "status")} == {k: plain[k] for k in ("n", "mean", "status")}
    assert split["split"]["before"]["n"] == 3 and math.isclose(split["split"]["before"]["mean"], 0.035)
    assert split["split"]["after"]["n"] == 2 and math.isclose(split["split"]["after"]["mean"], -0.045)
```

Append to `tests/test_web.py`:

```python
def test_method_page_describes_the_fcs_two_stage_fit(web_cfg):
    web_cfg.prep.fcs_pooling = "conference"
    web_cfg.model.fcs_two_stage = True
    web_cfg.model.fcs_weight = 0.3
    _write_season(web_dir(web_cfg), datetime.now(UTC))
    html = " ".join(TestClient(create_app(web_cfg)).get("/method").text.split())
    assert "counts for 30% of a normal game" in html
    assert "restated in November 2026" in html
```

Run: `uv run pytest tests/test_verdict.py tests/test_web.py -v -k "split or fcs_two_stage"`
Expected: FAIL (`verdict_split_at` isn't a field; the Method text is missing).

- [ ] **Step 4: Implement the verdict split.** `src/cfbr/config.py`, class `Bets`, after `verdict_min_bets`:

```python
    # When the published model changed mid-season (DECISIONS "FCS games, two-stage fit"): bets recorded
    # at or after this ISO time are shown separately, as context only; the verdict still pools all bets.
    verdict_split_at: str | None = None
```

`src/cfbr/bets/verdict.py`: add `from datetime import datetime`, then add:

```python
def _split(ledger: pl.DataFrame, table: PricingTable, split_at: str) -> dict:
    """Context only: closing-implied ROI before and after a mid-season model switch."""
    cut = datetime.fromisoformat(split_at)
    roi = closing_implied_roi(ledger, table)
    after = pl.Series([r is not None and datetime.fromisoformat(r) >= cut for r in ledger["recorded_at"]])
    out = {"at": split_at}
    for name, sel in (("before", ~after), ("after", after)):
        v = roi.filter(sel).drop_nulls()
        out[name] = {"n": v.len(), "mean": float(v.mean()) if v.len() else math.nan}
    return out
```

In `verdict()`, add `"split": None` to the initial `out` dict. Then, right after `out["n"] = n` (before the `if n == 0` early return), add:

```python
    if cfg.verdict_split_at:
        out["split"] = _split(ledger, table, cfg.verdict_split_at)
```

`src/cfbr/bets/ledger.py`: after the `lines += [f"Bets with a snapshot close: …"]` line inside the `else:` branch, add:

```python
        if v["split"]:
            s = v["split"]
            lines += [f"Context only (the verdict pools all bets): before the {s['at'][:10]} model switch "
                      f"{s['before']['n']} bets, mean {_fmt(s['before']['mean'], pct=True)}; after it "
                      f"{s['after']['n']} bets, mean {_fmt(s['after']['mean'], pct=True)}.", ""]
```

`src/cfbr/web/templates/bets.html`: after the `{% if verdict.n %}…{% endif %}` block (line ~124), add:

```html
{% if verdict.split %}<p class="muted">Context only, the verdict pools all bets: before the {{ verdict.split.at[:10] }} model switch {{ verdict.split.before.n }} bets, mean {{ verdict.split.before.mean|pct }}; after it {{ verdict.split.after.n }} bets, mean {{ verdict.split.after.mean|pct }}.</p>{% endif %}
```

`src/cfbr/web/bets.py:518` (`/api/bets`): `clean_summary` only cleans top-level NaN. Change that call to `clean_summary({**v, "split": _clean_split(v["split"])})` with `v = verdict(...)` bound first, and add:

```python
def _clean_split(s: dict | None) -> dict | None:
    if s is None:
        return None
    return {"at": s["at"], **{k: clean_summary(s[k]) for k in ("before", "after")}}
```

- [ ] **Step 5: Implement the Method text.** `src/cfbr/web/templates/method.html:14`: replace the FCS `<li>` with:

```html
  <li>FCS opponents are pooled {% if cfg.prep.fcs_pooling == "conference" %}by conference, one rated team per FCS conference{% else %}into a single rated team{% endif %}.{% if cfg.model.fcs_two_stage %} A game against an FCS opponent counts for {{ "%.0f"|format(cfg.model.fcs_weight * 100) }}% of a normal game in an FBS team's rating. FCS ratings are fit separately, against FBS ratings held fixed, and are used only to predict those games. 2026 ratings were restated in November 2026 when this changed; the original outputs are archived.{% endif %}</li>
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_verdict.py tests/test_web.py tests/test_fcs_two_stage.py -v`
Expected: all PASS. `/api/bets` tests still pass (JSON has no NaN).

- [ ] **Step 7: Commit**

```bash
git add config/ src/cfbr/config.py src/cfbr/bets/verdict.py src/cfbr/bets/ledger.py src/cfbr/web/ tests/test_fcs_two_stage.py tests/test_verdict.py tests/test_web.py
git commit -m "feat: FCS two-stage: publish tuned config; Method text; verdict before/after context"
```

- [ ] **Step 8: Write the failing restate tests** — create `tests/test_restate.py`:

```python
import pytest

from cfbr.report.restate import archive_outputs


def test_archive_moves_week_dirs_and_accuracy_only(tmp_path):
    for w in ("w06", "w07", "w11"):
        (tmp_path / w).mkdir()
        (tmp_path / w / "rankings.csv").write_text("x")
    for f in ("accuracy.csv", "bets.csv", "ledger_chain.jsonl", "prior_comparison.csv"):
        (tmp_path / f).write_text("x")
    weeks = archive_outputs(tmp_path, through_week=10)
    assert weeks == [6, 7]
    assert (tmp_path / "_v1" / "w06" / "rankings.csv").exists() and (tmp_path / "_v1" / "accuracy.csv").exists()
    assert not (tmp_path / "w06").exists() and (tmp_path / "w11").exists()
    for f in ("bets.csv", "ledger_chain.jsonl", "prior_comparison.csv"):
        assert (tmp_path / f).exists()


def test_archive_refuses_to_overwrite(tmp_path):
    (tmp_path / "w06").mkdir()
    archive_outputs(tmp_path, through_week=10)
    (tmp_path / "w06").mkdir()
    with pytest.raises(FileExistsError):
        archive_outputs(tmp_path, through_week=10)
```

Run: `uv run pytest tests/test_restate.py -v`
Expected: `ModuleNotFoundError: No module named 'cfbr.report.restate'`.

- [ ] **Step 9: Implement `src/cfbr/report/restate.py`**

```python
"""Restate a live season under the published config (DECISIONS "FCS games, two-stage fit").

Moves the existing weekly outputs aside (never deletes them), then rebuilds rankings, the accuracy
line and the season simulation for every week that had been published. The paper-bet ledger and
other season-level files are never touched."""

from __future__ import annotations

import logging
from pathlib import Path

from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.model.engine import Engine
from cfbr.report.rank import append_accuracy, run_rank, season_accuracy

log = logging.getLogger(__name__)


def archive_outputs(season_dir: Path, through_week: int, name: str = "_v1") -> list[int]:
    """Move out/{season}/wNN/ (NN <= through_week) and accuracy.csv into season_dir/name/.
    Returns the archived week numbers, i.e. the weeks to rebuild. Refuses if the archive exists."""
    dest = season_dir / name
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; refusing to overwrite an archive")
    weeks = sorted(int(p.name[1:]) for p in season_dir.glob("w[0-9][0-9]")
                   if p.is_dir() and int(p.name[1:]) <= through_week)
    dest.mkdir(parents=True)
    for w in weeks:
        (season_dir / f"w{w:02d}").rename(dest / f"w{w:02d}")
    acc = season_dir / "accuracy.csv"
    if acc.exists():
        acc.rename(dest / "accuracy.csv")
    return weeks


def restate_season(cfg: Config, season: int, weeks: list[int]) -> Engine:
    ds = Dataset.load(cfg)
    eng = Engine(ds, cfg, last_completed=season - 1)
    acc_path = cfg.paths.out_dir / str(season) / "accuracy.csv"
    for w in weeks:
        log.info("restating %d week %d", season, w)
        run_rank(cfg, season, w, last_completed=season - 1, ds=ds)
        append_accuracy(acc_path, season, w, season_accuracy(eng, season, w))
        if cfg.sim.enabled:
            from cfbr.sim.report import run_sim

            run_sim(eng, season, w)
    return eng
```

- [ ] **Step 10: Extract the shadow block and add `restate` to `src/cfbr/cli.py`.** Move the body of `_weekly_run` from `shadow_failed = False` through the end of the calibration-v3 `try/except` into a new function, unchanged except that it returns the flag:

```python
def _run_shadows(cfg, ds, eng, season: int) -> bool:
    """Prior, portal, totals and calibration shadows; each isolated. Returns True if any failed."""
    shadow_failed = False
    plain_shadow_failed = False
    # ... the moved lines, verbatim ...
    return shadow_failed
```

In `_weekly_run`, replace the moved lines with `shadow_failed = _run_shadows(cfg, ds, eng, season)`. Then add the command:

```python
@app.command()
def restate(
    season: Annotated[int, typer.Option()],
    through_week: Annotated[int, typer.Option(help="Last completed regular-season week")],
    config: ConfigOpt = None,
) -> None:
    """Restate a live season under the published config (DECISIONS "FCS games, two-stage fit"): moves
    out/{season}/wNN/ and accuracy.csv to out/{season}/_v1/, re-ranks those weeks, rebuilds the accuracy
    line, season sims, shadows and the web season. Never touches the paper-bet ledger. Server only."""
    from cfbr.report.restate import archive_outputs, restate_season
    from cfbr.web.build import build_season, web_dir

    cfg = _default_config(config)
    log.info("config: %s", cfg.source)
    weeks = archive_outputs(cfg.paths.out_dir / str(season), through_week)
    eng = restate_season(cfg, season, weeks)
    failed = _run_shadows(cfg, eng.ds, eng, season)
    web_out = build_season(eng, season, web_dir(cfg))
    typer.echo(f"restated {season} weeks {weeks}; archive {cfg.paths.out_dir / str(season) / '_v1'}; web {web_out}")
    if failed:
        raise typer.Exit(code=6)
```

- [ ] **Step 11: Run the tests**

Run: `uv run pytest tests/test_restate.py -v`
Run: `uv run pytest -q`
Expected: all pass. (`weekly` behaves the same; its shadow block was only moved.)

- [ ] **Step 12: Dry-run the restatement locally on a copy** (proves the command end to end without touching the real `out/`)

```bash
mkdir -p /private/tmp/claude-501/-Users-abtreece-Projects-cfbr/cbe622d2-8c10-41ae-8e4d-3a378152fc0e/scratchpad/restate-dry/out/2026
```

```bash
scp -q -r 'javelin:~/cfb-ratings/out/2026/w0[6-9]' 'javelin:~/cfb-ratings/out/2026/w10' 'javelin:~/cfb-ratings/out/2026/accuracy.csv' 'javelin:~/cfb-ratings/out/2026/*_comparison.csv' /private/tmp/claude-501/-Users-abtreece-Projects-cfbr/cbe622d2-8c10-41ae-8e4d-3a378152fc0e/scratchpad/restate-dry/out/2026/
```

Write `/private/tmp/claude-501/-Users-abtreece-Projects-cfbr/cbe622d2-8c10-41ae-8e4d-3a378152fc0e/scratchpad/restate-dry/dry.toml`: the contents of `config/tuned.toml` plus a `[paths]` section setting `out_dir` to the absolute `.../restate-dry/out` and `db_path` to the worktree's `data/cfb.duckdb` (absolute paths).

Run: `uv run cfbr restate --season 2026 --through-week 10 --config /private/tmp/claude-501/-Users-abtreece-Projects-cfbr/cbe622d2-8c10-41ae-8e4d-3a378152fc0e/scratchpad/restate-dry/dry.toml`
Expected: `restated 2026 weeks [6, 7, 8, 9, 10] …`, and `_v1/` holds the five originals. Then check the invariant: `rankings.csv` in the new `w10/` matches the gate-3 v2 top 25 (same teams, ranks and ratings).

- [ ] **Step 13: Commit**

```bash
git add src/cfbr/report/restate.py src/cfbr/cli.py tests/test_restate.py
git commit -m "feat: FCS two-stage: restate command for a live season"
```

- [ ] **Step 14: DECISIONS lines** (append two):

```
- FCS two-stage adopted mid-2026 (<YYYY-MM-DD>): config/tuned.toml is the v2 candidate; the v1 config is kept as config/tuned_v1.toml so 2026 v1 numbers stay reproducible. 2026 restated with `cfbr restate` on deploy: weekly outputs and accuracy.csv archived to out/2026/_v1/, every published week re-ranked. The paper-bet ledger is untouched; `[bets].verdict_split_at` marks the switch, and bets.md and /bets show the before/after closing-implied ROI as context only, while the verdict keeps one series. The season sim's tau stays 6.0 (chosen under v1; out of scope) — Model (FCS)
- Shadow bases moved (supersedes the "every other setting frozen" wording of the roster-prior, portal, calibration v2 and v3 rules, for the base only): from the FCS two-stage adoption, both arms of each shadow are computed on the v2 base (alpha, half-life, FCS handling). The comparison code rebuilds both arms from the published config on every run, so window games already scored are rescored automatically. Thresholds, windows and decision dates are unchanged. Totals v2 doesn't use the margin ratings and is unaffected — Model (FCS)
```

```bash
git add DECISIONS.md
git commit -m "docs: FCS two-stage adopted; shadow bases moved"
```

---

### Task 7: PR, deploy and restatement on javelin

**Files:** none (process)

- [ ] **Step 1: Full verification**

Run: `uv run pytest -q`
Expected: all pass. Paste the summary line in the PR.

- [ ] **Step 2: Update the spec status line** in `docs/superpowers/specs/2026-11-08-fcs-two-stage-design.md` (`Status: rule fixed, not built.` → `Status: built; <adopted | not adopted> <YYYY-MM-DD>, see DECISIONS.md.`) and commit (`docs: FCS two-stage spec status`).

- [ ] **Step 3: Run `check-yourself` on the PR body before posting** (user global rule for public-facing text).

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/fcs-two-stage
```

```bash
gh pr create --title "feat: FCS games count for almost nothing (two-stage fit)" --body-file /private/tmp/claude-501/-Users-abtreece-Projects-cfbr/cbe622d2-8c10-41ae-8e4d-3a378152fc0e/scratchpad/pr_body.md
```

The PR body covers: the problem (Wake/Merrimack), the design, the rule commit `55fb749`, the tuned values, the three gate lines with run ids, the week-10 top-25 change, and the restatement plan. It ends with the attribution line from the session's system reminder (`🤖 Generated with [Claude Code](https://claude.com/claude-code)`); commits carry no Co-Authored-By per the user's CLAUDE.md.

- [ ] **Step 5: Confirm CI passes** (`gh pr checks`) and resolve any review threads.

- [ ] **Step 6: STOP — ask the user** before merging and deploying (stop-and-ask: publishing/deploying). State: what deploy changes on the site immediately (`deploy.sh` rebuilds `out/web/` under the new config), and that `cfbr restate` must follow right away to bring `out/2026/wNN/`, `accuracy.csv`, sims and shadows into line.

- [ ] **Step 7 (after approval): Merge and deploy**

```bash
gh pr merge --squash --delete-branch
```

```bash
deploy/deploy.sh
```

Expected: deploy passes its tests and health checks on javelin.

- [ ] **Step 8 (after approval): Restate on javelin.** It must not overlap a cron run. `restate` doesn't take the weekly lock, so check that no `cfbr weekly` is running first:

```bash
ssh javelin 'pgrep -af "cfbr weekly" || echo idle'
```

```bash
ssh javelin 'cd ~/cfb-ratings && ~/.local/bin/uv run cfbr restate --season 2026 --through-week 10'
```

Expected: `restated 2026 weeks [6, 7, 8, 9, 10] …`.

- [ ] **Step 9: Verify live**

```bash
ssh javelin 'cd ~/cfb-ratings && ls out/2026/_v1 && head -12 out/2026/w10/rankings.csv'
```

Expected: `_v1/` has `w06…w10` and `accuracy.csv`; the new `w10` top matches gate 3's v2 column. Load `/rankings`, `/method` and `/bets` on the site and confirm the Method sentence and the verdict context line.

- [ ] **Step 10: Clean up** after merge is confirmed (`gh pr view --json state` = MERGED): remove the worktree and the local branch.
