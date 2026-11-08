# What Moved Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split each team's weekly rating change into own game / older plays fading / other games, show it with per-game context on the team page and as rankings tooltips, and move the Prior column out of "More columns".

**Architecture:** `Engine._fit` gets an optional play mask and keeps each fit's home coefficient. A new `report/moves.py` computes the three parts (leave-own-game refits through the published fit path) and per-game efficiency vs expectation. `cfbr rank` writes `moves.csv`; the web build adds the move columns to `ratings.parquet` (live season only) and the efficiency columns to `games.parquet`. Model-derived columns join the existing free-viewer lock lists.

**Tech Stack:** Python 3.12, numpy, polars, FastAPI/Jinja, pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-11-08-what-moved-design.md` (9cb7fa2). Display only: published ratings, ranks and predictions must not change.
- `Engine._fit(..., mask=None)` must be byte-identical to today (the published pipeline goes through it).
- `move_decay + move_own + move_others == rating_change` exactly (before rounding) for every FBS team.
- Point-in-time: expected per-play edges come from the game's pre-game slot snapshot.
- Model-derived values are locked for free viewers in a live season, like ratings.
- Moves are computed for the live season only in the web build (cost ~2 s/week).
- Local work uses the production DB copy symlinked at `data/cfb.duckdb`; no API calls; don't run `weekly`, `bets`, `odds-snapshot`, `backup`, `check`.
- Separate Bash calls instead of `&&` chains; conventional commits; no Co-Authored-By.

## Deviations from the spec (simplifications; flagged to the user)

1. **Storage:** instead of two new files (`moves.parquet`, `move_games.parquet`), the team move columns go into `ratings.parquet` and the game context into `games.parquet`. The existing store loads them with no changes, and the existing lock lists (`MODEL_RATING_COLS`, `MODEL_GAME_COLS`) cover them.
2. **Earlier weeks:** shown through the three new columns in the existing week-by-week table instead of a separate week selector. The "What moved" section shows the latest week.
3. **`moves.csv`** holds team rows only. Game context lives in the web games data.
4. **Actual per-play stats** (`home_ppa`, `away_ppa`, `home_sr`, `away_sr`) are public facts from CFBD plays, not model output, so they stay visible to free viewers. Only the expected edges are locked.

## File map

| File | Change |
|---|---|
| `src/cfbr/model/engine.py` | `_fit(mask=)`, `home_ppa`/`home_sr` snapshot columns |
| `src/cfbr/report/moves.py` (new) | `rating_moves`, `game_efficiency`, `moves_csv_frame` |
| `src/cfbr/report/rank.py` | `run_rank` writes `moves.csv` |
| `src/cfbr/web/build.py` | move columns in `season_ratings`, efficiency columns in `season_games` |
| `src/cfbr/web/app.py` | lock lists, `team_games` columns, `what_moved()`, team page/API context |
| `src/cfbr/web/templates/team.html`, `rankings.html`, `method.html` | display |
| `tests/test_moves.py` (new), `tests/test_leakage.py`, `tests/test_web.py` | tests |
| `DECISIONS.md`, spec status | docs |

---

### Task 1: Masked fit and home coefficients in the engine

**Files:**
- Modify: `src/cfbr/model/engine.py` (`_fit`)
- Test: `tests/test_moves.py` (new)

**Interfaces:**
- Produces: `Engine._fit(season: int, prior: pl.DataFrame | None, as_of=None, mask: np.ndarray | None = None) -> pl.DataFrame`. `mask` is a bool array of length `arrays(season).n_before(as_of)`; True keeps a play. The output gains columns `home_ppa`, `home_sr` (the fit's home coefficients, the same value on every row).

- [ ] **Step 1: Write the failing tests** — create `tests/test_moves.py`:

```python
"""What-moved breakdown (spec 2026-11-08-what-moved-design.md). Real-data tests skip without the DB."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cfbr.model.engine import RATING_COLS, Engine, _ns
from cfbr.model.ridge import fit_efficiency_arrays, fit_fcs_stage2, play_weights, recency_weights


@pytest.fixture(params=["default", "two_stage"])
def model_cfg(request, cfg, cfg_two_stage):
    return cfg if request.param == "default" else cfg_two_stage


def test_unmasked_fit_is_unchanged_and_carries_home_coefficients(model_cfg, dataset):
    eng = Engine(dataset, model_cfg, last_completed=2025)
    as_of = eng.slot_as_of(2018, "r07")
    k = eng.arrays(2018).n_before(as_of)
    plain = eng._fit(2018, eng.priors(2018), as_of)
    full_mask = eng._fit(2018, eng.priors(2018), as_of, mask=np.ones(k, bool))
    assert_frame_equal(plain, full_mask)
    assert {"home_ppa", "home_sr"} <= set(plain.columns)
    assert plain["home_ppa"].n_unique() == 1 and plain["home_sr"].n_unique() == 1


def test_masked_fit_equals_explicit_fit_on_remaining_plays(model_cfg, dataset):
    eng = Engine(dataset, model_cfg, last_completed=2025)
    m, a = model_cfg.model, eng.arrays(2018)
    as_of = eng.slot_as_of(2018, "r07")
    k = a.n_before(as_of)
    keep = np.ones(k, bool)
    keep[k // 2:k // 2 + 500] = False
    got = eng._fit(2018, eng.priors(2018), as_of, mask=keep)
    pr = pl.DataFrame({"team": a.teams}).join(eng.priors(2018), on="team", how="left").fill_null(0.0)
    p = {c: pr[c].to_numpy() for c in RATING_COLS}
    w = play_weights(a, k, _ns(as_of), m)
    w = None if w is None else w[keep]
    off, de, h = a.off[:k][keep], a.de[:k][keep], a.home[:k][keep]
    ref = fit_efficiency_arrays(off, de, h, a.ppa[:k][keep], a.teams, m.alpha_ppa, p["o_ppa"], p["d_ppa"], w, a.center)
    if m.fcs_two_stage:
        rec = recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)
        ref = fit_fcs_stage2(off, de, h, a.ppa[:k][keep], ~a.center, m.alpha_ppa, ref, p["o_ppa"], p["d_ppa"],
                             None if rec is None else rec[keep])
    assert np.abs(got["o_ppa"].to_numpy() - ref.o).max() < 1e-9
    assert np.abs(got["d_ppa"].to_numpy() - ref.d).max() < 1e-9
    assert got["home_ppa"][0] == pytest.approx(ref.home, abs=1e-12)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_moves.py -v`
Expected: FAIL (`_fit() got an unexpected keyword argument 'mask'`).

- [ ] **Step 3: Implement in `src/cfbr/model/engine.py`.** Replace `_fit` with:

```python
    def _fit(self, season: int, prior: pl.DataFrame | None, as_of=None, mask: np.ndarray | None = None) -> pl.DataFrame:
        """Fit both components on season plays with kickoff < as_of (all plays when None).
        ``mask`` (bool, length n_before(as_of)) keeps only some of those plays; the what-moved
        breakdown uses it for leave-one-game-out refits through this same path."""
        a = self.arrays(season)
        k = a.n_before(as_of)
        teams = a.teams
        if prior is not None:
            pr = pl.DataFrame({"team": teams}).join(prior, on="team", how="left").fill_null(0.0)
            p = {c: pr[c].to_numpy() for c in RATING_COLS}
        else:
            p = {c: None for c in RATING_COLS}
        m = self.cfg.model
        w = play_weights(a, k, None if as_of is None else _ns(as_of), m)
        sel = slice(None) if mask is None else np.asarray(mask, dtype=bool)
        off, de, h = a.off[:k][sel], a.de[:k][sel], a.home[:k][sel]
        ppa, success = a.ppa[:k][sel], a.success[:k][sel]
        w = None if w is None else w[sel]
        f_ppa = fit_efficiency_arrays(off, de, h, ppa, teams, m.alpha_ppa, p["o_ppa"], p["d_ppa"], w, a.center)
        f_sr = fit_efficiency_arrays(off, de, h, success, teams, m.alpha_sr, p["o_sr"], p["d_sr"], w, a.center)
        if m.fcs_two_stage:
            # Stage 2 (DECISIONS "FCS games, two-stage fit"): FCS rows from FCS plays at full recency
            # weight, Stage 1 FBS ratings held fixed. w_off/w_def below stay the Stage 1 weights.
            rec = None if as_of is None else recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)
            rec = None if rec is None else rec[sel]
            fcs = ~a.center
            f_ppa = fit_fcs_stage2(off, de, h, ppa, fcs, m.alpha_ppa, f_ppa, p["o_ppa"], p["d_ppa"], rec)
            f_sr = fit_fcs_stage2(off, de, h, success, fcs, m.alpha_sr, f_sr, p["o_sr"], p["d_sr"], rec)
        n = len(teams)
        w_off = np.bincount(off, weights=w, minlength=n).astype(float)
        w_def = np.bincount(de, weights=w, minlength=n).astype(float)
        return pl.DataFrame({
            "team": teams, "o_ppa": f_ppa.o, "d_ppa": f_ppa.d, "o_sr": f_sr.o, "d_sr": f_sr.d,
            "n_off": f_ppa.n_off, "n_def": f_ppa.n_def, "w_off": w_off, "w_def": w_def,
            "home_ppa": np.full(n, f_ppa.home), "home_sr": np.full(n, f_sr.home),
        })
```

(`a.off[:k][slice(None)]` is the same array as `a.off[:k]`, so the unmasked path computes exactly what it did before.)

- [ ] **Step 4: Run the new tests and the leakage tests**

Run: `uv run pytest tests/test_moves.py tests/test_leakage.py -q`
Expected: all pass.

- [ ] **Step 5: Prove published output unchanged**

Run: `uv run cfbr backtest --seasons 2015-2021 --config config/tuned.toml --tag what-moved-check`
Then: `uv run python -c "import glob, polars as pl; a=pl.read_parquet(glob.glob('/Users/abtreece/Projects/cfbr/results/20261108T165939Z-tuned-v2-tuning/predictions.parquet')[0]).sort('game_id'); b=pl.read_parquet(sorted(glob.glob('results/*-what-moved-check/predictions.parquet'))[-1]).sort('game_id'); print((a['pred_margin']==b['pred_margin']).all(), a.height, b.height)"`
Expected: `True 5844 5844` (the tuned-v2 tuning run is the v2 published config on 2015–2021).

- [ ] **Step 6: Full suite, commit**

Run: `uv run pytest -q` (expect all pass)

```bash
git add src/cfbr/model/engine.py tests/test_moves.py
git commit -m "feat: what-moved: masked engine fit and home coefficients in snapshots"
```

---

### Task 2: `report/moves.py` — the breakdown and game efficiency

**Files:**
- Create: `src/cfbr/report/moves.py`
- Test: `tests/test_moves.py`

**Interfaces:**
- Consumes: `Engine._fit(..., mask=)`, snapshot columns `home_ppa`/`home_sr` (Task 1); `team_ratings_points` (engine).
- Produces:
  - `rating_moves(eng: Engine, season: int, prev_as_of, as_of) -> pl.DataFrame`, one row per FBS team: `team, rating, prev_rating, rating_change, rank, prev_rank, move_decay, move_own, move_others, passed_by (list[str], ≤ 5), passed (list[str], ≤ 5), n_passed_by (Int32), n_passed (Int32)`. Unrounded floats.
  - `game_efficiency(eng: Engine, season: int) -> pl.DataFrame`: `game_id, home_ppa, away_ppa, home_sr, away_sr, exp_ppa_edge, exp_sr_edge` (edges from the home side's view; per-play units).
  - `moves_csv_frame(m: pl.DataFrame) -> pl.DataFrame`: rounded (2 dp) CSV form, lists joined with `"; "`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_moves.py`):

```python
from cfbr.report.moves import game_efficiency, moves_csv_frame, rating_moves
from cfbr.report.rank import fbs_table, week_as_of

SEASON = 2023


@pytest.fixture(scope="module")
def eng(cfg_two_stage, dataset):
    return Engine(dataset, cfg_two_stage, last_completed=2025)


@pytest.fixture(scope="module")
def week7(eng):
    return rating_moves(eng, SEASON, week_as_of(eng, SEASON, 6), week_as_of(eng, SEASON, 7))


def test_parts_sum_to_the_published_change(eng, week7):
    cur = fbs_table(eng, SEASON, week_as_of(eng, SEASON, 7)).select("team", "rating", "rank")
    prev = fbs_table(eng, SEASON, week_as_of(eng, SEASON, 6)).select("team", prev="rating")
    ref = cur.join(prev, on="team").with_columns(change=pl.col("rating") - pl.col("prev"))
    m = week7.join(ref, on="team", suffix="_ref")
    assert m.height == week7.height == ref.height
    total = m["move_decay"] + m["move_own"] + m["move_others"]
    assert (total - m["change"]).abs().max() < 1e-9
    assert (m["rank"] == m["rank_ref"]).all()


def test_bye_week_has_no_own_game(eng, week7):
    g = eng.games.filter((pl.col("season") == SEASON) & (pl.col("week") == 7))
    played = set(g["home"].to_list()) | set(g["away"].to_list())
    byes = week7.filter(~pl.col("team").is_in(list(played)))
    assert byes.height > 0 and (byes["move_own"] == 0).all()


def test_week_one_has_nothing_to_fade(eng):
    w1 = rating_moves(eng, SEASON, eng.slot_as_of(SEASON, "r01"), week_as_of(eng, SEASON, 1))
    assert w1["move_decay"].abs().max() < 1e-12


def test_passed_lists_match_the_ranks(week7):
    ranks = {r["team"]: (r["prev_rank"], r["rank"]) for r in week7.iter_rows(named=True)}
    for r in week7.iter_rows(named=True):
        pr, cr = r["prev_rank"], r["rank"]
        by = [t for t, (p, c) in ranks.items() if p > pr and c < cr]
        gone = [t for t, (p, c) in ranks.items() if p < pr and c > cr]
        assert r["n_passed_by"] == len(by) and r["n_passed"] == len(gone)
        assert set(r["passed_by"]) <= set(by) and len(r["passed_by"]) == min(5, len(by))
        assert set(r["passed"]) <= set(gone) and len(r["passed"]) == min(5, len(gone))


def test_game_efficiency_actuals_and_point_in_time_expectation(eng):
    ge = game_efficiency(eng, SEASON)
    g = eng.games.filter((pl.col("season") == SEASON) & pl.col("completed") & (pl.col("week") == 7))
    row = g.row(0, named=True)
    e = ge.filter(pl.col("game_id") == row["game_id"]).row(0, named=True)
    plays = eng.season_plays(SEASON).filter(pl.col("game_id") == row["game_id"])
    hp = plays.filter(pl.col("offense") == row["home"])
    assert e["home_ppa"] == pytest.approx(hp["ppa"].mean())
    assert e["home_sr"] == pytest.approx(hp["success"].cast(pl.Float64).mean())
    as_of = eng.slot_as_of(SEASON, row["slot"])
    assert as_of <= row["kickoff_ts"]
    snap = eng.snapshot(SEASON, as_of)
    r = {t: x for t, x in zip(snap["team"], snap.iter_rows(named=True))}
    hm = 0.0 if row["neutral_site"] else 1.0
    exp = (r[row["home"]]["o_ppa"] + r[row["home"]]["d_ppa"] - r[row["away"]]["o_ppa"] - r[row["away"]]["d_ppa"]
           + 2 * snap["home_ppa"][0] * hm)
    assert e["exp_ppa_edge"] == pytest.approx(exp, abs=1e-12)


def test_csv_frame_rounds_and_joins(week7):
    c = moves_csv_frame(week7)
    assert c.schema["passed_by"] == pl.Utf8 and c["move_own"].round(2).equals(c["move_own"])
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_moves.py -q -k "sum or bye or week_one or passed or efficiency or csv"`
Expected: `ModuleNotFoundError: No module named 'cfbr.report.moves'`.

- [ ] **Step 3: Implement `src/cfbr/report/moves.py`:**

```python
"""What moved (spec 2026-11-08-what-moved-design.md): split a week's rating change per FBS team
into older plays fading, the team's own game and other games, plus per-game efficiency against
the pre-game expectation. Display only; every fit goes through Engine._fit (the published path)."""

from __future__ import annotations

import numpy as np
import polars as pl

from cfbr.model.engine import Engine, team_ratings_points

MAX_NAMES = 5


def _points(eng: Engine, season: int, snap: pl.DataFrame) -> pl.DataFrame:
    return team_ratings_points(snap, eng.calibration(season), eng.cfg).select("team", "rating")


def rating_moves(eng: Engine, season: int, prev_as_of, as_of) -> pl.DataFrame:
    a, fbs = eng.arrays(season), eng.fbs(season)
    priors = eng.priors(season)
    prev = _points(eng, season, eng.snapshot(season, prev_as_of)).rename({"rating": "prev_rating"})
    cur = _points(eng, season, eng.snapshot(season, as_of))
    k_prev, k = a.n_before(prev_as_of), a.n_before(as_of)
    old = np.zeros(k, bool)
    old[:k_prev] = True
    faded = _points(eng, season, eng._fit(season, priors, as_of, mask=old)).rename({"rating": "faded"})
    idx = {t: i for i, t in enumerate(a.teams)}
    cur_rating = dict(cur.iter_rows())
    own = {}
    for team in fbs:
        i = idx[team]
        mine = ~old & ((a.off[:k] == i) | (a.de[:k] == i))
        if not mine.any():
            own[team] = 0.0
            continue
        without = _points(eng, season, eng._fit(season, priors, as_of, mask=~mine))
        own[team] = cur_rating[team] - without.filter(pl.col("team") == team)["rating"].item()
    rank = pl.col("rating").rank("ordinal", descending=True).cast(pl.Int32)
    df = (cur.filter(pl.col("team").is_in(fbs)).with_columns(rank=rank)
          .join(prev.filter(pl.col("team").is_in(fbs))
                .with_columns(prev_rank=pl.col("prev_rating").rank("ordinal", descending=True).cast(pl.Int32)),
                on="team")
          .join(faded, on="team")
          .with_columns(rating_change=pl.col("rating") - pl.col("prev_rating"),
                        move_decay=pl.col("faded") - pl.col("prev_rating"),
                        move_own=pl.col("team").replace_strict(own, return_dtype=pl.Float64))
          .with_columns(move_others=pl.col("rating_change") - pl.col("move_decay") - pl.col("move_own"))
          .sort("rank"))
    ranks = list(zip(df["team"], df["prev_rank"], df["rank"]))  # sorted by current rank
    by, gone = [], []
    for team, pr, cr in ranks:
        by.append([t for t, p, c in ranks if p > pr and c < cr])
        gone.append([t for t, p, c in ranks if p < pr and c > cr])
    return df.with_columns(
        passed_by=pl.Series([x[:MAX_NAMES] for x in by], dtype=pl.List(pl.Utf8)),
        passed=pl.Series([x[:MAX_NAMES] for x in gone], dtype=pl.List(pl.Utf8)),
        n_passed_by=pl.Series([len(x) for x in by], dtype=pl.Int32),
        n_passed=pl.Series([len(x) for x in gone], dtype=pl.Int32),
    ).select("team", "rating", "prev_rating", "rating_change", "rank", "prev_rank", "move_decay", "move_own",
             "move_others", "passed_by", "passed", "n_passed_by", "n_passed")


def game_efficiency(eng: Engine, season: int) -> pl.DataFrame:
    """Per completed game: each side's PPA per play and success rate (the plays the fit used) and
    the expected edges from the game's pre-game slot snapshot, from the home side's view:
    (O_h + D_h) - (O_a + D_a) + 2·H·home, home = 0 at a neutral site. The intercept cancels."""
    if eng.as_of_mode != "slot":
        raise ValueError("game_efficiency needs slot as-of mode")
    stats = (eng.season_plays(season)
             .group_by("game_id", "offense")
             .agg(ppa=pl.col("ppa").mean(), sr=pl.col("success").cast(pl.Float64).mean()))
    g = eng.season_games_with_snapshots(season).filter(pl.col("completed"))
    home_coef = {slot: (s["home_ppa"][0], s["home_sr"][0]) for slot, s in eng.snapshots(season).items()}
    hm = pl.when(pl.col("neutral_site")).then(0.0).otherwise(1.0)
    g = g.with_columns(
        h_ppa=pl.col("slot").replace_strict({k: v[0] for k, v in home_coef.items()}, return_dtype=pl.Float64),
        h_sr=pl.col("slot").replace_strict({k: v[1] for k, v in home_coef.items()}, return_dtype=pl.Float64),
    )
    side = lambda s: stats.rename({"offense": s, "ppa": f"{s}_ppa", "sr": f"{s}_sr"})  # noqa: E731
    return (g.join(side("home"), on=["game_id", "home"], how="left")
            .join(side("away"), on=["game_id", "away"], how="left")
            .with_columns(
                exp_ppa_edge=pl.col("o_ppa_h") + pl.col("d_ppa_h") - pl.col("o_ppa_a") - pl.col("d_ppa_a")
                + 2 * pl.col("h_ppa") * hm,
                exp_sr_edge=pl.col("o_sr_h") + pl.col("d_sr_h") - pl.col("o_sr_a") - pl.col("d_sr_a")
                + 2 * pl.col("h_sr") * hm)
            .select("game_id", "home_ppa", "away_ppa", "home_sr", "away_sr", "exp_ppa_edge", "exp_sr_edge"))


def moves_csv_frame(m: pl.DataFrame) -> pl.DataFrame:
    floats = ["rating", "prev_rating", "rating_change", "move_decay", "move_own", "move_others"]
    return m.with_columns([pl.col(c).round(2) for c in floats]
                          + [pl.col(c).list.join("; ") for c in ("passed_by", "passed")])
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_moves.py -q`
Expected: all pass. (If `eng.season_plays` lacks `game_id`, check `prep/plays.py` SCRIMMAGE_SQL: it selects `p.game_id`, so `model_plays` has it.)

- [ ] **Step 5: Timing sanity**

Run: `uv run python -c "import time; from cfbr.config import load_config; from cfbr.data import Dataset; from cfbr.model.engine import Engine; from cfbr.report.moves import rating_moves; from cfbr.report.rank import week_as_of; c=load_config('config/tuned.toml'); e=Engine(Dataset.load(c),c,last_completed=2025); t=time.perf_counter(); m=rating_moves(e,2026,week_as_of(e,2026,9),week_as_of(e,2026,10)); print(round(time.perf_counter()-t,1),'s'); print(m.filter(m['team']=='Ohio State').to_dicts())"`
Expected: a few seconds; Ohio State `move_own` ≈ −1.88, `move_decay` ≈ −0.22, `move_others` ≈ +0.14, `passed_by` = ['Georgia', 'Alabama'] (the prototype's numbers).

- [ ] **Step 6: Commit**

```bash
git add src/cfbr/report/moves.py tests/test_moves.py
git commit -m "feat: what-moved: rating change breakdown and game efficiency"
```

---

### Task 3: Write the data — `moves.csv`, web parquet columns, lock lists

**Files:**
- Modify: `src/cfbr/report/rank.py` (`run_rank`)
- Modify: `src/cfbr/web/build.py` (`season_ratings`, `season_games`)
- Modify: `src/cfbr/web/app.py` (`MODEL_RATING_COLS`, `MODEL_GAME_COLS`)
- Test: `tests/test_moves.py`, `tests/test_web.py`

**Interfaces:**
- Consumes: `rating_moves`, `game_efficiency`, `moves_csv_frame` (Task 2).
- Produces: `out/{season}/wNN/moves.csv`; `ratings.parquet` columns `move_decay, move_own, move_others, passed_by, passed, n_passed_by, n_passed` (null outside a live season and at the preseason point); `games.parquet` columns `home_ppa, away_ppa, home_sr, away_sr, exp_ppa_edge, exp_sr_edge`. `season_ratings(eng, season, now, moves: bool | None = None)` (None = only when the season is live).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_moves.py`):

```python
def test_rank_csv_matches_web_rows(eng, tmp_path):
    from datetime import UTC, datetime

    from cfbr.report.rank import run_rank
    from cfbr.web.build import season_ratings

    cfg = eng.cfg.model_copy(deep=True)
    cfg.paths.out_dir = tmp_path
    out = run_rank(cfg, SEASON, 7, last_completed=2025, ds=eng.ds)
    csv = pl.read_csv(out / "moves.csv")
    web = (season_ratings(eng, SEASON, datetime.now(UTC), moves=True)
           .filter((pl.col("week") == 7) & pl.col("fbs")))
    web = moves_csv_frame(web.select("team", "rating", "move_decay", "move_own", "move_others", "passed_by",
                                     "passed").with_columns(prev_rating=pl.lit(0.0), rating_change=pl.lit(0.0)))
    m = csv.join(web, on="team", suffix="_web")
    assert m.height == csv.height
    for c in ("move_decay", "move_own", "move_others"):
        assert (m[c] - m[f"{c}_web"]).abs().max() < 0.006
    assert (m["passed_by"].fill_null("") == m["passed_by_web"].fill_null("")).all()
```

In `tests/test_web.py`, extend the free-viewer API test. Find `test_free_api_withholds_live_model_output` and add these lines at its end:

```python
    from cfbr.web.app import MODEL_GAME_COLS, MODEL_RATING_COLS

    assert {"move_own", "move_decay", "move_others", "passed_by", "passed"} <= set(MODEL_RATING_COLS)
    assert {"exp_ppa_edge", "exp_sr_edge"} <= set(MODEL_GAME_COLS)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_moves.py -q -k csv_matches` (expect FAIL: no `moves.csv` / `moves` kwarg)
Run: `uv run pytest tests/test_web.py -q -k free_api_withholds` (expect FAIL: columns not in lists)

- [ ] **Step 3: `run_rank`** in `src/cfbr/report/rank.py`, after the `predictions.csv` line:

```python
    from cfbr.report.moves import moves_csv_frame, rating_moves

    prev_as_of = week_as_of(eng, season, week - 1) if week > 1 else eng.slot_as_of(season, "r01")
    moves_csv_frame(rating_moves(eng, season, prev_as_of, as_of)).write_csv(out / "moves.csv")
```

(`prev_as_of` matches `build_rankings`' previous table, so `moves.csv`'s `rating_change` equals `rankings.csv`'s.)

- [ ] **Step 4: Web build.** In `src/cfbr/web/build.py`, add `from cfbr.report.moves import game_efficiency, rating_moves` to the imports. Change `season_ratings`:

```python
MOVE_COLS = ["move_decay", "move_own", "move_others", "passed_by", "passed", "n_passed_by", "n_passed"]


def season_ratings(eng: Engine, season: int, now: datetime, moves: bool | None = None) -> pl.DataFrame:
    """``moves`` adds the what-moved columns (None = only for a live season; ~2 s per week)."""
    cal = eng.calibration(season)
    live = not season_over(eng, season, now)
    moves = live if moves is None else moves
    fbs = eng.fbs(season)
    conf = eng.ds.team_status.filter(pl.col("season") == season).select("team", "conference")
    parts = []
    prev_as_of = None
    for week, label, as_of in timeline(eng, season, now):
        snap = _snapshot(eng, season, as_of)
        t = (team_ratings_points(snap, cal, eng.cfg)
             .join(snap.select("team", *RATING_COLS), on="team")
             .with_columns(fbs=pl.col("team").is_in(fbs)))
        rank = t.filter(pl.col("fbs")).select(
            "team", rank=pl.col("rating").rank("ordinal", descending=True).cast(pl.Int32))
        rec = records(eng, season, as_of).select("team", "w", "l")
        polls, _ = timeline_polls(eng.ds.polls, season, week)
        part = (t.join(rank, on="team", how="left").join(rec, on="team", how="left")
                .join(conf, on="team", how="left").join(polls, on="team", how="left"))
        if moves and prev_as_of is not None and week != FINAL:
            part = part.join(rating_moves(eng, season, prev_as_of, as_of).select("team", *MOVE_COLS),
                             on="team", how="left")
        parts.append(part.with_columns(season=pl.lit(season, pl.Int32), week=pl.lit(week, pl.Int32),
                                       label=pl.lit(label), as_of=pl.lit(as_of),
                                       w=pl.col("w").fill_null(0), l=pl.col("l").fill_null(0)))
        prev_as_of = as_of
    df = pl.concat(parts, how="diagonal_relaxed").sort("team", "week")
    for c, dt in (("move_decay", pl.Float64), ("move_own", pl.Float64), ("move_others", pl.Float64),
                  ("passed_by", pl.List(pl.Utf8)), ("passed", pl.List(pl.Utf8)),
                  ("n_passed_by", pl.Int32), ("n_passed", pl.Int32)):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dt).alias(c))
    return (df.with_columns(
        rating_change=pl.col("rating") - pl.col("rating").shift(1).over("team"),
        rank_change=pl.col("rank").shift(1).over("team") - pl.col("rank"),
    ).select("season", "week", "label", "as_of", "team", "conference", "fbs", "rank", "w", "l",
             "rating", "offense", "defense", "prior_share", "rating_change", "rank_change", *POLLS, *RATING_COLS,
             *MOVE_COLS)
            .sort("week", "rank", "team", nulls_last=True))
```

In `season_games`, after the `spread` join:

```python
    g = g.join(game_efficiency(eng, season), on="game_id", how="left")
```

- [ ] **Step 5: Lock lists** in `src/cfbr/web/app.py`:

```python
MODEL_RATING_COLS = ["rank", "rating", "offense", "defense", "prior_share", "rating_change", "rank_change",
                     "o_ppa", "d_ppa", "o_sr", "d_sr",
                     "move_decay", "move_own", "move_others", "passed_by", "passed", "n_passed_by", "n_passed"]
MODEL_GAME_COLS = ["pred_margin", "win_prob", "prior_pred", "massey_pred", "sigma", "prior_share_h", "prior_share_a",
                   "exp_ppa_edge", "exp_sr_edge"]
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_moves.py tests/test_web.py -q`
Expected: all pass. (The csv/web test fits ~14 weeks: expect ~30 s.)

- [ ] **Step 7: Full suite, commit**

Run: `uv run pytest -q`

```bash
git add src/cfbr/report/rank.py src/cfbr/web/build.py src/cfbr/web/app.py tests/test_moves.py tests/test_web.py
git commit -m "feat: what-moved: moves.csv, web data columns, free-viewer lock"
```

---

### Task 4: Display — team page, rankings tooltips, Prior column, Method

**Files:**
- Modify: `src/cfbr/web/app.py` (`team_games`, new `what_moved`, `team_page`, `api_team`)
- Modify: `src/cfbr/web/templates/team.html`, `rankings.html`, `method.html`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: the ratings/games columns from Task 3.
- Produces: `what_moved(hist: list[dict], tgames: list[dict]) -> dict | None`. Keys: `label, prev_rank, rank, rank_change, prev_rating, rating, rating_change, own, decay, others, passed_by, passed, n_passed_by, n_passed, games`. Each game dict has `opponent, is_home, neutral_site, team_points, opp_points, t_pred, t_ppa, opp_ppa, t_sr, opp_sr, t_exp_ppa_edge`. Returns None when the latest week has no move data (preseason, finished season, or locked).

- [ ] **Step 1: Extend the synthetic fixture** in `tests/test_web.py`. In `_ratings()`, add to every row dict:

```python
                "move_decay": None if week == 0 else -0.2, "move_own": None if week == 0 else 1.5,
                "move_others": None if week == 0 else -0.3,
                "passed_by": None if week == 0 else (["Ohio State"] if i == 0 else []),
                "passed": None if week == 0 else [], "n_passed_by": None if week == 0 else (1 if i == 0 else 0),
                "n_passed": None if week == 0 else 0,
```

and extend `schema_overrides` with `"move_decay": pl.Float64, "move_own": pl.Float64, "move_others": pl.Float64, "passed_by": pl.List(pl.Utf8), "passed": pl.List(pl.Utf8), "n_passed_by": pl.Int32, "n_passed": pl.Int32`. In `_game()`, add `"home_ppa": 0.25, "away_ppa": 0.05, "home_sr": 0.48, "away_sr": 0.38, "exp_ppa_edge": 0.10, "exp_sr_edge": 0.04,`.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_web.py`):

```python
def test_team_page_shows_what_moved(client):
    html = " ".join(client.get(f"/team/{SEASON}/Texas%20A%26M").text.split())
    assert "What moved" in html and "Own game" in html and "+1.50" in html
    assert "Older plays fading" in html and "−0.20" in html and "Other games" in html
    assert "Beat" in html and "31–10" in html and "expected to win by 10.0" in html
    assert "PPA +0.250 vs +0.050" in html and "48.0% vs 38.0%" in html
    assert "Passed by</th><td></td><td>Ohio State" in html  # fixture: index 0 passed_by = ["Ohio State"]


def test_team_page_bye_week(client):
    html = " ".join(client.get(f"/team/{SEASON}/%3Cb%3EEvil%3C%2Fb%3E").text.split())
    assert "What moved" in html


def test_rankings_tooltips_and_prior_column(client):
    html = client.get(f"/rankings?season={SEASON}&week=1").text
    assert 'title="Own game +1.50 · fading −0.20 · others −0.30' in html
    assert 'class="num opt" title="How much of the rating' in html  # Prior no longer behind More columns


def test_free_viewers_get_no_what_moved(free_client):
    html = free_client.get(f"/team/{SEASON}/Texas%20A%26M").text
    assert "What moved" not in html
```

`test_team_page_bye_week` can't use the HTML fixture: all four fixture teams play in week 1. Test the bye case on `what_moved` directly instead:

```python
def test_what_moved_bye_week():
    from cfbr.web.app import what_moved

    t0 = T0
    hist = [{"label": "Preseason", "as_of": t0, "rating": 1.0, "rank": 2, "move_own": None},
            {"label": "Week 1", "as_of": t0 + timedelta(days=7), "rating": 0.8, "rank": 3, "rank_change": -1,
             "rating_change": -0.2, "move_own": 0.0, "move_decay": -0.2, "move_others": 0.0,
             "passed_by": ["X"], "passed": [], "n_passed_by": 1, "n_passed": 0}]
    m = what_moved(hist, [])
    assert m["games"] == [] and m["own"] == 0.0 and m["passed_by"] == ["X"]
    assert what_moved(hist[:1], []) is None
```

Use it in place of `test_team_page_bye_week` above.

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_web.py -q -k "what_moved or bye_week or tooltips"`
Expected: FAIL (section and tooltips don't exist).

- [ ] **Step 4: `team_games` columns** in `src/cfbr/web/app.py`, add inside its `with_columns(...)`:

```python
        t_ppa=pl.when(home).then(pl.col("home_ppa")).otherwise(pl.col("away_ppa")),
        opp_ppa=pl.when(home).then(pl.col("away_ppa")).otherwise(pl.col("home_ppa")),
        t_sr=pl.when(home).then(pl.col("home_sr")).otherwise(pl.col("away_sr")),
        opp_sr=pl.when(home).then(pl.col("away_sr")).otherwise(pl.col("home_sr")),
        t_exp_ppa_edge=pl.when(home).then(pl.col("exp_ppa_edge")).otherwise(-pl.col("exp_ppa_edge")),
```

Before the `return`, guard old web data: at the top of `team_games`, after `g = ...`, add
`g = _ensure(g, ["home_ppa", "away_ppa", "home_sr", "away_sr", "exp_ppa_edge", "exp_sr_edge"])` (import `_ensure` from `cfbr.web.store`; it adds missing columns as nulls).

- [ ] **Step 5: `what_moved`** in `src/cfbr/web/app.py` (after `team_games`):

```python
def what_moved(hist: list[dict], tgames: list[dict]) -> dict | None:
    """The latest week's rating-change breakdown for the team page (spec 2026-11-08-what-moved)."""
    if len(hist) < 2 or hist[-1].get("move_own") is None:
        return None
    cur, prev = hist[-1], hist[-2]
    games = [g for g in tgames if g.get("completed") and prev["as_of"] <= g["kickoff_ts"] < cur["as_of"]]
    return {"label": cur["label"], "prev_rank": prev.get("rank"), "rank": cur.get("rank"),
            "rank_change": cur.get("rank_change"), "prev_rating": prev["rating"], "rating": cur["rating"],
            "rating_change": cur["rating_change"], "own": cur["move_own"], "decay": cur["move_decay"],
            "others": cur["move_others"], "passed_by": cur.get("passed_by") or [], "passed": cur.get("passed") or [],
            "n_passed_by": cur.get("n_passed_by") or 0, "n_passed": cur.get("n_passed") or 0, "games": games}
```

In `team_page`, pass `moved=None if locked else what_moved(hist.to_dicts(), games_raw)` to `render`, where `games_raw = team_games(s, team).to_dicts()` (compute it once and reuse it for the existing `games=` list: `games = [game_row(g, now, live, cfg.bets.threshold) for g in games_raw]`).

In `api_team`, add `"t_ppa", "opp_ppa", "t_sr", "opp_sr", "t_exp_ppa_edge"` to the `select(...)` list.

- [ ] **Step 6: Team template.** In `src/cfbr/web/templates/team.html`, immediately before `<h2>Explain the rating</h2>`, add:

```html
{% if moved %}
<h2 id="moved">What moved <span class="muted">{{ moved.label|lower }}</span></h2>
<p class="note">Rank {{ moved.prev_rank }} → {{ moved.rank }}{% if moved.rank_change %} ({{ delta(moved.rank_change, 0) }}){% endif %} · rating {{ moved.prev_rating|signed }} → {{ moved.rating|signed }} ({{ delta(moved.rating_change, 2) }}). Ratings come from per-play efficiency against what was expected, not from wins and losses. <a href="/method#moves">How this is split</a></p>
<table class="data moved">
  <tbody>
    <tr><th scope="row">Own game</th><td class="num strong">{{ moved.own|signed(2) }}</td>
      <td>{% if moved.games %}{% for g in moved.games %}
        {% set won = g.team_points > g.opp_points %}{{ "Beat" if won else ("Lost to" if g.team_points < g.opp_points else "Tied") }} {{ g.opponent }} {{ g.team_points }}–{{ g.opp_points }}{% if g.t_pred is not none %} · expected to {{ "win" if g.t_pred > 0 else "lose" }} by {{ "%.1f"|format(g.t_pred|abs) }}{% endif %}
        {% if g.t_ppa is not none %}<br><span class="muted">per play: PPA {{ g.t_ppa|signed(3) }} vs {{ g.opp_ppa|signed(3) }}{% if g.t_exp_ppa_edge is not none %} (edge {{ (g.t_ppa - g.opp_ppa)|signed(3) }}, expected {{ g.t_exp_ppa_edge|signed(3) }}){% endif %} · success {{ "%.1f"|format(100 * g.t_sr) }}% vs {{ "%.1f"|format(100 * g.opp_sr) }}%</span>{% endif %}{% if not loop.last %}<br>{% endif %}
      {% endfor %}{% else %}No game this week{% endif %}</td></tr>
    <tr><th scope="row">Older plays fading</th><td class="num">{{ moved.decay|signed(2) }}</td><td class="muted">last week's plays count a little less each week</td></tr>
    <tr><th scope="row">Other games</th><td class="num">{{ moved.others|signed(2) }}</td><td class="muted">other results re-rating this team's opponents</td></tr>
    {% if moved.passed_by %}<tr><th scope="row">Passed by</th><td></td><td>{{ moved.passed_by|join(", ") }}{% if moved.n_passed_by > moved.passed_by|length %} and {{ moved.n_passed_by - moved.passed_by|length }} more{% endif %}</td></tr>{% endif %}
    {% if moved.passed %}<tr><th scope="row">Passed</th><td></td><td>{{ moved.passed|join(", ") }}{% if moved.n_passed > moved.passed|length %} and {{ moved.n_passed - moved.passed|length }} more{% endif %}</td></tr>{% endif %}
  </tbody>
</table>
{% endif %}
```

Check the `signed` filter's signature in `app.py` (it's used as `|signed` and `|signed(3)` already, so `signed(2)` works; it renders a true minus "−"). In the week-by-week table, add three header cells after `Δ Rating`: `<th class="num" title="Own game">Own</th><th class="num" title="Older plays fading">Fade</th><th class="num" title="Other games">Others</th>`, and the matching cells after the Δ Rating cell:

```html
        <td class="num">{{ h.move_own|signed(2) if h.move_own is not none else "—" }}</td>
        <td class="num">{{ h.move_decay|signed(2) if h.move_decay is not none else "—" }}</td>
        <td class="num">{{ h.move_others|signed(2) if h.move_others is not none else "—" }}</td>
```

- [ ] **Step 7: Rankings template.** In `src/cfbr/web/templates/rankings.html`:
  - Prior header: change `class="num opt extra" title="{{ col["Prior"] }}"` to `class="num opt" title="{{ col["Prior"] }}"`, and the Prior cell's `class="num opt extra"` to `class="num opt"`.
  - Add a macro-free tooltip variable at the start of the row loop (right after `<tr>`):

```html
    {% set tip = ("Own game " ~ (r.move_own|signed(2)) ~ " · fading " ~ (r.move_decay|signed(2)) ~ " · others " ~ (r.move_others|signed(2)) ~ ((" · passed by " ~ (r.passed_by|join(", "))) if r.passed_by else "")) if r.move_own is not none else "" %}
```

  - On the ± cell and the Δ Rating cell, add `{% if tip %} title="{{ tip }}"{% endif %}` to the `<td ...>` tag.

- [ ] **Step 8: Method page.** In `src/cfbr/web/templates/method.html`, after the `#rating` section (before `<h2 id="point-in-time">`), add:

```html
<h2 id="moves">What moved</h2>
<p>Each week's rating change on a team page is split into three parts that add up exactly to it. <strong>Own game:</strong> the rating with this week's game, minus the same fit without it. <strong>Older plays fading:</strong> last week's plays refit at this week's date; recent plays count more, so every week older plays count a little less and ratings drift toward the preseason prior. <strong>Other games:</strong> the rest, mostly other results re-rating this team's opponents. The per-play line compares the game's points per play and success rate with what the model expected before kickoff.</p>
```

- [ ] **Step 9: Run tests**

Run: `uv run pytest tests/test_web.py -q`
Expected: all pass, including the existing Method and rankings tests (update any assertion that pins Prior as `opt extra`).

- [ ] **Step 10: Look at it.** Build web data for the live season against the local DB and render the Ohio State page:

Run: `uv run cfbr web build --season 2026` (writes to the gitignored `out/web/`)
Run (background): `uv run cfbr web serve --port 8011`
Then: `curl -s http://127.0.0.1:8011/team/2026/Ohio%20State | tr -s ' \n' ' ' | grep -o 'What moved.*Explain the rating' | head -c 1500`
Expected: the section with own game ≈ −1.88, "Lost to Oregon 28–41 · expected to win by 13.7", the per-play line, fading ≈ −0.22, others ≈ +0.14, "Passed by Georgia, Alabama". Stop the server afterwards.

- [ ] **Step 11: Commit**

```bash
git add src/cfbr/web/app.py src/cfbr/web/templates/team.html src/cfbr/web/templates/rankings.html src/cfbr/web/templates/method.html tests/test_web.py
git commit -m "feat: what-moved: team page section, rankings tooltips, Prior column, Method"
```

---

### Task 5: Docs, PR

**Files:** `DECISIONS.md`, `docs/superpowers/specs/2026-11-08-what-moved-design.md`, `README.md` (if it lists `out/{season}/wNN/` files)

- [ ] **Step 1: DECISIONS line** (append):

```
- What-moved breakdown (2026-11-08, display only): each live-season week's rating change per FBS team is split into own game (full fit minus the fit without that team's week plays), older plays fading (last week's plays refit at this week's as-of) and other games (the remainder), through Engine._fit with a play mask so it is the published fit path; parts sum exactly to rating_change. Shown on team pages (with score vs pre-game prediction and per-play PPA/success vs the pre-game expected edge) and as rankings tooltips; written to out/{season}/wNN/moves.csv by `cfbr rank`; model-derived parts locked like ratings. Live season only in the web build (~2 s/week). Rejected first: an "in-season only" rating (published model without the prior) ranked Wake Forest #1 and Liberty #2 after 2026 week 10 with compressed ratings (SD 7.7 vs 11.0); prior scaled 0–1 on 2015–2021 under v2 loses MAE at every scale (13.295 full, 15.047 none; no-prior cost +6.0 weeks 1–2, +0.54 weeks 11+). Prior column moved out of "More columns" — Web
```

- [ ] **Step 2: Spec status** — change `Status: design approved, not built.` to `Status: built 2026-11-08; storage simplified (columns in ratings/games parquet), see plan.`

- [ ] **Step 3: README** — if `README.md` lists the files in `out/{season}/wNN/`, add `moves.csv`. Check with `grep -n "predictions.csv" README.md`.

- [ ] **Step 4: Full suite**

Run: `uv run pytest -q` — all pass; paste the summary line in the PR.

- [ ] **Step 5: Commit, run `check-yourself` on the PR body, push, open PR**

```bash
git add DECISIONS.md docs/superpowers/specs/2026-11-08-what-moved-design.md README.md
git commit -m "docs: what-moved breakdown"
```

```bash
git push -u origin feat/what-moved
```

`gh pr create --title "feat: what-moved breakdown on team pages and rankings" --body-file <scratchpad>/pr_body_moves.md` (body: problem, the three parts, the Ohio State example, deviations from the spec, tests; ends with the session's PR attribution line).

- [ ] **Step 6: STOP — ask the user** before merging and deploying. Deploy needs no restatement: the web build adds the columns for 2026; Sunday's `weekly` writes `w11/moves.csv`.
