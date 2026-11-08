# FCS games, two-stage fit — design

Date: 2026-11-08. Status: rule fixed, not built.

## Problem

FBS-vs-FCS games move FBS ratings far more than they should. In 2026 week 10,
Wake Forest went 17.07 → 23.22 (#11 → #3) after a 77–0 win over Merrimack;
refit without that game, Wake is 15.69 (#13). Two causes:

1. **One fit does two jobs.** FCS plays sit in the same weighted ridge as FBS
   plays, so a blowout of an FCS opponent is weighted like any other game.
   Many FBS teams play FCS opponents early (when there's little other data)
   or once late in the season (when recency weighting makes it count the most).
2. **FCS groups shrink toward the wrong place.** Pooled FCS entities are
   shrunk toward 0 (an average FBS team) in the no-prior `final()` fit, and
   that final carries over as next season's prior (`pooled_priors`). Thin
   groups barely move off 0: FCS Independents (about 250 plays a season)
   finished at −13.5 in 2024 and −7.6 in 2025, while the other groups were
   −15 to −41. So in 2026 the model expected Merrimack to play like a weak FBS
   team, and credited the 77–0 win to Wake.

User requirement: an FCS game should count for almost nothing in an FBS team's
rating, early or late in the season. How much exactly is tuned on 2015–2021.

## Design

### Config (`config.py`, `config/default.toml`)

- `model.fcs_two_stage: bool = false`. When false, the fit is today's single
  fit, bit for bit, so `tuned.toml` (to be kept as `tuned_v1.toml`) reproduces.
- `model.fcs_weight: float = 1.0`, in [0, 1]. Used only when two-stage is on.

Both are optional with defaults, so every existing config keeps loading.

### Stage 1: FBS ratings (`Engine._fit`, used by `snapshot()` and `final()`)

The per-play weight becomes the recency weight × (`fcs_weight` if the offense
or defense is a pooled FCS entity, else 1). Everything else is the existing
ridge: same priors, alpha, home term and intercept. The FBS ratings come from
Stage 1. Stage 1's FCS entity ratings are thrown away.

Per-play weights are computed by a single helper,
`play_weights(arrays, k, as_of, cfg)` in `model/ridge.py`. `Engine._fit`,
`model/uncertainty.py` and `model/matchup.py` all call it. Today the last two
call `recency_weights` themselves, so without this change they would silently
ignore `fcs_weight` and stop matching the published fit. Neither is published
(both models were rejected), but they stay consistent diagnostics.

### Stage 2: FCS ratings (new function in `model/ridge.py`)

- **Input:** only plays with a pooled FCS side that kicked off before the same
  as-of time, at full recency weight (no `fcs_weight`).
- **Fixed offsets:** each FBS side's Stage 1 rating, Stage 1's home coefficient
  and intercept. Stage 2 never changes them.
- **Parameters, separately for PPA and success rate:** an FCS offense level
  `L_o` and an FCS defense level `L_d`, shared by every FCS group, plus each
  group's offense and defense deviation from those levels. A group's rating =
  level + deviation.
- **Penalties:**
  - Deviations get the same alpha as Stage 1 (`alpha_ppa` / `alpha_sr`),
    penalized toward the group's prior deviation (its pooled prior minus the
    mean of the FCS groups' pooled priors).
  - The levels get a negligible penalty (1e-6 × alpha) toward the mean of the
    pooled priors. That keeps them defined when there are no FCS plays yet
    (before week 1) and has no practical effect otherwise.
  - In `final()` (no prior), the deviations shrink toward 0, meaning *the FCS
    average*, not an average FBS team. That fixes cause 2.
- **Output:** these FCS ratings replace Stage 1's FCS rows in the snapshot or
  final. `pooled_priors` keeps its current logic (carry over last season's
  final, else use the mean) and now carries over the corrected finals.

### Unchanged

Calibration (base, v2, v3), slots, as-of times, `schemes.training_seasons`,
point-in-time rules, prior regression for FBS teams, and the web/rank code
paths. FBS ratings and published ranks depend only on Stage 1.

## Tuning

- Same harness as Phase 6 (`backtest/tune.py`): 256 Optuna TPE trials
  (multivariate, seed 0) in batches of 30, objective = overall MAE of the
  predicted margin over 2015–2021 under the hybrid scheme.
- **Fixed in every trial:** `fcs_two_stage = true`, `fcs_pooling =
  "conference"`, garbage-time filter off, recency on.
- **Searched:** `fcs_weight` uniform [0, 1]; `alpha_ppa` and `alpha_sr` log
  20–3000; `half_life_weeks` 2–26.
- **Output:** the best trial goes to a new `config/tuned_v2.toml`.
  `config/tuned.toml` stays untouched unless the gate passes.
- The Massey baseline settings are not re-tuned (only the baseline uses them).

## Evaluation rule (fixed before any trial runs)

Comparator: the published model (`config/tuned.toml` as of this commit). All
comparisons are paired: the same games, each predicted point-in-time from its
own slot. Metric: paired MAE gain = MAE(v1) − MAE(v2), with a 95% CI.

| # | Data | Role | Pass condition |
|---|---|---|---|
| 1 | 2015–2021 | Implementation check (in-sample: tuned here) | gain > 0 |
| 2 | 2022–2025 | Out of sample (viewed before, never tuned on) | gain > 0 and CI lower bound > −0.10 |
| 3 | 2026, every completed game through week 10 | Freshest data, never used for margin tuning | gain > 0 and CI lower bound > −0.10 |

All three must pass. If any fails, nothing published changes, the result goes
in DECISIONS.md, and there is no second round of tuning on these seasons.

Disclosure for gate 3: the 2026 week-9 and week-10 rankings and the Wake
Forest/Merrimack refit were looked at while diagnosing this problem. No MAE
comparison of this change on 2026 games was computed before this rule.

**Why "CI lower bound > −0.10" and not "CI excludes 0":** a few seasons can't
resolve a gain of about 0.05. Gates 2 and 3 exist to show the change isn't
worse on data it was not tuned on. This is the bar the shadow-adoption rules
already use.

**Reported, not gated:**

- The tuned `fcs_weight`, alpha and half-life next to the v1 values.
- MAE split into FBS-vs-FBS and FBS-vs-FCS games, per gate.
- ATS vs the closing line.
- Pooled-FCS slope of margin on prediction (track-record review: 0.68 in v1).
- 2026 week-10 top 25 under both models, and what the Merrimack game does to
  Wake Forest under v2.

Tuning seasons carry the documented second-order LOSO leak (calibration and
prior fit within 2015–2021).

## If the gate passes: adopt mid-2026 and restate

1. **Config:** `tuned.toml` → `config/tuned_v1.toml`; `tuned_v2.toml` →
   `config/tuned.toml`. Priors and calibration (including σ) refit from the
   new config. The season sim's `tau` stays 6.0, the value chosen under v1
   (DECISIONS.md, "Sim calibration"). It's a separate setting and not part of
   this rule.
2. **Restate 2026** (on javelin, after deploy, which is a stop-and-ask step):
   move `out/2026/w01…wNN/` to `out/2026/_v1/`, rerun `cfbr rank` for every
   completed 2026 week, rebuild `accuracy.csv`, then `cfbr web build`. The
   rule that the site matches `cfbr rank` keeps holding because both are
   regenerated. The Method page gets a one-line note: "2026 ratings restated
   on <date> after the FCS-game change; original outputs archived."
3. **Paper-bet ledger:** append-only and untouched. Bets already recorded
   keep their v1 predictions; bets from the switch on use v2. The CLV verdict
   (#40) judges the published model's live picks and keeps counting all bets
   as one series. The switch date is recorded, and the verdict report shows
   the before/after split as context only. `bets_retro.*` is left alone.
4. **Shadows:** the roster prior, calibration v2/v3 and portal comparisons
   are paired "every other setting frozen" tests, so both arms are rebuilt on
   the v2 base and window games already scored are rescored. Decision rules,
   windows and dates are unchanged. A superseding DECISIONS.md line records
   the base change. Totals v2 doesn't use the margin ratings and is untouched.
5. **Timing:** the goal is to finish before the Sunday 2026-11-15 cron. If it
   isn't ready, week 11 publishes under v1 and the restatement covers it later.

## Testing

**Unit tests (synthetic data, no DB):**

- Two-stage off → ratings identical to the current single fit.
- Two-stage on, `fcs_weight = 1` → Stage 1 FBS ratings equal the single fit.
- `fcs_weight = 0` → adding or removing an FBS team's FCS game leaves every
  FBS rating unchanged.
- Stage 2, a thin FCS group with no prior → shrinks toward the FCS level, not 0.
- Stage 2 → the FBS offsets it sees are exactly Stage 1's ratings, and they
  are unchanged afterward.

**Existing invariants:** the `cfg` fixture (`tests/conftest.py`) loads
`config/default.toml`, which has two-stage off and `half_life_weeks = 0.0`. So
the existing dataset tests never exercise the published config. Add a
`cfg_two_stage` fixture: `default.toml` plus `fcs_two_stage = true`,
`fcs_weight = 0.3`, `half_life_weeks = 3.7`. Then run these tests under both
fixtures:

- `tests/test_leakage.py`: the max-kickoff assertion, the canary future game,
  and `test_engine_prefix_fit_matches_explicit_slice`. That last test checks
  the engine against an explicit `fit_efficiency` call with no weights. Under
  two-stage, its reference fit gets the same `play_weights`, and it checks the
  FBS rows against Stage 1 and the FCS rows against Stage 2.
- `tests/test_web.py`: `test_web_ratings_match_rank_output`.
- `tuned_v1.toml` still loads, and the `default.toml` defaults don't change
  behavior.

**Done means:** `uv run pytest` passes (dataset tests against the local DB
copy), and the gate numbers and reported items above are written up.
