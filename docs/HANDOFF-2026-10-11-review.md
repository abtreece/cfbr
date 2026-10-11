# Handoff: review of the model, paper bets and site (2026-10-11)

Implementation handoff from a full review of `cfbr`. Everything below was verified by reading code or by running the checks in the appendix against the frozen backtest runs. Nothing here has been built yet.

*Revised 2026-10-11 after a second review: the M1 day-mode as-of rule fixed (the first draft leaked plays from games still in progress), the M1 shadow clarified, W1 downgraded (CFBD normally renumbers a rescheduled game's week), the M2 spec matched to what was measured, the O1 timestamp option corrected, the priority table aligned with the sections, line references fixed.*

Read `CLAUDE.md` and `DECISIONS.md` first. Rules that bind every item:

- **Pre-register before you look.** Any model or bet-rule change gets an evaluation rule written into `DECISIONS.md` *before* the result is computed, with the metric, the data window, and the adopt/reject threshold. Follow the pattern of DECISIONS lines 71–73 and 106–110.
- **2022–2025 is no longer a clean holdout.** It has been viewed for roster, totals, matchup, uncertainty, and (in this review) for the calibration slope and week bias. Treat it as a sanity check only. The only unseen window is **2026 regular-season week 7 onward plus the 2026 postseason**, shared with the roster-prior and totals-v2 shadows. Don't change what those shadows compute.
- **Point-in-time.** `tests/test_leakage.py` must keep passing. Any doubt is a reason to stop.
- **API quota.** Free tier, 1,000 calls/month. Nothing here needs a backfill except item M1's live variant (see there).
- Add a one-line entry to `DECISIONS.md` for every judgment call. Commit per item with the usual prefixes (`Model:`, `Paper bets:`, `Web app:`, `Ops:`).

## Headline finding (context, not a task)

The model carries no out-of-sample information beyond the closing line. OLS of actual margin on `(-spread, pred_margin)`:

| seasons | line weight | model weight | MAE line | MAE model | MAE blend |
|---|---|---|---|---|---|
| 2015–2021 (tuning) | 0.945 | 0.051 | 12.496 | 13.351 | 12.493 |
| 2022–2025 (holdout) | 0.976 | 0.045 | 12.026 | 13.000 | 12.033 |

Holdout ATS, model side vs close, |edge| ≥ 3: 49.6% (n=2,086). By edge bucket on holdout: 0–3 49.1%, 3–6 50.5%, 6–10 48.9%, 10+ 48.5%. The 54.2% "Large" figure in `bets.md` is tuning-season only.

The paper-bet program is testing a hypothesis the backtest already rejects. That doesn't mean stop it (CLV is well powered and is the pre-registered primary metric), but the reporting must say so (B2) and the model work below is where any edge would have to come from.

---

## Priority order

Do these in order. P0 is correctness and data safety; P1 is the two model items with real measured headroom; P2 is the betting EV machinery; P3 is site features that are cheap because the data is already precomputed, plus W1 (moved down from P0: rarer than first thought).

| # | item | size | why first |
|---|---|---|---|
| O1 | backups of `data/raw`, ledger, chain | S | irreplaceable data, no copy |
| B1 | `cfbr bets` can record bets after kickoff | S | correctness |
| B2 | honest holdout numbers in bets.md (do with B1) | S | reporting integrity |
| M1 | finer as-of times (intra-week plays) | M | largest legal information gap |
| M2 | shrinkage-aware calibration | S | measured −0.07 MAE holdout |
| M3 | FCS indicator in calibration | S | measured +3–4 pt bias |
| M5 | backtest report diagnostics | S | would have caught M2/M3 at tuning |
| M4 | cheap ceiling checks (optional) | S each | check, then decide |
| M6 | weekly-run efficiency (optional) | S | ops speed |
| B3 | points→cents table and key numbers | M | prerequisite for B4–B6 |
| B4 | calibrated cover probability | M | prerequisite for any sizing |
| B5 | market-implied power ratings | M | diagnostic |
| B6 | consensus consistency, snapshot cron gaps | S | CLV quality |
| W1 | postponed/rescheduled game handling | S | rare live-site edge case (see W1) |
| W2–W4 | site robustness, features, tests | S–M | visitor value |
| O2–O3 | heartbeat, news archive decision | S | ops hygiene |

---

## P0 — correctness and data safety

### O1. Backups (none exist today)

`data/raw` (2.3 GB, cannot be refetched under the quota), `out/{season}/bets.csv`, `out/{season}/ledger_chain.jsonl`, `out/{season}/ledger_history/`, `data/news/` are all gitignored and no rsync/restic/snapshot exists in crontab, `README.md` or `deploy/`.

- Add a nightly cron (`deploy/` + README) that syncs `data/raw`, `data/cfb.duckdb` (optional, rebuildable), `out/*/bets*.csv`, `out/*/ledger_chain.jsonl`, `out/*/ledger_history/` and `data/news` to an off-machine target. Restic or `rsync -a --delete` to a second disk/NAS/S3 bucket are all fine; the user picks the target.
- **Outside timestamp for the ledger chain** (DECISIONS 116 asks for this): at the end of `_weekly_run` in `src/cfbr/cli.py`, include the chain's latest entry hash (and entry count) in the ntfy message body. ntfy's server keeps the message timestamp, so this is off-machine. Committing the chain to a local git branch does **not** give an outside timestamp (same disk, rewritable); it only counts if pushed to a remote, and a cron job committing in the shared repo conflicts with the worktree workflow, so prefer ntfy. Keep `out/` gitignored for everything else.
- Acceptance: restore drill documented in README; health check (`src/cfbr/bets/health.py`) flags a backup older than 8 days.

### B1. Standalone `cfbr bets` can record a bet after kickoff

`src/cfbr/bets/ledger.py:145` filters candidates with `kickoff_ts > placed_at`, where `placed_at` is the **snapshot** time, never the clock. `cfbr weekly` always takes a fresh snapshot so it is safe. `cfbr bets --season --week` (`src/cfbr/cli.py:365-380`) reuses the newest snapshot; run on a Saturday after a failed Sunday it would place Thursday/Friday bets whose results are known. The integrity chain won't catch it (it only checks that recorded terms never change).

- In `run_bets` (`src/cfbr/bets/run.py`) also require `kickoff_ts > now_utc`; add a `recorded_at` column (ISO, UTC) to the ledger schema. Keep `placed_at` as the line-capture time. `read_ledger` must load old ledgers without the column (forward-compatible, like `season_type` in DECISIONS 77).
- Add `recorded_at` to the immutable-terms list in `src/cfbr/bets/integrity.py`.
- Test in `tests/test_bets.py`: a snapshot older than a game's kickoff produces no bet for that game.

---

## P1 — model: the two items with measured headroom

### M1. Predict with finer as-of times (intra-week plays)

`slot_as_of` (`src/cfbr/data.py:114-117`) sets a slot's as-of to the **first kickoff of the whole CFBD week**. Saturday games are therefore predicted without that week's Tuesday–Friday plays, and CFBD week 1 includes Week 0. Measured on 2024 regular season: median lag from as-of to kickoff 3.7 days; 60% of games kick off more than 2 days after their as-of. The closing line has that information; the model declines to use it. Legal point-in-time data.

Two layers:

1. **Backtest/eval variant (no API cost).** Add `Engine.predict(season, as_of_mode="slot" | "day")`. In "day" mode, a game's as-of is the start of its kickoff day in **US Eastern time** (local midnight, as a UTC instant), never the slot's. Implementation: `SeasonArrays.n_before` already takes any timestamp; group games by (slot, Eastern kickoff date), fit one snapshot per group (~5 extra ridge pairs per week, ~50 ms each), attach.

   **The as-of must exclude games still in progress.** The engine selects training plays by *game kickoff* < as-of (`engine.py:48-52`), and the calibration and Massey baselines select games by kickoff < as-of plus `completed` (`engine.py:228`), where `completed` is the game's eventual state. So any as-of falling mid-game leaks that game's full result. With UTC midnight, a 10:30 PM ET Saturday kickoff (02:30 UTC Sunday) would get every play of a 7 PM ET game still being played. Eastern midnight has the same problem one day over: a 10:30 PM ET Friday game is still running at midnight Saturday. (The slot as-of is safe only because it is the week's first kickoff.) **Hard rule, enforced in code:** the effective cutoff for a day group is `max(slot_as_of, day_as_of − GAME_DURATION_BOUND)` with `GAME_DURATION_BOUND = 5h` (a constant in `data.py`), applied to plays (`n_before(cutoff)`, still a kickoff prefix), recency weights, the calibration set and the Massey set alike. A game is then used only if it kicked off at least 5 h before the target day starts, which covers late-night and Hawaii games.

   Keep the "ratings for week w" display convention unchanged (`ratings()`, `report/rank.py`, the site) — only `predict()` changes.
2. **Live variant (needs ingest).** Live plays arrive only when a week completes (DECISIONS 12), so day-mode predictions in production need a mid-week plays fetch: one extra `/plays` call per (seasonType, week) taken Friday night for the completed Tue–Thu games, ~15 calls/season. Only build this if the ceiling check passes.

**Ceiling check first** (required by the project): run day-mode vs slot-mode on 2015–2021 under the frozen `config/tuned.toml`, report paired MAE gain overall and on games with lag > 2 days, 95% CI. Decide in advance: proceed to a shadow if the overall gain CI excludes 0.

**Evaluation rule to pre-register**: like the roster shadow (DECISIONS 73): window 2026 week 7 onward + postseason, paired MAE gain day − slot, adopt for 2027 if gain > 0 and CI lower bound > −0.10. State in the rule which form the 2026 test takes:

- *Live variant not built (default):* production can't make day-mode predictions, so there is no live shadow. The 2026 test is a walk-forward backtest run after the season on 2026 data, with the as-of rule above. It is still unseen data, but call it a post-season backtest in DECISIONS, not a shadow.
- *Live variant built:* a true shadow, using only plays actually ingested before each game's kickoff (store the fetch time with the snapshot). The backtest variant assumes a game's plays are available 5 h after its kickoff, which the live path can't match exactly. Report both.

Leakage guard: extend `tests/test_leakage.py` per (slot, day) group so it asserts that every training game **ended** before the target day's as-of (`kickoff_ts + GAME_DURATION_BOUND ≤ day_as_of`, or kickoff before the slot as-of), not just that it kicked off before. A max-kickoff assertion alone passes under the UTC-midnight bug above. Add a canary: a game kicking off 3 h before a day group's as-of (a late Friday kickoff vs. Saturday's group) must not reach that group's fit.

### M2. Shrinkage-aware calibration

`fit_calibration` (`src/cfbr/model/calibrate.py:40-46`) fits one `(b_home, b_ppa, b_sr)` for all games, no intercept. The residual structure depends on how much of the rating is still prior (`prior_share`, already computed per game as `prior_share_h/_a`):

| mean prior share | n (tuning) | ideal slope on pred | n (holdout) | ideal slope |
|---|---|---|---|---|
| < 0.3 | 783 | 0.95 | 433 | 0.89 |
| 0.3–0.5 | 2,595 | 0.95 | 1,678 | 0.87 |
| 0.5–0.7 | 1,344 | 1.00 | 833 | 0.94 |
| 0.7–0.9 | 515 | 1.05 | 330 | 1.02 |
| ≥ 0.9 (weeks 1–2) | 607 | 1.14 | 385 | 1.12 |

Prior-based predictions are under-dispersed; data-based ones slightly over-dispersed. A correction fit on tuning only, `resid ~ pred + pred·ps + home·ps` with `ps = (prior_share_h + prior_share_a)/2`, gives coefficients `[-0.178, +0.316, -0.023]` and on holdout: overall MAE 12.995 → 12.927, week 1 14.05 → 13.82, week 3 13.65 → 13.60, weeks 2 and 4 unchanged. This gain is larger than any experiment rejected in DECISIONS.md and consistent across both sets. It also explains the week-1 home bias of +2.5 to +3.3 points better than a week-1 home constant does (a pure week-1/2 home constant fit on tuning gains only 0.008 MAE on holdout).

Implementation — build exactly the model that was measured, a second stage on the base prediction: `pred' = pred + c₁·pred + c₂·pred·ps + c₃·home·ps`, where `pred` is the existing `predict_margin` output and `(c₁, c₂, c₃)` are fit by OLS on `margin − pred` over the same training games and seasons as the base calibration (`schemes.training_seasons`, point-in-time `pred` from each game's own slot snapshot). Store them as `Calibration.c_pred, c_pred_ps, c_home_ps` (default 0 so old pickles/configs load). Don't split the interaction into separate `Δppa×ps` and `Δsr×ps` terms: that is a different model with an extra parameter and wasn't measured. If someone wants to try it, it's a separate pre-registered item. `ps` is available in `season_games_with_snapshots`. `team_ratings_points` (the published ratings) should keep using the base coefficients so the rating definition ("expected margin vs an average team") is unchanged; only game predictions use the interaction. Say so in DECISIONS.

**Evaluation rule to pre-register** (write before running): **primary = the 2026 week 7+ shadow**, paired MAE gain (M2 − base), adopt for 2027 if gain > 0 and CI lower bound > −0.10 (same as M1). Both 2015–2021 and 2022–2025 have now been viewed for this exact correction, so neither can serve as the primary test. Secondary, as a gate for starting the shadow at all: refit under the hybrid scheme and require the 2015–2021 paired MAE gain CI to exclude 0. This is expected to pass and only catches implementation bugs. Note in DECISIONS that both windows were viewed for this item on 2026-10-11.

Caveat on tuning numbers: the tuning-season calibration carries a second-order LOSO leak (a season's finals reach its own calibration through other seasons' LOSO priors; `engine.py:149-159` and `:214-220`). Holdout is clean. Don't try to fix the scheme (user-approved); just report it where tuning MAE is quoted.

### M3. FCS indicator in the calibration

Games with a pooled FCS side in weeks 1–2: residual bias +4.1 (tuning, n=444) and +2.9 (holdout, n=327) — FBS wins by more than predicted. Pooled priors carry last season's final (`src/cfbr/model/prior.py:129-138`) with the same alpha as FBS teams.

- Add `fcs_game` (either side pooled) as a column in `design()`; fit with M2 in the same OLS so there is one calibration change, one rule, one shadow. Expected gain ~0.02 MAE overall, ~0.2 on FCS games.
- Possibly fold the T1 bet-filter logic (FBS vs FBS) into the same indicator later; don't change T1 now.

### M4. Cheap ceiling checks worth one afternoon each (not pre-approved; check, then decide)

All use `results/*/predictions.parquet` residuals or the existing `SeasonArrays`; none need new data.

- **Explosiveness** as a third ridge target (mean PPA on successful plays, or winsorized PPA). Fit the third ridge for 2015–2016, add to the calibration, compare MAE. If the gain is under 0.03, drop it.
- **Special teams points per game** from the dropped play types (`prep/plays.py` drop list) regressed on calibration residuals.
- **Rest/bye days** difference from `games`, regressed on residuals. Expect ≈ 0.
- **Per-team home field**: shrunk team dummies on home-game residuals, out-of-sample within training seasons. Expect ≈ 0 (per-season HFA residual is within ±1 point except 2020).

Already rejected, don't redo: roster prior (71–72), availability (89–90), uncertainty Step A (106–107), totals v1 (108–109), matchup splits (111–112), prior decay by week (113), garbage-time filter (48).

### M5. Backtest report diagnostics

`backtest/metrics.py:104-131 report()` should print, per run, so these patterns are visible at tuning time:

- bias and slope of `margin ~ pred` by week bucket (1, 2, 3–4, 5–8, 9–12, 13+, post) and by prior-share bucket;
- ATS cover by edge bucket (0–3, 3–6, 6–10, 10+) vs the closing line, with Wilson CI;
- the line+model blend weights (OLS of margin on `-spread`, `pred`) — the number the paper bets implicitly bet on;
- a win-probability reliability table (10 bins).

Write these into `metrics.json` and show them on `/accuracy` for finished seasons. Also fix the stale DECISIONS line 28 (Massey now cap 42, alpha 1.0 per tuned.toml), and document that `cfbr rank` on a 2015–2021 season uses the expanding scheme, not LOSO, so it differs from the backtest for those seasons (`engine.py:143-144`).

### M6. Minor efficiency (optional)

- `calibration(2026)` and `priors()` recompute every past season's snapshots on each weekly run (~370 ridge solves, ~20 s, 1.2 GB). A disk cache under `out/cache/` keyed by (config hash, season, last_completed) would cut weekly `rank` to a few seconds. Invalidate when `tuned.toml` or the store changes.
- `Dataset.load` reads `SELECT *` on `scrimmage_plays` (`data.py:73`); project to the ~10 columns the engine uses.
- Log games with zero model plays in `prep` (a handful per season; they feed calibration and Massey but not ratings).

---

## P2 — paper bets

### B2. Honest reporting (do with B1)

In `src/cfbr/bets/ledger.py` (`BACKTEST_TIER_COVER`, line ~51) and `bets.md`/`/bets`:

- Show holdout 2022–2025 tier cover rates next to the tuning ones (compute from `results/20261011T145524Z-tuned-holdout/predictions.parquet`; all are 48.5–50.5%).
- State that the 500-bet minimum is powered for CLV (SE ≈ 0.07 pts), not for win rate. For the win-rate side, compute the bets needed for 80% power from B3's points→cover-probability table and show the working; the first draft's "~6,000 bets for a 2-point edge" was not derived and shouldn't be quoted until B3 exists. Add a Monte Carlo line: "a 500-bet sample from a 50% coin shows ROI ≥ +X% Y% of the time".
- Show `n_providers` at capture and the number of books in the closing consensus beside each bet's CLV (22 of 30 open week-7 bets had a single-book "consensus" at capture).

### B3. Points→cents and key numbers (prerequisite for B4–B6)

A 3-point edge at −4.5 crosses 3 and 7; a 3-point edge at −17.5 crosses nothing that matters. The tracker treats them identically, and best-line selection (`ledger.py:110-116`) takes the better number before the better price, so +3 at −125 beats +2.5 at −105.

- Build `bets/pricing.py`: from 2015–2025 final margins in DuckDB (`games`), the empirical probability mass at each integer margin, then for any (line, half-point move) the cover-probability change and its fair price in cents. Cache the table in `out/`.
- Express each bet's edge and CLV in **cents** (fair-price difference) alongside points; grade the best-line record against the **best-book closing number**, not the consensus close (`ledger.py:280`).
- Best-line selection becomes max expected value over (number, price) across `pricing_books`.
- Rule change scope: this changes *reporting and best-line pricing*, not which games are bet or the consensus record. Record the start date in DECISIONS, as with lines 79 and 96. Existing bets keep their terms.

### B4. Calibrated cover probability

DECISIONS 76 notes the model's own Φ(edge/σ) is far too confident. Fit a logistic regression of `cover ∈ {0,1}` on (edge, |line|, key-number crossing from B3) over the 2015–2021 walk-forward predictions, shrink toward 50%, and compare the implied probability to the price's break-even. Only after this exists does any Kelly or confidence sizing make sense; until then keep flat stakes. Pre-register: it is a reporting column in 2026, and may become the bet rule for 2027 only if its selected subset's live CLV in 2026 week 7+ beats the all-bets CLV with a CI above 0 (same shape as T1/T2, DECISIONS 91–93). Don't add a third tracked test mid-season; this is a 2027 candidate.

### B5. Market-implied power ratings (diagnostic)

Invert closing lines into a Vegas rating per team per week with the Massey ridge (`model/massey.py`, margins = −spread, no cap). Compare to the model's ratings by conference, week, FCS status and home field. Publish as a diagnostic table in `results/` and optionally on the team page for finished seasons. This is the fastest way to find or kill a systematic disagreement, and it costs nothing.

### B6. Consistency and cron gaps

- Capture consensus mixes CFBD + Odds API (`bets/run.py:124-146`); closing consensus is Odds API only (`bets/closing.py:46-50`). Use the same book set for both, or at least record both counts (B2).
- `snapshot_closing` (`closing.py:39-42`) uses only the last pre-kickoff snapshot; if that snapshot lacks the game (book pulled the market), fall back to the previous snapshot within the 12 h window before falling to CFBD.
- Odds-snapshot cron (README): Friday 04:00 UTC kickoffs and Saturday games after 01:00 UTC close 2–6 h stale; Sunday/Monday games are never snapshotted. Add Fri 9 PM CT, Sat 9:30 PM CT and a Sun/Mon run gated on `games_kicking_off`; ~4 credits/week.
- `games_kicking_off` reads DuckDB `games`, rebuilt only on Sunday, so mid-week kickoff changes are invisible. Cheap fix: `odds-snapshot` reads the newest cached CFBD games file instead.
- Odds API team matching (`ingest/odds.py:150-157`): prefix scoring lets "Georgia State Panthers" score 1 against "Georgia"; log a warning when both teams match only at score 1.
- OddsPapi: `pricing_lines` (`ingest/oddspapi.py:120-130`) spends the fixtures call before checking that the budget covers any game; reorder.
- Health check (`bets/health.py`) additions: CFBD lines snapshot freshness; Odds API snapshot discarded for a >6 h gap (bets silently DK-only); zero new bets on a Sunday with games; Odds API event match rate < 90%; backup age (O1).
- Tests missing: `run.current_books` gap logic, `fill_books`, `games_kicking_off`, B1.

---

## P3 — website and ops

### W1. Postponed or rescheduled games (moved from P0)

`completed_weeks` (`src/cfbr/web/build.py:44-51`) and `fetch.completed_play_weeks` (`src/cfbr/ingest/fetch.py:138-150`) require every game in a CFBD week to be `completed`, except that the web side ignores games stale for `STALE_GAME` (`build.py:33`, 4 days). If a game kept its week number but moved to a later date, the week would stay "pending" and `weekly` would rank week N−1 (`cli.py:307-311`) until the makeup was played.

In practice CFBD renumbers a rescheduled game into its new week. Checked in the store: regular-season games more than 10 days from their week's median kickoff number 0 in every season 2014–2026 (2020 included, despite its COVID reschedulings) except 4 in 2025. The remaining risk is a lag between CFBD updating the date and updating the week. **Don't change `assign_slots`**: it would alter historical slotting and the frozen backtests for almost no benefit.

- Live completed-week logic only: in `completed_play_weeks` and `completed_weeks`, treat a game whose `kickoff_ts` is past the following week's first kickoff as belonging to that later week, and log a warning when this fires (it means CFBD hasn't renumbered yet). Mind DECISIONS 12: plays for the original week are then fetched without the moved game, and the moved game's plays arrive with its new week.
- Surface cancelled/postponed state: `templates/team.html:90` and `games.html:141` key only on `completed`; show "postponed"/"cancelled" for stale games, and make `sim/season.py:125-133` skip stale games in live mode as it does in historical mode.
- Tests: `completed_weeks` with one game moved three weeks out but still carrying its old week number; `default_slot` with a stale game.

### W2. Robustness

- `sim/report.py:42-43` writes `season_sim.csv/.md` in place; use tmp+rename like `build.py:427-430`. `store.py:196-203` reads the `.md` unguarded.
- Cold-cache gap: a server restart between the `ratings.parquet` and `meta.json` renames pairs new ratings with old games once. Write all season files to a temp dir and rename the directory, or write `meta.json` last and have `Store` refuse a season whose parquet mtimes are newer than meta.
- `/accuracy` re-reads `accuracy.csv` and recomputes `summarize()` over every season per request (`store.py:208-211`, `app.py:414-418`); `store.runs()` stats every `results/*/metrics.json`. Memoize by mtime like the rest of `Store`.
- `/api/seasons` (`app.py:427`) 500s if a `meta.json` disappears between `seasons()` and `season()`; catch and skip.
- Finished-season pages send no `Cache-Control`/ETag; add `public, max-age=3600` for seasons that are over. Live-season responses must stay `private, no-store` (tested).
- `weekly` exit 3 (quota) aborts before `build_store`/`rank` even when the cache would suffice; let rank proceed on cached data and still exit 3.

### W3. Visitor features, cheapest first (data already in `out/web/{season}/`)

All respect the premium gate: anything derived from live-season ratings goes through `public_view` (`app.py:225-228`) and is free once the season ends.

1. **Explain the rating**: `o_ppa, d_ppa, o_sr, d_sr` are in `ratings.parquet` but never rendered; add a tile row on `team.html` with league percentile for each.
2. **Model vs AP/CFP**: `ap_rank` is in ratings rows; add a sortable `rank − ap_rank` column and a "most over/under-rated" box on `/rankings`.
3. **Upset alerts** on `/games`: Vegas favorite with model `win_prob < 0.5`, or |edge| ≥ 3 (same threshold as the bets, so it doesn't leak more than `/games` already shows).
4. **Head-to-head tool**: two ratings plus the calibrated home constant. Export `b_home` (and after M2, the interaction coefficients) to `meta.json` at build time; pure client-side arithmetic, no server fit.
5. **Last-built timestamp**: `meta.built_at` exists and is shown nowhere; put it in the footer.
6. **Strength of schedule / résumé**: join opponent ratings at build time; show SOS rank, best win, worst loss, record vs top 25 by rating.
7. **Conference standings projections** from the existing sim output (`season_sim.*` has title odds but no projected conference W-L table).
8. **Methodology / about page** generated from `README.md` + the relevant DECISIONS lines; a changelog from `git log` with the phase prefixes.
9. Cross-team rating-history comparison chart (data present).

### W4. Tests to add

Build atomicity (mid-rebuild read), `completed_weeks`/`season_over` with a postponed game (W1), `default_slot` with a stale game, `accuracy_log`/`runs` memoization, `_weekly_run` exit-code ordering (3/5, web build last) directly rather than via the alerts wrapper, `/api/seasons` with a missing meta.

### O2. Heartbeat

Alerts fire on failure only. Add a dead-man's switch: `weekly` posts a success ping (ntfy or a healthchecks.io-style URL), and a separate daily cron (or the health check) alerts if no successful run is recorded since the last Sunday during the season. Also rotate `data/*.log` (the weekly log includes full prep output per run).

### O3. News/availability archive: decide

`src/cfbr/news/` is read by nothing but its own CLI commands (`cli.py:506-537`). The hourly cron runs headless Chromium for 5 conference pages and 8 sources, 10 MB so far. The availability feature it was meant to feed was rejected (DECISIONS 90), and the remaining idea (pre-game official reports, DECISIONS 82) needs SEC/Pac-12/MWC/Sun Belt/CUSA coverage the config says is unavailable. Options: (a) keep collecting as a cheap archive for a later point-in-time join on `first_seen_at`; (b) pause the cron and keep the code. Needs the user's call; don't delete data without it (see DECISIONS 88 for the purge rule when a source is dropped).

---

## Appendix: reproducing the review numbers

All from the frozen runs; nothing refit. Run with `uv run python`.

```python
import polars as pl, numpy as np
def prep(f):
    p = pl.read_parquet(f).filter(pl.col("completed") & pl.col("margin").is_not_null())
    return p.with_columns(line=-pl.col("spread"), ps=(pl.col("prior_share_h")+pl.col("prior_share_a"))/2)
t = prep("results/20261011T144444Z-tuned-tuning/predictions.parquet")
h = prep("results/20261011T145524Z-tuned-holdout/predictions.parquet")
mae = lambda y, yh: float(np.mean(np.abs(y - yh)))

# Blend weights: does the model add anything to the line?
for name, p in [("tuning", t), ("holdout", h)]:
    q = p.filter(pl.col("line").is_not_null())
    X = np.column_stack([q["line"], q["pred_margin"]]); y = q["margin"].to_numpy()
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(name, c, mae(y, X[:, 0]), mae(y, X[:, 1]), mae(y, X @ c))

# Shrinkage-aware calibration: fit on tuning, apply to holdout
def XY(p):
    y = p["margin"].to_numpy(); x = p["pred_margin"].to_numpy(); ps = p["ps"].to_numpy()
    hm = (~p["neutral_site"]).cast(pl.Float64).to_numpy()
    return y, x, np.column_stack([x, x*ps, hm*ps])
y, x, X = XY(t); c, *_ = np.linalg.lstsq(X, y - x, rcond=None)
y, x, X = XY(h); print("holdout", mae(y, x), "->", mae(y, x + X @ c))

# ATS by edge bucket on holdout
q = h.filter(pl.col("line").is_not_null()).with_columns(edge=pl.col("pred_margin")-pl.col("line"))
q = q.with_columns(res=pl.col("margin")-pl.col("line")).filter(pl.col("res") != 0)
q = q.with_columns(win=((pl.col("res") > 0) == (pl.col("edge") > 0)),
                   b=pl.when(pl.col("edge").abs()<3).then(0).when(pl.col("edge").abs()<6).then(3)
                      .when(pl.col("edge").abs()<10).then(6).otherwise(10))
print(q.group_by("b").agg(n=pl.len(), win=pl.col("win").mean()).sort("b"))
```

Slot lag (M1), 2024 regular season:

```python
from cfbr.config import load_config; from cfbr.data import Dataset, slot_as_of
cfg = load_config(); ds = Dataset.load(cfg)
g = ds.pooled_games(cfg).filter((pl.col("season")==2024)&(pl.col("season_type")=="regular"))
g = g.join(slot_as_of(g).select("season","slot","as_of"), on=["season","slot"])
lag = (g["kickoff_ts"]-g["as_of"]).dt.total_hours()/24
print(lag.median(), (lag > 2).mean())   # 3.7 days, 0.60
```
