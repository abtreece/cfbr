# Handoff: follow-ups from the track-record review (2026-10-31)

What is left after PRs #37–#40 (opening-line comparator, calibration v3, portal-era roster prior, closing-line-value verdict) were merged and deployed to javelin on 2026-10-31 (deploy 197ca2a → e2d77fc; 408 tests passed on real data; both frozen runs refreshed with `cfbr backtest-lines --run …`; `cfbr-web` restarted). The plan those PRs executed is `docs/superpowers/plans/2026-10-30-track-record-improvements.md`; the rules and results are in `DECISIONS.md` (lines cited below are as of e2d77fc; search by the quoted phrase if they move).

Every item below was verified against the code at e2d77fc. Nothing here has been built yet.

Read `CLAUDE.md` and `DECISIONS.md` first. Rules that bind every item:

- **Pre-register before you look.** Any change to a window, metric or threshold gets its own `DECISIONS.md` line, committed alone, *before* the code or the result. Result lines cite the rule's commit hash. The four pre-registration hashes from this round (`5738f8f` v3 rule, `859c017` portal rule, `d35e0ac` verdict rule) survive only as PR head refs after the squash: `git fetch origin refs/pull/38/head` (resp. 39, 40) to see them.
- **Nothing is deployed without asking.** `deploy/deploy.sh` is a stop-and-ask step; a shadow file that must exist before a deadline is a reason to deploy early, not to run anything by hand on javelin.
- **Ledger schema changes are a user decision** (CLAUDE.md "When to Stop and Ask"). Item 2 is one.
- **2022–2025 is a sanity window, not a test.** 2026 weeks 7+ and the postseason belong to the shadows fixed on 2026-10-11; no new variant enters them.

---

## Deadlines

| when | item | kind |
|---|---|---|
| **Tue 2026-11-03 18:00 CT** (first week-10 kickoff) | 1. Calibration v3 shadow window: amend or accept | user decision; code only if amended |
| before the first 2027 `cfbr weekly` (pricing table gains 2026) | 2. CLV verdict: pooling, per-bet freeze, decision freeze | ledger schema change; stop-and-ask |
| before 2027 week 1 | 3. Shadow decisions already scheduled (roster, v2, v3, totals) | user decisions on pre-registered rules |
| before the 2027 preseason | 4. 2027 roster-prior candidate needs its own rule | pre-registration |
| before 2028 week 1 | 5. Portal-era roster prior decision | user decision; nothing to build |

---

## 1. Calibration v3 shadow window (decide before Tue 2026-11-03 18:00 CT)

**Where it stands.** The v3 shadow (`out/{season}/calibration_v3_comparison.{csv,md}`) is written by `cfbr weekly` from this deploy; the first file appears after the Sunday 2026-11-01 run. Its window is `in_window_v3()` in `src/cfbr/report/shadow.py` (`V3_WINDOW_FIRST_WEEK = 10`, `WINDOW_SEASON = 2026`, plus the postseason). Adoption rule (DECISIONS line 187): window gain v3 − v2 > 0 with 95% CI lower bound > −0.10, decided with v2 before 2027 week 1, v3 only if v2 is adopted.

**The problem** (DECISIONS line 188, "Limitation of the shadow window"): regular-season weeks 10+ plus bowls hold about 5 FBS-vs-FCS games in 2026, and 637 of the 693 tuning-season FCS games fall outside that window. So the window measures v3's side effects on FBS games (the joint refit moves the `home_ps` and `fcs` coefficients for every game; mean |v3 − v2| ≈ 0.36 pts in the window), not the FCS slope v3 exists to fix. On the same window in past seasons v3 vs v2 was −0.022 (2015–2021) and −0.002 (2022–2025), so expect a close or slightly negative reading as written. Also disclosed there: the primary 2015–2021 test was not blind (an ad-hoc LOSO look on 2026-10-30 preceded the rule), so the 2022–2025 sanity (+0.017 [−0.024, +0.057]) is the only clean estimate.

**Options.** Either is acceptable; (a) needs no code.

- **(a) Accept the limitation.** Add nothing. The decision before 2027 week 1 is then read as "v3 does not hurt FBS games" (CI lower bound > −0.10) rather than as evidence for the slope fix. The gate record (`results/20261031T034624Z-calib-v3-gate`, `…T034712Z-calib-v3-sanity`) stays the evidence for the slope.
- **(b) Pre-register an amendment.** One DECISIONS line, committed alone before 18:00 CT Tuesday, then the code. Two candidates discussed in the #38 review:
  - *FCS-games row, reported not gated:* add to `calibration_v3_markdown` a row for window games with exactly one pooled-FCS side (gain, n, slope of margin on prediction under v2 and v3). With ~5 games it can't gate anything; it just makes the slope visible on fresh data.
  - *FBS non-inferiority as the criterion:* restate the adoption test as "v3 − v2 on window games with no pooled-FCS side has CI lower bound > −0.10", which is what the window can actually measure, and let the slope evidence rest on the gate.
  Code for either: `src/cfbr/report/shadow.py` (`calibration_v3_comparison` already carries the `fcs` term through `stage2_design`, so the FCS flag is derivable from the design, or add an `fcs` column to the comparison frame), `tests/test_report.py::test_calibration_v3_shadow_window_and_gain`, and the method copy in `src/cfbr/web/templates/method.html` if the criterion changes. Don't touch `V3_WINDOW_FIRST_WEEK`: moving the window earlier would include week 7–9 games whose v3 predictions were never shadowed point-in-time.

**Don't:** rerun `cfbr shadow --season 2026` on javelin by hand to backfill; the weekly run writes the file, and the window opens at week 10 either way.

---

## 2. CLV verdict: cross-season pooling, per-bet ROI freeze, decision-run freeze (before the first 2027 weekly)

**Where it stands.** `src/cfbr/bets/verdict.py` implements the rule in DECISIONS line 192 (300 eligible bets; edge if the 95% CI lower bound of mean closing-implied ROI > 0, no edge if the upper bound < 0, else undecided; decision point = first weekly run after the 2027 regular season with n ≥ 300, else 2028). It is shown in `bets.md` (`ledger.summary_markdown`, which imports `verdict` inside the function), on `/bets` and in `/api/bets` (`web/bets.py`). Today: n = 57, mean −0.87% [−3.64%, +1.90%], at-close reference −4.39%, status "too few". The three gaps are recorded in DECISIONS line 193.

**Gap A — one season only.** `verdict()` receives the ledger `run.py` and `web/bets.py` hand it, which is `out/{season}/bets.csv` for a single season (`Store.bets(season)`, `web/store.py`). The rule pools every settled bet from the 57 on, across seasons. Nothing pools today; 2027's verdict as written would start from zero.

**Gap B — the ROI is recomputed from a moving table.** `closing_implied_roi` calls `PricingTable.expected_value` at read time with whatever `out/pricing_table.json` holds. `run.py:108` rebuilds that table "when its seasons change" (completed seasons; it is 2021–2025 now and will take in 2026 at the first 2027 run). So every historical bet's ROI, and the CI the verdict is judged on, silently changes when the table does. `web/bets.py:season_pricing_table` already refuses to price a past season with a table that includes that season (the page then shows "Verdict not recomputed for past seasons"), which is the right instinct but leaves past seasons with no verdict at all.

**Gap C — no final state.** The status is recomputed every run; nothing detects the decision run or records the pre-registered reading once.

**Design sketch** (a proposal to decide on, not a rule; the metric and threshold stay as pre-registered):

1. *Freeze per bet at settlement.* Add two nullable columns to `LEDGER_SCHEMA` in `src/cfbr/bets/ledger.py`: `closing_implied_roi` (the bet's ROI at the table in force when it settled) and `at_close_roi` (the same table's price of taking the close itself; today's `_at_close` reference). Set them in `settle()` (`ledger.py:285`) using the `table` it already receives. `read_ledger` back-fills missing columns with nulls (`ledger.py:334`), so old ledgers load; `tests/test_bets.py::test_legacy_ledger_without_new_columns_loads` covers that path and needs the new columns.
2. *Backfill the 57 (and every 2026 bet settled before the change) once, under the 2021–2025 table,* with a one-off command or a guarded step in `cfbr bets` that fills only null `closing_implied_roi` on settled rows. This must land **before the pricing table takes in 2026**, i.e. before the first 2027 weekly run; after that the historical ROI can no longer be reproduced from the table the rule was written against. Record the backfill (date, table seasons, n) in DECISIONS.
3. *Pool across seasons.* `verdict()` takes the concatenation of every `out/*/bets.csv` (`Store.bet_seasons()` lists them) and reads the frozen column instead of recomputing; `_eligible` stays (settled, `closing_source == "snapshot"`, non-null close). `at_close` becomes the mean of the frozen `at_close_roi`. The `table` argument becomes unnecessary for the verdict, which also removes the `season_pricing_table`/`verdict_withheld` special case on `/bets`.
4. *Freeze the decision.* When a weekly run after the 2027 regular season first sees n ≥ 300, write `out/verdict.json` (status, n, mean, sd, CI, at_close, run timestamp, rule hash `d35e0ac`) and have every later run, `bets.md` and `/bets` display that record rather than recompute. If the status is "undecided" at that run, the rule says it continues: keep recomputing and write the file only when the CI clears 0 on either side or at the 2028 decision point. The 2028 rollover needs its own wording in the same DECISIONS line.
5. *Surfaces.* `bets.md` verdict section (`ledger.py` ~591), `/bets` template `templates/bets.html` ~119–121 (the "not recomputed for past seasons" message goes away), `/api/bets` (`web/bets.py:518`), `bets_guide.html` copy, README line 185.

**Tests to update:** `tests/test_verdict.py` (statuses; the 300 pin at line ~77 stays), `tests/test_bets.py::test_summary_markdown_has_the_verdict_section`, `tests/test_web.py::test_bets_page_and_api_carry_the_verdict`, `::test_bets_computed_verdict`, `::test_bets_verdict_withheld_for_past_season` (becomes a pooled-verdict test), plus the legacy-ledger load test. `tests/test_web.py` also asserts no page outside `web/bets.py` carries ledger data; `out/verdict.json` must be read only there.

**Known and kept as-is** (DECISIONS line 193 (2) and (3)): per-bet SD of closing-implied ROI is 0.107, so at n = 300 the CI half-width is about ±1.2%, not the ±0.6% the rule assumed; the at-close reference varies per bet (−14.9% to +5.8%; 5 of 57 positive) because `PricingTable.margin_pmf` buckets by `int(round(|center|))` (half-to-even) with ±2-point pooling and favorites and underdogs cover at different rates. Both were seen after the rule, so neither the threshold nor the metric changes.

---

## 3. Decisions already scheduled before 2027 week 1 (nothing to build; don't forget them)

All pre-registered in DECISIONS on 2026-10-11 and 2026-10-30; the weekly job writes the files. Read each file once, after the 2026 postseason, and record the decision as a DECISIONS line citing the rule.

| shadow | file | window | adopt if | rule |
|---|---|---|---|---|
| roster prior (plain) | `out/2026/prior_comparison.*` | 2026 reg. wk 7+ & post | gain v1−roster > 0, CI low > −0.10 | line 73 |
| calibration v2 | `out/2026/calibration_comparison.*` | same | gain base−v2 > 0, CI low > −0.10 | line 122 |
| calibration v3 | `out/2026/calibration_v3_comparison.*` | 2026 reg. wk 10+ & post | gain v2−v3 > 0, CI low > −0.10, **and v2 adopted** | lines 187–188, item 1 above |
| totals v2 | `out/2026/totals_comparison.*` | wk 7+ & post | gain > 0, CI low > −0.10 | line 110 |

Matchup ratings (line 112) and team-specific home field (line 170) both failed their 2015–2021 gates on 2026-10-11/12 and have no shadow; nothing to decide there.

Interactions written into the rules: if two of these are adopted, their **combination is checked on 2015–2021 before 2027 week 1** (line 122). Whichever prior is published in 2027 becomes the "published" column of the portal shadow (item 5). Adopting any of them is a `config/tuned.toml` change, which is otherwise frozen; do it as its own PR with the DECISIONS line first.

After the 2026 season ends, **do not rerun `cfbr shadow --season 2026`** after a published-config change: it recomputes the 2026 "published" columns under the new config (README "Prior shadow comparison").

---

## 4. 2027 roster-prior candidate (rule before the 2027 preseason)

DECISIONS line 98: career-weighted player production (last 2–3 seasons) plus QB experience (career pass attempts, starts). Not in any 2026 window on purpose. It needs its own evaluation rule written before the 2027 preseason, judged against v1 and the current roster prior on data not used to design it. The 2026 window and 2022–2025 are both spent for roster features, so the rule has to name a window that is still clean (2027 regular season, shared with the portal shadow, which the portal rule permits only for the prior it names; resolve that in the rule). Code entry points: `src/cfbr/model/prior.py:fit_prior_models(train, mode, roster_min_season)`, roster feature build under `ingest/`/`prep/` (see `tests/test_roster.py`).

---

## 5. Portal-era roster prior (decide before 2028 week 1)

Running since this deploy: `out/{season}/prior_portal_comparison.{csv,md}`, three columns (published, plain roster, portal). Window is **2027 only** (`in_window_portal()` in `shadow.py`, `season == 2027`); the 2026 rows are kept for the record and never enter the status (DECISIONS line 191 superseded the 2026-postseason part of `859c017`). Adopt only if portal beats the published model (gain > 0, CI low > −0.10) **and** beats plain roster (gain > 0) over the window. Context recorded on line 190: head-to-head on 2022–2025 portal loses to plain roster (−0.028), so the hypothesis is unsupported going in. Nothing to build; if the plain roster prior is adopted for 2027, confirm the "published" column tracks it (`portal_comparison` reads `prior_comparison.csv`, which already follows the published config).

---

## Noted, not planned

- The two frozen backtest runs (`results/20261011T144444Z-tuned-tuning`, `…T145524Z-tuned-holdout`) predate the 2026-10-11 books-only closing-consensus fix, so their stored close differs from the current table on ~8% of games (48.5% of tuning-season games have a different `n_providers`). `refresh_lines` works around this by adding only the opener block and comparing opener to a books-only current close. Re-freezing the runs would change headline metrics and touch the holdout; not proposed.
- CFBD opening lines are Bovada-only for 2021–22 (DraftKings from 2023, ESPN Bet 2024–25), disclosed in DECISIONS and on `/lab`; no fix available on the free tier.
