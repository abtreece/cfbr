# CFB Ratings Engine: Build Spec

## Objective

Build a predictive college football rating system that produces a weekly ranking of all FBS teams. A team's rating is its expected point margin against an average FBS team on a neutral field. Accuracy is measured out-of-sample: how well the ratings predict future game margins, compared against Vegas closing lines.

v1 is a transparent, backtestable statistical model. It runs on CPU. No neural nets. No LLM is involved in producing a rating.

## Operating rules for the agent

- Work through the phases in order. Each phase has a Definition of Done (DoD). Do not start a phase until the previous phase's DoD passes.
- When a choice isn't specified here, pick the conventional option, record it in `DECISIONS.md` (one line each: decision, reason, phase), and keep going. Stop and ask only if something would make the backtest invalid. The main example is any doubt about data leakage.
- Never use betting lines as model inputs. They are evaluation data only.
- Every prediction for a game in week `w` of season `s` must be produced using only data that was available before week `w` kicked off. This is the most important correctness property in the project, and it gets tested explicitly (Phase 5).
- Commit at the end of each phase with a message that names the phase.
- Before relying on any CFBD endpoint or field, verify its name and shape against the live API docs (https://apinext.collegefootballdata.com). The names in this spec are believed correct but have not been verified.

## Environment

- Host: Dell PowerEdge R730 (dual Xeon) with an NVIDIA T4 (16GB). v1 uses the CPU only. The GPU is reserved for Phase 8.
- Python 3.12. Use `uv` for dependency management.
- Core deps: `cfbd` (official client) or `httpx`, `duckdb`, `polars` (or pandas), `numpy`, `scipy`, `scikit-learn`, `pydantic`, `typer`, `pytest`.
- Read the API key from `CFBD_API_KEY`. Never commit it.

## Repo layout

```
cfb-ratings/
  pyproject.toml
  DECISIONS.md
  README.md
  config/default.toml        # all tunables live here
  src/cfbr/
    ingest/                  # CFBD client, caching, loaders
    prep/                    # play filtering, garbage time, FCS pooling
    model/                   # ridge fit, priors, margin calibration
    backtest/                # walk-forward harness, metrics
    report/                  # ranking output
    cli.py
  data/
    raw/                     # cached API responses (gitignored)
    cfb.duckdb               # normalized store (gitignored)
  tests/
```

## Phase 1: Ingestion and storage

Pull from CFBD into DuckDB for seasons 2014 through the current season (2014 is the first year returning-production data exists).

| Dataset | Endpoint (verify) | Grain |
|---|---|---|
| Teams / FBS membership by year | `/teams/fbs?year=` | team-season |
| Games | `/games?year=&seasonType=` | game |
| Plays (with PPA/EPA) | `/plays?year=&week=` | play |
| Betting lines | `/lines?year=` | game-provider |
| Returning production | `/player/returning?year=` | team-season |
| Team talent | `/talent?year=` | team-season |
| Recruiting team rankings | `/recruiting/teams?year=` | team-season |
| Calendar | `/calendar?year=` | week |

Requirements:
- Cache every raw response under `data/raw/{endpoint}/{year}/{week}.json`. Re-runs use the cache unless `--refresh` is passed. Check the account's call quota on the CFBD tiers page and log the number of calls each command makes. The full backfill has to fit within the free-tier quota, possibly spread over several days. If it can't, write that down in DECISIONS.md and implement a resumable backfill.
- Store each game's kickoff timestamp and each team's FBS/FCS status for that year. The leakage guard depends on both.
- Include postseason games.
- Weekly incremental ingest: `cfbr ingest --season 2026 --week N`.

DoD: all tables are populated for 2014 through the latest completed week; row-count sanity checks pass (roughly 750+ FBS games per season, roughly 120k+ scrimmage plays per season); a second run of ingest makes zero API calls.

## Phase 2: Play preparation

Build a clean play-level table, `model_plays`, with one row per scrimmage play.

- Keep: rushes, passes (completions, incompletions, sacks, interceptions), and fumbles on scrimmage plays. Drop: kickoffs, punts, field goals, extra points, timeouts, end-of-period rows, and penalty-only plays with no snap result. Record the exact play-type mapping in DECISIONS.md.
- Garbage-time filter (configurable). Drop a play when the score margin at the snap is greater than: Q2 38, Q3 28, Q4 22. No filter in Q1. Overtime plays are kept.
- Columns: `season, week, game_id, kickoff_ts, offense, defense, offense_fcs, defense_fcs, home_team, neutral_site, ppa (EPA), success (bool)`.
- Success definition: on 1st down, gain ≥ 50% of yards to go; on 2nd down, ≥ 70%; on 3rd/4th down, 100%.
- FCS handling: replace every FCS team with a single pooled `FCS` entity. Make this configurable so tiered pooling (by FCS conference) can be tried later.
- Rows with null PPA: drop them and log the share dropped per season.

DoD: `model_plays` is built for all seasons; unit tests cover the garbage-time thresholds, the success definition, and FCS pooling; the null-PPA share is under 3% for every season (otherwise investigate and record the finding).

## Phase 3: Core rating model

### 3a. Opponent-adjusted efficiency (ridge)

Fit one regression per as-of date:

```
ppa_play = intercept + O[offense] - D[defense] + H * home_indicator + error
```

- One-hot coefficients for each team's offense and each team's defense. `home_indicator` is +1 when the offense is the home team, -1 when the defense is the home team, and 0 at a neutral site.
- Ridge penalty `alpha` (tuned in Phase 6). Use scipy sparse matrices together with `sklearn.linear_model.Ridge` (or a direct sparse solve).
- **Shrink toward the prior, not toward zero.** Implement this by fitting on residuals: subtract each team's prior O and D values (Phase 4) from the target first, fit the ridge on what remains, then add the priors back. As data accumulates, the ridge solution moves away from the prior automatically.
- Optional sample weights: recency decay within the season (`half_life_weeks`, default off). Tune in Phase 6.
- Fit the same model a second time with `success` (as 0/1) as the target, which gives success-rate ratings O_sr and D_sr.

### 3b. Margin calibration

Map the efficiency ratings to points. On training seasons, using as-of ratings only, fit a linear regression:

```
game_margin = b0*home + b1*(O_ppa_home - D_ppa_away) - b2*(O_ppa_away - D_ppa_home) + (same terms for success rate) + e
```

Simplify where the data supports it, for example by using a single net rating per team. A team's **Rating** is its predicted margin against the average FBS team on a neutral field. Record the residual SD `sigma` for computing win probabilities, `P(win) = Phi(pred_margin / sigma)`.

### 3c. Baseline for comparison

Also implement a score-only Massey/SRS model: ridge regression on game margins, with margins capped at a configurable value (default 28). This is the baseline the efficiency model has to beat.

DoD: given `(season, week)`, the model returns a ratings table for all FBS teams plus pooled FCS; fitting one week takes under 10 seconds on the R730; tests confirm that with synthetic data generated from known team strengths, the model recovers those strengths (correlation > 0.95).

## Phase 4: Preseason prior

For each season `s`, build each team's prior O and D values from features that were known before week 1 of season `s`:

- the team's final rating from season `s-1` (offense and defense separately)
- returning production (offense and defense percentages)
- team talent composite
- a 4-year weighted average of recruiting team rating

Fit a linear model on past seasons that predicts the final ratings of season `s` from these features. Use leave-one-season-out fits so that no season's prior is built from its own outcome. New FBS programs get the average of the bottom quartile; record this in DECISIONS.md.

How strongly the prior pulls on the in-season fit is controlled by the ridge `alpha`, applied to the residual-from-prior parameterization above. Optionally the prior's influence can also be weighted by games played (`prior_weight = k / (k + games_played)`); tune `k` in Phase 6. The goal is that the prior's influence is small by week 7 or 8. Verify this in the backtest output.

DoD: the prior alone (week 1 predictions) is backtested, and its MAE is reported per season.

## Phase 5: Walk-forward backtest harness

`cfbr backtest --seasons 2015-2025 --config config/default.toml`

For each season and each week `w` from 1 through the end of the postseason:
1. Compute the as-of time as the earliest kickoff of week `w`.
2. Fit priors and ratings using only plays and games with a kickoff before the as-of time.
3. Predict every game in week `w`.

Metrics, reported per season, per week bucket (weeks 1–3, 4–7, 8+, postseason), and overall:
- MAE and RMSE of the predicted margin
- Log loss and Brier score of win probability
- Straight-up winner accuracy
- MAE of the closing spread (choose a consensus provider or the median across providers; record the choice) on the same games, as the benchmark
- Win rate against the closing spread, along with its binomial confidence interval. Without a confidence interval the number is noise.

**Leakage tests (required):**
- A test asserts that for a sampled set of (season, week) pairs, the maximum `kickoff_ts` among training rows is earlier than the as-of time.
- A canary test: inject a fake future game with an extreme margin and confirm that ratings for earlier weeks don't change.

Outputs go to `results/{run_id}/`: a metrics JSON file, per-game predictions as parquet, and a resolved copy of the config.

DoD: the backtest runs end to end; leakage tests pass; the efficiency model beats the Massey baseline on overall MAE.

## Phase 6: Tuning

- Tuning seasons: 2015–2021. Holdout seasons: 2022–2025. Do not look at holdout metrics until the tuning config is frozen.
- Tunables: ridge `alpha` (for the PPA and success-rate fits), garbage-time thresholds, `half_life_weeks`, prior weight `k`, the FCS pooling strategy, and the margin cap (baseline model only).
- Run each search in parallel across CPU cores (joblib). Use a grid or a small Bayesian search (Optuna). The objective is overall MAE on the tuning seasons.
- Write `results/tuning_report.md`, which states the best config, how sensitive MAE is to each parameter, and the frozen holdout metrics.

**Targets (expectations, not pass/fail):** closing lines typically have an MAE of roughly 12–13 points. A good v1 lands within about 1 point of the closing line overall and gets close to it after week 8. Report the actual numbers honestly whatever they are.

DoD: the config is frozen in `config/tuned.toml`; holdout metrics are reported.

## Phase 7: Weekly ranking output

`cfbr rank --season 2026 --week N`

Output files:
- `out/2026/wNN/rankings.csv` with columns: rank, team, conference, record, rating, offense rating, defense rating, rating change from last week, rank change, prior share (how much of the rating still comes from the prior)
- `out/2026/wNN/rankings.md` with the top 25 plus full table
- `out/2026/wNN/predictions.csv` with next week's games: predicted margin, win probability, and the closing spread once it's available (the spread is for display only)

Also provide `cfbr weekly`, which runs ingest, then rank, then a backtest update for the season so far, in that order. The command should be safe to run from cron on Sunday mornings Central time.

DoD: running `cfbr weekly` for the current 2026 week produces all output files, and a 2026 season-to-date accuracy line is appended to `out/2026/accuracy.csv`.

## Phase 8 (stretch, not part of v1): flagged extensions

Implement only after v1 has shipped. Each one must beat the frozen v1 on the holdout seasons before it gets merged.

1. **Hierarchical Bayesian model (GPU).** Write the model in NumPyro/JAX with partial pooling at the team, conference, and FCS levels, a prior defined in the model rather than through the ridge penalty, and NUTS sampling on the T4. Output a posterior distribution for each team's rating.
2. **Monte Carlo season simulation.** Draw from the rating posteriors (or from the v1 ratings with `sigma`) and simulate the remaining schedule at least 100k times to get conference title and playoff probabilities. Use JAX or CuPy on the T4.
3. **Availability adjustments.** An extraction step run by a local LLM (a quantized 7–8B model on the T4) reads injury and depth chart news and outputs structured QB availability per team, with a confidence score. Apply it as an adjustment to the predicted margin, with the size of the adjustment fit on historical data. This requires building a timestamped news archive first, and adjustments must be point-in-time correct to avoid leakage.

## Non-goals

- Resumé-style or committee-mimicking rankings.
- A web UI. File outputs are enough for v1.
- Live in-game data.
- Automated bet placement of any kind. (Amended 2026-10-11: a paper-trading tracker that flags model-vs-line edges, records hypothetical bets and grades them is in scope; it never places a bet, and betting lines still never feed the ratings.)
