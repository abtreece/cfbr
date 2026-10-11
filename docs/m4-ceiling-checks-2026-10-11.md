# M4 ceiling checks (2026-10-11)

Residual = margin − pred_margin from the frozen backtests (`results/20261011T144444Z-tuned-tuning`, 2015–2021, primary; `results/20261011T145524Z-tuned-holdout`, 2022–2025, sanity only). Every feature is point-in-time: season-to-date quantities use only games/plays with kickoff < the slot's `as_of`. Evaluation: OLS of residual on the feature, leave-one-season-out within 2015–2021 (fit on 6 seasons, predict the 7th); holdout uses a strict expanding window (fit on all seasons < s). Gain = mean(|r| − |r − fitted|), paired, 95% CI = ±1.96·sd/√n. Threshold: gain < 0.03 MAE → drop. Baseline MAE: 13.382 (tuning, n=5844), 12.995 (holdout, n=3659). Intercept-only reference gain: −0.002 (tuning), so none of the gains below are mean-shift artifacts.

Scripts: `common.py`, `check1_explosive.py` (+ `check1b_holdout.py`), `check2_st.py`, `check3_rest.py`, `check4_hfa.py` in this directory. Total runtime ≈ 2 min; no API calls.

## Summary

| check | feature | tuning gain (LOSO 2015–21) | holdout gain (2022–25) | verdict |
|---|---|---|---|---|
| 1 explosiveness | Δ ridge PPA-on-successful-plays (net) | −0.003 [−0.012, +0.006] | +0.008 [−0.005, +0.021] | **drop** |
| 1 explosiveness | same, incremental over prior-less PPA net | +0.008 | +0.022 | **drop** (< 0.03 on primary) |
| 2 special teams | Δ ST net points/game, Δ FG make rate | −0.007 [−0.016, +0.003] | +0.002 [−0.006, +0.010] | **drop** |
| 3 rest/bye | Δ rest days, Δ bye indicator | −0.003 [−0.014, +0.007] | +0.002 [−0.012, +0.017] | **drop** (≈ 0, as expected) |
| 4 per-team HFA | shrunk home-team dummies, best λ=300 | +0.003 [−0.006, +0.012] | +0.012 [−0.000, +0.025] | **drop** (≈ 0, as expected) |

None of the four candidates reaches the 0.03 threshold on the primary set. Do not build any of them.

## 1. Explosiveness as a third ridge target

Built properly, not proxied: for every (season, slot) `as_of`, I reused `Engine.arrays(season)` and `fit_efficiency_arrays` with the tuned `alpha_ppa`, the tuned recency weights and FBS centering, no prior (prior-less `_fit` structure, as the handoff suggested), on three targets: (x) PPA restricted to successful plays, (w) PPA winsorized at ±3 on all plays, and (p) plain PPA as a reference. Δ = (o_h − d_a) − (o_a − d_h). Runtime was 9 s per three seasons, so I ran all of 2015–2021 (and 2022–2025 for sanity) rather than subsampling.

| feature set | n | MAE before | MAE after | gain [95% CI] | coefficients (full tuning fit) |
|---|---|---|---|---|---|
| Δx (explosive) alone, tuning | 5844 | 13.382 | 13.385 | −0.003 [−0.012, +0.006] | x: +6.2 |
| Δx alone, holdout | 3659 | 12.995 | 12.987 | +0.008 [−0.005, +0.021] | |
| Δw (winsorized) alone, tuning | 5844 | 13.382 | 13.347 | +0.035 [+0.007, +0.064] | w: −14.7 (corr(Δw, Δp) = 0.98) |
| Δp (prior-less PPA, reference) alone, tuning | 5844 | 13.382 | 13.343 | +0.039 [+0.009, +0.069] | p: −13.0 |
| Δx + Δp, tuning | 5844 | 13.382 | 13.336 | +0.047 [+0.014, +0.079] | x: +10.5, p: −13.9 |
| Δx + Δp, holdout | 3659 | 12.995 | 12.885 | +0.110 [+0.063, +0.157] | |
| Δp + pred_margin, tuning | 5844 | 13.382 | 13.300 | +0.082 [+0.035, +0.129] | p: −34.0, pm: +0.16 |
| Δx + Δp + pred_margin, tuning | 5844 | 13.382 | 13.278 | +0.104 [+0.053, +0.155] | x: +18.5, p: −38.2, pm: +0.18 |
| Δx + Δp + pred_margin, holdout | 3659 | 12.995 | 12.858 | +0.137 [+0.065, +0.209] | |

Explosiveness itself fails: alone it explains nothing (coefficient ≈ 0, gain −0.003), and its incremental gain over the reference feature is +0.008 (tuning) / +0.022 (holdout), both under 0.03; its coefficient is unstable across specifications (6 → 10 → 18). The winsorized variant is 0.98-correlated with plain PPA and carries no explosiveness information. Weeks ≥ 4 only: Δx +0.015 [−0.009, +0.039].

**Incidental finding (not explosiveness, not pre-registered, reported as a lead only):** the residual is *negatively* related to the prior-less, recency-weighted season-to-date PPA net (Δp, coefficient −13 alone; −34 with pred_margin, whose own slope turns +0.16). It replicates on the holdout (+0.088 alone, +0.104 with pred_margin; per season 2022 +0.13, 2023 +0.13, 2024 +0.08, 2025 +0.03, with the last two CIs crossing zero). Reading: for a given prediction, games where the in-season PPA fit favours the home team are over-predicted, i.e. the published blend leans too hard on in-season PPA relative to the prior/success rate. This is a shrinkage/calibration-weight question (alpha_ppa, PPA-vs-SR weight, or a calibration term), close in spirit to the calibration v2 prior-share slope already shadowed for 2026; whether v2 already captures it was not checked here. The direction is consistent with the holdout gain fading in 2024–2025. It conflicts with alpha_ppa having been tuned on 2015–2021, so it should be treated as a hypothesis to pre-register, not a result. Nothing here recommends re-tuning.

## 2. Special-teams points per game

Special-teams plays are in `plays_raw` (not in `scrimmage_plays`): Punt, Kickoff, Field Goal Good/Missed, Blocked Punt/FG (and TD variants), Punt/Kickoff Return TD, Missed FG Return (TD), Defensive 2pt. Points on an ST play were derived from the change in offense_score + defense_score versus the previous score-reliable row (per `SCORE_UNRELIABLE_TYPES`), clamped to [0, 8]; the attribution checked out (FG Good 2.92 mean, 97% positive; blocked-punt TD 6.6; kickoffs 0). Feature A: team season-to-date (ST points scored − allowed) per game. Feature B: shrunk FG make rate (fgm+2.4)/(fga+3). Both Δ = home − away; FCS/no-data sides get the neutral value. Punt/kickoff *field position* value was not modelled (no EPA on those rows), so this is a points-only check, which is what the handoff asked for.

| feature | n | MAE before | MAE after | gain [95% CI] | coefficients |
|---|---|---|---|---|---|
| Δ ST net pts/game, tuning | 5844 | 13.382 | 13.388 | −0.005 [−0.012, +0.002] | +0.056 pts per ST pt/game |
| Δ FG make rate, tuning | 5844 | 13.382 | 13.384 | −0.002 [−0.011, +0.007] | +1.86 |
| both, tuning | 5844 | 13.382 | 13.389 | −0.007 [−0.016, +0.003] | 0.038, 1.57 |
| both, holdout | 3659 | 12.995 | 12.993 | +0.002 [−0.006, +0.010] | |
| both, weeks ≥ 4, tuning | 4141 | 13.014 | 12.998 | +0.017 [−0.010, +0.043] | (intercept-only on weeks ≥ 4 already gives +0.010) |

Nothing. The coefficient on ST points (0.056 margin points per ST point per game) says the residual barely moves with past ST scoring; the model's PPA/SR already captures the drive outcomes that lead to FGs.

## 3. Rest / bye days

From `games` (all games, including FCS): days since the team's previous game in the season, season openers set to 14, everything capped at 21; bye = rest ≥ 13. Δ = home − away. 15% of games have a bye differential.

| feature | n | MAE before | MAE after | gain [95% CI] | coefficients |
|---|---|---|---|---|---|
| Δ rest days, tuning | 5844 | 13.382 | 13.384 | −0.002 [−0.011, +0.007] | +0.10 pts/day |
| Δ bye, tuning | 5844 | 13.382 | 13.385 | −0.002 [−0.013, +0.008] | +0.93 pts |
| Δ bye excluding openers, tuning | 5844 | 13.382 | 13.387 | −0.004 [−0.014, +0.005] | +0.86 |
| Δ rest + Δ bye, holdout | 3659 | 12.995 | 12.993 | +0.002 [−0.012, +0.017] | |

A bye is worth about +0.9 points in-sample, but that is too small and too rare to move MAE out of sample. ≈ 0, as the handoff expected.

## 4. Per-team home-field advantage

Home (non-neutral) games only, n = 5430 tuning / 3390 holdout. Residual ~ intercept + team dummy for the home team, ridge-shrunk with λ in {5, 10, 20, 50, 100, 300} (in units of games; closed form (Σr − n·μ)/(n + λ)). λ = 0 is undefined (teams unseen in training folds).

| λ | tuning gain [95% CI] | holdout gain [95% CI] | team-effect sd (full fit) |
|---|---|---|---|
| 5 | −0.123 [−0.190, −0.057] | −0.026 [−0.105, +0.053] | 2.44 |
| 20 | −0.054 [−0.103, −0.005] | +0.004 [−0.058, +0.066] | 1.81 |
| 50 | −0.013 [−0.045, +0.019] | +0.018 [−0.025, +0.062] | 1.22 |
| 100 | −0.001 [−0.021, +0.020] | +0.020 [−0.010, +0.049] | 0.79 |
| 300 | +0.003 [−0.006, +0.012] | +0.012 [−0.000, +0.025] | 0.33 |

Weak shrinkage hurts; the best λ (300) leaves an optimistic bound of +0.003 on tuning and +0.012 on holdout, both with CIs touching zero and the holdout value picked after seeing six λ values. Per-team HFA is noise at this sample size. ≈ 0, as expected.

## Approximations

- Check 1 uses prior-less ridges (as the handoff suggested) rather than the prior-shrunk in-season fit; early-slot values are therefore noisier than a production feature would be, which if anything flatters the Δp finding less than a prior-shrunk version might.
- Check 2 measures ST *points* only; field-position value of punts/kickoffs is not in the data as EPA.
- Check 3 counts games against any opponent (incl. FCS) for rest; the first game of a season is treated as a 14-day rest.
- Holdout numbers everywhere are a sanity check with a strict expanding window; no parameter was chosen on them except the λ sweep in check 4, which is flagged.
