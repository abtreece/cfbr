# Penalty ceiling check (2026-11-08)

**Verdict: MISS.** No penalty feature is built. Officiating-crew data (refwatch.ca, RefMetrics, conference gamebooks) is not pursued for the margin model.

Rule: DECISIONS.md "Penalty ceiling check" (committed before any feature values were computed) and its pre-run amendment (drops |yards| > 99, a parse error). Script: `docs/penalty-ceiling-check-2026-11-08.py`. It reads only cached data and makes no API calls.

## Motivation

The question was whether officiating data, such as crew assignments from refwatch.ca, could improve the ratings. CFBD has no officials data, and a crew's flags fall on both teams, so a crew effect on the margin can't be larger than the team penalty effect it works through. Team penalty behaviour is already in `plays_raw` at no cost, so it served as the ceiling test.

## Setup

- **Residual:** margin − pred_margin from the frozen tuned backtests (`results/20261011T144444Z-tuned-tuning`, `results/20261011T145524Z-tuned-holdout`), the same as M4.
- **Data:** `play_type = 'Penalty'` rows, which the model drops (P2) and which carry no PPA. `yards_gained` is signed from the offense's point of view through 2025 week 8. Same season only, kickoff < `as_of`.
- **F1, primary:** net penalty-only yards per game, (Σ net)/(g + 3). Δ = home − away.
- **F2, secondary:** penalties committed per game, shrunk toward the league mean.
- **Evaluation:** OLS of residual on intercept + feature. Leave-one-season-out within 2015–2021, paired MAE gain with a 95% CI. Pass requires gain ≥ +0.03 and CI lower bound > 0.

## Results

| spec | n | MAE base | MAE new | gain [95% CI] | coefficients |
|---|---|---|---|---|---|
| intercept only, 2015–2021 LOSO | 5844 | 13.382 | 13.384 | −0.0019 [−0.0058, +0.0020] | int +0.13 |
| **F1 net yards (primary), 2015–2021 LOSO** | 5844 | 13.382 | 13.376 | **+0.0065 [−0.0134, +0.0265]** | d_f1 +0.0706 |
| F2 committed, 2015–2021 LOSO | 5844 | 13.382 | 13.381 | +0.0018 [−0.0146, +0.0182] | d_f2 −0.63 |
| F1 + F2, 2015–2021 LOSO | 5844 | 13.382 | 13.377 | +0.0050 [−0.0157, +0.0257] | d_f1 +0.056, d_f2 −0.26 |
| intercept only, 2022–2024 expanding (sanity) | 2725 | 13.143 | 13.144 | −0.0005 [−0.0039, +0.0029] | int +0.08 |
| F1, 2022–2024 expanding (sanity) | 2725 | 13.143 | 13.103 | +0.0407 [+0.0092, +0.0722] | d_f1 +0.0713 |
| F2, 2022–2024 expanding (sanity) | 2725 | 13.143 | 13.119 | +0.0244 [−0.0006, +0.0495] | d_f2 −0.62 |
| F1 + F2, 2022–2024 expanding (sanity) | 2725 | 13.143 | 13.100 | +0.0427 [+0.0103, +0.0751] | d_f1 +0.058, d_f2 −0.24 |

The sd of Δ F1 is 10.9 yards per game (2015–2021). Corr(ΔF1, ΔF2) is −0.60. Corr(ΔF1, pred_margin) is −0.13.

**Descriptive, not pre-registered:** F1 gain by season, LOSO:

| season | gain [95% CI] |
|---|---|
| 2015 | −0.010 [−0.063, +0.043] |
| 2016 | +0.002 [−0.056, +0.060] |
| 2017 | +0.030 [−0.014, +0.074] |
| 2018 | +0.047 [+0.006, +0.088] |
| 2019 | +0.020 [−0.022, +0.062] |
| 2020 | −0.025 [−0.098, +0.048] |
| 2021 | −0.030 [−0.091, +0.031] |

Expanding window: 2022 +0.012, 2023 +0.042, 2024 +0.067 [+0.013, +0.122]. Corr(ΔF1, residual) is positive in every season from 2015 to 2025, ranging from +0.006 (2020) to +0.096 (2018).

## Reading

- **The primary test misses by a wide margin.** The 2015–2021 gain is a fifth of the bar, and its CI includes zero.
- **The sign is consistent but the effect is small.** The coefficient is +0.07 points of margin per net penalty yard per game in both windows. That works out to about +0.77 points per sd. The direction makes sense: penalty-only yards are field position that the model's plays never see. But the effect is too small and too noisy season to season to clear 0.03 on the tuning seasons.
- **The 2022–2024 sanity gain (+0.041) doesn't pass the rule.** It is reported but not gated, as fixed in advance, and it rests mostly on 2023–2024. Like the M4 PPA lead, it is an incidental lead, not a result. Testing it would need its own rule on data not yet seen. A live feature would also need penalty attribution parsed from `play_text`, because CFBD stopped signing penalty yards in 2025 week 9.
- **The officiating-crew implication follows from the rule.** A crew's flags split across both teams, so a crew effect on the margin is a second-order version of F1. With F1 itself below the bar, crew data is not worth sourcing for the margin model. Totals were out of scope (the v2 shadow window is running).
