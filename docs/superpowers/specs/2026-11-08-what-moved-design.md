# What moved — design

Date: 2026-11-08. Status: built 2026-11-08; storage simplified (columns in ratings/games parquet), see the plan.

## Problem

Readers can't tell why a team's rating or rank changed in a week. After 2026 week 10, the
complaints were "Ohio State lost to Oregon and only fell to #2", "Notre Dame beat Miami and
fell", "Utah beat BYU and fell". Each one was explained by splitting the rating change by hand
into the team's own game, older plays fading, and other teams moving. This feature puts that
split on the site.

Rejected alternative (recorded so it isn't re-proposed): an "in-season only" rating, meaning the
published model with the preseason prior removed. After week 10 it ranked Wake Forest #1 and
Liberty #2, and its ratings are compressed (SD 7.7 vs 11.0). Ten weeks don't separate team
quality from schedule strength well enough without the prior. It would add confusion rather
than remove it.

Display only. The published ratings, ranks and predictions don't change. No model or
evaluation rule is involved.

## Computation (`src/cfbr/report/moves.py`, new)

For each FBS team and each completed regular-season week *w*, comparing "after week *w*" with
"after week *w* − 1" (week 1 compares with the preseason prior):

| Part | Definition |
|---|---|
| `move_decay` (older plays fading) | the plays behind the week *w* − 1 rating (kickoff < `week_as_of(w − 1)`), refit with recency weights at `week_as_of(w)`, minus the week *w* − 1 rating |
| `move_own` (own game) | the week-*w* rating minus the week-*w* fit with that team's week-*w* plays (kickoff in [`week_as_of(w − 1)`, `week_as_of(w)`)) left out; 0 on a bye |
| `move_others` (other games) | `rating_change − move_decay − move_own` |

The three parts add up exactly to the published `rating_change`. Week 1: `move_decay` = 0 (no
earlier plays). As-of times are those of `report.rank.week_as_of`, so they match `cfbr rank`.

Every fit uses the published path: the same priors, `play_weights` (recency and FCS weight),
Stage 2 when `fcs_two_stage` is on, and `calibration(season)` to convert to points. To avoid a
second copy of the fit, `Engine._fit` gains an optional `mask: np.ndarray | None` over the
first *k* plays. With `mask=None` its output is byte-identical to today.

**Rank movement:** `passed_by` is the teams ranked below this team after week *w* − 1 and above
it after week *w*. `passed` is the reverse. Each is the first five names by week-*w* rank, plus
counts `n_passed_by` and `n_passed`.

**Game context**, one row per team per game played in week *w*:

- `opponent`, `home`/`away`/`neutral`, `points_for`, `points_against`
- `pred_margin`: the model's pre-game predicted margin from the team's view. It's taken from
  that game's slot snapshot, which is already point-in-time.
- Actual per-play stats from `model_plays` (the plays the fit used): `ppa_for`, `ppa_against`,
  `sr_for`, `sr_against`.
- Expected per-play edges, from the pre-game snapshot of the game's slot:
  `exp_ppa_edge = (o_ppa_t + d_ppa_t) − (o_ppa_opp + d_ppa_opp) + 2·H_ppa·home`, and the same
  for success rate. Here home = +1 home, −1 away, 0 neutral. The intercept cancels out in this
  difference. `_fit` additionally keeps each fit's home coefficients (`home_ppa`, `home_sr`) so
  the snapshot carries them.

Postseason slots aren't covered, matching the weekly ranking pages.

**Cost:** one extra fit per week for decay, plus one leave-own-game fit per FBS team. The
prototype ran about 15 ms per fit, about 2 s per week.

## Where it runs and where it's stored

- `web/build.py`: `out/web/{season}/moves.parquet` (team rows) and
  `out/web/{season}/move_games.parquet` (game rows), for every completed regular-season week of
  the **live season only**. Completed seasons get none; at about 6 minutes per deploy for all
  of history, it isn't worth it.
- `cfbr rank` (and so `weekly`): `out/{season}/wNN/moves.csv` for that week. Team rows, with the
  game context joined on; a team with two games that week gets two rows.
- **Access:** it's model output, gated exactly like ratings for a live season (`view`/premium
  checks on every page and API that shows it).

## Display

**Team page, new section "What moved"**, above "Explain the rating". It shows the latest
completed week, with a week selector for earlier completed weeks. Example (2026 week 10; values
from the prototype and from data; the expected per-play edge is computed at build time):

> **After week 10: rank 1 → 3 (−2), rating 24.95 → 22.98 (−1.97)**
>
> | | |
> |---|---|
> | Own game | **−1.88** · Lost to Oregon 28–41 · expected to win by 13.7 |
> | | per play: PPA +0.029 vs +0.166, success 40.8% vs 35.1% (expected edge shown) |
> | Older plays fading | −0.22 |
> | Other games | +0.14 |
> | Passed by | Georgia, Alabama |

- Bye week: "No game this week", then the fading and other-games lines.
- One-sentence note: ratings come from per-play efficiency against what was expected, not from
  wins and losses. Links to `/method#moves`.
- The existing collapsed week-by-week table gets three narrow columns: own game, fading, others.

**Rankings page:** the ± (rank change) and Δ Rating cells get a `title` tooltip, e.g.
"Own game −1.88 · fading −0.22 · others +0.14 · passed by Georgia, Alabama". No new columns.
In the same PR, the Prior column moves out of "More columns" (drop its `extra` class).

**Method page:** a short `#moves` section defining the three parts and noting that they add up
to the change.

**API:** `/api/rankings` rows and `/api/team/{season}/{team}` get `move_own`, `move_decay`,
`move_others`, `passed_by` and `passed`. `/api/team` also gets the game-context rows. Both are
already gated for a live season.

## Tests

- Sum: for every FBS team in a live-season week, `move_decay + move_own + move_others` equals
  `rating_change` before rounding (≤ 1e-9).
- `Engine._fit(mask=None)` is byte-identical to today's output. A masked fit equals an explicit
  fit on the remaining plays. Both checks run under the default and the two-stage test configs.
- Bye week: `move_own` = 0. Week 1: `move_decay` = 0.
- `passed_by`/`passed` are consistent with the two weeks' ranks.
- Point-in-time: the expected edge for a game uses its pre-game slot snapshot. A test asserts
  the snapshot's as-of is ≤ the game's kickoff. The actual per-play stats equal a direct
  `model_plays` aggregate for that game.
- `cfbr rank`'s `moves.csv` equals the web parquet rows for the same week.
- Web: the section renders for a game week, a bye week and the week selector. Tooltips appear on
  the rankings page. The API fields are present. Both are locked for free viewers in a live
  season, like ratings. The Prior column is visible by default.

## Rollout

A normal PR and deploy. The deploy's web build creates `moves.parquet` and `move_games.parquet`
for 2026 weeks 1–10. Published ratings don't change. Sunday's `weekly` adds week 11 and its
`moves.csv`. One DECISIONS.md line records the feature and the rejected in-season rating.
