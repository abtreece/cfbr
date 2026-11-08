# Data anomaly checks and LLM investigation — design

Date: 2026-11-08. Status: design approved, not built.

## Problem

`cfbr weekly` publishes ratings from whatever CFBD returned. The existing
post-run health check (`bets/health.py`) covers the betting pipeline (ledger,
snapshots, budgets) but nothing checks the play-by-play or the ratings it
produces. A truncated feed, a duplicated drive or a misclassified team would
reach the site, the paper bets and the accuracy line unnoticed.

Some of these are self-explanatory once detected ("game X has no plays").
Others need judgment to tell a data bug from real football: a 9-point score
gap, a 105-play game, a 6-point rating drop. That second step is the only part
of this design that uses an LLM, and it is the only unattended job in cfbr
that needs one.

Monthly API credits ($200, Ultra plan) expire at the end of each billing cycle
and hard-stop when exhausted (support article 17154008). The design spends
more per investigation (effort, turns, every flagged item, a one-time
historical shakedown) rather than investigating more things.

## Decisions (user, 2026-11-08)

- All eight checks below are in v1, including B3 (model vs. line).
- Advisory only: nothing blocks publication; exit codes and the publish path
  of `weekly` are unchanged.
- Investigation runs as a separate cron step (`cfbr investigate`), not inline.
- Agent tools: one read-only SQL tool plus a whitelisted output-file reader;
  domain notes in a frozen system prompt.
- Model `claude-opus-5-5`, effort `xhigh`, 40 turns per investigation, up to
  12 investigations per run, local guard $50/month.
- A historical shakedown (2022–2025) runs before go-live and gates it.

## Anomaly checks

Baselines measured on `data/cfb.duckdb` (dev copy, 2026-11-08), completed
games with at least one FBS team, 2022–2026 week 8: 4,225 games.

### A. Ingest integrity (each newly ingested week; `plays_raw`, `games`)

| ID | Check | Baseline | Threshold | Handling |
|---|---|---|---|---|
| A1 | Completed game with an FBS team has no play-by-play | 0 / 4,225 | any | plain |
| A2 | Score gap: \|PBP max home score − final\| + \|PBP max away score − final\| | > 8 in 46 games | > 8 | investigate |
| A3 | Play count per game outside the historical range | p1 ≈ 140, p50 ≈ 177, p99 ≈ 225; < 100 in 11 games | < 110 or > 250 | investigate |
| A4 | Duplicate plays: same game, period, clock, down, distance, yards to goal, play text (Timeout/End-of-period rows excluded) | 134–282 games/season 2022–2025; 7 in 2026 | > 3 duplicate rows in one game | investigate |
| A5 | FBS status change: a team whose first game of the season is this week and whose FBS status differs from last season's `team_status` (or that is FBS with no row last season) | 1–2 per season (JMU 2022 … NDSU, Sacramento State 2026) | any | plain |

A5 originally checked for team names absent from `team_status`; that can
never fire, because `team_status` is derived from `games` (`ingest/store.py`).
The status change is the event that matters: it changes FCS pooling.

A2 uses the maximum score in the PBP, not the last play's: ordered by numeric
`play_id`, the last play's score misses the final in 5–10% of games (late-game
play order is unreliable), while the maximum matches in 95.7%.

### B. Model output plausibility (after `rank`; `out/{season}/wNN/`)

| ID | Check | Threshold | Handling |
|---|---|---|---|
| B1 | \|Δ rating\| week over week, per team | > p99.5 for the week bucket (weeks 1–3, 4–7, 8+) | investigate |
| B2 | Single-game offensive PPA/play outside the season distribution (2026: p0.5 −0.53, p99.5 +0.70) | outside p0.25–p99.75 | context for B1 only, not alerted on its own |
| B3 | \|model margin − current consensus spread\| on next week's games, from `predictions.csv` (`pred_margin + closing_spread`) | > p99.5 of backtest \|model − opener\| (2021+) | investigate |

B1 and B3 thresholds are recomputed by `cfbr anomalies --baselines`
(`baselines.py`) with the Engine under the production config (`tuned.toml`,
including the FCS two-stage fit, PR #43): B1 from `build_rankings` over 2015
to the last completed season, B3 from point-in-time predictions vs the opener
(2021+). They are frozen to `config/anomaly_thresholds.toml`, like `tuned.toml`.
`weekly` checks only the latest completed week (`--history` covers the rest),
and A5 fires only in the team's first scheduled week.
B3 is calibrated against the opener because Sunday's lines are early-week
numbers, not closes. B3 reads lines only to compare; nothing it produces is
read by the model or bets (see Isolation).

Not checked: line-provider disagreement (bets health covers the Odds API
side), null-PPA share (steady near 25%, mostly non-scrimmage rows), last-play
score order (too noisy).

## Architecture

New package `src/cfbr/anomalies/`:

| Module | Responsibility | Depends on |
|---|---|---|
| `checks.py` | The eight checks as pure functions over a DuckDB connection and `out/` frames; return `Anomaly` records (id, check, severity, season, week, game/team keys, value, threshold, `investigate`) | duckdb, polars |
| `baselines.py` | Computes B1/B3 thresholds; writes `config/anomaly_thresholds.toml` | `results/` frozen runs |
| `tools.py` | `query(sql)` and `read_output(name)` (see Agent) | duckdb |
| `prompt.py` | Frozen system prompt: schemas, domain notes, verdict rubric | — |
| `agent.py` | One investigation: tool loop, caps, refusal fallback, `submit_verdict`, grounding check | `anthropic` SDK |
| `ledger.py` | `data/anthropic_calls.jsonl`; month-to-date spend; guard | — |
| `run.py` | Orchestration for the entry points | all above |

A1–A4 thresholds are constants in `config/default.toml` under `[anomalies]`;
`[investigate]` holds `model`, `effort`, `max_turns` (40), `max_items` (12),
`wall_clock_minutes` (15), `monthly_budget_usd` (50).

### Entry points

- `cfbr weekly`: after `rank` and the accuracy line, before bets, calls
  `run.run_step`; each check is guarded. Writes
  `out/{season}/wNN/anomalies.{json,md}` and sends one ntfy (priority
  `default`) listing the items new in this run — all of them until PR 2, then
  plain items immediately and investigate items marked "investigating". A
  same-week re-run does not re-alert items already seen. B2 rows are stored
  with status `context` and never alerted. A crash here is alerted and never
  changes the exit code.
- `cfbr investigate`: cron, hourly Sunday 10:00–18:00 Central. No-op unless
  the latest `anomalies.json` has `pending` items.
- `cfbr anomalies --season S [--week W] [--history] [--baselines]`
  (PR 1; `--history` runs the database checks over every completed week for
  the shakedown's first step) and later `[--investigate] [--sample-patterns]
  [--force ID] [--eval]` (PR 2–3).

### Isolation

No module in `model/`, `bets/`, `report/` or `web/` imports `cfbr.anomalies`
(tested by an import scan). The site does not show anomalies in v1. Lines
reach the agent only through B3 records and the `lines` table, and agent
output is written only to `anomalies.*`, transcripts and the ledger.

## Data flow

```
cfbr weekly (holds .weekly.lock)
  ingest → build_store → prep → rank → accuracy
  → checks.run  → out/{S}/wNN/anomalies.json  (investigate items: status "pending"; others "plain")
                → A1/A5 lines in the run's ntfy
  → bets → web build   (unchanged)

cfbr investigate (hourly Sun 10–18 CT)
  no pending items                → exit 0, silent
  ledger month-to-date ≥ budget   → mark "skipped_budget", alert once
  flock .weekly.lock non-blocking → held: exit 0, next hour retries
      copy cfb.duckdb → data/investigations/snapshot.duckdb (~315 MB, seconds)
      release the lock
  for item in pending (severity order: A2, A3, A4, B1, B3), at most max_items:
      verdict = agent.investigate(item)
      write verdict into anomalies.json (atomic rename, after each item)
      transcript → data/investigations/{S}/wNN/{id}.json
  render anomalies.md; send one ntfy
```

Items beyond `max_items` are listed in the alert as not investigated (more
than 12 in a week suggests a systemic break). A written verdict is never
re-run automatically; `--force ID` re-runs one by hand.

The lock is held only while copying the store, so a long run (up to 12
items × 15 minutes) never makes the 3 PM `weekly` exit 4. The agent queries
the snapshot, which `build_store` cannot replace mid-investigation.

Anomaly ids are a stable hash of check id and keys (season, week, game or
team). When `weekly` re-runs for the same week (the 3 PM run), `checks.run`
keeps the existing status and verdict of every id that fires again, adds new
ids as `pending`, and marks ids that no longer fire `resolved` (verdict
kept). Writes to `anomalies.json` from `weekly` and `investigate` are
read-merge-write under an atomic rename; `investigate` re-reads the file
before writing each verdict, so a concurrent `weekly` re-run is not
overwritten.

## Agent

### Tools

- `query(sql)`: a connection to the run's snapshot with `read_only=True` and
  `SET enable_external_access = false` (blocks `read_csv`, `ATTACH`, `COPY`
  and other file access). Results capped at 200 rows and 20,000 characters,
  with a truncation note. Each executed query gets a sequential `query_id`;
  SQL and full result are kept for the grounding check. SQL errors return as
  `is_error` tool results.
- `read_output(name)`: only these files for the anomaly's season and week:
  `ratings.csv`, the previous week's `ratings.csv`, `anomalies.json`,
  `accuracy.csv`. Anything else is refused.
- `submit_verdict(...)`: `strict: true`; ends the loop. Forced `tool_choice`
  is rejected by Opus 5.5, so the prompt instructs the agent to finish with
  this call.

As built, each `query` runs in its own child process (killed at the timeout)
behind a parse-tree gate that allows exactly one SELECT; see DECISIONS.md
(Ops, anomalies) for the full hardening.

### System prompt

Frozen text (cacheable across a run's investigations): table schemas;
domain notes taken from DECISIONS.md and CLAUDE.md — `offense_score` and
`defense_score` are post-play; `play_id` must be ordered numerically, and
late-game order is unreliable; penalty `yards_gained` is signed through 2025
week 8 and unsigned after; plays with both sides non-FBS are dropped in prep;
non-FBS teams are pooled; play types dropped in prep; FCS two-stage fit — and
the verdict rubric, including how to judge `model_impact` from prep rules.
The anomaly record and any B2 context go in the first user message.

### Settings

`claude-opus-5-5`, adaptive thinking, `output_config.effort = "xhigh"`,
task budget about 60K output tokens (beta), server-side refusal fallback
(`fallbacks: "default"`), streaming. Every response's usage, cost and
request id go to the ledger.

### Verdict

```json
{
  "verdict": "data_bug | real | inconclusive",
  "confidence": "low | medium | high",
  "summary": "≤280 chars",
  "evidence": [{"claim": "...", "query_id": 7, "excerpt": "≤500 chars"}],
  "affected": {"game_ids": [], "teams": []},
  "model_impact": "none | small | material | unknown",
  "suggested_action": "≤200 chars, advisory, never executed"
}
```

`run.py` adds model, tokens, cost, turns, request ids and duration.

**Grounding check** (deterministic, after `submit_verdict`): every
`evidence[].query_id` must name a query that ran without error in this
session, and `excerpt` must be a substring of its stored result. Any failure
downgrades the verdict to `inconclusive` with reason `ungrounded`.

### Alert

One ntfy per `investigate` run. Priority `default`, or `high` when any item is
`data_bug` with `material` impact.

```
cfbr anomalies w09 — 1 data bug, 2 real, 0 inconclusive ($0.84)
• A4 Rice–UTSA: data_bug/high — drive 14 duplicated (38 plays); material
• B1 Iowa −6.2: real/medium — lost QB; SR fell 0.48→0.31 in 2 games
• B3 Army–Navy edge 19.5: real/high — no data issue on either side
```

## Error handling

| Condition | Behavior |
|---|---|
| A check raises inside `weekly` | Becomes a plain anomaly ("check A3 failed: …"); exit code unchanged |
| `anomalies.json` write fails inside `weekly` | Logged and alerted; web rebuild still runs |
| No `ANTHROPIC_API_KEY` (env or `.keys`) | `investigate` logs and exits 0 — this is the off switch |
| Lock held | Exit 0, silent |
| 429 / 5xx / connection error after SDK retries | Item stays `pending`, `attempts += 1`; at 3 → `error`, alert |
| 401 / 403 | Alert once, stop; items stay `pending` |
| Credits exhausted (billing refusal) or local guard reached | Remaining items `skipped_budget`, alert once |
| `refusal` after fallback | `inconclusive`, reason `refused`, `stop_details.category` kept |
| Turn cap or wall clock | `inconclusive`, reason `budget` |
| Grounding check fails | `inconclusive`, reason `ungrounded` |
| Alert delivery fails | Never raises (existing `alerts.send`) |

## Testing

- Checks: per check, a synthetic DuckDB fixture that fires and one that
  doesn't. A `dataset`-fixture test that 2022–2025 firing counts stay within
  the measured bands.
- Tool sandbox: `INSERT`, `read_csv('/etc/passwd')`, `ATTACH`, `COPY … TO`
  and `../` in `read_output` are refused; row and character caps hold.
- Agent: a fake Anthropic client with scripted responses covers the turn cap,
  refusal, missing or malformed `submit_verdict`, grounding pass and fail,
  and ledger writes. No network in `pytest`.
- Ledger: cost math, month boundary, guard.
- Merge: a `weekly` re-run keeps verdicts of ids that fire again, adds new
  ids as `pending`, marks vanished ids `resolved`.
- Isolation: import scan over `model/`, `bets/`, `report/`, `web/`.
- Live eval (spends credits, manual): `cfbr anomalies --eval` replays
  `tests/fixtures/anomaly_eval.jsonl` and reports agreement.

## Shakedown (go-live gate)

1. Run the checks over 2022–2025 (no API calls). Report counts per check and
   season.
2. Group fired items by check and shape (A4: whole duplicated drive vs.
   scattered single plays; A2: gap size and side; A3: low vs. high; B1/B3 by
   week bucket) and sample up to 3 per group, about 40 investigations
   (estimated $20–40).
3. Each verdict is checked against the play-by-play, then reviewed by the
   user. Labeled items become `tests/fixtures/anomaly_eval.jsonl`.
4. Iterate on the system prompt until the labeled set shows **≥ 90%
   agreement and no wrong `high`-confidence verdict**. This criterion is
   fixed now, before any verdict exists.
5. Findings about training data (e.g. duplicate plays in 2015–2025) are
   recorded in DECISIONS.md and a separate issue. A prep fix would change the
   frozen backtests and needs its own pre-registered rule; it is out of scope.

The shakedown runs in the dev checkout with an API key supplied as an
environment variable only (never a repo `.keys`), with the user's approval.

## Rollout

1. PR 1: `checks.py`, `baselines.py`, thresholds file, `weekly` integration,
   `cfbr anomalies` (deterministic). Useful on its own.
2. PR 2: `tools.py`, `prompt.py`, `agent.py`, `ledger.py`, `run.py`,
   `cfbr investigate`, tests. Adds the `anthropic` dependency (2.x, which
   brings `httpx2`); DECISIONS line explaining the exception to "HTTP through
   httpx directly".
3. Shakedown (above).
4. PR 3: eval fixture, README (command table, cron line, ops), DECISIONS
   lines, `cfbr quota` / `cfbr check` showing Anthropic month-to-date spend.
5. User steps (stop-and-ask): deploy, `ANTHROPIC_API_KEY` in javelin's
   `.keys`, crontab line for `cfbr investigate`. Confirm the Console org has
   no purchased credits or auto-reload if the $200 hard stop is relied on.

## Cost

Per investigation about $0.50–2 at `xhigh` with 40 turns; 0–12 a week;
about $4–15 a month in season, plus the one-time shakedown. The local guard
($50) catches runaway loops; the credits' hard stop is the backstop.

## Open observations (not verified; for the shakedown)

- Duplicate-play rows fell from ~1,200 extra rows (2025) to 8 (2026). Either
  CFBD changed or the 2026 cache was fetched differently; 2015–2025 training
  data may carry duplicates.
- Late-game `play_id` order is unreliable. Prep derives score at the snap
  from that order (DECISIONS P2); garbage time is disabled in `tuned.toml`,
  so model impact is likely nil.
