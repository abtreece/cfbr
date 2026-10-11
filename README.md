# cfb-ratings

A predictive college football rating system. A team's **rating** is its expected point margin against an average FBS team on a neutral field. Ratings come from opponent-adjusted play efficiency (EPA/PPA and success rate) fit with ridge regression, shrunk toward a preseason prior, and calibrated to points. They are backtested walk-forward against Vegas closing lines.

See `cfb-ratings-spec.md` for the build spec, `DECISIONS.md` for every judgment call, and `results/tuning_report.md` for tuned accuracy.

## Setup

```bash
uv sync
echo "CFBD_API_KEY=..." > .keys      # or export CFBD_API_KEY; .keys is gitignored
chmod 600 .keys
```

The CFBD free tier allows 1,000 calls a month. A full backfill for 2014–present takes about 300 calls. Every API call is logged to `data/raw/_calls.jsonl`, and `cfbr quota` shows usage. Raw responses are cached under `data/raw/`, so re-runs make no calls (use `--refresh` to re-fetch).

## Commands

| Command | What it does |
|---|---|
| `cfbr ingest --backfill` | Pull all seasons from 2014 through the current one into the cache and rebuild `data/cfb.duckdb` |
| `cfbr ingest --season 2026 --week 7` | Incremental weekly ingest (about 4 calls: games, lines, polls, plays; +1 per re-run while that week's polls are pending) |
| `cfbr prep` | Build `scrimmage_plays` / `model_plays` |
| `cfbr ratings --season 2025 --week 8` | Print ratings available before that week's first kickoff |
| `cfbr backtest --seasons 2015-2021` | Walk-forward backtest; writes `results/{run_id}/` |
| `cfbr tune` | Optuna search on the tuning seasons; freezes `config/tuned.toml` |
| `cfbr holdout` | Evaluate the frozen config on the holdout seasons (run once) |
| `cfbr rank --season 2026 --week 6` | Write `out/2026/w06/` rankings and next-week predictions |
| `cfbr weekly` | Cron entry point: ingest → prep → rank → season-to-date accuracy line → web data for the season |
| `cfbr web build` | Precompute week-by-week ratings and game predictions for the web app into `out/web/` (about 30 s for all seasons) |
| `cfbr web serve` | Run the local web app on http://127.0.0.1:8000 |
| `cfbr quota` | Show API usage this month |

`rank` and `weekly` use `config/tuned.toml` when it exists. Pass `--config` to override.

## Web app

A local web app for browsing the ratings:

```bash
uv run cfbr web build          # once; `cfbr weekly` keeps the current season up to date
uv run cfbr web serve          # http://127.0.0.1:8000 (add --host 0.0.0.0 --port 8000 for your LAN; there is no login)
```

- **Rankings**: any season, after any week (or preseason/final), filterable by conference and sortable. Includes rank and rating changes, the prior share, and a trend sparkline.
- **Team pages**: rating, offense and defense through the season, plus a game log with the model's prediction, the closing line and the result.
- **Games**: each week's predictions against the closing line, plus results, errors and against-the-spread grading once games are final.
- **Accuracy**: walk-forward accuracy per season and per week against the closing line and the Massey baseline. Also shows the `cfbr weekly` run log and every backtest run under `results/`.
- **JSON API** under `/api/` (docs at `/api/docs`).

The server never fits a model. It reads `out/web/{season}/` and picks up new files from `cfbr weekly` without a restart. The ratings shown "after week N" are exactly what `cfbr rank` writes to `out/{season}/wNN/`.

## Cron (Sunday mornings, Central time)

```cron
# System clock is UTC and Ubuntu's cron ignores CRON_TZ: each run fires at both possible
# UTC hours, and only the one matching the Chicago local hour proceeds (correct across DST).
# 9 AM: early rankings + predictions. 3 PM: re-run to pick up the AP/Coaches polls.
0 14,15 * * 0  [ "$(TZ=America/Chicago date +\%H)" = "09" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr weekly >> data/weekly.log 2>&1
0 20,21 * * 0  [ "$(TZ=America/Chicago date +\%H)" = "15" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr weekly >> data/weekly.log 2>&1
```

`weekly` takes a lock, so overlapping runs are refused. A week is only ingested once all of its FBS games are final, and re-running a week replaces that week's line in `out/{season}/accuracy.csv` instead of adding a duplicate. Exit codes: 3 = API quota nearly exhausted, 4 = another run in progress, 5 = no completed weeks yet.

## Paper bets

`cfbr weekly` also maintains a paper-trading ledger. It never places a bet, and betting lines never feed the ratings. On each run it:

1. grades open bets whose games are final, and records the closing line and closing-line value (CLV),
2. records a new 1-unit bet at −110 for every upcoming game where the model and the consensus line (median across books in the newest line snapshot) differ by 3 or more points. A bet is recorded once, before kickoff, and never changed.

Outputs: `out/{season}/bets.csv` (the ledger) and `out/{season}/bets.md` (record, ROI, CLV, open bets). Each bet is graded twice: at the consensus line, and at the best number across every sportsbook CFBD lists, with no book excluded. Settings are in `[bets]` in `config/default.toml`. Run it by hand with `cfbr bets --season 2026 --week 6`. The summary draws no conclusion before 500 settled bets. Treat positive average CLV as the first sign of a real edge; win rate alone is mostly noise at small samples.

## Point-in-time guarantee

Every prediction for a game in slot *w* (regular-season week, or a 7-day postseason window) uses only plays and games that kicked off before that slot's first kickoff. Cross-season models (the preseason prior and the points calibration) use leave-one-season-out within the tuning seasons, and only earlier seasons for the holdout and live seasons. `tests/test_leakage.py` checks this with an explicit max-kickoff assertion and a canary future game.

## Tests

```bash
uv run pytest
```

Tests that need real data skip automatically when `data/cfb.duckdb` is absent.
