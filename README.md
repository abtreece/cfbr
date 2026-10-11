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

Paper bets also use The Odds API for US sportsbook spreads with prices. Its key goes in `.keys` as `ODDS_API_KEY` (or `ODD_API_KEY`). The free tier gives 500 credits a month, and each `cfbr weekly` run spends 2 (one per region in `[odds].regions`). Calls are logged to `data/raw/_odds_calls.jsonl`, and `cfbr quota` shows the remaining credits without spending any.

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
| `cfbr quota` | Show API usage this month (CFBD and The Odds API) |

`rank` and `weekly` use `config/tuned.toml` when it exists. Pass `--config` to override.

## Web app

A local web app for browsing the ratings:

```bash
uv run cfbr web build          # once; `cfbr weekly` keeps the current season up to date
uv run cfbr web serve          # http://127.0.0.1:8000 (add --host 0.0.0.0 --port 8000 for your LAN; there is no login)
```

- **Rankings**: any season, after any week (or preseason/final), filterable by conference and sortable. Includes rank and rating changes, the prior share, a trend sparkline, and AP, Coaches and CFP poll ranks for comparison.
- **Team pages**: rating, offense and defense through the season, poll ranks, plus a game log with the model's prediction, the closing line and the result.
- **Games**: each week's predictions against the closing line, plus results, errors and against-the-spread grading once games are final. Ranked teams show their AP rank going into the game.
- **Accuracy**: walk-forward accuracy per season and per week against the closing line and the Massey baseline. Also shows the `cfbr weekly` run log and every backtest run under `results/`.
- **Bets**: the paper-betting ledger from `out/{season}/bets.csv`, with the verdict against the pre-set minimum sample, record, units, ROI and closing-line value (at the consensus line and at the best line), open bets, and settled bets. `/bets/retro` (same gate) shows the retroactive replay from `cfbr bets-retro`, labelled as retroactive and kept separate from the live record.
- **Rosters** (`/roster/{season}/{team}`, linked from each team page): who's on the roster with position, class, size and hometown; returner / transfer (with the previous school) / no prior stats; last season's production at any school; departures and where they went; and the team-level roster summary. Data comes from CFBD rosters via `cfbr.report.roster_web`, written by the web build. Rosters are public facts, so they're free even during a live season.
- **JSON API** under `/api/` (docs at `/api/docs`).

**Premium content.** The betting section and all model output for a season still in progress (ratings, ranks, predicted margins, win probabilities, for every game) are premium; anything less leaks the picks (see DECISIONS.md). Free viewers get past seasons, the live season's schedule, scores, closing lines and polls, and aggregate accuracy. Everyone is premium today. To charge for it, implement `is_premium()` in `src/cfbr/web/access.py`. Every route already depends on it, and live-season data is masked in one place (`public_view` in `src/cfbr/web/app.py`) before any page or API code sees it.

To run it as a service that starts at boot and restarts if it crashes (systemd user service, no root needed):

```bash
mkdir -p ~/.config/systemd/user && ln -sf ~/cfb-ratings/deploy/cfbr-web.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now cfbr-web
loginctl enable-linger "$USER"          # start user services at boot without a login
systemctl --user restart cfbr-web       # after pulling new web code (new data needs no restart)
```

It listens on 0.0.0.0:8000 and logs to `data/web.log` (also `journalctl --user -u cfbr-web`).

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

Outputs: `out/{season}/bets.csv` (the ledger) and `out/{season}/bets.md` (record, ROI, CLV, open bets). Lines come from the newest CFBD snapshot plus the newest Odds API snapshot from the same run, which adds FanDuel, theScore Bet, BetRivers, BetMGM, Hard Rock Bet and others with their prices. Each bet is graded twice: at the consensus line (−110), and at the best number across every non-offshore sportsbook, at that book's price. No book is excluded by state; offshore books (e.g. Bovada) count toward the consensus line but never price a bet. Settings are in `[bets]` in `config/default.toml`. Run it by hand with `cfbr bets --season 2026 --week 6`. `cfbr bets-retro --season 2026 --through-week 6` replays the same rules on weeks already played, for both the published (v1) and roster-prior models, against opening lines, into a separate `out/{season}/bets_retro.csv`/`.md` that never counts toward the live ledger. The summary draws no conclusion before 500 settled bets. Treat positive average CLV as the first sign of a real edge; win rate alone is mostly noise at small samples.

## Prior shadow comparison

The published ratings use the v1 preseason prior. `cfbr weekly` also computes every game's point-in-time prediction with the roster-based prior (player rosters, transfers, defensive returning production) and writes `out/{season}/prior_comparison.csv` and `.md`. The roster predictions are never published. Under a rule fixed in DECISIONS.md, the roster prior becomes the default before 2027 week 1 only if it beats v1 on 2026 games from week 7 onward and the postseason. Run it by hand with `cfbr shadow --season 2026`.

## Point-in-time guarantee

Every prediction for a game in slot *w* (regular-season week, or a 7-day postseason window) uses only plays and games that kicked off before that slot's first kickoff. Cross-season models (the preseason prior and the points calibration) use leave-one-season-out within the tuning seasons, and only earlier seasons for the holdout and live seasons. `tests/test_leakage.py` checks this with an explicit max-kickoff assertion and a canary future game.

## Tests

```bash
uv run pytest
```

Tests that need real data skip automatically when `data/cfb.duckdb` is absent.
