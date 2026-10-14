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

Paper bets also use The Odds API for US sportsbook spreads with prices. Its key goes in `.keys` as `ODDS_API_KEY` (or `ODD_API_KEY`). The free tier gives 500 credits a month, and each `cfbr weekly` run spends 2 (one per region in `[odds].regions`). Calls are logged to `data/raw/_odds_calls.jsonl`, and `cfbr quota` shows the remaining credits without spending any. bet365 lines come from OddsPapi (`ODDSPAPI_KEY` in `.keys`, free tier of 250 requests a month): one request per game, fetched only for games the tracker is about to bet, logged to `data/raw/_oddspapi_calls.jsonl` and counted by `cfbr quota`.

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
| `cfbr check` | Daily dead-man's switch: alert when no successful `weekly` run is recorded since the last Sunday in season, or the backup is stale |
| `cfbr backup` | Copy the raw cache, news archive and paper-bet ledgers (with hash chains and history) to `[backup].target`; cron runs it nightly |
| `cfbr scores` | Fill in final scores on the site for games played since the last `weekly` (one CFBD call, only when a game should be over); display only, ratings and paper bets wait for Sunday. Cron runs it nightly |
| `cfbr odds-snapshot` | Pull an Odds API snapshot when a game kicks off within 6 h (the closing lines for paper-bet CLV); cron runs it Tue–Mon, plus weekdays 3× with `--if-games-within 168` for fresher site lines |
| `cfbr market-ratings --season 2026` | Diagnostic: invert the closing lines into market-implied ratings per slot and compare with the model's (`results/market_ratings_{season}.md`); no API calls |
| `cfbr moneyline-eval` | Market-only test: is a side's moneyline a better buy than its spread at −110, both priced with the points-to-cents table (`results/moneyline_pricing.md`); no API calls |
| `cfbr sim --season 2026 --week 6` | Simulate the rest of the season (win totals, conference titles, playoff odds) into `out/{season}/wNN/season_sim.*`; `cfbr weekly` runs it too |

`rank` and `weekly` use `config/tuned.toml` when it exists. Pass `--config` to override.

## Web app

A local web app for browsing the ratings:

```bash
uv run cfbr web build          # once; `cfbr weekly` keeps the current season up to date
uv run cfbr web serve          # http://127.0.0.1:8000 (add --host 0.0.0.0 --port 8000 for your LAN; there is no login)
```

The nav has four sections (Rankings, Games, Projections, Bets); routes are unchanged from before the reorganization, so old links keep working.

- **Rankings** (`/rankings`): any season, after any week (or preseason/final), filterable by conference and sortable. Shows rank and rating changes, AP rank and `vs AP`, and a trend sparkline; "More columns" adds conference, offense/defense, Coaches, CFP and prior share.
  - **Team pages** (`/team/{season}/{team}`): rating tiles, the rating explained, the season outlook from the latest simulation, schedule strength and résumé, a game log with the model's prediction, the line and the result. A **Roster** tab (`/roster/{season}/{team}`) lists who's on the roster with position, class, size and hometown; returner / transfer (with the previous school) / no prior stats; last season's production at any school; and departures and where they went. Player pages hang off it. Rosters come from CFBD via `cfbr.report.roster_web` and are free even during a live season.
- **Games** (`/games`): each week's predictions against the line, plus results, errors and against-the-spread grading once games are final. Flagged games (the model picks the underdog, or is `[bets].threshold`+ points off the line) are highlighted, and `?show=flagged` / `?show=upsets` narrows the table to them; `?conf=SEC` or `?conf=top25` filters by conference or AP top 25, and the table sorts by kickoff, win %, edge and error. The **Any matchup** tab (`/matchup`) prices any two FBS teams in the browser; `?home=&away=&neutral=1` makes a matchup shareable.
- **Projections** (`/odds`): the weekly season simulation from `cfbr weekly` (`out/{season}/wNN/season_sim.csv`): expected wins and range, and each team's chances to go undefeated, be bowl eligible, reach and win its conference title game, make the playoff, earn a bye, and win it all, plus per-conference title odds. The simulation's own notes (from `season_sim.md`) are collapsed under "How these are simulated". Premium while the season is live; free once it's over.
- **Bets** (`/bets`): the paper-betting ledger from `out/{season}/bets.csv`, led by open bets (each with its line at bet and the current line, so a bet the line has since moved toward is explained), then "How it works", the bankroll chart, results, tiers and tracked tests (collapsed until a bet settles), and settled bets. The **Replay** tab (`/bets/retro`, same gate) shows the retroactive replay from `cfbr bets-retro`, labelled as retroactive and kept separate from the live record. The **Shortlist** tab (`/bets/shortlist`, `/api/bets/shortlist`, same gate) narrows the paper bets to the ones worth a real bet at the viewer's chosen sportsbooks (rules in DECISIONS.md "Real-money shortlist"); the choice is a cookie set from a form on the page, and `?books=DraftKings,FanDuel` overrides it in the API.
- **Track record** (`/accuracy`, a tab under Rankings): walk-forward accuracy per season and per week against the closing line and the Massey baseline, and every season side by side. The `cfbr weekly` run log and every backtest run under `results/` (with diagnostics) are on `/lab`, which is linked from nowhere and sits behind the same `bets_access` gate as Bets (its diagnostics include edge-vs-close research).
- **Method** (`/method`, linked from the footer): every shared definition and convention, once.
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

### Deploying code changes

Code is written on a dev machine and reaches the server (javelin, `~/cfb-ratings`) only through GitHub: branch → PR → squash-merge to `main` → deploy. Never edit or commit on the server. From a local checkout:

```bash
deploy/deploy.sh    # DEPLOY_HOST / DEPLOY_DIR override javelin and cfb-ratings
```

It sends itself over ssh and, on the server: refuses a dirty working tree; holds the `cfbr weekly` lock (refuses while a run is in progress, and no run starts mid-deploy); fast-forwards to `origin/main` (refuses anything else); `uv sync --frozen`; runs the tests against the real data; rebuilds `out/web/`; restarts `cfbr-web`; and checks `/` (after its redirect to `/rankings`), `/accuracy`, `/bets` and `/api/bets` for 200 and `data/web.log` for new tracebacks. If any step after the fast-forward fails, it resets to the previous commit, rebuilds `out/web/` and restarts.

The restart is required: Jinja reloads templates from disk at once, but the Python already loaded (routes, row builders, filters) stays the old version until the process restarts, and a new template asking for a column the old code never produces returns a 500. Cron jobs run `uv run` from the checkout and pick up new code on their next run. Data-only changes (a `cfbr weekly` run, new files under `out/`) need no deploy.

For local development, copy the server's data once. Nothing in it needs an API call to rebuild, and the dev machine has no `.keys`, so it can't spend CFBD quota:

```bash
rsync -a --exclude='*.log' --exclude='.*.lock' javelin:cfb-ratings/data javelin:cfb-ratings/out javelin:cfb-ratings/results ./
```

On macOS, `tests/test_backup.py` needs GNU rsync (`brew install rsync`); the system `openrsync` handles `--relative --delete --backup` differently.

## Cron (Sunday mornings, Central time)

```cron
# System clock is UTC and Ubuntu's cron ignores CRON_TZ: each run fires at both possible
# UTC hours, and only the one matching the Chicago local hour proceeds (correct across DST).
# 9 AM: early rankings + predictions. 3 PM: re-run to pick up the AP/Coaches polls.
0 14,15 * * 0  [ "$(TZ=America/Chicago date +\%H)" = "09" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr weekly >> data/weekly.log 2>&1
0 20,21 * * 0  [ "$(TZ=America/Chicago date +\%H)" = "15" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr weekly >> data/weekly.log 2>&1
```

`odds-snapshot` pulls the pre-kickoff Odds API snapshot that grades paper-bet CLV. It reads kickoffs from the cached CFBD games file (no CFBD call) and exits without spending credits unless a game kicks off within 6 h, so the Sunday and Monday 4:40 PM runs cost nothing on a normal week. The weekday 9 AM, 1 PM and 9 PM runs pass `--if-games-within 168`, so they pull whenever a game is within a week: they keep the site's lines and the shortlist's "now" numbers fresh between Sunday's capture and the weekend (free in the offseason). About 46–50 credits a week in all, roughly 215 a month with `cfbr weekly`'s, against the free tier's 500. After saving a snapshot it also refreshes the site's Vegas lines for unplayed games from that snapshot (the median across books) and each book's line (`book_lines.parquet`, for the shortlist), patching `out/web/{season}/` and rewriting `meta.json` last; it skips the refresh while `cfbr weekly` or a deploy holds the lock, and a refresh failure alerts but never loses the snapshot. Finished games keep CFBD's closing line. No restart is needed: the server reloads on the new `meta.json`. Extra snapshots never move a paper bet: only `cfbr weekly` records bets, from the snapshot it takes itself, and closes use the newest snapshot in the 12 h before kickoff.

```cron
# Odds API snapshots (Central). Every weekday 9 AM, 1 PM and 9 PM whenever a game is within a week;
# closing snapshots Tue-Fri 4:40 PM, Sat 10:40 AM, 2:10 PM, 6:10 PM, 9:30 PM, and Sun and Mon 4:40 PM
# (for the rare Sunday/Monday game; free when nothing kicks off).
# Runs that land after UTC midnight (9 PM; Sat 6:10 PM CST = 00:10 UTC Sunday) get their own line.
5 14,15 * * 1-5   [ "$(TZ=America/Chicago date +\%H)" = "09" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot --if-games-within 168 >> data/odds-snapshot.log 2>&1
5 18,19 * * 1-5   [ "$(TZ=America/Chicago date +\%H)" = "13" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot --if-games-within 168 >> data/odds-snapshot.log 2>&1
0 2,3 * * 2-6     [ "$(TZ=America/Chicago date +\%H)" = "21" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot --if-games-within 168 >> data/odds-snapshot.log 2>&1
40 21,22 * * 2-5  [ "$(TZ=America/Chicago date +\%H)" = "16" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot >> data/odds-snapshot.log 2>&1
40 15,16 * * 6    [ "$(TZ=America/Chicago date +\%H)" = "10" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot >> data/odds-snapshot.log 2>&1
10 19,20 * * 6    [ "$(TZ=America/Chicago date +\%H)" = "14" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot >> data/odds-snapshot.log 2>&1
10 23 * * 6       [ "$(TZ=America/Chicago date +\%H)" = "18" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot >> data/odds-snapshot.log 2>&1
10 0 * * 0        [ "$(TZ=America/Chicago date +\%H)" = "18" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot >> data/odds-snapshot.log 2>&1
30 2,3 * * 0      [ "$(TZ=America/Chicago date +\%H)" = "21" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot >> data/odds-snapshot.log 2>&1
40 21,22 * * 0,1  [ "$(TZ=America/Chicago date +\%H)" = "16" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr odds-snapshot >> data/odds-snapshot.log 2>&1
```

`scores` fills in midweek final scores on the site (Tuesday–Friday games, and Saturday's early ones on Saturday night), so they don't sit unplayed until Sunday. It makes one CFBD `/games` call only when a game on the site is 3.5 h to 4 days past kickoff and not yet final, so it's free on nights without games; about 1 call per game night, 15–25 a month. It stops calling once 120 calls are left in the month (`[ingest].scores_quota_reserve`) so it never spends Sunday's. The response is saved as `data/raw/games/{season}/scores.json`, which the store never reads: ratings stay "after week N" and paper bets are settled only by `weekly`. Like the lines refresh, it skips while `weekly` or a deploy holds the lock, writes `meta.json` last, and needs no restart.

```cron
# Midweek scores (Central): 11 PM nightly, and a 9 AM catch-up Mon-Sat for late finishes.
# 11 PM Central is 04:00 or 05:00 UTC the next day, so every day-of-week runs.
0 4,5 * * *       [ "$(TZ=America/Chicago date +\%H)" = "23" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr scores >> data/scores.log 2>&1
15 14,15 * * 1-6  [ "$(TZ=America/Chicago date +\%H)" = "09" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr scores >> data/scores.log 2>&1
```

`weekly` takes a lock, so overlapping runs are refused. A week is only ingested once all of its FBS games are final, and re-running a week replaces that week's line in `out/{season}/accuracy.csv` instead of adding a duplicate. Exit codes: 3 = API quota nearly exhausted (the run still completes on cached data, so rankings for the last ingested week are published), 4 = another run in progress, 5 = no completed weeks yet. Failures (except 5) and post-run health-check problems send a push alert through ntfy: install the ntfy app and subscribe to the topic in `.keys` (`NTFY_TOPIC`). A successful run also sends a silent (priority `min`) "cfbr weekly ok" message carrying the paper-bet ledger's hash-chain head, which doubles as a heartbeat and as a timestamp for the ledger kept off this machine.

### Daily check (dead-man's switch) and log rotation

`cfbr weekly` records each successful run in `out/last_success.json`. `cfbr check` alerts (ntfy) when, during the season, no success is recorded since the last Sunday window (6 PM Central), or when the backup is stale; outside the season it is quiet. The same cron line rotates `data/*.log` with `deploy/logrotate.conf` (weekly, 12 kept, compressed).

```cron
# Daily at 8 PM Central.
0 1,2 * * *  [ "$(TZ=America/Chicago date +\%H)" = "20" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr check >> data/check.log 2>&1; /usr/sbin/logrotate -s /home/abtreece/cfb-ratings/data/.logrotate.state /home/abtreece/cfb-ratings/deploy/logrotate.conf
```

## Backups

`data/raw` can't be refetched under the free-tier quota, and the paper-bet ledgers (`out/{season}/bets*.csv`, `ledger_chain.jsonl`, `ledger_history/`) and `data/news` exist nowhere else, all gitignored. `cfbr backup` rsyncs them to `[backup].target` (default `~/cfbr-backup`; point it at another disk or an off-machine mount when one is available). Files deleted or changed since the previous run are kept under `target/_previous/{date}/` for 30 days. The weekly health check alerts when the last backup is more than 8 days old.

```cron
# Nightly at 3:30 AM Central.
30 8,9 * * *  [ "$(TZ=America/Chicago date +\%H)" = "03" ] && cd /home/abtreece/cfb-ratings && /home/abtreece/.local/bin/uv run cfbr backup >> data/backup.log 2>&1
```

Restore drill (run it once a season against a scratch directory, never over the live checkout):

```bash
T=~/cfbr-backup; R=$(mktemp -d)
rsync -a "$T/data" "$T/out" "$R/"                       # what a restore would put back
diff -rq data/raw "$R/data/raw" && echo raw ok          # identical to the live cache
uv run python -c "from pathlib import Path; from cfbr.bets.integrity import verify; print(verify(Path('$R/out'), 2026) or 'ledger chain ok')"
```

For a real restore, stop the cron jobs and `cfbr-web`, then `rsync -a "$T/data/" data/` and `rsync -a "$T/out/" out/`, then `cfbr ingest` with no flags (makes no API calls; it only rebuilds `data/cfb.duckdb` from the cache) and `cfbr prep`.

## Paper bets

`cfbr weekly` also maintains a paper-trading ledger. It never places a bet, and betting lines never feed the ratings. On each run it:

1. grades open bets whose games are final, and records the closing line and closing-line value (CLV),
2. records a new 1-unit bet at −110 for every upcoming game where the model and the consensus line (median across books in the newest line snapshot) differ by 3 or more points. A bet is recorded once, before kickoff, and never changed.

Outputs: `out/{season}/bets.csv` (the ledger) and `out/{season}/bets.md` (record, ROI, CLV, open bets). Lines come from the newest CFBD snapshot plus the newest Odds API snapshot from the same run, which adds FanDuel, theScore Bet, BetRivers, BetMGM, Hard Rock Bet and others with their prices. Each bet is graded twice: at the consensus line (−110), and at the best number across every non-offshore sportsbook, at that book's price. No book is excluded by state; offshore books (e.g. Bovada) count toward the consensus line but never price a bet. Settings are in `[bets]` in `config/default.toml`. Run it by hand with `cfbr bets --season 2026 --week 6`. `cfbr bets-retro --season 2026 --through-week 6` replays the same rules on weeks already played, for both the published (v1) and roster-prior models, against opening lines, into a separate `out/{season}/bets_retro.csv`/`.md` that never counts toward the live ledger. The summary draws no conclusion before 500 settled bets. Treat positive average CLV as the first sign of a real edge; win rate alone is mostly noise at small samples.

### Real bets

Bets you actually place are tracked in their own ledger, `out/{season}/bets_real.csv`, never mixed into the paper record, its hash chain or the 500-bet test. Record one on the server:

```bash
uv run cfbr real-bet --team Cincinnati --line 1.5 --odds -110 --stake 10 --placed "2026-10-11 23:06:17" --book DraftKings
```

`--line` is from the team's side (+1.5 → `1.5`), `--stake` is dollars risked, and `--placed` is Central time. The bet is matched to the team's next game after `--placed` in the cached schedule (no API call); pass `--game-id` if two games tie. It refuses a bet placed at or after kickoff and an exact duplicate. `cfbr weekly` settles open real bets at their own line and price and records CLV against the same close the paper bets use; a failure there exits 6 like the paper step. They show on `/bets` as "My real bets" only to the owner, never to other viewers (premium or not) and never in `/api/bets`. To become the owner, add `OWNER_KEY=<a long random string>` to `.keys` on the server (e.g. `openssl rand -hex 24`), restart `cfbr-web`, and open `/owner` once in your browser and enter that string (it is sent in a POST body, never the URL, so it stays out of `data/web.log` and browser history); it sets an HttpOnly cookie holding an HMAC of the key (180 days). `/owner/logout` clears it. With no `OWNER_KEY`, nobody is the owner. The site is plain HTTP, so the cookie can't be marked Secure; anyone who can sniff the network could copy it.

## Prior shadow comparison

The published ratings use the v1 preseason prior. `cfbr weekly` also computes every game's point-in-time prediction with the roster-based prior (player rosters, transfers, defensive returning production) and writes `out/{season}/prior_comparison.csv` and `.md`. The roster predictions are never published. Under a rule fixed in DECISIONS.md, the roster prior becomes the default before 2027 week 1 only if it beats v1 on 2026 games from week 7 onward and the postseason. Run it by hand with `cfbr shadow --season 2026`.

The same job runs a second shadow, calibration v2 (`out/{season}/calibration_comparison.*`): a second stage on each game prediction that corrects for how much of the two ratings is still prior and for games against a pooled FCS side. Ratings are unchanged and v2 predictions are never published; it is adopted for 2027 only under its pre-registered rule (DECISIONS.md, "M2+M3 calibration v2"). `cfbr shadow` writes both.

## Point-in-time guarantee

Every prediction for a game in slot *w* (regular-season week, or a 7-day postseason window) uses only plays and games that kicked off before that slot's first kickoff. Cross-season models (the preseason prior and the points calibration) use leave-one-season-out within the tuning seasons, and only earlier seasons for the holdout and live seasons. `tests/test_leakage.py` checks this with an explicit max-kickoff assertion and a canary future game.

## Tests

```bash
uv run pytest
```

Tests that need real data skip automatically when `data/cfb.duckdb` is absent.
