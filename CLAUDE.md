# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`cfbr` is a predictive college football rating engine. A team's rating is its expected point margin vs. an average FBS team on a neutral field: an opponent-adjusted ridge fit on play efficiency (PPA/EPA and success rate), shrunk toward a preseason prior and calibrated to points, and backtested walk-forward against Vegas closing lines. `cfb-ratings-spec.md` is the build spec (phases 1–7 are done, Phase 8 is a stretch goal). `DECISIONS.md` records every judgment call, one line each: decision — reason — phase. When you make a non-obvious choice, add a line there.

## Commands

```bash
uv sync                                   # Python 3.12 only (>=3.12,<3.13)
uv run pytest                             # all tests
uv run pytest tests/test_leakage.py       # one file
uv run pytest tests/test_bets.py -k clv   # one test by name
uv run cfbr --help                        # CLI (typer); see README for the full command table
```

Tests that use the `dataset` fixture (`tests/conftest.py`) need real data in `data/cfb.duckdb` and skip when it isn't there. There is no lint config.

The CFBD API key comes from `CFBD_API_KEY` or a gitignored `.keys` file in the repo root. The account is on the **free tier (1,000 calls/month)**. Every call is logged to `data/raw/_calls.jsonl`, and raw responses are cached under `data/raw/`, so re-runs make no calls. Don't run `--refresh`, a backfill, or anything else that hits the API without a good reason. `cfbr quota` is free (`/info` isn't counted).

## Pipeline and data flow

```
CFBD API → data/raw/ (JSON cache) → data/cfb.duckdb (build_store rebuilds it from the cache)
        → prep: scrimmage_plays / model_plays → Dataset (in-memory polars) → Engine
        → rank (out/{season}/wNN/), backtest (results/{run_id}/), bets (out/{season}/bets.*),
          web build (out/web/{season}/) → web serve (read-only FastAPI app)
```

- `ingest/`: `client.py` handles HTTP, caching and quota ledger, `fetch.py` decides what to fetch (backfill/weekly, completed-week logic, live line snapshots), and `store.py` rebuilds DuckDB from the raw cache. The store is always derived from the cache, never edited in place.
- `prep/plays.py`: play-type filtering, score at the snap, the garbage-time filter, success, and FCS pooling.
- `data.py`: `Dataset.load` reads DuckDB once per run. `assign_slots` defines **prediction slots**: `r{week}` for regular-season weeks and `p{n}` for 7-day postseason buckets. `training_slice` is the point-in-time filter.
- `model/engine.py`: `Engine` is the core. It lazily caches, per season, `final()` (no-prior full-season fit, used as prior targets and features), `priors()`, `snapshot(season, as_of)` (fit on plays with kickoff < as_of, a prefix found by binary search over kickoff-sorted arrays), `calibration()` and `predict()`. The ridge math is in `ridge.py`, the prior regression in `prior.py`, the points calibration in `calibrate.py`, and the Massey baseline in `massey.py`.
- `model/schemes.py`: which past seasons may train the cross-season models (prior, calibration). `hybrid` uses leave-one-season-out inside the tuning seasons (2015–2021) and a strict expanding window for holdout (2022–2025) and live seasons. This was a user-approved decision; don't change it casually.
- `Engine(..., last_completed=...)` sets which seasons count as complete. The CLI passes `current_season() - 1` (or `season - 1` for `rank`).
- `backtest/`: walk-forward harness, metrics (MAE vs. the closing line, ATS with Wilson CI, log loss, Brier), and Optuna tuning. `config/tuned.toml` is the frozen tuned config. The holdout was evaluated once (`cfbr holdout`). Don't re-tune or iterate on holdout seasons.
- `report/rank.py`: weekly ranking output and the season-to-date accuracy line.
- `bets/`: paper-betting tracker (it never places bets). `ledger.py` holds the append-only ledger, `read_ledger` and `summarize`. `run.py` settles open bets and records new ones. The rules were fixed before any results came in (see DECISIONS.md "Bets").
- `web/`: `build.py` precomputes everything into `out/web/{season}/`. The server (`app.py`, `store.py`, Jinja templates, no CDN assets) **never fits a model**; it only reads files. Betting data lives only in `web/bets.py` (`/bets`, `/api/bets`), and every route there is gated by `web/access.py:bets_access` so the section can become paid later. No other page or API response may include ledger data (tested in `tests/test_web.py`).

`cfbr weekly` is the cron entry point (Sunday 9am and 3pm Central; see README). It takes a lock and runs ingest → build_store → prep → rank → accuracy line → paper bets (isolated; a failure there exits with 6) → web season rebuild (last). Exit codes: 3 = quota, 4 = lock held, 5 = no completed weeks, 6 = bets step failed.

## Where things run

Production (cron, `cfbr-web`, the API key, the paper-bet ledger of record) is javelin, `~/cfb-ratings`. Development happens in a local checkout: branch → PR → squash-merge → `deploy/deploy.sh` (see README "Deploying code changes"). Never edit, commit or run git writes on javelin. The local checkout has a copy of `data/`, `out/` and `results/` but no `.keys`. Don't add one, and don't run `weekly`, `bets`, `odds-snapshot`, `backup` or `check` locally: they spend quota, write ledgers or send alerts that belong to production. Deploying is a stop-and-ask step.

## Invariants (don't break these)

- **Point-in-time:** a prediction for a game in slot *w* uses only plays and games that kicked off before that slot's first kickoff. Cross-season models follow `schemes.training_seasons`. `tests/test_leakage.py` enforces this with a max-kickoff assertion and a canary future game. Any doubt about leakage is a reason to stop and ask.
- **Betting lines are never model inputs.** They're used only for evaluation and the paper-bet ledger.
- **Published ratings (SP+, FPI, etc.) are never used at all**: not as inputs, prior targets, calibration targets or tuning objectives, and never ingested. Borrowed public conventions (Football Outsiders' success rate, Connelly's returning production and garbage-time thresholds) get credited (see DECISIONS.md "Independence from published ratings").
- The closing spread is the median of `spread` across providers (home perspective; Vegas home margin = −spread).
- The site's "after week N" ratings must exactly match `cfbr rank`'s `out/{season}/wNN/` output (tested).
- The `lines` table holds the latest row per game-provider. Live line snapshots are kept separately under `data/raw/lines/{season}/snapshots/`.
- Offshore books (`[bets].offshore_books`) count toward the consensus line but are never used to price a bet. CFBD non-books (projections, aggregators) never price a bet either. All books are shown regardless of jurisdiction.
- Config: every tunable lives in `config/default.toml` (pydantic schema in `config.py`). `rank`/`weekly`/`bets`/`web` use `config/tuned.toml` when it exists, and `--config` overrides. Keep old keys loadable (for example, `prior_k` stays in the schema even though it's unused) so `tuned.toml` keeps working.

## Conventions

- Dataframes use polars and storage uses DuckDB. HTTP goes through `httpx` directly, not the `cfbd` client.
- Commit messages name the phase or feature (for example, "Phase 6: …", "Paper bets: …", "Web app: …").
- `data/`, `out/` and `results/` are gitignored outputs.
