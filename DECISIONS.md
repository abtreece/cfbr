# Decisions

One line each: decision — reason — phase.

- Prior and margin-calibration coefficients use a hybrid fit: leave-one-season-out within tuning seasons 2015–2021, strict expanding window for holdout 2022–2025 and live seasons — keeps headline holdout metrics leak-free while giving early seasons stable weights (user-approved 2026-10-11) — P3/P4
- API usage budgeted against the free tier (1,000 calls/month); every command logs its call count and the backfill is resumable — free tier key; overage can disable the key — P1
- HTTP via `httpx` rather than the `cfbd` client — thin, explicit control over caching and call counting — P0
- Dataframes via polars; storage in DuckDB — spec default — P0
- API key loaded from `CFBD_API_KEY` env var, falling back to a gitignored `.keys` file in the repo root — key was provided in `.keys` — P0
- Games and lines fetched with `seasonType=both` (one call per season each); plays fetched one call per (seasonType, week) — minimizes calls; ~24 calls/season, full 2014–2026 backfill used 297 calls — P1
- Live-season games/lines cached under a week-keyed file (`both_wNN.json`) so a weekly re-run is served from cache, while new weeks fetch fresh data — P1
- Plays for a week are fetched only after every FBS game that week is marked completed — prevents caching a partial week — P1
- `/info` is not counted against the quota (verified: remaining unchanged after the call), so `cfbr quota` uses it freely — P1
- DuckDB table for returning production is `returning_production` (`returning` is a reserved word) — P1
- Team FBS/FCS status per season: FBS if listed by `/teams/fbs` that year, else the classification CFBD attaches in that season's games; anything not FBS (FCS, D-II, unknown) is pooled as FCS — P1
- `/talent` starts in 2015 (no 2014 rows); not needed since 2015 is the first season with a prior — P1
- `/player/returning` has offense-only fields (percentPPA, usage and passing/rushing/receiving splits); no defensive returning production exists, so defensive priors use previous rating, talent and recruiting only — P4
