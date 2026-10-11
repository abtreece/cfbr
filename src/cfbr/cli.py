"""cfbr command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from cfbr.config import load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("cfbr")

ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="TOML config path")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


@app.command()
def ingest(
    season: Annotated[Optional[int], typer.Option(help="Season for weekly incremental ingest")] = None,
    week: Annotated[Optional[int], typer.Option(help="Week for weekly incremental ingest")] = None,
    backfill: Annotated[bool, typer.Option(help="Pull all seasons from first_season to current")] = False,
    refresh: Annotated[bool, typer.Option(help="Ignore cache and re-fetch")] = False,
    config: ConfigOpt = None,
) -> None:
    """Fetch CFBD data into the raw cache and rebuild the DuckDB store."""
    from cfbr.ingest import fetch
    from cfbr.ingest.client import CFBDClient, QuotaExceeded
    from cfbr.ingest.store import build_store

    cfg = load_config(config)
    client = CFBDClient(cfg, refresh=refresh)
    stopped = False
    try:
        if backfill:
            fetch.backfill(client, cfg.ingest.first_season, fetch.current_season(), cfg.ingest.recruiting_lookback)
        elif season is not None and week is not None:
            fetch.weekly(client, season, week)
        elif season is not None or week is not None:
            raise typer.BadParameter("--season and --week must be given together")
    except QuotaExceeded as e:
        log.error("%s", e)
        stopped = True
    finally:
        client.close()
    log.info(
        "API calls this command: %d (cache hits %d); calls this month: %d of %d",
        client.calls, client.cache_hits, client.calls_this_month(), cfg.ingest.monthly_quota,
    )
    counts = build_store(cfg.paths.raw_dir, cfg.paths.db_path)
    log.info("store rebuilt: %s", counts)
    if stopped:
        raise typer.Exit(code=3)


@app.command()
def prep(config: ConfigOpt = None) -> None:
    """Build scrimmage_plays and model_plays from plays_raw."""
    from cfbr.prep.plays import build_prep

    cfg = load_config(config)
    log.info("prep: %s", build_prep(cfg))


@app.command()
def ratings(
    season: Annotated[int, typer.Option()],
    week: Annotated[int, typer.Option(help="Ratings available before this week kicks off")],
    top: Annotated[int, typer.Option()] = 25,
    config: ConfigOpt = None,
) -> None:
    """Print the ratings table for a (season, week)."""
    import time

    import polars as pl

    from cfbr.data import Dataset
    from cfbr.ingest.fetch import current_season
    from cfbr.model.engine import Engine

    cfg = load_config(config)
    eng = Engine(Dataset.load(cfg), cfg, last_completed=current_season() - 1)
    eng.calibration(season)
    t0 = time.perf_counter()
    tbl = eng.ratings(season, week)
    log.info("week fit %.2fs", time.perf_counter() - t0)
    with pl.Config(tbl_rows=top + 5, float_precision=2):
        typer.echo(tbl.select("team", "rating", "offense", "defense", "prior_share").head(top))
        typer.echo(tbl.filter(pl.col("team").str.starts_with("FCS")).select("team", "rating"))


@app.command()
def backtest(
    seasons: Annotated[str, typer.Option(help="e.g. 2015-2021 or 2015,2017")] = "2015-2021",
    config: ConfigOpt = None,
    tag: Annotated[str, typer.Option(help="Suffix for the run id")] = "",
) -> None:
    """Walk-forward backtest; writes results/{run_id}/."""
    from cfbr.backtest.harness import parse_seasons, run_backtest, write_results

    cfg = load_config(config)
    season_list = parse_seasons(seasons)
    preds, metrics = run_backtest(cfg, season_list)
    out = write_results(cfg, season_list, preds, metrics, tag)
    o = metrics["overall"]
    typer.echo(f"results: {out}")
    typer.echo(
        f"games {o['n_games']}  MAE {o['mae']:.2f}  RMSE {o['rmse']:.2f}  Massey MAE {o['massey_mae']:.2f}  "
        f"line MAE {o.get('line_mae', float('nan')):.2f} (model on same games {o.get('model_mae_lined', float('nan')):.2f})"
    )
    typer.echo(
        f"SU acc {o['su_accuracy']:.3f}  log loss {o['log_loss']:.3f}  Brier {o['brier']:.3f}  "
        f"ATS {o.get('ats_wins')}/{o.get('ats_n')} = {o.get('ats_win_rate', float('nan')):.3f} "
        f"[{o.get('ats_ci95_low', float('nan')):.3f}, {o.get('ats_ci95_high', float('nan')):.3f}]"
    )
    for b, m in metrics["by_bucket"].items():
        typer.echo(f"  {b:7s} n={m['n_games']:5d}  MAE {m['mae']:.2f}  line {m.get('line_mae', float('nan')):.2f}  "
                   f"Massey {m['massey_mae']:.2f}")


@app.command()
def tune(
    trials: Annotated[int, typer.Option()] = 256,
    jobs: Annotated[int, typer.Option()] = 30,
    config: ConfigOpt = None,
) -> None:
    """Tune on the tuning seasons, write config/tuned.toml and results/tuning_report.md (no holdout)."""
    import json

    from cfbr.backtest import tune as T
    from cfbr.backtest.harness import run_backtest, write_results
    from cfbr.config import REPO_ROOT

    cfg = load_config(config)
    t0, t1 = cfg.seasons.tuning
    seasons = list(range(t0, t1 + 1))
    cpath = str(config) if config else None
    out_dir = cfg.paths.results_dir / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = T.run_search(cpath, seasons, trials, jobs)
    trials_df = T.summarize_trials(results)
    trials_df.write_parquet(out_dir / "trials.parquet")
    best = min(results, key=lambda r: r["mae"])["params"]
    sens = T.sensitivity(cpath, best, seasons, jobs)
    mgrid = T.massey_grid(cpath, best, seasons, jobs)
    mbest = min(mgrid, key=lambda r: r["massey_mae"])
    best = best | {"margin_cap": mbest["margin_cap"], "massey_alpha": mbest["massey_alpha"]}
    (out_dir / "search.json").write_text(json.dumps(
        {"best": best, "sensitivity": sens, "massey_grid": mgrid}, indent=2, default=float))

    tuned_path = REPO_ROOT / "config" / "tuned.toml"
    tuned = T.write_frozen(cpath, best, tuned_path)
    preds, metrics = run_backtest(tuned, seasons)
    run_dir = write_results(tuned, seasons, preds, metrics, "tuned-tuning")
    default_metrics = run_backtest(load_config(config), seasons)[1]
    T.write_report(cfg.paths.results_dir / "tuning_report.md", best, metrics, default_metrics, sens, mgrid,
                   trials, run_dir)
    typer.echo(f"frozen: {tuned_path}\nreport: {cfg.paths.results_dir / 'tuning_report.md'}")


@app.command()
def holdout(config: ConfigOpt = None) -> None:
    """Evaluate the frozen config/tuned.toml on the holdout seasons (run once) and append to the report."""
    from cfbr.backtest import tune as T
    from cfbr.backtest.harness import run_backtest, write_results
    from cfbr.config import REPO_ROOT

    tuned = load_config(config or REPO_ROOT / "config" / "tuned.toml")
    h0, h1 = tuned.seasons.holdout
    seasons = list(range(h0, h1 + 1))
    preds, metrics = run_backtest(tuned, seasons)
    run_dir = write_results(tuned, seasons, preds, metrics, "tuned-holdout")
    T.append_holdout(tuned.paths.results_dir / "tuning_report.md", metrics, run_dir)
    typer.echo(f"holdout appended to {tuned.paths.results_dir / 'tuning_report.md'}")


def _default_config(config: Optional[Path]):
    """Explicit --config, else config/tuned.toml once frozen, else default."""
    from cfbr.config import REPO_ROOT

    tuned = REPO_ROOT / "config" / "tuned.toml"
    return load_config(config or (tuned if tuned.exists() else None))


@app.command()
def rank(
    season: Annotated[int, typer.Option()],
    week: Annotated[int, typer.Option(help="Last completed regular-season week")],
    config: ConfigOpt = None,
) -> None:
    """Write out/{season}/wNN/ rankings.csv, rankings.md, predictions.csv."""
    from cfbr.ingest.fetch import current_season
    from cfbr.report.rank import run_rank

    cfg = _default_config(config)
    log.info("config: %s", cfg.source)
    out = run_rank(cfg, season, week, last_completed=min(current_season(), season) - 1)
    typer.echo(f"wrote {out}")


EXIT_MEANING = {3: "API quota nearly used up", 4: "another run held the lock",
                6: "paper-bet step, prior shadow comparison or season simulation failed"}


@app.command()
def weekly(config: ConfigOpt = None) -> None:
    """Cron entry point: ingest -> prep -> rank -> season-to-date backtest line.

    Safe to re-run: a lock prevents overlapping runs, ingest is cached, and the
    accuracy line for a week is replaced rather than duplicated. Exit code 6
    means everything ran except the paper-bet step or the prior shadow comparison.
    Failures and post-run health-check problems send a push alert (see ``[alerts]``);
    exit 5 (no completed weeks yet, e.g. preseason) doesn't.
    """
    from cfbr.alerts import send
    from cfbr.bets.health import check
    from cfbr.ingest.fetch import current_season

    cfg = _default_config(config)
    code = 0
    try:
        _weekly_run(config)
    except typer.Exit as e:
        code = e.exit_code
        if code not in (0, 5):
            send(cfg, f"cfbr weekly failed (exit {code})", f"{EXIT_MEANING.get(code, 'see data/weekly.log')}. "
                 "Details in data/weekly.log.")
        raise
    except Exception as e:
        code = -1
        send(cfg, "cfbr weekly crashed", f"{type(e).__name__}: {e}"[:300] + " See data/weekly.log.")
        raise
    finally:
        if code in (0, 6):
            try:
                problems = check(cfg, current_season())
            except Exception as e:  # noqa: BLE001 - the check itself must not fail the run
                problems = [f"health check crashed: {type(e).__name__}: {e}"]
            if problems:
                log.warning("post-run check: %s", "; ".join(problems))
                send(cfg, "cfbr weekly: check found problems", "; ".join(problems))


def _weekly_run(config: ConfigOpt = None) -> None:
    import fcntl
    import json
    from datetime import UTC, datetime

    from cfbr.data import Dataset
    from cfbr.ingest import fetch
    from cfbr.ingest.client import CFBDClient, QuotaExceeded
    from cfbr.ingest.store import build_store
    from cfbr.model.engine import Engine
    from cfbr.prep.plays import build_prep
    from cfbr.report.rank import append_accuracy, run_rank, season_accuracy
    from cfbr.web.build import build_season, web_dir

    cfg = _default_config(config)
    cfg.paths.db_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(cfg.paths.db_path.parent / ".weekly.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("another cfbr weekly run holds the lock; exiting")
        raise typer.Exit(code=4)

    season = fetch.current_season()
    now = datetime.now(UTC)
    client = CFBDClient(cfg)
    try:
        cal = client.get("/calendar", {"year": season}, season, "season")
        started = [int(w["week"]) for w in cal if w["seasonType"] == "regular"
                   and datetime.fromisoformat(w["startDate"].replace("Z", "+00:00")) <= now]
        cal_week = max(started) if started else 1
        fetch.weekly(client, season, cal_week)
        games = fetch.latest_cached(client, "/games", season) or []
    except QuotaExceeded as e:
        log.error("%s", e)
        raise typer.Exit(code=3)
    finally:
        client.close()
    log.info("ingest API calls: %d; this month: %d of %d", client.calls, client.calls_this_month(),
             cfg.ingest.monthly_quota)
    # Odds API snapshot for the paper bets, taken right after the CFBD line snapshot. A
    # failure (no key, quota, network) only means bets are priced from CFBD lines alone.
    if cfg.bets.enabled and cfg.odds.enabled:
        from cfbr.ingest.odds import fetch_odds_snapshot, odds_api_key

        if odds_api_key() is None:
            log.info("no Odds API key; paper bets use CFBD lines only")
        else:
            try:
                fetch_odds_snapshot(cfg, season)
            except Exception as e:
                log.warning("Odds API snapshot failed (%s); paper bets use CFBD lines only", e)
    done = [wk for st, wk in fetch.completed_play_weeks(games) if st == "regular"]
    if not done:
        log.error("no completed weeks yet for %d", season)
        raise typer.Exit(code=5)
    week = max(done)
    log.info("season %d: calendar week %d, last completed week %d", season, cal_week, week)

    build_store(cfg.paths.raw_dir, cfg.paths.db_path)
    build_prep(cfg)
    ds = Dataset.load(cfg)
    out = run_rank(cfg, season, week, last_completed=season - 1, ds=ds)
    eng = Engine(ds, cfg, last_completed=season - 1)
    m = season_accuracy(eng, season, week)
    acc_path = cfg.paths.out_dir / str(season) / "accuracy.csv"
    append_accuracy(acc_path, season, week, m)
    log.info("season-to-date: %s", json.dumps({k: m.get(k) for k in ("n_games", "mae", "line_mae", "model_mae_lined",
                                                                     "ats_win_rate", "massey_mae")}, default=float))
    # Paper bets are isolated so a failure here can't block the web refresh.
    bets_failed = False
    if cfg.bets.enabled:
        from cfbr.bets.run import run_bets

        try:
            run_bets(cfg, ds, season, week)
        except Exception:
            log.exception("paper-bet step failed; continuing")
            bets_failed = True
    # Shadow prior comparison: also isolated, never affects published output.
    shadow_failed = False
    try:
        from cfbr.report.shadow import run_shadow

        run_shadow(cfg, ds, season, last_completed=season - 1)
    except Exception:
        log.exception("prior shadow comparison failed; continuing")
        shadow_failed = True
    try:
        from cfbr.report.shadow import run_totals_shadow

        run_totals_shadow(cfg, ds, season, last_completed=season - 1)
    except Exception:
        log.exception("totals shadow comparison failed; continuing")
        shadow_failed = True
    # Season simulation: isolated too.
    sim_failed = False
    if cfg.sim.enabled:
        try:
            from cfbr.sim.report import run_sim

            run_sim(eng, season, week)
        except Exception:
            log.exception("season simulation failed; continuing")
            sim_failed = True
    # Last, so a web-data failure can't cost the rankings or the accuracy line.
    web_out = build_season(eng, season, web_dir(cfg))
    typer.echo(f"wrote {out}, {acc_path} and {web_out}")
    if bets_failed or shadow_failed or sim_failed:
        raise typer.Exit(code=6)


@app.command()
def bets(
    season: Annotated[int, typer.Option()],
    week: Annotated[int, typer.Option(help="Last completed week (uses out/{season}/wNN/predictions.csv)")],
    config: ConfigOpt = None,
) -> None:
    """Settle finished paper bets and record new ones from the latest line snapshot."""
    from cfbr.bets.run import run_bets
    from cfbr.data import Dataset

    cfg = _default_config(config)
    path = run_bets(cfg, Dataset.load(cfg), season, week)
    typer.echo(f"wrote {path} and {path.with_name('bets.md')}")


@app.command()
def sim(
    season: Annotated[int, typer.Option()],
    week: Annotated[int, typer.Option(help="Last completed regular-season week (simulate from its ratings)")],
    n_sims: Annotated[Optional[int], typer.Option(help="Override [sim].n_sims")] = None,
    config: ConfigOpt = None,
) -> None:
    """Simulate the rest of the season: win totals, conference titles, playoff odds.
    Writes out/{season}/wNN/season_sim.csv and .md. Makes no API calls."""
    from cfbr.data import Dataset
    from cfbr.model.engine import Engine
    from cfbr.sim.report import run_sim

    cfg = _default_config(config)
    if n_sims:
        cfg.sim.n_sims = n_sims
    path = run_sim(Engine(Dataset.load(cfg), cfg, last_completed=season - 1), season, week)
    typer.echo(f"wrote {path} and {path.with_suffix('.md')}")


@app.command("bets-retro")
def bets_retro(
    season: Annotated[int, typer.Option()],
    through_week: Annotated[int, typer.Option(help="Last regular-season week to replay")],
    config: ConfigOpt = None,
) -> None:
    """Replay the paper-bet rules on weeks already played (v1 and roster models) into a
    separate out/{season}/bets_retro.* file. Makes no API calls."""
    from cfbr.bets.retro import retro_markdown, run_retro
    from cfbr.data import Dataset

    cfg = _default_config(config)
    led = run_retro(cfg, Dataset.load(cfg), season, through_week, last_completed=season - 1)
    out = cfg.paths.out_dir / str(season)
    out.mkdir(parents=True, exist_ok=True)
    led.write_csv(out / "bets_retro.csv")
    (out / "bets_retro.md").write_text(retro_markdown(led, cfg.bets, season, through_week))
    typer.echo(f"wrote {out / 'bets_retro.csv'} and {out / 'bets_retro.md'}")


@app.command()
def shadow(
    season: Annotated[int, typer.Option()],
    config: ConfigOpt = None,
) -> None:
    """Write out/{season}/prior_comparison.{csv,md}: v1 vs roster prior, point-in-time."""
    from cfbr.data import Dataset
    from cfbr.ingest.fetch import current_season
    from cfbr.report.shadow import run_shadow

    cfg = _default_config(config)
    path = run_shadow(cfg, Dataset.load(cfg), season, last_completed=min(current_season(), season) - 1)
    typer.echo(f"wrote {path} and {path.with_suffix('.md')}")


@app.command()
def quota(config: ConfigOpt = None) -> None:
    """Show this month's API call usage (local ledger and CFBD /info)."""
    from cfbr.ingest.client import CFBDClient

    cfg = load_config(config)
    client = CFBDClient(cfg)
    try:
        info = client.http.get("/info").json()  # /info does not count against the quota
    finally:
        client.close()
    typer.echo(f"local ledger: {client.calls_this_month()} calls this month")
    typer.echo(f"CFBD: {info.get('usedCalls')} used, {info.get('remainingCalls')} remaining, resets {info.get('resetAt')}")
    from cfbr.ingest.odds import odds_api_key, usage

    if odds_api_key() is not None:
        u = usage(cfg)  # /v4/sports does not cost credits
        typer.echo(f"The Odds API: {u['used']} credits used, {u['remaining']} remaining this month "
                   f"(each weekly run costs {len(cfg.odds.regions)})")
    from cfbr.ingest.oddspapi import calls_this_month, oddspapi_key

    if oddspapi_key() is not None:
        typer.echo(f"OddsPapi (bet365): {calls_this_month(cfg.paths.raw_dir)} of {cfg.oddspapi.monthly_quota} requests "
                   f"used this month (local ledger; {cfg.oddspapi.quota_reserve} held in reserve)")


news_app = typer.Typer(no_args_is_help=True, help="Timestamped news/availability archive.")
app.add_typer(news_app, name="news")


@news_app.command("collect")
def news_collect(
    source: Annotated[Optional[list[str]], typer.Option(help="Only these source names")] = None,
) -> None:
    """Fetch configured sources (config/news.toml) into data/news/. Safe to run hourly."""
    import fcntl

    from cfbr.news.collect import NEWS_DIR, collect

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    lock = open(NEWS_DIR / ".collect.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.error("another news collection is running; exiting")
        raise typer.Exit(code=4)
    results = collect(source)
    for name, res in results.items():
        typer.echo(f"{name:28s} {res}")


@news_app.command("purge")
def news_purge(
    source: Annotated[str, typer.Argument(help="Source name to delete from the archive")],
    apply: Annotated[bool, typer.Option(help="Actually delete (default: report only)")] = False,
) -> None:
    """Delete everything archived from a source that can't be relied on going forward."""
    from cfbr.news.collect import purge_source

    report = purge_source(source, apply=apply)
    typer.echo(("deleted: " if apply else "would delete (pass --apply): ") + str(report))


@news_app.command("status")
def news_status() -> None:
    """Per-source last run, result, and archive size."""
    import json

    from cfbr.news.collect import NEWS_DIR

    state = json.loads((NEWS_DIR / "state.json").read_text()) if (NEWS_DIR / "state.json").exists() else {}
    n_items = sum(1 for _ in open(NEWS_DIR / "items.jsonl")) if (NEWS_DIR / "items.jsonl").exists() else 0
    n_snaps = sum(1 for _ in open(NEWS_DIR / "snapshots.jsonl")) if (NEWS_DIR / "snapshots.jsonl").exists() else 0
    typer.echo(f"items: {n_items}  snapshots: {n_snaps}")
    for name, st in sorted(state.items()):
        typer.echo(f"{name:28s} {st.get('last_run', '-'):32s} {st.get('last_result', '-')}")


web_app = typer.Typer(no_args_is_help=True, help="Local web app over the ratings.")
app.add_typer(web_app, name="web")


@web_app.command("build")
def web_build(
    seasons: Annotated[Optional[str], typer.Option(help="e.g. 2015-2026 or 2026; default all seasons")] = None,
    config: ConfigOpt = None,
) -> None:
    """Precompute week-by-week ratings and game predictions into out/web/."""
    from cfbr.backtest.harness import parse_seasons
    from cfbr.web.build import build

    cfg = _default_config(config)
    log.info("config: %s", cfg.source)
    for out in build(cfg, parse_seasons(seasons) if seasons else None):
        typer.echo(f"wrote {out}")


@web_app.command("serve")
def web_serve(
    host: Annotated[str, typer.Option(help="Use 0.0.0.0 to allow other machines on the network")] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    config: ConfigOpt = None,
) -> None:
    """Serve the web app (reads out/web/; run `cfbr web build` first)."""
    import uvicorn

    from cfbr.web.app import create_app
    from cfbr.web.build import web_dir

    cfg = _default_config(config)
    if not any(web_dir(cfg).glob("*/meta.json")):
        log.warning("no web data in %s yet; run `cfbr web build`", web_dir(cfg))
    typer.echo(f"serving on http://{host}:{port}")
    uvicorn.run(create_app(cfg), host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
