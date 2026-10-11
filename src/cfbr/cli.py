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


@app.command()
def weekly(config: ConfigOpt = None) -> None:
    """Cron entry point: ingest -> prep -> rank -> season-to-date backtest line.

    Safe to re-run: a lock prevents overlapping runs, ingest is cached, and the
    accuracy line for a week is replaced rather than duplicated.
    """
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
    m = season_accuracy(Engine(ds, cfg, last_completed=season - 1), season, week)
    acc_path = cfg.paths.out_dir / str(season) / "accuracy.csv"
    append_accuracy(acc_path, season, week, m)
    log.info("season-to-date: %s", json.dumps({k: m.get(k) for k in ("n_games", "mae", "line_mae", "model_mae_lined",
                                                                     "ats_win_rate", "massey_mae")}, default=float))
    typer.echo(f"wrote {out} and {acc_path}")


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


if __name__ == "__main__":
    app()
