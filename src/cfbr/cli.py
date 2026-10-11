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
