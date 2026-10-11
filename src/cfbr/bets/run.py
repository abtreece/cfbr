"""Weekly paper-bet step: settle finished bets, then record new ones."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import consensus_spread
from cfbr.bets.closing import snapshot_closing
from cfbr.bets.integrity import record
from cfbr.bets.ledger import (add_bets, candidate_bets, combine_books, market, read_ledger, settle,
                              snapshot_frame, summarize, summary_markdown)
from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.ingest.fetch import latest_lines_snapshot
from cfbr.ingest.odds import latest_odds_snapshot, odds_frame, team_aliases
from cfbr.ingest.oddspapi import oddspapi_key, pricing_lines

log = logging.getLogger(__name__)


def run_bets(cfg: Config, ds: Dataset, season: int, week: int) -> Path | None:
    """``week`` is the last completed week; new bets come from that week's
    predictions.csv (the next week's games) against the newest line snapshot."""
    out_dir = cfg.paths.out_dir / str(season)
    ledger_path = out_dir / "bets.csv"
    ledger = fill_books(read_ledger(ledger_path), cfg.paths.raw_dir, season, cfg.bets.offshore_books)

    games = ds.games.filter(pl.col("season") == season)
    # season_type and fbs_game are game metadata, not bet terms, so bets recorded before
    # those columns existed get them filled.
    meta = games.select("game_id", pl.col("season_type").alias("_st"),
                        (~(pl.col("home_fcs") | pl.col("away_fcs"))).alias("_fbs"))
    ledger = (ledger.join(meta, on="game_id", how="left")
              .with_columns(season_type=pl.coalesce("season_type", "_st"), fbs_game=pl.coalesce("fbs_game", "_fbs"))
              .drop("_st", "_fbs"))
    closing = closing_lines(cfg, ds, games, ledger, season)
    ledger = settle(ledger, games.select("game_id", "completed", "home_points", "away_points"), closing)

    books = current_books(cfg, games, season)
    preds_path = out_dir / f"w{week:02d}" / "predictions.csv"
    added = 0
    if books is None:
        log.warning("no live line snapshot for %d; no new bets recorded", season)
    elif not preds_path.exists():
        log.warning("%s missing; run `cfbr rank` first", preds_path)
    else:
        placed_at, book_lines = books
        preds = pl.read_csv(preds_path)
        new = candidate_bets(preds, book_lines, placed_at, cfg.bets)
        fresh = new.filter(~pl.col("game_id").is_in(ledger["game_id"].to_list()))
        extra = extra_pricing_lines(cfg, games, season, fresh["game_id"].to_list())
        if extra is not None and extra.height:
            new = candidate_bets(preds, book_lines, placed_at, cfg.bets, extra)
        new = (new.drop("season", "week", "season_type", "fbs_game")
               .join(games.select("game_id", pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
                                  "season_type", fbs_game=~(pl.col("home_fcs") | pl.col("away_fcs"))),
                     on="game_id", how="left"))
        ledger, added = add_bets(ledger, new)

    out_dir.mkdir(parents=True, exist_ok=True)
    # Temp file + rename so readers (e.g. the web app) never see a partial file.
    tmp = ledger_path.with_suffix(".csv.tmp")
    ledger.sort("kickoff_ts", "game_id").write_csv(tmp)
    tmp.replace(ledger_path)
    record(cfg.paths.out_dir, season)
    md_tmp = out_dir / "bets.md.tmp"
    md_tmp.write_text(summary_markdown(ledger, cfg.bets, season))
    md_tmp.replace(out_dir / "bets.md")
    m = summarize(ledger, cfg.bets)
    log.info("paper bets: %d new, %d open, %d settled (%d-%d-%d), units %+.2f, avg CLV %s",
             added, m["open"], m["settled"], m["won"], m["lost"], m["push"], m["units"],
             "n/a" if m["n_clv"] == 0 else f"{m['avg_clv']:+.2f}")
    return ledger_path


def closing_lines(cfg: Config, ds: Dataset, games: pl.DataFrame, ledger: pl.DataFrame, season: int) -> pl.DataFrame:
    """Closing line per game: the pre-kickoff multi-book snapshot consensus where one exists
    (see bets.closing), otherwise CFBD's post-game consensus."""
    cfbd = consensus_spread(ds.lines, cfg.backtest.spread_consensus).select(
        "game_id", closing_line="spread").with_columns(closing_source=pl.lit("cfbd"))
    todo = games.filter(pl.col("game_id").is_in(ledger.filter(pl.col("status") == "open")["game_id"].to_list())
                        & pl.col("completed"))
    snap = snapshot_closing(cfg, todo.select("game_id", "home_team", "away_team", "kickoff_ts"), season)
    snap = snap.select("game_id", "closing_line").with_columns(closing_source=pl.lit("snapshot"))
    return pl.concat([snap, cfbd.filter(~pl.col("game_id").is_in(snap["game_id"].to_list()))])


def extra_pricing_lines(cfg: Config, games: pl.DataFrame, season: int, game_ids: list[int]) -> pl.DataFrame | None:
    """bet365 (OddsPapi) lines for the games about to be bet. Only those games are fetched,
    to stay inside the free tier. Any failure (no key, budget, network) just means no
    bet365 pricing this run."""
    if not game_ids or not cfg.oddspapi.enabled or oddspapi_key() is None:
        return None
    teams_path = cfg.paths.raw_dir / "teams_fbs" / str(season) / "season.json"
    aliases = team_aliases(json.loads(teams_path.read_text())) if teams_path.exists() else {}
    try:
        return pricing_lines(cfg, season, games.filter(pl.col("game_id").is_in(game_ids))
                             .select("game_id", "home_team", "away_team", "kickoff_ts"), aliases)
    except Exception as e:  # noqa: BLE001 - pricing extras must never block recording bets
        log.warning("OddsPapi pricing lines failed (%s); bets priced without them", e)
        return None


def current_books(cfg: Config, games: pl.DataFrame, season: int) -> tuple[datetime, pl.DataFrame] | None:
    """Book-level lines to price new bets: the newest CFBD snapshot, plus the newest
    Odds API snapshot when the two were taken within ``odds.max_snapshot_gap_hours``
    (Odds API rows win for a book in both, since they carry prices). Bets count as
    placed at the later of the two times, so only games kicking off after both qualify."""
    raw = cfg.paths.raw_dir
    snap = latest_lines_snapshot(raw, season)
    odds = latest_odds_snapshot(raw, season) if cfg.odds.enabled else None
    frames, times = [], []
    if odds is not None:
        gap = None if snap is None else abs((odds[0] - snap[0]).total_seconds()) / 3600
        if gap is not None and gap > cfg.odds.max_snapshot_gap_hours:
            log.warning("Odds API snapshot is %.1f h from the CFBD snapshot; not used", gap)
        else:
            teams_path = raw / "teams_fbs" / str(season) / "season.json"
            aliases = team_aliases(json.loads(teams_path.read_text())) if teams_path.exists() else {}
            f = odds_frame(odds[1], games, aliases)
            log.info("Odds API: %d events, %d matched to games, %d book lines", len(odds[1]),
                     f["game_id"].n_unique(), f.height)
            frames.append(f)
            times.append(odds[0])
    if snap is not None:
        frames.append(snapshot_frame(snap[1]))
        times.append(snap[0])
    if not frames:
        return None
    return max(times), combine_books(*frames)


def fill_books(ledger: pl.DataFrame, raw_dir: Path, season: int, offshore: list[str]) -> pl.DataFrame:
    """Fill the per-book line column for bets recorded before it existed, from
    the same snapshot each bet was placed against (matched by placed_at)."""
    todo = ledger.filter(pl.col("books").is_null())
    if todo.height == 0:
        return ledger
    fills = []
    for placed in todo["placed_at"].unique().to_list():
        ts = datetime.fromisoformat(placed)
        f = raw_dir / "lines" / str(season) / "snapshots" / f"{ts:%Y%m%dT%H%M%SZ}.json"
        if f.exists():
            m = market(snapshot_frame(json.loads(f.read_text())), offshore).select("game_id", "books")
            fills.append(m.with_columns(placed_at=pl.lit(placed)))
    if not fills:
        return ledger
    f = pl.concat(fills).rename({"books": "books_fill"})
    return (ledger.join(f, on=["game_id", "placed_at"], how="left")
            .with_columns(books=pl.coalesce("books", "books_fill")).drop("books_fill"))
