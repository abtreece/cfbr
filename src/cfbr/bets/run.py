"""Weekly paper-bet step: settle finished bets, then record new ones; and the midweek
settlement of bets whose games ended since (:func:`settle_midweek`, from ``cfbr scores``)."""

from __future__ import annotations

import fcntl
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import consensus_spread
from cfbr.bets.closing import snapshot_closing
from cfbr.bets.integrity import record
from cfbr.bets.ledger import (add_bets, candidate_bets, combine_books, market, read_ledger, settle,
                              snapshot_frame, summarize, summary_markdown)
from cfbr.bets.pricing import PricingTable, table_from_dataset
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
    table = pricing_table(cfg, ds, season)
    closing = closing_lines(cfg, ds, games, ledger, season)
    ledger = settle(ledger, games.select("game_id", "completed", "home_points", "away_points"), closing, table)

    books = current_books(cfg, games, season)
    preds_path = out_dir / f"w{week:02d}" / "predictions.csv"
    added = 0
    if books is None:
        log.warning("no live line snapshot for %d; no new bets recorded", season)
    elif not preds_path.exists():
        log.warning("%s missing; run `cfbr rank` first", preds_path)
    else:
        placed_at, book_lines = books
        now = datetime.now(UTC)
        preds = pl.read_csv(preds_path)
        new = candidate_bets(preds, book_lines, placed_at, cfg.bets, now=now, table=table)
        fresh = new.filter(~pl.col("game_id").is_in(ledger["game_id"].to_list()))
        extra = extra_pricing_lines(cfg, games, season, fresh["game_id"].to_list())
        if extra is not None and extra.height:
            new = candidate_bets(preds, book_lines, placed_at, cfg.bets, extra, now=now, table=table)
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
    md_tmp.write_text(summary_markdown(ledger, cfg.bets, season, table))
    md_tmp.replace(out_dir / "bets.md")
    m = summarize(ledger, cfg.bets)
    log.info("paper bets: %d new, %d open, %d settled (%d-%d-%d), units %+.2f, avg CLV %s",
             added, m["open"], m["settled"], m["won"], m["lost"], m["push"], m["units"],
             "n/a" if m["n_clv"] == 0 else f"{m['avg_clv']:+.2f}")
    return ledger_path


def run_real_bets(cfg: Config, ds: Dataset, season: int) -> Path | None:
    """Settle the owner's real bets (bets.real) against final scores and the paper ledger's
    close. Never touches bets.csv or its chain; does nothing when no real bet is recorded."""
    from cfbr.bets.real import ledger_path, read_real, settle_real, summarize_real, update_real

    path = ledger_path(cfg.paths.out_dir, season)
    if not path.exists():
        return None
    games = ds.games.filter(pl.col("season") == season)
    # Closes are looked up outside the lock (they read snapshot files); settling re-reads the
    # ledger under it, so a bet recorded meanwhile is kept (and settles on the next run).
    closing = closing_lines(cfg, ds, games, read_real(path), season)
    final = games.select("game_id", "completed", "home_points", "away_points")
    real = update_real(path, lambda df: settle_real(df, final, closing))
    m = summarize_real(real)
    log.info("real bets: %d open, %d settled (%d-%d-%d), %+.2f dollars", m["open"], m["settled"],
             m["won"], m["lost"], m["push"], m["dollars"])
    return path


def pricing_table(cfg: Config, ds: Dataset, season: int) -> PricingTable:
    """Points-to-cents table from every completed season before ``season`` with closing lines,
    cached at out/pricing_table.json (rebuilt when its seasons change). Lines never feed the ratings."""
    path = cfg.paths.out_dir / "pricing_table.json"
    want = sorted(ds.lines.filter(pl.col("season") < season)["season"].unique().to_list())
    if path.exists():
        try:
            t = PricingTable.load(path)
            if t.seasons == want:
                return t
        except (ValueError, KeyError) as e:  # a bad cache is just rebuilt
            log.warning("pricing table at %s unreadable (%s); rebuilding", path, e)
    t = table_from_dataset(ds, cfg, season - 1)
    t.save(path)
    log.info("pricing table built from %d games, seasons %s", t.n_games, t.seasons)
    return t


def closing_lines(cfg: Config, ds: Dataset, games: pl.DataFrame, ledger: pl.DataFrame, season: int) -> pl.DataFrame:
    """Closing line per game: the pre-kickoff multi-book snapshot consensus where one exists
    (see bets.closing), otherwise CFBD's post-game consensus."""
    cfbd = consensus_spread(ds.lines, cfg.backtest.spread_consensus).select(
        "game_id", closing_line="spread", closing_books=pl.col("n_providers").cast(pl.Int32)).with_columns(
        closing_source=pl.lit("cfbd"))
    snap = snapshot_closes(cfg, games, ledger, season)
    return pl.concat([snap, cfbd.filter(~pl.col("game_id").is_in(snap["game_id"].to_list()))], how="diagonal_relaxed")


def snapshot_closes(cfg: Config, games: pl.DataFrame, ledger: pl.DataFrame, season: int) -> pl.DataFrame:
    """game_id, closing_line, closing_books, closing_source="snapshot" for the final games
    (``games``: game_id, home_team, away_team, kickoff_ts, completed) of ``ledger``'s open bets
    that have a pre-kickoff Odds API snapshot close (see bets.closing)."""
    todo = games.filter(pl.col("game_id").is_in(ledger.filter(pl.col("status") == "open")["game_id"].to_list())
                        & pl.col("completed"))
    snap = snapshot_closing(cfg, todo.select("game_id", "home_team", "away_team", "kickoff_ts"), season)
    return snap.select("game_id", "closing_line", "closing_books").with_columns(closing_source=pl.lit("snapshot"))


def final_games(rows: list | None) -> pl.DataFrame:
    """The completed games in a CFBD /games response, as ``settle`` and ``snapshot_closes`` take them."""
    recs = [{"game_id": g["id"], "home_team": g.get("homeTeam"), "away_team": g.get("awayTeam"),
             "kickoff_ts": datetime.fromisoformat(g["startDate"].replace("Z", "+00:00")).astimezone(UTC),
             "completed": True, "home_points": g["homePoints"], "away_points": g["awayPoints"]}
            for g in rows or [] if g.get("completed") and g.get("startDate")
            and g.get("homePoints") is not None and g.get("awayPoints") is not None]
    return pl.DataFrame(recs, schema={"game_id": pl.Int64, "home_team": pl.Utf8, "away_team": pl.Utf8,
                                      "kickoff_ts": pl.Datetime("us", "UTC"), "completed": pl.Boolean,
                                      "home_points": pl.Int64, "away_points": pl.Int64})


def settle_midweek(cfg: Config, season: int, rows: list | None) -> tuple[int, int] | None:
    """Grade open paper bets (``bets.csv``) and real bets (``bets_real.csv``) whose games are
    final in ``rows`` (``cfbr scores``'s CFBD /games response), so /bets doesn't wait for
    Sunday. Same grading as :func:`run_bets` with the same close: only games with a pre-kickoff
    snapshot close are graded here, since one that would fall back to CFBD's close has to wait
    for Sunday's lines. Never records a bet. Takes the weekly lock (non-blocking). Returns None
    when it did nothing (lock held, no pricing table), else (paper bets graded, real bets graded)."""
    from cfbr.bets.real import ledger_path as real_path
    from cfbr.bets.real import read_real, settle_real, update_real

    final = final_games(rows)
    out_dir = cfg.paths.out_dir / str(season)
    ledger_path = out_dir / "bets.csv"
    table_path = cfg.paths.out_dir / "pricing_table.json"
    with open(cfg.paths.db_path.parent / ".weekly.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("weekly lock held; midweek settlement skipped (weekly settles them)")
            return None
        n_paper = n_real = 0
        ledger = read_ledger(ledger_path)
        closes = snapshot_closes(cfg, final, ledger, season)
        if closes.height:
            if not table_path.exists():
                log.warning("%s missing; paper bets wait for cfbr weekly", table_path)
                return None
            table = PricingTable.load(table_path)
            graded = final.filter(pl.col("game_id").is_in(closes["game_id"].to_list()))
            before = ledger.filter(pl.col("status") == "open").height
            ledger = settle(ledger, graded.select("game_id", "completed", "home_points", "away_points"), closes, table)
            n_paper = before - ledger.filter(pl.col("status") == "open").height
            if n_paper:
                tmp = ledger_path.with_suffix(".csv.tmp")
                ledger.sort("kickoff_ts", "game_id").write_csv(tmp)
                tmp.replace(ledger_path)
                record(cfg.paths.out_dir, season)
                md_tmp = out_dir / "bets.md.tmp"
                md_tmp.write_text(summary_markdown(ledger, cfg.bets, season, table))
                md_tmp.replace(out_dir / "bets.md")
        rp = real_path(cfg.paths.out_dir, season)
        if rp.exists():
            real = read_real(rp)
            rcloses = snapshot_closes(cfg, final, real, season)
            if rcloses.height:
                rgraded = final.filter(pl.col("game_id").is_in(rcloses["game_id"].to_list()))
                before = real.filter(pl.col("status") == "open").height
                real = update_real(rp, lambda df: settle_real(df, rgraded, rcloses))
                n_real = before - real.filter(pl.col("status") == "open").height
    if n_paper:
        m = summarize(ledger, cfg.bets)
        log.info("paper bets settled midweek: %d (season %d-%d-%d, avg CLV %s)", n_paper, m["won"], m["lost"],
                 m["push"], "n/a" if m["n_clv"] == 0 else f"{m['avg_clv']:+.2f}")
    return n_paper, n_real


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
