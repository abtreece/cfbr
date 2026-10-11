import json
from datetime import UTC, datetime, timedelta

import duckdb
import polars as pl

from cfbr.bets.closing import games_kicking_off, games_kicking_off_cached
from cfbr.bets.ledger import LEDGER_SCHEMA
from cfbr.bets.run import fill_books

NOW = datetime(2026, 10, 15, 18, tzinfo=UTC)


def _games_db(path, rows):
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE games (game_id BIGINT, season INTEGER, completed BOOLEAN, kickoff_ts TIMESTAMPTZ)")
        con.executemany("INSERT INTO games VALUES (?, ?, ?, ?)", rows)


def test_games_kicking_off_counts_unstarted_games_in_the_window(tmp_path):
    db = tmp_path / "x.duckdb"
    _games_db(db, [(1, 2026, False, NOW + timedelta(hours=2)),     # in the window
                   (2, 2026, False, NOW + timedelta(hours=5)),     # in the window
                   (3, 2026, False, NOW + timedelta(hours=7)),     # too far out
                   (4, 2026, False, NOW - timedelta(hours=1)),     # already kicked off
                   (5, 2026, True, NOW + timedelta(hours=1)),      # completed (stale flag) never counts
                   (6, 2025, False, NOW + timedelta(hours=1))])    # other season
    assert games_kicking_off(db, 2026, 6, NOW) == 2
    assert games_kicking_off(db, 2026, 8, NOW) == 3
    assert games_kicking_off(db, 2026, 1, NOW) == 0


def _cached(raw, season, rows, name="both.json"):
    d = raw / "games" / str(season)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(rows))


def _row(gid, start, completed=False):
    return {"id": gid, "homeTeam": "A", "awayTeam": "B", "startDate": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "completed": completed}


def test_games_kicking_off_cached_prefers_the_cache_and_falls_back_to_the_store(tmp_path):
    raw = tmp_path / "raw"
    db = tmp_path / "x.duckdb"
    _games_db(db, [(1, 2026, False, NOW + timedelta(hours=2))])
    # No cached file: the store answers (and 0 without a store).
    assert games_kicking_off_cached(raw, 2026, 6, NOW, db_path=db) == 1
    assert games_kicking_off_cached(raw, 2026, 6, NOW) == 0
    # A weekly file is used when both.json is absent; the newest one wins.
    _cached(raw, 2026, [_row(1, NOW + timedelta(hours=2))], "both_w06.json")
    _cached(raw, 2026, [_row(1, NOW + timedelta(days=3)),               # kickoff moved mid-week
                        _row(2, NOW + timedelta(hours=3)),
                        _row(3, NOW + timedelta(hours=1), completed=True)], "both_w07.json")
    assert games_kicking_off_cached(raw, 2026, 6, NOW, db_path=db) == 1
    _cached(raw, 2026, [_row(1, NOW + timedelta(hours=2)), _row(2, NOW + timedelta(hours=3))])
    assert games_kicking_off_cached(raw, 2026, 6, NOW, db_path=db) == 2


def test_fill_books_uses_the_snapshot_each_bet_was_placed_against(tmp_path):
    snaps = tmp_path / "lines" / "2026" / "snapshots"
    snaps.mkdir(parents=True)
    (snaps / "20261011T140000Z.json").write_text(json.dumps([
        {"id": 1, "lines": [{"provider": "DraftKings", "spread": -3.0}, {"provider": "Bovada", "spread": -3.5},
                            {"provider": "numberfire", "spread": -2.0}]},
        {"id": 2, "lines": [{"provider": "DraftKings", "spread": 7.0}]}]))
    rows = [{"game_id": 1, "placed_at": "2026-10-11T14:00:00+00:00", "books": None},
            {"game_id": 2, "placed_at": "2026-10-11T14:00:00+00:00", "books": "kept"},    # already filled
            {"game_id": 3, "placed_at": "2026-10-04T14:00:00+00:00", "books": None}]      # no snapshot file
    ledger = pl.DataFrame(rows).cast({"game_id": pl.Int64, "placed_at": pl.Utf8, "books": pl.Utf8})
    out = fill_books(ledger, tmp_path, 2026, ["Bovada"])
    got = dict(out.select("game_id", "books").iter_rows())
    assert got[1] == "Bovada -3.5 (offshore); DraftKings -3"
    assert got[2] == "kept" and got[3] is None
    assert out.columns == ledger.columns and out.height == 3
    # Nothing to fill: the ledger comes back unchanged.
    assert fill_books(out.filter(pl.col("game_id") != 3), tmp_path, 2026, []).height == 2
    assert fill_books(pl.DataFrame(schema=LEDGER_SCHEMA), tmp_path, 2026, []).height == 0
