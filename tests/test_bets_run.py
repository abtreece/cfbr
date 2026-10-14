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


# -- midweek settlement (cfbr scores) -------------------------------------------------

def _midweek(tmp_path):
    from cfbr.bets.pricing import PricingTable
    from tests.test_closing_integrity import KICK, ROW, TEAMS, _cfg, _ledger, _snap

    cfg = _cfg(tmp_path)
    t = cfg.paths.raw_dir / "teams_fbs" / "2026"
    t.mkdir(parents=True)
    (t / "season.json").write_text(json.dumps(TEAMS))
    _snap(cfg, KICK - timedelta(hours=3), {"DraftKings": 6.0, "FanDuel": 6.0, "BetMGM": 5.0})  # the close: 6
    # Game 1 (Tennessee +6.5 vs Alabama) has a pre-kickoff close; game 2 has none.
    _ledger(cfg, [ROW, ROW | {"game_id": 2, "team": "Ohio", "kickoff_ts": "2026-10-17T23:00:00Z"}])
    pmf = {3: 0.2, 7: 0.1, -3: 0.05, 1: 0.1, 4: 0.1, 10: 0.15, 14: 0.1, 2: 0.1, -10: 0.1}
    PricingTable(seasons=[2025], n_games=1000, window=2.0, min_games=300,
                 pmf={m: pmf for m in range(36)}).save(cfg.paths.out_dir / "pricing_table.json")
    rows = [{"id": 1, "homeTeam": "Tennessee", "awayTeam": "Alabama", "startDate": "2026-10-17T19:30:00.000Z",
             "completed": True, "homePoints": 20, "awayPoints": 24},
            {"id": 2, "homeTeam": "Ohio", "awayTeam": "Toledo", "startDate": "2026-10-17T23:00:00.000Z",
             "completed": True, "homePoints": 30, "awayPoints": 10},
            {"id": 3, "homeTeam": "Iowa", "awayTeam": "Purdue", "startDate": "2026-10-18T00:00:00.000Z",
             "completed": False, "homePoints": None, "awayPoints": None}]
    return cfg, rows


def test_midweek_settlement_grades_bets_with_a_snapshot_close_like_sunday(tmp_path):
    from cfbr.bets import integrity
    from cfbr.bets.ledger import read_ledger, settle
    from cfbr.bets.pricing import PricingTable
    from cfbr.bets.run import final_games, settle_midweek, snapshot_closes

    cfg, rows = _midweek(tmp_path)
    path = cfg.paths.out_dir / "2026" / "bets.csv"
    before = read_ledger(path)
    assert settle_midweek(cfg, 2026, rows) == (1, 0)
    led = read_ledger(path).sort("game_id")
    g1, g2 = led.to_dicts()
    # Tennessee +6.5 lost by 4: covered. Close 6 (median of 6, 6, 5): +0.5 CLV from the snapshot.
    assert (g1["status"], g1["home_points"], g1["away_points"]) == ("won", 20, 24)
    assert (g1["closing_line"], g1["closing_source"], g1["clv"]) == (6.0, "snapshot", 0.5)
    assert g1["clv_cents"] is not None and g1["best_status"] == "won"
    assert g2["status"] == "open"  # final, but its close would be CFBD's: Sunday grades it
    # Exactly what Sunday's settle gives for the same scores and close.
    final = final_games(rows)
    closes = snapshot_closes(cfg, final, before, 2026)
    sunday = settle(before, final.filter(pl.col("game_id") == 1).select("game_id", "completed", "home_points",
                                                                        "away_points"),
                    closes, PricingTable.load(cfg.paths.out_dir / "pricing_table.json"))
    assert sunday.sort("game_id").to_dicts()[0] == g1
    # Through the hash chain, with bets.md rewritten; a second run changes nothing.
    assert len(integrity._entries(cfg.paths.out_dir / "2026" / "ledger_chain.jsonl")) == 1
    assert integrity.verify(cfg.paths.out_dir, 2026) == []
    assert (cfg.paths.out_dir / "2026" / "bets.md").exists()
    assert settle_midweek(cfg, 2026, rows) == (0, 0)
    assert len(integrity._entries(cfg.paths.out_dir / "2026" / "ledger_chain.jsonl")) == 1


def test_midweek_settlement_grades_real_bets_and_waits_for_the_lock(tmp_path):
    import fcntl

    from cfbr.bets.ledger import read_ledger
    from cfbr.bets.real import ledger_path, read_real
    from cfbr.bets.run import settle_midweek

    cfg, rows = _midweek(tmp_path)
    real = ledger_path(cfg.paths.out_dir, 2026)
    pl.DataFrame([{"game_id": 1, "season": 2026, "week": 8, "kickoff_ts": "2026-10-17T19:30:00Z",
                   "home_team": "Tennessee", "away_team": "Alabama", "side": "home", "team": "Tennessee",
                   "line": 7.0, "odds": -105, "risk_dollars": 21.0, "book": "FanDuel",
                   "placed_at": "2026-10-12T15:00:00+00:00", "recorded_at": "2026-10-12T15:01:00+00:00",
                   "status": "open"}]).write_csv(real)
    cfg.paths.db_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.paths.db_path.parent / ".weekly.lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert settle_midweek(cfg, 2026, rows) is None
    assert read_ledger(cfg.paths.out_dir / "2026" / "bets.csv")["status"].to_list() == ["open", "open"]
    assert settle_midweek(cfg, 2026, rows) == (1, 1)
    r = read_real(real).to_dicts()[0]
    assert (r["status"], r["dollars"], r["closing_line"], r["clv"]) == ("won", 20.0, 6.0, 1.0)
    # Nothing final in the response (or no response): nothing to do.
    assert settle_midweek(cfg, 2026, None) == (0, 0)
