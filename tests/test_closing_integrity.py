import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cfbr.bets import integrity
from cfbr.bets.closing import snapshot_closing
from cfbr.config import Config, Paths

KICK = datetime(2026, 10, 17, 19, 30, tzinfo=UTC)
TEAMS = [{"school": "Tennessee", "mascot": "Volunteers", "alternateNames": []},
         {"school": "Alabama", "mascot": "Crimson Tide", "alternateNames": []}]


def _cfg(tmp_path):
    return Config(paths=Paths(raw_dir=tmp_path / "raw", db_path=tmp_path / "x.duckdb", results_dir=tmp_path,
                              out_dir=tmp_path / "out"))


def _snap(cfg, ts, spreads):
    d = cfg.paths.raw_dir / "odds" / "2026" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    books = [{"key": b, "title": b, "markets": [{"key": "spreads", "outcomes": [
        {"name": "Tennessee Volunteers", "point": s, "price": -110},
        {"name": "Alabama Crimson Tide", "point": -s, "price": -110}]}]} for b, s in spreads.items()]
    ev = [{"id": "e1", "commence_time": KICK.strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": "Tennessee Volunteers",
           "away_team": "Alabama Crimson Tide", "bookmakers": books}]
    (d / f"{ts:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(ev))


def test_closing_uses_the_last_snapshot_before_kickoff(tmp_path):
    cfg = _cfg(tmp_path)
    t = cfg.paths.raw_dir / "teams_fbs" / "2026"
    t.mkdir(parents=True)
    (t / "season.json").write_text(json.dumps(TEAMS))
    games = pl.DataFrame({"game_id": [1], "home_team": ["Tennessee"], "away_team": ["Alabama"], "kickoff_ts": [KICK]})
    assert snapshot_closing(cfg, games, 2026).height == 0                      # no snapshots yet
    _snap(cfg, KICK - timedelta(days=5), {"DraftKings": 7.0})                # too old to be a close
    assert snapshot_closing(cfg, games, 2026).height == 0
    _snap(cfg, KICK - timedelta(hours=3), {"DraftKings": 6.5, "FanDuel": 6.5, "BetMGM": 5.5})
    _snap(cfg, KICK + timedelta(minutes=5), {"DraftKings": 1.0})             # after kickoff: ignored
    r = snapshot_closing(cfg, games, 2026).row(0, named=True)
    assert (r["closing_line"], r["closing_books"]) == (6.5, 3)


def _ledger(cfg, rows):
    d = cfg.paths.out_dir / "2026"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(d / "bets.csv")


ROW = {"game_id": 1, "placed_at": "2026-10-11T16:24:08+00:00", "kickoff_ts": "2026-10-17T19:30:00Z", "side": "home",
       "team": "Tennessee", "model_margin": 1.0, "line": 6.5, "edge": 7.5, "best_line": 7.0, "best_provider": "FanDuel",
       "odds": -110, "stake": 1.0, "best_odds": -105, "status": "open", "fbs_game": None}


def test_ledger_chain_records_changes_and_catches_tampering(tmp_path):
    cfg = _cfg(tmp_path)
    out = cfg.paths.out_dir
    t0 = datetime(2026, 10, 11, 16, tzinfo=UTC)
    _ledger(cfg, [ROW])
    e1 = integrity.record(out, 2026, t0)
    assert e1["bets"] == 1 and integrity.record(out, 2026, t0) is None        # unchanged: no new entry
    # Allowed: metadata filled in, results added at settlement.
    _ledger(cfg, [ROW | {"fbs_game": True, "status": "won"}])
    integrity.record(out, 2026, t0 + timedelta(days=7))
    assert integrity.verify(out, 2026) == []
    # Not allowed: a recorded bet's line changing.
    _ledger(cfg, [ROW | {"fbs_game": True, "status": "won", "line": 7.5}])
    assert "bets.csv changed outside the weekly run (not in the hash chain)" in integrity.verify(out, 2026)
    integrity.record(out, 2026, t0 + timedelta(days=8))
    assert any("recorded bet 1 changed terms" in p for p in integrity.verify(out, 2026))
    # Editing a history file breaks its hash.
    hist = sorted((out / "2026" / "ledger_history").glob("*.csv"))[0]
    hist.write_text(hist.read_text().replace("16:24:08", "16:24:09"))
    assert any("missing or altered" in p for p in integrity.verify(out, 2026))


def test_dropped_bets_are_flagged(tmp_path):
    cfg = _cfg(tmp_path)
    out = cfg.paths.out_dir
    t0 = datetime(2026, 10, 11, 16, tzinfo=UTC)
    _ledger(cfg, [ROW, ROW | {"game_id": 2}])
    integrity.record(out, 2026, t0)
    _ledger(cfg, [ROW])
    integrity.record(out, 2026, t0 + timedelta(hours=1))
    assert any("1 recorded bets missing" in p for p in integrity.verify(out, 2026))
