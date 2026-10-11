"""The paper-bet health check's snapshot, match-rate and Sunday-bets problems (all read-only)."""

import json
from datetime import UTC, datetime, timedelta

import polars as pl

from cfbr.backup import STAMP
from cfbr.bets import health
from cfbr.bets.ledger import LEDGER_SCHEMA
from cfbr.config import Config, Paths

SUNDAY = datetime(2026, 10, 11, 14, 30, tzinfo=UTC)      # a Sunday run
TEAMS = [{"school": "Tennessee", "mascot": "Volunteers", "alternateNames": []},
         {"school": "Alabama", "mascot": "Crimson Tide", "alternateNames": []},
         {"school": "Texas", "mascot": "Longhorns", "alternateNames": []},
         {"school": "Oklahoma", "mascot": "Sooners", "alternateNames": []}]


def _cfg(tmp_path):
    cfg = Config(paths=Paths(raw_dir=tmp_path / "raw", db_path=tmp_path / "x.duckdb", results_dir=tmp_path,
                             out_dir=tmp_path / "out"))
    cfg.backup.target = str(tmp_path / "backup")
    (tmp_path / "backup").mkdir()
    (tmp_path / "backup" / STAMP).write_text(json.dumps({"finished_at": SUNDAY.isoformat()}))
    return cfg


def _ledger(cfg, rows):
    d = cfg.paths.out_dir / "2026"
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=LEDGER_SCHEMA).write_csv(d / "bets.csv")


def _bet(gid, recorded_at, placed_at=None):
    return {"game_id": gid, "season": 2026, "week": 7, "status": "open", "side": "home",
            "kickoff_ts": (SUNDAY + timedelta(days=6)).strftime("%Y-%m-%dT%H:%M:%SZ"), "odds": -110, "stake": 1.0,
            "line": -3.0, "placed_at": placed_at or recorded_at, "recorded_at": recorded_at}


def _lines_snapshot(cfg, ts):
    d = cfg.paths.raw_dir / "lines" / "2026" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{ts:%Y%m%dT%H%M%SZ}.json").write_text("[]")


def _odds_snapshot(cfg, ts, events):
    d = cfg.paths.raw_dir / "odds" / "2026" / "snapshots"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{ts:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(events))


def _event(eid, home, away, kick):
    return {"id": eid, "commence_time": kick.strftime("%Y-%m-%dT%H:%M:%SZ"), "home_team": home, "away_team": away,
            "bookmakers": []}


def _games(cfg, rows):
    d = cfg.paths.raw_dir / "games" / "2026"
    d.mkdir(parents=True, exist_ok=True)
    (d / "both.json").write_text(json.dumps(rows))


def _game(gid, home, away, kick, completed=False):
    return {"id": gid, "homeTeam": home, "awayTeam": away, "startDate": kick.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "completed": completed}


def _only(problems, text):
    return [p for p in problems if text in p]


def test_check_never_raises_without_files(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    cfg = _cfg(tmp_path)
    assert health.check(cfg, 2026, SUNDAY) == []           # no ledger: nothing else is checked
    _ledger(cfg, [_bet(1, SUNDAY.isoformat())])
    got = health.check(cfg, 2026, SUNDAY)
    assert "no CFBD lines snapshot this season (bets can't be priced)" in got
    assert not [p for p in got if "check failed" in p or "itself failed" in p]
    # A broken dependency surfaces as a problem string, never an exception.
    monkeypatch.setattr(health, "cfbd_snapshot_problems", lambda *a: 1 / 0)
    assert any("CFBD snapshot freshness check failed: ZeroDivisionError" in p for p in health.check(cfg, 2026, SUNDAY))
    monkeypatch.setattr(health, "_check", lambda *a: 1 / 0)
    assert health.check(cfg, 2026, SUNDAY) == ["health check itself failed: ZeroDivisionError: division by zero"]


def test_cfbd_lines_snapshot_must_be_fresh(tmp_path):
    cfg = _cfg(tmp_path)
    _ledger(cfg, [_bet(1, SUNDAY.isoformat())])
    _lines_snapshot(cfg, SUNDAY - timedelta(hours=5))
    assert _only(health.check(cfg, 2026, SUNDAY), "CFBD lines snapshot is 5 h old")
    _lines_snapshot(cfg, SUNDAY - timedelta(minutes=20))
    assert not _only(health.check(cfg, 2026, SUNDAY), "lines snapshot")


def test_discarded_odds_snapshot_is_reported(tmp_path):
    cfg = _cfg(tmp_path)
    _ledger(cfg, [_bet(1, SUNDAY.isoformat())])
    _lines_snapshot(cfg, SUNDAY - timedelta(minutes=20))
    _odds_snapshot(cfg, SUNDAY - timedelta(hours=8), [])                 # 7.7 h from the CFBD snapshot
    got = health.check(cfg, 2026, SUNDAY)
    assert _only(got, "Odds API snapshot is 7.7 h from the CFBD snapshot")
    assert _only(got, "priced from CFBD lines only")
    _odds_snapshot(cfg, SUNDAY - timedelta(minutes=10), [])
    assert not _only(health.check(cfg, 2026, SUNDAY), "from the CFBD snapshot")
    cfg.odds.enabled = False
    _odds_snapshot(cfg, SUNDAY - timedelta(hours=9), [])
    assert not _only(health.check(cfg, 2026, SUNDAY), "from the CFBD snapshot")


def test_sunday_without_new_bets_is_reported_only_when_games_are_coming(tmp_path):
    cfg = _cfg(tmp_path)
    kick = SUNDAY + timedelta(days=6)
    _ledger(cfg, [_bet(1, (SUNDAY - timedelta(days=7)).isoformat())])  # last week's bet only
    assert not _only(health.check(cfg, 2026, SUNDAY), "no bets recorded today")   # no games file: silent
    _games(cfg, [_game(1, "Tennessee", "Alabama", kick), _game(2, "Texas", "Oklahoma", SUNDAY - timedelta(days=1), True)])
    assert _only(health.check(cfg, 2026, SUNDAY), "no bets recorded today although 1 games kick off")
    # Not a Sunday: quiet. A bet recorded today: quiet. Older rows count by placed_at when recorded_at is null.
    assert not _only(health.check(cfg, 2026, SUNDAY + timedelta(days=1)), "no bets recorded today")
    _ledger(cfg, [_bet(1, SUNDAY.isoformat())])
    assert not _only(health.check(cfg, 2026, SUNDAY), "no bets recorded today")
    _ledger(cfg, [_bet(1, None, placed_at=(SUNDAY - timedelta(hours=1)).isoformat())])
    assert not _only(health.check(cfg, 2026, SUNDAY), "no bets recorded today")
    # Games file with nothing in the next 7 days (bye week / season over): quiet.
    _games(cfg, [_game(1, "Tennessee", "Alabama", SUNDAY + timedelta(days=10))])
    _ledger(cfg, [_bet(1, (SUNDAY - timedelta(days=7)).isoformat())])
    assert not _only(health.check(cfg, 2026, SUNDAY), "no bets recorded today")


def test_low_odds_api_match_rate_is_reported(tmp_path):
    cfg = _cfg(tmp_path)
    _ledger(cfg, [_bet(1, SUNDAY.isoformat())])
    kick = SUNDAY + timedelta(days=6)
    t = cfg.paths.raw_dir / "teams_fbs" / "2026"
    t.mkdir(parents=True)
    (t / "season.json").write_text(json.dumps(TEAMS))
    _games(cfg, [_game(1, "Tennessee", "Alabama", kick), _game(2, "Texas", "Oklahoma", kick)])
    # No odds snapshot yet: nothing to measure.
    assert not _only(health.check(cfg, 2026, SUNDAY), "Odds API events matched")
    events = [_event("a", "Tennessee Volunteers", "Alabama Crimson Tide", kick),
              _event("b", "Texas Longhorns", "Oklahoma Sooners", kick),
              _event("c", "Ohio State Buckeyes", "Michigan Wolverines", kick),     # not in the games file
              _event("d", "Ohio Bobcats", "Toledo Rockets", kick)]
    _odds_snapshot(cfg, SUNDAY - timedelta(minutes=5), events)
    assert _only(health.check(cfg, 2026, SUNDAY), "only 2 of 4 Odds API events matched a CFBD game")
    _games(cfg, [_game(1, "Tennessee", "Alabama", kick), _game(2, "Texas", "Oklahoma", kick),
                 _game(3, "Ohio State", "Michigan", kick), _game(4, "Ohio", "Toledo", kick)])
    assert not _only(health.check(cfg, 2026, SUNDAY), "Odds API events matched")
    # A games file the check can't parse is a problem string, not a crash.
    (cfg.paths.raw_dir / "games" / "2026" / "both.json").write_text("{not json")
    assert _only(health.check(cfg, 2026, SUNDAY), "Odds API match rate check failed")
