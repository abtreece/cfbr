import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
import typer

from cfbr import alerts, cli, ops
from cfbr.bets import health
from cfbr.config import Config, Paths

# A Saturday night in season.
NOW = datetime(2026, 10, 17, 23, 0, tzinfo=UTC)


def _cfg(tmp_path):
    cfg = Config(paths=Paths(raw_dir=tmp_path / "raw", db_path=tmp_path / "cfb.duckdb", results_dir=tmp_path,
                             out_dir=tmp_path / "out"))
    cfg.backup.target = str(tmp_path / "backup")
    (tmp_path / "backup").mkdir()
    (tmp_path / "backup" / "LAST_BACKUP.json").write_text(json.dumps({"finished_at": NOW.isoformat()}))
    return cfg


def _games(cfg, kickoffs):
    d = cfg.paths.raw_dir / "games" / "2026"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"id": i, "homeTeam": "H", "awayTeam": "A", "startDate": k.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
             "completed": k < NOW} for i, k in enumerate(kickoffs, 1)]
    (d / "both.json").write_text(json.dumps(rows))


def test_expected_since_is_the_last_closed_sunday_window():
    central = ops.CENTRAL
    sat = datetime(2026, 10, 17, 20, 0, tzinfo=central)          # Saturday evening
    assert ops.expected_since(sat).date() == datetime(2026, 10, 11).date()
    sun_noon = datetime(2026, 10, 18, 12, 0, tzinfo=central)     # Sunday before the 6 PM deadline
    assert ops.expected_since(sun_noon).date() == datetime(2026, 10, 11).date()
    sun_night = datetime(2026, 10, 18, 19, 0, tzinfo=central)    # Sunday after the deadline
    assert ops.expected_since(sun_night).date() == datetime(2026, 10, 18).date()


def test_season_active_needs_recent_and_upcoming_games():
    g = lambda *ks: pl.DataFrame({"kickoff_ts": list(ks)}, schema={"kickoff_ts": pl.Datetime("us", "UTC")})  # noqa: E731
    assert ops.season_active(g(NOW - timedelta(days=3), NOW + timedelta(days=4)), NOW)
    assert not ops.season_active(g(NOW - timedelta(days=30), NOW + timedelta(days=4)), NOW)   # preseason
    assert not ops.season_active(g(NOW - timedelta(days=3)), NOW)                            # season over
    assert not ops.season_active(None, NOW) and not ops.season_active(g(), NOW)


def test_heartbeat_alerts_only_in_season_and_only_when_stale(tmp_path):
    cfg = _cfg(tmp_path)
    assert ops.heartbeat_problems(cfg, 2026, NOW) == []                      # no cached games: not in season
    _games(cfg, [NOW - timedelta(days=4), NOW + timedelta(days=3)])
    assert any("recorded at all" in p for p in ops.heartbeat_problems(cfg, 2026, NOW))
    ops.record_success(cfg, 2026, 0, now=NOW - timedelta(days=2))            # Thursday: after last Sunday
    assert ops.heartbeat_problems(cfg, 2026, NOW) == []
    ops.record_success(cfg, 2026, 0, now=NOW - timedelta(days=9))            # before last Sunday
    assert any("since Sun 2026-10-11" in p for p in ops.heartbeat_problems(cfg, 2026, NOW))
    assert ops.last_success(cfg) == NOW - timedelta(days=9)
    (cfg.paths.out_dir / ops.STAMP).write_text("{broken")
    assert ops.last_success(cfg) is None
    # Out of season but backup stale: only the backup problem.
    probs = ops.daily_check(cfg, 2026, NOW + timedelta(days=40))
    assert probs and all("backup" in p for p in probs)


def test_check_command_alerts_and_exits_nonzero(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(alerts, "send", lambda cfg, title, msg, priority="high": sent.append((title, msg)) or True)
    monkeypatch.setattr(ops, "daily_check", lambda cfg, season, now=None: ["no successful `cfbr weekly` run since Sun"])
    with pytest.raises(typer.Exit) as e:
        cli.check(None)
    assert e.value.exit_code == 1 and sent[0][0] == "cfbr check found problems"
    monkeypatch.setattr(ops, "daily_check", lambda cfg, season, now=None: [])
    cli.check(None)   # quiet


def test_weekly_records_a_success_stamp(monkeypatch):
    stamped = []
    monkeypatch.setattr(cli, "_weekly_run", lambda config: None)
    monkeypatch.setattr(alerts, "send", lambda cfg, title, msg, priority="high": True)
    monkeypatch.setattr(health, "check", lambda cfg, season: [])
    monkeypatch.setattr(ops, "record_success", lambda cfg, season, code, now=None: stamped.append(code))
    cli.weekly(None)
    assert stamped == [0]
    stamped.clear()
    monkeypatch.setattr(cli, "_weekly_run", lambda config: (_ for _ in ()).throw(typer.Exit(code=4)))
    with pytest.raises(typer.Exit):
        cli.weekly(None)
    assert stamped == []   # a failed run leaves the stamp alone


def test_weekly_run_continues_on_cached_data_when_the_quota_is_hit(monkeypatch, tmp_path):
    """Exit 3 takes precedence over 5 when nothing is cached, and comes last when the run completed."""
    from cfbr.ingest import fetch
    from cfbr.ingest.client import CFBDClient, QuotaExceeded

    cfg = _cfg(tmp_path)
    monkeypatch.setattr(cli, "_default_config", lambda config: cfg)
    monkeypatch.setattr(fetch, "current_season", lambda: 2026)
    monkeypatch.setattr(CFBDClient, "get", lambda self, *a, **k: (_ for _ in ()).throw(QuotaExceeded("quota")))
    monkeypatch.setattr(CFBDClient, "calls_this_month", lambda self: 999)
    monkeypatch.setattr(fetch, "latest_cached", lambda client, endpoint, year: [])
    with pytest.raises(typer.Exit) as e:
        cli._weekly_run(None)
    assert e.value.exit_code == 3


def _status_error(code):
    import httpx

    req = httpx.Request("GET", "https://apinext.collegefootballdata.com/lines")
    return httpx.HTTPStatusError(f"{code}", request=req, response=httpx.Response(code, request=req))


def _outage_run(monkeypatch, tmp_path, errors, exit_code=5):
    """Run _weekly_run with fetch.weekly raising ``errors`` in turn, then succeeding; nothing
    completed, so a run that gets past ingest ends with exit 5. Returns the sleeps taken."""
    import time

    from cfbr.ingest import fetch
    from cfbr.ingest.client import CFBDClient

    cfg = _cfg(tmp_path)
    cfg.ingest.outage_retries = 2
    monkeypatch.setattr(cli, "_default_config", lambda config: cfg)
    monkeypatch.setattr(fetch, "current_season", lambda: 2026)
    monkeypatch.setattr(CFBDClient, "get", lambda self, *a, **k: [])
    monkeypatch.setattr(CFBDClient, "calls_this_month", lambda self: 0)
    monkeypatch.setattr(fetch, "latest_cached", lambda client, endpoint, year: [])
    pending = list(errors)

    def weekly(client, season, week):
        if pending:
            raise pending.pop(0)

    monkeypatch.setattr(fetch, "weekly", weekly)
    sleeps = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    with pytest.raises(typer.Exit) as e:
        cli._weekly_run(None)
    assert e.value.exit_code == exit_code
    return sleeps


def test_weekly_run_waits_out_a_cfbd_outage(monkeypatch, tmp_path):
    import httpx

    sleeps = _outage_run(monkeypatch, tmp_path, [_status_error(522), httpx.ConnectTimeout("timeout")])
    assert sleeps == [600, 600]


def test_weekly_run_stops_when_the_quota_runs_out_on_a_retry(monkeypatch, tmp_path):
    """Exit 3 straight from ingest (not 5 after it): the failed attempt may have cached games
    without their plays."""
    from cfbr.ingest.client import QuotaExceeded

    sleeps = _outage_run(monkeypatch, tmp_path, [_status_error(502), QuotaExceeded("quota")], exit_code=3)
    assert sleeps == [600]


def test_weekly_run_gives_up_after_the_last_retry(monkeypatch, tmp_path):
    import httpx

    with pytest.raises(httpx.HTTPStatusError):
        _outage_run(monkeypatch, tmp_path, [_status_error(502)] * 3)


def test_weekly_run_does_not_retry_a_client_error(monkeypatch, tmp_path):
    import httpx

    with pytest.raises(httpx.HTTPStatusError):
        _outage_run(monkeypatch, tmp_path, [_status_error(401)])
