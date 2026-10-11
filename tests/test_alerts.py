import json
from datetime import UTC, datetime, timedelta

import duckdb
import polars as pl
import pytest
import typer

from cfbr import alerts, cli
from cfbr.bets import health
from cfbr.bets.ledger import LEDGER_SCHEMA
from cfbr.config import Config, Paths

NOW = datetime(2026, 10, 18, 14, 30, tzinfo=UTC)


def _cfg(tmp_path):
    cfg = Config(paths=Paths(raw_dir=tmp_path / "raw", db_path=tmp_path / "cfb.duckdb", results_dir=tmp_path,
                             out_dir=tmp_path / "out"))
    cfg.odds.enabled = cfg.oddspapi.enabled = False
    cfg.backup.target = str(tmp_path / "backup")
    (tmp_path / "backup").mkdir()
    (tmp_path / "backup" / "LAST_BACKUP.json").write_text(json.dumps({"finished_at": NOW.isoformat()}))
    return cfg


def test_send_posts_to_ntfy_only_with_a_topic(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    posts = []
    monkeypatch.setattr(alerts.httpx, "post", lambda url, **k: posts.append((url, k)) or
                        type("R", (), {"raise_for_status": lambda self: None})())
    monkeypatch.setattr(alerts, "ntfy_topic", lambda: None)
    assert alerts.send(cfg, "t", "m") is False and posts == []
    monkeypatch.setattr(alerts, "ntfy_topic", lambda: "cfbr-secret")
    assert alerts.send(cfg, "cfbr weekly failed", "boom") is True
    url, kw = posts[0]
    assert url == "https://ntfy.sh/cfbr-secret" and kw["headers"]["Title"] == "cfbr weekly failed" and kw["content"] == b"boom"
    monkeypatch.setattr(alerts.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert alerts.send(cfg, "t", "m") is False  # never raises


def _ledger(cfg, rows):
    d = cfg.paths.out_dir / "2026"
    d.mkdir(parents=True)
    pl.DataFrame(rows, schema=LEDGER_SCHEMA).write_csv(d / "bets.csv")


def _bet(gid, status="open", kickoff=NOW - timedelta(days=1), units=None):
    return {"game_id": gid, "season": 2026, "week": 7, "status": status, "units": units, "side": "home",
            "kickoff_ts": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"), "odds": -110, "stake": 1.0, "line": -3.0}


def test_health_check_flags_settlement_and_ledger_problems(tmp_path):
    cfg = _cfg(tmp_path)
    assert health.check(cfg, 2026, NOW) == []  # no ledger yet
    _ledger(cfg, [_bet(1), _bet(2, "won", units=1.0), _bet(3, "lost"), _bet(4, kickoff=NOW - timedelta(days=6)),
                  _bet(5), _bet(5)])
    with duckdb.connect(str(cfg.paths.db_path)) as con:
        con.execute("CREATE TABLE games AS SELECT * FROM (VALUES (1, 2026, true), (5, 2026, false)) t(game_id, season, completed)")
    got = health.check(cfg, 2026, NOW)
    assert "1 games bet twice in the ledger" in got
    assert "1 settled bets have no units" in got
    assert "1 bets on finished games are still open (settlement failed?)" in got
    assert "1 open bets kicked off over 4 days ago" in got


@pytest.mark.parametrize("outcome,alerted,checked", [
    (None, [], True),                                   # clean run: check only
    (typer.Exit(code=6), ["cfbr weekly failed (exit 6)"], True),
    (typer.Exit(code=5), [], False),                    # preseason: quiet
    (typer.Exit(code=4), ["cfbr weekly failed (exit 4)"], False),
    (RuntimeError("disk full"), ["cfbr weekly crashed"], False),
])
def test_weekly_alerts_on_failure_and_runs_the_check(monkeypatch, outcome, alerted, checked):
    sent, ran = [], []
    def run(config):
        if outcome is not None:
            raise outcome
    monkeypatch.setattr(cli, "_weekly_run", run)
    monkeypatch.setattr(alerts, "send", lambda cfg, title, msg, priority="high":
                        (priority != "min" and sent.append(title)) or True)
    monkeypatch.setattr(health, "check", lambda cfg, season: ran.append(season) or [])
    if outcome is None:
        cli.weekly(None)
    else:
        with pytest.raises(type(outcome)):
            cli.weekly(None)
    assert sent == alerted and bool(ran) == checked


def test_weekly_pushes_check_problems(monkeypatch):
    sent = []
    monkeypatch.setattr(cli, "_weekly_run", lambda config: None)
    monkeypatch.setattr(alerts, "send", lambda cfg, title, msg, priority="high":
                        (priority != "min" and sent.append((title, msg))) or True)
    monkeypatch.setattr(health, "check", lambda cfg, season: ["2 bets on finished games are still open"])
    cli.weekly(None)
    assert sent == [("cfbr weekly: check found problems", "2 bets on finished games are still open")]


def test_weekly_success_ping_carries_the_ledger_chain_head(monkeypatch, tmp_path):
    from cfbr import backup

    sent = []
    monkeypatch.setattr(cli, "_weekly_run", lambda config: None)
    monkeypatch.setattr(alerts, "send", lambda cfg, title, msg, priority="high": sent.append((title, msg, priority)))
    monkeypatch.setattr(health, "check", lambda cfg, season: [])
    monkeypatch.setattr(backup, "chain_head", lambda out_dir, season: "ledger chain 2026: 3 entries, head abc")
    cli.weekly(None)
    assert sent == [("cfbr weekly ok", "ledger chain 2026: 3 entries, head abc", "min")]


def test_health_check_flags_a_missing_or_stale_backup(tmp_path):
    cfg = _cfg(tmp_path)
    assert health.check(cfg, 2026, NOW + timedelta(days=7)) == []
    assert any("days old" in p for p in health.check(cfg, 2026, NOW + timedelta(days=9)))
    cfg.backup.target = str(tmp_path / "nowhere")
    assert any("no backup found" in p for p in health.check(cfg, 2026, NOW))
