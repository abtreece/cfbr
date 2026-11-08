import fcntl
import json
from datetime import UTC, datetime

import duckdb
import pytest
from typer.testing import CliRunner

from cfbr.anomalies import ledger
from cfbr.anomalies.records import Anomaly, merge, read_doc, update_doc, write_doc
from cfbr.anomalies.shakedown import pattern_group, sample_patterns, shakedown_doc
from cfbr.anomalies.run import classify_error, investigate_alert, investigate_doc, plan_targets
from cfbr.config import Config, Paths, anthropic_api_key
from tests.anomaly_fakes import FakeAPIError, FakeClient, message, tool_use


def test_investigate_config_defaults():
    i = Config().investigate
    assert (i.model, i.effort, i.max_turns, i.max_items, i.max_attempts) == ("claude-opus-5-5", "xhigh", 40, 12, 3)
    assert (i.max_tokens, i.task_budget_tokens, i.monthly_budget_usd) == (32000, 60000, 50.0)
    assert (i.wall_clock_minutes, i.query_timeout_seconds, i.query_max_rows, i.query_max_chars) == (15, 30, 200, 20000)


def test_anthropic_key_env_then_keys_file_then_none(monkeypatch, tmp_path):
    monkeypatch.setattr("cfbr.config.REPO_ROOT", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert anthropic_api_key() is None
    (tmp_path / ".keys").write_text("CFBD_API_KEY=abc\nANTHROPIC_API_KEY='sk-file'\n")
    assert anthropic_api_key() == "sk-file"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert anthropic_api_key() == "sk-env"


NOW = datetime(2026, 11, 15, 16, tzinfo=UTC)
GOOD = {"verdict": "real", "confidence": "medium", "summary": "Consistent PBP.",
        "evidence": [{"claim": "play text", "query_id": 1, "excerpt": "Rush for 4 yards"}],
        "affected": {"game_ids": [1], "teams": []}, "model_impact": "none", "suggested_action": ""}


def _ok_script():
    return [message([tool_use("query", {"sql": "SELECT 'Rush for 4 yards' AS play_text"}, "q")]),
            message([tool_use("submit_verdict", GOOD, "v")])]


@pytest.fixture
def env(tmp_path):
    cfg = Config(paths=Paths(raw_dir=tmp_path / "raw", db_path=tmp_path / "data" / "cfb.duckdb",
                             results_dir=tmp_path / "res", out_dir=tmp_path / "out"))
    cfg.investigate.ledger = tmp_path / "data" / "calls.jsonl"
    cfg.investigate.snapshot = tmp_path / "data" / "inv" / "snapshot.duckdb"
    cfg.investigate.transcripts = tmp_path / "data" / "inv"
    cfg.paths.db_path.parent.mkdir(parents=True)
    with duckdb.connect(str(cfg.paths.db_path)) as con:
        con.execute("CREATE TABLE games (game_id BIGINT, home_team VARCHAR)")
    items = [Anomaly("A2", 2026, 9, "1", "Navy @ Army: gap 19", value=19.0, investigate=True, game_id=1),
             Anomaly("A3", 2026, 9, "2", "Ohio @ Iowa: 105 plays", value=105.0, investigate=True, game_id=2),
             Anomaly("A1", 2026, 9, "3", "no pbp", investigate=False, game_id=3)]
    doc, _ = merge(None, items, 2026, 9, NOW)
    path = cfg.paths.out_dir / "2026" / "w09" / "anomalies.json"
    write_doc(path, doc)
    return cfg, path


def _status(path):
    return {r["check"]: r["status"] for r in read_doc(path)["items"]}


def test_plan_targets_orders_by_severity_and_splits_overflow():
    doc = {"items": [{"id": "b", "check": "B1", "key": "x", "status": "pending"},
                     {"id": "a", "check": "A2", "key": "y", "status": "pending"},
                     {"id": "c", "check": "A1", "key": "z", "status": "plain"},
                     {"id": "d", "check": "A3", "key": "w", "status": "investigated"}]}
    todo, over = plan_targets(doc, (), 1)
    assert [r["id"] for r in todo] == ["a"] and [r["id"] for r in over] == ["b"]
    todo2, over2 = plan_targets(doc, ("d",), 12)
    assert [r["id"] for r in todo2] == ["d"] and over2 == []


def test_investigates_pending_items_and_writes_verdicts(env):
    cfg, path = env
    summary = investigate_doc(cfg, path, FakeClient(_ok_script() + _ok_script()), now=NOW)
    assert summary["status"] == "done" and len(summary["items"]) == 2 and summary["cost_usd"] > 0
    assert _status(path) == {"A2": "investigated", "A3": "investigated", "A1": "plain"}
    row = next(r for r in read_doc(path)["items"] if r["check"] == "A2")
    assert row["verdict"]["verdict"] == "real" and row["investigated_at"] == NOW.isoformat()
    assert "real" in path.with_suffix(".md").read_text()
    transcript = cfg.investigate.transcripts / "2026" / "w09" / f"{row['id']}.json"
    assert json.loads(transcript.read_text())["queries"][0]["input"] == "SELECT 'Rush for 4 yards' AS play_text"


def test_volume_overflow_is_marked_and_reported(env):
    cfg, path = env
    cfg.investigate.max_items = 1
    summary = investigate_doc(cfg, path, FakeClient(_ok_script()), now=NOW)
    assert _status(path)["A3"] == "skipped_volume" and summary["skipped_volume"] == 1


def test_budget_guard_marks_items_and_stops_before_any_call(env):
    cfg, path = env
    cfg.investigate.ledger.write_text(json.dumps({"ts": NOW.isoformat(), "cost_usd": 60.0}) + "\n")
    client = FakeClient([])
    summary = investigate_doc(cfg, path, client, now=NOW)
    assert summary["status"] == "budget" and client.calls == []
    assert _status(path)["A2"] == "skipped_budget"


def test_lock_held_skips_quietly(env):
    cfg, path = env
    with open(cfg.paths.db_path.parent / ".weekly.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = investigate_doc(cfg, path, FakeClient([]), now=NOW)
    assert summary["status"] == "locked" and _status(path)["A2"] == "pending"


def test_transient_errors_count_attempts_then_error(env):
    cfg, path = env
    cfg.investigate.max_attempts = 2
    investigate_doc(cfg, path, FakeClient([FakeAPIError(529, "overloaded_error"), FakeAPIError(500)]), now=NOW)
    row = next(r for r in read_doc(path)["items"] if r["check"] == "A2")
    assert row["status"] == "pending" and row["attempts"] == 1
    investigate_doc(cfg, path, FakeClient([FakeAPIError(529), FakeAPIError(529)]), now=NOW)
    assert _status(path) == {"A2": "error", "A3": "error", "A1": "plain"}


def test_auth_error_stops_the_run_and_leaves_items_pending(env):
    cfg, path = env
    summary = investigate_doc(cfg, path, FakeClient([FakeAPIError(401, "authentication_error")]), now=NOW)
    assert summary["stopped"] == "auth" and _status(path)["A2"] == "pending"


def test_billing_error_marks_remaining_skipped_budget(env):
    cfg, path = env
    summary = investigate_doc(cfg, path, FakeClient([FakeAPIError(402, "billing_error")]), now=NOW)
    assert summary["stopped"] == "budget" and _status(path) == {"A2": "skipped_budget", "A3": "skipped_budget", "A1": "plain"}


def test_concurrent_weekly_rerun_is_not_lost(env):
    cfg, path = env

    def rerun_then_query(kw):
        update_doc(path, lambda d: {**d, "items": d["items"] + [{"id": "new", "check": "B1", "key": "Iowa",
                                                               "status": "pending", "message": "m"}]})
        return message([tool_use("query", {"sql": "SELECT 'Rush for 4 yards' AS play_text"}, "q")])

    investigate_doc(cfg, path, FakeClient([rerun_then_query, message([tool_use("submit_verdict", GOOD, "v")])]
                                          + _ok_script()), now=NOW)
    ids = {r["id"] for r in read_doc(path)["items"]}
    assert "new" in ids and _status(path)["A2"] == "investigated"


def test_classify_error():
    assert classify_error(FakeAPIError(429, "rate_limit_error")) == "transient"
    assert classify_error(FakeAPIError(503)) == "transient"
    assert classify_error(FakeAPIError(403, "permission_error")) == "auth"
    assert classify_error(FakeAPIError(402, "billing_error")) == "budget"
    assert classify_error(FakeAPIError(400, "invalid_request_error")) == "item"
    assert classify_error(ConnectionError("reset")) == "transient"


def test_investigate_alert_text():
    s = {"status": "done", "label": "w09", "cost_usd": 0.84, "skipped_volume": 0, "skipped_budget": 0,
         "errors": 0, "stopped": None,
         "items": [{"check": "A4", "message": "Rice–UTSA dup", "verdict": "data_bug", "confidence": "high",
                    "summary": "drive 14 duplicated", "model_impact": "material", "reason": None},
                   {"check": "B1", "message": "Iowa −6.2", "verdict": "real", "confidence": "medium",
                    "summary": "lost QB", "model_impact": "small", "reason": None}]}
    title, body, priority = investigate_alert(s)
    assert title == "cfbr anomalies w09 — 1 data bug, 1 real, 0 inconclusive ($0.84)"
    assert body.splitlines()[0] == "• A4 Rice–UTSA dup: data_bug/high — drive 14 duplicated; material"
    assert priority == "high"
    assert investigate_alert({"status": "nothing"}) is None and investigate_alert({"status": "locked"}) is None


def _hold(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w")
    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fh


def test_overlapping_run_is_busy(env):
    cfg, path = env
    client = FakeClient([])
    with _hold(cfg.investigate.snapshot.parent / ".investigate.lock"):
        summary = investigate_doc(cfg, path, client, now=NOW)
    assert summary["status"] == "busy" and client.calls == []
    assert _status(path)["A2"] == "pending" and investigate_alert(summary) is None


def test_snapshot_is_replaced_atomically(env):
    cfg, path = env
    investigate_doc(cfg, path, FakeClient(_ok_script() + _ok_script()), now=NOW)
    assert cfg.investigate.snapshot.exists()
    assert list(cfg.investigate.snapshot.parent.glob("*.tmp")) == []


def test_wal_present_skips(env):
    cfg, path = env
    cfg.paths.db_path.with_name(cfg.paths.db_path.name + ".wal").write_text("")
    client = FakeClient([])
    summary = investigate_doc(cfg, path, client, now=NOW)
    assert summary["status"] == "locked" and client.calls == [] and _status(path)["A2"] == "pending"
    assert list(cfg.investigate.snapshot.parent.glob("*.tmp")) == []


def test_resolved_during_investigation_restores_investigated(env):
    cfg, path = env

    def resolve_then_query(kw):
        def apply(d):
            for r in d["items"]:
                if r["check"] == "A2":
                    r["status"] = "resolved"
            return d
        update_doc(path, apply)
        return message([tool_use("query", {"sql": "SELECT 'Rush for 4 yards' AS play_text"}, "q")])

    investigate_doc(cfg, path, FakeClient([resolve_then_query, message([tool_use("submit_verdict", GOOD, "v")])]
                                          + _ok_script()), now=NOW)
    row = next(r for r in read_doc(path)["items"] if r["check"] == "A2")
    assert row["status"] == "resolved" and row["resolved_from"] == "investigated" and row["verdict"]["verdict"] == "real"


def test_alert_cost_includes_errored_items(env):
    cfg, path = env

    def spend_then_fail(kw):
        ledger.record(cfg.investigate.ledger, model="claude-opus-5-5", usage=message([]).usage,
                      purpose="investigate", request_id="r", now=NOW)
        raise FakeAPIError(500)

    summary = investigate_doc(cfg, path, FakeClient(_ok_script() + [spend_then_fail]), now=NOW)
    total = sum(json.loads(line)["cost_usd"] for line in cfg.investigate.ledger.read_text().splitlines())
    assert len(summary["items"]) == 1 and summary["cost_usd"] == round(total, 2) and total > 0


def test_auth_alert():
    s = {"status": "done", "label": "w09", "cost_usd": 0.0, "skipped_volume": 0, "skipped_budget": 0,
         "errors": 0, "stopped": "auth", "auth_error": "FakeAPIError: HTTP 401", "items": []}
    title, body, priority = investigate_alert(s)
    assert title == "cfbr investigate: Anthropic key rejected" and priority == "high"
    assert "HTTP 401" in body


def test_doc_write_failure_is_not_an_item_error(env, monkeypatch):
    cfg, path = env
    real_patch = __import__("cfbr.anomalies.run", fromlist=["_patch"])._patch

    def failing_patch(doc_path, ids, patch):
        if patch.get("status") == "investigated":
            raise OSError("No space left on device")
        return real_patch(doc_path, ids, patch)

    monkeypatch.setattr("cfbr.anomalies.run._patch", failing_patch)
    with pytest.raises(OSError, match="No space left"):
        investigate_doc(cfg, path, FakeClient(_ok_script()), now=NOW)
    row = next(r for r in read_doc(path)["items"] if r["check"] == "A2")
    assert row["status"] == "pending" and "attempts" not in row  # not misrecorded as a transient/item error


def _h(check, value, key):
    return {"id": f"{check}{key}", "check": check, "season": 2023, "week": 5, "key": key, "message": "m",
            "value": value, "threshold": None, "investigate": True, "alert": True, "game_id": int(key), "team": None}


def test_pattern_groups():
    assert pattern_group(_h("A2", None, "1"), 110) == "A2 no scores"
    assert [pattern_group(_h("A2", v, "1"), 110) for v in (9, 17, 31)] == ["A2 gap 9-16", "A2 gap 17-30", "A2 gap >30"]
    assert [pattern_group(_h("A3", v, "1"), 110) for v in (90, 300)] == ["A3 low", "A3 high"]
    assert [pattern_group(_h("A4", v, "1"), 110) for v in (4, 11, 41)] == ["A4 4-10", "A4 11-40", "A4 >40"]
    assert pattern_group(_h("A1", None, "1"), 110) is None and pattern_group(_h("B2", 1.5, "1"), 110) is None


def test_sample_patterns_is_deterministic_and_capped():
    rows = [_h("A4", 5, str(k)) for k in range(10)] + [_h("A2", 20, "99")]
    picked = sample_patterns(rows, 3, 110)
    assert [r["group"] for r in picked].count("A4 4-10") == 3 and len(picked) == 4
    assert picked == sample_patterns(list(reversed(rows)), 3, 110)


def test_shakedown_doc_is_pending_and_titled():
    doc = shakedown_doc(sample_patterns([_h("A3", 90, "1")], 3, 110), [2022, 2023], NOW)
    assert doc["title"] == "shakedown 2022\u20132023" and doc["items"][0]["status"] == "pending"
    assert doc["items"][0]["verdict"] is None and doc["items"][0]["group"] == "A3 low"


def test_cli_investigate_without_key_exits_zero(env, monkeypatch):
    from cfbr import cli
    cfg, path = env
    monkeypatch.setattr(cli, "_default_config", lambda c: cfg)
    monkeypatch.setattr("cfbr.config.anthropic_api_key", lambda: None)
    result = CliRunner().invoke(cli.app, ["investigate", "--doc", str(path)])
    assert result.exit_code == 0 and "investigation is off" in result.output
    assert _status(path)["A2"] == "pending"


def test_cli_investigate_dry_run_lists_targets_without_a_key(env, monkeypatch):
    from cfbr import cli
    cfg, path = env
    monkeypatch.setattr(cli, "_default_config", lambda c: cfg)
    monkeypatch.setattr("cfbr.config.anthropic_api_key", lambda: None)
    result = CliRunner().invoke(cli.app, ["investigate", "--doc", str(path), "--dry-run"])
    assert result.exit_code == 0
    assert "A2 Navy @ Army: gap 19" in result.output and "A3 Ohio @ Iowa: 105 plays" in result.output
    assert "would investigate 2" in result.output


def test_load_config_resolves_investigate_paths_against_repo_root(monkeypatch, tmp_path):
    from cfbr.config import REPO_ROOT, load_config
    monkeypatch.chdir(tmp_path)
    i = load_config().investigate
    for p in (i.ledger, i.snapshot, i.transcripts):
        assert p.is_absolute() and p.is_relative_to(REPO_ROOT)


def test_cli_investigate_crash_sends_an_alert_and_fails(env, monkeypatch):
    from cfbr import cli
    cfg, path = env
    sent = []
    monkeypatch.setattr(cli, "_default_config", lambda c: cfg)
    monkeypatch.setattr("cfbr.config.anthropic_api_key", lambda: "sk-test")
    monkeypatch.setattr("cfbr.alerts.send", lambda cfg_, title, message, priority="high": sent.append(title) or True)

    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("cfbr.anomalies.run.investigate_doc", boom)
    result = CliRunner().invoke(cli.app, ["investigate", "--doc", str(path)])
    assert result.exit_code != 0
    assert sent == ["cfbr investigate crashed"]


def test_cli_sample_patterns_refuses_to_overwrite(env, monkeypatch):
    from cfbr import cli
    cfg, _ = env
    monkeypatch.setattr(cli, "_default_config", lambda c: cfg)
    out = cfg.paths.results_dir / "anomaly_shakedown" / "anomalies.json"
    out.parent.mkdir(parents=True)
    out.write_text("paid verdicts")
    result = CliRunner().invoke(cli.app, ["anomalies", "--sample-patterns", "--seasons", "2022-2022"])
    assert result.exit_code == 1 and "exists" in result.output
    assert out.read_text() == "paid verdicts"


def test_classify_error_credit_balance_400_is_budget():
    e = FakeAPIError(400, "invalid_request_error")
    e.args = ("Your credit balance is too low to access the Anthropic API.",)
    assert classify_error(e) == "budget"
