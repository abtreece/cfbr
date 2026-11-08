import fcntl
import json
from datetime import UTC, datetime

import duckdb
import pytest

from cfbr.anomalies.records import Anomaly, merge, read_doc, update_doc, write_doc
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
