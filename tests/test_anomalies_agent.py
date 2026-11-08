import json

import duckdb
import pytest

from cfbr.anomalies.agent import BETAS, grounded, investigate
from cfbr.anomalies.prompt import TOOLS, VERDICT_SCHEMA, build_system, item_message, schema_text
from cfbr.anomalies.tools import ToolSession
from cfbr.config import Investigate
from tests.anomaly_fakes import FakeAPIError, FakeClient, message, text_block, tool_use


def _strict_ok(schema):
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for sub in schema["properties"].values():
            _strict_ok(sub)
    if schema.get("type") == "array":
        _strict_ok(schema["items"])


def test_tools_are_strict_and_complete():
    assert [t["name"] for t in TOOLS] == ["query", "read_output", "submit_verdict"]
    for t in TOOLS:
        assert t["strict"] is True
        _strict_ok(t["input_schema"])
    assert VERDICT_SCHEMA["properties"]["verdict"]["enum"] == ["data_bug", "real", "inconclusive"]
    assert TOOLS[1]["input_schema"]["properties"]["name"]["enum"] == [
        "rankings.csv", "rankings_prev.csv", "predictions.csv", "accuracy.csv"]


def test_system_prompt_carries_schema_and_domain_notes(tmp_path):
    db = tmp_path / "s.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE games (game_id BIGINT, home_team VARCHAR)")
    schema = schema_text(db)
    assert schema == "games(game_id BIGINT, home_team VARCHAR)"
    system = build_system(schema)
    for needle in ["games(game_id BIGINT, home_team VARCHAR)", "play_id::DECIMAL(38,0)", "after the play",
                   "2025 week 8", "submit_verdict", "result_id", "model_plays", "a PPA value",
                   "garbage-time filter is off"]:
        assert needle in system, needle
    assert build_system(schema) == system  # byte-identical, so the prompt caches across a run


def test_item_message_has_the_record_and_context_but_no_status():
    row = {"id": "abc", "check": "B1", "season": 2026, "week": 9, "key": "Iowa", "message": "Iowa: rating -6.2",
           "value": -6.2, "threshold": 6.26, "game_id": None, "team": "Iowa", "status": "pending", "verdict": None}
    ctx = [{"check": "B2", "message": "Iowa in game 7: offense -0.60 PPA/play", "team": "Iowa", "game_id": 7}]
    msg = item_message(row, ctx)
    payload = json.loads(msg.split("\n", 1)[1].rsplit("\n\n", 1)[0])
    assert payload["anomaly"]["check"] == "B1" and "status" not in payload["anomaly"]
    assert payload["context"] == [{"check": "B2", "message": "Iowa in game 7: offense -0.60 PPA/play"}]


ROW = {"id": "abc123", "check": "A4", "season": 2026, "week": 9, "key": "1", "message": "UTSA @ Rice: 4 duplicate play rows",
       "value": 4.0, "threshold": 3.0, "game_id": 1, "team": None, "status": "pending", "verdict": None}


def _verdict(**over):
    v = {"verdict": "data_bug", "confidence": "high", "summary": "Drive 7 is listed twice.",
         "evidence": [{"claim": "four extra rows", "query_id": 1, "excerpt": "extra\n4"}],
         "affected": {"game_ids": [1], "teams": ["Rice"]}, "model_impact": "small",
         "suggested_action": "Re-fetch the game's plays."}
    v.update(over)
    return v


@pytest.fixture
def session(tmp_path):
    db = tmp_path / "s.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute("CREATE TABLE t AS SELECT 4 AS extra")
    s = ToolSession(db, tmp_path / "out", 2026, 9, Investigate())
    yield s
    s.close()


def _run(client, session, tmp_path, cfg=None, clock=None):
    kw = {"clock": clock} if clock else {}
    return investigate(client, ROW, [], session, "SYSTEM", cfg or Investigate(), tmp_path / "calls.jsonl", **kw)


def test_grounded_verdict_is_kept_and_request_shape_is_right(session, tmp_path):
    client = FakeClient([message([tool_use("query", {"sql": "SELECT extra FROM t"}, "tu1")]),
                         message([tool_use("submit_verdict", _verdict(), "tu2")])])
    out = _run(client, session, tmp_path)
    assert out["verdict"] == "data_bug" and out["reason"] is None
    assert out["meta"]["turns"] == 2 and out["meta"]["queries"] == 1 and out["meta"]["cost_usd"] > 0
    first = client.calls[0]
    assert first["model"] == "claude-opus-5-5" and first["betas"] == BETAS and first["fallbacks"] == "default"
    assert first["output_config"] == {"effort": "xhigh", "task_budget": {"type": "tokens", "total": 60000}}
    assert first["cache_control"] == {"type": "ephemeral"} and "thinking" not in first and "tool_choice" not in first
    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and second[2]["content"][0]["tool_use_id"] == "tu1"
    assert second[2]["content"][0]["is_error"] is False and "result_id 1" in second[2]["content"][0]["content"]
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 2


def test_ungrounded_excerpt_downgrades_to_inconclusive(session, tmp_path):
    bad = _verdict(evidence=[{"claim": "x", "query_id": 1, "excerpt": "38 duplicate rows"}])
    client = FakeClient([message([tool_use("query", {"sql": "SELECT extra FROM t"}, "tu1")]),
                         message([tool_use("submit_verdict", bad, "tu2")])])
    out = _run(client, session, tmp_path)
    assert (out["verdict"], out["confidence"], out["reason"]) == ("inconclusive", "low", "ungrounded")
    assert out["original_verdict"]["verdict"] == "data_bug"


def test_grounded_rules():
    results = {1: "result_id 1 (1 rows)\nextra\n4"}
    assert grounded(_verdict(), results)
    assert not grounded(_verdict(evidence=[]), results)                       # data_bug needs evidence
    assert grounded(_verdict(verdict="inconclusive", evidence=[]), results)
    assert not grounded(_verdict(evidence=[{"claim": "c", "query_id": 9, "excerpt": "4"}]), results)
    assert not grounded(_verdict(evidence=[{"claim": "c", "query_id": 1, "excerpt": "  "}]), results)
    assert grounded(_verdict(evidence=[{"claim": "c", "query_id": 1, "excerpt": "extra   4"}]), results)


def test_turn_cap_gives_inconclusive_budget(session, tmp_path):
    client = FakeClient([message([tool_use("query", {"sql": "SELECT 1"}, f"tu{i}")]) for i in range(3)])
    out = _run(client, session, tmp_path, cfg=Investigate(max_turns=3))
    assert (out["verdict"], out["reason"]) == ("inconclusive", "budget") and len(client.calls) == 3


def test_wall_clock_gives_inconclusive_budget(session, tmp_path):
    ticks = iter([0.0, 0.0, 10_000.0, 10_000.0])
    client = FakeClient([message([tool_use("query", {"sql": "SELECT 1"}, "tu1")])])
    out = _run(client, session, tmp_path, clock=lambda: next(ticks))
    assert out["reason"] == "budget" and len(client.calls) == 1


def test_refusal_is_recorded(session, tmp_path):
    from types import SimpleNamespace as NS
    client = FakeClient([message([], stop="refusal", details=NS(category="cyber"))])
    out = _run(client, session, tmp_path)
    assert (out["verdict"], out["reason"]) == ("inconclusive", "refused")
    assert out["meta"]["refusal_category"] == "cyber"


def test_text_only_turn_gets_a_nudge_then_verdict(session, tmp_path):
    client = FakeClient([message([text_block("Thinking it over.")], stop="end_turn"),
                         message([tool_use("submit_verdict", _verdict(verdict="inconclusive", evidence=[]), "tu1")])])
    out = _run(client, session, tmp_path)
    assert out["verdict"] == "inconclusive" and out["reason"] is None
    assert client.calls[1]["messages"][-1] == {"role": "user", "content": "Finish by calling submit_verdict."}


def test_query_errors_go_back_as_tool_errors_and_long_fields_are_clipped(session, tmp_path):
    long = _verdict(verdict="inconclusive", evidence=[], summary="s" * 400, suggested_action="a" * 300)
    client = FakeClient([message([tool_use("query", {"sql": "SELECT nope FROM t"}, "tu1")]),
                         message([tool_use("submit_verdict", long, "tu2")])])
    out = _run(client, session, tmp_path)
    assert client.calls[1]["messages"][2]["content"][0]["is_error"] is True
    assert len(out["summary"]) == 280 and len(out["suggested_action"]) == 200


def test_api_errors_propagate(session, tmp_path):
    with pytest.raises(FakeAPIError):
        _run(FakeClient([FakeAPIError(529, "overloaded_error")]), session, tmp_path)
