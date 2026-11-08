import json

import duckdb

from cfbr.anomalies.prompt import TOOLS, VERDICT_SCHEMA, build_system, item_message, schema_text


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
                   "2025 week 8", "submit_verdict", "result_id", "model_plays"]:
        assert needle in system, needle


def test_item_message_has_the_record_and_context_but_no_status():
    row = {"id": "abc", "check": "B1", "season": 2026, "week": 9, "key": "Iowa", "message": "Iowa: rating -6.2",
           "value": -6.2, "threshold": 6.26, "game_id": None, "team": "Iowa", "status": "pending", "verdict": None}
    ctx = [{"check": "B2", "message": "Iowa in game 7: offense -0.60 PPA/play", "team": "Iowa", "game_id": 7}]
    msg = item_message(row, ctx)
    payload = json.loads(msg.split("\n", 1)[1].rsplit("\n\n", 1)[0])
    assert payload["anomaly"]["check"] == "B1" and "status" not in payload["anomaly"]
    assert payload["context"] == [{"check": "B2", "message": "Iowa in game 7: offense -0.60 PPA/play"}]
