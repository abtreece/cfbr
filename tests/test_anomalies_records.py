import json
import threading
from datetime import UTC, datetime, timedelta

from cfbr.anomalies.records import Anomaly, anomaly_id, initial_status, merge, read_doc, render_md, update_doc, write_doc
from cfbr.config import Config

T0 = datetime(2026, 11, 8, 15, 0, tzinfo=UTC)


def _a(check="A2", key="401", investigate=True, alert=True, msg="gap"):
    return Anomaly(check=check, season=2026, week=9, key=key, message=msg, investigate=investigate, alert=alert)


def test_config_has_anomaly_defaults():
    a = Config().anomalies
    assert a.enabled and a.a2_max_score_gap == 8 and (a.a3_min_plays, a.a3_max_plays) == (110, 250)
    assert a.a4_max_duplicates == 3 and a.b2_quantiles == (0.0025, 0.9975) and a.baseline_quantile == 0.995


def test_id_is_stable_and_key_sensitive():
    assert _a().id == anomaly_id("A2", 2026, 9, "401") == _a(msg="different text").id
    assert len(_a().id) == 12
    assert _a(key="402").id != _a().id
    assert _a(check="A3").id != _a().id


def test_initial_status():
    assert initial_status(_a(investigate=True)) == "pending"
    assert initial_status(_a(investigate=False)) == "plain"
    assert initial_status(_a(investigate=False, alert=False)) == "context"


def test_merge_first_run_adds_everything_sorted_by_severity():
    doc, added = merge(None, [_a(check="B1", key="Iowa"), _a(check="A1", key="400", investigate=False)], 2026, 9, T0)
    assert [r["check"] for r in doc["items"]] == ["A1", "B1"]
    assert {r["check"] for r in added} == {"A1", "B1"}
    assert doc["season"] == 2026 and doc["week"] == 9
    row = doc["items"][1]
    assert row["status"] == "pending" and row["verdict"] is None and row["first_seen_at"] == T0.isoformat()


def test_merge_rerun_keeps_status_and_verdict_and_adds_only_new():
    doc, _ = merge(None, [_a(key="401")], 2026, 9, T0)
    doc["items"][0]["status"] = "investigated"
    doc["items"][0]["verdict"] = {"verdict": "real"}
    doc["items"][0]["attempts"] = 2
    t1 = T0 + timedelta(hours=6)
    doc2, added = merge(doc, [_a(key="401"), _a(key="402")], 2026, 9, t1)
    by_key = {r["key"]: r for r in doc2["items"]}
    assert by_key["401"]["status"] == "investigated" and by_key["401"]["verdict"] == {"verdict": "real"}
    assert by_key["401"]["first_seen_at"] == T0.isoformat() and by_key["401"]["updated_at"] == t1.isoformat()
    assert by_key["401"]["attempts"] == 2
    assert [r["key"] for r in added] == ["402"]


def test_merge_marks_vanished_items_resolved_and_restores_on_refire():
    doc, _ = merge(None, [_a(key="401")], 2026, 9, T0)
    doc2, added = merge(doc, [], 2026, 9, T0)
    assert doc2["items"][0]["status"] == "resolved" and doc2["items"][0]["resolved_from"] == "pending"
    assert added == []
    doc3, added3 = merge(doc2, [_a(key="401")], 2026, 9, T0)
    assert doc3["items"][0]["status"] == "pending" and added3 == []


def test_merge_resolved_and_still_absent_stays_resolved_unchanged():
    doc, _ = merge(None, [_a(key="401")], 2026, 9, T0)
    doc2, _ = merge(doc, [], 2026, 9, T0)
    t2 = T0 + timedelta(hours=6)
    doc3, added = merge(doc2, [], 2026, 9, t2)
    assert added == []
    assert doc3["items"][0]["status"] == "resolved"
    assert doc3["items"][0]["updated_at"] == T0.isoformat()


def test_write_doc_is_atomic_and_round_trips(tmp_path):
    p = tmp_path / "w09" / "anomalies.json"
    doc, _ = merge(None, [_a()], 2026, 9, T0)
    write_doc(p, doc)
    assert read_doc(p) == doc
    assert not list(p.parent.glob("*.tmp"))
    assert read_doc(tmp_path / "missing.json") is None
    assert json.loads(p.read_text())["items"][0]["id"] == _a().id


def test_render_md_lists_items_and_says_when_empty():
    doc, _ = merge(None, [_a(msg="Rice @ UTSA: 38 duplicate play rows", check="A4")], 2026, 9, T0)
    md = render_md(doc)
    assert "2026 week 9" in md and "Rice @ UTSA: 38 duplicate play rows" in md and "| pending | A4 |" in md
    empty, _ = merge(None, [], 2026, 9, T0)
    assert "No anomalies." in render_md(empty)
    multi, _ = merge(None, [_a(msg="Binder Error:\n\nbad column", check="ERR")], 2026, 9, T0)
    assert "Binder Error: bad column" in render_md(multi)


def test_update_doc_serializes_concurrent_writers(tmp_path):
    p = tmp_path / "w09" / "anomalies.json"

    def bump(doc):
        doc = doc or {"n": 0}
        doc["n"] += 1
        return doc

    threads = [threading.Thread(target=lambda: [update_doc(p, bump) for _ in range(25)]) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert read_doc(p)["n"] == 100
    assert not list(p.parent.glob("*.tmp"))


def test_render_md_uses_title_when_present():
    doc, _ = merge(None, [_a()], 2026, 9, T0)
    doc["title"] = "Shakedown 2022–2025"
    assert render_md(doc).startswith("# Data anomalies — Shakedown 2022–2025")
