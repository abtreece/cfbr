import json
from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest

from cfbr.anomalies import ledger


def _u(i=0, o=0, cw=0, cr=0):
    return NS(input_tokens=i, output_tokens=o, cache_creation_input_tokens=cw, cache_read_input_tokens=cr)


def test_call_cost_opus_5_5():
    # 1M of each: 4 + 20 + 5 + 0.20
    assert ledger.call_cost("claude-opus-5-5", _u(10**6, 10**6, 10**6, 10**6)) == pytest.approx(29.2)


def test_unknown_model_is_priced_conservatively_and_none_counts_as_zero():
    u = NS(input_tokens=10**6, output_tokens=None)
    assert ledger.call_cost("some-fallback-model", u) == pytest.approx(ledger.FALLBACK_PRICE[0])


def test_record_appends_and_month_to_date_sums_only_this_month(tmp_path):
    p = tmp_path / "calls.jsonl"
    ledger.record(p, model="claude-opus-5-5", usage=_u(o=10**6), purpose="a1", request_id="r1",
                  now=datetime(2026, 10, 31, 23, 0, tzinfo=UTC))
    c = ledger.record(p, model="claude-opus-5-5", usage=_u(i=10**6), purpose="a2", request_id="r2",
                      now=datetime(2026, 11, 1, 1, 0, tzinfo=UTC))
    assert c == pytest.approx(4.0)
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert [r["purpose"] for r in rows] == ["a1", "a2"] and rows[1]["request_id"] == "r2"
    assert ledger.month_to_date(p, datetime(2026, 11, 8, tzinfo=UTC)) == pytest.approx(4.0)
    assert ledger.month_to_date(p, datetime(2026, 10, 15, tzinfo=UTC)) == pytest.approx(20.0)
    assert ledger.month_to_date(tmp_path / "missing.jsonl") == 0.0
