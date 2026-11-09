"""The anomaly-investigation shakedown, kept as a labeled record (no API calls here).

tests/fixtures/anomaly_eval.jsonl holds one row per shakedown item (2022–2025, sampled by
`cfbr anomalies --sample-patterns`): the investigator's verdict and a label made blind to it.
These tests pin the file's shape and the go-live gate fixed before any verdict existed: at
least 90% agreement and no wrong high-confidence verdict. Re-running the shakedown means
regenerating the file; the gate test then says whether a prompt or model change still passes.

Rows in the "live" groups come from production runs after go-live. Their labels were made
after seeing the verdict, so they are kept out of the gate and only checked for wrong
high-confidence verdicts. Verdicts are recorded as the grounding rule of the day left them.
"""

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "anomaly_eval.jsonl"
VERDICTS = {"data_bug", "real", "inconclusive"}
LABELS = {"data_bug", "real", "unclear"}
IMPACTS = {"none", "small", "material", "unknown"}
CHECKS = {"A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3"}


def _rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def _is_live(row: dict) -> bool:
    return " live " in f" {row['group']} "


def _wrong_high(rows: list[dict]) -> list[str]:
    return [r["id"] for r in rows
            if r["agent"]["confidence"] == "high" and r["agent"]["verdict"] != "inconclusive"
            and r["agent"]["verdict"] != r["label"]["verdict"]]


def test_fixture_rows_are_well_formed():
    rows = _rows()
    assert len({r["id"] for r in rows}) == len(rows)
    assert sum(not _is_live(r) for r in rows) == 22
    for r in rows:
        assert r["check"] in CHECKS and r["group"].startswith(r["check"])
        assert r["label"]["verdict"] in LABELS and r["label"]["confidence"] in {"low", "medium", "high"}
        assert r["label"]["model_impact"] in IMPACTS and r["label"]["evidence"]
        assert r["agent"]["verdict"] in VERDICTS and r["agent"]["model_impact"] in IMPACTS
        assert r["agent"]["cost_usd"] >= 0 and r["agent"]["turns"] >= 1


def test_shakedown_passes_the_preregistered_gate():
    rows = [r for r in _rows() if not _is_live(r)]
    agree = sum(r["agent"]["verdict"] == r["label"]["verdict"] for r in rows)
    assert agree / len(rows) >= 0.90, f"agreement {agree}/{len(rows)}"
    assert _wrong_high(rows) == []


def test_live_rows_have_no_wrong_high_confidence_verdict():
    assert _wrong_high([r for r in _rows() if _is_live(r)]) == []
