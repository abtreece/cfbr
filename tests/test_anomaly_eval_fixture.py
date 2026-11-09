"""The anomaly-investigation shakedown, kept as a labeled record (no API calls here).

tests/fixtures/anomaly_eval.jsonl holds one row per shakedown item (2022–2025, sampled by
`cfbr anomalies --sample-patterns`): the investigator's verdict and a label made blind to it.
These tests pin the file's shape and the go-live gate fixed before any verdict existed: at
least 90% agreement and no wrong high-confidence verdict. Re-running the shakedown means
regenerating the file; the gate test then says whether a prompt or model change still passes.
"""

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "anomaly_eval.jsonl"
VERDICTS = {"data_bug", "real", "inconclusive"}
LABELS = {"data_bug", "real", "unclear"}
IMPACTS = {"none", "small", "material", "unknown"}


def _rows() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]


def test_fixture_rows_are_well_formed():
    rows = _rows()
    assert len(rows) == 22 and len({r["id"] for r in rows}) == 22
    for r in rows:
        assert r["check"] in {"A2", "A3", "A4"} and r["group"].startswith(r["check"])
        assert r["label"]["verdict"] in LABELS and r["label"]["confidence"] in {"low", "medium", "high"}
        assert r["label"]["model_impact"] in IMPACTS and r["label"]["evidence"]
        assert r["agent"]["verdict"] in VERDICTS and r["agent"]["model_impact"] in IMPACTS
        assert r["agent"]["cost_usd"] >= 0 and r["agent"]["turns"] >= 1


def test_shakedown_passes_the_preregistered_gate():
    rows = _rows()
    agree = sum(r["agent"]["verdict"] == r["label"]["verdict"] for r in rows)
    wrong_high = [r["id"] for r in rows
                  if r["agent"]["confidence"] == "high" and r["agent"]["verdict"] != "inconclusive"
                  and r["agent"]["verdict"] != r["label"]["verdict"]]
    assert agree / len(rows) >= 0.90, f"agreement {agree}/{len(rows)}"
    assert wrong_high == []
