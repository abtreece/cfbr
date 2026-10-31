import math

import polars as pl

from cfbr.bets.ledger import LEDGER_SCHEMA
from cfbr.bets.verdict import closing_implied_roi, verdict
from cfbr.config import Bets


class FakeTable:
    """expected_value from the point gap only: +2% per point the number beats the center, minus 4.5% vig."""

    def expected_value(self, side, number, odds, center):
        gap = (number - center) if side == "home" else (center - number)
        return -0.045 + 0.02 * gap


def _row(i, status, source, side, line, close):
    return {**{k: None for k in LEDGER_SCHEMA}, "game_id": i, "season": 2026, "week": 7, "side": side, "team": "T",
            "line": line, "odds": -110, "stake": 1.0, "status": status, "closing_source": source, "closing_line": close}


def _ledger(rows):
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def test_closing_implied_roi_only_for_settled_snapshot_closes():
    led = _ledger([_row(1, "won", "snapshot", "home", -3.0, -5.0),     # home at -3, close -5: 2 pts of CLV
                   _row(2, "lost", "cfbd", "home", -3.0, -5.0),         # fallback close: excluded
                   _row(3, "open", "snapshot", "away", 3.0, 3.0)])       # not settled: excluded
    r = closing_implied_roi(led, FakeTable())
    assert r.to_list()[0] == -0.045 + 0.04 and r.to_list()[1] is None and r.to_list()[2] is None


def test_verdict_statuses():
    cfg = Bets(pricing_books=[], verdict_min_bets=3)
    assert verdict(_ledger([]), cfg, None)["status"] == "unavailable"
    few = _ledger([_row(1, "won", "snapshot", "home", -3.0, -5.0)])
    assert verdict(few, cfg, FakeTable())["status"] == "too few"
    edge = _ledger([_row(i, "won", "snapshot", "home", -3.0, -7.0) for i in range(4)])          # +3.5% each
    v = verdict(edge, cfg, FakeTable())
    assert v["status"] == "edge" and math.isclose(v["mean"], 0.035) and v["n"] == 4 and math.isclose(v["at_close"], -0.045)
    none = _ledger([_row(i, "lost", "snapshot", "home", -3.0, -3.0) for i in range(4)])          # -4.5% each
    assert verdict(none, cfg, FakeTable())["status"] == "no edge"
    mixed = _ledger([_row(1, "won", "snapshot", "home", -3.0, -7.0), _row(2, "lost", "snapshot", "home", -3.0, -3.0),
                     _row(3, "won", "snapshot", "home", -3.0, -5.0), _row(4, "lost", "snapshot", "home", -3.0, -3.5)])
    assert verdict(mixed, cfg, FakeTable())["status"] == "undecided"
