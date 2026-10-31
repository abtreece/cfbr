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


def test_closing_implied_roi_sign_with_real_pricing_table():
    import numpy as np

    from cfbr.bets.pricing import PricingTable
    rng = np.random.default_rng(1)
    n = 2000
    spread = rng.choice([-3.0, -3.5, -7.0, -10.0, -14.0, 3.0, 6.5], n)
    margin = (-spread + rng.normal(0, 14, n)).round()
    table = PricingTable.build(pl.DataFrame({"season": [2020] * n, "margin": margin, "spread": spread}), min_games=200)
    # both bets beat a -7 close by 3 points: home at -4, away at -10 (home perspective)
    for side, line, close in [("home", -4.0, -7.0), ("away", -10.0, -7.0)]:
        led = _ledger([_row(1, "won", "snapshot", side, line, close)])
        roi = closing_implied_roi(led, table).to_list()[0]
        assert roi > table.expected_value(side, close, -110, close)


def test_decision_point_and_empty_ledger():
    cfg = Bets(pricing_books=[], verdict_min_bets=3)
    dp = verdict(_ledger([]), cfg, None)["decision_point"]
    assert "after the 2027 regular season" in dp and "at least 3 settled" in dp and "(else it carries to 2028)" in dp
    v = verdict(_ledger([]), cfg, FakeTable())
    assert v["status"] == "too few" and v["n"] == 0
    one = verdict(_ledger([_row(1, "won", "snapshot", "home", -3.0, -5.0)]), Bets(pricing_books=[], verdict_min_bets=1), FakeTable())
    assert one["status"] == "undecided" and math.isnan(one["sd"]) and math.isnan(one["ci_low"])


def test_threshold_is_pre_registered():
    # The 300-bet threshold is part of the verdict rule pre-registered in DECISIONS.md (2026-10-30);
    # changing it breaks that pre-registration.
    from cfbr.config import load_config

    assert load_config("config/default.toml").bets.verdict_min_bets == 300
    assert load_config("config/tuned.toml").bets.verdict_min_bets == 300
