"""Verdict by closing-line value (DECISIONS 2026-10-30, "Paper-bet verdict by closing-line value").

Closing-implied ROI of a settled bet = units won per unit risked at the recorded number and price if
the pre-kickoff consensus close were the true line (``PricingTable.expected_value``, key numbers
included). The mean bettor taking the close itself earns about the vig, -4.5% at -110.
The status is recomputed every run and is provisional until the decision point.
"""

from __future__ import annotations

import math

import polars as pl

from cfbr.bets.ledger import SETTLED
from cfbr.bets.pricing import PricingTable
from cfbr.config import Bets

DECISION_POINT = ("the first weekly run after the 2027 regular season at which at least {n} settled bets have a "
                  "pre-kickoff snapshot close (else it carries to 2028)")
_Z = 1.96


def _eligible(r: dict) -> bool:
    return r["status"] in SETTLED and r["closing_source"] == "snapshot" and r["closing_line"] is not None


def closing_implied_roi(ledger: pl.DataFrame, table: PricingTable) -> pl.Series:
    """Per-row closing-implied ROI; null for rows that are unsettled or lack a pre-kickoff snapshot close."""
    values = [table.expected_value(r["side"], r["line"], int(r["odds"]), r["closing_line"]) if _eligible(r) else None
              for r in ledger.iter_rows(named=True)]
    return pl.Series("closing_implied_roi", values, dtype=pl.Float64)


def _at_close(ledger: pl.DataFrame, table: PricingTable) -> float:
    """Mean ROI of taking the close itself at the same prices: the vig the bets must beat."""
    vals = [table.expected_value(r["side"], r["closing_line"], int(r["odds"]), r["closing_line"])
            for r in ledger.iter_rows(named=True) if _eligible(r)]
    return sum(vals) / len(vals) if vals else math.nan


def verdict(ledger: pl.DataFrame, cfg: Bets, table: PricingTable | None) -> dict:
    out = {"status": "unavailable", "n": 0, "min_bets": cfg.verdict_min_bets, "mean": math.nan, "sd": math.nan,
           "ci_low": math.nan, "ci_high": math.nan, "at_close": math.nan, "decision_point": DECISION_POINT.format(n=cfg.verdict_min_bets)}
    if table is None:
        return out
    roi = closing_implied_roi(ledger, table).drop_nulls()
    n = roi.len()
    out["n"] = n
    if n == 0:
        out["status"] = "too few"
        return out
    out["mean"] = float(roi.mean())
    out["sd"] = float(roi.std()) if n > 1 else math.nan
    half = _Z * out["sd"] / math.sqrt(n)
    out["ci_low"], out["ci_high"] = out["mean"] - half, out["mean"] + half
    out["at_close"] = _at_close(ledger, table)
    if n < cfg.verdict_min_bets:
        out["status"] = "too few"
    elif out["ci_low"] > 0:
        out["status"] = "edge"
    elif out["ci_high"] < 0:
        out["status"] = "no edge"
    else:
        out["status"] = "undecided"
    return out
