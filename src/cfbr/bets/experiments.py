"""Tracked tests on the live paper-bet ledger.

Each test splits the recorded bets into an "in" group and an "out" group by a
rule fixed before its evaluation window (see DECISIONS.md, "Tracked tests").
Nothing here changes which bets are recorded or how they're graded; the tests
only report the two groups side by side, week to week, until the decision
point. The primary metric is the gap in average closing-line value (CLV),
which is far less noisy than win rate at these sample sizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import polars as pl

from cfbr.config import Bets

DECISION_POINT = "after the 2026 postseason"
MIN_CLV_BETS = 50  # per group; below this at the decision point, the test carries over to 2027


@dataclass(frozen=True)
class Test:
    key: str
    name: str
    question: str
    in_label: str
    out_label: str
    rule: Callable[[pl.DataFrame, Bets], pl.Expr]   # (ledger with helper columns, bet rules) -> boolean "in" expression
    decision: str


def _core(d: pl.DataFrame) -> pl.Expr:
    week1 = (pl.col("season_type").fill_null("regular") == "regular") & (pl.col("week") == 1)
    return pl.col("fbs_game").fill_null(False) & (pl.col("line").abs() <= 21) & ~week1


def _second_look(d: pl.DataFrame) -> pl.Expr:
    return pl.col("_placed") > pl.col("_first_capture")


def _shortlisted(d: pl.DataFrame, cfg: Bets) -> pl.Expr:
    from cfbr.bets.shortlist import shortlist_ids  # shortlist imports this module

    return pl.col("game_id").is_in(shortlist_ids(d, cfg))


TESTS = [
    Test("core_filter", "T1 · Core filter",
         "Does the model beat the market more on competitive FBS games?",
         "FBS vs FBS, line ≤ 21, not week 1", "FCS opponent, line > 21, or week 1", lambda d, cfg: _core(d),
         "Adopt the filter as the bet rule from 2027 if the CLV gap (in − out) has a 95% CI above 0 "
         "and the in-group ROI is at least the all-bets ROI; otherwise keep betting every edge."),
    Test("second_look", "T2 · Second-look bets",
         "Are bets added after the week's first capture (a game crossing the threshold later) worse?",
         "First capture of the week", "Added at a later run", lambda d, cfg: ~_second_look(d),
         "Bet only at each week's first capture from 2027 if the CLV gap (first − later) has a 95% CI above 0; "
         "otherwise keep recording at every run."),
    Test("shortlist", "T3 · Shortlist",
         "Do the shortlisted bets (core, first capture, a reference book at the line) beat the market more than the rest?",
         "Shortlist (DraftKings, FanDuel, bet365)", "Every other paper bet", _shortlisted,
         "Keep publishing the shortlist if the CLV gap (in − out) has a 95% CI above 0; otherwise the page says "
         "it hasn't beaten the full list, and it is reconsidered. Mostly T1 ∩ T2: read it alongside them."),
]


def with_helpers(ledger: pl.DataFrame) -> pl.DataFrame:
    """Adds ``_placed`` and ``_first_capture`` (earliest placement among that week's bets; the
    postseason is one bucket, as in the week-by-week report)."""
    post = (pl.col("season_type") == "postseason").fill_null(False)
    return (ledger.with_columns(_placed=pl.col("placed_at").str.to_datetime(time_zone="UTC"),
                                _wk=pl.when(post).then(None).otherwise(pl.col("week")), _post=post)
            .with_columns(_first_capture=pl.col("_placed").min().over("season", "_post", "_wk")))


def clv_gap(a: pl.Series, b: pl.Series) -> dict:
    """Mean CLV of ``a`` minus ``b`` with a Welch 95% CI (NaN when either side has < 2 values)."""
    a, b = a.drop_nulls(), b.drop_nulls()
    if a.len() < 2 or b.len() < 2:
        return {"gap": math.nan, "ci95": (math.nan, math.nan), "n_in": a.len(), "n_out": b.len()}
    gap = float(a.mean() - b.mean())
    se = math.sqrt(float(a.var()) / a.len() + float(b.var()) / b.len())
    return {"gap": gap, "ci95": (gap - 1.96 * se, gap + 1.96 * se), "n_in": a.len(), "n_out": b.len()}


def tracked_tests(ledger: pl.DataFrame, cfg: Bets) -> list[dict]:
    from cfbr.bets.ledger import summarize

    d = with_helpers(ledger)
    out = []
    for t in TESTS:
        d2 = d.with_columns(_in=t.rule(d, cfg))
        inn, rest = d2.filter(pl.col("_in")), d2.filter(~pl.col("_in"))
        gap = clv_gap(inn["clv"], rest["clv"])
        ready = min(gap["n_in"], gap["n_out"]) >= MIN_CLV_BETS
        out.append({"key": t.key, "name": t.name, "question": t.question, "decision": t.decision,
                    "decision_point": DECISION_POINT, "min_clv_bets": MIN_CLV_BETS, "enough_data": ready,
                    "groups": [{"label": t.in_label, **summarize(inn, cfg)},
                               {"label": t.out_label, **summarize(rest, cfg)}],
                    "all_roi": summarize(ledger, cfg)["roi"], **gap})
    return out
