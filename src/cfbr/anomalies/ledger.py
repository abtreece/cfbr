"""Anthropic API spend ledger: one JSON line per call in data/anthropic_calls.jsonl, like the
CFBD quota ledger. `cfbr investigate` refuses to start once month-to-date spend (UTC calendar
month) reaches [investigate].monthly_budget_usd. It is a runaway guard; the account's monthly
credits, which hard-stop when used up, are the backstop."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

# USD per million tokens: input, output, cache write (5-minute), cache read.
PRICES = {"claude-opus-5-5": (4.0, 20.0, 5.0, 0.20)}
# Any other model (e.g. one a refusal fallback routed to) is priced at the top tier, so spend
# is over-counted rather than under-counted.
FALLBACK_PRICE = (10.0, 50.0, 12.5, 1.0)

_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


def _tokens(usage) -> list[int]:
    return [int(getattr(usage, f, None) or 0) for f in _FIELDS]


def call_cost(model: str, usage) -> float:
    prices = PRICES.get(model, FALLBACK_PRICE)
    return sum(n * p for n, p in zip(_tokens(usage), prices)) / 1e6


def record(path: Path, *, model: str, usage, purpose: str, request_id: str | None,
           now: datetime | None = None) -> float:
    cost = call_cost(model, usage)
    row = {"ts": (now or datetime.now(UTC)).isoformat(), "model": model, "purpose": purpose,
           "request_id": request_id, **dict(zip(_FIELDS, _tokens(usage))), "cost_usd": round(cost, 6)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    return cost


def month_to_date(path: Path, now: datetime | None = None) -> float:
    if not path.exists():
        return 0.0
    month = (now or datetime.now(UTC)).strftime("%Y-%m")
    total = 0.0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["ts"].startswith(month):
            total += row["cost_usd"]
    return total
