"""Real-money shortlist: the paper bets worth placing at the viewer's books.

Rule fixed in DECISIONS.md ("Real-money shortlist", before any of its bets settled). A paper
bet is on the shortlist when it passes T1's core filter, was recorded at its week's first
capture (T2), and, at capture, one of the viewer's books offered its side at the recorded
consensus line or better. Only data recorded at capture is used, so membership never changes.
Nothing here touches the paper ledger, its rules or its grading.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from cfbr.bets.experiments import _core, with_helpers
from cfbr.bets.ledger import book_key
from cfbr.bets.pricing import PricingTable
from cfbr.config import Bets

# One entry of the ledger's ``books`` column: "DraftKings -7.5 (-110/-110)", home spread and
# home/away prices, optionally followed by " (offshore)" or " (pricing only)".
BOOK_RE = re.compile(r"^(?P<book>.+?) (?P<spread>[+-]\d+(?:\.\d+)?)(?: \((?P<hp>[+-]\d+)/(?P<ap>[+-]\d+)\))?(?P<rest>.*)$")


@dataclass(frozen=True)
class BookLine:
    book: str
    home_spread: float
    home_price: int | None
    away_price: int | None
    offshore: bool
    pricing_only: bool


def parse_books(books: str | None) -> list[BookLine]:
    """The per-book lines recorded with a bet (unparseable entries are skipped)."""
    out = []
    for entry in (books or "").split("; "):
        m = BOOK_RE.match(entry)
        if m is None:
            continue
        out.append(BookLine(m["book"], float(m["spread"]), int(m["hp"]) if m["hp"] else None,
                            int(m["ap"]) if m["ap"] else None, "(offshore)" in m["rest"], "(pricing only)" in m["rest"]))
    return out


def _payout(odds: int) -> float:
    return odds / 100 if odds > 0 else 100 / -odds


def bet_at(side: str, line: float, entries: list[BookLine], allowed: set[str], default_odds: int,
           table: PricingTable | None = None) -> BookLine | None:
    """The best of the allowed books' entries that offer ``side`` at ``line`` (home spread) or
    better (an empty ``allowed`` means any non-offshore book), chosen as ``ledger.market`` chooses the best line: highest expected value against
    the consensus with ``table``, else most points then best price. None if no book qualifies."""
    flip = 1 if side == "home" else -1
    price = (lambda e: e.home_price) if side == "home" else (lambda e: e.away_price)
    ok = [e for e in entries if not e.offshore and (not allowed or book_key(e.book) in allowed)
          and flip * e.home_spread >= flip * line]
    if not ok:
        return None
    odds = lambda e: price(e) if price(e) is not None else default_odds  # noqa: E731
    if table is not None:
        return max(ok, key=lambda e: (table.expected_value(side, e.home_spread, odds(e), line),
                                      flip * e.home_spread, _payout(odds(e))))
    return max(ok, key=lambda e: (flip * e.home_spread, _payout(odds(e))))


def shortlist(ledger: pl.DataFrame, books: Iterable[str], cfg: Bets,
              table: PricingTable | None = None) -> pl.DataFrame:
    """The ledger's shortlisted bets for these books, with ``bet_at`` (team perspective),
    ``bet_at_book`` and ``bet_at_odds``, ordered by kickoff."""
    allowed = {book_key(b) for b in books}
    d = with_helpers(ledger)
    d = d.filter(_core(d) & (pl.col("_placed") == pl.col("_first_capture")))
    picks = [bet_at(r["side"], r["line"], parse_books(r["books"]), allowed, cfg.odds, table)
             for r in d.select("side", "line", "books").iter_rows(named=True)]
    flips = [1 if s == "home" else -1 for s in d["side"].to_list()]
    side_price = [None if p is None else (p.home_price if f == 1 else p.away_price) for p, f in zip(picks, flips)]
    d = d.with_columns(
        bet_at=pl.Series([None if p is None else f * p.home_spread for p, f in zip(picks, flips)], dtype=pl.Float64),
        bet_at_book=pl.Series([None if p is None else p.book for p in picks], dtype=pl.Utf8),
        bet_at_odds=pl.Series(side_price, dtype=pl.Int32),
    )
    return (d.filter(pl.col("bet_at_book").is_not_null())
            .drop([c for c in d.columns if c.startswith("_")])
            .sort("kickoff_ts", "game_id"))


def shortlist_ids(ledger: pl.DataFrame, cfg: Bets) -> list[int]:
    """Game ids on the shortlist under the reference books (``pricing_books``): T3's in-group."""
    return shortlist(ledger, cfg.pricing_books, cfg)["game_id"].to_list()
