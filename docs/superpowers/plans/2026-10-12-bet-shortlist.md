# Real-money shortlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show premium viewers a short list of the week's paper bets worth placing for real money at *their* sportsbooks, and track that list's record as tracked test T3.

**Architecture:** A pure function in `bets/shortlist.py` derives the list from the paper ledger (the T1 core filter, first capture, and the viewer's books' numbers at capture). `bets/experiments.py` tracks it as T3 under the reference book set. `web/access.py:viewer_books` is the single place that knows the book choice comes from a cookie. `web/bets.py` adds `/bets/shortlist`, a POST to set books, and `/api/bets/shortlist`, all behind `bets_access`.

**Tech Stack:** Python 3.12, polars, FastAPI, Jinja2, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-12-bet-shortlist-design.md`. **Rule of record:** DECISIONS.md "Real-money shortlist", "T3 shortlist", "Shortlist book choice" (commit c4d5520).

## Global Constraints

- The paper ledger, its rules, and T1/T2 membership are never changed. The shortlist only reads `bets.csv`.
- Membership uses only capture-time data: `fbs_game`, `line`, `week`, `season_type`, `placed_at`, `books`.
- Ledger `line` is the **home spread**. `books` is `"Book ±spread (hp/ap)[ (offshore)| (pricing only)]"` entries joined by `"; "`, with home spread and home/away prices.
- Offshore entries never qualify a bet. Edge size, tier, and cover probability never order or filter picks. Picks are ordered by kickoff.
- Reference book set = `bets.pricing_books` (DraftKings, FanDuel, bet365). No choice means the reference set.
- Ledger data appears only in `web/bets.py` routes, behind `bets_access`, and is sent with `Cache-Control: private, no-store`.
- The server never fits a model and writes nothing to disk.
- Code style: 2-space indentation is the user default, **but this repo uses 4-space Python**, so match the repo. Use polars and the stdlib form parsing (`parse_qs`), and add no new dependencies.
- Commits: conventional (`feat(bets): …`, `feat(web): …`), one logical change each, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run tests with `uv run pytest …` from the worktree root.

---

## File Structure

- **Create** `src/cfbr/bets/shortlist.py`: `BookLine`, `parse_books`, `BOOK_RE`, `shortlist`, `shortlist_ids`. Pure, with no web imports.
- **Create** `tests/test_shortlist.py`: unit tests for the module above.
- **Modify** `src/cfbr/bets/experiments.py`: `Test.rule` takes `(d, cfg)`; add T3.
- **Modify** `src/cfbr/config.py` and `config/default.toml`: add `bets.selectable_books`.
- **Modify** `src/cfbr/web/access.py`: `BOOKS_COOKIE`, `book_cookie_value`, `choose_books`, `viewer_books`.
- **Modify** `src/cfbr/web/store.py`: `pricing_table()`.
- **Modify** `src/cfbr/web/bets.py`: import `BOOK_RE` (drop the local `_BOOK`); shortlist routes.
- **Create** `src/cfbr/web/templates/bets_shortlist.html`.
- **Modify** `src/cfbr/web/templates/bets.html` and `bets_retro.html`: tabs Live | Shortlist | Replay.
- **Modify** `tests/test_bets.py` and `tests/test_web.py`.
- **Modify** `README.md`: the Bets bullet mentions the Shortlist tab.

---

### Task 1: `bets/shortlist.py` — parse capture books and select the shortlist

**Files:**
- Create: `src/cfbr/bets/shortlist.py`
- Create: `tests/test_shortlist.py`
- Modify: `src/cfbr/web/bets.py:69` (replace the local `_BOOK` regex with an import)

**Interfaces:**
- Consumes: `cfbr.bets.experiments._core(d) -> pl.Expr` and `with_helpers(ledger) -> pl.DataFrame` (adds `_placed`, `_first_capture`), `cfbr.bets.ledger.book_key(name) -> str`, `cfbr.bets.pricing.PricingTable.expected_value(side, number, odds, center) -> float`, `cfbr.config.Bets`.
- Produces:
  - `BOOK_RE: re.Pattern` (groups `book`, `spread`, `hp`, `ap`, `rest`)
  - `BookLine(book: str, home_spread: float, home_price: int | None, away_price: int | None, offshore: bool, pricing_only: bool)`
  - `parse_books(books: str | None) -> list[BookLine]`
  - `shortlist(ledger: pl.DataFrame, books: Iterable[str], cfg: Bets, table: PricingTable | None = None) -> pl.DataFrame`: the ledger's columns plus `bet_at: Float64` (team perspective), `bet_at_book: Utf8`, `bet_at_odds: Int32` (null when the book posted no price), sorted by `kickoff_ts, game_id`.
  - `shortlist_ids(ledger: pl.DataFrame, cfg: Bets) -> list[int]`: game ids on the shortlist under `cfg.pricing_books`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_shortlist.py`:

```python
import polars as pl
import pytest

from cfbr.bets.ledger import LEDGER_SCHEMA
from cfbr.bets.shortlist import BookLine, parse_books, shortlist, shortlist_ids
from cfbr.config import Bets

CFG = Bets()
FIRST, LATER = "2026-10-11T14:05:00+00:00", "2026-10-11T20:05:00+00:00"
REF = frozenset(CFG.pricing_books)


def _row(gid, side="home", line=-7.0, books="DraftKings -7 (-110/-110)", week=7, fbs=True, placed=FIRST,
         season_type="regular", kickoff=None):
    return {"game_id": gid, "season": 2026, "week": week, "season_type": season_type, "side": side,
            "team": f"H{gid}" if side == "home" else f"A{gid}", "home_team": f"H{gid}", "away_team": f"A{gid}",
            "line": line, "edge": 4.0, "model_margin": 11.0, "odds": -110, "stake": 1.0, "status": "open",
            "fbs_game": fbs, "placed_at": placed, "books": books,
            "kickoff_ts": kickoff or f"2026-10-{10 + gid:02d}T23:00:00Z"}


def _ledger(rows):
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def test_parse_books_reads_spread_prices_and_markers():
    got = parse_books("Bovada +3.5 (-115/-105) (offshore); DraftKings +3 (-110/-110); "
                      "bet365 +3.5 (-105/-115) (pricing only); BetMGM +3; garbage")
    assert got == [
        BookLine("Bovada", 3.5, -115, -105, True, False),
        BookLine("DraftKings", 3.0, -110, -110, False, False),
        BookLine("bet365", 3.5, -105, -115, False, True),
        BookLine("BetMGM", 3.0, None, None, False, False),
    ]
    assert parse_books(None) == [] and parse_books("") == []


def test_core_rule():
    d = _ledger([_row(1), _row(2, line=-24.5), _row(3, fbs=False), _row(4, fbs=None), _row(5, week=1),
                 _row(6, week=1, season_type="postseason")])
    assert shortlist(d, REF, CFG)["game_id"].to_list() == [1, 6]  # postseason "week 1" is not week 1


def test_first_capture_rule():
    d = _ledger([_row(1), _row(2, placed=LATER), _row(3, week=8, placed=LATER)])
    # Game 2 came after week 7's first capture; game 3 is week 8's only (so first) capture.
    assert shortlist(d, REF, CFG)["game_id"].to_list() == [1, 3]


def test_your_number_or_better_from_each_side():
    d = _ledger([
        _row(1, "home", -7.0, "DraftKings -7.5 (-110/-110)"),   # home wants a higher spread: -7.5 is worse
        _row(2, "home", -7.0, "DraftKings -6.5 (-110/-110)"),   # better
        _row(3, "away", 7.0, "DraftKings +7.5 (-110/-110)"),    # away wants a lower spread: +7.5 is worse
        _row(4, "away", 7.0, "DraftKings +6.5 (-110/-110)"),    # better: away lays 6.5 instead of 7
    ])
    s = shortlist(d, REF, CFG)
    assert s["game_id"].to_list() == [2, 4]
    assert s["bet_at"].to_list() == [6.5, -6.5] and s["bet_at_book"].to_list() == ["DraftKings"] * 2


def test_only_the_viewers_books_and_never_offshore():
    books = "Bovada -6 (-110/-110) (offshore); BetMGM -6.5 (-110/-110); DraftKings -7.5 (-110/-110)"
    d = _ledger([_row(1, "home", -7.0, books), _row(2, "home", -7.0, None)])
    assert shortlist(d, REF, CFG).is_empty()                       # DK is short; Bovada never counts; null books out
    s = shortlist(d, {"BetMGM"}, CFG)
    assert s["game_id"].to_list() == [1] and s["bet_at_book"].to_list() == ["BetMGM"]
    assert shortlist(d, {"Bovada"}, CFG).is_empty()


def test_bet365_pricing_only_entries_count_and_names_match_by_key():
    d = _ledger([_row(1, "home", -7.0, "DraftKings -7.5 (-110/-110); bet365 -7 (-105/-115) (pricing only)")])
    s = shortlist(d, {"Bet365"}, CFG)
    assert s["bet_at_book"].to_list() == ["bet365"] and s["bet_at_odds"].to_list() == [-105]


def test_bet_at_prefers_points_then_price_without_a_table():
    books = "DraftKings -7 (-115/-105); FanDuel -7 (-105/-115); BetMGM -6.5 (-125/+105)"
    d = _ledger([_row(1, "home", -7.0, books)])
    s = shortlist(d, {"DraftKings", "FanDuel"}, CFG)
    assert (s["bet_at"][0], s["bet_at_book"][0], s["bet_at_odds"][0]) == (-7.0, "FanDuel", -105)
    s = shortlist(d, {"DraftKings", "FanDuel", "BetMGM"}, CFG)
    assert (s["bet_at"][0], s["bet_at_book"][0]) == (-6.5, "BetMGM")


def test_unpriced_entries_count_as_the_default_odds():
    d = _ledger([_row(1, "home", -7.0, "DraftKings -7; FanDuel -7 (-112/-108)")])
    s = shortlist(d, {"DraftKings", "FanDuel"}, CFG)
    assert s["bet_at_book"].to_list() == ["DraftKings"] and s["bet_at_odds"].to_list() == [None]


def test_bet_at_uses_expected_value_with_a_table():
    class Table:  # half a point is worth less than 20 cents here
        def expected_value(self, side, number, odds, center):
            payout = odds / 100 if odds > 0 else 100 / -odds
            return 0.5 * payout - 0.5 + (number - center) * 0.01

    books = "DraftKings -6.5 (-130/+110); FanDuel -7 (-105/-115)"
    d = _ledger([_row(1, "home", -7.0, books)])
    assert shortlist(d, {"DraftKings", "FanDuel"}, CFG)["bet_at_book"].to_list() == ["DraftKings"]
    assert shortlist(d, {"DraftKings", "FanDuel"}, CFG, Table())["bet_at_book"].to_list() == ["FanDuel"]


def test_shortlist_ids_use_the_reference_books():
    d = _ledger([_row(1), _row(2, books="BetMGM -6 (-110/-110)"), _row(3, line=-30.0)])
    assert shortlist_ids(d, CFG) == [1]


def test_sorted_by_kickoff_and_keeps_ledger_columns():
    d = _ledger([_row(1, kickoff="2026-10-18T00:00:00Z"), _row(2, kickoff="2026-10-17T16:00:00Z")])
    s = shortlist(d, REF, CFG)
    assert s["game_id"].to_list() == [2, 1]
    assert set(LEDGER_SCHEMA) <= set(s.columns) and not any(c.startswith("_") for c in s.columns)
```

Sign conventions: `line` and `books` are home spreads, and `bet_at` is the bet team's number. An away bet at home spread +7 has the away team at −7, so a book at home +6.5 (away −6.5) is better for it. A home bet at −7 is better off at −6.5.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_shortlist.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'cfbr.bets.shortlist'`.

- [ ] **Step 3: Implement `src/cfbr/bets/shortlist.py`**

```python
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
    better, chosen as ``ledger.market`` chooses the best line: highest expected value against
    the consensus with ``table``, else most points then best price. None if no book qualifies."""
    flip = 1 if side == "home" else -1
    price = (lambda e: e.home_price) if side == "home" else (lambda e: e.away_price)
    ok = [e for e in entries if not e.offshore and book_key(e.book) in allowed
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
```

- [ ] **Step 4: Point `web/bets.py` at the shared regex**

In `src/cfbr/web/bets.py`, delete the line

```python
_BOOK = re.compile(r"^(?P<book>.+?) (?P<spread>[+-]\d+(?:\.\d+)?)(?: \((?P<hp>[+-]\d+)/(?P<ap>[+-]\d+)\))?(?P<rest>.*)$")
```

add `from cfbr.bets.shortlist import BOOK_RE` to the imports, and in `books_from_side` change `m = _BOOK.match(entry)` to `m = BOOK_RE.match(entry)`. If `re` is no longer used in the file (`grep -n 're\.' src/cfbr/web/bets.py`), remove `import re`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_shortlist.py tests/test_web.py -q`
Expected: all pass. Dataset-backed web tests may show as skipped if there's no `data/cfb.duckdb` in the worktree, which is fine.

- [ ] **Step 6: Commit**

```bash
git add src/cfbr/bets/shortlist.py tests/test_shortlist.py src/cfbr/web/bets.py
git commit -m "feat(bets): select the real-money shortlist from the paper ledger

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Track the shortlist as T3

**Files:**
- Modify: `src/cfbr/bets/experiments.py` (`Test.rule` signature, `TESTS`, `tracked_tests`)
- Modify: `tests/test_bets.py:239-268` (`test_tracked_tests_split_the_ledger_by_fixed_rules`)
- Modify: `tests/test_web.py:506`

**Interfaces:**
- Consumes: `cfbr.bets.shortlist.shortlist_ids(ledger, cfg) -> list[int]` (Task 1). It's imported inside the rule function, because `shortlist.py` imports `experiments.py`.
- Produces: `tracked_tests(ledger, cfg)` returns three dicts, with keys `core_filter`, `second_look`, `shortlist`. The `shortlist` entry is what the web page shows (Task 4).

- [ ] **Step 1: Update the failing test**

In `tests/test_bets.py`, inside `test_tracked_tests_split_the_ledger_by_fixed_rules`:

1. Change `def row(gid, week, line, fbs, placed, clv_, status="won", season_type="regular"):` so it also sets `books`:

```python
    def row(gid, week, line, fbs, placed, clv_, status="won", season_type="regular", books=None):
        r = _settled(gid, week, 4, status, season_type)
        return r | {"line": line, "fbs_game": fbs, "placed_at": placed, "clv": clv_,
                    "books": books if books is not None else f"DraftKings {line:+g} (-110/-110)"}
```

2. Change the first row to have a game that passes T1 and T2 but whose reference books are short (so T3 differs from T1 ∩ T2):

```python
    d = _ledger([
        row(1, 7, -7.0, True, early, 1.0),                  # core, first capture, DK at the line: shortlist
        row(2, 7, -24.5, True, early, -0.5, "lost"),        # line > 21: out
        row(3, 7, -3.0, False, late, 0.0, "lost"),          # FCS opponent: out; added at a later run
        row(4, 1, -3.0, True, early, 0.5),                  # week 1: out
        row(5, 1, -3.0, True, early, 0.5, "won", "postseason", "BetMGM -3 (-110/-110)"),  # core + first, but no reference book
        row(6, 8, 2.5, None, late, 2.0),                    # unknown FBS status: out; first capture of week 8
    ])
    t1, t2, t3 = tracked_tests(d, CFG)
```

3. After the existing T2 assertion, add:

```python
    # Shortlist: core + first capture + a reference book at the line. Game 5 has only BetMGM.
    assert t3["key"] == "shortlist" and [g["settled"] for g in t3["groups"]] == [1, 5]
    assert t3["n_in"] == 1 and math.isnan(t3["gap"])
    assert "### T3 · Shortlist" in md
```

(Place the T3 assertion line after `md = summary_markdown(...)`.)

In `tests/test_web.py:506` change the expected keys to `["core_filter", "second_look", "shortlist"]`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_bets.py -k tracked_tests -q`
Expected: FAIL with `ValueError: not enough values to unpack (expected 3, got 2)`.

- [ ] **Step 3: Implement T3**

In `src/cfbr/bets/experiments.py`:

Change the `rule` field and its comment:

```python
    rule: Callable[[pl.DataFrame, Bets], pl.Expr]   # (ledger with helper columns, bet rules) -> boolean "in" expression
```

Add after `_second_look`:

```python
def _shortlisted(d: pl.DataFrame, cfg: Bets) -> pl.Expr:
    from cfbr.bets.shortlist import shortlist_ids  # shortlist imports this module

    return pl.col("game_id").is_in(shortlist_ids(d, cfg))
```

Replace `TESTS` with:

```python
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
```

In `tracked_tests`, change `d2 = d.with_columns(_in=t.rule(d))` to `d2 = d.with_columns(_in=t.rule(d, cfg))`.

Check nothing else calls `.rule(`: `grep -rn '\.rule(' src tests`.

`shortlist_ids` receives `d` (already with helper columns). `with_helpers` recomputes `_placed`/`_first_capture` with `with_columns`, which overwrites them, so this is safe.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_bets.py tests/test_shortlist.py tests/test_web.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/bets/experiments.py tests/test_bets.py tests/test_web.py
git commit -m "feat(bets): track the shortlist as T3

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Book choice — config and the `viewer_books` seam

**Files:**
- Modify: `src/cfbr/config.py:92` (after `pricing_books`)
- Modify: `config/default.toml` (`[bets]`, after `pricing_books`)
- Modify: `src/cfbr/web/access.py` (append a section)
- Test: `tests/test_web.py` (append)

**Interfaces:**
- Consumes: `cfbr.bets.ledger.book_key`, `cfbr.config.Bets`.
- Produces (used in Task 4):
  - `Bets.selectable_books: list[str]`
  - `BOOKS_COOKIE = "cfbr_books"`, `BOOKS_COOKIE_DAYS = 365`
  - `choose_books(names: Iterable[str], bets: Bets) -> frozenset[str]`: canonical names from `selectable_books`; the reference set if none are valid.
  - `book_cookie_value(books: Iterable[str]) -> str`: book keys joined with `.` (cookie-safe: alphanumerics and dots only).
  - `viewer_books(request: Request, bets: Bets) -> frozenset[str]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web.py`:

```python
# -- shortlist book choice ----------------------------------------------------------------

def test_choose_books_validates_against_the_selectable_list():
    from cfbr.config import Bets
    from cfbr.web.access import book_cookie_value, choose_books

    b = Bets()
    assert choose_books(["draftkings", "Bovada", "nope", " FanDuel "], b) == {"DraftKings", "FanDuel"}
    assert choose_books([], b) == choose_books(["Bovada"], b) == frozenset(b.pricing_books)  # offshore isn't selectable
    assert "Bovada" not in b.selectable_books and set(b.pricing_books) <= set(b.selectable_books)
    v = book_cookie_value({"Bally Bet", "DraftKings"})
    assert v == "ballybet.draftkings" and choose_books(v.split("."), b) == {"Bally Bet", "DraftKings"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_web.py -k choose_books -q`
Expected: FAIL with `ImportError: cannot import name 'book_cookie_value'`.

- [ ] **Step 3: Implement**

In `src/cfbr/config.py`, after the `pricing_books` line in `class Bets`:

```python
    # Books a viewer may pick for the real-money shortlist (never offshore; DECISIONS
    # "Shortlist book choice"). No choice = pricing_books, the set T3 tracks.
    selectable_books: list[str] = ["DraftKings", "FanDuel", "bet365", "BetMGM", "BetRivers", "Bally Bet", "betPARX",
                                   "theScore Bet"]
```

In `config/default.toml`, after the `pricing_books = [...]` line:

```toml
# Books a viewer may pick for the real-money shortlist; never offshore. No choice = pricing_books.
selectable_books = ["DraftKings", "FanDuel", "bet365", "BetMGM", "BetRivers", "Bally Bet", "betPARX", "theScore Bet"]
```

Append to `src/cfbr/web/access.py`, with `from collections.abc import Iterable`, `from cfbr.bets.ledger import book_key` and `from cfbr.config import Bets` added to the imports:

```python
# -- the viewer's sportsbooks for the real-money shortlist --------------------------------
# The site has no accounts, so the choice lives in a cookie. It is a preference, not a
# credential. viewer_books is the only code that knows that; with accounts, only it changes
# (DECISIONS "Shortlist book choice").

BOOKS_COOKIE = "cfbr_books"
BOOKS_COOKIE_DAYS = 365


def choose_books(names: Iterable[str], bets: Bets) -> frozenset[str]:
    """The selectable books among ``names`` (any spelling ``book_key`` matches), spelled as in
    config; the reference set (``pricing_books``) when none are."""
    by_key = {book_key(b): b for b in bets.selectable_books}
    chosen = frozenset(by_key[k] for n in names if (k := book_key(n.strip())) in by_key)
    return chosen or frozenset(bets.pricing_books)


def book_cookie_value(books: Iterable[str]) -> str:
    """Cookie-safe encoding of a book choice: the books' keys, sorted, joined with dots."""
    return ".".join(sorted(book_key(b) for b in books))


def viewer_books(request: Request, bets: Bets) -> frozenset[str]:
    """This viewer's sportsbooks (from the cookie; the reference set when unset or invalid)."""
    return choose_books((request.cookies.get(BOOKS_COOKIE) or "").split("."), bets)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_web.py -k "choose_books or bets" -q`
Expected: all pass. (`body["rules"]` in `test_bets_api_matches_ledger_summary` now also carries `selectable_books`, which is harmless.)

- [ ] **Step 5: Commit**

```bash
git add src/cfbr/config.py config/default.toml src/cfbr/web/access.py tests/test_web.py
git commit -m "feat(web): let viewers choose their sportsbooks for the shortlist

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Shortlist page, book form and API

**Files:**
- Modify: `src/cfbr/web/store.py` (add `pricing_table`)
- Modify: `src/cfbr/web/bets.py` (imports and three routes inside `make_router`, after `api_bets`)
- Create: `src/cfbr/web/templates/bets_shortlist.html`
- Modify: `src/cfbr/web/templates/bets.html:5` and `src/cfbr/web/templates/bets_retro.html:6` (tabs)
- Modify: `README.md:59` (the Bets bullet)
- Test: `tests/test_web.py` (append)

**Interfaces:**
- Consumes: `shortlist(ledger, books, cfg.bets, table)` (Task 1); `tracked_tests(...)` entry `key == "shortlist"` (Task 2); `choose_books`, `book_cookie_value`, `viewer_books`, `BOOKS_COOKIE`, `BOOKS_COOKIE_DAYS` (Task 3); the existing `bet_rows`, `with_line_now`, `summarize`, `clean`, `clean_summary`, `NO_STORE`, `pick`.
- Produces: `GET /bets/shortlist`, `POST /bets/shortlist/books` (303 to `/bets/shortlist`), `GET /api/bets/shortlist?season=&books=`, and `Store.pricing_table() -> PricingTable | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web.py`. The fixture ledger from `_write_ledger` has `books` null and no `fbs_game`, so this writes its own:

```python
def _write_shortlist_ledger(cfg):
    from cfbr.bets.ledger import LEDGER_SCHEMA

    first, later = "2026-10-11T14:05:00+00:00", "2026-10-11T20:05:00+00:00"
    rows = [
        _bet(1, "home", -3.5) | {"fbs_game": True, "placed_at": first,
                                 "books": "DraftKings -3.5 (-110/-110); BetMGM -3 (-115/-105)"},
        _bet(2, "away", 4.0) | {"fbs_game": True, "placed_at": first, "books": "BetMGM +3.5 (-110/-110)"},
        _bet(3, "home", -3.0) | {"fbs_game": True, "placed_at": later, "books": "DraftKings -3 (-110/-110)"},
        _bet(5, "home", -7.0, "won", 31, 14, -8.0, week=6) | {"fbs_game": True, "placed_at": "2026-10-04T14:00:00+00:00",
                                                              "books": "FanDuel -7 (-110/-110)"},
    ]
    d = cfg.paths.out_dir / str(SEASON)
    d.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=LEDGER_SCHEMA).write_csv(d / "bets.csv")


@pytest.fixture
def shortlist_client(web_cfg):
    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_shortlist_ledger(web_cfg)
    return TestClient(create_app(web_cfg), follow_redirects=False)


def test_shortlist_defaults_to_the_reference_books(shortlist_client):
    r = shortlist_client.get("/bets/shortlist")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
    html = " ".join(r.text.split())
    assert "Shortlist" in html and "Texas A&amp;M" in html       # game 1 (home, DK at the line)
    assert "−3.5 or better at DraftKings" in html
    assert "Hawai" not in html.split('id="settled"')[0]          # game 2: only BetMGM has its number
    body = shortlist_client.get("/api/bets/shortlist").json()
    assert [p["game_id"] for p in body["picks"]] == [1, 5] and body["books"] == ["DraftKings", "FanDuel", "bet365"]
    assert body["summary"]["won"] == 1


def test_shortlist_follows_the_viewers_books(shortlist_client):
    c = shortlist_client
    r = c.post("/bets/shortlist/books", content="books=BetMGM&books=Bovada",
               headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 303 and r.headers["location"] == "/bets/shortlist"
    assert "cfbr_books=betmgm" in r.headers["set-cookie"] and "HttpOnly" in r.headers["set-cookie"]
    body = c.get("/api/bets/shortlist").json()
    assert body["books"] == ["BetMGM"] and [p["game_id"] for p in body["picks"]] == [1, 2]
    assert [p["bet_at_book"] for p in body["picks"]] == ["BetMGM", "BetMGM"]
    # An explicit ?books= wins over the cookie; junk falls back to the reference set.
    assert c.get("/api/bets/shortlist?books=FanDuel").json()["books"] == ["FanDuel"]
    c.cookies.set("cfbr_books", "nope.bovada")
    assert c.get("/api/bets/shortlist").json()["books"] == ["DraftKings", "FanDuel", "bet365"]
    # Submitting no books clears the choice.
    r = c.post("/bets/shortlist/books", content="", headers={"content-type": "application/x-www-form-urlencoded"})
    assert r.status_code == 303 and ("Max-Age=0" in r.headers["set-cookie"] or 'cfbr_books=""' in r.headers["set-cookie"])


def test_shortlist_is_gated_and_linked(web_cfg):
    from cfbr.web.access import bets_access

    _write_season(web_dir(web_cfg), datetime.now(UTC), polls=True)
    _write_shortlist_ledger(web_cfg)
    app = create_app(web_cfg)
    c = TestClient(app)
    assert 'href="/bets/shortlist?season=2026"' in c.get("/bets").text
    app.dependency_overrides[bets_access] = lambda: (_ for _ in ()).throw(HTTPException(402, "no"))
    assert c.get("/bets/shortlist").status_code == 402
    assert c.get("/api/bets/shortlist").status_code == 402
    assert c.post("/bets/shortlist/books", content="books=FanDuel").status_code == 402
```

Also extend the existing gating test at `tests/test_web.py:576-597` (the one that sets `deny` and loops over free URLs). After `assert c.get("/bets/retro").status_code == 402 …` add:

```python
    assert c.get("/bets/shortlist").status_code == 402 and c.get("/api/bets/shortlist").status_code == 402
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_web.py -k shortlist -q`
Expected: FAIL with 404s on `/bets/shortlist`.

- [ ] **Step 3: Add `Store.pricing_table`**

In `src/cfbr/web/store.py`, after `def bets(...)`:

```python
    def pricing_table(self):
        """The points-to-cents table ``cfbr weekly`` writes (out/pricing_table.json), or None.
        Read-only; without it the shortlist picks a book by points, then price."""
        from cfbr.bets.pricing import PricingTable

        path = self.out_dir / "pricing_table.json"
        if not path.exists():
            return None
        try:
            return PricingTable.load(path)
        except (ValueError, KeyError) as e:
            log.warning("pricing table at %s unreadable (%s); shortlist picks by points then price", path, e)
            return None
```

If `store.py` has no module logger yet (`grep -n '^log = ' src/cfbr/web/store.py`), add `import logging` and `log = logging.getLogger(__name__)` at the top, following `bets/run.py`.

- [ ] **Step 4: Add the routes**

In `src/cfbr/web/bets.py`, extend the imports:

```python
from urllib.parse import parse_qs

from fastapi.responses import HTMLResponse, RedirectResponse

from cfbr.bets.shortlist import BOOK_RE, shortlist
from cfbr.web.access import (BOOKS_COOKIE, BOOKS_COOKIE_DAYS, bets_access, book_cookie_value, choose_books,
                             owner_viewer, viewer_books)
```

(This merges with the existing `HTMLResponse` and `access` imports and keeps Task 1's `BOOK_RE` import.)

Inside `make_router`, after `api_bets`:

```python
    # -- real-money shortlist (DECISIONS "Real-money shortlist"): same gate, read-only ------------

    def shortlist_rows(sl: pl.DataFrame, s) -> tuple[list[dict], list[dict]]:
        """Open picks (with the current line, and whether it moved past the viewer's number) and
        settled picks, newest first."""
        rows = bet_rows(sl, cfg)
        open_ = with_line_now([r for r in rows if r["status"] == "open"], s, cfg.bets.threshold)
        open_ = [r | {"moved_past": r["line_now"] is not None and r["line_now"] < r["bet_at"]} for r in open_]
        return open_, [r for r in reversed(rows) if r["status"] != "open"]

    @router.get("/bets/shortlist", response_class=HTMLResponse, include_in_schema=False)
    def shortlist_page(request: Request, season: int | None = None):
        season, ledger = pick(season)
        books = viewer_books(request, cfg.bets)
        ctx = {"bet_seasons": store.bet_seasons(), "season": season, "rules": cfg.bets, "books": books,
               "reference": books == frozenset(cfg.bets.pricing_books),
               "has_retro": season in store.retro_seasons()}
        if ledger is not None:
            s = store.season(season)
            sl = shortlist(ledger, books, cfg.bets, store.pricing_table())
            open_picks, settled = shortlist_rows(sl, s)
            fbs = set() if s is None else set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
            ctx |= {"open_picks": open_picks, "settled": settled, "summary": summarize(sl, cfg.bets),
                    "n_paper": ledger.height, "fbs_teams": fbs,
                    "test": next(t for t in tracked_tests(ledger, cfg.bets) if t["key"] == "shortlist")}
        resp = render(request, "bets_shortlist.html", **ctx)
        resp.headers.update(NO_STORE)
        return resp

    @router.post("/bets/shortlist/books", include_in_schema=False)
    async def shortlist_books(request: Request):
        # Standard-library form parsing (repeated "books" fields), as /owner does.
        body = (await request.body())[:4096].decode("utf-8", "replace")
        names = parse_qs(body).get("books", [])
        resp = RedirectResponse("/bets/shortlist", status_code=303)
        if names:
            resp.set_cookie(BOOKS_COOKIE, book_cookie_value(choose_books(names, cfg.bets)),
                            max_age=BOOKS_COOKIE_DAYS * 86400, httponly=True, samesite="lax", path="/")
        else:
            resp.delete_cookie(BOOKS_COOKIE, path="/")
        resp.headers.update(NO_STORE)
        return resp

    @router.get("/api/bets/shortlist")
    def api_shortlist(request: Request, response: Response, season: int | None = None, books: str | None = None):
        from cfbr.web.app import clean

        season, ledger = pick(season)
        if ledger is None:
            raise HTTPException(404, "No paper bets recorded yet.")
        chosen = choose_books(books.split(","), cfg.bets) if books else viewer_books(request, cfg.bets)
        sl = shortlist(ledger, chosen, cfg.bets, store.pricing_table())
        response.headers.update(NO_STORE)
        return clean({"season": season, "books": sorted(chosen, key=str.lower),
                      "summary": clean_summary(summarize(sl, cfg.bets)), "picks": sl.to_dicts()})
```

`sorted(chosen, key=str.lower)` gives `["DraftKings", "FanDuel", "bet365"]` for the reference set, matching the test.

- [ ] **Step 5: Create the template**

Create `src/cfbr/web/templates/bets_shortlist.html`:

```html
{% extends "base.html" %}
{% from "macros.html" import season_select, tile, team_link, tabs %}
{% block title %}{% if season %}{{ season }} {% endif %}shortlist · CFB Ratings{% endblock %}
{% block content %}
{% set q = "?season=" ~ season if season else "" %}
{% set t = [("/bets" ~ q, "Live", false), ("/bets/shortlist" ~ q, "Shortlist", true)] %}
{% if has_retro %}{% set t = t + [("/bets/retro" ~ q, "Replay", false)] %}{% endif %}
{{ tabs(t, "Paper bets") }}
{% if bet_seasons|length > 1 %}
<form class="filters" method="get" action="/bets/shortlist">
  {{ season_select(bet_seasons, season) }}
  <noscript><button type="submit">Show</button></noscript>
</form>
{% endif %}

<h1>{% if season %}{{ season }} {% endif %}shortlist</h1>
<p class="lede">The paper bets worth a real bet at your sportsbooks: competitive FBS games, found at the week's first look,
  where one of your books had the number. Not betting advice. Check the live line before you bet.</p>

<form class="filters" method="post" action="/bets/shortlist/books">
  <fieldset><legend>Your books{% if reference %} <span class="muted">(default)</span>{% endif %}</legend>
  {% for b in rules.selectable_books %}
    <label><input type="checkbox" name="books" value="{{ b }}"{% if b in books %} checked{% endif %}> {{ b }}</label>
  {% endfor %}
  </fieldset>
  <button type="submit">Save</button>
</form>

{% if open_picks is not defined %}
<p class="muted">No paper bets recorded yet.</p>
{% else %}
<div class="tiles">
  {{ tile("Record", summary.won ~ "–" ~ summary.lost ~ "–" ~ summary.push, "won–lost–push") }}
  {{ tile("Avg CLV", (summary.avg_clv|signed(2) ~ " pts") if summary.n_clv else "—",
          ("beat the close " ~ summary.beat_close|pct(0) ~ " of " ~ summary.n_clv) if summary.n_clv else "after games close") }}
  {{ tile("On the list", (open_picks|length + settled|length) ~ " of " ~ n_paper, "paper bets this season") }}
</div>

<h2>Open picks</h2>
{% if open_picks %}
<div class="table-wrap">
<table class="data">
  <thead><tr><th class="opt">Kickoff ({{ tz_label }})</th><th>Game</th><th>Bet</th><th class="num opt">Line now</th><th class="opt">Recorded at</th></tr></thead>
  <tbody>
  {% for b in open_picks %}
    <tr>
      <td class="opt">{{ b.kickoff|kickoff }}</td>
      <td class="wrap">{{ team_link(b.away_team, season, fbs_teams) }} @ {{ team_link(b.home_team, season, fbs_teams) }}</td>
      <td class="strong wrap">{{ b.team }} {{ b.bet_at|spread }} or better at {{ b.bet_at_book }}{% if b.bet_at_odds is not none %} ({{ b.bet_at_odds|price }}){% endif %}
        {% if b.moved_past %}<br><span class="muted">Moved past your number ({{ b.line_now|spread }} now): pass</span>{% endif %}</td>
      <td class="num opt">{{ b.line_now|spread }}</td>
      <td class="opt">{{ b.team_line|spread }} consensus</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>
{% else %}
<p class="muted">No open picks{% if not reference %} at your books{% endif %}. A pick needs a competitive FBS game found at the week's first look, with one of your books at the number.</p>
{% endif %}

<details class="rules">
  <summary>How it works</summary>
  <ul>
    <li>Every pick is a <a href="/bets{{ q }}">paper bet</a> that also passes three rules fixed before any of them settled:
      FBS vs FBS with a line of 21 or less, not week 1; recorded at the week's first look; and one of your books offered the side at the paper line or better when it was recorded.</li>
    <li>"Or better at" is the best number among your books at that moment. Lines are captured on Sundays, so check the live number: if it has moved past yours, pass.</li>
    <li>Picks are in kickoff order. Bigger edges haven't covered more often in the holdout seasons, so they aren't ranked by edge.</li>
    <li>With no books chosen you see the default set (DraftKings, FanDuel, bet365), the list tracked below.</li>
  </ul>
</details>

{% if test %}
<h2>Is the shortlist better than the full list?</h2>
<p class="note">{{ test.question }}</p>
<div class="table-wrap">
<table class="data">
  <thead><tr><th>Group</th><th class="num opt">Settled</th><th class="num">W–L–P</th><th class="num opt">ROI</th><th class="num opt">Avg CLV</th></tr></thead>
  <tbody>
  {% for g in test.groups %}
    <tr><td class="team wrap">{{ g.label }}</td><td class="num opt">{{ g.settled }}</td><td class="num">{{ g.won }}–{{ g.lost }}–{{ g.push }}</td>
      <td class="num opt">{{ g.roi|pct }}</td><td class="num opt">{{ g.avg_clv|signed(2) if g.n_clv else "—" }}</td></tr>
  {% endfor %}
  </tbody>
</table>
</div>
<p class="note">{% if test.enough_data %}CLV gap: <strong>{{ test.gap|signed(2) }} pts</strong> [{{ test.ci95[0]|signed(2) }}, {{ test.ci95[1]|signed(2) }}].
  {% else %}<strong>No conclusion yet.</strong> {{ test.n_in }} and {{ test.n_out }} bets with a closing line, of the {{ test.min_clv_bets }} per group needed by {{ test.decision_point }}.{% endif %}
  {{ test.decision }}</p>
{% endif %}

<h2 id="settled">Settled picks</h2>
{% if settled %}
<div class="table-wrap">
<table class="data">
  <thead><tr><th>Game</th><th>Bet</th><th class="num">Result</th><th class="num opt">CLV</th></tr></thead>
  <tbody>
  {% for b in settled %}
    <tr><td class="wrap">{{ b.away_team }} @ {{ b.home_team }}</td><td class="wrap">{{ b.team }} {{ b.bet_at|spread }} at {{ b.bet_at_book }}</td>
      <td class="num">{{ b.status }}</td><td class="num opt">{{ b.clv|signed(1) if b.clv is not none else "—" }}</td></tr>
  {% endfor %}
  </tbody>
</table>
</div>
{% else %}<p class="muted">None settled yet.</p>{% endif %}
{% endif %}
{% endblock %}
```

Before relying on them, check that `kickoff`, `spread`, `price`, `pct`, `signed` and the `tz_label` global exist: `grep -n "filters\[\|globals\[" src/cfbr/web/app.py`. They're all used by `bets.html` today.

- [ ] **Step 6: Tabs on /bets and /bets/retro**

In `src/cfbr/web/templates/bets.html`, replace line 5:

```html
{% if has_retro %}{{ tabs([("/bets?season=" ~ season, "Live", true), ("/bets/retro?season=" ~ season, "Replay", false)], "Paper bets") }}{% endif %}
```

with:

```html
{% if season %}{% set t = [("/bets?season=" ~ season, "Live", true), ("/bets/shortlist?season=" ~ season, "Shortlist", false)] %}
{% if has_retro %}{% set t = t + [("/bets/retro?season=" ~ season, "Replay", false)] %}{% endif %}{{ tabs(t, "Paper bets") }}{% endif %}
```

In `src/cfbr/web/templates/bets_retro.html`, replace line 6:

```html
{{ tabs([("/bets" ~ q, "Live", false), ("/bets/retro" ~ q, "Replay", true)], "Paper bets") }}
```

with:

```html
{{ tabs([("/bets" ~ q, "Live", false), ("/bets/shortlist" ~ q, "Shortlist", false), ("/bets/retro" ~ q, "Replay", true)], "Paper bets") }}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_web.py -q`
Expected: all pass.

- [ ] **Step 8: README**

In `README.md`, in the Bets bullet (line 59), after the sentence about the **Replay** tab, add:

```markdown
The **Shortlist** tab (`/bets/shortlist`, `/api/bets/shortlist`, same gate) narrows the paper bets to the ones worth a real bet at the viewer's chosen sportsbooks (rules in DECISIONS.md "Real-money shortlist"); the choice is a cookie set from a form on the page, and `?books=DraftKings,FanDuel` overrides it in the API.
```

- [ ] **Step 9: Full suite and a manual look**

Run: `uv run pytest -q`
Expected: all pass (dataset tests skip if there's no local data).

Then check it against real data. The worktree has no `data/` or `out/` (they're gitignored), so link the main checkout's copies (read-only use; nothing here writes): `ln -s /Users/abtreece/Projects/cfbr/data data && ln -s /Users/abtreece/Projects/cfbr/out out`, then `uv run cfbr web serve` (http://127.0.0.1:8000). Check that:

- `/bets/shortlist` lists the week-7 first-capture core bets with "or better at DraftKings";
- ticking only BetMGM and saving changes the list;
- the Live, Shortlist and Replay tabs link to each other.

Stop the server and `rm data out` (the symlinks only) before committing.

- [ ] **Step 10: Commit**

```bash
git add src/cfbr/web/store.py src/cfbr/web/bets.py src/cfbr/web/templates/bets_shortlist.html src/cfbr/web/templates/bets.html src/cfbr/web/templates/bets_retro.html tests/test_web.py README.md
git commit -m "feat(web): shortlist page and API filtered by the viewer's books

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
