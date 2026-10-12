# Real-money shortlist — design

Date: 2026-10-12. Rule of record: DECISIONS.md "Real-money shortlist", "T3 shortlist", "Shortlist book choice" (commit c4d5520, before the first week-7 kickoff at 2026-10-14 00:00 UTC). This spec covers how it's built; if the two ever disagree, DECISIONS wins.

## Goal

Each week the paper ledger records every game with |edge| ≥ 3 (30 bets in week 7). That's the right sample for the pre-registered 500-bet test and the wrong list to bet real money from. The shortlist is an opinionated subset for real bets, filtered by the viewer's sportsbooks, with its own tracked record so it can be judged.

Out of scope: changing the paper rule, staking/sizing, user accounts, mid-week line refreshes (they'd spend Odds API credits).

## The rule

A paper bet is on the shortlist when all three hold:

1. **T1 core.** `fbs_game`, |`line`| ≤ 21, not regular-season week 1 (the same expression as `experiments._core`).
2. **T2 first capture.** `placed_at` equals the week's first capture (`experiments.with_helpers`; the postseason is one bucket).
3. **Your number.** At capture, at least one of the viewer's books offered the bet's side at the recorded consensus `line` or better. Books come from the ledger's `books` column (home spread, home/away prices; `(offshore)` and `(pricing only)` markers). Offshore entries never count. A bet whose `books` is null doesn't qualify.

"Better" is from the bettor's side: team number = home spread × (+1 home, −1 away), and it qualifies when team number ≥ the team's recorded line.

Each pick carries **bet at {number} or better at {book} ({price})**: the best entry among the viewer's qualifying books, chosen the way `ledger.market` chooses the paper best line. With the points-to-cents `PricingTable`, that means the highest expected value against the consensus. Without it, the most points, then the better price, with unpriced entries counting as −110.

Ordering is by kickoff. Edge, tier and cover probability aren't used (holdout tiers and B4 showed no information).

Membership uses only capture-time data, so it's fixed once recorded.

## Units

### `bets/shortlist.py` (new, pure)

- `parse_books(books: str | None) -> list[BookLine]`: `BookLine(book, home_spread, home_price, away_price, offshore, pricing_only)`. It reuses the `_BOOK` regex, which moves from `web/bets.py` into this module, and `web/bets.py` imports it back. Unparseable entries are skipped.
- `shortlist(ledger, books: frozenset[str], cfg, table=None) -> pl.DataFrame`: the ledger rows on the shortlist, plus `bet_at`, `bet_at_book` and `bet_at_odds` (team perspective). Book names match through `ledger.book_key` (so "BetOnline.ag" matches "BetOnline").
- `in_shortlist(ledger, cfg, table=None) -> pl.Series[bool]`: membership under `cfg.bets.pricing_books`, for T3.

It depends on `experiments.with_helpers` and `_core` and on `ledger.book_key`, and it has no web imports.

### `bets/experiments.py`

Add `T3 · Shortlist` to `TESTS`, with its rule from `in_shortlist`. Its labels and decision text come from DECISIONS. `DECISION_POINT` and `MIN_CLV_BETS` stay shared with T1/T2.

`Test.rule` takes the ledger with helpers. T3 needs the pricing table to pick `bet_at`, but not for membership: membership only asks whether *any* allowed book reaches the line. So T3's rule doesn't need the table.

### Config (`config.py`, `config/default.toml`)

Add `bets.selectable_books: list[str]`. Its default is the non-offshore books observed in the 2026 week-7 captures (DraftKings, FanDuel, BetMGM, BetRivers, Bally Bet, betPARX, theScore Bet) plus bet365. bet365 arrives only as OddsPapi "pricing only" entries for games being bet, and none show in the local ledger copy yet. Names not present in a given week just never match. Adding a book later is a config change. Old configs stay loadable because the field has a default.

### `web/access.py`

- `BOOKS_COOKIE = "cfbr_books"`.
- `viewer_books(request) -> frozenset[str]`: a FastAPI dependency. It reads the cookie (comma-separated) and keeps only names in `selectable_books`, compared by `book_key`. If that leaves nothing, it falls back to `pricing_books`. This is the seam: accounts would replace only this function's body.

The cookie is a preference, not a credential: HttpOnly, SameSite=Lax, 1 year, path `/`.

### `web/bets.py` (all behind `bets_access`)

- `GET /bets/shortlist?season=`: the season's shortlist for `viewer_books`. Open picks come first, showing kickoff, team, bet at / book / price, consensus at capture, `line_now` from the existing `with_line_now`, and a **"moved past your number — pass"** flag when `line_now` is worse than `bet_at`. Below that are settled picks, a summary (`ledger.summarize` on the shortlisted rows) and T3's tracked-test block. A small form lists `selectable_books` as checkboxes, with the current choice checked.
- `POST /bets/shortlist/books`: a form body of `books` values. It sets or clears the cookie and returns a 303 to `/bets/shortlist`. Clearing means the reference set.
- `GET /api/bets/shortlist?season=&books=`: JSON. An explicit `books` query takes priority over the cookie, so API users don't need cookies. It's validated the same way.
- /bets links to the shortlist. The shortlist template is `templates/bets_shortlist.html`, in the existing bets style.

No other route or response gets shortlist or ledger data, which keeps the existing invariant.

## Data flow

`out/{season}/bets.csv` → `read_ledger` → `shortlist(ledger, viewer_books, cfg, table)` → template/JSON. The server computes it per request from the ledger file it already reads, and it never fits a model. Nothing new is written to disk, and the web build is unchanged.

## Error handling

- A missing or bad cookie gives the reference set.
- Unknown book names are dropped.
- If no bets qualify, an empty state says why: "none of your books matched," or no first-capture core bets this week.
- A null `books` field means the bet doesn't qualify and isn't flagged.
- A missing ledger file behaves the way /bets does today.

## Testing

- `tests/test_shortlist.py` (synthetic ledgers, no dataset needed):
  - each of the three rules, in and out;
  - home vs away "or better";
  - offshore entries ignored, and "pricing only" bet365 counted;
  - unpriced entries;
  - `book_key` aliasing;
  - null `books`;
  - `bet_at` choice with and without a table;
  - the default set equals `in_shortlist`.
- `tests/test_web.py`:
  - the cookie round-trip;
  - an invalid cookie falls back;
  - the `?books=` override;
  - 402 when not premium;
  - the existing "no ledger data outside /bets" test still passes with the new routes.
- `tests/test_bets.py` (where T1/T2 are tested): T3 appears and uses `pricing_books`.

## Later: accounts

When accounts exist, `viewer_books` reads the saved preference from the account instead of the cookie, and maybe migrates the cookie's value on first login. Nothing in `bets/shortlist.py`, the routes or the templates changes.
