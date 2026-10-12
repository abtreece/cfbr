"""Access control for premium content in the web app.

Premium content is (1) the paper-betting section and (2) all model output for
a season still in progress: ratings, ranks, offense/defense, predicted margins
and win probabilities, for every game, finished or not. Each of those can be
turned back into the picks: predicted margin = home rating - away rating +
home field, and even ratings or past predictions from earlier weeks recreate
most picks (measured in DECISIONS.md). Free viewers still get past seasons,
the live season's schedule, scores, closing lines and polls, and aggregate
accuracy.

Everything is open today. To make premium content paid (pay-per-view,
subscription), implement :func:`is_premium`, e.g. check a session cookie or
signed token on ``request``. Every route depends on :func:`premium_viewer`, so
nothing else needs to change.

Before charging, re-review the data sources' terms (DECISIONS.md): the betting
pages show third-party odds (The Odds API, OddsPapi's bet365 lines), and
OddsPapi forbids redistributing its data as a standalone product, which a paid
page could be read as.

Separately, the *owner* (:func:`owner_viewer`) is whoever holds ``OWNER_KEY``: the
owner's real bets are shown only to them, whether or not premium is open.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from collections.abc import Iterable

from fastapi import Depends, HTTPException, Request

from cfbr.bets.ledger import book_key
from cfbr.config import REPO_ROOT, Bets

OWNER_KEY_NAME = "OWNER_KEY"
OWNER_COOKIE = "cfbr_owner"


def is_premium(request: Request) -> bool:
    """Whether this viewer may see premium content. Currently everyone may."""
    return True


def premium_viewer(request: Request) -> bool:
    """FastAPI dependency: the single switch every route consults."""
    return is_premium(request)


def bets_access(premium: bool = Depends(premium_viewer)) -> None:
    """Gate for /bets and /api/bets: the whole section is premium."""
    if not premium:
        raise HTTPException(402, "The betting section is for subscribers.")


# -- owner: the site owner's own data (real bets), separate from premium --------------
# The owner opens /owner?key=<OWNER_KEY> once; that sets an HttpOnly cookie holding an
# HMAC of the key (never the key). With no OWNER_KEY configured nobody is the owner.

def owner_key() -> str | None:
    """OWNER_KEY from the environment, else the gitignored .keys file; None if unset."""
    if os.environ.get(OWNER_KEY_NAME):
        return os.environ[OWNER_KEY_NAME]
    keys_file = REPO_ROOT / ".keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == OWNER_KEY_NAME and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def owner_token(key: str) -> str:
    """What the cookie holds: an HMAC of the key, so the cookie never carries the key itself."""
    return hmac.new(key.encode(), b"cfbr-owner-v1", hashlib.sha256).hexdigest()


def is_owner(request: Request) -> bool:
    key = owner_key()
    cookie = request.cookies.get(OWNER_COOKIE)
    return bool(key and cookie) and hmac.compare_digest(cookie, owner_token(key))


def owner_viewer(request: Request) -> bool:
    """FastAPI dependency: whether this viewer is the site owner (independent of premium)."""
    return is_owner(request)


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
