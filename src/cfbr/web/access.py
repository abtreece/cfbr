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

from fastapi import Depends, HTTPException, Request

from cfbr.config import REPO_ROOT

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
