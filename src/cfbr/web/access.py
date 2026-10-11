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
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request


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
