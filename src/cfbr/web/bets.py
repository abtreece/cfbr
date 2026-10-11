"""Paper-betting section of the web app (``/bets`` and ``/api/bets``).

Self-contained so it can be gated or split out later: its own router, its own
template, its own data (``out/{season}/bets.csv``, written by ``cfbr weekly``),
and every route depends on :func:`cfbr.web.access.bets_access`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse

from cfbr.bets.ledger import risk, summarize
from cfbr.config import Config
from cfbr.web.access import bets_access
from cfbr.web.store import Store

# Betting data is per-viewer once the section is gated; never let a shared cache keep it.
NO_STORE = {"Cache-Control": "private, no-store"}


def spread_str(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return "PK" if x == 0 else f"{x:+g}".replace("-", "−")


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None  # 3.11+ parses a trailing "Z"


def bet_rows(ledger: pl.DataFrame) -> list[dict]:
    """Ledger rows with display fields from the bet team's side (lines are stored home-perspective)."""
    out = []
    for r in ledger.sort("kickoff_ts", "game_id").iter_rows(named=True):
        home = r["side"] == "home"
        flip = 1 if home else -1
        risked, to_win = risk(r["odds"], r["stake"])
        out.append(r | {
            "kickoff": _ts(r["kickoff_ts"]), "placed": _ts(r["placed_at"]),
            "team_line": r["line"] * flip,
            "best": None if r["best_line"] is None else r["best_line"] * flip,
            "close": None if r["closing_line"] is None else r["closing_line"] * flip,
            "model_team": r["model_margin"] * flip,  # model's expected margin for the bet team
            "edge_pts": abs(r["edge"]), "risked": risked, "to_win": to_win,
        })
    return out


def breakdown(ledger: pl.DataFrame, cfg: Config) -> list[dict]:
    """Same slices as ``bets.md``: all, by edge size, by side, by week."""
    t = cfg.bets.threshold
    rows = [{"label": "All bets, best line", **summarize(ledger, cfg.bets, "best")}]
    groups = [("All bets", ledger),
              (f"Edge {t:g}–6 pts", ledger.filter(pl.col("edge").abs() < 6)),
              ("Edge 6+ pts", ledger.filter(pl.col("edge").abs() >= 6)),
              ("Home side", ledger.filter(pl.col("side") == "home")),
              ("Away side", ledger.filter(pl.col("side") == "away"))]
    for wk in sorted(ledger.filter(pl.col("status") != "open")["week"].drop_nulls().unique().to_list()):
        groups.append((f"Week {wk}", ledger.filter(pl.col("week") == wk)))
    out = [{"label": label, **summarize(df, cfg.bets)} for label, df in groups]
    return out[:1] + rows + out[1:]


def units_chart(ledger: pl.DataFrame) -> dict | None:
    """Cumulative units after each week with settled bets (needs at least two weeks)."""
    s = (ledger.filter(pl.col("status").is_in(["won", "lost", "push"]))
         .group_by("week").agg(units=pl.col("units").sum(), n=pl.len()).sort("week"))
    if s.height < 2:
        return None
    return {"signed": True, "x": [f"Week {w}" for w in s["week"].to_list()], "n": s["n"].to_list(),
            "series": [{"name": "Cumulative units", "values": s["units"].cum_sum().round(2).to_list()}]}


def clean_summary(m: dict) -> dict:
    return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in m.items()}


def make_router(cfg: Config, store: Store, render: Callable[..., HTMLResponse]) -> APIRouter:
    router = APIRouter(dependencies=[Depends(bets_access)])

    def pick(season: int | None) -> tuple[int | None, pl.DataFrame | None]:
        seasons = store.bet_seasons()
        if season is None:
            season = seasons[-1] if seasons else None
        elif season not in seasons:
            raise HTTPException(404, f"No paper bets for {season}.")
        return season, (store.bets(season) if season is not None else None)

    @router.get("/bets", response_class=HTMLResponse, include_in_schema=False)
    def bets_page(request: Request, season: int | None = None):
        season, ledger = pick(season)
        ctx = {"bet_seasons": store.bet_seasons(), "season": season, "rules": cfg.bets}
        if ledger is not None:
            rows = bet_rows(ledger)
            fbs = set()
            s = store.season(season)
            if s is not None:
                fbs = set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
            ctx |= {"summary": summarize(ledger, cfg.bets), "best": summarize(ledger, cfg.bets, "best"),
                    "breakdown": breakdown(ledger, cfg),
                    "open_bets": [r for r in rows if r["status"] == "open"],
                    "settled": [r for r in reversed(rows) if r["status"] != "open"],
                    "chart": units_chart(ledger), "fbs_teams": fbs}
        resp = render(request, "bets.html", **ctx)
        resp.headers.update(NO_STORE)
        return resp

    @router.get("/api/bets")
    def api_bets(response: Response, season: int | None = None):
        from cfbr.web.app import clean

        season, ledger = pick(season)
        if ledger is None:
            raise HTTPException(404, "No paper bets recorded yet.")
        response.headers.update(NO_STORE)
        return clean({
            "season": season,
            "rules": cfg.bets.model_dump(),
            "summary": clean_summary(summarize(ledger, cfg.bets)),
            "summary_best_line": clean_summary(summarize(ledger, cfg.bets, "best")),
            "bets": ledger.sort("kickoff_ts", "game_id").to_dicts(),
        })

    return router
