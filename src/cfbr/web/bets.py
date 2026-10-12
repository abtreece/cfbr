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

from cfbr.bets.experiments import tracked_tests
from cfbr.bets.ledger import by_tier, risk, summarize, tier_expr, weekly, what_if
from cfbr.config import Config
from cfbr.web.access import bets_access
from cfbr.web.store import Store

# Betting data is per-viewer once the section is gated; never let a shared cache keep it.
NO_STORE = {"Cache-Control": "private, no-store"}


def spread_str(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return "PK" if x == 0 else f"{x:+g}".replace("-", "−")


def price_str(odds: int | None) -> str:
    """American odds for display, e.g. +105 or −115."""
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return ""
    return f"{int(odds):+d}".replace("-", "−")


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None  # 3.11+ parses a trailing "Z"


def bet_rows(ledger: pl.DataFrame, cfg: Config) -> list[dict]:
    """Ledger rows with display fields from the bet team's side (lines are stored home-perspective)."""
    out = []
    uv = cfg.bets.unit_value
    for r in ledger.with_columns(tier=tier_expr(cfg.bets)).sort("kickoff_ts", "game_id").iter_rows(named=True):
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
            "risk_dollars": risked * uv, "win_dollars": to_win * uv,
            "dollars": None if r["units"] is None else r["units"] * uv,
        })
    return out


def breakdown(ledger: pl.DataFrame, cfg: Config) -> list[dict]:
    """Same slices as ``bets.md``'s top table: all (consensus, best line), by side."""
    groups = [("All bets", ledger),
              ("Home side", ledger.filter(pl.col("side") == "home")),
              ("Away side", ledger.filter(pl.col("side") == "away"))]
    out = [{"label": label, **summarize(df, cfg.bets)} for label, df in groups]
    return out[:1] + [{"label": "All bets, best line", **summarize(ledger, cfg.bets, "best")}] + out[1:]


def what_if_chart(plans: list[dict], bankroll: float) -> dict | None:
    """One bankroll line per staking plan (the live flat-1% plan first, so it gets the first series color)."""
    if not plans or not plans[0]["weeks"]:
        return None
    return {"signed": False, "format": "dollars", "x": ["Start"] + plans[0]["weeks"],
            # Whole dollars keep the end labels short enough to fit at phone width.
            "series": [{"name": p["name"], "values": [round(bankroll)] + [round(v) for v in p["path"]]} for p in plans]}


def money_chart(weeks: list[dict], bankroll: float) -> dict | None:
    """Bankroll from the start through each week with settled bets."""
    if not weeks:
        return None
    return {"signed": False, "format": "dollars", "x": ["Start"] + [w["label"] for w in weeks],
            "series": [{"name": "Bankroll ($)", "values": [round(bankroll, 2)] + [round(w["balance"], 2) for w in weeks]}]}


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
        risked, to_win = risk(cfg.bets.odds, cfg.bets.stake)
        ctx = {"bet_seasons": store.bet_seasons(), "season": season, "rules": cfg.bets,
               "unit_risk": risked * cfg.bets.unit_value, "unit_win": to_win * cfg.bets.unit_value}
        if ledger is not None:
            rows = bet_rows(ledger, cfg)
            weeks = weekly(ledger, cfg.bets)
            fbs = set()
            s = store.season(season)
            if s is not None:
                fbs = set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
            open_bets = [r for r in rows if r["status"] == "open"]
            ctx |= {"summary": summarize(ledger, cfg.bets), "best": summarize(ledger, cfg.bets, "best"),
                    "breakdown": breakdown(ledger, cfg), "weeks": weeks, "tiers": by_tier(ledger, cfg.bets),
                    "tests": tracked_tests(ledger, cfg.bets),
                    "what_if": (wi := what_if(ledger, cfg.bets)), "what_if_chart": what_if_chart(wi, cfg.bets.bankroll),
                    "open_bets": open_bets,
                    # Older ledgers predate these columns; don't show a column of dashes.
                    "show_cents": any(r.get("edge_cents") is not None for r in open_bets),
                    "show_cover": any(r.get("cover_prob") is not None for r in open_bets),
                    "settled": [r for r in reversed(rows) if r["status"] != "open"],
                    "chart": money_chart(weeks, cfg.bets.bankroll), "fbs_teams": fbs}
        ctx["has_retro"] = season in store.retro_seasons()
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
            "weekly": [clean_summary(w) for w in weekly(ledger, cfg.bets)],
            "tiers": [clean_summary(t) for t in by_tier(ledger, cfg.bets)],
            "tests": [clean_summary(t) | {"groups": [clean_summary(g) for g in t["groups"]]}
                      for t in tracked_tests(ledger, cfg.bets)],
            "what_if": what_if(ledger, cfg.bets),
            "bets": (ledger.with_columns(tier=tier_expr(cfg.bets), dollars=pl.col("units") * cfg.bets.unit_value)
                     .sort("kickoff_ts", "game_id").to_dicts()),
        })

    # -- retroactive replay (``cfbr bets-retro``): same gate, never mixed with the live ledger --

    def pick_retro(season: int | None) -> tuple[int | None, pl.DataFrame | None]:
        seasons = store.retro_seasons()
        if season is None:
            season = seasons[-1] if seasons else None
        elif season not in seasons:
            raise HTTPException(404, f"No retroactive paper bets for {season}.")
        return season, (store.retro_bets(season) if season is not None else None)

    @router.get("/bets/retro", response_class=HTMLResponse, include_in_schema=False)
    def retro_page(request: Request, season: int | None = None):
        season, led = pick_retro(season)
        ctx = {"retro_seasons": store.retro_seasons(), "season": season, "rules": cfg.bets}
        if led is not None and led.height:
            s = store.season(season)
            fbs = set(s.ratings.filter(pl.col("fbs"))["team"].to_list()) if s is not None else set()
            ctx |= {"models": retro_models(led, cfg), "overlap": retro_overlap(led),
                    "through_week": led["week"].max(), "fbs_teams": fbs,
                    "breakeven": summarize(led, cfg.bets)["breakeven"]}
        resp = render(request, "bets_retro.html", **ctx)
        resp.headers.update(NO_STORE)
        return resp

    @router.get("/api/bets/retro")
    def api_retro(response: Response, season: int | None = None):
        from cfbr.web.app import clean

        season, led = pick_retro(season)
        if led is None:
            raise HTTPException(404, "No retroactive paper bets.")
        response.headers.update(NO_STORE)
        return clean({
            "season": season, "retroactive": True, "rules": cfg.bets.model_dump(),
            "models": [{"model": m["model"], "label": m["label"], "summary": clean_summary(m["summary"]),
                        "summary_best_line": clean_summary(m["best"]),
                        "weekly": [clean_summary(w) for w in m["weeks"]],
                        "tiers": [clean_summary(t) for t in m["tiers"]], "what_if": m["what_if"]}
                       for m in retro_models(led, cfg)],
            "bets": (led.with_columns(tier=tier_expr(cfg.bets), dollars=pl.col("units") * cfg.bets.unit_value)
                     .sort("model", "kickoff_ts", "game_id").to_dicts()),
        })

    return router


def retro_models(led: pl.DataFrame, cfg: Config) -> list[dict]:
    """Per-model record, week-by-week, tiers and bet rows for the retro page (models in a fixed order)."""
    from cfbr.bets.retro import MODELS

    out = []
    for model, label in MODELS.items():
        d = led.filter(pl.col("model") == model)
        if d.height == 0:
            continue
        weeks = weekly(d, cfg.bets)
        out.append({"model": model, "label": label[:1].upper() + label[1:], "summary": summarize(d, cfg.bets),
                    "best": summarize(d, cfg.bets, "best"), "weeks": weeks, "tiers": by_tier(d, cfg.bets),
                    "chart": money_chart(weeks, cfg.bets.bankroll), "rows": list(reversed(bet_rows(d, cfg))),
                    "what_if": (wi := what_if(d, cfg.bets)), "what_if_chart": what_if_chart(wi, cfg.bets.bankroll)})
    return out


def retro_overlap(led: pl.DataFrame) -> dict | None:
    sides = led.pivot(on="model", index="game_id", values="side", aggregate_function="first")
    if not {"v1", "roster"} <= set(sides.columns):
        return None
    both = sides.filter(pl.col("v1").is_not_null() & pl.col("roster").is_not_null())
    return {"both": both.height, "same_side": int((both["v1"] == both["roster"]).sum()),
            "v1_only": int(sides["v1"].is_not_null().sum()) - both.height,
            "roster_only": int(sides["roster"].is_not_null().sum()) - both.height}
