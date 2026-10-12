"""Paper-betting section of the web app (``/bets`` and ``/api/bets``).

Self-contained so it can be gated or split out later: its own router, its own
template, its own data (``out/{season}/bets.csv``, written by ``cfbr weekly``),
and every route depends on :func:`cfbr.web.access.bets_access`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime
from urllib.parse import parse_qs

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from cfbr.bets.experiments import tracked_tests
from cfbr.bets.ledger import by_tier, risk, summarize, tier_expr, week_label, weekly, what_if
from cfbr.bets.shortlist import BOOK_RE, shortlist
from cfbr.config import Config
from cfbr.web import MODEL_NAME
from cfbr.web.access import (BOOKS_COOKIE, BOOKS_COOKIE_DAYS, bets_access, book_cookie_value, choose_books,
                             owner_viewer, viewer_books)
from cfbr.web.lines import older_line
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
            "books_side": books_from_side(r.get("books"), home),
        })
    return out


def books_from_side(books: str | None, home: bool) -> str | None:
    """The ledger's per-book lines ("DraftKings +5.5 (-110/-110); ...", home spread and home/away
    prices) rewritten from the bet team's side: "DraftKings −5.5 (−110); ...". Entries that don't
    parse are kept as they are."""
    if not books:
        return books
    out = []
    for entry in books.split("; "):
        m = BOOK_RE.match(entry)
        if m is None:
            out.append(entry)
            continue
        spread = float(m["spread"]) * (1 if home else -1)
        price = m["hp"] if home else m["ap"]
        out.append(f"{m['book']} {spread_str(spread)}" + (f" ({price_str(int(price))})" if price else "") + m["rest"])
    return "; ".join(out)


def with_line_now(rows: list[dict], s, threshold: float) -> list[dict]:
    """Add the current line (the one /games shows, ``games.spread``) to open bets, from the
    side bet on, and whether the model is still ``threshold``+ points off it. A bet's own
    line never changes; this only explains why /games may no longer flag the game.

    ``line_now_older`` marks a line older than the page's ``lines_as_of``: Games keeps CFBD's
    line when the newest odds snapshot had none for the game, and stars it the same way."""
    # The ledger stores game ids as text; the season's games as integers.
    games = None if s is None else s.games
    cols = [c for c in ("spread", "line_source", "line_as_of") if games is not None and c in games.columns]
    lines = {} if games is None else {str(r["game_id"]): r for r in games.select("game_id", *cols).iter_rows(named=True)}
    lines_as_of = None if s is None else s.meta.get("lines_as_of")
    out = []
    for r in rows:
        g = lines.get(str(r["game_id"]), {})
        spread = g.get("spread")
        now = None if spread is None or spread != spread else (spread if r["side"] == "home" else -spread)
        edge_now = None if now is None else r["model_team"] + now
        older = now is not None and g.get("line_source") == "cfbd" and older_line(g.get("line_as_of"), lines_as_of)
        out.append(r | {"line_now": now, "edge_now": edge_now,
                        "line_now_older": older, "line_now_as_of": g.get("line_as_of") if older else None,
                        "below_threshold": edge_now is not None and edge_now < threshold})
    return out


def real_bets_ctx(real: pl.DataFrame | None, s) -> dict | None:
    """The owner's real bets for /bets: rows from the bettor's side plus a summary. Never
    mixed into the paper ledger's numbers, and never in /api/bets."""
    if real is None or real.height == 0:
        return None
    from cfbr.bets.real import summarize_real, win_dollars

    spreads = {} if s is None else {k: v for k, v in s.games.select("game_id", "spread").iter_rows()}
    rows = []
    for r in real.sort("kickoff_ts", "placed_at").to_dicts():
        flip = 1 if r["side"] == "home" else -1
        spread = spreads.get(r["game_id"])
        rows.append(r | {"kickoff": _ts(r["kickoff_ts"]), "placed": _ts(r["placed_at"]),
                         "team_line": r["line"] * flip, "to_win": win_dollars(r["odds"], r["risk_dollars"]),
                         "line_now": None if spread is None or spread != spread or r["status"] != "open" else spread * flip,
                         "close_team": None if r["closing_line"] is None else r["closing_line"] * flip})
    return {"rows": rows, "summary": summarize_real(real)}


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

    def pick(season: int | None, seasons: list[int] | None = None) -> tuple[int | None, pl.DataFrame | None]:
        """The season to show (default: newest of ``seasons``, the paper-ledger seasons unless
        given) and its paper ledger (None when that season has only real bets)."""
        seasons = store.bet_seasons() if seasons is None else seasons
        if season is None:
            season = seasons[-1] if seasons else None
        elif season not in seasons:
            raise HTTPException(404, f"No paper bets for {season}.")
        has_paper = season is not None and season in store.bet_seasons()
        return season, (store.bets(season) if has_paper else None)

    @router.get("/bets", response_class=HTMLResponse, include_in_schema=False)
    def bets_page(request: Request, season: int | None = None, owner: bool = Depends(owner_viewer)):
        # The owner also reaches seasons that have only real bets; everyone else (and the API)
        # sees paper-ledger seasons only.
        seasons = sorted(set(store.bet_seasons()) | set(store.real_seasons())) if owner else store.bet_seasons()
        season, ledger = pick(season, seasons)
        risked, to_win = risk(cfg.bets.odds, cfg.bets.stake)
        ctx = {"bet_seasons": seasons, "season": season, "rules": cfg.bets,
               "close_max_age": cfg.odds.closing_max_age_hours,
               "unit_risk": risked * cfg.bets.unit_value, "unit_win": to_win * cfg.bets.unit_value}
        if ledger is not None:
            rows = bet_rows(ledger, cfg)
            weeks = weekly(ledger, cfg.bets)
            fbs = set()
            s = store.season(season)
            if s is not None:
                fbs = set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
            open_bets = with_line_now([r for r in rows if r["status"] == "open"], s, cfg.bets.threshold)
            ctx |= {"summary": summarize(ledger, cfg.bets), "best": summarize(ledger, cfg.bets, "best"),
                    "breakdown": breakdown(ledger, cfg), "weeks": weeks, "tiers": by_tier(ledger, cfg.bets),
                    "tests": tracked_tests(ledger, cfg.bets),
                    "what_if": (wi := what_if(ledger, cfg.bets)), "what_if_chart": what_if_chart(wi, cfg.bets.bankroll),
                    "open_bets": open_bets,
                    # Bets whose "consensus" was one book (recorded before multi-book pricing, or
                    # on a run where the Odds API was unavailable); explained under "How it works".
                    "n_single_book": ledger.filter(pl.col("n_providers") == 1).height,
                    "n_below": sum(1 for r in open_bets if r["below_threshold"]),
                    # Older ledgers predate these columns; don't show a column of dashes.
                    "show_cents": any(r.get("edge_cents") is not None for r in open_bets),
                    "show_cover": any(r.get("cover_prob") is not None for r in open_bets),
                    "settled": [r for r in reversed(rows) if r["status"] != "open"],
                    "chart": money_chart(weeks, cfg.bets.bankroll), "fbs_teams": fbs,
                    "lines_as_of": s.meta.get("lines_as_of") if s is not None else None}
        # The owner's real bets: only for the owner (owner_viewer), never for other viewers,
        # premium or not, and never in /api/bets.
        ctx["real"] = (real_bets_ctx(store.real_bets(season), store.season(season))
                       if owner and season is not None else None)
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

    # -- real-money shortlist (DECISIONS "Real-money shortlist"): same gate, read-only ------------

    def shortlist_rows(sl: pl.DataFrame, s) -> tuple[list[dict], list[dict]]:
        """Open picks (with the current line, and whether the consensus has moved against the bettor
        since capture) and settled picks, newest first."""
        rows = bet_rows(sl, cfg)
        open_ = with_line_now([r for r in rows if r["status"] == "open"], s, cfg.bets.threshold)
        open_ = [r | {"moved_past": r["line_now"] is not None and r["line_now"] < r["team_line"]} for r in open_]
        return open_, [r for r in reversed(rows) if r["status"] != "open"]

    def season_pricing_table(season: int):
        """The pricing table only if it was built from seasons before ``season``; a later table
        would re-rank a past season's picks with its own results."""
        table = store.pricing_table()
        return table if table is not None and (not table.seasons or max(table.seasons) < season) else None

    @router.get("/bets/shortlist", response_class=HTMLResponse, include_in_schema=False)
    def shortlist_page(request: Request, season: int | None = None):
        season, ledger = pick(season)
        books = viewer_books(request, cfg.bets)
        ctx = {"bet_seasons": store.bet_seasons(), "season": season, "rules": cfg.bets, "books": books,
               "reference": books == frozenset(cfg.bets.pricing_books),
               "has_retro": season in store.retro_seasons()}
        if ledger is not None:
            s = store.season(season)
            sl = shortlist(ledger, books, cfg.bets, season_pricing_table(season))
            open_picks, settled = shortlist_rows(sl, s)
            fbs = set() if s is None else set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
            ctx |= {"open_picks": open_picks, "settled": settled, "summary": summarize(sl, cfg.bets),
                    "n_paper": ledger.height, "fbs_teams": fbs,
                    # The open picks' week(s), in kickoff order ("Week 7", "Postseason").
                    "open_weeks": list(dict.fromkeys(week_label(r["week"], r["season_type"]) for r in open_picks)),
                    "lines_as_of": s.meta.get("lines_as_of") if s is not None else None,
                    "test": next(t for t in tracked_tests(ledger, cfg.bets) if t["key"] == "shortlist")}
        resp = render(request, "bets_shortlist.html", **ctx)
        resp.headers.update(NO_STORE)
        return resp

    @router.post("/bets/shortlist/books", include_in_schema=False)
    async def shortlist_books(request: Request):
        # Standard-library form parsing (repeated "books" fields), as /owner does.
        body = (await request.body())[:4096].decode("utf-8", "replace")
        form = parse_qs(body)
        names = form.get("books", [])
        try:  # only a parsed int goes into the Location header, never the raw input
            where = f"/bets/shortlist?season={int(form['season'][0])}"
        except (KeyError, IndexError, ValueError):
            where = "/bets/shortlist"
        resp = RedirectResponse(where, status_code=303)
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
        chosen = choose_books(books.split(","), cfg.bets) if books is not None else viewer_books(request, cfg.bets)
        sl = shortlist(ledger, chosen, cfg.bets, season_pricing_table(season))
        response.headers.update(NO_STORE)
        return clean({"season": season, "books": sorted(chosen),
                      "summary": clean_summary(summarize(sl, cfg.bets)), "picks": sl.to_dicts()})

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


# Visitor-facing names for the retro models; the ledger and reports keep the internal ones
# (cfbr.bets.retro.MODELS: "published model (v1 prior)", "roster-prior shadow model").
RETRO_LABELS = {"v1": f"{MODEL_NAME} (published)", "roster": "Experimental roster model"}


def retro_models(led: pl.DataFrame, cfg: Config) -> list[dict]:
    """Per-model record, week-by-week, tiers and bet rows for the retro page (models in a fixed order)."""
    from cfbr.bets.retro import MODELS

    out = []
    for model in MODELS:
        d = led.filter(pl.col("model") == model)
        if d.height == 0:
            continue
        weeks = weekly(d, cfg.bets)
        out.append({"model": model, "label": RETRO_LABELS.get(model, model), "summary": summarize(d, cfg.bets),
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
