"""Paper-betting section of the web app (``/bets`` and ``/api/bets``).

Self-contained so it can be gated or split out later: its own router, its own
template, its own data (``out/{season}/bets.csv``, written by ``cfbr weekly``),
and every route depends on :func:`cfbr.web.access.bets_access`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from cfbr.bets.experiments import tracked_tests
from cfbr.bets.ledger import (HOLDOUT_TIER_COVER, OPENER_TIER_COVER, book_key, break_even, by_tier, risk, summarize, tier_expr,
                              tier_reference, tiers, week_label, weekly, what_if)
from cfbr.bets.shortlist import BOOK_RE, shortlist
from cfbr.bets.verdict import verdict
from cfbr.config import Config
from cfbr.web import MODEL_NAME
from cfbr.web.access import (BOOKS_COOKIE, BOOKS_COOKIE_DAYS, bets_access, book_cookie_value, choose_books,
                             owner_viewer, viewer_books)
from cfbr.web.lines import older_line
from cfbr.web.store import Store

# Betting data is per-viewer once the section is gated; never let a shared cache keep it.
NO_STORE = {"Cache-Control": "private, no-store"}
# A one-shot notice ("Recorded: ..." / "Voided.") after a save or void. Carried in a short-lived
# cookie, not the redirect URL, so it shows on the page the action lands on and never again on a
# reload, a bookmark or the back button. Only the owner's routes set or read it.
FLASH_COOKIE = "cfbr_flash"
FLASH_SECONDS = 60

# A same-terms real bet submitted again this soon (placed time blank) is a double-submit, not a second wager.
RESUBMIT_WINDOW = timedelta(minutes=10)
# The entry form lists games this far ahead: the week's slate and a little beyond, not the season.
FORM_WINDOW = timedelta(days=10)
# ...and games that kicked off this recently, for a live bet entered during or after the game.
LIVE_LOOKBACK = timedelta(days=3)

# Prices in the guide's break-even table, from more juice to a plus-money price.
GUIDE_PRICES = (-120, -115, -110, -105, 100, 105)


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


# Margins college football games end on most often; a line moving across one costs (or gains)
# more than its size suggests. Display only (DECISIONS "Shortlist line moves").
KEY_NUMBERS = (3, 7, 10, 14)


def keys_crossed(was: float, now: float) -> list[int]:
    """Key numbers whose result a move from ``was`` to ``now`` (the bettor's spread) changes:
    final margins (bet team's view) from -was to -now, counting a whole-number end, which
    turns a win or loss into a push."""
    lo, hi = sorted((-was, -now))
    return [k for k in KEY_NUMBERS if any(lo <= m <= hi for m in (k, -k))] if was != now else []


def line_move(r: dict, books: pl.DataFrame | None, threshold: float) -> dict:
    """What an open shortlist pick's book offers now, from the newest odds snapshot, against
    the number shown at capture (``bet_at``). Falls back to the consensus (``line_now`` against
    the paper line) when that book has no line for the game. ``move`` is from the bettor's side:
    positive means a better number. ``edge_now`` is the model's margin against the number now.
    Display only: it never changes the shortlist or a bet's grading."""
    flip = 1 if r["side"] == "home" else -1
    hit = None
    if books is not None and r.get("bet_at_book"):
        want = book_key(r["bet_at_book"])
        hit = next((b for b in books.filter(pl.col("game_id") == int(r["game_id"])).iter_rows(named=True)
                    if book_key(b["book"]) == want), None)
    if hit is not None:
        now, was = flip * hit["home_spread"], r["bet_at"]
        price = hit["home_price"] if flip == 1 else hit["away_price"]
        source = "book"
    elif r.get("line_now") is not None:
        now, was, price, source = r["line_now"], r["team_line"], None, "consensus"
    else:
        return {"move_source": None}
    edge_now = r["model_team"] + now
    return {"move_source": source, "move_now": now, "move_price": price, "move": now - was,
            "move_keys": keys_crossed(was, now), "move_edge": edge_now, "move_below": edge_now < threshold}


def real_bets_ctx(real: pl.DataFrame | None, s) -> dict:
    """The owner's real bets for /bets: rows from the bettor's side plus a summary (empty when
    none are recorded yet, so the owner still gets the entry form). Never mixed into the paper
    ledger's numbers, and never in /api/bets."""
    from cfbr.bets.real import REAL_SCHEMA, summarize_real, win_dollars

    if real is None:
        real = pl.DataFrame(schema=REAL_SCHEMA)
    spreads = {} if s is None else {k: v for k, v in s.games.select("game_id", "spread").iter_rows()}
    rows = []
    for r in real.sort("kickoff_ts", "placed_at").to_dicts():
        flip = 1 if r["side"] == "home" else -1
        spread = spreads.get(r["game_id"])
        ml = r["market"] == "moneyline"
        rows.append(r | {"kickoff": _ts(r["kickoff_ts"]), "placed": _ts(r["placed_at"]), "moneyline": ml,
                         "team_line": None if ml else r["line"] * flip, "to_win": win_dollars(r["odds"], r["risk_dollars"]),
                         "line_now": None if ml or spread is None or spread != spread or r["status"] != "open" else spread * flip,
                         "close_team": None if r["closing_line"] is None else r["closing_line"] * flip})
    return {"rows": rows, "summary": summarize_real(real)}


# -- the owner's entry form for a real bet (DECISIONS "Real-bet entry form") -------------------

def upcoming_games(s, now: datetime, keep: int | None = None) -> list[dict]:
    """One choice per game kicking off in the next ``FORM_WINDOW`` (the games with lines up) or
    in the last ``LIVE_LOOKBACK`` (for a live bet; ``started`` marks these), plus game ``keep``
    (a prefill link's game) whenever it is: away and home teams, neutral site, kickoff and the
    consensus home spread (a hint; the owner types the number they got). Kickoff order. The side
    is chosen separately, by team name, so the two teams of a game never sit in one list."""
    if s is None:
        return []
    ts = s.games.schema["kickoff_ts"]  # the parquet's zone name varies (UTC vs Etc/UTC)
    soon = (pl.col("kickoff_ts") > pl.lit(now).cast(ts)) & (pl.col("kickoff_ts") <= pl.lit(now + FORM_WINDOW).cast(ts))
    recent = (pl.col("kickoff_ts") <= pl.lit(now).cast(ts)) & (pl.col("kickoff_ts") > pl.lit(now - LIVE_LOOKBACK).cast(ts))
    choose = (~pl.col("completed") & soon) | recent
    if keep is not None:  # a prefill link's game, even if final: a backdated bet still needs a pre-kickoff placed time
        choose = choose | (pl.col("game_id") == keep)
    g = s.games.filter(choose).sort("kickoff_ts", "game_id")
    return [{"game_id": r["game_id"], "away_team": r["away_team"], "home_team": r["home_team"],
             "neutral": bool(r.get("neutral_site")), "kickoff": r["kickoff_ts"], "started": r["kickoff_ts"] <= now,
             "spread": None if r.get("spread") is None or r["spread"] != r["spread"] else r["spread"]}
            for r in g.to_dicts()]


def real_form_defaults(params, cfg: Config) -> dict:
    """Prefill for the entry form from query parameters (the "Record" links on the paper bets
    and shortlist put the pick's game, side, line, price and book there). Anything missing or
    unparsable is just left blank: the form validates on submit."""
    def num(key, cast):
        try:
            return cast(params.get(key, ""))
        except (TypeError, ValueError):
            return None
    return {"game": num("game", int), "side": params.get("side") if params.get("side") in ("home", "away") else None,
            "market": "spread", "live": False,
            "line": num("line", float), "odds": num("odds", int) or cfg.bets.odds,
            "stake": num("stake", float) or cfg.bets.unit_value, "book": (params.get("book") or "")[:40]}


def record_link(season: int, game_id: int, side: str, team_line: float | None, odds: int | None,
                book: str | None) -> str:
    """The owner's one-click prefill of the entry form for a paper pick."""
    from urllib.parse import urlencode

    q = {"season": season, "game": game_id, "side": side}
    if team_line is not None and team_line == team_line:
        q["line"] = f"{team_line:g}"
    if odds is not None and odds == odds:
        q["odds"] = int(odds)
    if book:
        q["book"] = book
    return f"/bets?{urlencode(q)}#record"


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


def parse_pick(game: str | None, side: str | None) -> tuple[int, str]:
    """The form's game select and team choice."""
    from cfbr.bets.real import RealBetError

    if not (game or "").isdigit():
        raise RealBetError("pick a game")
    if side not in ("home", "away"):
        raise RealBetError("pick the team you bet on")
    return int(game), side


def describe_bet(r: dict) -> str:
    """"BYU ML +158 (live)", "Texas A&M +3.5 −105": the bet as the owner would say it."""
    terms = "ML" if r["market"] == "moneyline" else spread_str(r["line"] if r["side"] == "home" else -r["line"])
    return f"{r['team']} {terms} {price_str(r['odds'])}" + (" (live)" if r["live"] else "")


def parse_number(s: str | None, what: str, cast):
    from cfbr.bets.real import RealBetError

    try:
        # Only the Unicode minus is normalized; int/float already accept a leading "+" and
        # reject one anywhere else, so "3+5" is an error rather than 35.
        return cast((s or "").replace("\u2212", "-"))
    except ValueError as e:
        raise RealBetError(f"{what} must be a number") from e


def parse_placed(s: str | None, now: datetime) -> datetime:
    """When the bet was placed: the form's datetime-local value (Central), else now."""
    from cfbr.bets.real import RealBetError
    from cfbr.web.app import DISPLAY_TZ

    if not s:
        return now
    try:
        t = datetime.fromisoformat(s)
    except ValueError as e:
        raise RealBetError("placed time must be a date and time") from e
    t = t.replace(tzinfo=DISPLAY_TZ) if t.tzinfo is None else t
    if t > now:
        raise RealBetError("placed time is in the future")
    return t.astimezone(UTC)


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

    def bets_ctx(request: Request, season: int | None, owner: bool) -> dict:
        """Everything /bets renders except the owner's section (``real``)."""
        # The owner also reaches seasons with only real bets, or with web data but no bets yet
        # (to record the first one); everyone else (and the API) sees paper-ledger seasons only.
        seasons = (sorted(set(store.bet_seasons()) | set(store.real_seasons()) | set(store.seasons()))
                   if owner else store.bet_seasons())
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
                    "verdict": verdict(ledger, cfg.bets, season_pricing_table(season)),
                    "verdict_withheld": store.pricing_table() is not None and season_pricing_table(season) is None,
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
            if owner:  # one-click prefill of the entry form, per open paper bet
                for r in open_bets:
                    r["record_link"] = record_link(season, r["game_id"], r["side"], r["best"] if r["best"] is not None else r["team_line"],
                                                   r.get("best_odds"), r.get("best_provider"))
        ctx["has_retro"] = season in store.retro_seasons()
        return ctx

    @router.get("/bets", response_class=HTMLResponse, include_in_schema=False)
    def bets_page(request: Request, season: int | None = None, owner: bool = Depends(owner_viewer)):
        ctx = bets_ctx(request, season, owner)
        season = ctx["season"]
        # The owner's real bets: only for the owner (owner_viewer), never for other viewers,
        # premium or not, and never in /api/bets.
        flash = request.cookies.get(FLASH_COOKIE) if owner else None
        ctx["real"] = real_ctx(season, request.query_params, flash=flash) if owner and season is not None else None
        resp = render(request, "bets.html", **ctx)
        if flash:  # shown once: this view consumes it
            resp.delete_cookie(FLASH_COOKIE, path="/bets")
        resp.headers.update(NO_STORE)
        return resp

    def flash_redirect(season: int, notice: str) -> RedirectResponse:
        """Back to the owner's section with a one-shot notice for the next page view."""
        resp = RedirectResponse(f"/bets?season={season}#mine", status_code=303)
        resp.set_cookie(FLASH_COOKIE, notice, max_age=FLASH_SECONDS, httponly=True, samesite="lax", path="/bets")
        resp.headers.update(NO_STORE)
        return resp

    def real_ctx(season: int, params, error: str | None = None, form: dict | None = None,
                 flash: str | None = None) -> dict:
        """The owner's section: recorded bets plus the entry form (upcoming games, prefill)."""
        s = store.season(season)
        ctx = real_bets_ctx(store.real_bets(season), s)
        form = form or real_form_defaults(params, cfg)
        keep = form["game"] if isinstance(form.get("game"), int) else None
        # After a save, name the bet just recorded (the newest row), so a wrong team is caught now.
        newest = max(ctx["rows"], key=lambda r: r["recorded_at"] or "", default=None)
        ctx |= {"games": upcoming_games(s, datetime.now(UTC), keep), "form": form, "error": error,
                "saved": describe_bet(newest) if flash == "saved" and newest else None,
                "voided": flash == "voided"}
        return ctx

    @router.post("/bets/real", response_class=HTMLResponse, include_in_schema=False)
    async def record_real_bet(request: Request, owner: bool = Depends(owner_viewer)):
        """Record one real bet from the owner's form, through the same ledger code as
        ``cfbr real-bet`` (locked, append-only, duplicates refused). Nobody but the owner can
        reach it; for everyone else the route doesn't exist. Nothing here touches the paper ledger."""
        from cfbr.bets.real import MARKETS, RealBetError, add_real, ledger_path, new_bet, same_terms, update_real

        if not owner:
            raise HTTPException(404, "Not found.")
        body = (await request.body())[:4096].decode("utf-8", "replace")
        form = {k: v[0].strip() for k, v in parse_qs(body).items()}
        try:
            season = int(form.get("season", ""))
        except ValueError as e:
            raise HTTPException(400, "Bad season.") from e
        s = store.season(season)
        if s is None:
            raise HTTPException(404, f"No data for {season}.")
        try:
            game_id, side = parse_pick(form.get("game"), form.get("side"))
            market = form.get("market") or "spread"
            if market not in MARKETS:
                raise RealBetError("pick a bet type")
            live = form.get("live") == "on"
            line = parse_number(form.get("line"), "line", float) if market == "spread" else None
            odds = parse_number(form.get("odds"), "price", int)
            stake = parse_number(form.get("stake"), "stake", float)
            book = form.get("book") or None
            now = datetime.now(UTC)
            placed_at = parse_placed(form.get("placed"), now)
            rows = s.games.filter(pl.col("game_id") == game_id).to_dicts()
            if not rows:
                raise RealBetError(f"no game {game_id} in {season}")
            game = rows[0]
            team = game[f"{side}_team"]
            bet = new_bet(game, team, line, odds, stake, book, placed_at, now, season, game.get("week"),
                          market=market, live=live)

            def add(df: pl.DataFrame) -> pl.DataFrame:
                # The ledger only refuses an exact repeat, placed time included. With the placed
                # time defaulted to "now", a double-submit would slip past that, so a same-terms
                # bet recorded in the last few minutes is refused unless a placed time was typed.
                if not form.get("placed"):
                    terms = ["game_id", "side", "line", "odds", "risk_dollars", "book", "market", "live"]
                    recent = df.filter(same_terms(bet, terms) & (pl.col("status") != "void")
                                       & (pl.col("recorded_at").str.to_datetime(time_zone="UTC") > now - RESUBMIT_WINDOW))
                    if recent.height:
                        raise RealBetError("the same bet was recorded a moment ago; set the placed time if this is a second wager")
                return add_real(df, bet)

            # The locked read-modify-write (a blocking flock plus CSV I/O) runs off the event loop
            # so a held ledger lock can't stall the worker's other requests.
            await run_in_threadpool(update_real, ledger_path(store.out_dir, season), add)
        except RealBetError as e:
            # Back to the page with the problem and what was typed; nothing was written.
            ctx = bets_ctx(request, season, owner=True)
            ctx["real"] = real_ctx(season, request.query_params, error=str(e),
                                   form={"game": int(form["game"]) if form.get("game", "").isdigit() else None,
                                         "side": form.get("side") if form.get("side") in ("home", "away") else None,
                                         "market": form.get("market") or "spread",
                                         "live": form.get("live") == "on", "line": form.get("line"), "odds": form.get("odds"),
                                         "stake": form.get("stake"), "book": form.get("book", ""),
                                         "placed": form.get("placed", "")})
            resp = render(request, "bets.html", **ctx)
            resp.status_code = 422
            resp.headers.update(NO_STORE)
            return resp
        return flash_redirect(season, "saved")

    @router.post("/bets/real/void", include_in_schema=False)
    async def void_real_bet(request: Request, owner: bool = Depends(owner_viewer)):
        """Void one open real bet (a mistaken entry): the row stays in the ledger, marked void,
        and counts for nothing. Owner-only like recording; a settled bet can't be voided."""
        from cfbr.bets.real import RealBetError, ledger_path, update_real, void_real

        if not owner:
            raise HTTPException(404, "Not found.")
        body = (await request.body())[:1024].decode("utf-8", "replace")
        form = {k: v[0].strip() for k, v in parse_qs(body).items()}
        try:
            season, game_id = int(form.get("season", "")), int(form.get("game_id", ""))
        except ValueError as e:
            raise HTTPException(400, "Bad bet.") from e
        recorded_at, now = form.get("recorded_at", ""), datetime.now(UTC)
        try:
            await run_in_threadpool(update_real, ledger_path(store.out_dir, season),
                                    lambda df: void_real(df, game_id, recorded_at, now))
        except RealBetError as e:
            raise HTTPException(409, f"Not voided: {e}.") from e
        return flash_redirect(season, "voided")

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
            "verdict": clean_summary(verdict(ledger, cfg.bets, season_pricing_table(season))),
            "tests": [clean_summary(t) | {"groups": [clean_summary(g) for g in t["groups"]]}
                      for t in tracked_tests(ledger, cfg.bets)],
            "what_if": what_if(ledger, cfg.bets),
            "bets": (ledger.with_columns(tier=tier_expr(cfg.bets), dollars=pl.col("units") * cfg.bets.unit_value)
                     .sort("kickoff_ts", "game_id").to_dicts()),
        })

    # -- real-money shortlist (DECISIONS "Real-money shortlist"): same gate, read-only ------------

    def shortlist_rows(sl: pl.DataFrame, s) -> tuple[list[dict], list[dict]]:
        """Open picks (with the current line, and how the pick's book has moved since capture:
        :func:`line_move`) and settled picks, newest first."""
        rows = bet_rows(sl, cfg)
        open_ = with_line_now([r for r in rows if r["status"] == "open"], s, cfg.bets.threshold)
        books = None if s is None else s.book_lines
        open_ = [r | line_move(r, books, cfg.bets.threshold) for r in open_]
        return open_, [r for r in reversed(rows) if r["status"] != "open"]

    def season_pricing_table(season: int):
        """The pricing table only if it was built from seasons before ``season``; a later table
        would re-rank a past season's picks with its own results."""
        table = store.pricing_table()
        return table if table is not None and (not table.seasons or max(table.seasons) < season) else None

    @router.get("/bets/shortlist", response_class=HTMLResponse, include_in_schema=False)
    def shortlist_page(request: Request, season: int | None = None, owner: bool = Depends(owner_viewer)):
        season, ledger = pick(season)
        books = viewer_books(request, cfg.bets)
        ctx = {"bet_seasons": store.bet_seasons(), "season": season, "rules": cfg.bets, "books": books,
               "reference": books == frozenset(cfg.bets.pricing_books),
               "has_retro": season in store.retro_seasons()}
        if ledger is not None:
            s = store.season(season)
            sl = shortlist(ledger, books, cfg.bets, season_pricing_table(season))
            open_picks, settled = shortlist_rows(sl, s)
            if owner:  # one-click prefill of the real-bet form with the pick at its book
                for r in open_picks:
                    at = r["move_now"] if r.get("move_source") == "book" else r["bet_at"]
                    price = r["move_price"] if r.get("move_source") == "book" else r.get("bet_at_odds")
                    r["record_link"] = record_link(season, r["game_id"], r["side"], at, price, r["bet_at_book"])
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

    @router.get("/bets/guide", response_class=HTMLResponse, include_in_schema=False)
    def guide_page(request: Request):
        # Definitions and made-up examples only, never the ledger; numbers come from config and
        # the backtest constants so the page can't drift from the rules it explains.
        risked, to_win = risk(cfg.bets.odds, cfg.bets.stake)
        resp = render(request, "bets_guide.html", rules=cfg.bets, has_retro=bool(store.retro_seasons()),
                      prices=[{"odds": o, "risk": risk(o, 1.0)[0] * 100, "win": risk(o, 1.0)[1] * 100,
                               "break_even": break_even(o)} for o in GUIDE_PRICES],
                      break_even_std=break_even(cfg.bets.odds),
                      tiers=[t | {"holdout": tier_reference(t, HOLDOUT_TIER_COVER),
                                    "opener": tier_reference(t, OPENER_TIER_COVER)} for t in tiers(cfg.bets)],
                      unit_risk=risked * cfg.bets.unit_value, unit_win=to_win * cfg.bets.unit_value)
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
