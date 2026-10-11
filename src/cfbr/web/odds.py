"""Season-simulation odds (``/odds``, ``/api/odds``): conference-title and playoff
probabilities from ``cfbr weekly``'s Monte Carlo (``out/{season}/wNN/season_sim.csv``).

These are model output built from the live ratings, so while a season is in
progress they are premium (see cfbr.web.access); a finished season's odds are free.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable

import polars as pl
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from markupsafe import Markup

from cfbr.web.access import premium_viewer
from cfbr.web.bets import NO_STORE
from cfbr.web.store import Store

PROB_COLS = ["p_undefeated", "p_bowl_eligible", "p_title_game", "p_conf_title", "p_playoff", "p_bye", "p_champion"]


def prob(p) -> str:
    """Percent for a probability, keeping rare-but-possible distinct from impossible/certain."""
    if p is None:
        return "—"
    if p <= 0:
        return "0%"
    if p >= 1:
        return "100%"
    if p < 0.005:
        return "<1%"
    if p > 0.995:
        return ">99%"
    return f"{100 * p:.0f}%"


def notes_html(md: str) -> Markup:
    """The simulation's own notes (plain text with **bold**), escaped, as paragraphs."""
    paras = []
    for line in md.splitlines():
        safe = html.escape(line)
        paras.append("<p>" + re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe) + "</p>")
    return Markup("\n".join(paras))


def conference_odds(df: pl.DataFrame, min_p: float = 0.01) -> list[dict]:
    """Per conference: teams with at least ``min_p`` title odds, best first, and the rest's share."""
    out = []
    for (conf,), g in df.filter(pl.col("p_conf_title") > 0).group_by("conference", maintain_order=True):
        g = g.sort("p_conf_title", descending=True)
        top = g.filter(pl.col("p_conf_title") >= min_p)
        out.append({"conference": conf, "teams": top.select("team", "p_conf_title", "p_title_game").to_dicts(),
                    "rest": float(g.filter(pl.col("p_conf_title") < min_p)["p_conf_title"].sum()),
                    "leader": float(g["p_conf_title"].max())})
    return sorted(out, key=lambda c: c["conference"])


def make_router(store: Store, render: Callable[..., HTMLResponse]) -> APIRouter:
    router = APIRouter()

    def pick(season: int | None, week: int | None, premium: bool):
        seasons = [y for y in store.seasons() if store.sim_weeks(y)]
        if not seasons:
            raise HTTPException(404, "No season simulations yet; `cfbr weekly` writes them.")
        season = season or seasons[-1]
        weeks = store.sim_weeks(season)
        if not weeks:
            raise HTTPException(404, f"No season simulation for {season}.")
        week = week if week is not None else weeks[-1]
        if week not in weeks:
            raise HTTPException(404, f"No season simulation for {season} week {week}.")
        s = store.season(season)
        live = bool(s.meta.get("live")) if s is not None else True
        return seasons, season, weeks, week, live, live and not premium

    @router.get("/odds", response_class=HTMLResponse, include_in_schema=False)
    def odds_page(request: Request, season: int | None = None, week: int | None = None, conf: str | None = None,
                  premium: bool = Depends(premium_viewer)):
        seasons, season, weeks, week, live, locked = pick(season, week, premium)
        ctx = {"sim_seasons": seasons, "season": season, "weeks": weeks, "week": week, "locked": locked}
        if not locked:
            df, notes = store.season_sim(season, week)
            conferences = sorted(c for c in df["conference"].unique().to_list() if c)
            rows = df.filter(pl.col("conference") == conf) if conf else df
            ctx |= {"rows": rows.to_dicts(), "notes": notes_html(notes), "conf": conf or None,
                    "conferences": conferences, "conf_odds": conference_odds(df)}
        resp = render(request, "odds.html", **ctx)
        if live:
            resp.headers.update(NO_STORE)
        return resp

    @router.get("/api/odds")
    def api_odds(response: Response, season: int | None = None, week: int | None = None,
                 premium: bool = Depends(premium_viewer)):
        from cfbr.web.app import clean

        _, season, _, week, live, locked = pick(season, week, premium)
        if locked:
            raise HTTPException(402, f"{season} season odds are for subscribers until the season ends.")
        if live:
            response.headers.update(NO_STORE)
        df, notes = store.season_sim(season, week)
        return clean({"season": season, "week": week, "notes": notes, "teams": df.to_dicts()})

    return router
