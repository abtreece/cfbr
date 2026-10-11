"""Roster pages (``/roster/{season}/{team}``, ``/api/roster/...``).

Rosters are public facts (who is on the team, where transfers came from, who
left, last season's production), not model output, so they are free even for a
live season. The published v1 model doesn't use the roster features (the roster
prior was evaluated and not adopted), so showing them can't recreate picks.
Data comes from ``cfbr.report.roster_web`` via the web build.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import polars as pl
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from cfbr.report.roster_web import CAREER_LABELS, PLAYER_EXTRA_LABELS, ROSTER_SUMMARY_LABELS
from cfbr.web.store import Roster, Store

CLASS = {1: "Fr", 2: "So", 3: "Jr", 4: "Sr", 5: "Gr"}
POSITION_ORDER = ["QB", "RB", "FB", "WR", "TE", "OL", "OT", "G", "C", "DL", "DE", "DT", "EDGE", "LB", "DB", "CB", "S",
                  "K", "P", "LS", "ATH"]
SHARES = {"off_ret_share", "def_ret_tackles", "def_ret_havoc"}
PPA_TOTALS = {"off_in_fbs", "off_in_fcs", "off_out", "qb_in_ppa"}


def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def height_str(inches) -> str:
    if _missing(inches) or inches <= 0:
        return "—"
    return f"{int(inches) // 12}-{int(inches) % 12}"


def count_str(x) -> str:
    """Counts as whole numbers; half-credit stats (e.g. 0.5 TFL) keep one decimal."""
    if _missing(x):
        return "—"
    return f"{x:,.0f}" if abs(x - round(x)) < 0.05 else f"{x:,.1f}"


def feature_value(key: str, x) -> str:
    if _missing(x):
        return "—"
    if key in SHARES:
        return f"{100 * x:.0f}%"
    if key == "qb_returns":
        return "Yes" if x >= 0.5 else "No"
    if key == "qb_in_ppa" and x == 0:
        # 0 means no incoming QB with 100+ pass attempts, not "no transfer QB".
        return "None with 100+ att."
    if key in PPA_TOTALS:
        return f"{x:.0f} PPA".replace("-", "−")
    return count_str(x)


def summary_tiles(row: dict) -> list[dict]:
    return [{"key": k, "label": label, "value": feature_value(k, row.get(k)), "definition": definition}
            for k, (label, definition) in ROSTER_SUMMARY_LABELS.items()]


def incoming_qb(summary: dict | None) -> dict | None:
    """The incoming transfer QB with the most pass attempts last season (any count), if any.
    Complements qb_in_ppa, which only counts QBs with 100+ attempts."""
    if not summary or not summary.get("in_qb_name"):
        return None
    att, avg = summary.get("in_qb_pass_att"), summary.get("in_qb_pass_ppa_avg")
    parts = [f"from {summary['in_qb_from']}" if summary.get("in_qb_from") else None,
             f"{count_str(att)} pass att." if not _missing(att) else None,
             f"{avg:+.2f} PPA/att.".replace("-", "−") if not _missing(avg) else None]
    return {"name": summary["in_qb_name"], "detail": " · ".join(p for p in parts if p)}


def player_rows(players: pl.DataFrame) -> list[dict]:
    order = {p: i for i, p in enumerate(POSITION_ORDER)}
    rows = []
    for r in players.iter_rows(named=True):
        rows.append(r | {
            "class_label": CLASS.get(r["class_year"], "—") if r["class_year"] is not None else "—",
            "height_label": height_str(r["height"]),
            "hometown": ", ".join(x for x in (r["home_city"], r["home_state"]) if x) or "—",
            "pos_order": order.get(r["position"], len(order)),
        })
    return sorted(rows, key=lambda r: (r["pos_order"], r["jersey"] if r["jersey"] is not None else 999, r["name"] or ""))


def team_roster(roster: Roster, team: str) -> tuple[pl.DataFrame, pl.DataFrame, dict | None]:
    p = roster.players.filter(pl.col("team") == team)
    d = roster.departures.filter(pl.col("team") == team)
    s = roster.summary.filter(pl.col("team") == team)
    return p, d, (s.to_dicts()[0] if s.height else None)


def roster_card(roster: Roster | None, team: str) -> dict | None:
    """A few headline numbers for the team page; None if there is no roster for the team."""
    if roster is None:
        return None
    p, d, s = team_roster(roster, team)
    if p.height == 0:
        return None
    counts = dict(p.group_by("status").len().iter_rows())
    return {"size": p.height, "returners": counts.get("returner", 0), "transfers_in": counts.get("transfer_in", 0),
            "no_stats": counts.get("new", 0), "departures": d.height, "transfers_out": d["to_team"].is_not_null().sum(),
            "ret_offense": feature_value("off_ret_share", s.get("off_ret_share")) if s else "—",
            "qb_back": feature_value("qb_returns", s.get("qb_returns")) if s else "—",
            "as_of": s.get("roster_as_of") if s else None}


# Career table column groups; a group is shown only if the player has any value in it.
CAREER_GROUPS = {
    "Passing": ["qb_starts", "pass_att", "pass_ppa_avg"],
    "Rushing / receiving": ["rush_car", "rec"],
    "Offense": ["plays", "ppa_avg"],
    "Defense": ["tackles", "tfl", "sacks", "pd", "hurries", "ints"],
}


def career_view(careers: pl.DataFrame | None, player_id: str) -> tuple[list[dict], list[dict]]:
    """(rows, columns) for a player's career table, oldest season first. Columns carry
    their CAREER_LABELS label and definition; empty groups are left out."""
    if careers is None:
        return [], []
    rows = careers.filter(pl.col("player_id") == player_id).sort("career_season").to_dicts()
    passer = any((r.get("pass_att") or 0) > 0 for r in rows)
    phone = {"qb_starts", "pass_att"} if passer else {"plays", "tackles"}  # columns kept on phones
    cols = []
    for group, keys in CAREER_GROUPS.items():
        present = [k for k in keys if any(not _missing(r.get(k)) and r.get(k) != 0 for r in rows)]
        cols += [{"key": k, "group": group, "label": CAREER_LABELS[k][0], "definition": CAREER_LABELS[k][1],
                  "opt": k not in phone} for k in present]
    return rows, cols


def career_cell(key: str, x) -> str:
    if _missing(x):
        return "—"
    if key.endswith("_avg"):
        return f"{x:+.2f}".replace("-", "−")
    return count_str(x)


def make_router(store: Store, render: Callable[..., HTMLResponse]) -> APIRouter:
    router = APIRouter()

    def load(season: int, team: str) -> tuple[pl.DataFrame, pl.DataFrame, dict | None]:
        roster = store.roster(season)
        if roster is None:
            raise HTTPException(404, f"No roster data for {season}.")
        p, d, s = team_roster(roster, team)
        if p.height == 0:
            raise HTTPException(404, f"No {season} roster for {team!r}.")
        return p, d, s

    @router.get("/roster/{season}/{team:path}", response_class=HTMLResponse, include_in_schema=False)
    def roster_page(request: Request, season: int, team: str):
        p, d, s = load(season, team)
        rows = player_rows(p)
        s_obj = store.season(season)
        fbs = set(s_obj.ratings.filter(pl.col("fbs"))["team"].to_list()) if s_obj is not None else set()
        positions = sorted({r["position"] for r in rows if r["position"]},
                           key=lambda x: (POSITION_ORDER.index(x) if x in POSITION_ORDER else 99, x))
        return render(request, "roster.html", season=season, team=team, rows=rows, positions=positions,
                      departures=d.sort("prev_plays", "prev_tackles", descending=True, nulls_last=True).to_dicts(),
                      tiles=summary_tiles(s) if s else [], as_of=s.get("roster_as_of") if s else None,
                      in_qb=incoming_qb(s),
                      card=roster_card(store.roster(season), team), fbs_teams=fbs,
                      roster_seasons=[y for y in store.seasons() if store.roster(y) is not None])

    @router.get("/api/roster/{season}/{team:path}")
    def api_roster(season: int, team: str):
        from cfbr.web.app import clean

        p, d, s = load(season, team)
        return clean({"season": season, "team": team, "summary": s,
                      "summary_labels": {k: {"label": a, "definition": b} for k, (a, b) in ROSTER_SUMMARY_LABELS.items()},
                      "players": p.to_dicts(), "departures": d.to_dicts()})

    def load_player(season: int, player_id: str) -> tuple[dict, Roster]:
        roster = store.roster(season)
        if roster is None:
            raise HTTPException(404, f"No roster data for {season}.")
        p = roster.players.filter(pl.col("player_id") == player_id)
        if p.height == 0:
            raise HTTPException(404, f"No player {player_id!r} on a {season} roster.")
        return player_rows(p)[0], roster

    @router.get("/player/{season}/{player_id}", response_class=HTMLResponse, include_in_schema=False)
    def player_page(request: Request, season: int, player_id: str):
        player, roster = load_player(season, player_id)
        rows, cols = career_view(roster.careers, player_id)
        # Link a career team to its team page when we have that season and it was FBS then.
        fbs_by_season = {}
        for y in {r["career_season"] for r in rows} | {season}:
            s = store.season(y)
            fbs_by_season[y] = set(s.ratings.filter(pl.col("fbs"))["team"].to_list()) if s is not None else set()
        return render(request, "player.html", season=season, player=player, rows=rows, cols=cols,
                      fbs_by_season=fbs_by_season, extra_labels=PLAYER_EXTRA_LABELS,
                      has_careers=roster.careers is not None, cell=career_cell)

    @router.get("/api/player/{season}/{player_id}")
    def api_player(season: int, player_id: str):
        from cfbr.web.app import clean

        player, roster = load_player(season, player_id)
        rows, _ = career_view(roster.careers, player_id)
        drop = {"pos_order", "class_label", "height_label", "hometown"}
        return clean({"season": season, "player": {k: v for k, v in player.items() if k not in drop},
                      "career": rows,
                      "career_labels": {k: {"label": a, "definition": b} for k, (a, b) in CAREER_LABELS.items()}})

    return router
