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
OFFENSE = {"QB", "RB", "FB", "WR", "TE", "OL", "OT", "G", "C"}
DEFENSE = {"DL", "DE", "DT", "NT", "EDGE", "LB", "DB", "CB", "S"}
SPECIAL = {"PK", "K", "P", "LS", "PR"}

# Roster sections: (key, heading, stat columns). A column is (row key, label, definition,
# format, phone class). Rows carry last season's numbers (prev_*); formats: "count", "avg"
# (signed, 2 dp), "pct" (completion %), "made_att" (FGM/FGA). Every section shares the bio
# columns, so the stat columns are the only thing that differs.
SECTIONS: list[tuple[str, str, list[tuple[str, str, str, str, str]]]] = [
    ("qb", "Quarterbacks", [
        ("prev_pass_att", "Att", "Pass attempts last season", "count", ""),
        ("cmp_pct", "Cmp%", "Completion percentage last season", "pct", "opt"),
        ("prev_pass_yds", "Yds", "Passing yards last season", "count", ""),
        ("prev_pass_td", "TD", "Passing touchdowns last season", "count", "opt-xs"),
        ("prev_pass_int", "INT", "Interceptions thrown last season", "count", "opt"),
        ("prev_rush_yds", "Rush yds", "Rushing yards last season", "count", "opt"),
        ("prev_pass_ppa_avg", "PPA/att", "Passing expected points added per attempt last season", "avg", "opt"),
    ]),
    ("offense", "Offense", [
        ("prev_rush_car", "Car", "Carries last season", "count", "opt"),
        ("prev_rush_yds", "Rush yds", "Rushing yards last season", "count", ""),
        ("prev_rec", "Rec", "Receptions last season", "count", "opt"),
        ("prev_rec_yds", "Rec yds", "Receiving yards last season", "count", ""),
        ("off_td", "TD", "Rushing + receiving touchdowns last season", "count", "opt-xs"),
        ("prev_ppa_avg", "PPA/play", "Expected points added per play last season (CFBD player PPA)", "avg", "opt"),
    ]),
    ("defense", "Defense", [
        ("prev_tackles", "Tkl", "Total tackles last season", "count", ""),
        ("prev_tfl", "TFL", "Tackles for loss last season", "count", ""),
        ("prev_sacks", "Sacks", "Sacks last season", "count", "opt-xs"),
        ("prev_pd", "PD", "Passes defended last season", "count", "opt"),
        ("prev_ints", "INT", "Interceptions last season", "count", "opt"),
    ]),
    ("special", "Special teams", [
        ("fg", "FG", "Field goals made / attempted last season", "made_att", ""),
        ("prev_fg_long", "Long", "Longest field goal made last season (yards)", "count", "opt"),
        ("xp", "XP", "Extra points made / attempted last season", "made_att", "opt"),
        ("prev_punts", "Punts", "Punts last season", "count", ""),
        ("punt_avg", "Avg", "Yards per punt last season", "avg1", "opt-xs"),
        ("prev_punt_in20", "In 20", "Punts downed inside the opponent's 20 last season", "count", "opt"),
    ]),
]
SECTION_COLS = {key: cols for key, _, cols in SECTIONS}
# Production on the other side of the ball worth a note (special-teams tackles fall below this).
OTHER_SIDE_MIN = 10

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


def last_qb(players: pl.DataFrame, departures: pl.DataFrame, summary: dict | None) -> dict:
    """Last season's starting QB (most pass plays for the team, the model's definition) and where
    he is now: back, drafted, at another school, or gone. Replaces a bare Yes/No on the tiles."""
    cands: list[tuple[float, str, str]] = []
    back = players.filter((pl.col("status") == "returner") & (pl.col("prev_pass_plays") > 0))
    cands += [(float(r["prev_pass_plays"]), r["name"], "returns") for r in back.iter_rows(named=True)]
    if "prev_pass_plays" in departures.columns:  # older builds lack the column
        for r in departures.filter(pl.col("prev_pass_plays") > 0).iter_rows(named=True):
            reason = r.get("reason")
            if reason == "drafted":
                where = "drafted" + (f" (Rd {int(r['draft_round'])})" if not _missing(r.get("draft_round")) else "")
            elif r.get("to_team"):
                where = f"now at {r['to_team']}"
            elif r.get("portal_destination"):
                where = f"now at {r['portal_destination']}"
            elif r.get("in_portal"):
                where = "entered the portal"
            else:
                where = "not on an FBS roster"
            cands.append((float(r["prev_pass_plays"]), r["name"], where))
    feat = summary.get("qb_returns") if summary else None
    if not cands or (cands and max(cands)[2] == "returns" and not _missing(feat) and feat < 0.5):
        # No pass-play data for the departed starter (or none at all): fall back to the feature.
        if _missing(feat):
            return {"name": "—", "detail": None}
        return ({"name": "Back", "detail": "last season's starter is on the roster"} if feat >= 0.5
                else {"name": "Not back", "detail": "last season's starter is off the roster"})
    att, name, where = max(cands)
    return {"name": name, "detail": f"{where} · {count_str(att)} pass att. last season"}


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


def _num(r: dict, key: str) -> float:
    x = r.get(key)
    return 0.0 if _missing(x) else float(x)


def side_of(position: str | None) -> str | None:
    """'offense', 'defense', 'special', or None (ATH, missing or unknown)."""
    if position in OFFENSE:
        return "offense"
    if position in DEFENSE:
        return "defense"
    if position in SPECIAL:
        return "special"
    return None


def section_of(r: dict) -> str:
    """Roster section for a row: by position, QBs on their own; ATH or unknown positions go
    where last season's production was (offense when there was none)."""
    if r.get("position") == "QB":
        return "qb"
    side = side_of(r.get("position"))
    if side is not None:
        return side
    if _num(r, "prev_tackles") > _num(r, "prev_plays"):
        return "defense"
    if _num(r, "prev_fga") + _num(r, "prev_punts") > 0 and _num(r, "prev_plays") == 0:
        return "special"
    return "offense"


def _ratio(num, den):
    return None if _missing(num) or _missing(den) or den == 0 else num / den


def _sum_known(*xs):
    known = [x for x in xs if not _missing(x)]
    return sum(known) if known else None


def derived_stats(r: dict) -> dict:
    """Section columns computed from prev_* stats (rows from older builds lack them: all None)."""
    return {"cmp_pct": _ratio(r.get("prev_pass_cmp"), r.get("prev_pass_att")),
            "off_td": _sum_known(r.get("prev_rush_td"), r.get("prev_rec_td")),
            "fg": (r.get("prev_fgm"), r.get("prev_fga")), "xp": (r.get("prev_xpm"), r.get("prev_xpa")),
            "punt_avg": _ratio(r.get("prev_punt_yds"), r.get("prev_punts"))}


def other_side_note(r: dict, section: str) -> str | None:
    """A short note when a player's position no longer says where his numbers came from: a
    side-of-ball switch since last season, or real production on the other side (10+ plays or
    tackles; special-teams tackles fall below that)."""
    parts = []
    prev_pos = r.get("prev_position")
    if prev_pos and prev_pos != r.get("position") and side_of(prev_pos) not in (None, side_of(r.get("position"))):
        parts.append(f"moved from {prev_pos}")
    on_offense = section in ("qb", "offense")
    if not on_offense and _num(r, "prev_plays") >= OTHER_SIDE_MIN:
        parts.append(f"{count_str(r['prev_plays'])} offensive plays last season")
    if section != "defense" and _num(r, "prev_tackles") >= OTHER_SIDE_MIN:
        parts.append(f"{count_str(r['prev_tackles'])} tackles last season")
    return " · ".join(parts) or None


def stat_cell(fmt: str, x) -> tuple[str, float]:
    """(display text, sort value) for a section stat column."""
    if fmt == "made_att":
        made, att = x
        if _missing(att) or att == 0:
            return "—", -1.0
        return f"{count_str(made or 0)}/{count_str(att)}", float(made or 0) + att / 1000
    if _missing(x):
        return "—", -999.0
    if fmt == "pct":
        return f"{100 * x:.1f}%", float(x)
    if fmt == "avg":
        return f"{x:+.2f}".replace("-", "−"), float(x)
    if fmt == "avg1":
        return f"{x:.1f}", float(x)
    return count_str(x), float(x)


def section_weight(section: str, r: dict) -> float:
    """Main production measure for a section, used to order departures (most first)."""
    if section == "qb":
        return _num(r, "prev_pass_att") or _num(r, "prev_pass_plays")
    if section == "offense":
        return _num(r, "prev_plays")
    if section == "defense":
        return _num(r, "prev_tackles")
    return _num(r, "prev_fga") + _num(r, "prev_xpa") + _num(r, "prev_punts")


def _with_section(r: dict) -> dict:
    section = section_of(r)
    d = r | derived_stats(r)
    d["section"] = section
    d["cells"] = [stat_cell(fmt, d.get(key)) for key, _, _, fmt, _ in SECTION_COLS[section]]
    d["note"] = other_side_note(r, section)
    return d


def sections(rows: list[dict]) -> list[dict]:
    """Rows grouped into SECTIONS order, empty sections left out, keeping row order."""
    out = []
    for key, heading, cols in SECTIONS:
        sec_rows = [r for r in rows if r["section"] == key]
        if sec_rows:
            positions = list(dict.fromkeys(r["position"] for r in sec_rows if r["position"]))
            out.append({"key": key, "heading": heading, "cols": cols, "rows": sec_rows, "positions": positions,
                        "legend": [(label, definition) for _, label, definition, _, _ in cols],
                        "has_linemen": any(r["position"] in ("OL", "OT", "G", "C") for r in sec_rows)})
    return out


def player_rows(players: pl.DataFrame) -> list[dict]:
    order = {p: i for i, p in enumerate(POSITION_ORDER)}
    rows = []
    for r in players.iter_rows(named=True):
        rows.append(_with_section(r) | {
            "class_label": CLASS.get(r["class_year"], "—") if r["class_year"] is not None else "—",
            "height_label": height_str(r["height"]),
            "hometown": ", ".join(x for x in (r["home_city"], r["home_state"]) if x) or "—",
            "pos_order": order.get(r["position"], len(order)),
        })
    return sorted(rows, key=lambda r: (r["pos_order"], r["jersey"] if r["jersey"] is not None else 999, r["name"] or ""))


REASON_ORDER = {"drafted": 0, "transfer": 1, "portal_transfer": 2, "portal_no_school": 3, "unknown": 4}


def departure_rows(departures: pl.DataFrame) -> list[dict]:
    """Departures by section, most production in the section first, with display fields for
    why they left. Files built before draft/portal data existed have no ``reason``: fall back
    to "transfer" when on another roster, else "unknown"."""
    rows = []
    for r in departures.iter_rows(named=True):
        r = _with_section(r)
        reason = r.get("reason") or ("transfer" if r.get("to_team") else "unknown")
        cls = r.get("prev_class_year")
        draft = None
        if reason == "drafted":
            parts = [f"Rd {r['draft_round']}" if r.get("draft_round") else None,
                     f"#{r['draft_overall']} overall" if r.get("draft_overall") else None,
                     # Full franchise name ("Los Angeles Rams") when known; the city alone is ambiguous.
                     r.get("draft_nfl_team_name") or r.get("draft_nfl_team")]
            draft = " · ".join(p for p in parts if p)
        rows.append(r | {"reason": reason, "reason_order": REASON_ORDER.get(reason, 9), "draft_label": draft,
                         "prev_class_label": CLASS.get(cls, "—") if cls is not None else "—"})
    return sorted(rows, key=lambda r: (-section_weight(r["section"], r), r.get("name") or ""))


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
            "newcomers": counts.get("new", 0), "departures": d.height, "transfers_out": d["to_team"].is_not_null().sum(),
            "drafted": int((d["reason"] == "drafted").sum()) if "reason" in d.columns else None,
            "ret_offense": feature_value("off_ret_share", s.get("off_ret_share")) if s else "—",
            "last_qb": last_qb(p, d, s),
            "as_of": s.get("roster_as_of") if s else None}


# Career table column groups and the side of the ball each belongs to. A group is shown
# only if the player has a value in it; another side's group also needs real production
# there (OTHER_SIDE_MIN plays or tackles in some season), so a receiver's special-teams
# tackles don't add a defense block. Kicking and punting show whenever they have data.
CAREER_GROUPS = {
    "Passing": ("offense", ["qb_starts", "pass_cmp", "pass_att", "pass_yds", "pass_td", "pass_int", "pass_ppa_avg"]),
    "Rushing": ("offense", ["rush_car", "rush_yds", "rush_td"]),
    "Receiving": ("offense", ["rec", "rec_yds", "rec_td"]),
    "Offense": ("offense", ["plays", "ppa_avg"]),
    "Defense": ("defense", ["tackles", "tfl", "sacks", "pd", "hurries", "ints", "def_td"]),
    "Kicking": ("special", ["fgm", "fga", "fg_long", "xpm", "xpa"]),
    "Punting": ("special", ["punts", "punt_yds", "punt_in20"]),
}
# Columns kept on phones, by roster section.
CAREER_PHONE = {"qb": {"pass_att", "pass_yds", "pass_td"}, "offense": {"rush_yds", "rec_yds"},
                "defense": {"tackles", "sacks"}, "special": {"fgm", "fga", "punts"}}


def career_view(careers: pl.DataFrame | None, player_id: str,
                player: dict | None = None) -> tuple[list[dict], list[dict]]:
    """(rows, columns) for a player's career table, oldest season first. Columns carry
    their CAREER_LABELS label and definition; empty groups, and other-side groups without
    real production, are left out. ``player`` (a roster row) sets his side of the ball."""
    if careers is None:
        return [], []
    rows = careers.filter(pl.col("player_id") == player_id).sort("career_season").to_dicts()
    section = section_of(player) if player is not None else None
    if section is None:  # no roster row: the side with the most career production
        off, de = sum(_num(r, "plays") for r in rows), sum(_num(r, "tackles") for r in rows)
        section = "defense" if de > off else "offense"
    own = "offense" if section == "qb" else section
    real = {"offense": any(_num(r, "plays") >= OTHER_SIDE_MIN for r in rows),
            "defense": any(_num(r, "tackles") >= OTHER_SIDE_MIN for r in rows), "special": True}
    # CFBD's "pass" PPA for a receiver is on passes thrown to him, so Passing needs attempts.
    passer = any(_num(r, "pass_att") > 0 for r in rows)
    phone = CAREER_PHONE[section]
    cols = []
    for group, (side, keys) in CAREER_GROUPS.items():
        if (side != own and not real[side]) or (group == "Passing" and not passer):
            continue
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
        s_obj = store.season(season)
        fbs = set(s_obj.ratings.filter(pl.col("fbs"))["team"].to_list()) if s_obj is not None else set()
        return render(request, "roster.html", season=season, team=team, sections=sections(player_rows(p)),
                      departure_sections=sections(departure_rows(d)), has_reasons="reason" in d.columns,
                      portal_data="in_portal" in d.columns and d["in_portal"].is_not_null().any(),
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
        rows, cols = career_view(roster.careers, player_id, player)
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
        rows, _ = career_view(roster.careers, player_id, player)
        drop = {"pos_order", "class_label", "height_label", "hometown", "cells", "note", "fg", "xp", "cmp_pct",
                "off_td", "punt_avg"}
        return clean({"season": season, "player": {k: v for k, v in player.items() if k not in drop},
                      "career": rows,
                      "career_labels": {k: {"label": a, "definition": b} for k, (a, b) in CAREER_LABELS.items()}})

    return router
