"""Local web app: rankings, team pages, game predictions, and model accuracy.

Reads only the files written by ``cfbr web build`` (and ``cfbr weekly``),
backtest results under ``results/``, and ``out/{season}/accuracy.csv``.
"""

from __future__ import annotations

import math
import threading
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import polars as pl
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cfbr.backtest.metrics import summarize
from cfbr.config import Config
from cfbr.web.access import premium_viewer
from cfbr.web.bets import NO_STORE, price_str, spread_str

# Finished seasons never change (a rebuild reproduces them), so shared caches may keep them an hour.
PUBLIC_HOUR = {"Cache-Control": "public, max-age=3600"}
from cfbr.web.bets import make_router as bets_router
from cfbr.web.build import FINAL, STALE_GAME, web_dir
from cfbr.web.odds import make_router as odds_router
from cfbr.web.odds import prob
from cfbr.web.roster import count_str, roster_card
from cfbr.web.roster import make_router as roster_router
from cfbr.web.store import Season, Store

HERE = Path(__file__).parent
DISPLAY_TZ = ZoneInfo("America/Chicago")
DISPLAY_TZ_LABEL = "CT"


# -- formatting -----------------------------------------------------------------

def _missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def num(x, digits: int = 1) -> str:
    return "—" if _missing(x) else f"{x:.{digits}f}"


def signed(x, digits: int = 1) -> str:
    return "—" if _missing(x) else f"{x:+.{digits}f}".replace("-", "−")


def money(x, signed: bool = True) -> str:
    if _missing(x):
        return "—"
    s = f"${abs(x):,.0f}" if round(x, 2) == round(x) else f"${abs(x):,.2f}"
    return (("+" if x > 0 else "−" if x < 0 else "") + s) if signed else s


def pct(x, digits: int = 1) -> str:
    return "—" if _missing(x) else f"{100 * x:.{digits}f}%".replace("-", "−")


def kickoff(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    local = ts.astimezone(DISPLAY_TZ)
    return f"{local:%a %b} {local.day}, {local:%I:%M %p}".replace(" 0", " ") + f" {DISPLAY_TZ_LABEL}"


def short_ts(ts: datetime | None) -> str:
    """Compact local timestamp for narrow columns, e.g. "Oct 11, 11:24 AM"."""
    if ts is None:
        return "—"
    local = ts.astimezone(DISPLAY_TZ)
    return f"{local:%b} {local.day}, {local:%I:%M %p}".replace(" 0", " ")


def iso_ts(s: str | None) -> str:
    """An ISO timestamp from meta.json (e.g. ``built_at``) in display time."""
    if not s:
        return "—"
    try:
        ts = datetime.fromisoformat(s)
    except ValueError:
        return s
    return short_ts(ts if ts.tzinfo else ts.replace(tzinfo=UTC)) + f" {DISPLAY_TZ_LABEL}"


def slot_label(slot: str) -> str:
    return f"Week {int(slot[1:])}" if slot.startswith("r") else f"Postseason {int(slot[1:])}"


POLL_LABELS = {"ap_rank": "AP", "coaches_rank": "Coaches", "cfp_rank": "CFP"}


def poll_note(s: Season, week: int) -> list[str]:
    """Which poll each column shows, e.g. "AP: week 7 poll"; polls not yet released are left out."""
    used = s.meta.get("polls", {}).get(str(week), {})
    out = []
    for col, label in POLL_LABELS.items():
        wk = used.get(col)
        if wk is None:
            continue
        if wk == "final":
            out.append(f"{label}: final poll")
        elif week == FINAL:
            out.append(f"{label}: week {wk} (last released)")
        elif wk == week + 1:
            out.append(f"{label}: {'preseason poll' if wk == 1 else f'week {wk} poll'}")
        else:
            out.append(f"{label}: week {wk} poll (week {week + 1} not released yet)")
    return out


def sparkline(values: list[float | None], lo: float, hi: float, w: int = 72, h: int = 18) -> str:
    """SVG polyline points on a shared scale so trends are comparable across rows."""
    pts = [v for v in values if not _missing(v)]
    if len(pts) < 2 or hi <= lo:
        return ""
    step = w / (len(pts) - 1)
    return " ".join(f"{i * step:.1f},{h - (v - lo) / (hi - lo) * h:.1f}" for i, v in enumerate(pts))


def clean(obj):
    """JSON-safe: NaN -> null, datetimes -> ISO strings."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# -- view models ------------------------------------------------------------------

def season_kind(cfg: Config, season: int) -> str:
    if cfg.seasons.tuning[0] <= season <= cfg.seasons.tuning[1]:
        return "tuning"
    if cfg.seasons.holdout[0] <= season <= cfg.seasons.holdout[1]:
        return "holdout"
    return "live"


def default_week(s: Season) -> int:
    """Latest point in the season (the Final table once a season is over)."""
    return s.weeks[-1]


TOP25 = "top25"  # pseudo-conference on the rankings filter: the model's top 25


def rankings_rows(s: Season, week: int, conference: str | None = None) -> list[dict]:
    r = s.ratings.filter(pl.col("fbs"))
    trend = (r.filter(pl.col("week") <= week).sort("week").group_by("team")
             .agg(trend=pl.col("rating")))
    cur = r.filter(pl.col("week") == week)
    if conference == TOP25:
        cur = cur.filter(pl.col("rank") <= 25)
    elif conference:
        cur = cur.filter(pl.col("conference") == conference)
    # Model rank minus AP rank: positive means the poll has the team higher than the model does.
    if "ap_rank" in cur.columns:
        cur = cur.with_columns(vs_ap=(pl.col("rank") - pl.col("ap_rank")).cast(pl.Int32))
    return cur.join(trend, on="team", how="left").sort("rank").to_dicts()


def ap_disagreements(rows: list[dict], n: int = 3) -> dict:
    """Where the model and the AP poll differ most: ``over`` (AP rank well above the model's),
    ``under`` (model well above the AP rank) and ``unranked`` (model top 25, no AP vote)."""
    ranked = [r for r in rows if r.get("ap_rank") is not None and r.get("vs_ap") is not None]
    over = sorted((r for r in ranked if r["vs_ap"] > 0), key=lambda r: (-r["vs_ap"], r["ap_rank"]))[:n]
    under = sorted((r for r in ranked if r["vs_ap"] < 0), key=lambda r: (r["vs_ap"], r["rank"]))[:n]
    unranked = [r for r in rows if r.get("ap_rank") is None and r.get("rank") is not None and r["rank"] <= 25][:n]
    return {"over": over, "under": under, "unranked": unranked}


EXPLAIN = (("o_ppa", "Offense: points per play", "opponent-adjusted PPA, {} in FBS"),
           ("d_ppa", "Defense: points per play", "opponent-adjusted PPA allowed, {} in FBS"),
           ("o_sr", "Offense: success rate", "opponent-adjusted, {} in FBS"),
           ("d_sr", "Defense: success rate", "opponent-adjusted, {} in FBS"))


def explain_rating(s: Season, team: str, week: int) -> list[dict]:
    """The four efficiency terms behind a rating with the team's league percentile for each.
    All four are signed deviations from the FBS average, where higher is better (defense
    is already from the defending team's side)."""
    fbs = s.ratings.filter(pl.col("fbs") & (pl.col("week") == week))
    me = s.ratings.filter((pl.col("team") == team) & (pl.col("week") == week))
    if me.height == 0 or fbs.height == 0:
        return []
    out = []
    for col, label, sub in EXPLAIN:
        v = me[col][0]
        if _missing(v):
            return []  # withheld (free viewer on a live season)
        pct_rank = float((fbs[col] <= v).mean())
        out.append({"label": label, "value": v, "pct": pct_rank, "kind": "ppa" if col.endswith("ppa") else "sr",
                    "sub": sub.format(ordinal(int(round(100 * pct_rank))) + " percentile")})
    return out


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def schedule_strength(s: Season, week: int) -> pl.DataFrame:
    """Per FBS team: mean opponent rating (at ``week``) over the full schedule and over the
    games played, with the schedule rank (1 = hardest). Pooled FCS opponents use their pool's rating."""
    r = s.ratings.filter(pl.col("week") == week).select("team", "rating", "rank")
    fbs = set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
    g = s.games
    sides = pl.concat([
        g.select(team=pl.col("home_team"), opp=pl.col("away"), done=pl.col("completed")),
        g.select(team=pl.col("away_team"), opp=pl.col("home"), done=pl.col("completed")),
    ]).filter(pl.col("team").is_in(list(fbs)))
    sides = sides.join(r.rename({"team": "opp", "rating": "opp_rating"}).select("opp", "opp_rating"), on="opp", how="left")
    out = sides.group_by("team").agg(
        sos=pl.col("opp_rating").mean(),
        sos_played=pl.col("opp_rating").filter(pl.col("done")).mean(),
        n_games=pl.len(),
    )
    return out.with_columns(sos_rank=pl.col("sos").rank("ordinal", descending=True).cast(pl.Int32))


def resume(s: Season, team: str, week: int) -> dict | None:
    """Strength of schedule and résumé facts for the team page: SOS rank, best win, worst
    loss and the record against the model's top 25 as of ``week``. None when withheld."""
    sos = schedule_strength(s, week)
    me = sos.filter(pl.col("team") == team)
    if me.height == 0 or _missing(me["sos"][0]):
        return None
    r = s.ratings.filter(pl.col("week") == week).select("team", "rating", "rank")
    g = (team_games(s, team).filter(pl.col("completed") & pl.col("t_margin").is_not_null())
         .join(r.rename({"team": "opp_entity", "rating": "opp_rating", "rank": "opp_rank"}), on="opp_entity", how="left"))
    wins = g.filter(pl.col("t_margin") > 0).sort("opp_rating", descending=True, nulls_last=True)
    losses = g.filter(pl.col("t_margin") < 0).sort("opp_rating", nulls_last=True)
    top = g.filter(pl.col("opp_rank") <= 25)
    row = me.row(0, named=True)
    return {
        "sos": row["sos"], "sos_played": row["sos_played"], "sos_rank": row["sos_rank"], "n_fbs": sos.height,
        "best_win": wins.row(0, named=True) if wins.height else None,
        "worst_loss": losses.row(0, named=True) if losses.height else None,
        "top25_w": int((top["t_margin"] > 0).sum()), "top25_l": int((top["t_margin"] < 0).sum()),
    }


def team_games(s: Season, team: str) -> pl.DataFrame:
    """Games for ``team`` from its own perspective (positive = team favored / won)."""
    g = s.games.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
    home = pl.col("home_team") == team
    vegas_home = -pl.col("spread")
    return g.with_columns(
        is_home=home,
        opponent=pl.when(home).then(pl.col("away_team")).otherwise(pl.col("home_team")),
        opp_entity=pl.when(home).then(pl.col("away")).otherwise(pl.col("home")),
        opp_ap_rank=pl.when(home).then(pl.col("away_ap_rank")).otherwise(pl.col("home_ap_rank")),
        team_points=pl.when(home).then(pl.col("home_points")).otherwise(pl.col("away_points")),
        opp_points=pl.when(home).then(pl.col("away_points")).otherwise(pl.col("home_points")),
        t_pred=pl.when(home).then(pl.col("pred_margin")).otherwise(-pl.col("pred_margin")),
        t_win_prob=pl.when(home).then(pl.col("win_prob")).otherwise(1 - pl.col("win_prob")),
        t_vegas=pl.when(home).then(vegas_home).otherwise(-vegas_home),
        t_margin=pl.when(home).then(pl.col("margin")).otherwise(-pl.col("margin")),
    ).sort("kickoff_ts")


def game_status(g: dict, now: datetime | None, live: bool) -> str:
    """``final``, ``upcoming``, or for a game still not completed STALE_GAME after its
    kickoff: ``postponed`` while the season is in progress (it may still be made up),
    ``cancelled`` once the season is over."""
    if g.get("completed"):
        return "final"
    ts = g.get("kickoff_ts")
    if now is not None and ts is not None and ts < now - STALE_GAME:
        return "postponed" if live else "cancelled"
    return "upcoming"


def game_row(g: dict, now: datetime | None = None, live: bool = True, threshold: float = 3.0) -> dict:
    """Display fields for one game (home perspective in the data). Model fields
    are None when the prediction is withheld (see :func:`public_view`). ``alert`` names
    an upset alert: the model takes the other side of the Vegas favorite, or differs from
    the closing line by ``threshold`` points or more (the paper-bet threshold)."""
    pred, margin, spread = g["pred_margin"], g["margin"], g["spread"]
    home, away = g["home_team"], g["away_team"]
    has_pred = not _missing(pred)
    row = {**g, "model_pick": None, "fav_prob": None, "vegas_pick": None, "edge": None, "ats": None,
           "model_err": None, "vegas_err": None, "status": game_status(g, now, live), "alert": None}
    if has_pred:
        fav = home if pred >= 0 else away
        row["model_pick"] = f"{fav} by {abs(pred):.1f}" if abs(pred) >= 0.05 else "Toss-up"
        row["fav_prob"] = max(g["win_prob"], 1 - g["win_prob"])
    if not _missing(spread):
        vegas_home = -spread
        if vegas_home == 0:
            row["vegas_pick"] = "Pick'em"
        else:
            vfav = home if vegas_home > 0 else away
            row["vegas_pick"] = f"{vfav} −{abs(vegas_home):g}"
        if has_pred:
            diff = pred - vegas_home
            row["edge"] = (f"{abs(diff):.1f} toward {home if diff > 0 else away}") if abs(diff) >= 0.05 else "agrees"
            win_prob = g.get("win_prob")
            fav_prob = None if _missing(win_prob) else (win_prob if vegas_home > 0 else 1 - win_prob)
            if vegas_home != 0 and fav_prob is not None and fav_prob < 0.5:
                dog = away if vegas_home > 0 else home
                row["alert"] = f"{dog} to win outright ({100 * (1 - fav_prob):.0f}%)"
            elif abs(diff) >= threshold:
                row["alert"] = f"{home if diff > 0 else away} by {abs(diff):.1f} more than the line"
    if g["completed"] and not _missing(margin):
        if not _missing(spread):
            row["vegas_err"] = abs(-spread - margin)
        if has_pred:
            row["model_err"] = abs(pred - margin)
            if not _missing(spread):
                vegas_home = -spread
                result, pick = math.copysign(1, margin - vegas_home), math.copysign(1, pred - vegas_home)
                if margin == vegas_home:
                    row["ats"] = "push"
                elif pred == vegas_home:
                    row["ats"] = "no pick"
                else:
                    row["ats"] = "win" if result == pick else "loss"
    return row


# Model output: anything here, for any game or week of a season in progress, can be
# turned back into the picks (see cfbr.web.access). Free viewers get these as nulls.
MODEL_RATING_COLS = ["rank", "rating", "offense", "defense", "prior_share", "rating_change", "rank_change",
                     "o_ppa", "d_ppa", "o_sr", "d_sr"]
MODEL_GAME_COLS = ["pred_margin", "win_prob", "prior_pred", "massey_pred", "sigma", "prior_share_h", "prior_share_a"]


def _null_out(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns([pl.lit(None, df.schema[c]).alias(c) for c in cols if c in df.columns])


def public_view(s: Season) -> Season:
    """A live season as a free viewer may see it: schedule, scores, lines, polls and
    records, with every model column nulled before any page or API code sees it."""
    return Season(s.meta, _null_out(s.ratings, MODEL_RATING_COLS), _null_out(s.games, MODEL_GAME_COLS))


def slots(s: Season) -> list[str]:
    return s.games.select("slot", "slot_order").unique().sort("slot_order")["slot"].to_list()


def default_slot(s: Season, now: datetime) -> str:
    """The next slot with games still to play in a live season, else the last one."""
    order = slots(s)
    if s.meta.get("live"):
        g = s.games
        cutoff = pl.lit(now - STALE_GAME).cast(g.schema["kickoff_ts"])
        upcoming = g.filter(~pl.col("completed") & (pl.col("kickoff_ts") > cutoff))
        if upcoming.height:
            return upcoming.sort("slot_order")["slot"][0]
    return order[-1]


def accuracy_by_slot(s: Season) -> pl.DataFrame:
    """Per-week MAE on completed games that have a closing line (same games for all three).
    Postseason windows are pooled into one point; some hold only a game or two."""
    d = s.games.filter(pl.col("completed") & pl.col("margin").is_not_null() & pl.col("spread").is_not_null())
    post = pl.col("slot").str.starts_with("p")
    d = d.with_columns(label=pl.when(post).then(pl.lit("Postseason"))
                       .otherwise(pl.format("Week {}", pl.col("slot").str.slice(1).cast(pl.Int32))))
    return (d.group_by("label").agg(
        slot=pl.col("slot").sort_by("slot_order").first(),
        slot_order=pl.col("slot_order").min(),
        n=pl.len(),
        model=(pl.col("pred_margin") - pl.col("margin")).abs().mean(),
        vegas=(-pl.col("spread") - pl.col("margin")).abs().mean(),
        massey=(pl.col("massey_pred") - pl.col("margin")).abs().mean(),
    ).sort("slot_order"))


_SUMMARIES: dict[int, tuple[Season, dict]] = {}
_SUMMARIES_LOCK = threading.Lock()


def season_summary(s: Season) -> dict:
    """Aggregate accuracy for a season's completed games, computed once per loaded Season
    (the Store hands out the same object until the files change)."""
    key = id(s)
    with _SUMMARIES_LOCK:
        hit = _SUMMARIES.get(key)
        if hit is not None and hit[0] is s:
            return hit[1]
    out = summarize(s.games.filter(pl.col("completed")))
    with _SUMMARIES_LOCK:
        _SUMMARIES[key] = (s, out)
        if len(_SUMMARIES) > 64:  # keep the cache bounded; entries for dropped Seasons are just stale ids
            for k in list(_SUMMARIES)[:-32]:
                del _SUMMARIES[k]
    return out


# -- app --------------------------------------------------------------------------

def create_app(cfg: Config, store: Store | None = None) -> FastAPI:
    store = store or Store(web_dir(cfg), cfg.paths.results_dir, cfg.paths.out_dir)
    app = FastAPI(title="CFB Ratings", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    env = templates.env
    # Templates load once per process, like the Python code. With auto-reload, a template
    # landing on main went live immediately against the old code still running (e.g. a
    # filter the old process didn't define) and broke the page until a restart.
    env.auto_reload = False
    app.state.templates = templates
    env.filters |= {"num": num, "signed": signed, "pct": pct, "money": money, "kickoff": kickoff, "short_ts": short_ts, "slot_label": slot_label,
                    "iso_ts": iso_ts,
                    "spread": spread_str, "price": price_str, "count": count_str,
                    "q": lambda s: quote(str(s), safe="")}
    env.globals |= {"sparkline": sparkline, "FINAL": FINAL, "tz_label": DISPLAY_TZ_LABEL, "prob": prob}

    def get_season(season: int | None) -> Season:
        season = season or store.latest()
        s = store.season(season) if season else None
        if s is None:
            if not store.seasons():
                raise HTTPException(503, "No web data yet. Run `cfbr web build` first.")
            if season and store.rebuilding(season):
                raise HTTPException(503, f"Season {season} is being rebuilt; try again in a moment.")
            raise HTTPException(404, f"No data for season {season}.")
        return s

    def render(request: Request, name: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"seasons": store.seasons(), "locked": False, **ctx})

    def view(s: Season, premium: bool) -> tuple[Season, bool]:
        """The season as this viewer may see it, and whether model output was withheld."""
        locked = bool(s.meta.get("live")) and not premium
        return (public_view(s) if locked else s), locked

    def private(resp: Response, s: Season, shared: bool = True) -> Response:
        # A live season's pages differ for free and premium viewers; never let a shared cache mix them.
        # A finished season's pages are the same for everyone and change only on a rebuild.
        if s.meta.get("live") or not shared:
            resp.headers.update(NO_STORE)
        else:
            resp.headers.update(PUBLIC_HOUR)
        return resp

    def latest_free_season() -> int | None:
        done = [y for y in store.seasons() if (x := store.season(y)) is not None and not x.meta.get("live")]
        return done[-1] if done else None

    def pick_week(s: Season, week: int | None) -> int:
        if week is None:
            return default_week(s)
        if week not in s.weeks:
            raise HTTPException(404, f"No ratings for {s.meta['season']} week {week}.")
        return week

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return templates.TemplateResponse(request, "error.html", {"seasons": store.seasons(), "status": exc.status_code,
                                                                  "detail": exc.detail}, status_code=exc.status_code)

    # -- pages ----------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index():
        return RedirectResponse("/rankings")

    @app.get("/rankings", response_class=HTMLResponse, include_in_schema=False)
    def rankings_page(request: Request, season: int | None = None, week: int | None = None,
                      conf: str | None = None, premium: bool = Depends(premium_viewer)):
        live_season = None
        if season is None and not premium:
            # Free viewers land on the latest finished season; the live one is premium.
            latest = store.season(store.latest()) if store.latest() else None
            if latest is not None and latest.meta.get("live"):
                live_season, season = latest.meta["season"], latest_free_season()
        s = get_season(season)
        if view(s, premium)[1]:
            return private(render(request, "rankings.html", s=s, season=s.meta["season"], locked=True,
                                  free_season=latest_free_season()), s)
        week = pick_week(s, week)
        conf = conf or None
        rows = rankings_rows(s, week, conf)
        fbs = s.ratings.filter(pl.col("fbs"))
        conferences = sorted(c for c in fbs["conference"].unique().to_list() if c)
        lo, hi = fbs["rating"].min(), fbs["rating"].max()
        as_of = fbs.filter(pl.col("week") == week)["as_of"][0]
        show_cfp = fbs.filter(pl.col("week") == week)["cfp_rank"].is_not_null().any()
        show_ap = fbs.filter(pl.col("week") == week)["ap_rank"].is_not_null().any()
        resp = render(request, "rankings.html", s=s, season=s.meta["season"], week=week, conf=conf, rows=rows,
                      polls=poll_note(s, week), show_cfp=show_cfp, show_ap=show_ap,
                      ap=ap_disagreements(rows if not conf else rankings_rows(s, week)) if show_ap else None,
                      conferences=conferences, lo=lo, hi=hi, as_of=as_of, kind=season_kind(cfg, s.meta["season"]),
                      live_season=live_season)
        # The landing page with a live-season notice is viewer-specific (free viewers only see it).
        return private(resp, s, shared=live_season is None)

    @app.get("/team/{season}/{team:path}", response_class=HTMLResponse, include_in_schema=False)
    def team_page(request: Request, season: int, team: str, premium: bool = Depends(premium_viewer)):
        s, locked = view(get_season(season), premium)
        hist = s.ratings.filter(pl.col("team") == team).sort("week")
        if hist.height == 0:
            raise HTTPException(404, f"No team named {team!r} in {season}.")
        fbs_teams = set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
        now, live = datetime.now(UTC), bool(s.meta.get("live"))
        games = [game_row(g, now, live, cfg.bets.threshold) for g in team_games(s, team).to_dicts()]
        team_seasons = [y for y in store.seasons()
                        if (other := store.season(y)) is not None and other.ratings["team"].is_in([team]).any()]
        last_week = hist["week"][-1]
        chart = {
            "signed": True,
            "x": [s.label(w) for w in hist["week"].to_list()],
            "series": [{"name": n, "values": hist[c].to_list()} for n, c in
                       (("Rating", "rating"), ("Offense", "offense"), ("Defense", "defense"))],
        }
        resp = render(request, "team.html", s=s, season=season, team=team, hist=hist.to_dicts(),
                      latest=hist.to_dicts()[-1], games=games, fbs_teams=fbs_teams, team_seasons=team_seasons,
                      show_cfp=hist["cfp_rank"].is_not_null().any(), polls=poll_note(s, hist["week"][-1]),
                      chart=None if locked else clean(chart), is_fbs=team in fbs_teams, locked=locked,
                      explain=[] if locked else explain_rating(s, team, last_week),
                      resume=None if locked or team not in fbs_teams else resume(s, team, last_week),
                      roster=roster_card(store.roster(season), team))
        return private(resp, s)

    @app.get("/games", response_class=HTMLResponse, include_in_schema=False)
    def games_page(request: Request, season: int | None = None, slot: str | None = None,
                   premium: bool = Depends(premium_viewer)):
        s, locked = view(get_season(season), premium)
        order = slots(s)
        slot = slot or default_slot(s, datetime.now(UTC))
        if slot not in order:
            raise HTTPException(404, f"No games in slot {slot!r} of {s.meta['season']}.")
        g = s.games.filter(pl.col("slot") == slot)
        now = datetime.now(UTC)
        rows = [game_row(r, now, bool(s.meta.get("live")), cfg.bets.threshold) for r in g.to_dicts()]
        done = g.filter(pl.col("completed"))
        summary = summarize(done) if done.height and not locked else None
        fbs_teams = set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
        resp = render(request, "games.html", s=s, season=s.meta["season"], slot=slot, slot_order=order, rows=rows,
                      summary=summary, fbs_teams=fbs_teams, any_final=done.height > 0, locked=locked,
                      alerts=[] if locked else [r for r in rows if r["alert"]], threshold=cfg.bets.threshold)
        return private(resp, s)

    @app.get("/matchup", response_class=HTMLResponse, include_in_schema=False)
    def matchup_page(request: Request, season: int | None = None, week: int | None = None,
                     premium: bool = Depends(premium_viewer)):
        """Head-to-head tool: two ratings plus the calibrated home edge, priced in the browser
        from numbers already in the build (meta.calibration); the server fits nothing."""
        s, locked = view(get_season(season), premium)
        week = pick_week(s, week)
        cal = s.meta.get("calibration")
        teams = []
        if not locked:
            fbs = s.ratings.filter(pl.col("fbs") & (pl.col("week") == week)).sort("team")
            teams = fbs.select("team", "conference", "rating", "rank", "prior_share").to_dicts()
        resp = render(request, "matchup.html", s=s, season=s.meta["season"], week=week, locked=locked,
                      cal=clean(cal) if cal else None, teams=clean(teams),
                      kind=season_kind(cfg, s.meta["season"]))
        return private(resp, s)

    @app.get("/accuracy", response_class=HTMLResponse, include_in_schema=False)
    def accuracy_page(request: Request, season: int | None = None):
        # Aggregate accuracy only (MAE, ATS record), which can't be turned back into picks,
        # so it is free even for a live season.
        s = get_season(season)
        season = s.meta["season"]
        by_slot = accuracy_by_slot(s)
        chart = {
            "signed": False,
            "x": by_slot["label"].to_list(),
            "n": by_slot["n"].to_list(),
            "series": [{"name": n, "values": by_slot[c].to_list()} for n, c in
                       (("Model", "model"), ("Closing line", "vegas"), ("Massey baseline", "massey"))],
        }
        per_season = []
        for y in store.seasons():
            other = store.season(y)
            if other is not None:
                per_season.append({"season": y, "kind": season_kind(cfg, y), **season_summary(other)})
        resp = render(request, "accuracy.html", s=s, season=season, kind=season_kind(cfg, season),
                      summary=season_summary(s), chart=clean(chart), by_slot=by_slot.to_dicts(),
                      per_season=per_season, runs=store.runs(), log=store.accuracy_log(season), cfg=cfg)
        # Aggregate only, the same for every viewer; a finished season's page may be cached.
        return private(resp, s) if not s.meta.get("live") else resp

    # -- JSON API ---------------------------------------------------------------

    @app.get("/api/seasons")
    def api_seasons():
        # A season can vanish (or be mid-rebuild) between the listing and the read: skip it.
        return [clean(s.meta) for y in store.seasons() if (s := store.season(y)) is not None]

    @app.get("/api/rankings")
    def api_rankings(response: Response, season: int | None = None, week: int | None = None, conf: str | None = None,
                     premium: bool = Depends(premium_viewer)):
        s = get_season(season)
        if view(s, premium)[1]:
            raise HTTPException(402, f"{s.meta['season']} ratings are for subscribers until the season ends.")
        private(response, s)
        week = pick_week(s, week)
        rows = [{k: v for k, v in r.items() if k != "trend"} for r in rankings_rows(s, week, conf or None)]
        return clean({"season": s.meta["season"], "week": week, "label": s.label(week), "teams": rows})

    @app.get("/api/team/{season}/{team:path}")
    def api_team(response: Response, season: int, team: str, premium: bool = Depends(premium_viewer)):
        s, _ = view(get_season(season), premium)
        private(response, s)
        hist = s.ratings.filter(pl.col("team") == team).sort("week")
        if hist.height == 0:
            raise HTTPException(404, f"No team named {team!r} in {season}.")
        games = team_games(s, team).select(
            "game_id", "kickoff_ts", "slot", "completed", "is_home", "neutral_site", "opponent", "opp_ap_rank", "team_points",
            "opp_points", "t_pred", "t_win_prob", "t_vegas", "t_margin")
        return clean({"season": season, "team": team, "weeks": hist.to_dicts(), "games": games.to_dicts()})

    @app.get("/api/games")
    def api_games(response: Response, season: int | None = None, slot: str | None = None,
                  premium: bool = Depends(premium_viewer)):
        s, _ = view(get_season(season), premium)
        private(response, s)
        slot = slot or default_slot(s, datetime.now(UTC))
        g = s.games.filter(pl.col("slot") == slot)
        if g.height == 0:
            raise HTTPException(404, f"No games in slot {slot!r} of {s.meta['season']}.")
        return clean({"season": s.meta["season"], "slot": slot, "games": g.to_dicts()})

    # Paper bets: a separate router behind its own access gate (see cfbr.web.access).
    app.include_router(bets_router(cfg, store, render))
    # Rosters: public facts, free for everyone (see cfbr.web.roster).
    app.include_router(roster_router(store, render))
    # Season-simulation odds: premium while the season is live (see cfbr.web.odds).
    app.include_router(odds_router(store, render))

    return app
