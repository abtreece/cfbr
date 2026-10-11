"""Local web app: rankings, team pages, game predictions, and model accuracy.

Reads only the files written by ``cfbr web build`` (and ``cfbr weekly``),
backtest results under ``results/``, and ``out/{season}/accuracy.csv``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import polars as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from cfbr.backtest.metrics import summarize
from cfbr.config import Config
from cfbr.web.build import FINAL, STALE_GAME, web_dir
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


def pct(x, digits: int = 1) -> str:
    return "—" if _missing(x) else f"{100 * x:.{digits}f}%"


def kickoff(ts: datetime | None) -> str:
    if ts is None:
        return "—"
    local = ts.astimezone(DISPLAY_TZ)
    return f"{local:%a %b} {local.day}, {local:%I:%M %p}".replace(" 0", " ") + f" {DISPLAY_TZ_LABEL}"


def slot_label(slot: str) -> str:
    return f"Week {int(slot[1:])}" if slot.startswith("r") else f"Postseason {int(slot[1:])}"


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
    if isinstance(obj, list):
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


def rankings_rows(s: Season, week: int, conference: str | None = None) -> list[dict]:
    r = s.ratings.filter(pl.col("fbs"))
    trend = (r.filter(pl.col("week") <= week).sort("week").group_by("team")
             .agg(trend=pl.col("rating")))
    cur = r.filter(pl.col("week") == week)
    if conference:
        cur = cur.filter(pl.col("conference") == conference)
    return cur.join(trend, on="team", how="left").sort("rank").to_dicts()


def team_games(s: Season, team: str) -> pl.DataFrame:
    """Games for ``team`` from its own perspective (positive = team favored / won)."""
    g = s.games.filter((pl.col("home_team") == team) | (pl.col("away_team") == team))
    home = pl.col("home_team") == team
    vegas_home = -pl.col("spread")
    return g.with_columns(
        is_home=home,
        opponent=pl.when(home).then(pl.col("away_team")).otherwise(pl.col("home_team")),
        opp_entity=pl.when(home).then(pl.col("away")).otherwise(pl.col("home")),
        team_points=pl.when(home).then(pl.col("home_points")).otherwise(pl.col("away_points")),
        opp_points=pl.when(home).then(pl.col("away_points")).otherwise(pl.col("home_points")),
        t_pred=pl.when(home).then(pl.col("pred_margin")).otherwise(-pl.col("pred_margin")),
        t_win_prob=pl.when(home).then(pl.col("win_prob")).otherwise(1 - pl.col("win_prob")),
        t_vegas=pl.when(home).then(vegas_home).otherwise(-vegas_home),
        t_margin=pl.when(home).then(pl.col("margin")).otherwise(-pl.col("margin")),
    ).sort("kickoff_ts")


def game_row(g: dict) -> dict:
    """Display fields for one game (home perspective in the data)."""
    pred, margin, spread = g["pred_margin"], g["margin"], g["spread"]
    home, away = g["home_team"], g["away_team"]
    fav = home if pred >= 0 else away
    row = {**g, "model_pick": f"{fav} by {abs(pred):.1f}" if abs(pred) >= 0.05 else "Toss-up", "fav_prob": max(g["win_prob"], 1 - g["win_prob"]),
           "vegas_pick": None, "edge": None, "ats": None, "model_err": None, "vegas_err": None}
    if not _missing(spread):
        vegas_home = -spread
        if vegas_home == 0:
            row["vegas_pick"] = "Pick'em"
        else:
            vfav = home if vegas_home > 0 else away
            row["vegas_pick"] = f"{vfav} −{abs(vegas_home):g}"
        diff = pred - vegas_home
        row["edge"] = (f"{abs(diff):.1f} toward {home if diff > 0 else away}") if abs(diff) >= 0.05 else "agrees"
    if g["completed"] and not _missing(margin):
        row["model_err"] = abs(pred - margin)
        if not _missing(spread):
            vegas_home = -spread
            row["vegas_err"] = abs(vegas_home - margin)
            result, pick = math.copysign(1, margin - vegas_home), math.copysign(1, pred - vegas_home)
            if margin == vegas_home:
                row["ats"] = "push"
            elif pred == vegas_home:
                row["ats"] = "no pick"
            else:
                row["ats"] = "win" if result == pick else "loss"
    return row


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


def season_summary(s: Season) -> dict:
    return summarize(s.games.filter(pl.col("completed")))


# -- app --------------------------------------------------------------------------

def create_app(cfg: Config, store: Store | None = None) -> FastAPI:
    store = store or Store(web_dir(cfg), cfg.paths.results_dir, cfg.paths.out_dir)
    app = FastAPI(title="CFB Ratings", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    env = templates.env
    env.filters |= {"num": num, "signed": signed, "pct": pct, "kickoff": kickoff, "slot_label": slot_label,
                    "q": lambda s: quote(str(s), safe="")}
    env.globals |= {"sparkline": sparkline, "FINAL": FINAL, "tz_label": DISPLAY_TZ_LABEL}

    def get_season(season: int | None) -> Season:
        season = season or store.latest()
        s = store.season(season) if season else None
        if s is None:
            if not store.seasons():
                raise HTTPException(503, "No web data yet. Run `cfbr web build` first.")
            raise HTTPException(404, f"No data for season {season}.")
        return s

    def render(request: Request, name: str, **ctx) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"seasons": store.seasons(), **ctx})

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
                      conf: str | None = None):
        s = get_season(season)
        week = pick_week(s, week)
        conf = conf or None
        rows = rankings_rows(s, week, conf)
        fbs = s.ratings.filter(pl.col("fbs"))
        conferences = sorted(c for c in fbs["conference"].unique().to_list() if c)
        lo, hi = fbs["rating"].min(), fbs["rating"].max()
        as_of = fbs.filter(pl.col("week") == week)["as_of"][0]
        return render(request, "rankings.html", s=s, season=s.meta["season"], week=week, conf=conf, rows=rows,
                      conferences=conferences, lo=lo, hi=hi, as_of=as_of, kind=season_kind(cfg, s.meta["season"]))

    @app.get("/team/{season}/{team:path}", response_class=HTMLResponse, include_in_schema=False)
    def team_page(request: Request, season: int, team: str):
        s = get_season(season)
        hist = s.ratings.filter(pl.col("team") == team).sort("week")
        if hist.height == 0:
            raise HTTPException(404, f"No team named {team!r} in {season}.")
        fbs_teams = set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
        games = [game_row(g) for g in team_games(s, team).to_dicts()]
        team_seasons = [y for y in store.seasons()
                        if (other := store.season(y)) is not None and other.ratings["team"].is_in([team]).any()]
        chart = {
            "signed": True,
            "x": [s.label(w) for w in hist["week"].to_list()],
            "series": [{"name": n, "values": hist[c].to_list()} for n, c in
                       (("Rating", "rating"), ("Offense", "offense"), ("Defense", "defense"))],
        }
        return render(request, "team.html", s=s, season=season, team=team, hist=hist.to_dicts(),
                      latest=hist.to_dicts()[-1], games=games, fbs_teams=fbs_teams, team_seasons=team_seasons,
                      chart=clean(chart), is_fbs=team in fbs_teams)

    @app.get("/games", response_class=HTMLResponse, include_in_schema=False)
    def games_page(request: Request, season: int | None = None, slot: str | None = None):
        s = get_season(season)
        order = slots(s)
        slot = slot or default_slot(s, datetime.now(UTC))
        if slot not in order:
            raise HTTPException(404, f"No games in slot {slot!r} of {s.meta['season']}.")
        g = s.games.filter(pl.col("slot") == slot)
        rows = [game_row(r) for r in g.to_dicts()]
        done = g.filter(pl.col("completed"))
        summary = summarize(done) if done.height else None
        fbs_teams = set(s.ratings.filter(pl.col("fbs"))["team"].to_list())
        return render(request, "games.html", s=s, season=s.meta["season"], slot=slot, slot_order=order, rows=rows,
                      summary=summary, fbs_teams=fbs_teams, any_final=done.height > 0)

    @app.get("/accuracy", response_class=HTMLResponse, include_in_schema=False)
    def accuracy_page(request: Request, season: int | None = None):
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
        return render(request, "accuracy.html", s=s, season=season, kind=season_kind(cfg, season),
                      summary=season_summary(s), chart=clean(chart), by_slot=by_slot.to_dicts(),
                      per_season=per_season, runs=store.runs(), log=store.accuracy_log(season), cfg=cfg)

    # -- JSON API ---------------------------------------------------------------

    @app.get("/api/seasons")
    def api_seasons():
        return [clean(store.season(y).meta) for y in store.seasons()]

    @app.get("/api/rankings")
    def api_rankings(season: int | None = None, week: int | None = None, conf: str | None = None):
        s = get_season(season)
        week = pick_week(s, week)
        rows = [{k: v for k, v in r.items() if k != "trend"} for r in rankings_rows(s, week, conf or None)]
        return clean({"season": s.meta["season"], "week": week, "label": s.label(week), "teams": rows})

    @app.get("/api/team/{season}/{team:path}")
    def api_team(season: int, team: str):
        s = get_season(season)
        hist = s.ratings.filter(pl.col("team") == team).sort("week")
        if hist.height == 0:
            raise HTTPException(404, f"No team named {team!r} in {season}.")
        games = team_games(s, team).select(
            "game_id", "kickoff_ts", "slot", "completed", "is_home", "neutral_site", "opponent", "team_points",
            "opp_points", "t_pred", "t_win_prob", "t_vegas", "t_margin")
        return clean({"season": season, "team": team, "weeks": hist.to_dicts(), "games": games.to_dicts()})

    @app.get("/api/games")
    def api_games(season: int | None = None, slot: str | None = None):
        s = get_season(season)
        slot = slot or default_slot(s, datetime.now(UTC))
        g = s.games.filter(pl.col("slot") == slot)
        if g.height == 0:
            raise HTTPException(404, f"No games in slot {slot!r} of {s.meta['season']}.")
        return clean({"season": s.meta["season"], "slot": slot, "games": g.to_dicts()})

    return app
