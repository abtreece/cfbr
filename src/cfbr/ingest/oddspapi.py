"""OddsPapi (v4): bet365 spreads for pricing paper bets.

Neither CFBD nor The Odds API carries bet365 for US college football, so the bet
tracker asks OddsPapi for bet365's main spread line, but only for games it is
about to bet (odds come one fixture per request, and the free tier allows 250
requests a month). These lines can set a bet's best line; they never enter the
consensus, which stays the CFBD + Odds API market. OddsPapi's "bet365 NJ" is a
clone of the international bet365 feed, so prices may differ slightly from a
US bet365 app.

Every call is appended to ``{raw_dir}/_oddspapi_calls.jsonl``; the compact lines
used are kept under ``{raw_dir}/oddspapi/{season}/snapshots/`` (raw responses are
~2.5 MB per game and aren't kept). Betting lines are never model inputs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from cfbr.config import REPO_ROOT, Config
from cfbr.ingest.odds import BOOK_SCHEMA, match_events

log = logging.getLogger(__name__)

BASE_URL = "https://api.oddspapi.io/v4"
SPORT_ID = 14  # American football (NFL, CFL and others too; matching to CFBD games filters them out)
KEY_NAME = "ODDSPAPI_KEY"


class OddsPapiBudget(RuntimeError):
    pass


def oddspapi_key() -> str | None:
    if os.environ.get(KEY_NAME):
        return os.environ[KEY_NAME]
    keys_file = REPO_ROOT / ".keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == KEY_NAME and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def _ledger(raw_dir: Path) -> Path:
    return raw_dir / "_oddspapi_calls.jsonl"


def calls_this_month(raw_dir: Path) -> int:
    path = _ledger(raw_dir)
    if not path.exists():
        return 0
    month = datetime.now(UTC).strftime("%Y-%m")
    return sum(1 for line in path.read_text().splitlines() if line.strip() and json.loads(line)["ts"].startswith(month))


def requests_left(cfg: Config) -> int:
    o = cfg.oddspapi
    return o.monthly_quota - o.quota_reserve - calls_this_month(cfg.paths.raw_dir)


def _get(cfg: Config, path: str, params: dict):
    key = oddspapi_key()
    if key is None:
        raise RuntimeError(f"{KEY_NAME} not set (env or .keys)")
    if requests_left(cfg) <= 0:
        raise OddsPapiBudget("OddsPapi monthly budget (minus reserve) used up")
    resp = httpx.get(BASE_URL + path, params={**params, "apiKey": key}, timeout=60.0)
    with open(_ledger(cfg.paths.raw_dir), "a") as f:
        f.write(json.dumps({"ts": datetime.now(UTC).isoformat(), "endpoint": "/v4" + path, "params": params,
                            "status": resp.status_code}) + "\n")
    resp.raise_for_status()
    return resp.json()


def market_catalog(cfg: Config) -> dict[str, dict]:
    """Spread markets by marketId: handicap (from participant 1's side) and outcomeId -> "1"/"2".
    Fetched once and cached; the catalogue is static."""
    path = cfg.paths.raw_dir / "oddspapi" / "markets_spreads.json"
    if not path.exists():
        rows = _get(cfg, "/markets", {"sportId": SPORT_ID})
        rows = rows if isinstance(rows, list) else list(rows.values())
        spreads = {str(m["marketId"]): {"handicap": m["handicap"], "period": m.get("period"),
                                        "outcomes": {str(o["outcomeId"]): o["outcomeName"] for o in m.get("outcomes") or []}}
                   for m in rows if m.get("sportId") == SPORT_ID and m.get("marketType") == "spreads"}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spreads))
    return json.loads(path.read_text())


def main_spread(odds: dict, slug: str, catalog: dict[str, dict]) -> tuple[float, int, int] | None:
    """(participant-1 handicap, participant-1 price, participant-2 price) for ``slug``'s main
    full-game spread, or None when the book has no live main line for the game."""
    b = (odds.get("bookmakerOdds") or {}).get(slug)
    if not b or b.get("suspended") or b.get("bookmakerIsActive") is False:
        return None
    for mid, m in (b.get("markets") or {}).items():
        meta = catalog.get(str(mid))
        if not meta or meta.get("period") != "result":
            continue
        prices = {}
        for oid, o in (m.get("outcomes") or {}).items():
            p = (o.get("players") or {}).get("0") or {}
            side = meta["outcomes"].get(str(oid))
            if side in ("1", "2") and p.get("mainLine") and p.get("active", True) and p.get("priceAmerican") is not None:
                prices[side] = int(float(p["priceAmerican"]))
        if len(prices) == 2:
            return float(meta["handicap"]), prices["1"], prices["2"]
    return None


def pricing_lines(cfg: Config, season: int, games: pl.DataFrame, aliases: dict[str, set[str]],
                  now: datetime | None = None) -> pl.DataFrame:
    """Lines from ``cfg.oddspapi.books`` (bet365) for ``games`` (game_id, home_team, away_team,
    kickoff_ts), home perspective like CFBD's ``spread``. Fetches as many games as the
    monthly budget allows (soonest kickoffs first) and saves the compact result."""
    now = now or datetime.now(UTC)
    if games.height == 0:
        return pl.DataFrame(schema=BOOK_SCHEMA)
    # The fixture list costs a request too: don't spend it unless at least one odds call can follow.
    if requests_left(cfg) <= 1:
        log.warning("OddsPapi: %d requests left this month; not enough for the fixture list plus a game",
                    max(requests_left(cfg), 0))
        return pl.DataFrame(schema=BOOK_SCHEMA)
    catalog = market_catalog(cfg)
    fixtures = _get(cfg, "/fixtures", {"sportId": SPORT_ID, "from": f"{now:%Y-%m-%d}",
                                       "to": f"{now + timedelta(days=9):%Y-%m-%d}"})
    events = [{"id": f["fixtureId"], "commence_time": f["startTime"], "home_team": f.get("participant1Name") or "",
               "away_team": f.get("participant2Name") or ""} for f in fixtures if f.get("hasOdds")]
    matched = {gid: (fid, flipped) for fid, (gid, flipped) in match_events(events, games, aliases).items()}
    todo = games.filter(pl.col("game_id").is_in(list(matched))).sort("kickoff_ts")
    budget = requests_left(cfg)
    if todo.height > budget:
        log.warning("OddsPapi budget allows %d of %d games this run", max(budget, 0), todo.height)
        todo = todo.head(max(budget, 0))
    rows, fetched = [], []
    for gid in todo["game_id"].to_list():
        fid, flipped = matched[gid]
        odds = _get(cfg, "/odds", {"fixtureId": fid})
        fetched.append(fid)
        for slug in cfg.oddspapi.books:
            line = main_spread(odds, slug, catalog)
            if line is None:
                continue
            h, p1, p2 = line
            # Participant 1 is the home side unless the feeds disagree on home/away (neutral sites).
            rows.append({"game_id": gid, "provider": slug, "spread": -h if flipped else h,
                         "home_price": p2 if flipped else p1, "away_price": p1 if flipped else p2})
    out = pl.DataFrame(rows, schema=BOOK_SCHEMA)
    snap_dir = cfg.paths.raw_dir / "oddspapi" / str(season) / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / f"{now:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps({"fixtures": fetched, "lines": out.to_dicts()}))
    log.info("OddsPapi: %d of %d games matched, %d fetched, %d book lines; %d requests left this month",
             len(matched), games.height, len(fetched), out.height, requests_left(cfg))
    return out
