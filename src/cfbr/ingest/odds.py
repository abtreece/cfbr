"""The Odds API: live sportsbook spreads, with prices, for the paper-bet tracker.

CFBD's line feed carries few books (in 2026 only DraftKings and offshore Bovada),
so each weekly run also snapshots The Odds API, which covers the US books
(FanDuel, theScore Bet, BetRivers, ...). Free tier: 500 credits a month; an odds
call costs one credit per region, and ``/v4/sports`` is free. Every call is
appended to ``{raw_dir}/_odds_calls.jsonl`` and each response is kept as a
timestamped snapshot under ``{raw_dir}/odds/{season}/snapshots/``.

Betting lines are never model inputs; these feed only the paper-bet ledger.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from cfbr.config import REPO_ROOT, Config

log = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com"
SPORT = "americanfootball_ncaaf"
KEY_NAMES = ("ODDS_API_KEY", "ODD_API_KEY")

BOOK_SCHEMA = {"game_id": pl.Int64, "provider": pl.Utf8, "spread": pl.Float64,
               "home_price": pl.Int32, "away_price": pl.Int32}


class OddsQuotaExceeded(RuntimeError):
    pass


def odds_api_key() -> str | None:
    """The Odds API key from the environment or the gitignored .keys file (None if absent)."""
    for name in KEY_NAMES:
        if os.environ.get(name):
            return os.environ[name]
    keys_file = REPO_ROOT / ".keys"
    if keys_file.exists():
        for line in keys_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() in KEY_NAMES and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def _ledger(raw_dir: Path) -> Path:
    return raw_dir / "_odds_calls.jsonl"


def credits_left(raw_dir: Path, monthly_quota: int) -> int:
    """Remaining credits: the API's own count from this month's last call, else quota minus logged use."""
    path = _ledger(raw_dir)
    if not path.exists():
        return monthly_quota
    month = datetime.now(UTC).strftime("%Y-%m")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["ts"].startswith(month)]
    for r in reversed(rows):
        if r.get("remaining") is not None:
            return int(r["remaining"])
    return monthly_quota - sum(int(r.get("last") or 0) for r in rows)


def _get(cfg: Config, path: str, params: dict) -> httpx.Response:
    key = odds_api_key()
    if key is None:
        raise RuntimeError(f"The Odds API key not set ({' or '.join(KEY_NAMES)} in env or .keys)")
    resp = httpx.get(BASE_URL + path, params={**params, "apiKey": key}, timeout=60.0)
    hdr = resp.headers
    entry = {"ts": datetime.now(UTC).isoformat(), "endpoint": path, "params": params, "status": resp.status_code,
             "remaining": hdr.get("x-requests-remaining"), "used": hdr.get("x-requests-used"),
             "last": hdr.get("x-requests-last")}
    with open(_ledger(cfg.paths.raw_dir), "a") as f:
        f.write(json.dumps(entry) + "\n")
    resp.raise_for_status()
    return resp


def fetch_odds_snapshot(cfg: Config, season: int) -> Path:
    """Pull current spreads for every listed game into a new timestamped snapshot."""
    o = cfg.odds
    left = credits_left(cfg.paths.raw_dir, o.monthly_quota)
    if left - len(o.regions) < o.quota_reserve:
        raise OddsQuotaExceeded(f"The Odds API: {left} credits left, reserve {o.quota_reserve}; skipping")
    params = {"regions": ",".join(o.regions), "markets": "spreads", "oddsFormat": "american", "dateFormat": "iso"}
    resp = _get(cfg, f"/v4/sports/{SPORT}/odds", params)
    snap_dir = cfg.paths.raw_dir / "odds" / str(season) / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(resp.json()))
    tmp.replace(path)
    log.info("The Odds API: %d events, %s credits left", len(resp.json()), resp.headers.get("x-requests-remaining"))
    return path


def usage(cfg: Config) -> dict:
    """Credits used/remaining from the free ``/v4/sports`` endpoint."""
    h = _get(cfg, "/v4/sports", {}).headers
    return {"used": h.get("x-requests-used"), "remaining": h.get("x-requests-remaining")}


def latest_odds_snapshot(raw_dir: Path, season: int) -> tuple[datetime, list] | None:
    snaps = sorted((raw_dir / "odds" / str(season) / "snapshots").glob("*.json"))
    if not snaps:
        return None
    ts = datetime.strptime(snaps[-1].stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return ts, json.loads(snaps[-1].read_text())


# -- matching Odds API events to CFBD games -------------------------------------------

def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[.'’`]", "", s.lower()).replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9()]+", " ", s).split())


def team_aliases(teams_json: list) -> dict[str, set[str]]:
    """CFBD team -> normalized full names the Odds API may use ("School Mascot" for
    the school and each alternate name)."""
    out: dict[str, set[str]] = {}
    for t in teams_json:
        names = {t["school"], *(t.get("alternateNames") or [])}
        mascot = t.get("mascot") or ""
        out[t["school"]] = {norm_name(f"{n} {mascot}") for n in names if n}
    return out


def _name_score(odds_name: str, team: str, aliases: dict[str, set[str]]) -> int:
    """2 = exact "School Mascot" match, 1 = the Odds API name starts with the CFBD
    name (covers FCS teams, which have no mascot data), 0 = no match."""
    o = norm_name(odds_name)
    if o in aliases.get(team, ()):
        return 2
    t = norm_name(team)
    return 1 if o == t or o.startswith(t + " ") else 0


def match_events(events: list, games: pl.DataFrame, aliases: dict[str, set[str]],
                 window: timedelta = timedelta(hours=36)) -> dict[str, tuple[int, bool]]:
    """Odds API event id -> (CFBD game_id, flipped). ``flipped`` means the two feeds
    disagree on which team is home (neutral sites). Both teams must match and the
    kickoffs must be within ``window``; ambiguous or unmatched events are skipped."""
    g = games.select("game_id", "home_team", "away_team",
                     pl.col("kickoff_ts").dt.convert_time_zone("UTC")).to_dicts()
    out = {}
    for e in events:
        t0 = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        best: list[tuple[int, int, bool]] = []
        for r in g:
            if r["kickoff_ts"] is None or abs(r["kickoff_ts"] - t0) > window:
                continue
            for flipped, (h, a) in ((False, (r["home_team"], r["away_team"])), (True, (r["away_team"], r["home_team"]))):
                sh, sa = _name_score(e["home_team"], h, aliases), _name_score(e["away_team"], a, aliases)
                if sh and sa:
                    best.append((sh + sa, r["game_id"], flipped))
        if not best:
            log.debug("no CFBD game for %s @ %s", e["away_team"], e["home_team"])
            continue
        best.sort(reverse=True)
        if len(best) > 1 and best[1][0] == best[0][0] and best[1][1] != best[0][1]:
            log.warning("ambiguous match for %s @ %s; skipped", e["away_team"], e["home_team"])
            continue
        out[e["id"]] = (best[0][1], best[0][2])
    return out


def odds_frame(events: list, games: pl.DataFrame, aliases: dict[str, set[str]]) -> pl.DataFrame:
    """Book-level home spreads and prices (home perspective, like CFBD's ``spread``)."""
    matched = match_events(events, games, aliases)
    rows = []
    for e in events:
        if e["id"] not in matched:
            continue
        gid, flipped = matched[e["id"]]
        # CFBD's home team is the Odds API's away team when flipped.
        cfbd_home = e["away_team"] if flipped else e["home_team"]
        for b in e.get("bookmakers") or []:
            for m in b.get("markets") or []:
                if m.get("key") != "spreads":
                    continue
                oc = {o["name"]: o for o in m.get("outcomes") or []}
                home = oc.get(cfbd_home)
                away = next((o for n, o in oc.items() if n != cfbd_home), None)
                if home is None or away is None or home.get("point") is None:
                    continue
                rows.append({"game_id": gid, "provider": b["title"], "spread": float(home["point"]),
                             "home_price": home.get("price"), "away_price": away.get("price")})
    return pl.DataFrame(rows, schema=BOOK_SCHEMA)
