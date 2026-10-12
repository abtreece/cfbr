"""Real bets the owner placed, tracked apart from the paper ledger.

``out/{season}/bets_real.csv`` is append-only like ``bets.csv`` (rows are only ever added, and
an open row only ever gains its result), but it is a separate record: it never enters the paper
ledger, its hash chain, or the pre-registered 500-bet test. Each bet is graded at its own number
and price, and its CLV is measured against the same close the paper bets use
(``bets.run.closing_lines``).

Stakes are dollars risked ("$10 at −110" risks $10 to win $9.09), not the paper ledger's units.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from cfbr.bets.ledger import clv, grade
from cfbr.ingest.odds import norm_name

REAL_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Int64, "season": pl.Int32, "week": pl.Int32, "kickoff_ts": pl.Utf8,
    "home_team": pl.Utf8, "away_team": pl.Utf8,
    # Terms: never change once recorded. ``line`` is the home spread, like the paper ledger.
    "side": pl.Utf8, "team": pl.Utf8, "line": pl.Float64, "odds": pl.Int32, "risk_dollars": pl.Float64,
    "book": pl.Utf8, "placed_at": pl.Utf8, "recorded_at": pl.Utf8,
    # Result: filled once, when the game is final.
    "status": pl.Utf8, "home_points": pl.Int32, "away_points": pl.Int32, "dollars": pl.Float64,
    "closing_line": pl.Float64, "closing_source": pl.Utf8, "clv": pl.Float64,
}


class RealBetError(ValueError):
    """A bet that can't be recorded as given (no matching game, after kickoff, duplicate)."""


def ledger_path(out_dir: Path, season: int) -> Path:
    return out_dir / str(season) / "bets_real.csv"


def read_real(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=REAL_SCHEMA)
    df = pl.read_csv(path, schema_overrides=REAL_SCHEMA)
    missing = [c for c in REAL_SCHEMA if c not in df.columns]
    df = df.with_columns(pl.lit(None, dtype=REAL_SCHEMA[c]).alias(c) for c in missing)
    return df.select(pl.col(c).cast(t) for c, t in REAL_SCHEMA.items())


def write_real(df: pl.DataFrame, path: Path) -> None:
    """Temp file + rename, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.sort("kickoff_ts", "placed_at").write_csv(tmp)
    tmp.replace(path)


def win_dollars(odds: int, risked: float) -> float:
    """What a winning bet of ``risked`` dollars pays at American ``odds``."""
    return risked * 100 / -odds if odds < 0 else risked * odds / 100


def find_game(games: pl.DataFrame, team: str, placed_at: datetime) -> dict:
    """``team``'s next game after ``placed_at`` (from ``games``: game_id, home_team, away_team,
    kickoff_ts in UTC), matched on normalized name. Raises if the team has no game left, or if
    the name is ambiguous (two teams' games kick off next at the same time)."""
    want = norm_name(team)
    rows = [g for g in games.to_dicts()
            if g["kickoff_ts"] is not None and g["kickoff_ts"] > placed_at
            and want in (norm_name(g["home_team"] or ""), norm_name(g["away_team"] or ""))]
    if not rows:
        raise RealBetError(f"no {team!r} game after {placed_at:%Y-%m-%d %H:%M} UTC")
    first = min(g["kickoff_ts"] for g in rows)
    nxt = [g for g in rows if g["kickoff_ts"] == first]
    if len(nxt) > 1:
        raise RealBetError(f"{team!r} matches {len(nxt)} games at {first:%Y-%m-%d %H:%M} UTC; pass --game-id")
    return nxt[0]


def new_bet(game: dict, team: str, team_line: float, odds: int, risked: float, book: str | None,
            placed_at: datetime, recorded_at: datetime, season: int, week: int | None) -> dict:
    """One row. ``team_line`` is from the bettor's side (Cincinnati +1.5); stored as the home spread."""
    kickoff = game["kickoff_ts"]
    if placed_at >= kickoff:
        raise RealBetError(f"placed at {placed_at:%Y-%m-%d %H:%M} UTC, at or after kickoff ({kickoff:%Y-%m-%d %H:%M} UTC)")
    if odds == 0 or -100 < odds < 100:
        raise RealBetError(f"{odds} is not an American price")
    if risked <= 0:
        raise RealBetError("stake must be positive")
    side = "home" if norm_name(team) == norm_name(game["home_team"]) else "away"
    return {"game_id": game["game_id"], "season": season, "week": week, "kickoff_ts": kickoff.isoformat(),
            "home_team": game["home_team"], "away_team": game["away_team"],
            "side": side, "team": game[f"{side}_team"], "line": team_line if side == "home" else -team_line,
            "odds": odds, "risk_dollars": risked, "book": book, "placed_at": placed_at.isoformat(),
            "recorded_at": recorded_at.isoformat(), "status": "open"}


def add_real(ledger: pl.DataFrame, bet: dict) -> pl.DataFrame:
    """Append ``bet``; refuse an exact repeat (same game, side and placed_at)."""
    dup = ledger.filter((pl.col("game_id") == bet["game_id"]) & (pl.col("side") == bet["side"])
                        & (pl.col("placed_at") == bet["placed_at"]))
    if dup.height:
        raise RealBetError("this bet is already recorded")
    return pl.concat([ledger, pl.DataFrame([bet], schema=REAL_SCHEMA)])


def settle_real(ledger: pl.DataFrame, games: pl.DataFrame, closing: pl.DataFrame) -> pl.DataFrame:
    """Grade open bets whose game is final, at the bet's own line and price; CLV against
    ``closing`` (game_id, closing_line, closing_source), the paper ledger's close. Terms and
    already-settled rows are never touched."""
    final = {g["game_id"]: g for g in games.to_dicts()
             if g["completed"] and g["home_points"] is not None and g["away_points"] is not None}
    close = {c["game_id"]: c for c in closing.to_dicts()}
    rows = []
    for r in ledger.to_dicts():
        g = final.get(r["game_id"])
        if r["status"] == "open" and g is not None:
            status = grade(r["side"], r["line"], g["home_points"], g["away_points"])
            r |= {"status": status, "home_points": g["home_points"], "away_points": g["away_points"],
                  "dollars": {"won": win_dollars(r["odds"], r["risk_dollars"]), "lost": -r["risk_dollars"]}.get(status, 0.0)}
            c = close.get(r["game_id"])
            if c is not None and c["closing_line"] is not None:
                r |= {"closing_line": c["closing_line"], "closing_source": c.get("closing_source"),
                      "clv": clv(r["side"], r["line"], c["closing_line"])}
        rows.append(r)
    return pl.DataFrame(rows, schema=REAL_SCHEMA)


def summarize_real(ledger: pl.DataFrame) -> dict:
    settled = ledger.filter(pl.col("status").is_in(["won", "lost", "push"]))
    n = {s: settled.filter(pl.col("status") == s).height for s in ("won", "lost", "push")}
    risked = float(settled["risk_dollars"].sum()) if settled.height else 0.0
    dollars = float(settled["dollars"].sum()) if settled.height else 0.0
    with_clv = settled.filter(pl.col("clv").is_not_null())
    open_ = ledger.filter(pl.col("status") == "open")
    return {"bets": ledger.height, "open": open_.height, "open_risk": float(open_["risk_dollars"].sum()) if open_.height else 0.0,
            "settled": settled.height, **n, "dollars": dollars, "risked": risked,
            "roi": dollars / risked if risked else None,
            "avg_clv": float(with_clv["clv"].mean()) if with_clv.height else None, "n_clv": with_clv.height}
