"""Paper-betting tracker: flags model-vs-market edges, records them as
hypothetical flat bets at the line captured when flagged, and grades them.

Nothing here places a bet. Betting lines never feed the ratings; they are
read only to compare against finished predictions.

Rules (fixed before any results, see DECISIONS.md):
  - Bet a side when |model home margin - market home margin| >= threshold.
  - Market = median home spread across providers in the newest line snapshot.
  - A bet is recorded once, before kickoff, and never changed afterwards.
  - Graded at the recorded consensus line; risk |odds|/100 units to win 1 at
    negative odds (e.g. 1.1 to win 1 at -110).
  - Closing-line value (CLV) = points the recorded line beat the closing line by.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import wilson_ci
from cfbr.config import Bets

LEDGER_SCHEMA = {
    "game_id": pl.Int64, "season": pl.Int32, "week": pl.Int32, "placed_at": pl.Utf8, "kickoff_ts": pl.Utf8,
    "home_team": pl.Utf8, "away_team": pl.Utf8, "side": pl.Utf8, "team": pl.Utf8,
    "model_margin": pl.Float64, "line": pl.Float64, "edge": pl.Float64, "n_providers": pl.Int32,
    "best_line": pl.Float64, "best_provider": pl.Utf8, "odds": pl.Int32, "stake": pl.Float64,
    "status": pl.Utf8, "home_points": pl.Int32, "away_points": pl.Int32, "closing_line": pl.Float64,
    "clv": pl.Float64, "units": pl.Float64,
}


def snapshot_frame(lines_json: list) -> pl.DataFrame:
    """Provider-level spreads from a raw /lines snapshot."""
    rows = [{"game_id": g["id"], "provider": ln.get("provider"), "spread": ln.get("spread")}
            for g in lines_json for ln in (g.get("lines") or []) if ln.get("spread") is not None]
    return pl.DataFrame(rows, schema={"game_id": pl.Int64, "provider": pl.Utf8, "spread": pl.Float64})


def market(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Consensus (median) home spread per game, plus the best number for each side:
    a home bettor wants the highest spread, an away bettor the lowest."""
    return (snapshot.group_by("game_id").agg(
        line=pl.col("spread").median(), n_providers=pl.len().cast(pl.Int32),
        best_home=pl.col("spread").max(), best_home_provider=pl.col("provider").sort_by("spread").last(),
        best_away=pl.col("spread").min(), best_away_provider=pl.col("provider").sort_by("spread").first(),
    ))


def candidate_bets(preds: pl.DataFrame, snapshot: pl.DataFrame, placed_at: datetime, cfg: Bets) -> pl.DataFrame:
    """Bets to record from ``preds`` (predictions.csv rows: game_id, kickoff_ts,
    home_team, away_team, pred_margin) against a line snapshot taken at ``placed_at``.
    Only games that kick off after the snapshot are eligible."""
    preds = preds.with_columns(pl.col("kickoff_ts").cast(pl.Utf8).str.to_datetime(time_zone="UTC")
                               if preds.schema["kickoff_ts"] == pl.Utf8 else pl.col("kickoff_ts"))
    m = market(snapshot)
    d = (preds.join(m, on="game_id", how="inner")
         .filter(pl.col("kickoff_ts") > placed_at)
         .with_columns(edge=pl.col("pred_margin") + pl.col("line"))  # model margin - (-spread)
         .filter(pl.col("edge").abs() >= cfg.threshold))
    home = pl.col("edge") > 0
    return d.select(
        "game_id",
        pl.lit(None, pl.Int32).alias("season"), pl.lit(None, pl.Int32).alias("week"),
        pl.lit(placed_at.isoformat(timespec="seconds")).alias("placed_at"),
        pl.col("kickoff_ts").dt.strftime("%Y-%m-%dT%H:%M:%SZ").alias("kickoff_ts"),
        "home_team", "away_team",
        pl.when(home).then(pl.lit("home")).otherwise(pl.lit("away")).alias("side"),
        pl.when(home).then(pl.col("home_team")).otherwise(pl.col("away_team")).alias("team"),
        pl.col("pred_margin").alias("model_margin"), "line", "edge", "n_providers",
        pl.when(home).then(pl.col("best_home")).otherwise(pl.col("best_away")).alias("best_line"),
        pl.when(home).then(pl.col("best_home_provider")).otherwise(pl.col("best_away_provider")).alias("best_provider"),
        pl.lit(cfg.odds, pl.Int32).alias("odds"), pl.lit(cfg.stake).alias("stake"),
        pl.lit("open").alias("status"),
        pl.lit(None, pl.Int32).alias("home_points"), pl.lit(None, pl.Int32).alias("away_points"),
        pl.lit(None, pl.Float64).alias("closing_line"), pl.lit(None, pl.Float64).alias("clv"),
        pl.lit(None, pl.Float64).alias("units"),
    )


def risk(odds: int, stake: float) -> tuple[float, float]:
    """(amount risked, amount won) for a ``stake``-unit bet: at negative odds you
    risk enough to win ``stake``; at positive odds you risk ``stake``."""
    if odds < 0:
        return stake * -odds / 100, stake
    return stake, stake * odds / 100


def grade(side: str, line: float, home_points: int, away_points: int) -> str:
    margin = home_points - away_points
    cover = margin + line if side == "home" else -(margin + line)
    return "won" if cover > 0 else ("lost" if cover < 0 else "push")


def clv(side: str, line: float, closing: float) -> float:
    """Points the recorded line beat the close by (home spread convention:
    a home bettor gains when the spread falls, an away bettor when it rises)."""
    return line - closing if side == "home" else closing - line


def settle(ledger: pl.DataFrame, games: pl.DataFrame, closing: pl.DataFrame) -> pl.DataFrame:
    """Grade open bets whose games are final. ``games``: game_id, completed,
    home_points, away_points. ``closing``: game_id, closing_line (consensus)."""
    if ledger.height == 0:
        return ledger
    g = games.select("game_id", "completed", pl.col("home_points").alias("hp"), pl.col("away_points").alias("ap"))
    c = closing.select("game_id", pl.col("closing_line").alias("cl"))
    d = ledger.join(g, on="game_id", how="left").join(c, on="game_id", how="left")
    rows = []
    for r in d.iter_rows(named=True):
        if r["status"] == "open" and r["completed"] and r["hp"] is not None and r["ap"] is not None:
            r["status"] = grade(r["side"], r["line"], r["hp"], r["ap"])
            r["home_points"], r["away_points"] = r["hp"], r["ap"]
            risked, won = risk(r["odds"], r["stake"])
            r["units"] = {"won": won, "lost": -risked, "push": 0.0}[r["status"]]
            if r["cl"] is not None:
                r["closing_line"] = r["cl"]
                r["clv"] = clv(r["side"], r["line"], r["cl"])
        rows.append({k: r[k] for k in LEDGER_SCHEMA})
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def read_ledger(path: Path) -> pl.DataFrame:
    if path.exists():
        return pl.read_csv(path, schema_overrides=LEDGER_SCHEMA).select(list(LEDGER_SCHEMA))
    return pl.DataFrame(schema=LEDGER_SCHEMA)


def add_bets(ledger: pl.DataFrame, new: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Append bets for games not already in the ledger. Existing bets are never changed."""
    fresh = new.filter(~pl.col("game_id").is_in(ledger["game_id"].to_list()))
    return pl.concat([ledger, fresh.cast(LEDGER_SCHEMA).select(list(LEDGER_SCHEMA))]), fresh.height


def summarize(ledger: pl.DataFrame, cfg: Bets) -> dict:
    s = ledger.filter(pl.col("status").is_in(["won", "lost", "push"]))
    w = int((s["status"] == "won").sum())
    l = int((s["status"] == "lost").sum())
    p = int((s["status"] == "push").sum())
    lo, hi = wilson_ci(w, w + l)
    risked = float(sum(risk(o, st)[0] for o, st in s.select("odds", "stake").iter_rows())) if s.height else 0.0
    units = float(s["units"].sum()) if s.height else 0.0
    c = s.filter(pl.col("clv").is_not_null())
    be = (-cfg.odds / (100 - cfg.odds)) if cfg.odds < 0 else 100 / (100 + cfg.odds)
    return {
        "open": int((ledger["status"] == "open").sum()), "settled": s.height, "won": w, "lost": l, "push": p,
        "win_rate": w / (w + l) if w + l else math.nan, "ci95": (lo, hi), "breakeven": be,
        "units": units, "risked": risked, "roi": units / risked if risked else math.nan,
        "avg_clv": float(c["clv"].mean()) if c.height else math.nan,
        "beat_close": float((c["clv"] > 0).mean()) if c.height else math.nan,
        "n_clv": c.height,
    }


def _fmt(x: float, pct: bool = False, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if pct:
        return f"{100 * x:.1f}%"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def summary_markdown(ledger: pl.DataFrame, cfg: Bets, season: int) -> str:
    def block(title: str, df: pl.DataFrame) -> list[str]:
        m = summarize(df, cfg)
        return [f"| {title} | {m['settled']} | {m['won']}-{m['lost']}-{m['push']} | {_fmt(m['win_rate'], pct=True)} "
                f"[{_fmt(m['ci95'][0], pct=True)}, {_fmt(m['ci95'][1], pct=True)}] | {_fmt(m['units'], signed=True)} | "
                f"{_fmt(m['roi'], pct=True)} | {_fmt(m['avg_clv'], signed=True)} | {_fmt(m['beat_close'], pct=True)} |"]

    m = summarize(ledger, cfg)
    if m["settled"] < cfg.min_sample:
        verdict = (f"**No conclusion yet.** {m['settled']} settled bets; the pre-set minimum is {cfg.min_sample}. "
                   "Treat these numbers as noise until then.")
    else:
        verdict = ("Minimum sample reached. An edge needs both positive average CLV and a win-rate interval "
                   f"whose lower bound clears break-even ({_fmt(m['breakeven'], pct=True)}).")
    head = ["| bets | settled | W-L-P | win % [95% CI] | units | ROI | avg CLV (pts) | beat close |",
            "|---|---|---|---|---|---|---|---|"]
    lines = [
        f"# {season} paper bets", "",
        "Hypothetical flat bets only; nothing is wagered. Rules: bet the model's side when it differs from the "
        f"consensus line by ≥ {cfg.threshold:g} points, at the line captured when flagged, {cfg.odds} odds, "
        f"{cfg.stake:g} unit to win. Break-even win rate: {_fmt(m['breakeven'], pct=True)}.", "",
        verdict, "",
        *head, *block("all", ledger),
        *block(f"edge {cfg.threshold:g}–6", ledger.filter(pl.col("edge").abs() < 6)),
        *block("edge 6+", ledger.filter(pl.col("edge").abs() >= 6)),
        *block("home side", ledger.filter(pl.col("side") == "home")),
        *block("away side", ledger.filter(pl.col("side") == "away")), "",
        f"Open bets: {m['open']}. CLV > 0 means the recorded line was better than where the market closed.", "",
    ]
    weeks = sorted(ledger.filter(pl.col("status") != "open")["week"].unique().to_list())
    if weeks:
        lines += ["## By week", "", *head]
        for wk in weeks:
            lines += block(f"week {wk}", ledger.filter(pl.col("week") == wk))
        lines.append("")
    opn = ledger.filter(pl.col("status") == "open").sort("kickoff_ts")
    if opn.height:
        lines += ["## Open bets", "", "| kickoff (UTC) | game | bet | home spread | model (home margin) | edge | best available |",
                  "|---|---|---|---|---|---|---|"]
        for r in opn.iter_rows(named=True):
            team_line = r["line"] if r["side"] == "home" else -r["line"]
            best = r["best_line"] if r["side"] == "home" else -r["best_line"]
            lines.append(f"| {r['kickoff_ts'][:16].replace('T', ' ')} | {r['away_team']} @ {r['home_team']} | "
                         f"{r['team']} {team_line:+g} | {r['line']:+g} | {r['model_margin']:+.1f} | {r['edge']:+.1f} | "
                         f"{best:+g} ({r['best_provider']}) |")
        lines.append("")
    return "\n".join(lines)
