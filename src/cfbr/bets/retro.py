"""Retroactive paper bets: replay the live betting rules on weeks already played.

For each prediction slot, the model's point-in-time prediction (fit only on games
that kicked off before the slot's first kickoff) is compared with the opening
line, and bets are flagged, priced and graded exactly as the live tracker does
(same threshold, odds, tiers). Opening lines stand in for the line a bet would
have been recorded at on the Sunday before; closing-line value is measured
against the close as usual.

These bets were never recorded in real time, so they live in their own file
(``out/{season}/bets_retro.csv``) and never count toward the live ledger or its
minimum sample. Betting lines are never model inputs.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from cfbr.backtest.metrics import consensus_spread
from cfbr.bets.ledger import (LEDGER_SCHEMA, by_tier, candidate_bets, money, settle, summarize, weekly, _fmt)
from cfbr.config import Bets, Config
from cfbr.data import Dataset
from cfbr.ingest.odds import BOOK_SCHEMA
from cfbr.model.engine import Engine

MODELS = {"v1": "published model (v1 prior)", "roster": "roster-prior shadow model"}


def opening_books(lines: pl.DataFrame, season: int) -> pl.DataFrame:
    """Book-level opening home spreads (CFBD has no prices)."""
    return (lines.filter((pl.col("season") == season) & pl.col("spread_open").is_not_null())
            .select("game_id", "provider", spread="spread_open",
                    home_price=pl.lit(None, pl.Int32), away_price=pl.lit(None, pl.Int32))
            .cast(BOOK_SCHEMA))


def retro_ledger(eng: Engine, season: int, through_week: int, cfg: Bets, model: str) -> pl.DataFrame:
    """Graded retro bets for finished regular-season games through ``through_week``."""
    ds = eng.ds
    preds = eng.predict(season).filter(
        pl.col("completed") & (pl.col("season_type") == "regular") & (pl.col("week") <= through_week))
    books = opening_books(ds.lines, season)
    out = []
    for (slot,), p in preds.group_by("slot", maintain_order=True):
        # Stamped just before the slot's first kickoff: every game in the slot is still eligible.
        placed_at = p["as_of"].min() - timedelta(seconds=1)
        c = candidate_bets(p.select("game_id", "kickoff_ts", "home_team", "away_team", "pred_margin"),
                           books.filter(pl.col("game_id").is_in(p["game_id"].to_list())), placed_at, cfg)
        out.append(c)
    if not out:
        return pl.DataFrame(schema=LEDGER_SCHEMA | {"model": pl.Utf8})
    games = ds.games.filter(pl.col("season") == season)
    led = (pl.concat(out).drop("season", "week", "season_type")
           .join(games.select("game_id", pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
                              "season_type"), on="game_id", how="left")
           .cast(LEDGER_SCHEMA).select(list(LEDGER_SCHEMA)))
    closing = consensus_spread(ds.lines, eng.cfg.backtest.spread_consensus).select("game_id", closing_line="spread")
    led = settle(led, games.select("game_id", "completed", "home_points", "away_points"), closing)
    return led.with_columns(model=pl.lit(model)).sort("kickoff_ts", "game_id")


def run_retro(cfg: Config, ds: Dataset, season: int, through_week: int, last_completed: int) -> pl.DataFrame:
    frames = []
    for model in MODELS:
        c = cfg.model_copy(deep=True)
        c.prior.features = model
        frames.append(retro_ledger(Engine(ds, c, last_completed=last_completed), season, through_week, cfg.bets, model))
    return pl.concat(frames)


def retro_markdown(led: pl.DataFrame, cfg: Bets, season: int, through_week: int) -> str:
    lines = [
        f"# {season} retroactive paper bets (weeks 1–{through_week})", "",
        "**Not live bets.** These replay the live rules on games already played: each week's point-in-time "
        "prediction (fit only on games before that week) against the **opening** line, "
        f"bet when they differ by ≥ {cfg.threshold:g} points, {cfg.odds} odds, one unit = "
        f"{money(cfg.unit_value, signed=False)}. Lines come from CFBD (DraftKings, plus offshore Bovada in the "
        "consensus); other US books have no history on the free tier. Graded at the opening consensus; CLV is "
        "against the closing consensus. They never count toward the live ledger's 500-bet minimum.", "",
        "The roster model was judged on these same weeks (DECISIONS: roster prior result), so its numbers here "
        "are in-sample in that sense and flatter it.", "",
        "| model | bets | W-L-P | win % [95% CI] | units | $ | ROI | avg CLV (pts) | beat close | best line $ |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model, label in MODELS.items():
        d = led.filter(pl.col("model") == model)
        m, b = summarize(d, cfg), summarize(d, cfg, "best")
        lines.append(f"| {label} | {m['settled']} | {m['won']}-{m['lost']}-{m['push']} | "
                     f"{_fmt(m['win_rate'], pct=True)} [{_fmt(m['ci95'][0], pct=True)}, {_fmt(m['ci95'][1], pct=True)}] | "
                     f"{_fmt(m['units'], signed=True)} | {money(m['dollars'])} | {_fmt(m['roi'], pct=True)} | "
                     f"{_fmt(m['avg_clv'], signed=True)} | {_fmt(m['beat_close'], pct=True)} | {money(b['dollars'])} |")
    lines += ["", f"Break-even win rate at {cfg.odds}: {_fmt(summarize(led, cfg)['breakeven'], pct=True)}.", ""]
    for model, label in MODELS.items():
        d = led.filter(pl.col("model") == model)
        lines += [f"## {label}: week by week", "",
                  "| week | bets | W-L-P | units | $ | running $ | ROI | avg CLV (pts) |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {w['label']} | {w['settled']} | {w['won']}-{w['lost']}-{w['push']} | {_fmt(w['units'], signed=True)} | "
                  f"{money(w['dollars'])} | {money(w['cum_dollars'])} | {_fmt(w['roi'], pct=True)} | "
                  f"{_fmt(w['avg_clv'], signed=True)} |" for w in weekly(d, cfg)]
        lines += ["", f"### {label}: by edge tier", "",
                  "| tier | bets | W-L-P | win % | $ | ROI | avg CLV (pts) | backtest cover |",
                  "|---|---|---|---|---|---|---|---|"]
        for t in by_tier(d, cfg):
            bt = "–" if t["backtest"] is None else _fmt(t["backtest"]["rate"], pct=True)
            lines.append(f"| {t['label']} | {t['settled']} | {t['won']}-{t['lost']}-{t['push']} | "
                         f"{_fmt(t['win_rate'], pct=True)} | {money(t['dollars'])} | {_fmt(t['roi'], pct=True)} | "
                         f"{_fmt(t['avg_clv'], signed=True)} | {bt} |")
        lines.append("")
    both = led.pivot(on="model", index="game_id", values="side", aggregate_function="first")
    if {"v1", "roster"} <= set(both.columns):
        agree = both.filter(pl.col("v1").is_not_null() & pl.col("roster").is_not_null())
        lines += ["## Overlap", "",
                  f"{agree.height} games were bet by both models "
                  f"({int((agree['v1'] == agree['roster']).sum())} on the same side); "
                  f"{int(both['v1'].is_not_null().sum()) - agree.height} by v1 only, "
                  f"{int(both['roster'].is_not_null().sum()) - agree.height} by roster only.", ""]
    return "\n".join(lines)
