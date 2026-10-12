"""Moneyline vs spread pricing: a market-only test (DECISIONS "Moneyline vs spread pricing").

Is a side's moneyline a better buy than the same side's spread at -110? Both bets are
priced against the points-to-cents table at the consensus line, so the model never
enters and the 2022-2025 holdout is not re-spent. Moneylines are read straight from
the raw CFBD cache (the store keeps none); games, scores and the consensus come from
the store. Every setting below was fixed in the rule before anything was computed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from cfbr.backtest.metrics import consensus_spread
from cfbr.bets.pricing import PricingTable, table_from_dataset
from cfbr.ingest.store import _games_file, _read

JUDGED = ("DraftKings", [2023, 2024, 2025])
REPORTED = [("Bovada", [2021, 2022, 2023, 2024, 2025]), ("ESPN Bet", [2024, 2025])]
ATS_ODDS = -110
MOVE = 1.5               # precheck: games whose consensus moved at least this much open -> close
BANDS = [(0.0, 0.20, "<20%"), (0.20, 0.45, "20–45%"), (0.45, 0.55, "45–55%"),
         (0.55, 0.80, "55–80%"), (0.80, 1.01, "≥80%")]
N_BOOT, SEED = 10_000, 0
SIDES_SCHEMA = {"season": pl.Int32, "game_id": pl.Int64, "provider": pl.Utf8, "side": pl.Utf8,
                "odds": pl.Float64, "number": pl.Float64, "center": pl.Float64, "p_win": pl.Float64,
                "ev_ml": pl.Float64, "ev_ats": pl.Float64, "pred": pl.Float64,
                "ret_ml": pl.Float64, "ret_ats": pl.Float64, "real": pl.Float64}


def payout(odds: float) -> float:
    """Units won per unit risked at American ``odds``."""
    return odds / 100 if odds > 0 else 100 / -odds


def implied(odds: float) -> float:
    """Break-even probability of American ``odds`` (vig included)."""
    return 1 / (1 + payout(odds))


def raw_moneylines(raw_dir: Path, seasons: list[int]) -> pl.DataFrame:
    """One row per (game, provider) with both moneylines, plus that book's spread and opening spread."""
    rows = []
    for season in seasons:
        p = _games_file(raw_dir, "lines", season)
        if p is None:
            continue
        for g in _read(p):
            for ln in g.get("lines") or []:
                if ln.get("homeMoneyline") is None or ln.get("awayMoneyline") is None:
                    continue
                rows.append({"game_id": g["id"], "season": g["season"], "provider": ln.get("provider"),
                             "home_ml": ln["homeMoneyline"], "away_ml": ln["awayMoneyline"],
                             "spread": ln.get("spread"), "spread_open": ln.get("spreadOpen")})
    return pl.DataFrame(rows, schema={"game_id": pl.Int64, "season": pl.Int32, "provider": pl.Utf8,
                                      "home_ml": pl.Float64, "away_ml": pl.Float64,
                                      "spread": pl.Float64, "spread_open": pl.Float64})


def with_market(ml: pl.DataFrame, games: pl.DataFrame, lines: pl.DataFrame, method: str) -> pl.DataFrame:
    """Joins completed games' home margin and the opening and closing consensus."""
    close = consensus_spread(lines, method, "spread").select("game_id", cons_close="spread")
    opening = consensus_spread(lines, method, "spread_open").select("game_id", cons_open="spread_open")
    done = (games.filter(pl.col("completed") & pl.col("home_points").is_not_null() & pl.col("away_points").is_not_null())
            .select("game_id", margin=(pl.col("home_points") - pl.col("away_points")).cast(pl.Float64)))
    return (ml.with_columns(pl.col("game_id").cast(done["game_id"].dtype))
            .join(done, on="game_id").join(close, on="game_id", how="left").join(opening, on="game_id", how="left"))


def precheck(d: pl.DataFrame, tables: dict[int, PricingTable]) -> dict:
    """Do the moneylines sit closer to the table at the opening or at the closing consensus?
    Judged on games whose consensus moved at least MOVE points; no-vig home probability."""
    m = d.filter(pl.col("cons_open").is_not_null() & pl.col("cons_close").is_not_null()
                 & ((pl.col("cons_close") - pl.col("cons_open")).abs() >= MOVE))
    gap_open, gap_close = [], []
    for season, hml, aml, co, cc in m.select("season", "home_ml", "away_ml", "cons_open", "cons_close").iter_rows():
        qh, qa = implied(hml), implied(aml)
        nv = qh / (qh + qa)
        t = tables[season]
        gap_open.append(abs(nv - t.win_push("home", 0.0, co)[0]))
        gap_close.append(abs(nv - t.win_push("home", 0.0, cc)[0]))
    go = float(np.mean(gap_open)) if gap_open else float("nan")
    gc = float(np.mean(gap_close)) if gap_close else float("nan")
    return {"n_moved": m.height, "gap_open": go, "gap_close": gc, "use_open": bool(go < gc)}


def game_sides(d: pl.DataFrame, tables: dict[int, PricingTable], use_open: bool) -> pl.DataFrame:
    """Both sides of every game: predicted Δ (table EV of the moneyline minus the spread at -110)
    and realized Δ (moneyline return minus spread return, 1 unit risked each, pushes 0)."""
    center_col, number_col = ("cons_open", "spread_open") if use_open else ("cons_close", "spread")
    d = d.filter(pl.col(center_col).is_not_null() & pl.col(number_col).is_not_null())
    rows = []
    for r in d.iter_rows(named=True):
        t = tables[r["season"]]
        center, number, margin = r[center_col], r[number_col], r["margin"]
        for side, odds in (("home", r["home_ml"]), ("away", r["away_ml"])):
            p_win = t.win_push(side, 0.0, center)[0]
            ev_ml = t.expected_value(side, 0.0, odds, center)
            ev_ats = t.expected_value(side, number, ATS_ODDS, center)
            won = margin > 0 if side == "home" else margin < 0
            ret_ml = payout(odds) if won else -1.0
            cover = margin + number if side == "home" else -(margin + number)
            ret_ats = payout(ATS_ODDS) if cover > 0 else (0.0 if cover == 0 else -1.0)
            rows.append({"season": r["season"], "game_id": r["game_id"], "provider": r["provider"], "side": side,
                         "odds": odds, "number": number, "center": center, "p_win": p_win,
                         "ev_ml": ev_ml, "ev_ats": ev_ats, "pred": ev_ml - ev_ats,
                         "ret_ml": ret_ml, "ret_ats": ret_ats, "real": ret_ml - ret_ats})
    return pl.DataFrame(rows, schema=SIDES_SCHEMA)


def cluster_bootstrap(sel: pl.DataFrame, col: str = "real") -> tuple[float, float]:
    """95% percentile CI of the mean of ``col``, resampling whole games (both sides together)."""
    # Sorted so a fixed seed draws the same games every run (group_by order is not stable).
    per = sel.group_by("game_id").agg(s=pl.col(col).sum(), n=pl.len()).sort("game_id")
    if per.height < 2:
        return float("nan"), float("nan")
    s, n = per["s"].to_numpy(), per["n"].to_numpy()
    idx = np.random.default_rng(SEED).integers(0, per.height, size=(N_BOOT, per.height))
    means = s[idx].sum(axis=1) / n[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def primary(sides: pl.DataFrame) -> dict:
    sel = sides.filter(pl.col("pred") > 0)
    lo, hi = cluster_bootstrap(sel)
    return {"n_sides": sides.height, "n_games": sides["game_id"].n_unique() if sides.height else 0,
            "n_sel": sel.height, "n_sel_games": sel["game_id"].n_unique() if sel.height else 0,
            "pred": float(sel["pred"].mean()) if sel.height else float("nan"),
            "real": float(sel["real"].mean()) if sel.height else float("nan"),
            "ci95": (lo, hi), "adopt": bool(lo > 0)}


def bands(sides: pl.DataFrame) -> list[dict]:
    """Predicted vs realized Δ and realized moneyline ROI by the side's fair P(win)."""
    out = []
    for lo, hi, label in BANDS:
        b = sides.filter((pl.col("p_win") >= lo) & (pl.col("p_win") < hi))
        means = {c: float(b[c].mean()) if b.height else float("nan") for c in ("pred", "real", "ret_ml", "ret_ats")}
        out.append({"band": label, "n": b.height, "pred": means["pred"], "real": means["real"],
                    "ml_roi": means["ret_ml"], "ats_roi": means["ret_ats"]})
    return out


def hold(d: pl.DataFrame) -> float:
    """Mean moneyline overround (sum of implied probabilities − 1)."""
    return float(np.mean([implied(h) + implied(a) - 1 for h, a in d.select("home_ml", "away_ml").iter_rows()]))


def evaluate(ds, cfg) -> tuple[dict, pl.DataFrame]:
    seasons = sorted({s for _, ss in [JUDGED, *REPORTED] for s in ss})
    tables = {s: table_from_dataset(ds, cfg, s - 1) for s in seasons}
    method = cfg.backtest.spread_consensus
    d = with_market(raw_moneylines(cfg.paths.raw_dir, seasons), ds.games, ds.lines, method)

    book, judged_seasons = JUDGED
    dj = d.filter((pl.col("provider") == book) & pl.col("season").is_in(judged_seasons))
    pre = precheck(dj, tables)
    sides = game_sides(dj, tables, pre["use_open"])
    report = {"book": book, "seasons": judged_seasons, "precheck": pre, "primary": primary(sides),
              "bands": bands(sides), "hold": hold(dj), "by_season": [], "reported": []}
    for season in judged_seasons:
        report["by_season"].append({"season": season, **primary(sides.filter(pl.col("season") == season))})
    all_sides = [sides]
    for other, other_seasons in REPORTED:
        do = d.filter((pl.col("provider") == other) & pl.col("season").is_in(other_seasons))
        so = game_sides(do, tables, pre["use_open"])
        all_sides.append(so)
        report["reported"].append({"book": other, "seasons": other_seasons, "hold": hold(do) if do.height else float("nan"),
                                   "primary": primary(so), "bands": bands(so)})
    return report, pl.concat(all_sides, how="diagonal_relaxed")


def _f(x: float, nd: int = 3) -> str:
    return "—" if x != x else f"{x:+.{nd}f}"


def markdown(r: dict) -> str:
    pre, p = r["precheck"], r["primary"]
    lines = ["# Moneyline vs spread pricing (market-only)", "",
             f"Judged: {r['book']} {r['seasons'][0]}–{r['seasons'][-1]}. Rule: DECISIONS.md, \"Moneyline vs spread pricing\".", "",
             "## Precheck: are CFBD moneylines openers or closers?", "",
             f"{pre['n_moved']} games moved ≥ {MOVE} pts. Mean |no-vig P(home) − table P(home)|: "
             f"opening consensus {pre['gap_open']:.4f}, closing consensus {pre['gap_close']:.4f} → "
             f"**{'opening' if pre['use_open'] else 'closing'}** lines used.", "",
             "## Primary", "",
             f"Game-sides with predicted Δ > 0: {p['n_sel']} of {p['n_sides']} ({p['n_sel_games']} games). "
             f"Mean predicted Δ {_f(p['pred'])}, realized Δ {_f(p['real'])} units per side, "
             f"95% CI [{_f(p['ci95'][0])}, {_f(p['ci95'][1])}] (bootstrap over games, {N_BOOT:,} draws, seed {SEED}).", "",
             f"**{'ADOPT' if p['adopt'] else 'NOT ADOPTED'}** (adopt only if the CI is above 0).", "",
             "| Season | Sides (Δ>0) | Pred Δ | Realized Δ | 95% CI |", "|---|---|---|---|---|"]
    for s in r["by_season"]:
        lines.append(f"| {s['season']} | {s['n_sel']} | {_f(s['pred'])} | {_f(s['real'])} | "
                     f"[{_f(s['ci95'][0])}, {_f(s['ci95'][1])}] |")
    lines += ["", f"{r['book']} moneyline hold: {r['hold']:.2%}.", "",
              "## By fair P(win) (reported, not judged)", "", *_band_table(r["bands"]),
              "", "## Other books (reported, not judged)", ""]
    for o in r["reported"]:
        q = o["primary"]
        lines += [f"### {o['book']} {o['seasons'][0]}–{o['seasons'][-1]}", "",
                  f"Hold {o['hold']:.2%}. Predicted Δ > 0: {q['n_sel']} of {q['n_sides']} sides; "
                  f"pred {_f(q['pred'])}, realized {_f(q['real'])}, 95% CI [{_f(q['ci95'][0])}, {_f(q['ci95'][1])}].", "",
                  *_band_table(o["bands"]), ""]
    return "\n".join(lines) + "\n"


def _band_table(rows: list[dict]) -> list[str]:
    out = ["| Fair P(win) | Sides | Pred Δ | Realized Δ | ML ROI | ATS ROI |", "|---|---|---|---|---|---|"]
    for b in rows:
        out.append(f"| {b['band']} | {b['n']} | {_f(b['pred'])} | {_f(b['real'])} | {_f(b['ml_roi'])} | {_f(b['ats_roi'])} |")
    return out


def run_moneyline_eval(ds, cfg) -> Path:
    report, sides = evaluate(ds, cfg)
    out = cfg.paths.results_dir / "moneyline_pricing.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    sides.write_parquet(out)
    out.with_suffix(".md").write_text(markdown(report))
    return out
