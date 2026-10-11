"""Paper-betting tracker: flags model-vs-market edges, records them as
hypothetical flat bets at the line captured when flagged, and grades them.

Nothing here places a bet. Betting lines never feed the ratings; they are
read only to compare against finished predictions.

Rules (fixed before any results, see DECISIONS.md):
  - Bet a side when |model home margin - market home margin| >= threshold.
  - Market = median home spread across providers in the newest line snapshot.
  - A bet is recorded once, before kickoff, and never changed afterwards.
  - Graded twice: at the recorded consensus line (does the model beat the
    market?) and at the best number across the books a bet could be priced at
    (what shopping them would have got). No book is excluded by jurisdiction,
    but offshore books are market data only: they count toward the consensus
    and never supply the best-line number.
  - The consensus record is graded at the configured odds (-110). Risk |odds|/100
    units to win 1 at negative odds (e.g. 1.1 to win 1 at -110). The best-line
    record uses the book's own price when the feed has one (The Odds API does,
    CFBD doesn't): the best number for the side, ties broken by the better price.
  - Closing-line value (CLV) = points the recorded line beat the closing line by.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import polars as pl

from cfbr.backtest.metrics import NON_BOOKS, wilson_ci
from cfbr.config import Bets
from cfbr.ingest.odds import BOOK_SCHEMA

LEDGER_SCHEMA = {
    "game_id": pl.Int64, "season": pl.Int32, "week": pl.Int32, "placed_at": pl.Utf8, "kickoff_ts": pl.Utf8,
    "home_team": pl.Utf8, "away_team": pl.Utf8, "side": pl.Utf8, "team": pl.Utf8,
    "model_margin": pl.Float64, "line": pl.Float64, "edge": pl.Float64, "n_providers": pl.Int32,
    "best_line": pl.Float64, "best_provider": pl.Utf8, "odds": pl.Int32, "stake": pl.Float64,
    "status": pl.Utf8, "home_points": pl.Int32, "away_points": pl.Int32, "closing_line": pl.Float64,
    "clv": pl.Float64, "units": pl.Float64,
    # Added later; older ledgers get these as nulls.
    "books": pl.Utf8, "best_status": pl.Utf8, "best_units": pl.Float64, "best_clv": pl.Float64,
    "season_type": pl.Utf8, "best_odds": pl.Int32, "fbs_game": pl.Boolean,
}

SETTLED = ["won", "lost", "push"]

# Walk-forward backtest cover rate by edge tier, as (wins, decided bets): the frozen
# tuned config over the tuning seasons 2015-2021, model side vs the closing consensus,
# pushes excluded. Keyed by (min edge, max edge or None). A reference for what each
# tier has historically returned; the model's own cover probabilities are far more
# confident than this. See DECISIONS.md.
BACKTEST_TIER_COVER = {(3.0, 6.0): (860, 1706), (6.0, 10.0): (493, 972), (10.0, None): (284, 524)}

# Non-books (numberfire projections, teamrankings/consensus aggregates) can't be
# bet, so they never price a bet.


def snapshot_frame(lines_json: list) -> pl.DataFrame:
    """Sportsbook-level spreads from a raw CFBD /lines snapshot (non-books dropped; CFBD has no prices)."""
    rows = [{"game_id": g["id"], "provider": ln.get("provider"), "spread": ln.get("spread")}
            for g in lines_json for ln in (g.get("lines") or [])
            if ln.get("spread") is not None and (ln.get("provider") or "").lower() not in NON_BOOKS]
    return pl.DataFrame(rows, schema=BOOK_SCHEMA)


def book_key(name: str) -> str:
    """Provider name normalized across feeds ("BetOnline.ag" and "BetOnline" are one book)."""
    n = name.lower().removesuffix(".ag")
    return "".join(ch for ch in n if ch.isalnum())


def combine_books(*frames: pl.DataFrame) -> pl.DataFrame:
    """One row per game and book across feeds; earlier frames win (pass the priced feed first)."""
    d = pl.concat([f.select(list(BOOK_SCHEMA)).cast(BOOK_SCHEMA) for f in frames])
    return (d.with_columns(_k=pl.col("provider").map_elements(book_key, return_dtype=pl.Utf8))
            .unique(["game_id", "_k"], keep="first", maintain_order=True).drop("_k"))


def payout(odds: pl.Expr) -> pl.Expr:
    """Amount won per unit risked at American odds (higher is a better price)."""
    return pl.when(odds > 0).then(odds / 100).otherwise(100 / -odds)


def _price(x: int | None) -> str:
    return "" if x is None else f"{x:+d}"


def market(snapshot: pl.DataFrame, offshore: list[str] | None = None, default_odds: int = -110) -> pl.DataFrame:
    """Consensus (median) home spread per game over all books, plus the best
    number for each side over non-offshore books only (null if none posted):
    a home bettor wants the highest spread, an away bettor the lowest, and
    between equal spreads the better price (unpriced books count as ``default_odds``)."""
    off = {book_key(b) for b in (offshore or [])}
    snap = snapshot.with_columns(
        offshore=pl.col("provider").map_elements(book_key, return_dtype=pl.Utf8).is_in(list(off)),
        hp=payout(pl.col("home_price").fill_null(default_odds)),
        ap=payout(pl.col("away_price").fill_null(default_odds)))
    prices = pl.when(pl.col("home_price").is_not_null()).then(pl.format(
        " ({}/{})", pl.col("home_price").map_elements(_price, return_dtype=pl.Utf8),
        pl.col("away_price").map_elements(_price, return_dtype=pl.Utf8))).otherwise(pl.lit(""))
    label = pl.format("{} {}{}{}", pl.col("provider"),
                      pl.col("spread").map_elements(lambda x: f"{x:+g}", return_dtype=pl.Utf8), prices,
                      pl.when(pl.col("offshore")).then(pl.lit(" (offshore)")).otherwise(pl.lit("")))

    # Best number per side over non-offshore books: rank by (points for the bettor, price), take the top row.
    on = snap.filter(~pl.col("offshore"))
    best_home = (on.sort("spread", "hp").group_by("game_id", maintain_order=True).last()
                 .select("game_id", best_home="spread", best_home_provider="provider", best_home_price="home_price"))
    best_away = (on.with_columns(_pts=-pl.col("spread")).sort("_pts", "ap").group_by("game_id", maintain_order=True)
                 .last().select("game_id", best_away="spread", best_away_provider="provider",
                                best_away_price="away_price"))
    return (snap.group_by("game_id").agg(line=pl.col("spread").median(), n_providers=pl.len().cast(pl.Int32),
                                         books=label.sort().str.join("; "))
            .join(best_home, on="game_id", how="left").join(best_away, on="game_id", how="left"))


def candidate_bets(preds: pl.DataFrame, snapshot: pl.DataFrame, placed_at: datetime, cfg: Bets) -> pl.DataFrame:
    """Bets to record from ``preds`` (predictions.csv rows: game_id, kickoff_ts,
    home_team, away_team, pred_margin) against a line snapshot taken at ``placed_at``.
    Only games that kick off after the snapshot are eligible."""
    preds = preds.with_columns(pl.col("kickoff_ts").cast(pl.Utf8).str.to_datetime(time_zone="UTC")
                               if preds.schema["kickoff_ts"] == pl.Utf8 else pl.col("kickoff_ts"))
    m = market(snapshot, cfg.offshore_books, cfg.odds)
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
        "books", pl.lit(None, pl.Utf8).alias("best_status"), pl.lit(None, pl.Float64).alias("best_units"),
        pl.lit(None, pl.Float64).alias("best_clv"), pl.lit(None, pl.Utf8).alias("season_type"),
        pl.when(home).then(pl.col("best_home_price")).otherwise(pl.col("best_away_price")).alias("best_odds"),
        pl.lit(None, pl.Boolean).alias("fbs_game"),
    )


def tiers(cfg: Bets) -> list[dict]:
    """Edge tiers in ascending order: name, lo (min |edge|), hi (exclusive, None for the top), label."""
    items = sorted(cfg.tiers.items(), key=lambda kv: kv[1])
    out = []
    for i, (name, lo) in enumerate(items):
        hi = items[i + 1][1] if i + 1 < len(items) else None
        span = f"{lo:g}+" if hi is None else f"{lo:g}–{hi:g}"
        out.append({"name": name, "lo": lo, "hi": hi, "label": f"{name} edge ({span} pts)"})
    return out


def tier_expr(cfg: Bets) -> pl.Expr:
    """Tier name from |edge|; anything below the lowest tier falls into it."""
    ts = tiers(cfg)
    expr = pl.lit(ts[0]["name"])
    for t in ts[1:]:
        expr = pl.when(pl.col("edge").abs() >= t["lo"]).then(pl.lit(t["name"])).otherwise(expr)
    return expr


def week_label(week: int | None, season_type: str | None) -> str:
    # CFBD numbers postseason weeks from 1 again, so they get their own bucket.
    if season_type == "postseason":
        return "Postseason"
    return "Week ?" if week is None else f"Week {week}"


def by_week(ledger: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """Bets grouped by week in calendar order, all postseason games in one bucket last."""
    post = (pl.col("season_type") == "postseason").fill_null(False)
    d = ledger.with_columns(_post=post, _wk=pl.when(post).then(None).otherwise(pl.col("week")))
    return [(week_label(wk, "postseason" if p else None), g.drop("_post", "_wk"))
            for (p, wk), g in d.sort("_post", "_wk", nulls_last=True).group_by("_post", "_wk", maintain_order=True)]


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
            best = r["best_line"]  # null when only offshore books had a line
            if best is not None:
                # At the book's own price when known, else the configured odds.
                b_risked, b_won = risk(r["best_odds"] if r["best_odds"] is not None else r["odds"], r["stake"])
                r["best_status"] = grade(r["side"], best, r["hp"], r["ap"])
                r["best_units"] = {"won": b_won, "lost": -b_risked, "push": 0.0}[r["best_status"]]
            if r["cl"] is not None:
                r["closing_line"] = r["cl"]
                r["clv"] = clv(r["side"], r["line"], r["cl"])
                if best is not None:
                    r["best_clv"] = clv(r["side"], best, r["cl"])
        rows.append({k: r[k] for k in LEDGER_SCHEMA})
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)


def read_ledger(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame(schema=LEDGER_SCHEMA)
    df = pl.read_csv(path, schema_overrides={k: v for k, v in LEDGER_SCHEMA.items()})
    missing = [pl.lit(None, t).alias(c) for c, t in LEDGER_SCHEMA.items() if c not in df.columns]
    return df.with_columns(missing).select(list(LEDGER_SCHEMA)).cast(LEDGER_SCHEMA)


def add_bets(ledger: pl.DataFrame, new: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Append bets for games not already in the ledger. Existing bets are never changed."""
    fresh = new.filter(~pl.col("game_id").is_in(ledger["game_id"].to_list()))
    return pl.concat([ledger, fresh.cast(LEDGER_SCHEMA).select(list(LEDGER_SCHEMA))]), fresh.height


def summarize(ledger: pl.DataFrame, cfg: Bets, basis: str = "consensus") -> dict:
    """Record at the consensus line, or at the best number across books (basis="best")."""
    st, un, cv = ("status", "units", "clv") if basis == "consensus" else ("best_status", "best_units", "best_clv")
    s = ledger.filter(pl.col(st).is_in(SETTLED))
    w = int((s[st] == "won").sum())
    l = int((s[st] == "lost").sum())
    p = int((s[st] == "push").sum())
    lo, hi = wilson_ci(w, w + l)
    price = pl.col("odds") if basis == "consensus" else pl.coalesce("best_odds", "odds")
    risked = float(sum(risk(o, k)[0] for o, k in s.select(price, "stake").iter_rows())) if s.height else 0.0
    units = float(s[un].sum()) if s.height else 0.0
    c = s.filter(pl.col(cv).is_not_null())
    be = (-cfg.odds / (100 - cfg.odds)) if cfg.odds < 0 else 100 / (100 + cfg.odds)
    opn = ledger.filter(pl.col("status") == "open")
    at_risk = float(sum(risk(o, k)[0] for o, k in opn.select("odds", "stake").iter_rows()))
    return {
        "open": opn.height, "settled": s.height, "won": w, "lost": l, "push": p,
        "win_rate": w / (w + l) if w + l else math.nan, "ci95": (lo, hi), "breakeven": be,
        "units": units, "risked": risked, "roi": units / risked if risked else math.nan,
        # Dollar figures are units x unit_value; reporting only.
        "dollars": units * cfg.unit_value, "risked_dollars": risked * cfg.unit_value,
        "balance": cfg.bankroll + units * cfg.unit_value,
        "open_risk_dollars": at_risk * cfg.unit_value,
        "avg_clv": float(c[cv].mean()) if c.height else math.nan,
        "beat_close": float((c[cv] > 0).mean()) if c.height else math.nan,
        "n_clv": c.height,
    }


def weekly(ledger: pl.DataFrame, cfg: Bets) -> list[dict]:
    """Win/loss by week for weeks with settled bets, with running totals and the
    bankroll before and after each week. ``over_bankroll`` flags a week whose
    stakes (all placed before any of them settled) exceeded the bankroll going in."""
    out, cum_u = [], 0.0
    for label, g in by_week(ledger):
        m = summarize(g, cfg)
        if m["settled"] == 0:
            continue
        start = cfg.bankroll + cum_u * cfg.unit_value
        cum_u += m["units"]
        out.append({"label": label, **m, "cum_units": cum_u, "cum_dollars": cum_u * cfg.unit_value,
                    "start_balance": start, "balance": cfg.bankroll + cum_u * cfg.unit_value,
                    "over_bankroll": m["risked_dollars"] > start})
    return out


def by_tier(ledger: pl.DataFrame, cfg: Bets) -> list[dict]:
    """Record per edge tier, with the backtest cover rate for that tier when one exists."""
    d = ledger.with_columns(_tier=tier_expr(cfg))
    out = []
    for t in tiers(cfg):
        m = summarize(d.filter(pl.col("_tier") == t["name"]).drop("_tier"), cfg)
        ref = BACKTEST_TIER_COVER.get((t["lo"], t["hi"]))
        bt = None
        if ref:
            w, n = ref
            bt = {"wins": w, "n": n, "rate": w / n, "ci95": wilson_ci(w, n)}
        out.append({**t, **m, "bets": m["open"] + m["settled"], "backtest": bt})
    return out


def money(x: float | None, signed: bool = True) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    s = f"${abs(x):,.0f}" if round(x, 2) == round(x) else f"${abs(x):,.2f}"
    if not signed:
        return s
    return ("+" if x > 0 else "-" if x < 0 else "") + s


def _fmt(x: float, pct: bool = False, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if pct:
        return f"{100 * x:.1f}%"
    return f"{x:+.2f}" if signed else f"{x:.2f}"


def tests_markdown(ledger: pl.DataFrame, cfg: Bets) -> list[str]:
    from cfbr.bets.experiments import tracked_tests

    lines = ["## Tracked tests", "",
             "Pre-registered splits of the bets above (rules fixed 2026-10-11, see DECISIONS.md). They don't change "
             "which bets are recorded. Primary metric: the gap in average CLV between the two groups.", ""]
    for t in tracked_tests(ledger, cfg):
        lines += [f"### {t['name']}", "", t["question"], "",
                  "| group | settled | W-L-P | win % | $ | ROI | avg CLV (pts) |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {g['label']} | {g['settled']} | {g['won']}-{g['lost']}-{g['push']} | {_fmt(g['win_rate'], pct=True)} | "
                  f"{money(g['dollars'])} | {_fmt(g['roi'], pct=True)} | {_fmt(g['avg_clv'], signed=True)} |"
                  for g in t["groups"]]
        gap = (f"CLV gap (first group − second): {_fmt(t['gap'], signed=True)} "
               f"[{_fmt(t['ci95'][0], signed=True)}, {_fmt(t['ci95'][1], signed=True)}]")
        if t["enough_data"]:
            status = f"{gap}. Decision {t['decision_point']}."
        else:
            status = (f"**No conclusion yet**: {t['n_in']} and {t['n_out']} bets with a closing line; the pre-set minimum "
                      f"is {t['min_clv_bets']} per group by {t['decision_point']} (otherwise it carries over to 2027). "
                      f"So far, {gap[0].lower() + gap[1:]}, which is noise until then.")
        lines += ["", status, "", f"Decision rule: {t['decision']}", ""]
    return lines


def summary_markdown(ledger: pl.DataFrame, cfg: Bets, season: int) -> str:
    def row(title: str, m: dict) -> str:
        return (f"| {title} | {m['settled']} | {m['won']}-{m['lost']}-{m['push']} | {_fmt(m['win_rate'], pct=True)} "
                f"[{_fmt(m['ci95'][0], pct=True)}, {_fmt(m['ci95'][1], pct=True)}] | {_fmt(m['units'], signed=True)} | "
                f"{money(m['dollars'])} | {_fmt(m['roi'], pct=True)} | {_fmt(m['avg_clv'], signed=True)} | "
                f"{_fmt(m['beat_close'], pct=True)} |")

    def block(title: str, df: pl.DataFrame, basis: str = "consensus") -> list[str]:
        return [row(title, summarize(df, cfg, basis))]

    m = summarize(ledger, cfg)
    risked, to_win = risk(cfg.odds, cfg.stake)
    if m["settled"] < cfg.min_sample:
        verdict = (f"**No conclusion yet.** {m['settled']} settled bets; the pre-set minimum is {cfg.min_sample}. "
                   "Treat these numbers as noise until then.")
    else:
        verdict = ("Minimum sample reached. An edge needs both positive average CLV and a win-rate interval "
                   f"whose lower bound clears break-even ({_fmt(m['breakeven'], pct=True)}).")
    head = ["| bets | settled | W-L-P | win % [95% CI] | units | $ | ROI | avg CLV (pts) | beat close |",
            "|---|---|---|---|---|---|---|---|---|"]
    lines = [
        f"# {season} paper bets", "",
        "Hypothetical flat bets only; nothing is wagered. Rules: bet the model's side when it differs from the "
        f"consensus line by ≥ {cfg.threshold:g} points, at the line captured when flagged, {cfg.odds} odds, "
        f"{cfg.stake:g} unit to win. Break-even win rate: {_fmt(m['breakeven'], pct=True)}. "
        f"Starting bankroll {money(cfg.bankroll, signed=False)}; every bet is a flat {money(cfg.unit_value, signed=False)} unit "
        f"({100 * cfg.unit_value / cfg.bankroll:g}% of the starting bankroll, not compounding), "
        f"so each bet risks {money(risked * cfg.unit_value, signed=False)} to win {money(to_win * cfg.unit_value, signed=False)}. "
        f"Bankroll now: {money(m['balance'], signed=False)}.", "",
        verdict, "",
        "Two records: **at consensus** (median line across books, measures whether the model beats the market) "
        "and **best line** (the best number across non-offshore books at that book's own price when known, what "
        "shopping them would get; offshore books count toward the consensus but are never used to price a bet).",
        "",
        *head, *block("all — at consensus", ledger), *block("all — best line", ledger, "best"),
        *block("home side", ledger.filter(pl.col("side") == "home")),
        *block("away side", ledger.filter(pl.col("side") == "away")), "",
        f"Open bets: {m['open']} ({money(m['open_risk_dollars'], signed=False)} at risk). "
        "CLV > 0 means the recorded line was better than where the market closed.", "",
    ]
    wk = weekly(ledger, cfg)
    if wk:
        lines += ["## Week by week", "",
                  "| week | settled | W-L-P | units | staked | $ | bankroll | ROI | avg CLV (pts) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {w['label']} | {w['settled']} | {w['won']}-{w['lost']}-{w['push']} | {_fmt(w['units'], signed=True)} | "
                  f"{money(w['risked_dollars'], signed=False)}{' ⚠' if w['over_bankroll'] else ''} | {money(w['dollars'])} | "
                  f"{money(w['balance'], signed=False)} | {_fmt(w['roi'], pct=True)} | "
                  f"{_fmt(w['avg_clv'], signed=True)} |" for w in wk]
        lines.append("")
    lines += ["## By edge tier", "",
              "Every bet is the same flat stake, so the payout per bet is identical; tiers group bets by how far the "
              "model is from the market. Backtest = how often that tier covered the closing line in the 2015–2021 "
              "walk-forward backtest (pushes excluded). No tier has cleared break-even there with confidence.", "",
              "| tier | bets | open | settled | W-L-P | win % [95% CI] | units | $ | ROI | backtest cover [95% CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for t in by_tier(ledger, cfg):
        bt = t["backtest"]
        bt_txt = "–" if bt is None else (f"{_fmt(bt['rate'], pct=True)} [{_fmt(bt['ci95'][0], pct=True)}, "
                                          f"{_fmt(bt['ci95'][1], pct=True)}] (n={bt['n']})")
        lines.append(f"| {t['label']} | {t['bets']} | {t['open']} | {t['settled']} | {t['won']}-{t['lost']}-{t['push']} | "
                     f"{_fmt(t['win_rate'], pct=True)} [{_fmt(t['ci95'][0], pct=True)}, {_fmt(t['ci95'][1], pct=True)}] | "
                     f"{_fmt(t['units'], signed=True)} | {money(t['dollars'])} | {_fmt(t['roi'], pct=True)} | {bt_txt} |")
    lines.append("")
    lines += tests_markdown(ledger, cfg)
    opn = ledger.filter(pl.col("status") == "open").with_columns(tier=tier_expr(cfg)).sort("kickoff_ts")
    if opn.height:
        lines += ["## Open bets", "", "| kickoff (UTC) | game | bet | tier | home spread | model (home margin) | edge | best available | all books (home spread, home/away price) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in opn.iter_rows(named=True):
            team_line = r["line"] if r["side"] == "home" else -r["line"]
            if r["best_line"] is None:
                best_txt = "– (offshore only)"
            else:
                best = r["best_line"] if r["side"] == "home" else -r["best_line"]
                price = f" {r['best_odds']:+d}" if r["best_odds"] is not None else ""
                best_txt = f"{best:+g}{price} ({r['best_provider']})"
            lines.append(f"| {r['kickoff_ts'][:16].replace('T', ' ')} | {r['away_team']} @ {r['home_team']} | "
                         f"{r['team']} {team_line:+g} | {r['tier']} | {r['line']:+g} | {r['model_margin']:+.1f} | {r['edge']:+.1f} | "
                         f"{best_txt} | {r['books'] or '–'} |")
        lines.append("")
    return "\n".join(lines)
