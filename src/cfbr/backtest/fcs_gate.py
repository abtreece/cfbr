"""FCS two-stage gates (DECISIONS "FCS games, two-stage fit", rule committed in 55fb749).

v1 = the published config, v2 = the tuned two-stage candidate, paired by game, each game predicted
point-in-time from its own slot. This module stays as the record of the test whether or not it passes."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from cfbr.backtest.calib_gate import _slope, paired_gain
from cfbr.backtest.harness import run_backtest
from cfbr.config import Config
from cfbr.data import Dataset
from cfbr.model.engine import Engine
from cfbr.report.rank import build_rankings

GATES = {
    1: {"seasons": list(range(2015, 2022)), "max_week": None, "role": "implementation check (in-sample)",
        "ci_floor": None},
    2: {"seasons": list(range(2022, 2026)), "max_week": None, "role": "out of sample", "ci_floor": -0.10},
    3: {"seasons": [2026], "max_week": 10, "role": "2026 regular season through week 10", "ci_floor": -0.10},
}


def gate_frame(v1: Config, v2: Config, ds: Dataset, seasons: list[int], max_week: int | None) -> pl.DataFrame:
    p1, _ = run_backtest(v1, seasons, ds)
    p2, _ = run_backtest(v2, seasons, ds)
    df = (p1.filter(pl.col("completed") & pl.col("margin").is_not_null())
          .select("game_id", "season", "week", "season_type", "home", "away", "home_team", "away_team", "margin",
                  "spread", pred_v1="pred_margin")
          .join(p2.select("game_id", pred_v2="pred_margin"), on="game_id", how="inner"))
    if max_week is not None:
        df = df.filter((pl.col("season_type") == "regular") & (pl.col("week") <= max_week))
    return df.with_columns(fcs_game=pl.col("home").str.starts_with("FCS") | pl.col("away").str.starts_with("FCS"))


def ats(df: pl.DataFrame, col: str) -> dict:
    """Against the closing line (Vegas home margin = -spread); pushes and unlined games skipped."""
    d = df.filter(pl.col("spread").is_not_null())
    line = -d["spread"].to_numpy()
    result, pick = np.sign(d["margin"].to_numpy() - line), np.sign(d[col].to_numpy() - line)
    ok = (result != 0) & (pick != 0)
    n, wins = int(ok.sum()), int((result[ok] == pick[ok]).sum())
    return {"n": n, "wins": wins, "rate": wins / n if n else math.nan}


def gate_report(df: pl.DataFrame, gate: int) -> dict:
    spec = GATES[gate]
    y = df["margin"].to_numpy()
    p1, p2 = df["pred_v1"].to_numpy(), df["pred_v2"].to_numpy()
    e1, e2 = np.abs(y - p1), np.abs(y - p2)
    fcs = df["fcs_game"].to_numpy()
    g = paired_gain(e1, e2)  # mean(e1 - e2): positive favours v2
    passes = bool(g["gain"] > 0 and (spec["ci_floor"] is None or g["ci95"][0] > spec["ci_floor"]))
    split = {}
    for name, mask in (("fbs_fbs", ~fcs), ("fbs_fcs", fcs)):
        split[name] = {**paired_gain(e1[mask], e2[mask]),
                       "mae_v1": float(e1[mask].mean()) if mask.any() else math.nan,
                       "mae_v2": float(e2[mask].mean()) if mask.any() else math.nan}
    return {"gate": gate, **spec, "n_games": int(y.size), "primary": {**g, "passes": passes},
            "mae": {"v1": float(e1.mean()), "v2": float(e2.mean())}, "split": split,
            "ats": {"v1": ats(df, "pred_v1"), "v2": ats(df, "pred_v2")},
            "fcs_slope": {"v1": _slope(p1[fcs], y[fcs]), "v2": _slope(p2[fcs], y[fcs])}}


def top25(cfg: Config, ds: Dataset, season: int, week: int) -> pl.DataFrame:
    rankings, _ = build_rankings(Engine(ds, cfg, last_completed=season - 1), season, week)
    return rankings.select("rank", "team", "rating").head(25)


def gate_markdown(r: dict, v1_model: dict, v2_model: dict, side_by_side: pl.DataFrame | None) -> str:
    def gain(x: dict) -> str:
        return f"{x['gain']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] (n={x['n']})"

    floor = "" if r["ci_floor"] is None else f" and 95% CI lower bound > {r['ci_floor']:+.2f}"
    lines = [f"# FCS two-stage gate {r['gate']}: {r['role']}", "",
             f"Seasons {r['seasons'][0]}–{r['seasons'][-1]}. Paired MAE gain v1 − v2: **{gain(r['primary'])}**. "
             f"Rule: gain > 0{floor} → **{'PASSES' if r['primary']['passes'] else 'FAILS'}**.", "",
             "| slice | MAE v1 | MAE v2 | gain |", "|---|---|---|---|",
             f"| all | {r['mae']['v1']:.3f} | {r['mae']['v2']:.3f} | {gain(r['primary'])} |"]
    for name in ("fbs_fbs", "fbs_fcs"):
        s = r["split"][name]
        lines.append(f"| {name} | {s['mae_v1']:.3f} | {s['mae_v2']:.3f} | {gain(s)} |")
    lines += ["", f"ATS vs close: v1 {r['ats']['v1']['wins']}/{r['ats']['v1']['n']} = {r['ats']['v1']['rate']:.3f}; "
                  f"v2 {r['ats']['v2']['wins']}/{r['ats']['v2']['n']} = {r['ats']['v2']['rate']:.3f}.",
              f"Slope of margin on prediction, FBS–FCS games: v1 {r['fcs_slope']['v1']:.3f}, "
              f"v2 {r['fcs_slope']['v2']:.3f}.", "",
              "| setting | v1 | v2 |", "|---|---|---|"]
    for k in ("alpha_ppa", "alpha_sr", "half_life_weeks", "fcs_two_stage", "fcs_weight"):
        lines.append(f"| {k} | {v1_model[k]} | {v2_model[k]} |")
    if side_by_side is not None:
        lines += ["", "## 2026 after week 10, top 25", "", "| rank | v1 | rating | v2 | rating |",
                  "|---|---|---|---|---|"]
        for row in side_by_side.iter_rows(named=True):
            lines.append(f"| {row['rank']} | {row['team']} | {row['rating']:.2f} | {row['team_v2']} | "
                         f"{row['rating_v2']:.2f} |")
    return "\n".join(lines) + "\n"


def latest_gate(results_dir: Path, gate: int) -> dict | None:
    runs = sorted(results_dir.glob(f"*-fcs-gate-{gate}"))
    return json.loads((runs[-1] / "gate.json").read_text()) if runs else None


def run_fcs_gate(v1: Config, v2: Config, ds: Dataset, gate: int) -> Path:
    spec = GATES[gate]
    df = gate_frame(v1, v2, ds, spec["seasons"], spec["max_week"])
    rep = gate_report(df, gate)
    rep["models"] = {"v1": v1.model.model_dump(), "v2": v2.model.model_dump()}
    side = None
    if gate == 3:
        side = top25(v1, ds, 2026, 10).join(
            top25(v2, ds, 2026, 10).rename({"team": "team_v2", "rating": "rating_v2"}), on="rank")
    out = v1.paths.results_dir / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-fcs-gate-{gate}")
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / "frame.parquet")
    (out / "gate.json").write_text(json.dumps(rep, indent=2, default=float))
    (out / "gate.md").write_text(gate_markdown(rep, rep["models"]["v1"], rep["models"]["v2"], side))
    return out
