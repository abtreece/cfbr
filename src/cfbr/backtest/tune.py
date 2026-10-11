"""Phase 6 tuning: Optuna (TPE) over the efficiency model, a grid for the
Massey baseline, one-at-a-time sensitivity sweeps, then a frozen config.

Only the tuning seasons are evaluated here. The holdout is evaluated by a
separate call (``holdout``) after ``config/tuned.toml`` is written.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import optuna
import polars as pl
from joblib import Parallel, delayed

from cfbr.config import Config, load_config

log = logging.getLogger(__name__)

_DS = None  # per-worker dataset cache


def apply_params(cfg: Config, params: dict) -> Config:
    cfg = cfg.model_copy(deep=True)
    m, gt, prep, base = cfg.model, cfg.prep.garbage_time, cfg.prep, cfg.baseline
    for k, v in params.items():
        if k in ("alpha_ppa", "alpha_sr", "half_life_weeks"):
            setattr(m, k, float(v))
        elif k == "gt_enabled":
            gt.enabled = bool(v)
        elif k in ("q2", "q3", "q4"):
            setattr(gt, k, int(v))
        elif k == "fcs_pooling":
            prep.fcs_pooling = v
        elif k in ("margin_cap", "massey_alpha"):
            if k == "margin_cap":
                base.margin_cap = int(v)
            else:
                base.alpha = float(v)
        else:
            raise KeyError(k)
    return cfg


def evaluate(config_path: str | None, params: dict, seasons: list[int]) -> dict:
    """Run one backtest in a worker process; returns overall metrics."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    global _DS
    from cfbr.backtest.harness import run_backtest
    from cfbr.data import Dataset

    cfg = apply_params(load_config(config_path), params)
    if _DS is None:
        _DS = Dataset.load(cfg)
    _, metrics = run_backtest(cfg, seasons, _DS)
    o = metrics["overall"]
    return {"params": params, "mae": o["mae"], "massey_mae": o["massey_mae"], "rmse": o["rmse"],
            "log_loss": o["log_loss"], "line_mae": o.get("line_mae"), "model_mae_lined": o.get("model_mae_lined"),
            "by_bucket": {b: m["mae"] for b, m in metrics["by_bucket"].items()},
            "prior_share_w07": next((r["mean_prior_share"] for r in metrics["prior_influence_by_slot"]
                                     if r["slot"] == "r07"), None)}


def suggest(trial: optuna.Trial) -> dict:
    p = {
        "alpha_ppa": trial.suggest_float("alpha_ppa", 20, 3000, log=True),
        "alpha_sr": trial.suggest_float("alpha_sr", 20, 3000, log=True),
        "fcs_pooling": trial.suggest_categorical("fcs_pooling", ["single", "conference"]),
        "gt_enabled": trial.suggest_categorical("gt_enabled", [True, False]),
    }
    hl = trial.suggest_categorical("recency", [False, True])
    p["half_life_weeks"] = trial.suggest_float("half_life_weeks", 2, 26) if hl else 0.0
    if p["gt_enabled"]:
        p["q2"] = trial.suggest_int("q2", 21, 52)
        p["q3"] = trial.suggest_int("q3", 14, 42)
        p["q4"] = trial.suggest_int("q4", 10, 35)
    return p


def run_search(config_path: str | None, seasons: list[int], n_trials: int, n_jobs: int, seed: int = 0) -> list[dict]:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(48, n_trials // 3), multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    pool = Parallel(n_jobs=n_jobs, backend="loky")
    results: list[dict] = []
    done = 0
    while done < n_trials:
        batch = [study.ask() for _ in range(min(n_jobs, n_trials - done))]
        params = [suggest(t) for t in batch]
        outs = pool(delayed(evaluate)(config_path, p, seasons) for p in params)
        for t, out in zip(batch, outs):
            study.tell(t, out["mae"])
            results.append(out)
        done += len(batch)
        best = min(results, key=lambda r: r["mae"])
        log.info("trials %d/%d  best MAE %.4f  %s", done, n_trials, best["mae"], best["params"])
    return results


def sensitivity(config_path: str | None, best: dict, seasons: list[int], n_jobs: int) -> dict[str, list[dict]]:
    """One-at-a-time sweeps around the best params."""
    grids: dict[str, list] = {
        "alpha_ppa": [best["alpha_ppa"] * f for f in (0.25, 0.5, 0.75, 1, 1.5, 2, 4)],
        "alpha_sr": [best["alpha_sr"] * f for f in (0.25, 0.5, 0.75, 1, 1.5, 2, 4)],
        "half_life_weeks": [0.0, 3, 6, 10, 16, 26],
        "fcs_pooling": ["single", "conference"],
        "gt_enabled": [True, False],
    }
    if best.get("gt_enabled"):
        for q, span in (("q2", 12), ("q3", 10), ("q4", 8)):
            c = best[q]
            grids[q] = sorted({max(7, c + d) for d in range(-span, span + 1, max(1, span // 4))})
    jobs = []
    for name, values in grids.items():
        for v in values:
            p = deepcopy(best)
            p[name] = v
            if name == "gt_enabled" and v and "q2" not in p:
                p |= {"q2": 38, "q3": 28, "q4": 22}
            jobs.append((name, v, p))
    outs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(evaluate)(config_path, p, seasons) for _, _, p in jobs)
    res: dict[str, list[dict]] = {}
    for (name, v, _), out in zip(jobs, outs):
        res.setdefault(name, []).append({"value": v, "mae": out["mae"]})
    return res


def massey_grid(config_path: str | None, best: dict, seasons: list[int], n_jobs: int) -> list[dict]:
    caps = [10, 14, 17, 21, 24, 28, 35, 42, 100]
    alphas = [0.3, 1, 3, 10, 30]
    jobs = [best | {"margin_cap": c, "massey_alpha": a} for c in caps for a in alphas]
    outs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(evaluate)(config_path, p, seasons) for p in jobs)
    return [{"margin_cap": p["margin_cap"], "massey_alpha": p["massey_alpha"], "massey_mae": o["massey_mae"]}
            for p, o in zip(jobs, outs)]


def to_toml(cfg: Config) -> str:
    """Serialize the tunable sections (paths and ingest come from defaults)."""
    d = cfg.model_dump(mode="json", exclude={"paths", "source"})

    def fmt(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        if isinstance(v, float):
            return repr(round(v, 4))
        return str(v)

    lines = ["# Frozen by `cfbr tune`. Do not edit after holdout evaluation.", ""]

    def section(name: str, body: dict):
        scalars = {k: v for k, v in body.items() if not isinstance(v, dict)}
        if scalars:
            lines.append(f"[{name}]")
            lines.extend(f"{k} = {fmt(v)}" for k, v in scalars.items())
            lines.append("")
        for k, v in body.items():
            if isinstance(v, dict):
                section(f"{name}.{k}", v)

    for name, body in d.items():
        section(name, body)
    return "\n".join(lines)


def write_frozen(base_config: str | None, best: dict, path: Path) -> Config:
    cfg = apply_params(load_config(base_config), best)
    path.write_text(to_toml(cfg))
    return load_config(path)


def summarize_trials(results: list[dict]) -> pl.DataFrame:
    rows = [{**r["params"], "mae": r["mae"], "massey_mae": r["massey_mae"], "prior_share_w07": r["prior_share_w07"]}
            for r in results]
    return pl.DataFrame(rows, infer_schema_length=None).sort("mae")


def describe_sensitivity(sens: dict[str, list[dict]], best_mae: float) -> list[str]:
    out = []
    for name, pts in sens.items():
        maes = np.array([p["mae"] for p in pts])
        out.append(f"| {name} | " + ", ".join(
            f"{_fmtv(p['value'])}: {p['mae']:.3f}" for p in pts) + f" | {maes.max() - maes.min():.3f} |")
    return out


def _fmtv(v) -> str:
    return f"{v:.0f}" if isinstance(v, float) and v >= 10 else (f"{v:.2f}" if isinstance(v, float) else str(v))


def _metric_rows(metrics: dict) -> list[str]:
    rows = ["| slice | games | MAE | RMSE | line MAE (same games) | model MAE (lined games) | Massey MAE | "
            "SU acc | log loss | Brier | ATS win % [95% CI] |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]

    def row(name: str, m: dict) -> str:
        ats = (f"{100 * m['ats_win_rate']:.1f} [{100 * m['ats_ci95_low']:.1f}, {100 * m['ats_ci95_high']:.1f}]"
               if m.get("ats_n") else "–")
        return (f"| {name} | {m['n_games']} | {m['mae']:.2f} | {m['rmse']:.2f} | {m.get('line_mae', float('nan')):.2f} | "
                f"{m.get('model_mae_lined', float('nan')):.2f} | {m['massey_mae']:.2f} | {m['su_accuracy']:.3f} | "
                f"{m['log_loss']:.3f} | {m['brier']:.3f} | {ats} |")

    rows.append(row("**overall**", metrics["overall"]))
    for b, m in metrics["by_bucket"].items():
        rows.append(row(b, m))
    for s, m in metrics["by_season"].items():
        rows.append(row(s, m))
    return rows


def _prior_rows(metrics: dict) -> list[str]:
    rows = ["| slot | mean prior share | model MAE | prior-only MAE | games |", "|---|---|---|---|---|"]
    for r in metrics["prior_influence_by_slot"]:
        rows.append(f"| {r['slot']} | {r['mean_prior_share']:.2f} | {r['mae']:.2f} | {r['prior_only_mae']:.2f} | "
                    f"{r['n_games']} |")
    return rows


def write_report(path: Path, best: dict, metrics: dict, default_metrics: dict, sens: dict, mgrid: list[dict],
                 n_trials: int, run_dir: Path) -> None:
    o, d = metrics["overall"], default_metrics["overall"]
    mb = min(mgrid, key=lambda r: r["massey_mae"])
    lines = [
        "# Tuning report", "",
        f"Tuning seasons {', '.join(metrics['by_season'])}; objective = overall MAE of predicted margin. "
        f"{n_trials} Optuna TPE trials run in parallel, then one-at-a-time sensitivity sweeps around the best point. "
        "The holdout seasons were not evaluated until `config/tuned.toml` was written.", "",
        "## Best config", "",
        "| parameter | value |", "|---|---|",
        *[f"| {k} | {_fmtv(v) if not isinstance(v, float) or v >= 10 else f'{v:.3f}'} |" for k, v in best.items()],
        "",
        f"Tuning-season MAE: **{o['mae']:.3f}** tuned vs {d['mae']:.3f} with `config/default.toml`. "
        f"Closing line on the same games: {o['line_mae']:.3f} (model {o['model_mae_lined']:.3f}, "
        f"gap {o['model_mae_lined'] - o['line_mae']:+.3f}).", "",
        f"Massey baseline grid best: cap {mb['margin_cap']}, alpha {mb['massey_alpha']} → MAE {mb['massey_mae']:.3f}.",
        "",
        "## Sensitivity (MAE on tuning seasons, one parameter varied at a time)", "",
        "| parameter | value: MAE | range |", "|---|---|---|",
        *describe_sensitivity(sens, o["mae"]), "",
        "## Tuning-season metrics (tuned config)", "", *_metric_rows(metrics), "",
        "## Prior influence by week (tuning seasons, tuned config)", "", *_prior_rows(metrics), "",
        f"Run: `{run_dir}`", "",
    ]
    path.write_text("\n".join(lines))


def append_holdout(path: Path, metrics: dict, run_dir: Path) -> None:
    o = metrics["overall"]
    lines = [
        "## Holdout metrics (frozen config, evaluated once)", "",
        f"Overall MAE {o['mae']:.3f} vs closing line {o['line_mae']:.3f} on the same games "
        f"(model {o['model_mae_lined']:.3f}, gap {o['model_mae_lined'] - o['line_mae']:+.3f}); "
        f"Massey baseline {o['massey_mae']:.3f}.", "",
        *_metric_rows(metrics), "",
        "### Prior influence by week (holdout)", "", *_prior_rows(metrics), "",
        f"Run: `{run_dir}`", "",
    ]
    with open(path, "a") as f:
        f.write("\n".join(lines))
