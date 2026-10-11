"""Weekly season-simulation output: out/{season}/wNN/season_sim.csv and season_sim.md."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from cfbr.config import Config
from cfbr.model.engine import Engine
from cfbr.sim.season import Rules, build_inputs, simulate, summary

log = logging.getLogger(__name__)


def rules_from(cfg: Config) -> Rules:
    s = cfg.sim
    return Rules(power_conferences=list(s.power_conferences), group_of_six_bid=s.group_of_six_bid,
                 notre_dame_top=s.notre_dame_top, field_size=s.field_size, byes=s.byes,
                 losses_weight=s.losses_weight, champion_bonus=s.champion_bonus, power_bonus=s.power_bonus)


def run_sim(eng: Engine, season: int, week: int) -> Path:
    """Simulate the rest of ``season`` from the ratings available after regular-season
    ``week`` (the snapshot used for week ``week + 1`` predictions)."""
    cfg = eng.cfg
    as_of = eng.slot_as_of(season, f"r{week + 1:02d}")
    if as_of is None:  # regular season over: simulate from the start of the postseason
        as_of = eng.season_slots(season)["as_of"].max()
    inp = build_inputs(eng, season, as_of)
    out = simulate(inp, n_sims=cfg.sim.n_sims, tau=cfg.sim.tau, seed=cfg.sim.seed, rules=rules_from(cfg))
    s = summary(inp, out)
    d = cfg.paths.out_dir / str(season) / f"w{week:02d}"
    d.mkdir(parents=True, exist_ok=True)
    s.write_csv(d / "season_sim.csv")
    (d / "season_sim.md").write_text(markdown(s, cfg, season, week, len(inp.g_mean)))
    log.info("season sim: %d simulations, %d games left, %s", cfg.sim.n_sims, len(inp.g_mean), d / "season_sim.csv")
    return d / "season_sim.csv"


def _p(x: float) -> str:
    if x >= 0.995:
        return ">99%"
    if 0 < x < 0.005:
        return "<1%"
    return f"{100 * x:.0f}%"


def markdown(s: pl.DataFrame, cfg: Config, season: int, week: int, games_left: int) -> str:
    lines = [f"# {season} season simulation after week {week}", "",
             f"{cfg.sim.n_sims:,} simulated seasons of the {games_left} remaining regular-season games, from the "
             "ratings available after this week (point-in-time), then conference title games and the 12-team "
             "playoff. Probabilities, not predictions.", "",
             "**Approximations:** title-game pairings use conference win percentage, then strength (not each "
             "conference's own tiebreakers); the committee's ranking is approximated by team strength minus "
             f"{cfg.sim.losses_weight:g} points per loss plus {cfg.sim.power_bonus:g} for power-conference teams "
             "(fitted to the 2015–2021 final rankings: 86% of the top 12). Playoff field: SEC, Big Ten, ACC "
             f"and Big 12 champions, the highest-ranked Group of Six {cfg.sim.group_of_six_bid}, Notre Dame if "
             "top 12, then at-large; seeds 1–4 get byes.", "",
             "| team | conf | rating | record | exp. wins [80%] | undefeated | conf title | playoff | bye | champion |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    top = s.filter((pl.col("p_playoff") >= 0.01) | (pl.col("p_conf_title") >= 0.05)).sort("p_playoff", descending=True)
    for r in top.iter_rows(named=True):
        lines.append(f"| {r['team']} | {r['conference']} | {r['rating']:+.1f} | {r['wins_now']}-{r['losses_now']} | "
                     f"{r['exp_wins']:.1f} [{r['wins_p10']}–{r['wins_p90']}] | {_p(r['p_undefeated'])} | "
                     f"{_p(r['p_conf_title'])} | {_p(r['p_playoff'])} | {_p(r['p_bye'])} | {_p(r['p_champion'])} |")
    lines += ["", "## Conference title odds", ""]
    for conf in sorted(set(s["conference"]) - {"FBS Independents"}):
        c = s.filter(pl.col("conference") == conf).sort("p_conf_title", descending=True).head(5)
        lines.append(f"- **{conf}:** " + ", ".join(f"{r['team']} {_p(r['p_conf_title'])}" for r in c.iter_rows(named=True)
                                                   if r["p_conf_title"] >= 0.005))
    return "\n".join(lines) + "\n"
