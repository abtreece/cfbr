"""Monte Carlo season simulation: conference titles, playoff odds and win totals.

Everything is point-in-time: ratings come from the engine's snapshot as of the
simulation date (plays that kicked off before it), finished games keep their real
results, and every remaining game is predicted from that one snapshot.

Each simulated season draws a per-team strength offset once (``tau`` points, so a
team's remaining games move together) and adds independent game noise with
``sqrt(sigma^2 - 2 tau^2)``, which keeps each game's total spread equal to the
calibrated ``sigma``. ``tau`` is chosen on the tuning seasons (see DECISIONS.md).
When the model session's rating draws are approved they can replace the offsets.

Postseason rules (2026, CFP 12-team format with power-conference auto bids) are
approximations, labelled as such in every output:
  - Conference title game: the top two in conference win percentage; ties broken
    by simulated strength (head-to-head and the conferences' own tiebreakers aren't
    modelled). Played at a neutral site.
  - CFP ranking proxy: simulated strength minus 6.5 points per loss, plus 10 for a
    power-conference team (fitted to the 2015-2021 final committee rankings: 86% of
    the top 12). The committee's judgment isn't modelled beyond that.
  - Field: SEC, Big Ten, ACC, Big 12 champions; the highest-ranked Group of Six
    champion (configurable); Notre Dame if ranked in the top 12; at-large by rank to
    12. Seeds 1-12 by rank; seeds 1-4 get byes; first round at the higher seed,
    later rounds neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import duckdb
import numpy as np
import polars as pl

from cfbr.data import STALE_GAME
from cfbr.model.calibrate import Calibration
from cfbr.model.engine import Engine

POWER = ["SEC", "Big Ten", "ACC", "Big 12"]
INDEPENDENT = "FBS Independents"


@dataclass
class Rules:
    power_conferences: list[str] = field(default_factory=lambda: list(POWER))
    group_of_six_bid: str = "champion"   # "champion" or "team"
    notre_dame_top: int = 12
    field_size: int = 12
    byes: int = 4
    title_games: bool = True
    # Committee-ranking proxy: strength - losses_weight * losses (+ bonuses); fitted on the
    # tuning seasons' final committee rankings (see DECISIONS.md).
    losses_weight: float = 6.5
    champion_bonus: float = 0.0
    power_bonus: float = 10.0


@dataclass
class Inputs:
    season: int
    teams: list[str]                 # FBS teams (real names)
    conference: list[str]
    strength: np.ndarray             # points vs an average FBS team, neutral field
    b_home: float
    sigma: float
    wins0: np.ndarray                # results so far (FBS teams)
    losses0: np.ndarray
    conf_wins0: np.ndarray
    conf_losses0: np.ndarray
    g_home: np.ndarray               # remaining games: team index, or -1 for a non-FBS opponent
    g_away: np.ndarray
    g_mean: np.ndarray               # expected home margin from the snapshot
    g_conf: np.ndarray               # conference game
    g_opp_strength_home: np.ndarray  # strength used for a non-FBS side (pooled FCS rating)
    g_opp_strength_away: np.ndarray
    as_of: object = None
    # Conference title games already on the schedule: conference -> (team a, team b, winner or -1).
    scheduled_titles: dict = field(default_factory=dict)


def conference_game_ids(db_path, season: int) -> set[int]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute("SELECT game_id FROM games WHERE season = ? AND conference_game", [season]).fetchall()
    return {r[0] for r in rows}


def title_game_ids(games: pl.DataFrame, conf_ids: set[int]) -> set[int]:
    """Conference title games (CFBD carries no flag): for each conference, its last
    December conference game, when it comes in a later week than all of that
    conference's other games. Army-Navy is skipped: both are American Athletic members
    and it's played the week after title games. Some title games are at the higher seed's stadium, not a neutral site."""
    army_navy = (pl.col("home_team").is_in(["Army", "Navy"]) & pl.col("away_team").is_in(["Army", "Navy"]))
    c = games.filter((pl.col("season_type") == "regular") & pl.col("game_id").is_in(list(conf_ids))
                     & pl.col("home_conference").is_not_null() & ~army_navy)
    out = set()
    for (conf,), g in c.group_by("home_conference"):
        dec = g.filter(pl.col("kickoff_ts").dt.month() == 12).sort("kickoff_ts")
        if dec.height == 0:
            continue
        last = dec.row(-1, named=True)
        if g.filter(pl.col("game_id") != last["game_id"])["week"].max() < last["week"]:
            out.add(last["game_id"])
    return out


def build_inputs(eng: Engine, season: int, as_of, only_completed_later: bool = False,
                 now: datetime | None = None) -> Inputs:
    """``only_completed_later``: for historical tests, simulate only remaining games that
    were eventually played (cancelled games had no result to compare against). Live mode
    likewise skips games still not completed ``STALE_GAME`` after their kickoff (postponed
    or cancelled; see cfbr.data): they are not games left to play."""
    now = now or datetime.now(UTC)
    cal: Calibration = eng.calibration(season)
    snap = eng.snapshot(season, as_of)
    rating, _, _ = cal.rating(snap["o_ppa"], snap["d_ppa"], snap["o_sr"], snap["d_sr"])
    strength_of = dict(zip(snap["team"].to_list(), np.asarray(rating, dtype=float).tolist()))
    ts = eng.ds.team_status.filter((pl.col("season") == season) & (pl.col("classification") == "fbs"))
    teams = sorted(ts["team"].to_list())
    conf = dict(zip(ts["team"].to_list(), ts["conference"].to_list()))
    idx = {t: i for i, t in enumerate(teams)}
    conf_ids = conference_game_ids(eng.cfg.paths.db_path, season)
    g = eng.games.filter((pl.col("season") == season) & (pl.col("season_type") == "regular"))
    # Title games on the schedule are played in the postseason step, never as regular games.
    titles = title_game_ids(g, conf_ids)
    scheduled = {}
    for r in g.filter(pl.col("game_id").is_in(list(titles))).iter_rows(named=True):
        a, b = idx.get(r["home_team"], -1), idx.get(r["away_team"], -1)
        if a < 0 or b < 0:
            continue
        played = r["completed"] and r["margin"] is not None and r["kickoff_ts"] < as_of
        if only_completed_later and not (r["completed"] and r["margin"] is not None):
            continue
        scheduled[conf[r["home_team"]]] = (a, b, (a if r["margin"] > 0 else b) if played else -1)
    g = g.filter(~pl.col("game_id").is_in(list(titles)))
    done = g.filter(pl.col("completed") & (pl.col("kickoff_ts") < as_of) & pl.col("margin").is_not_null())
    todo = g.filter(pl.col("kickoff_ts") >= as_of)
    if only_completed_later:
        todo = todo.filter(pl.col("completed") & pl.col("margin").is_not_null())
    else:
        stale = pl.lit(now - STALE_GAME).cast(g.schema["kickoff_ts"])
        todo = todo.filter(pl.col("completed") | (pl.col("kickoff_ts") > stale))

    n = len(teams)
    w, l, cw, cl = (np.zeros(n) for _ in range(4))
    for r in done.select("game_id", "home_team", "away_team", "margin").iter_rows(named=True):
        hw = r["margin"] > 0
        for team, won in ((r["home_team"], hw), (r["away_team"], not hw)):
            if team in idx:
                i = idx[team]
                w[i] += won
                l[i] += not won
                if r["game_id"] in conf_ids:
                    cw[i] += won
                    cl[i] += not won

    def side(team, pooled):
        return (idx.get(team, -1), strength_of.get(pooled, 0.0))

    hs = [side(t, p) for t, p in zip(todo["home_team"], todo["home"])]
    as_ = [side(t, p) for t, p in zip(todo["away_team"], todo["away"])]
    home_adv = np.where(todo["neutral_site"].to_numpy(), 0.0, cal.b_home)
    mean = np.array([h[1] - a[1] for h, a in zip(hs, as_)]) + home_adv
    return Inputs(
        season=season, teams=teams, conference=[conf[t] for t in teams],
        strength=np.array([strength_of.get(t, 0.0) for t in teams]), b_home=cal.b_home, sigma=cal.sigma,
        wins0=w, losses0=l, conf_wins0=cw, conf_losses0=cl,
        g_home=np.array([h[0] for h in hs], dtype=int), g_away=np.array([a[0] for a in as_], dtype=int),
        g_mean=mean, g_conf=todo["game_id"].is_in(list(conf_ids)).to_numpy(),
        g_opp_strength_home=np.array([h[1] for h in hs]), g_opp_strength_away=np.array([a[1] for a in as_]),
        as_of=as_of, scheduled_titles=scheduled)


def _incidence(team_idx: np.ndarray, n_teams: int) -> np.ndarray:
    m = np.zeros((len(team_idx), n_teams), dtype=np.float32)
    ok = team_idx >= 0
    m[np.arange(len(team_idx))[ok], team_idx[ok]] = 1.0
    return m


def _play(mean: np.ndarray, sd: float, rng) -> np.ndarray:
    """True where the first side wins."""
    return (mean + rng.standard_normal(mean.shape) * sd) > 0


def simulate(inp: Inputs, n_sims: int = 100_000, tau: float = 0.0, seed: int = 0, rules: Rules | None = None,
             postseason: bool = True, chunk: int = 10_000) -> dict[str, np.ndarray]:
    """Per-team outcome counts over ``n_sims`` seasons: reg-season win distribution,
    conference titles, playoff berths, byes and national titles."""
    rules = rules or Rules()
    rng = np.random.default_rng(seed)
    n = len(inp.teams)
    g_sd = float(np.sqrt(max(inp.sigma ** 2 - 2 * tau ** 2, 1e-6)))
    H, A = _incidence(inp.g_home, n), _incidence(inp.g_away, n)
    Hc, Ac = H * inp.g_conf[:, None], A * inp.g_conf[:, None]
    out = {k: np.zeros(n) for k in ("conf_title", "title_game", "playoff", "bye", "champion", "undefeated")}
    out["wins_hist"] = np.zeros((n, 20))
    out["wins_sum"] = np.zeros(n)
    conf_names = sorted({c for c in inp.conference if c != INDEPENDENT})
    conf_arr = np.array(inp.conference)
    done = 0
    while done < n_sims:
        k = min(chunk, n_sims - done)
        u = rng.standard_normal((k, n)) * tau if tau > 0 else np.zeros((k, n))
        # Team offsets shift each game's mean; non-FBS sides get none.
        uh = np.where(inp.g_home >= 0, u[:, np.clip(inp.g_home, 0, None)], 0.0)
        ua = np.where(inp.g_away >= 0, u[:, np.clip(inp.g_away, 0, None)], 0.0)
        home_win = _play(inp.g_mean[None, :] + uh - ua, g_sd, rng).astype(np.float32)
        wins = inp.wins0 + home_win @ H + (1 - home_win) @ A
        games_left = H.sum(0) + A.sum(0)
        losses = inp.losses0 + games_left - (wins - inp.wins0)
        cwins = inp.conf_wins0 + home_win @ Hc + (1 - home_win) @ Ac
        cgames = inp.conf_wins0 + inp.conf_losses0 + Hc.sum(0) + Ac.sum(0)
        out["wins_sum"] += wins.sum(0)
        wi = np.clip(wins.astype(int), 0, 19)
        for t in range(n):
            out["wins_hist"][t] += np.bincount(wi[:, t], minlength=20)[:20]
        out["undefeated"] += (losses == 0).sum(0)
        if postseason:
            _postseason(inp, rules, rng, g_sd, u + inp.strength[None, :], wins, losses, cwins, cgames,
                        conf_names, conf_arr, out)
        done += k
    out["n_sims"] = np.array(n_sims)
    return out


def _postseason(inp, rules, rng, g_sd, strength, wins, losses, cwins, cgames, conf_names, conf_arr, out):
    k, n = wins.shape
    losses = losses.copy()
    champ_of = {}
    pct = np.where(cgames > 0, cwins / np.maximum(cgames, 1), 0.0)
    rows = np.arange(k)
    for c in conf_names:
        members = np.where(conf_arr == c)[0]
        if len(members) < 2:
            continue
        key = pct[:, members] * 1000 + strength[:, members] / 100     # win pct, then strength
        order = np.argsort(-key, axis=1)
        a, b = members[order[:, 0]], members[order[:, 1]]
        if c in inp.scheduled_titles:                                  # the real pairing (and result, once played)
            ta, tb, winner = inp.scheduled_titles[c]
            a, b = np.full(k, ta), np.full(k, tb)
            out["title_game"] += np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
            if winner >= 0:
                champ = np.full(k, winner)
                losses[rows, tb if winner == ta else ta] += 1
            else:
                a_wins = _play(strength[rows, a] - strength[rows, b], g_sd, rng)
                champ = np.where(a_wins, a, b)
                losses[rows, np.where(a_wins, b, a)] += 1
        elif rules.title_games:
            out["title_game"] += np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
            a_wins = _play(strength[rows, a] - strength[rows, b], g_sd, rng)
            champ = np.where(a_wins, a, b)
            loser = np.where(a_wins, b, a)
            losses[rows, loser] += 1
        else:
            champ = a
        champ_of[c] = champ
        out["conf_title"] += np.bincount(champ, minlength=n)
    is_champ = np.zeros((k, n), dtype=bool)
    for c, ch in champ_of.items():
        is_champ[rows, ch] = True
    power = np.isin(conf_arr, rules.power_conferences)[None, :]
    # Committee-ranking proxy (see Rules).
    rank_key = (strength - rules.losses_weight * losses + rules.champion_bonus * is_champ
                + rules.power_bonus * power)
    order = np.argsort(-rank_key, axis=1)                  # best first
    rank = np.empty_like(order)
    rank[rows[:, None], order] = np.arange(n)[None, :]
    g6 = np.isin(conf_arr, [c for c in conf_names if c not in rules.power_conferences])
    in_field = np.zeros((k, n), dtype=bool)
    for c in rules.power_conferences:
        if c in champ_of:
            in_field[rows, champ_of[c]] = True
    g6_pool = g6[None, :] & (is_champ if rules.group_of_six_bid == "champion" else True)
    g6_rank = np.where(g6_pool, rank, n + 1)
    best_g6 = g6_rank.argmin(axis=1)
    has_g6 = g6_rank.min(axis=1) <= n
    in_field[rows[has_g6], best_g6[has_g6]] = True
    if "Notre Dame" in inp.teams:
        nd = inp.teams.index("Notre Dame")
        in_field[:, nd] |= rank[:, nd] < rules.notre_dame_top
    # At-large by rank until the field is full.
    for s in range(k):
        need = rules.field_size - int(in_field[s].sum())
        if need > 0:
            for t in order[s]:
                if not in_field[s, t]:
                    in_field[s, t] = True
                    need -= 1
                    if need == 0:
                        break
    field_rank = np.where(in_field, rank, n + 1)
    seeds = np.argsort(field_rank, axis=1)[:, :rules.field_size]   # seed 1..12 = best-ranked in field
    out["playoff"] += np.bincount(seeds.ravel(), minlength=n)
    out["bye"] += np.bincount(seeds[:, :rules.byes].ravel(), minlength=n)
    out["champion"] += np.bincount(_bracket(seeds, strength, inp.b_home, g_sd, rng), minlength=n)


def _bracket(seeds, strength, b_home, g_sd, rng):
    """12-team bracket (or the old 4-team one): 5v12, 6v11, 7v10, 8v9 at the higher seed; 1-4 host winners at
    neutral sites (1 v 8/9, 2 v 7/10, 3 v 6/11, 4 v 5/12); then 1/8 v 4/5 and 2/7 v 3/6."""
    rows = np.arange(len(seeds))

    def game(x, y, home):
        return np.where(_play(strength[rows, x] - strength[rows, y] + (b_home if home else 0.0), g_sd, rng), x, y)

    s = [None] + [seeds[:, i] for i in range(seeds.shape[1])]
    if seeds.shape[1] == 4:   # the 2014-2023 format: 1 v 4, 2 v 3, final; all neutral
        return game(game(s[1], s[4], False), game(s[2], s[3], False), False)
    if seeds.shape[1] != 12:
        raise ValueError("bracket supports 4- or 12-team fields")
    r1 = {8: game(s[8], s[9], True), 7: game(s[7], s[10], True), 6: game(s[6], s[11], True), 5: game(s[5], s[12], True)}
    q1, q2, q3, q4 = (game(s[1], r1[8], False), game(s[2], r1[7], False),
                      game(s[3], r1[6], False), game(s[4], r1[5], False))
    return game(game(q1, q4, False), game(q2, q3, False), False)


def cum_ge(hist: np.ndarray, w: int) -> np.ndarray:
    """P(regular-season wins >= w), title games excluded."""
    return hist[:, w:].sum(axis=1)


def summary(inp: Inputs, out: dict) -> pl.DataFrame:
    n_sims = float(out["n_sims"])
    hist = out["wins_hist"] / n_sims
    cum = hist.cumsum(axis=1)
    lo = (cum >= 0.10).argmax(axis=1)
    hi = (cum >= 0.90).argmax(axis=1)
    return pl.DataFrame({
        "team": inp.teams, "conference": inp.conference, "rating": np.round(inp.strength, 2),
        "wins_now": inp.wins0.astype(int), "losses_now": inp.losses0.astype(int),
        "exp_wins": np.round(out["wins_sum"] / n_sims, 2), "wins_p10": lo, "wins_p90": hi,
        "p_undefeated": out["undefeated"] / n_sims, "p_bowl_eligible": cum_ge(hist, 6),
        "p_title_game": out["title_game"] / n_sims,
        "p_conf_title": out["conf_title"] / n_sims, "p_playoff": out["playoff"] / n_sims,
        "p_bye": out["bye"] / n_sims, "p_champion": out["champion"] / n_sims,
    }).sort("p_playoff", "exp_wins", descending=True)
