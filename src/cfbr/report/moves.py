"""What moved (spec 2026-11-08-what-moved-design.md): split a week's rating change per FBS team
into older plays fading, the team's own game and other games, plus per-game efficiency against
the pre-game expectation. Display only; every fit goes through Engine._fit (the published path)."""

from __future__ import annotations

import numpy as np
import polars as pl

from cfbr.model.engine import Engine, team_ratings_points

MAX_NAMES = 5


def _points(eng: Engine, season: int, snap: pl.DataFrame) -> pl.DataFrame:
    return team_ratings_points(snap, eng.calibration(season), eng.cfg).select("team", "rating")


def rating_moves(eng: Engine, season: int, prev_as_of, as_of) -> pl.DataFrame:
    """One row per FBS team: the rating change from ``prev_as_of`` to ``as_of`` and its three parts
    (move_decay + move_own + move_others == rating_change), plus who passed whom in the ranks."""
    a, fbs = eng.arrays(season), eng.fbs(season)
    priors = eng.priors(season)
    prev = _points(eng, season, eng.snapshot(season, prev_as_of)).rename({"rating": "prev_rating"})
    cur = _points(eng, season, eng.snapshot(season, as_of))
    k_prev, k = a.n_before(prev_as_of), a.n_before(as_of)
    old = np.zeros(k, bool)
    old[:k_prev] = True
    faded = _points(eng, season, eng._fit(season, priors, as_of, mask=old)).rename({"rating": "faded"})
    idx = {t: i for i, t in enumerate(a.teams)}
    cur_rating = dict(cur.iter_rows())
    own = {}
    for team in fbs:
        i = idx[team]
        mine = ~old & ((a.off[:k] == i) | (a.de[:k] == i))
        if not mine.any():
            own[team] = 0.0
            continue
        without = _points(eng, season, eng._fit(season, priors, as_of, mask=~mine))
        own[team] = cur_rating[team] - without.filter(pl.col("team") == team)["rating"].item()
    df = (cur.filter(pl.col("team").is_in(fbs))
          .with_columns(rank=pl.col("rating").rank("ordinal", descending=True).cast(pl.Int32))
          .join(prev.filter(pl.col("team").is_in(fbs))
                .with_columns(prev_rank=pl.col("prev_rating").rank("ordinal", descending=True).cast(pl.Int32)),
                on="team")
          .join(faded, on="team")
          .with_columns(rating_change=pl.col("rating") - pl.col("prev_rating"),
                        move_decay=pl.col("faded") - pl.col("prev_rating"),
                        move_own=pl.col("team").replace_strict(own, return_dtype=pl.Float64))
          .with_columns(move_others=pl.col("rating_change") - pl.col("move_decay") - pl.col("move_own"))
          .sort("rank"))
    ranks = list(zip(df["team"], df["prev_rank"], df["rank"]))  # in current-rank order
    by, gone = [], []
    for _, pr, cr in ranks:
        by.append([t for t, p, c in ranks if p > pr and c < cr])
        gone.append([t for t, p, c in ranks if p < pr and c > cr])
    return df.with_columns(
        passed_by=pl.Series([x[:MAX_NAMES] for x in by], dtype=pl.List(pl.Utf8)),
        passed=pl.Series([x[:MAX_NAMES] for x in gone], dtype=pl.List(pl.Utf8)),
        n_passed_by=pl.Series([len(x) for x in by], dtype=pl.Int32),
        n_passed=pl.Series([len(x) for x in gone], dtype=pl.Int32),
    ).select("team", "rating", "prev_rating", "rating_change", "rank", "prev_rank", "move_decay", "move_own",
             "move_others", "passed_by", "passed", "n_passed_by", "n_passed")


def game_efficiency(eng: Engine, season: int) -> pl.DataFrame:
    """Per completed game: each side's PPA per play and success rate (the plays the fit used) and
    the expected edges from the game's pre-game slot snapshot, from the home side's view:
    (O_h + D_h) - (O_a + D_a) + 2·H·home, home = 0 at a neutral site. The intercept cancels."""
    if eng.as_of_mode != "slot":
        raise ValueError("game_efficiency needs slot as-of mode")
    stats = (eng.season_plays(season)
             .group_by("game_id", "offense")
             .agg(ppa=pl.col("ppa").mean(), sr=pl.col("success").cast(pl.Float64).mean()))
    g = eng.season_games_with_snapshots(season).filter(pl.col("completed"))
    snaps = eng.snapshots(season)
    h_ppa = {slot: float(s["home_ppa"][0]) for slot, s in snaps.items()}
    h_sr = {slot: float(s["home_sr"][0]) for slot, s in snaps.items()}
    hm = pl.when(pl.col("neutral_site")).then(0.0).otherwise(1.0)
    g = g.with_columns(h_ppa=pl.col("slot").replace_strict(h_ppa, return_dtype=pl.Float64),
                       h_sr=pl.col("slot").replace_strict(h_sr, return_dtype=pl.Float64))

    def side(s: str) -> pl.DataFrame:
        return stats.rename({"offense": s, "ppa": f"{s}_ppa", "sr": f"{s}_sr"})

    return (g.join(side("home"), on=["game_id", "home"], how="left")
            .join(side("away"), on=["game_id", "away"], how="left")
            .with_columns(
                exp_ppa_edge=pl.col("o_ppa_h") + pl.col("d_ppa_h") - pl.col("o_ppa_a") - pl.col("d_ppa_a")
                + 2 * pl.col("h_ppa") * hm,
                exp_sr_edge=pl.col("o_sr_h") + pl.col("d_sr_h") - pl.col("o_sr_a") - pl.col("d_sr_a")
                + 2 * pl.col("h_sr") * hm)
            .select("game_id", "home_ppa", "away_ppa", "home_sr", "away_sr", "exp_ppa_edge", "exp_sr_edge"))


def moves_csv_frame(m: pl.DataFrame) -> pl.DataFrame:
    """CSV form of rating_moves: floats to 2 dp, name lists joined with '; '."""
    floats = [c for c in ("rating", "prev_rating", "rating_change", "move_decay", "move_own", "move_others")
              if c in m.columns]
    return m.with_columns([pl.col(c).round(2) for c in floats]
                          + [pl.col(c).list.join("; ") for c in ("passed_by", "passed")])
