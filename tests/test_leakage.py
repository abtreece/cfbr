"""Point-in-time correctness: nothing from week w or later may reach a week-w prediction."""

import dataclasses
import random
from datetime import timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cfbr.data import training_slice
from cfbr.model.engine import Engine
from cfbr.model.schemes import training_seasons

ALL = range(2015, 2026)


# -- cross-season fit schemes -----------------------------------------------------

@pytest.mark.parametrize("target", range(2022, 2027))
def test_holdout_and_live_seasons_train_only_on_the_past(cfg, target):
    assert all(s < target for s in training_seasons(target, cfg, ALL))


@pytest.mark.parametrize("target", range(2015, 2022))
def test_tuning_seasons_leave_one_out_within_tuning_window(cfg, target):
    seasons = training_seasons(target, cfg, ALL)
    assert target not in seasons
    assert all(2015 <= s <= 2021 for s in seasons)


# -- within-season as-of slicing --------------------------------------------------

def test_training_rows_precede_as_of(cfg, dataset):
    eng = Engine(dataset, cfg, last_completed=2025)
    slots = eng.slots.filter(pl.col("season").is_between(2015, 2025))
    rng = random.Random(0)
    for r in rng.sample(list(slots.iter_rows(named=True)), 40):
        plays = training_slice(eng.plays, r["season"], r["as_of"])
        if plays.height:
            assert plays["kickoff_ts"].max() < r["as_of"]
            assert plays["season"].unique().to_list() == [r["season"]]
        games = training_slice(eng.games, r["season"], r["as_of"])
        if games.height:
            assert games["kickoff_ts"].max() < r["as_of"]


def test_engine_prefix_fit_matches_explicit_slice(cfg, dataset):
    """The engine's fast prefix slicing must equal fitting on training_slice rows."""
    from cfbr.model.ridge import fit_efficiency

    eng = Engine(dataset, cfg, last_completed=2025)
    season = 2018
    for slot in ("r01", "r05", "r12", "p1"):
        as_of = eng.slot_as_of(season, slot)
        snap = eng.snapshot(season, as_of)
        plays = training_slice(eng.plays, season, as_of)
        assert eng.arrays(season).n_before(as_of) == plays.height
        pr = pl.DataFrame({"team": eng.entities(season)}).join(eng.priors(season), on="team", how="left").fill_null(0.0)
        ref = fit_efficiency(plays, "ppa", cfg.model.alpha_ppa, eng.entities(season),
                             pr["o_ppa"].to_numpy(), pr["d_ppa"].to_numpy(), center_on=eng.fbs(season))
        assert abs(snap["o_ppa"].to_numpy() - ref.o).max() < 1e-9
        assert abs(snap["d_ppa"].to_numpy() - ref.d).max() < 1e-9


def test_as_of_is_earliest_kickoff_of_slot(cfg, dataset):
    eng = Engine(dataset, cfg, last_completed=2025)
    g = eng.games.filter(pl.col("season") == 2019)
    for r in eng.season_slots(2019).iter_rows(named=True):
        assert g.filter(pl.col("slot") == r["slot"])["kickoff_ts"].min() == r["as_of"]
        # No game in this slot kicks off before its as-of time.
        assert (g.filter(pl.col("slot") == r["slot"])["kickoff_ts"] >= r["as_of"]).all()


# -- canary: a fake future game must not move earlier ratings -----------------------

CANARY_SEASON = 2023
CANARY_SLOT = "r10"


def _with_canary(ds, eng):
    as_of = eng.slot_as_of(CANARY_SEASON, CANARY_SLOT)
    kickoff = as_of + timedelta(days=2)
    home, away = "Vanderbilt", "Alabama"
    game_id = 999_999_999
    n = 90
    plays = pl.DataFrame({
        "season": [CANARY_SEASON] * n, "week": [10] * n, "season_type": ["regular"] * n,
        "game_id": [game_id] * n, "play_id": [f"{game_id}{i:04d}" for i in range(n)],
        "kickoff_ts": [kickoff] * n,
        "offense_team": [home] * n, "defense_team": [away] * n,
        "offense_fcs": [False] * n, "defense_fcs": [False] * n,
        "offense_conference": ["SEC"] * n, "defense_conference": ["SEC"] * n,
        "home_team": [home] * n, "neutral_site": [False] * n,
        "period": [1] * n, "down": [1] * n, "distance": [10] * n, "yards_gained": [80] * n,
        "margin_at_snap": [0] * n, "play_type": ["Rushing Touchdown"] * n,
        "ppa": [12.0] * n, "success": [True] * n,
    }).cast(ds.scrimmage.schema)
    game = ds.games.filter(pl.col("season") == CANARY_SEASON).head(1).with_columns(
        game_id=pl.lit(game_id, pl.Int64), kickoff_ts=pl.lit(kickoff).cast(ds.games.schema["kickoff_ts"]),
        week=pl.lit(10, pl.Int32), slot=pl.lit(CANARY_SLOT), slot_order=pl.lit(10, ds.games.schema["slot_order"]),
        home_team=pl.lit(home), away_team=pl.lit(away), home_points=pl.lit(150, ds.games.schema["home_points"]),
        away_points=pl.lit(0, ds.games.schema["away_points"]), home_fcs=pl.lit(False), away_fcs=pl.lit(False),
        neutral_site=pl.lit(False), completed=pl.lit(True),
    )
    return dataclasses.replace(ds, scrimmage=pl.concat([ds.scrimmage, plays]), games=pl.concat([ds.games, game]))


def test_canary_future_game_does_not_change_earlier_ratings(cfg, dataset):
    clean = Engine(dataset, cfg, last_completed=2025)
    dirty = Engine(_with_canary(dataset, clean), cfg, last_completed=2025)

    # The slot's as-of time is unchanged by the canary (it kicks off later).
    assert_frame_equal(clean.season_slots(CANARY_SEASON), dirty.season_slots(CANARY_SEASON))
    assert_frame_equal(clean.priors(CANARY_SEASON), dirty.priors(CANARY_SEASON))

    order = clean.season_slots(CANARY_SEASON).select("slot", "slot_order")
    cutoff = order.filter(pl.col("slot") == CANARY_SLOT)["slot_order"].item()
    before = order.filter(pl.col("slot_order") <= cutoff)["slot"].to_list()
    after = order.filter(pl.col("slot_order") > cutoff)["slot"].to_list()

    for slot in before:
        as_of = clean.slot_as_of(CANARY_SEASON, slot)
        assert_frame_equal(clean.snapshot(CANARY_SEASON, as_of), dirty.snapshot(CANARY_SEASON, as_of))

    # Sanity: the canary is strong enough to move later ratings.
    as_of = clean.slot_as_of(CANARY_SEASON, after[0])
    c = clean.snapshot(CANARY_SEASON, as_of).filter(pl.col("team") == "Vanderbilt")["o_ppa"].item()
    d = dirty.snapshot(CANARY_SEASON, as_of).filter(pl.col("team") == "Vanderbilt")["o_ppa"].item()
    assert d > c + 0.05

    # Massey baseline: same guarantee.
    mc = clean.massey_predictions(CANARY_SEASON)
    md = dirty.massey_predictions(CANARY_SEASON).filter(pl.col("game_id") != 999_999_999)
    early = clean.games.filter((pl.col("season") == CANARY_SEASON) & pl.col("slot").is_in(before))["game_id"].to_list()
    assert_frame_equal(mc.filter(pl.col("game_id").is_in(early)).sort("game_id"),
                       md.filter(pl.col("game_id").is_in(early)).sort("game_id"))
    late = clean.games.filter((pl.col("season") == CANARY_SEASON) & pl.col("slot").is_in(after))["game_id"].to_list()
    assert not mc.filter(pl.col("game_id").is_in(late)).sort("game_id").equals(
        md.filter(pl.col("game_id").is_in(late)).sort("game_id"))


@pytest.mark.parametrize("fcs_slope", [False, True])
@pytest.mark.parametrize("target", [2018, 2023, 2026])
def test_calibration_v2_trains_on_the_same_seasons_as_the_base_calibration(cfg, target, fcs_slope, monkeypatch):
    """The second stage fits on training_seasons(target) only, with each game's own-season prediction."""
    import numpy as np

    eng = Engine.__new__(Engine)
    eng.cfg, eng.first_eval, eng.last_completed, eng._stage2 = cfg, cfg.ingest.first_season + 1, 2025, {}
    seen = []

    def fake(season):
        seen.append(season)
        g = pl.DataFrame({"completed": [True] * 3, "margin": [3.0, -7.0, 10.0], "neutral_site": [False] * 3,
                          "prior_share_h": [0.5] * 3, "prior_share_a": [0.4] * 3, "home_fcs": [False] * 3,
                          "away_fcs": [False, False, True]})
        return g, np.array([1.0, -2.0, 6.0])

    monkeypatch.setattr(eng, "base_predictions", fake)
    eng.stage2(target, fcs_slope=fcs_slope)
    assert seen == training_seasons(target, cfg, eng.prior_available())
    assert target not in seen


# -- day mode (DECISIONS "M1 intra-week as-of") -------------------------------------

def test_day_as_of_cutoffs_by_hand():
    from datetime import datetime, timezone

    from cfbr.data import day_as_of, slot_as_of

    def et(s):  # Eastern daylight time in October
        return datetime.fromisoformat(s + "-04:00").astimezone(timezone.utc)

    kick = {1: et("2026-10-15T19:30"),   # Thu night: slot as-of
            2: et("2026-10-16T22:30"),   # Fri late
            3: et("2026-10-17T12:00"),   # Sat noon: Saturday group starts here
            4: et("2026-10-17T22:30"),   # Sat late: same group as game 3
            5: et("2026-10-18T00:30")}   # Hawaii, Sat 6:30 PM HST = Sunday Eastern
    g = pl.DataFrame({"game_id": list(kick), "season": [2026] * 5, "slot": ["r08"] * 5, "slot_order": [8] * 5,
                      "kickoff_ts": list(kick.values())})
    d = {r["game_id"]: r["cutoff"] for r in day_as_of(g, slot_as_of(g)).iter_rows(named=True)}
    assert d[1] == kick[1]                                   # first day: the slot as-of
    assert d[2] == kick[2] - timedelta(hours=5)              # Fri 5:30 PM ET: Thursday's game is in
    assert d[3] == d[4] == kick[3] - timedelta(hours=5)      # 7 AM ET Saturday: Fri late game is in (8.5 h before)
    assert kick[2] < d[3] and not (kick[4] < d[4])           # the Sat late game never feeds its own group
    assert d[5] == kick[5] - timedelta(hours=5)              # 7:30 PM ET Sat: excludes the Sat 22:30 game...
    assert not (kick[4] < d[5]) and kick[3] < d[5]           # ...still in progress then; Sat noon is over


def test_day_mode_never_trains_on_a_game_in_progress(cfg, dataset):
    from cfbr.data import GAME_DURATION_BOUND

    eng = Engine(dataset, cfg, last_completed=2025, as_of_mode="day")
    season = 2024
    grp = eng.day_groups.filter(pl.col("season") == season).join(
        eng.season_slots(season).select("slot", "as_of"), on="slot")
    games = eng.games.filter(pl.col("season") == season).select("kickoff_ts")["kickoff_ts"]
    for r in grp.unique(["slot", "day"]).iter_rows(named=True):
        used = games.filter(games < r["cutoff"])
        # every game before the cutoff kicked off >= 5 h before the group's first game, or before the slot as-of
        ok = (used + GAME_DURATION_BOUND <= r["group_first"]) | (used < r["as_of"])
        assert ok.all(), (r["slot"], r["day"])
    p = eng.predict(season)
    assert (p["as_of"] <= p["kickoff_ts"]).all()
    # slot mode is untouched: same as_of as before for every game
    slot = Engine(dataset, cfg, last_completed=2025).group_cutoffs(season)
    assert slot.join(eng.season_slots(season).select("as_of"), on="as_of", how="anti").height == 0
