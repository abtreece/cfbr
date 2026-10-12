from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from cfbr.model.home_field import MIN_GAMES, attach, fit_c, reliability, team_games, team_gaps

T0 = datetime(2019, 9, 1)


def _preds(n: int, home_resid: float, away_resid: float, season: int = 2019) -> pl.DataFrame:
    """Team T alternates home vs H{i} and away at A{i}; T's residual is fixed by venue."""
    rows = []
    for i in range(n):
        at_home = i % 2 == 0
        r = home_resid if at_home else away_resid
        rows.append({"game_id": i, "season": season, "as_of": T0 + timedelta(days=7 * i),
                     "kickoff_ts": T0 + timedelta(days=7 * i), "completed": True, "neutral_site": False,
                     "home": "T" if at_home else f"A{i}", "away": f"H{i}" if at_home else "T",
                     "pred_margin": 0.0, "margin": r if at_home else -r})
    return pl.DataFrame(rows)


def test_gap_is_home_minus_away_from_the_team_side():
    tg = team_games(_preds(2 * MIN_GAMES, home_resid=5.0, away_resid=-1.0))
    gaps = team_gaps(tg, 2019, T0 + timedelta(days=10_000))
    assert gaps.filter(pl.col("team") == "T")["delta"].item() == pytest.approx(6.0)


def test_minimum_games_and_fcs_pools():
    p = _preds(2 * MIN_GAMES - 1, 5.0, -1.0)        # one away game short
    assert team_gaps(team_games(p), 2019, T0 + timedelta(days=10_000)).filter(pl.col("team") == "T").height == 0
    p = _preds(2 * MIN_GAMES, 5.0, -1.0).with_columns(home=pl.col("home").replace("T", "FCS"),
                                                       away=pl.col("away").replace("T", "FCS"))
    assert team_games(p).filter(pl.col("team").str.starts_with("FCS")).height == 0


def test_point_in_time_cutoff():
    """A game sees only residuals from games kicked off before its slot's as_of."""
    n = 2 * MIN_GAMES
    p = _preds(n + 1, 5.0, -1.0)
    # the last game (a home game for T) carries a huge residual; it must not inform its own feature
    p = p.with_columns(margin=pl.when(pl.col("game_id") == n).then(1000.0).otherwise(pl.col("margin")))
    out = attach(p, team_games(p))
    last = out.filter(pl.col("game_id") == n)
    assert last["delta_h"].item() == pytest.approx(6.0)
    assert last["x"].item() == pytest.approx(6.0 * 0.5)
    # before T has the minimum sample, the feature is 0
    assert out.filter(pl.col("game_id") < n - 1)["x"].abs().max() == 0.0


def test_lookback_window_and_neutral_sites():
    old = _preds(2 * MIN_GAMES, 9.0, -9.0, season=2015)      # outside S-4 for 2020
    new = _preds(2 * MIN_GAMES, 2.0, 0.0, season=2020).with_columns(
        kickoff_ts=pl.col("kickoff_ts") + timedelta(days=365 * 5), as_of=pl.col("as_of") + timedelta(days=365 * 5),
        game_id=pl.col("game_id") + 1000)
    tg = team_games(pl.concat([old, new]))
    assert team_gaps(tg, 2020, T0 + timedelta(days=10_000)).filter(pl.col("team") == "T")["delta"].item() == \
        pytest.approx(2.0)
    neutral = _preds(2 * MIN_GAMES, 5.0, -1.0).with_columns(neutral_site=pl.lit(True))
    assert team_games(neutral).height == 0


def test_neutral_site_feature_sign():
    """A strong-home team at a neutral site is expected to do slightly worse than its rating."""
    p = _preds(2 * MIN_GAMES + 1, 4.0, 0.0)
    p = p.with_columns(neutral_site=pl.col("game_id") == 2 * MIN_GAMES)
    last = attach(p, team_games(p)).filter(pl.col("game_id") == 2 * MIN_GAMES)
    assert last["x"].item() == pytest.approx(-0.5 * 4.0)


def test_fit_c_recovers_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    rows = pl.DataFrame({"x": x, "pred_margin": np.zeros(500), "margin": 0.4 * x + rng.normal(0, 0.01, 500)})
    assert fit_c(rows) == pytest.approx(0.4, abs=0.01)


def test_reliability_perfect_when_gaps_repeat():
    blocks = []
    for season in (2015, 2019):
        for k, gap in enumerate((1.0, 3.0, 6.0)):
            blocks.append(_preds(2 * MIN_GAMES, gap, 0.0, season=season).with_columns(
                home=pl.col("home").replace("T", f"T{k}"), away=pl.col("away").replace("T", f"T{k}"),
                game_id=pl.col("game_id") + 100 * k + 10_000 * (season - 2015)))
    rel = reliability(team_games(pl.concat(blocks)), (2015, 2017), (2018, 2021))
    assert rel["n_teams"] == 3
    assert rel["r"] == pytest.approx(1.0)
