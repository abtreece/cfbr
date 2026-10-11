from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from cfbr.model.availability import apply_adjustment, attach, availability_features, fit_adjustment


def _season():
    """Team T: QB1 starts games 0-1, QB2 starts game 2; RB1 plays games 0-1 then misses game 2."""
    rows = []

    def add(g, pid, att=0, rush=0, rec=0, tk=0):
        rows.append({"season": 2026, "season_type": "regular", "week": g + 1, "game_id": 100 + g, "team": "T",
                     "player_id": pid, "name": pid, "pass_cmp": 0.0, "pass_att": float(att), "rush_att": float(rush),
                     "rec": float(rec), "tackles": float(tk), "tfl": 0.0, "pd": 0.0, "ints": 0.0})
    for g in (0, 1):
        add(g, "QB1", att=30)
        add(g, "RB1", rush=20)
        add(g, "WR1", rec=5)
        add(g, "LB1", tk=10)
        add(g, "LB2", tk=5)
    add(2, "QB2", att=25)
    add(2, "QB1", att=3)
    add(2, "WR1", rec=6)
    add(2, "LB2", tk=7)
    add(3, "QB2", att=20)
    gp = pl.DataFrame(rows)
    t0 = datetime(2026, 9, 5)
    games = pl.DataFrame({"game_id": [100, 101, 102, 103], "kickoff_ts": [t0 + timedelta(days=7 * i) for i in range(4)]})
    return gp, games


def test_features_use_only_earlier_games():
    gp, games = _season()
    f = {r["game_id"]: r for r in availability_features(gp, games).iter_rows(named=True)}
    assert f[100]["has_info"] == 0.0 and f[100]["qb_out"] == 0.0          # first game: unknown
    assert f[102]["qb_out"] == 0.0 and f[102]["skill_missing"] == 0.0      # everyone played game 101
    # Before game 103: primary QB1 (63 att) didn't lead game 102 (QB2 did) -> backup started last time out.
    assert f[103]["qb_out"] == 1.0
    # RB1 (2 games, 40 touches) missed game 102: 40 / (40 + 16 WR1) of touches so far.
    assert f[103]["skill_missing"] == pytest.approx(40 / 56)
    # LB1 (2 games, 20 tackles) missed game 102: 20 / (20 + 17).
    assert f[103]["def_missing"] == pytest.approx(20 / 37)


def test_future_games_do_not_change_features():
    gp, games = _season()
    base = availability_features(gp, games).filter(pl.col("game_id") <= 102).sort("game_id")
    extra = pl.concat([gp, gp.filter(pl.col("game_id") == 103).with_columns(player_id=pl.lit("NEW"), game_id=pl.lit(103, pl.Int64))])
    again = availability_features(extra, games).filter(pl.col("game_id") <= 102).sort("game_id")
    assert base.equals(again)


def test_adjustment_recovers_known_effect():
    rng = np.random.default_rng(1)
    n = 4000
    d_qb = rng.choice([-1.0, 0.0, 0.0, 0.0, 1.0], n)
    pred = rng.normal(0, 10, n)
    margin = pred - 5.0 * d_qb + rng.normal(0, 14, n)
    rows = pl.DataFrame({"pred_margin": pred, "margin": margin, "d_qb_out": d_qb,
                         "d_skill_missing": np.zeros(n), "d_def_missing": np.zeros(n)})
    coef = fit_adjustment(rows)
    assert coef[0] == pytest.approx(-5.0, abs=1.0)
    assert apply_adjustment(rows, coef).len() == n


def test_attach_differences():
    preds = pl.DataFrame({"game_id": [1], "home_team": ["H"], "away_team": ["A"], "pred_margin": [3.0], "margin": [7.0]})
    feats = pl.DataFrame({"game_id": [1, 1], "team": ["H", "A"], "qb_out": [1.0, 0.0],
                          "skill_missing": [0.2, 0.1], "def_missing": [0.0, 0.3]})
    r = attach(preds, feats).row(0, named=True)
    assert r["d_qb_out"] == 1.0 and r["d_skill_missing"] == pytest.approx(0.1) and r["d_def_missing"] == pytest.approx(-0.3)
