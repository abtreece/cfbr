"""FCS two-stage fit (DECISIONS "FCS games, two-stage fit"): synthetic checks, no DB."""

import numpy as np
import pytest

from cfbr.config import Model, load_config
from cfbr.model.engine import SeasonArrays
from cfbr.model.ridge import fit_efficiency_arrays, play_weights, recency_weights

WEEK_NS = 7 * 86400 * 10**9
N_FBS = 20
THIN = 22  # index of the thin FCS group F2


def _season(thin_plays=40, seed=11):
    """20 FBS teams (true O/D ~ N(0, 0.1)); FCS groups F0, F1 (8 games each) and F2 (one short game),
    all with true O = D = -0.4. Plays sorted by kickoff like Engine.arrays."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(N_FBS)] + ["F0", "F1", "F2"]
    center = np.array([t.startswith("T") for t in teams])
    o_true = np.r_[rng.normal(0, 0.1, N_FBS), [-0.4] * 3]
    d_true = np.r_[rng.normal(0, 0.1, N_FBS), [-0.4] * 3]
    rows = []  # (offense, defense, home indicator, week, plays)
    for week in range(10):
        for a, b in rng.permutation(N_FBS).reshape(-1, 2):
            rows += [(a, b, 1.0, week, 60), (b, a, -1.0, week, 60)]
    for g, n_games, per_side in ((20, 8, 60), (21, 8, 60), (THIN, 1, thin_plays // 2)):
        for week in range(n_games):
            fbs = int(rng.integers(N_FBS))
            rows += [(fbs, g, 1.0, week, per_side), (g, fbs, -1.0, week, per_side)]
    off, de, h, wk = [], [], [], []
    for o, d, hh, w, c in rows:
        off += [o] * c
        de += [d] * c
        h += [hh] * c
        wk += [w] * c
    off, de, h, wk = np.array(off), np.array(de), np.array(h, dtype=float), np.array(wk)
    ppa = 0.05 + o_true[off] - d_true[de] + 0.02 * h + rng.normal(0, 1.2, off.size)
    order = np.argsort(wk, kind="stable")
    return SeasonArrays(teams=teams, center=center, kickoff_ns=(wk[order] * WEEK_NS).astype(np.int64),
                        off=off[order], de=de[order], home=h[order], ppa=ppa[order],
                        success=(ppa[order] > 0).astype(float))


def _stage1(a, w, alpha=300.0, keep=None):
    sel = np.ones(a.off.size, dtype=bool) if keep is None else keep
    return fit_efficiency_arrays(a.off[sel], a.de[sel], a.home[sel], a.ppa[sel], a.teams, alpha, None, None,
                                 None if w is None else w[sel], a.center)


def test_config_defaults_leave_behaviour_unchanged():
    for path in (None, "config/tuned.toml"):
        m = load_config(path).model
        assert m.fcs_two_stage is False and m.fcs_weight == 1.0


def test_fcs_weight_must_be_a_fraction():
    with pytest.raises(ValueError):
        Model(fcs_weight=1.5)


@pytest.mark.parametrize("two_stage,weight", [(False, 0.2), (True, 1.0)])
def test_play_weights_reduce_to_recency_exactly(two_stage, weight):
    a = _season()
    k = a.off.size
    as_of = int(a.kickoff_ns.max()) + WEEK_NS
    m = Model(half_life_weeks=3.7, fcs_two_stage=two_stage, fcs_weight=weight)
    np.testing.assert_array_equal(play_weights(a, k, as_of, m), recency_weights(a.kickoff_ns[:k], as_of, 3.7))
    assert play_weights(a, k, None, m) is None


def test_play_weights_scale_fcs_plays_only():
    a = _season()
    k = a.off.size
    w = play_weights(a, k, None, Model(fcs_two_stage=True, fcs_weight=0.25))
    fcs = ~a.center[a.off] | ~a.center[a.de]
    assert set(np.unique(w[fcs])) == {0.25}
    assert set(np.unique(w[~fcs])) == {1.0}


def test_play_weights_respect_the_prefix():
    a = _season()
    k = a.off.size // 2
    w = play_weights(a, k, int(a.kickoff_ns[k]), Model(half_life_weeks=3.7, fcs_two_stage=True, fcs_weight=0.3))
    assert w.size == k


def test_fcs_weight_zero_isolates_fbs_ratings():
    a = _season()
    k = a.off.size
    drop_thin_game = ~((a.off == THIN) | (a.de == THIN))
    w0 = play_weights(a, k, None, Model(fcs_two_stage=True, fcs_weight=0.0))
    full, part = _stage1(a, w0), _stage1(a, w0, keep=drop_thin_game)
    np.testing.assert_allclose(full.o[a.center], part.o[a.center], atol=1e-12)
    np.testing.assert_allclose(full.d[a.center], part.d[a.center], atol=1e-12)
    # Control: at full weight the same game does move FBS ratings.
    w1 = play_weights(a, k, None, Model(fcs_two_stage=True, fcs_weight=1.0))
    moved = np.abs(_stage1(a, w1).o[a.center] - _stage1(a, w1, keep=drop_thin_game).o[a.center]).max()
    assert moved > 1e-4
