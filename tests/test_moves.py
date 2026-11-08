"""What-moved breakdown (spec 2026-11-08-what-moved-design.md). Real-data tests skip without the DB."""

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cfbr.model.engine import RATING_COLS, Engine, _ns
from cfbr.model.ridge import fit_efficiency_arrays, fit_fcs_stage2, play_weights, recency_weights


@pytest.fixture(params=["default", "two_stage"])
def model_cfg(request, cfg, cfg_two_stage):
    return cfg if request.param == "default" else cfg_two_stage


def test_unmasked_fit_is_unchanged_and_carries_home_coefficients(model_cfg, dataset):
    eng = Engine(dataset, model_cfg, last_completed=2025)
    as_of = eng.slot_as_of(2018, "r07")
    k = eng.arrays(2018).n_before(as_of)
    plain = eng._fit(2018, eng.priors(2018), as_of)
    full_mask = eng._fit(2018, eng.priors(2018), as_of, mask=np.ones(k, bool))
    assert_frame_equal(plain, full_mask)
    assert {"home_ppa", "home_sr"} <= set(plain.columns)
    assert plain["home_ppa"].n_unique() == 1 and plain["home_sr"].n_unique() == 1


def test_masked_fit_equals_explicit_fit_on_remaining_plays(model_cfg, dataset):
    eng = Engine(dataset, model_cfg, last_completed=2025)
    m, a = model_cfg.model, eng.arrays(2018)
    as_of = eng.slot_as_of(2018, "r07")
    k = a.n_before(as_of)
    keep = np.ones(k, bool)
    keep[k // 2:k // 2 + 500] = False
    got = eng._fit(2018, eng.priors(2018), as_of, mask=keep)
    pr = pl.DataFrame({"team": a.teams}).join(eng.priors(2018), on="team", how="left").fill_null(0.0)
    p = {c: pr[c].to_numpy() for c in RATING_COLS}
    w = play_weights(a, k, _ns(as_of), m)
    w = None if w is None else w[keep]
    off, de, h = a.off[:k][keep], a.de[:k][keep], a.home[:k][keep]
    ref = fit_efficiency_arrays(off, de, h, a.ppa[:k][keep], a.teams, m.alpha_ppa, p["o_ppa"], p["d_ppa"], w, a.center)
    if m.fcs_two_stage:
        rec = recency_weights(a.kickoff_ns[:k], _ns(as_of), m.half_life_weeks)
        ref = fit_fcs_stage2(off, de, h, a.ppa[:k][keep], ~a.center, m.alpha_ppa, ref, p["o_ppa"], p["d_ppa"],
                             None if rec is None else rec[keep])
    assert np.abs(got["o_ppa"].to_numpy() - ref.o).max() < 1e-9
    assert np.abs(got["d_ppa"].to_numpy() - ref.d).max() < 1e-9
    assert got["home_ppa"][0] == pytest.approx(ref.home, abs=1e-12)


from cfbr.report.moves import game_efficiency, moves_csv_frame, rating_moves  # noqa: E402
from cfbr.report.rank import fbs_table, week_as_of  # noqa: E402

SEASON = 2023


@pytest.fixture(scope="module")
def eng(cfg_two_stage, dataset):
    return Engine(dataset, cfg_two_stage, last_completed=2025)


@pytest.fixture(scope="module")
def week7(eng):
    return rating_moves(eng, SEASON, week_as_of(eng, SEASON, 6), week_as_of(eng, SEASON, 7))


def test_parts_sum_to_the_published_change(eng, week7):
    cur = fbs_table(eng, SEASON, week_as_of(eng, SEASON, 7)).select("team", "rating", "rank")
    prev = fbs_table(eng, SEASON, week_as_of(eng, SEASON, 6)).select("team", prev="rating")
    ref = cur.join(prev, on="team").with_columns(change=pl.col("rating") - pl.col("prev"))
    m = week7.join(ref, on="team", suffix="_ref")
    assert m.height == week7.height == ref.height
    total = m["move_decay"] + m["move_own"] + m["move_others"]
    assert (total - m["change"]).abs().max() < 1e-9
    assert (m["rank"] == m["rank_ref"]).all()


def test_bye_week_has_no_own_game(eng, week7):
    g = eng.games.filter((pl.col("season") == SEASON) & (pl.col("week") == 7))
    played = set(g["home"].to_list()) | set(g["away"].to_list())
    byes = week7.filter(~pl.col("team").is_in(list(played)))
    assert byes.height > 0 and (byes["move_own"] == 0).all()


def test_week_one_has_nothing_to_fade(eng):
    w1 = rating_moves(eng, SEASON, eng.slot_as_of(SEASON, "r01"), week_as_of(eng, SEASON, 1))
    assert w1["move_decay"].abs().max() < 1e-12


def test_passed_lists_match_the_ranks(week7):
    ranks = {r["team"]: (r["prev_rank"], r["rank"]) for r in week7.iter_rows(named=True)}
    for r in week7.iter_rows(named=True):
        pr, cr = r["prev_rank"], r["rank"]
        by = [t for t, (p, c) in ranks.items() if p > pr and c < cr]
        gone = [t for t, (p, c) in ranks.items() if p < pr and c > cr]
        assert r["n_passed_by"] == len(by) and r["n_passed"] == len(gone)
        assert set(r["passed_by"]) <= set(by) and len(r["passed_by"]) == min(5, len(by))
        assert set(r["passed"]) <= set(gone) and len(r["passed"]) == min(5, len(gone))


def test_game_efficiency_actuals_and_point_in_time_expectation(eng):
    ge = game_efficiency(eng, SEASON)
    g = eng.games.filter((pl.col("season") == SEASON) & pl.col("completed") & (pl.col("week") == 7))
    row = g.row(0, named=True)
    e = ge.filter(pl.col("game_id") == row["game_id"]).row(0, named=True)
    plays = eng.season_plays(SEASON).filter(pl.col("game_id") == row["game_id"])
    hp = plays.filter(pl.col("offense") == row["home"])
    assert e["home_ppa"] == pytest.approx(hp["ppa"].mean())
    assert e["home_sr"] == pytest.approx(hp["success"].cast(pl.Float64).mean())
    as_of = eng.slot_as_of(SEASON, row["slot"])
    assert as_of <= row["kickoff_ts"]
    snap = eng.snapshot(SEASON, as_of)
    r = {t: x for t, x in zip(snap["team"], snap.iter_rows(named=True))}
    hm = 0.0 if row["neutral_site"] else 1.0
    exp = (r[row["home"]]["o_ppa"] + r[row["home"]]["d_ppa"] - r[row["away"]]["o_ppa"] - r[row["away"]]["d_ppa"]
           + 2 * snap["home_ppa"][0] * hm)
    assert e["exp_ppa_edge"] == pytest.approx(exp, abs=1e-12)


def test_csv_frame_rounds_and_joins(week7):
    c = moves_csv_frame(week7)
    assert c.schema["passed_by"] == pl.Utf8 and c["move_own"].round(2).equals(c["move_own"])


def test_rank_csv_matches_web_rows(eng, tmp_path):
    from datetime import UTC, datetime

    from cfbr.report.rank import run_rank
    from cfbr.web.build import season_ratings

    cfg = eng.cfg.model_copy(deep=True)
    cfg.paths.out_dir = tmp_path
    out = run_rank(cfg, SEASON, 7, last_completed=2025, ds=eng.ds)
    csv = pl.read_csv(out / "moves.csv")
    web = (season_ratings(eng, SEASON, datetime.now(UTC), moves=True)
           .filter((pl.col("week") == 7) & pl.col("fbs"))
           .select("team", "move_decay", "move_own", "move_others", "passed_by", "passed"))
    web = moves_csv_frame(web)
    m = csv.join(web, on="team", suffix="_web")
    assert m.height == csv.height
    for c in ("move_decay", "move_own", "move_others"):
        assert (m[c] - m[f"{c}_web"]).abs().max() < 0.006
    assert (m["passed_by"].fill_null("") == m["passed_by_web"].fill_null("")).all()
