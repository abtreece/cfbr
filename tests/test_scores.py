"""Midweek final scores on the site (cfbr scores) and pre-kickoff closes for those games."""

import fcntl
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from cfbr.web.build import web_dir
from cfbr.web.scores import GAME_LENGTH, apply_scores, due_games, refresh_scores, scores_path
from tests.test_web import SEASON, _odds_event, _write_season, web_cfg  # noqa: F401 (fixture)

NOW = datetime(2026, 10, 14, 4, tzinfo=UTC)  # Tuesday 11 PM Central


def _cfbd(gid, hp, ap, completed=True):
    return {"id": gid, "homePoints": hp, "awayPoints": ap, "completed": completed}


def _site_games():
    return pl.DataFrame({
        "game_id": [1, 2, 3, 4],
        "completed": [True, False, False, False],
        "kickoff_ts": [NOW - timedelta(days=4), NOW - timedelta(hours=5), NOW - timedelta(hours=4),
                       NOW + timedelta(days=3)],
        "home_points": [31, None, None, None], "away_points": [10, None, None, None],
        "margin": [21.0, None, None, None],
    }, schema_overrides={"home_points": pl.Int32, "away_points": pl.Int32})


def test_apply_scores_fills_in_only_games_not_yet_final():
    games = _site_games()
    rows = [_cfbd(1, 0, 0), _cfbd(2, 24, 27), _cfbd(3, 14, 7, completed=False), _cfbd(4, None, None)]
    out = apply_scores(games, rows).sort("game_id")
    # 1: the store's score stands. 2: filled in. 3: CFBD not final yet. 4: unplayed.
    assert out["completed"].to_list() == [True, True, False, False]
    assert out["home_points"].to_list() == [31, 24, None, None]
    assert out["margin"].to_list() == [21.0, -3.0, None, None]
    assert out.schema == games.schema
    assert apply_scores(games, None).equals(games) and apply_scores(games, []).equals(games)


def test_due_games_are_over_but_not_stale():
    games = _site_games().with_columns(
        kickoff_ts=pl.Series([NOW - timedelta(days=5), NOW - GAME_LENGTH, NOW - GAME_LENGTH + timedelta(minutes=1),
                              NOW - timedelta(days=5)]),
        completed=pl.Series([False, False, False, False]))
    # 1 and 4: past STALE_GAME (postponed on the site). 3: may still be in progress.
    assert due_games(games, NOW)["game_id"].to_list() == [2]
    # Build data carries local kickoff times; same answer.
    local = games.with_columns(pl.col("kickoff_ts").dt.convert_time_zone("America/Chicago"))
    assert due_games(local, NOW)["game_id"].to_list() == [2]


def _site(web_cfg, now):  # noqa: F811
    """Synthetic site with game 3 (Texas A&M @ Ohio State) kicked off 5 h before ``now``."""
    d = _write_season(web_dir(web_cfg), now)
    games = pl.read_parquet(d / "games.parquet")
    games = games.with_columns(
        kickoff_ts=pl.when(pl.col("game_id") == 3).then(pl.lit(now - timedelta(hours=5)))
        .otherwise(pl.col("kickoff_ts")).cast(games.schema["kickoff_ts"]),
        cfbd_spread=pl.col("spread"))
    games.write_parquet(d / "games.parquet")
    web_cfg.paths.raw_dir = web_cfg.paths.out_dir.parent / "raw"
    web_cfg.paths.db_path = web_cfg.paths.out_dir.parent / "db" / "cfb.duckdb"
    web_cfg.paths.db_path.parent.mkdir(parents=True)
    return d


def test_refresh_scores_calls_only_when_a_game_is_due(web_cfg):  # noqa: F811
    now = datetime.now(UTC)
    d = _site(web_cfg, now)
    calls = []

    def fetch():
        calls.append(1)
        rows = [_cfbd(3, 20, 23)]
        p = scores_path(web_cfg.paths.raw_dir, SEASON)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows))
        return rows

    # Two hours after kickoff: nothing due yet, no call.
    assert refresh_scores(web_cfg, SEASON, fetch, now=now - timedelta(hours=3)) is None and not calls
    before = pl.read_parquet(d / "games.parquet")
    assert refresh_scores(web_cfg, SEASON, fetch, now=now) == (1, 1) and len(calls) == 1
    after = pl.read_parquet(d / "games.parquet").sort("game_id")
    g3 = after.filter(pl.col("game_id") == 3).to_dicts()[0]
    assert g3["completed"] and (g3["home_points"], g3["away_points"], g3["margin"]) == (20, 23, -3.0)
    # Nothing else moves: predictions, lines, the other games.
    assert after.drop("completed", "home_points", "away_points", "margin").equals(
        before.sort("game_id").drop("completed", "home_points", "away_points", "margin"))
    assert json.loads((d / "meta.json").read_text())["scores_as_of"]
    # Game 3 is final now: the next run has nothing due and makes no call.
    assert refresh_scores(web_cfg, SEASON, fetch, now=now) is None and len(calls) == 1


def test_refresh_scores_keeps_trying_while_cfbd_has_no_final(web_cfg):  # noqa: F811
    now = datetime.now(UTC)
    d = _site(web_cfg, now)
    before = (d / "meta.json").read_bytes()
    assert refresh_scores(web_cfg, SEASON, lambda: [_cfbd(3, 14, 7, completed=False)], now=now) == (1, 0)
    assert (d / "meta.json").read_bytes() == before  # nothing filled: the site isn't rewritten
    assert refresh_scores(web_cfg, SEASON, lambda: [_cfbd(3, 21, 7)], now=now) == (1, 1)


@pytest.mark.parametrize("why", ["lock", "finished season", "no web data"])
def test_refresh_scores_makes_no_call(web_cfg, why):  # noqa: F811
    now = datetime.now(UTC)
    d = _site(web_cfg, now)

    def fetch():
        raise AssertionError("no CFBD call expected")

    if why == "finished season":
        (d / "meta.json").write_text(json.dumps(json.loads((d / "meta.json").read_text()) | {"live": False}))
    if why == "no web data":
        (d / "games.parquet").unlink()
    if why == "lock":
        with open(web_cfg.paths.db_path.parent / ".weekly.lock", "w") as held:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert refresh_scores(web_cfg, SEASON, fetch, now=now) is None
    else:
        assert refresh_scores(web_cfg, SEASON, fetch, now=now) is None


def test_store_and_odds_snapshot_ignore_the_scores_file(tmp_path):
    from cfbr.bets.closing import cached_games
    from cfbr.ingest.store import build_games

    d = tmp_path / "games" / str(SEASON)
    d.mkdir(parents=True)
    game = {"id": 1, "season": SEASON, "week": 7, "seasonType": "regular", "startDate": "2026-10-13T23:00:00.000Z",
            "completed": False, "homeTeam": "Middle Tennessee", "awayTeam": "Delaware"}
    (d / "both_w06.json").write_text(json.dumps([game]))
    (d / "scores.json").write_text(json.dumps([game | {"completed": True, "homePoints": 20, "awayPoints": 23}]))
    assert build_games(tmp_path)["completed"].to_list() == [False]
    assert [g["completed"] for g in cached_games(tmp_path, SEASON)] == [False]


def test_a_game_past_kickoff_shows_its_pre_kickoff_close_until_cfbd_has_one(tmp_path):
    from cfbr.web.lines import apply_lines, kickoff_closes

    cfbd_as_of = "2026-10-11T20:00:00+00:00"
    kick = datetime(2026, 10, 13, 23, tzinfo=UTC)
    games = pl.DataFrame({
        "game_id": [1, 2, 3], "home_team": ["Middle Tennessee", "Ohio State", "Hawai'i"],
        "away_team": ["Delaware", "Texas A&M", "Texas A&M"], "completed": [True, False, True],
        # 1: tonight's game, final from `cfbr scores`. 2: unplayed. 3: played before CFBD's fetch.
        "kickoff_ts": [kick, kick + timedelta(days=4), datetime(2026, 10, 10, 23, tzinfo=UTC)],
        "cfbd_spread": [2.5, -3.0, -7.0],
    })
    snaps = tmp_path / "odds" / str(SEASON) / "snapshots"
    snaps.mkdir(parents=True)

    def snap(ts, spread_mtsu):
        events = [_odds_event("e1", "Middle Tennessee", "Delaware", kick, [("DraftKings", spread_mtsu), ("FanDuel", 1.0)]),
                  _odds_event("e2", "Ohio State", "Texas A&M", kick + timedelta(days=4), [("DraftKings", -4.0)])]
        (snaps / f"{ts:%Y%m%dT%H%M%SZ}.json").write_text(json.dumps(events))

    snap(datetime(2026, 10, 10, 12, tzinfo=UTC), 9.0)  # before CFBD's lines: never used
    snap(datetime(2026, 10, 13, 14, tzinfo=UTC), 2.0)
    snap(datetime(2026, 10, 13, 21, 40, tzinfo=UTC), 1.0)  # the close
    snap(datetime(2026, 10, 14, 1, tzinfo=UTC), -20.0)  # after kickoff: live lines, never used
    now = datetime(2026, 10, 14, 4, tzinfo=UTC)
    closes = kickoff_closes(tmp_path, SEASON, games, cfbd_as_of, now)
    assert closes.to_dicts() == [{"game_id": 1, "close": 1.0, "close_as_of": "2026-10-13T21:40:00+00:00"}]
    fresh = (datetime(2026, 10, 14, 1, tzinfo=UTC), pl.DataFrame({"game_id": [1, 2], "spread": [-20.0, -4.0]}))
    out = apply_lines(games, fresh, cfbd_as_of, closes).sort("game_id")
    assert [(r["spread"], r["line_source"]) for r in out.to_dicts()] == [(1.0, "odds"), (-4.0, "odds"), (-7.0, "cfbd")]
    assert out["line_as_of"][0] == "2026-10-13T21:40:00+00:00"
    # After Sunday's run CFBD's line is newer than the kickoff: no close from the Odds API.
    assert kickoff_closes(tmp_path, SEASON, games, "2026-10-18T14:00:00+00:00", now).height == 0
    assert kickoff_closes(tmp_path, SEASON, games, None, now).height == 0
