from datetime import UTC, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from cfbr.bets.retro import opening_books, retro_ledger, retro_markdown
from cfbr.config import Bets, Config

T1 = datetime(2026, 9, 5, 16, tzinfo=UTC)
T2 = datetime(2026, 9, 12, 16, tzinfo=UTC)


class FakeEngine:
    """Two weekly slots of point-in-time predictions; game 3 is still unplayed."""

    def __init__(self):
        self.cfg = Config()
        games = pl.DataFrame({
            "game_id": [1, 2, 3], "season": [2026] * 3, "week": [1, 2, 2], "season_type": ["regular"] * 3,
            "completed": [True, True, False], "home_points": [30, 10, None], "away_points": [20, 24, None],
        })
        lines = pl.DataFrame({
            "game_id": [1, 1, 2, 3], "season": [2026] * 4, "provider": ["DraftKings", "Bovada", "DraftKings", "DraftKings"],
            "spread_open": [-3.0, -4.0, 2.0, -7.0], "spread": [-5.0, -5.0, 3.0, -7.0],
        })
        self.ds = SimpleNamespace(games=games, lines=lines)
        self._preds = pl.DataFrame({
            "game_id": [1, 2, 3], "season": [2026] * 3, "week": [1, 2, 2], "season_type": ["regular"] * 3,
            "slot": ["r01", "r02", "r02"], "as_of": [T1, T2, T2], "kickoff_ts": [T1, T2, T2],
            "completed": [True, True, False], "home_team": ["H1", "H2", "H3"], "away_team": ["A1", "A2", "A3"],
            "pred_margin": [10.0, -8.0, 20.0],
        })

    def predict(self, season):
        return self._preds


def test_retro_bets_use_opening_lines_and_point_in_time_stamps():
    led = retro_ledger(FakeEngine(), 2026, 6, Bets(), "v1")
    assert led["game_id"].to_list() == [1, 2]          # game 3 not played: never bet retroactively
    r1, r2 = led.iter_rows(named=True)
    # Game 1: opening consensus -3.5, model home +10 -> edge +6.5, home bet; won by 10.
    assert (r1["side"], r1["line"], r1["edge"], r1["status"]) == ("home", -3.5, 6.5, "won")
    assert r1["best_line"] == -3.0 and r1["best_provider"] == "DraftKings"  # Bovada is offshore
    assert r1["clv"] == pytest.approx(1.5) and r1["model"] == "v1"           # closed -5
    # Stamped just before the slot's first kickoff.
    assert datetime.fromisoformat(r1["placed_at"]) < T1 and datetime.fromisoformat(r2["placed_at"]) < T2
    # Game 2: opening +2, model home -8 -> edge -6, away bet; away won by 14.
    assert (r2["side"], r2["status"], r2["week"]) == ("away", "won", 2)
    md = retro_markdown(pl.concat([led, led.with_columns(model=pl.lit("roster"))]), Bets(), 2026, 6)
    assert "Not live bets" in md and "| Week 2 | 1 | 1-0-0 |" in md and "2 games were bet by both models" in md


def test_opening_books_skip_missing_opens():
    lines = pl.DataFrame({"game_id": [1, 2], "season": [2026, 2026], "provider": ["DraftKings"] * 2,
                          "spread_open": [-3.0, None], "spread": [-3.0, -1.0]})
    assert opening_books(lines, 2026)["game_id"].to_list() == [1]
