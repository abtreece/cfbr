"""Points-to-cents pricing for paper bets (review B3).

A half point is not worth the same everywhere: a spread move through 3 or 7 changes
the cover probability far more than one through 15. This module estimates the final
margin distribution from past games with a closing line and prices any (number, price)
pair against it, so an edge, a best line and closing-line value can all be stated in
cents (fair-price difference) as well as points.

Empirical table: for every integer line size ``m`` (0..MAX_CENTER), the probability
mass of each integer *favorite* margin over past completed games whose closing
consensus was within ``window`` points of ``m`` (widened until at least ``min_games``
games are in the pool). Conditioning on the line keeps the mass at 3 and 7 where the
market puts it; the pooling window is the approximation. Everything downstream is
arithmetic on that table. Lines never feed the ratings; this prices bets only.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

MAX_CENTER = 35
MARGIN_LO, MARGIN_HI = -45, 75


def to_cents(odds: int | float) -> float:
    """American odds on a continuous cents scale (-110 -> -10, +105 -> +5; -100 and +100 both 0)."""
    return odds + 100 if odds < 0 else odds - 100


def from_cents(cents: float) -> int:
    """Back to American odds, rounded."""
    return int(round(cents - 100)) if cents < 0 else int(round(cents + 100))


def fair_american(p_win: float, p_push: float = 0.0) -> float:
    """The American price at which a bet with these win/push probabilities has zero expected value."""
    p_loss = max(1.0 - p_win - p_push, 0.0)
    if p_win <= 0:
        return math.inf
    b = p_loss / p_win          # break-even payout per unit risked
    return 100.0 * b if b >= 1 else -100.0 / b


@dataclass
class PricingTable:
    seasons: list[int]
    n_games: int
    window: float
    min_games: int
    pmf: dict[int, dict[int, float]]   # integer line size -> favorite margin -> probability

    # -- construction ------------------------------------------------------------
    @classmethod
    def build(cls, games: pl.DataFrame, window: float = 2.0, min_games: int = 300) -> "PricingTable":
        """``games``: completed games with ``margin`` (home) and ``spread`` (home closing consensus)."""
        d = (games.filter(pl.col("margin").is_not_null() & pl.col("spread").is_not_null())
             .with_columns(size=pl.col("spread").abs(),
                           fav_margin=pl.when(pl.col("spread") <= 0).then(pl.col("margin"))
                           .otherwise(-pl.col("margin")).cast(pl.Int64).clip(MARGIN_LO, MARGIN_HI)))
        sizes = d["size"].to_numpy()
        fav = d["fav_margin"].to_numpy()
        pmf: dict[int, dict[int, float]] = {}
        for m in range(MAX_CENTER + 1):
            w = window
            while True:
                sel = abs(sizes - m) <= w + 1e-9
                if sel.sum() >= min_games or w > MAX_CENTER:
                    break
                w += 1.0
            vals, counts = _counts(fav[sel])
            n = counts.sum()
            pmf[m] = {int(k): float(c / n) for k, c in zip(vals, counts)} if n else {}
        seasons = sorted(set(d["season"].to_list())) if "season" in d.columns else []
        return cls(seasons, int(d.height), window, min_games, pmf)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"seasons": self.seasons, "n_games": self.n_games, "window": self.window,
                                   "min_games": self.min_games,
                                   "pmf": {str(m): {str(k): p for k, p in row.items()} for m, row in self.pmf.items()}}))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "PricingTable":
        j = json.loads(path.read_text())
        return cls(j["seasons"], j["n_games"], j["window"], j["min_games"],
                   {int(m): {int(k): p for k, p in row.items()} for m, row in j["pmf"].items()})

    # -- probabilities --------------------------------------------------------------
    def margin_pmf(self, center: float) -> dict[int, float]:
        """Favorite-margin distribution for a market line of size |center|."""
        return self.pmf.get(min(int(round(abs(center))), MAX_CENTER), {})

    def win_push(self, side: str, number: float, center: float) -> tuple[float, float]:
        """(P(win), P(push)) for a bet on ``side`` ("home"/"away") at home spread ``number``, when the
        market's true line (the consensus, the close, or the model's own margin) is ``center``."""
        pmf = self.margin_pmf(center)
        if not pmf:
            return 0.5, 0.0
        home_fav = center <= 0
        # Home covers iff margin > -number; away iff margin < -number. Flip to the favorite axis.
        t = -number if home_fav else number
        above = (side == "home") == home_fav
        p_win = sum(p for k, p in pmf.items() if (k > t if above else k < t))
        p_push = pmf.get(int(t), 0.0) if float(t).is_integer() else 0.0
        return p_win, p_push

    def fair_price(self, side: str, number: float, center: float) -> float:
        return fair_american(*self.win_push(side, number, center))

    def expected_value(self, side: str, number: float, odds: int, center: float) -> float:
        """Units won per unit risked at this number and price when the true line is ``center``."""
        p_win, p_push = self.win_push(side, number, center)
        payout = odds / 100 if odds > 0 else 100 / -odds
        return p_win * payout - (1 - p_win - p_push)

    # -- cents ------------------------------------------------------------------------
    def number_value_cents(self, side: str, number: float, center: float) -> float:
        """How much better ``number`` is than ``center`` for this side, in cents of fair price
        (positive = the number is worth laying more juice for)."""
        return to_cents(self.fair_price(side, center, center)) - to_cents(self.fair_price(side, number, center))

    def edge_cents(self, side: str, number: float, odds: int, model_center: float) -> float:
        """Value of a bet at ``number`` and ``odds`` if the model's margin were the true line
        (model_center = -model home margin). As over-confident as the model's own cover probability;
        reporting only."""
        return to_cents(odds) - to_cents(self.fair_price(side, number, model_center))

    def clv_cents(self, side: str, number: float, closing: float) -> float:
        """Closing-line value in cents: the fair price of the recorded number against the close,
        relative to the close itself."""
        return self.number_value_cents(side, number, closing)

    def key_numbers(self, center: float = 3.0, top: int = 8) -> list[tuple[int, float]]:
        """Integer margins with the most mass for a line of this size, largest first."""
        pmf = self.margin_pmf(center)
        return sorted(pmf.items(), key=lambda kv: -kv[1])[:top]


def _counts(values):
    import numpy as np

    return np.unique(values, return_counts=True)


def cover_probability_if_right(table: PricingTable, edge: float, centers=range(1, 22)) -> float:
    """Average P(cover) of a bet whose true line is ``edge`` points better than the number it got
    (no pushes: half-point numbers), over typical line sizes. Used for win-rate power."""
    ps = []
    for c in centers:
        number = -(c + 0.5)                       # home favored by c + 0.5
        ps.append(table.win_push("home", number, number - edge)[0])
    return sum(ps) / len(ps)


def bets_for_power(p: float, breakeven: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Flat bets needed for a one-sided test of win rate > break-even at this true rate ``p``."""
    if p <= breakeven:
        return 0
    z_a, z_b = 1.644854, {0.8: 0.841621, 0.9: 1.281552}.get(power, 0.841621)
    n = ((z_a * math.sqrt(breakeven * (1 - breakeven)) + z_b * math.sqrt(p * (1 - p))) / (p - breakeven)) ** 2
    return int(math.ceil(n))


def table_from_dataset(ds, cfg, through_season: int) -> PricingTable:
    """Build from the store: completed games through ``through_season`` with a closing consensus."""
    from cfbr.backtest.metrics import consensus_spread

    g = ds.games.filter(pl.col("completed") & (pl.col("season") <= through_season)).with_columns(
        margin=(pl.col("home_points") - pl.col("away_points")).cast(pl.Float64))
    g = g.join(consensus_spread(ds.lines, cfg.backtest.spread_consensus).select("game_id", "spread"), on="game_id")
    return PricingTable.build(g.select("season", "margin", "spread"))
