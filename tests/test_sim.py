import numpy as np
import pytest

from cfbr.sim.season import Inputs, Rules, simulate, summary


def _league(sigma=10.0):
    """Two 4-team conferences (SEC, Sun Belt) plus Notre Dame, 1 game left per pair as listed."""
    teams = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "Notre Dame"]
    conf = ["SEC"] * 4 + ["Sun Belt"] * 4 + ["FBS Independents"]
    strength = np.array([30, 10, 0, -10, 5, 0, -5, -10, 20.0])
    n = len(teams)
    games = [(0, 1, True), (2, 3, True), (4, 5, True), (6, 7, True), (8, 0, False), (8, -1, False)]
    home = np.array([g[0] for g in games]); away = np.array([g[1] for g in games])
    mean = np.array([strength[h] - (strength[a] if a >= 0 else -20.0) for h, a, _ in games])
    return Inputs(season=2026, teams=teams, conference=conf, strength=strength, b_home=0.0, sigma=sigma,
                  wins0=np.zeros(n), losses0=np.zeros(n), conf_wins0=np.zeros(n), conf_losses0=np.zeros(n),
                  g_home=home, g_away=away, g_mean=mean, g_conf=np.array([g[2] for g in games]),
                  g_opp_strength_home=np.zeros(len(games)), g_opp_strength_away=np.zeros(len(games)))


def test_totals_are_consistent():
    inp = _league()
    rules = Rules(power_conferences=["SEC"], field_size=4, byes=2)
    out = simulate(inp, n_sims=4000, seed=1, rules=rules)
    s = summary(inp, out)
    assert s["p_conf_title"].sum() == pytest.approx(2.0)      # one champion per conference
    assert s["p_title_game"].sum() == pytest.approx(4.0)      # two teams per title game
    assert s["p_playoff"].sum() == pytest.approx(4.0)         # field size
    assert s["p_bye"].sum() == pytest.approx(2.0)


def test_game_probabilities_match_the_normal_model():
    inp = _league(sigma=10.0)
    out = simulate(inp, n_sims=40_000, seed=3, postseason=False)
    s = {r["team"]: r for r in summary(inp, out).iter_rows(named=True)}
    # A3 vs A4: 10-point favorite, sigma 10 -> P(win) = Phi(1) = 0.841; it plays one game.
    assert s["A3"]["exp_wins"] == pytest.approx(0.841, abs=0.01)
    # Notre Dame: vs A1 (-10 -> 0.159) and vs a non-FBS side (+40 -> ~1).
    assert s["Notre Dame"]["exp_wins"] == pytest.approx(0.159 + 1.0, abs=0.01)


def test_team_offsets_keep_game_spread_but_correlate_results():
    inp = _league(sigma=10.0)
    base = summary(inp, simulate(inp, n_sims=40_000, tau=0.0, seed=5, postseason=False))
    corr = summary(inp, simulate(inp, n_sims=40_000, tau=6.0, seed=5, postseason=False))
    # Same single-game probability for a team with one game left...
    a3 = lambda s: s.filter(s["team"] == "A3")["exp_wins"].item()
    assert a3(base) == pytest.approx(a3(corr), abs=0.01)
    # ...but Notre Dame's two games now move together: more 0-win and 2-win seasons.
    def spread(s):
        r = s.filter(s["team"] == "Notre Dame").row(0, named=True)
        return r["wins_p90"] - r["wins_p10"]
    assert spread(corr) >= spread(base)


def test_notre_dame_gets_in_only_when_ranked_and_the_g6_champion_rule():
    inp = _league()
    rules = Rules(power_conferences=["SEC"], field_size=4, byes=2, notre_dame_top=4)
    s = {r["team"]: r for r in summary(inp, simulate(inp, n_sims=4000, seed=2, rules=rules)).iter_rows(named=True)}
    # Sun Belt (a Group of Six league here) always sends its champion.
    assert sum(s[t]["p_playoff"] for t in ("B1", "B2", "B3", "B4")) >= 0.999
    assert s["Notre Dame"]["p_playoff"] > 0.5 and s["Notre Dame"]["p_conf_title"] == 0


def test_title_games_are_found_and_army_navy_is_not_one():
    from datetime import UTC, datetime

    import polars as pl

    from cfbr.sim.season import title_game_ids
    d = lambda m, day: datetime(2026, m, day, 20, tzinfo=UTC)
    games = pl.DataFrame({
        "game_id": [1, 2, 3, 4, 5],
        "season_type": ["regular"] * 5,
        "week": [13, 14, 13, 16, 14],
        "kickoff_ts": [d(11, 28), d(12, 5), d(11, 28), d(12, 12), d(12, 5)],
        "home_conference": ["SEC", "SEC", "American Athletic", "American Athletic", "American Athletic"],
        "home_team": ["Georgia", "Georgia", "Tulane", "Army", "Tulane"],
        "away_team": ["Auburn", "Alabama", "Memphis", "Navy", "Navy"],
    })
    assert title_game_ids(games, {1, 2, 3, 4, 5}) == {2, 5}
    assert title_game_ids(games.filter(pl.col("game_id") != 5), {1, 2, 3, 4}) == {2}  # AAC title not scheduled yet
