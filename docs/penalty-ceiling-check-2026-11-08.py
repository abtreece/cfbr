"""Penalty ceiling check (rule: DECISIONS.md "Penalty ceiling check", 2026-11-08).

Residual = margin − pred_margin from the frozen tuned backtests. Features are
point-in-time team penalty stats from penalty-only rows of plays_raw.

  uv run python docs/penalty-ceiling-check-2026-11-08.py [repo_root]

repo_root defaults to the current directory and must hold data/cfb.duckdb and
results/. Reads only cached data; makes no API calls.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
TUNING_RUN = ROOT / 'results/20261011T144444Z-tuned-tuning/predictions.parquet'
HOLDOUT_RUN = ROOT / 'results/20261011T145524Z-tuned-holdout/predictions.parquet'
SHRINK_GAMES = 3.0
GAIN_THRESHOLD = 0.03

TEAM_GAMES_SQL = '''
WITH pen AS (
  SELECT p.game_id, p.season, p.offense, p.defense, p.yards_gained AS y
  FROM plays_raw p
  WHERE p.play_type = 'Penalty' AND p.yards_gained IS NOT NULL AND p.yards_gained <> 0
    AND abs(p.yards_gained) <= 99  -- amendment: |y| > 99 is a parse error
),
team_game AS (
  SELECT DISTINCT game_id, season, offense AS team FROM plays_raw WHERE offense IS NOT NULL
  UNION
  SELECT DISTINCT game_id, season, defense AS team FROM plays_raw WHERE defense IS NOT NULL
),
per_team AS (
  SELECT tg.game_id, tg.season, tg.team,
    COALESCE(SUM(CASE WHEN pen.offense = tg.team THEN pen.y
                      WHEN pen.defense = tg.team THEN -pen.y END), 0) AS net_yards,
    COALESCE(SUM(CASE WHEN (pen.offense = tg.team AND pen.y < 0)
                        OR (pen.defense = tg.team AND pen.y > 0) THEN 1 ELSE 0 END), 0) AS committed
  FROM team_game tg
  LEFT JOIN pen ON pen.game_id = tg.game_id AND (pen.offense = tg.team OR pen.defense = tg.team)
  GROUP BY 1, 2, 3
)
SELECT pt.*, g.kickoff_ts
FROM per_team pt JOIN games g USING (game_id)
'''


def load_team_games(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
  team_games = con.execute(TEAM_GAMES_SQL).pl()
  return team_games.with_columns(pl.col('kickoff_ts').dt.replace_time_zone('Etc/UTC'))


def features_for_slot(team_games: pl.DataFrame, season: int, as_of) -> pl.DataFrame:
  """Per-team F1/F2 using only that season's games that kicked off before as_of."""
  before = team_games.filter((pl.col('season') == season) & (pl.col('kickoff_ts') < as_of))
  if before.height == 0:
    return pl.DataFrame(schema={'team': pl.String, 'f1': pl.Float64, 'f2': pl.Float64})
  league_committed_per_team_game = before['committed'].sum() / before.height
  per_team = before.group_by('team').agg(
    pl.len().alias('g'),
    pl.col('net_yards').sum().alias('net'),
    pl.col('committed').sum().alias('committed'),
  )
  return per_team.select(
    'team',
    (pl.col('net') / (pl.col('g') + SHRINK_GAMES)).alias('f1'),
    ((pl.col('committed') + SHRINK_GAMES * league_committed_per_team_game) / (pl.col('g') + SHRINK_GAMES)
     - league_committed_per_team_game).alias('f2'),
  )


def attach_features(predictions: pl.DataFrame, team_games: pl.DataFrame) -> pl.DataFrame:
  parts = []
  for (season, as_of), slot_games in predictions.group_by(['season', 'as_of'], maintain_order=True):
    feats = features_for_slot(team_games, season, as_of)
    home = feats.rename({'team': 'home_team', 'f1': 'f1_h', 'f2': 'f2_h'})
    away = feats.rename({'team': 'away_team', 'f1': 'f1_a', 'f2': 'f2_a'})
    parts.append(slot_games.join(home, on='home_team', how='left').join(away, on='away_team', how='left'))
  joined = pl.concat(parts).with_columns(pl.col('^f[12]_[ha]$').fill_null(0.0))
  return joined.with_columns(
    (pl.col('margin') - pl.col('pred_margin')).alias('resid'),
    (pl.col('f1_h') - pl.col('f1_a')).alias('d_f1'),
    (pl.col('f2_h') - pl.col('f2_a')).alias('d_f2'),
  )


def ols_fit_predict(train: pl.DataFrame, test: pl.DataFrame, cols: list[str]) -> np.ndarray:
  def design(df):
    return np.column_stack([np.ones(df.height)] + [df[c].to_numpy() for c in cols])
  coef, *_ = np.linalg.lstsq(design(train), train['resid'].to_numpy(), rcond=None)
  return design(test) @ coef, coef


def paired_gain(resid: np.ndarray, fitted: np.ndarray) -> tuple[float, float, float]:
  diff = np.abs(resid) - np.abs(resid - fitted)
  mean = diff.mean()
  half_width = 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
  return mean, mean - half_width, mean + half_width


def evaluate(frame: pl.DataFrame, cols: list[str], splits) -> dict:
  resid_parts, fitted_parts, coefs = [], [], []
  for train_mask, test_mask in splits:
    train, test = frame.filter(train_mask), frame.filter(test_mask)
    fitted, coef = ols_fit_predict(train, test, cols)
    resid_parts.append(test['resid'].to_numpy())
    fitted_parts.append(fitted)
    coefs.append(coef)
  resid = np.concatenate(resid_parts)
  fitted = np.concatenate(fitted_parts)
  mean, lo, hi = paired_gain(resid, fitted)
  return {
    'n': len(resid), 'mae_base': np.abs(resid).mean(), 'mae_new': np.abs(resid - fitted).mean(),
    'gain': mean, 'lo': lo, 'hi': hi, 'coef_mean': np.mean(coefs, axis=0),
  }


def loso_splits(seasons: list[int]):
  return [(pl.col('season') != s, pl.col('season') == s) for s in seasons]


def expanding_splits(train_first: int, test_seasons: list[int]):
  return [((pl.col('season') >= train_first) & (pl.col('season') < s), pl.col('season') == s) for s in test_seasons]


def fmt(label: str, result: dict, cols: list[str]) -> str:
  coef = ', '.join(f'{c}: {v:+.4f}' for c, v in zip(['int'] + cols, result['coef_mean']))
  return (f"| {label} | {result['n']} | {result['mae_base']:.3f} | {result['mae_new']:.3f} | "
          f"{result['gain']:+.4f} [{result['lo']:+.4f}, {result['hi']:+.4f}] | {coef} |")


def main() -> None:
  con = duckdb.connect(str(ROOT / 'data/cfb.duckdb'), read_only=True)
  team_games = load_team_games(con)
  tuning = attach_features(pl.read_parquet(TUNING_RUN), team_games)
  holdout = attach_features(pl.read_parquet(HOLDOUT_RUN), team_games)
  combined = pl.concat([tuning, holdout], how='diagonal_relaxed')

  tuning_seasons = list(range(2015, 2022))
  sanity_seasons = [2022, 2023, 2024]
  specs = [('intercept only', []), ('F1 net yards (primary)', ['d_f1']), ('F2 committed', ['d_f2']),
           ('F1 + F2', ['d_f1', 'd_f2'])]

  print('Feature summary, 2015–2021 (per-team values before Δ):')
  print(tuning.select(
    pl.col('d_f1').std().alias('sd_d_f1'), pl.col('d_f2').std().alias('sd_d_f2'),
    pl.corr('d_f1', 'd_f2').alias('corr_f1_f2'),
    pl.corr('d_f1', 'resid').alias('corr_f1_resid'), pl.corr('d_f2', 'resid').alias('corr_f2_resid'),
    pl.corr('d_f1', 'pred_margin').alias('corr_f1_pred'),
  ))
  print()
  print('| spec | n | MAE base | MAE new | gain [95% CI] | mean coefficients |')
  print('|---|---|---|---|---|---|')
  results = {}
  for label, cols in specs:
    results[label] = evaluate(tuning, cols, loso_splits(tuning_seasons))
    print(fmt(f'{label}, 2015–2021 LOSO', results[label], cols))
  for label, cols in specs:
    sanity = evaluate(combined, cols, expanding_splits(2015, sanity_seasons))
    print(fmt(f'{label}, 2022–2024 expanding (sanity)', sanity, cols))

  primary = results['F1 net yards (primary)']
  passed = primary['gain'] >= GAIN_THRESHOLD and primary['lo'] > 0
  print()
  print(f"Primary verdict: {'PASS' if passed else 'MISS'} "
        f"(gain {primary['gain']:+.4f}, CI lower {primary['lo']:+.4f}; needs ≥ {GAIN_THRESHOLD} and > 0)")


if __name__ == '__main__':
  main()
