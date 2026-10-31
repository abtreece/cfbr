"""Preseason prior: predict each team's season-s ratings from preseason features.

Features (all known before week 1 of season s):
  - final (no-prior) ratings from season s-1, same component
  - returning production (offense components only; CFBD has no defensive split)
  - team talent composite
  - 4-year weighted recruiting points

Targets are each season's final no-prior ratings. One linear model per rating
component (o_ppa, d_ppa, o_sr, d_sr).
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression

from cfbr.model.roster import DEF_FEATURES, OFF_FEATURES

log = logging.getLogger(__name__)

COMPONENTS = ["o_ppa", "d_ppa", "o_sr", "d_sr"]
# Recruiting class weights by years before the season (0 = incoming class).
RECRUIT_WEIGHTS = {0: 0.15, 1: 0.30, 2: 0.30, 3: 0.25}
OFFENSE_RETURNING = ["percent_ppa", "usage"]


def _features_for(component: str, mode: str = "v1") -> list[str]:
    base = [f"prev_{component}", "talent", "recruit4"]
    feats = base + OFFENSE_RETURNING if component.startswith("o_") else base
    if mode == "roster":
        feats = feats + (OFF_FEATURES if component.startswith("o_") else DEF_FEATURES)
    elif mode != "v1":
        raise ValueError(f"unknown prior.features {mode!r}")
    return feats


def recruiting_avg(recruiting: pl.DataFrame, season: int) -> pl.DataFrame:
    parts = [
        recruiting.filter(pl.col("season") == season - lag).select(
            "team", (pl.col("points") * w).alias("wp"), pl.lit(w).alias("w"))
        for lag, w in RECRUIT_WEIGHTS.items()
    ]
    return (pl.concat(parts).group_by("team")
            .agg(recruit4=pl.col("wp").sum() / pl.col("w").sum()))


def season_features(
    season: int, teams: list[str], finals: dict[int, pl.DataFrame],
    returning: pl.DataFrame, talent: pl.DataFrame, recruiting: pl.DataFrame,
) -> pl.DataFrame:
    """Feature frame for the FBS ``teams`` of ``season``. Missing returning or
    talent values are imputed with that season's mean (both are published
    preseason, so the mean is preseason information too)."""
    df = pl.DataFrame({"team": teams})
    prev = finals.get(season - 1)
    if prev is not None:
        df = df.join(prev.select("team", *[pl.col(c).alias(f"prev_{c}") for c in COMPONENTS]), on="team", how="left")
    else:
        df = df.with_columns([pl.lit(None, pl.Float64).alias(f"prev_{c}") for c in COMPONENTS])
    df = df.join(returning.filter(pl.col("season") == season).select("team", *OFFENSE_RETURNING), on="team", how="left")
    df = df.join(talent.filter(pl.col("season") == season).select("team", "talent"), on="team", how="left")
    df = df.join(recruiting_avg(recruiting, season), on="team", how="left")
    fill = {}
    for c in OFFENSE_RETURNING + ["talent"]:
        mean = df[c].mean()
        fill[c] = pl.col(c).fill_null(mean if mean is not None else 0.0)
    rmin = df["recruit4"].min()
    fill["recruit4"] = pl.col("recruit4").fill_null(rmin if rmin is not None else 0.0)
    df = df.with_columns(**fill)
    return df.with_columns(season=pl.lit(season), new_program=pl.col("prev_o_ppa").is_null())


PriorModels = dict[str, list[tuple[list[str], LinearRegression]]]


def fit_prior_models(train: pl.DataFrame, mode: str = "v1", roster_min_season: int = 0) -> PriorModels:
    """``train`` = stacked season_features rows joined with targets (``o_ppa`` ...).

    Returns, per component, models in priority order. In roster mode the first
    model adds the roster features (fit on rows that have them) and the v1 model
    is the fallback for rows that don't (e.g. no defensive player stats before 2016).
    """
    rows = train.filter(~pl.col("new_program"))
    models: PriorModels = {}
    for comp in COMPONENTS:
        models[comp] = []
        modes = ["roster", "v1"] if mode == "roster" else ["v1"]
        for m in modes:
            feats = _features_for(comp, m)
            if any(f not in rows.columns for f in feats):
                continue
            pool = rows.filter(pl.col("season") >= roster_min_season) if m == "roster" and roster_min_season else rows
            r = pool.drop_nulls([comp, *feats])
            if r.height < 2 * len(feats) + 10:
                continue
            models[comp].append((feats, LinearRegression().fit(r.select(feats).to_numpy(), r[comp].to_numpy())))
    return models


def predict_priors(models: dict[str, LinearRegression], feats: pl.DataFrame) -> pl.DataFrame:
    """Prior per FBS team. New FBS programs (no prior-season FBS rating) get the
    mean of the bottom quartile of this season's predicted priors."""
    out = feats.select("team", "new_program")
    known = feats.filter(~pl.col("new_program"))
    cols = {}
    for comp, candidates in models.items():
        cols[comp] = {}
        for feats, m in candidates:  # first model whose features the row has
            todo = known.filter(~pl.col("team").is_in(list(cols[comp])))
            if any(f not in todo.columns for f in feats):
                continue
            rows = todo.drop_nulls(feats)
            if rows.height:
                cols[comp].update(zip(rows["team"].to_list(), m.predict(rows.select(feats).to_numpy())))
    preds = pl.DataFrame({"team": known["team"], **{c: [cols[c][t] for t in known["team"]] for c in COMPONENTS}})
    out = out.join(preds, on="team", how="left")
    if out["new_program"].any() and preds.height:
        net = preds.with_columns(net=pl.col("o_ppa") + pl.col("d_ppa"))
        q = net["net"].quantile(0.25)
        bottom = net.filter(pl.col("net") <= q)
        fill = {c: bottom[c].mean() for c in COMPONENTS}
        out = out.with_columns(**{c: pl.col(c).fill_null(v) for c, v in fill.items()})
    return out.with_columns([pl.col(c).fill_null(0.0) for c in COMPONENTS])


def pooled_priors(season: int, entities: list[str], finals: dict[int, pl.DataFrame]) -> pl.DataFrame:
    """Priors for pooled non-FBS entities: carry over last season's final
    rating for the same entity, else the mean over last season's pooled entities."""
    prev = finals.get(season - 1)
    if prev is None or not entities:
        return pl.DataFrame({"team": entities, **{c: [0.0] * len(entities) for c in COMPONENTS}})
    pooled_prev = prev.filter(pl.col("team").str.starts_with("FCS"))
    means = {c: (pooled_prev[c].mean() if pooled_prev.height else 0.0) for c in COMPONENTS}
    df = pl.DataFrame({"team": entities}).join(prev.select("team", *COMPONENTS), on="team", how="left")
    return df.with_columns(**{c: pl.col(c).fill_null(means[c]) for c in COMPONENTS})
