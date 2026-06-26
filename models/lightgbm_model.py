"""Global LightGBM forecaster over the full panel (Phase 2).

One model is trained across every series using the tabular feature suite, with
the series identity passed as categorical features. Three quantile models
(p10 / p50 / p90) give the probabilistic forecast; p50 is the point forecast.

The held-out horizon is produced by **recursive** multi-step forecasting: we
hide the validation actuals, then walk the horizon one step at a time, feeding
each step's p50 prediction back in as history before recomputing features for
the next step. That keeps lag/rolling features honest (no peeking at future
actuals) and reuses :func:`build_feature_matrix` so train/inference features
never diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from config import DatasetConfig
from features.pipeline import FeatureSpec, build_feature_matrix
from models.evaluate import QUANTILES, SeriesForecast

_LGBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)


def _fit_quantile_models(train: pd.DataFrame, spec: FeatureSpec) -> dict[float, LGBMRegressor]:
    """Fit one LightGBM per quantile on rows that have a target."""
    fit_rows = train.dropna(subset=["y"])
    x = fit_rows[spec.all]
    y = fit_rows["y"]
    models: dict[float, LGBMRegressor] = {}
    for q in QUANTILES:
        model = LGBMRegressor(objective="quantile", alpha=q, **_LGBM_PARAMS)
        model.fit(x, y, categorical_feature=spec.categorical)
        models[q] = model
    return models


def _series_weight(train_rows: pd.DataFrame, cfg: DatasetConfig) -> float:
    """WRMSSE weight: dollar-style volume if an exogenous volume col exists, else |target|."""
    if cfg.exogenous_cols and cfg.exogenous_cols[0] in train_rows:
        return float(train_rows[cfg.exogenous_cols[0]].abs().sum())
    return float(train_rows["y"].abs().sum())


def train_and_forecast(
    cfg: DatasetConfig, raw: pd.DataFrame
) -> tuple[list[SeriesForecast], dict[float, LGBMRegressor], FeatureSpec, dict]:
    """Train the global quantile models and recursively forecast the horizon."""
    frame, spec = build_feature_matrix(cfg, raw)
    all_dates = np.sort(frame["ds"].unique())
    test_dates = all_dates[-cfg.horizon :]
    train_frame = frame[frame["ds"] < test_dates[0]]

    models = _fit_quantile_models(train_frame, spec)

    # Recursive walk: hide validation actuals, predict step by step.
    work = raw.copy()
    work[cfg.date_col] = pd.to_datetime(work[cfg.date_col])
    from features.pipeline import series_key

    work["series_id"] = series_key(cfg, work).values
    is_test = work[cfg.date_col].isin(pd.to_datetime(test_dates))
    work.loc[is_test, cfg.target_col] = np.nan

    preds: dict[float, dict[tuple[str, pd.Timestamp], float]] = {q: {} for q in QUANTILES}
    for dt in pd.to_datetime(test_dates):
        fm, _ = build_feature_matrix(cfg, work)
        cur = fm[fm["ds"] == dt]
        x_cur = cur[spec.all]
        qpred = {q: models[q].predict(x_cur) for q in QUANTILES}
        # enforce non-crossing quantiles row-wise
        stacked = np.sort(np.vstack([qpred[q] for q in QUANTILES]), axis=0)
        for i, q in enumerate(QUANTILES):
            for sid, val in zip(cur["series_id"], stacked[i], strict=True):
                preds[q][(sid, dt)] = float(val)
        # feed p50 back in as the realised value for the next step's lags
        p50_map = {
            sid: v for sid, v in zip(cur["series_id"], stacked[QUANTILES.index(0.5)], strict=True)
        }
        rows = work[cfg.date_col] == dt
        work.loc[rows, cfg.target_col] = work.loc[rows, "series_id"].map(p50_map).to_numpy()

    return _assemble_forecasts(cfg, raw, test_dates, preds, models, spec)


def _assemble_forecasts(cfg, raw, test_dates, preds, models, spec):
    from features.pipeline import series_key

    df = raw.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df["series_id"] = series_key(cfg, df).values
    test_dates = pd.to_datetime(test_dates)

    forecasts: list[SeriesForecast] = []
    for sid, g in df.groupby("series_id", sort=False):
        g = g.sort_values(cfg.date_col)
        test_g = g[g[cfg.date_col].isin(test_dates)]
        if test_g.empty:
            continue
        order = test_g[cfg.date_col].to_list()
        y_true = test_g[cfg.target_col].to_numpy(dtype=float)
        quantiles = {
            q: np.array([preds[q][(sid, dt)] for dt in order], dtype=float) for q in QUANTILES
        }
        train_g = g[~g[cfg.date_col].isin(test_dates)]
        forecasts.append(
            SeriesForecast(
                series_id=sid,
                y_true=y_true,
                point=quantiles[0.5],
                quantiles=quantiles,
                train_history=train_g[cfg.target_col].to_numpy(dtype=float),
                weight=_series_weight(train_g, cfg),
            )
        )
    info = {"n_features": len(spec.all), "n_estimators": _LGBM_PARAMS["n_estimators"]}
    return forecasts, models, spec, info


__all__ = ["train_and_forecast", "_LGBM_PARAMS"]
