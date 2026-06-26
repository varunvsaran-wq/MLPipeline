"""Prophet across the whole panel (Phase 2 comparison).

Prophet is a *local* model — one fit per series — which is the natural contrast
to the global LightGBM. Its native 80% prediction interval maps directly to the
p10 / p90 quantiles (with yhat as p50), so it slots into the same
:class:`SeriesForecast` harness and the same pinball / WRMSSE scoring.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from prophet import Prophet

from config import DatasetConfig
from features.pipeline import series_key
from models.evaluate import SeriesForecast

# Prophet/cmdstanpy are extremely chatty; quiet them for the panel loop.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# 80% interval -> lower ~ p10, upper ~ p90.
_INTERVAL_WIDTH = 0.8


def _series_weight(train_df: pd.DataFrame, cfg: DatasetConfig) -> float:
    if cfg.exogenous_cols and cfg.exogenous_cols[0] in train_df:
        return float(train_df[cfg.exogenous_cols[0]].abs().sum())
    return float(train_df[cfg.target_col].abs().sum())


def forecast_panel(
    cfg: DatasetConfig, raw: pd.DataFrame, max_series: int | None = None
) -> list[SeriesForecast]:
    """Fit Prophet per series and return held-out forecasts for the panel."""
    df = raw.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df["series_id"] = series_key(cfg, df).values
    all_dates = np.sort(df["ds" if "ds" in df else cfg.date_col].unique())
    test_dates = pd.to_datetime(all_dates[-cfg.horizon :])

    forecasts: list[SeriesForecast] = []
    series_ids = list(dict.fromkeys(df["series_id"]))
    if max_series is not None:
        series_ids = series_ids[:max_series]

    for sid in series_ids:
        g = df[df["series_id"] == sid].sort_values(cfg.date_col)
        train_df = g[~g[cfg.date_col].isin(test_dates)]
        test_g = g[g[cfg.date_col].isin(test_dates)]
        if len(train_df) < 2 or test_g.empty:
            continue

        fit_frame = train_df.rename(columns={cfg.date_col: "ds", cfg.target_col: "y"})[["ds", "y"]]
        model = Prophet(
            interval_width=_INTERVAL_WIDTH,
            weekly_seasonality=False,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        if cfg.holiday_country:
            model.add_country_holidays(country_name=cfg.holiday_country)
        model.fit(fit_frame)

        future = pd.DataFrame({"ds": test_g[cfg.date_col].to_numpy()})
        fc = model.predict(future)
        quantiles = {
            0.1: fc["yhat_lower"].to_numpy(),
            0.5: fc["yhat"].to_numpy(),
            0.9: fc["yhat_upper"].to_numpy(),
        }
        forecasts.append(
            SeriesForecast(
                series_id=sid,
                y_true=test_g[cfg.target_col].to_numpy(dtype=float),
                point=quantiles[0.5],
                quantiles=quantiles,
                train_history=train_df[cfg.target_col].to_numpy(dtype=float),
                weight=_series_weight(train_df, cfg),
            )
        )
    return forecasts


__all__ = ["forecast_panel"]
