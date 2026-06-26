"""Feature pipeline.

Two layers:

* Phase 1 (Prophet): shape a single series into the ``(ds, y)`` frame Prophet
  expects and split it temporally.
* Phase 2 (LightGBM): a full tabular feature suite over the whole panel — lags,
  rolling stats, calendar, Fourier seasonality, and lagged exogenous regressors
  — driven entirely by :class:`config.FeatureConfig` so the same code serves
  weekly Avocado and daily M5.

The schema helper logs the feature contract to MLflow every run, which is what
lets us detect schema drift at inference time later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import DatasetConfig

# Fixed epoch so Fourier phases are comparable across runs/series.
_FOURIER_EPOCH = pd.Timestamp("2010-01-01")
# Approx days per base-frequency step, for converting Fourier periods to days.
_STEP_DAYS = {"D": 1.0, "W": 7.0, "M": 30.4375, "Q": 91.3125, "Y": 365.25}


@dataclass
class FeatureSpec:
    """Engineered columns produced by :func:`build_feature_matrix`."""

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical


def to_prophet_frame(cfg: DatasetConfig, series: pd.DataFrame) -> pd.DataFrame:
    """Rename a ``[date, target]`` series to Prophet's required ``[ds, y]``."""
    out = series.rename(columns={cfg.date_col: "ds", cfg.target_col: "y"})
    return out[["ds", "y"]].reset_index(drop=True)


def temporal_train_test_split(
    frame: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the final ``horizon`` rows as the validation set (no shuffling)."""
    if len(frame) <= horizon:
        raise ValueError(
            f"Series has {len(frame)} rows but horizon is {horizon}; need more history."
        )
    train = frame.iloc[:-horizon].reset_index(drop=True)
    test = frame.iloc[-horizon:].reset_index(drop=True)
    return train, test


def feature_schema(frame: pd.DataFrame) -> dict[str, dict]:
    """Describe each column (dtype + observed range) for MLflow schema logging.

    Logging this per run is what makes schema drift detectable: at inference we
    compare incoming columns/dtypes/ranges against the schema the model was
    trained on.
    """
    schema: dict[str, dict] = {}
    for col in frame.columns:
        s = frame[col]
        entry: dict = {"dtype": str(s.dtype)}
        if pd.api.types.is_numeric_dtype(s):
            entry["min"] = float(s.min())
            entry["max"] = float(s.max())
        elif pd.api.types.is_datetime64_any_dtype(s):
            entry["min"] = str(s.min())
            entry["max"] = str(s.max())
        schema[col] = entry
    return schema


# --- Phase 2: full tabular feature suite ----------------------------------


def _sanitize(name: str) -> str:
    """LightGBM rejects special chars in feature names; make them safe."""
    return "".join(c if c.isalnum() else "_" for c in name)


def series_key(cfg: DatasetConfig, df: pd.DataFrame) -> pd.Series:
    """A single string id per series (join of ``series_id_cols``)."""
    if not cfg.series_id_cols:
        return pd.Series([cfg.name] * len(df), index=df.index)
    return df[cfg.series_id_cols].astype(str).agg("|".join, axis=1)


def _add_lag_roll_features(d: pd.DataFrame, cfg: DatasetConfig, numeric: list[str]) -> None:
    """Lags and rolling mean/std/min/max of the target, grouped per series."""
    fc = cfg.features
    g = d.groupby("series_id", sort=False)[cfg.target_col]
    for lag in fc.lags:
        col = f"lag_{lag}"
        d[col] = g.transform(lambda s, lag=lag: s.shift(lag))
        numeric.append(col)
    for w in fc.rolling_windows:
        minp = max(2, w // 2)
        # shift(1) so the window never sees the current step (no leakage).
        for stat in ("mean", "std", "min", "max"):
            col = f"roll_{stat}_{w}"
            d[col] = g.transform(
                lambda s, w=w, minp=minp, stat=stat: getattr(
                    s.shift(1).rolling(w, min_periods=minp), stat
                )()
            )
            numeric.append(col)


def _add_exogenous_features(d: pd.DataFrame, cfg: DatasetConfig, numeric: list[str]) -> None:
    fc = cfg.features
    if not fc.use_exogenous:
        return
    for col in cfg.exogenous_cols:
        if col not in d.columns:
            continue
        out = f"{_sanitize(col)}_lag{fc.exogenous_lag}"
        d[out] = d.groupby("series_id", sort=False)[col].transform(
            lambda s, lag=fc.exogenous_lag: s.shift(lag)
        )
        numeric.append(out)


def _add_calendar_features(d: pd.DataFrame, cfg: DatasetConfig, numeric: list[str]) -> None:
    dt = d[cfg.date_col].dt
    d["month"] = dt.month
    d["weekofyear"] = dt.isocalendar().week.astype("int32")
    d["quarter"] = dt.quarter
    d["dayofweek"] = dt.dayofweek
    d["is_weekend"] = (dt.dayofweek >= 5).astype("int8")
    numeric += ["month", "weekofyear", "quarter", "dayofweek", "is_weekend"]

    if cfg.holiday_country:
        import holidays

        years = range(int(dt.year.min()), int(dt.year.max()) + 1)
        cal = holidays.country_holidays(cfg.holiday_country, years=years)
        d["is_holiday"] = d[cfg.date_col].dt.date.map(lambda x: x in cal).astype("int8")
    else:
        d["is_holiday"] = 0
    numeric.append("is_holiday")


def _add_fourier_features(d: pd.DataFrame, cfg: DatasetConfig, numeric: list[str]) -> None:
    fc = cfg.features
    if not fc.fourier:
        return
    step_days = _STEP_DAYS.get(cfg.freq.upper()[0], 1.0)
    t_days = (d[cfg.date_col] - _FOURIER_EPOCH).dt.days.to_numpy()
    for i, term in enumerate(fc.fourier):
        period_days = term.period * step_days
        for k in range(1, term.order + 1):
            ang = 2.0 * np.pi * k * t_days / period_days
            for fn_name, fn in (("sin", np.sin), ("cos", np.cos)):
                col = f"fourier{i}_{fn_name}{k}"
                d[col] = fn(ang)
                numeric.append(col)


def build_feature_matrix(cfg: DatasetConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
    """Build the full panel feature matrix for tree models.

    Returns the long frame (one row per series-step, with ``ds``/``y`` plus all
    engineered columns) and the :class:`FeatureSpec` listing model inputs.
    Lag/rolling NaNs at each series' start are left in place — LightGBM handles
    them natively.
    """
    d = df.copy()
    d[cfg.date_col] = pd.to_datetime(d[cfg.date_col])
    d["series_id"] = series_key(cfg, d).values
    d = d.sort_values(["series_id", cfg.date_col]).reset_index(drop=True)

    numeric: list[str] = []
    _add_lag_roll_features(d, cfg, numeric)
    _add_exogenous_features(d, cfg, numeric)
    _add_calendar_features(d, cfg, numeric)
    _add_fourier_features(d, cfg, numeric)

    # Series identity as categorical inputs for the global model.
    categorical: list[str] = []
    for col in cfg.series_id_cols:
        d[col] = d[col].astype("category")
        categorical.append(col)

    d["ds"] = d[cfg.date_col]
    d["y"] = d[cfg.target_col]
    return d, FeatureSpec(numeric=numeric, categorical=categorical)


def panel_train_test_split(
    frame: pd.DataFrame, horizon: int, date_col: str = "ds"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each series' final ``horizon`` steps into the test set (by date).

    Uses a global date cutoff so train never contains any timestamp from the
    validation window of any series.
    """
    cutoff = frame.sort_values(date_col)[date_col].unique()[-horizon]
    train = frame[frame[date_col] < cutoff].reset_index(drop=True)
    test = frame[frame[date_col] >= cutoff].reset_index(drop=True)
    return train, test


__all__ = [
    "to_prophet_frame",
    "temporal_train_test_split",
    "feature_schema",
    "build_feature_matrix",
    "panel_train_test_split",
    "series_key",
    "FeatureSpec",
]
