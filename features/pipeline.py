"""Feature pipeline.

Phase 1 is intentionally minimal: shape a single series into the ``(ds, y)``
frame Prophet expects and split it temporally. The full lag / rolling / Fourier
/ calendar suite arrives in Phase 2 (LightGBM on M5). The schema helper is here
from the start so every run logs the feature contract to MLflow, which is what
lets us detect schema drift at inference time later.
"""

from __future__ import annotations

import pandas as pd

from config import DatasetConfig


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


__all__ = ["to_prophet_frame", "temporal_train_test_split", "feature_schema"]
