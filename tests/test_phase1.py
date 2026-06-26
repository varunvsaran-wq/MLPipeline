"""Phase 1 unit tests: metrics, feature transforms, and series selection.

These exercise the dataset-agnostic plumbing without any heavy model deps. The
end-to-end Prophet run is covered by an integration test that skips when
prophet isn't installed (it lives in the optional ``[models]`` group).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DatasetConfig
from data.loader import select_series
from features.pipeline import (
    feature_schema,
    temporal_train_test_split,
    to_prophet_frame,
)
from models.metrics import forecast_bias, pinball_loss, wmape

# --- metrics ---------------------------------------------------------------


def test_wmape_perfect_forecast_is_zero():
    y = [10.0, 20.0, 30.0]
    assert wmape(y, y) == 0.0


def test_wmape_known_value():
    # errors |2|+|2| = 4 over sum|y| = 10+20 = 30 -> 0.1333...
    assert wmape([10.0, 20.0], [12.0, 18.0]) == pytest.approx(4 / 30)


def test_wmape_undefined_when_actuals_sum_to_zero():
    with pytest.raises(ValueError):
        wmape([0.0, 0.0], [1.0, 2.0])


def test_forecast_bias_sign():
    # predictions consistently high -> positive bias
    assert forecast_bias([10.0, 10.0], [12.0, 13.0]) == pytest.approx(2.5)


def test_pinball_loss_median_equals_half_mae():
    yt, yp = [10.0, 20.0], [12.0, 16.0]
    mae = np.mean(np.abs(np.array(yt) - np.array(yp)))
    assert pinball_loss(yt, yp, 0.5) == pytest.approx(mae / 2)


def test_pinball_rejects_bad_quantile():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 1.5)


# --- feature transforms ----------------------------------------------------


def _series(n=30):
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-05", periods=n, freq="W"),
            "AveragePrice": np.linspace(1.0, 2.0, n),
        }
    )


def test_to_prophet_frame_renames_columns():
    cfg = DatasetConfig.load("avocado")
    out = to_prophet_frame(cfg, _series())
    assert list(out.columns) == ["ds", "y"]


def test_temporal_split_holds_out_tail():
    frame = to_prophet_frame(DatasetConfig.load("avocado"), _series(30))
    train, test = temporal_train_test_split(frame, horizon=12)
    assert len(train) == 18 and len(test) == 12
    # no leakage: every training timestamp precedes the test window
    assert train["ds"].max() < test["ds"].min()


def test_temporal_split_rejects_too_short_series():
    frame = to_prophet_frame(DatasetConfig.load("avocado"), _series(10))
    with pytest.raises(ValueError):
        temporal_train_test_split(frame, horizon=12)


def test_feature_schema_records_dtype_and_range():
    frame = to_prophet_frame(DatasetConfig.load("avocado"), _series())
    schema = feature_schema(frame)
    assert set(schema) == {"ds", "y"}
    assert schema["y"]["min"] == pytest.approx(1.0)
    assert schema["y"]["max"] == pytest.approx(2.0)


# --- series selection ------------------------------------------------------


def test_select_series_filters_and_aggregates():
    cfg = DatasetConfig.load("avocado")
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-05", "2020-01-05", "2020-01-12"]),
            "AveragePrice": [1.0, 3.0, 2.0],
            "region": ["TotalUS", "California", "TotalUS"],
            "type": ["conventional", "conventional", "conventional"],
        }
    )
    out = select_series(cfg, raw)  # default_series -> region=TotalUS, type=conventional
    assert list(out["AveragePrice"]) == [1.0, 2.0]  # California row dropped


def test_select_series_raises_on_empty_selection():
    cfg = DatasetConfig.load("avocado")
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-05"]),
            "AveragePrice": [1.0],
            "region": ["California"],
            "type": ["organic"],
        }
    )
    with pytest.raises(ValueError):
        select_series(cfg, raw)
