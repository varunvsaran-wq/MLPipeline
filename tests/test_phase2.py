"""Phase 2 unit tests: feature suite correctness + scaled-error metrics.

No heavy model deps here — the LightGBM end-to-end run is a separate guarded
integration test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DatasetConfig, FeatureConfig, FourierTerm
from features.pipeline import build_feature_matrix, panel_train_test_split
from models.evaluate import SeriesForecast, aggregate
from models.metrics import rmsse, wrmsse


def _panel(n_per_series=60):
    """Two synthetic weekly series with a known structure."""
    frames = []
    for region, base in (("A", 10.0), ("B", 100.0)):
        dates = pd.date_range("2019-01-06", periods=n_per_series, freq="W")
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "AveragePrice": base + np.arange(n_per_series, dtype=float),
                    "Total Volume": np.linspace(1000, 2000, n_per_series),
                    "region": region,
                    "type": "conventional",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _cfg():
    cfg = DatasetConfig.load("avocado")
    # smaller, deterministic feature set for the synthetic series
    cfg.features = FeatureConfig(
        lags=[1, 2],
        rolling_windows=[3],
        fourier=[FourierTerm(period=52.143, order=2)],
        exogenous_lag=1,
        use_exogenous=True,
    )
    return cfg


# --- feature suite ---------------------------------------------------------


def test_lag1_equals_previous_target_within_series():
    cfg = _cfg()
    frame, spec = build_feature_matrix(cfg, _panel())
    a = frame[frame["region"] == "A"].sort_values("ds").reset_index(drop=True)
    # lag_1 at row i must equal y at row i-1 (and be NaN at the first row)
    assert np.isnan(a.loc[0, "lag_1"])
    assert a.loc[5, "lag_1"] == pytest.approx(a.loc[4, "y"])


def test_no_cross_series_leakage_at_series_boundary():
    cfg = _cfg()
    frame, _ = build_feature_matrix(cfg, _panel())
    # first row of each series has no prior within that series -> lag_1 is NaN,
    # never the last value of the other series. (Use the literal first row, not
    # groupby().first(), which skips NaN.)
    first_rows = frame.sort_values(["series_id", "ds"]).drop_duplicates("series_id", keep="first")
    assert first_rows["lag_1"].isna().all()


def test_expected_feature_columns_present():
    cfg = _cfg()
    _, spec = build_feature_matrix(cfg, _panel())
    for col in [
        "lag_1",
        "lag_2",
        "roll_mean_3",
        "roll_std_3",
        "month",
        "is_holiday",
        "fourier0_sin1",
        "fourier0_cos2",
    ]:
        assert col in spec.numeric, col
    assert spec.categorical == ["region", "type"]


def test_panel_split_holds_out_last_horizon_per_series():
    cfg = _cfg()
    frame, _ = build_feature_matrix(cfg, _panel(40))
    train, test = panel_train_test_split(frame, horizon=4)
    # 2 series x 4 held-out steps
    assert test["series_id"].nunique() == 2
    assert (test.groupby("series_id").size() == 4).all()
    assert train["ds"].max() < test["ds"].min()


# --- scaled-error metrics --------------------------------------------------


def test_rmsse_zero_for_perfect_forecast():
    hist = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmsse([5.0, 6.0], [5.0, 6.0], hist) == 0.0


def test_rmsse_flat_history_is_nan():
    # zero scale -> undefined, signalled as NaN so WRMSSE can skip it
    assert np.isnan(rmsse([1.0], [2.0], [3.0, 3.0, 3.0]))


def test_wrmsse_weights_dominant_series():
    # series with weight 100 has error, weight-1 series is perfect ->
    # weighted score is pulled toward the heavy series.
    hist = [1.0, 2.0, 3.0, 4.0]
    heavy = ([10.0, 10.0], [12.0, 12.0], hist, 100.0)
    light = ([10.0, 10.0], [10.0, 10.0], hist, 1.0)
    score = wrmsse([heavy, light])
    assert score == pytest.approx(rmsse(*heavy[:3]) * 100 / 101, rel=1e-6)


def test_aggregate_includes_all_metric_keys():
    fc = [
        SeriesForecast(
            series_id="A",
            y_true=np.array([10.0, 11.0]),
            point=np.array([10.0, 11.0]),
            quantiles={
                0.1: np.array([9.0, 10.0]),
                0.5: np.array([10.0, 11.0]),
                0.9: np.array([11.0, 12.0]),
            },
            train_history=np.array([1.0, 2.0, 3.0, 4.0]),
            weight=5.0,
        )
    ]
    m = aggregate(fc)
    for key in ["wmape", "wrmsse", "bias", "pinball_p10", "pinball_p50", "pinball_p90"]:
        assert key in m
