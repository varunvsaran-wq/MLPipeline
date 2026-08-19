"""Phase 4 integration test: the TFT trains and lands in the shared harness.

Skips when torch / pytorch-forecasting (the optional ``[deep]`` group) or the
avocado data are missing, so the core CI job stays light and green without a
multi-gigabyte install. Epoch count and hidden size are pinned tiny — this
checks the *contract* (shapes, quantile keys, non-crossing quantiles, WRMSSE
weights comparable to the other families), not forecast quality.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="torch not installed ([deep] extra)")
pytest.importorskip(
    "pytorch_forecasting", reason="pytorch-forecasting not installed ([deep] extra)"
)

from config import DatasetConfig  # noqa: E402
from data.loader import RAW_DIR, load_raw  # noqa: E402


@pytest.mark.integration
def test_tft_panel_forecast_matches_harness_contract():
    cfg = DatasetConfig.load("avocado")
    if not (RAW_DIR / cfg.raw_filename).exists():
        pytest.skip("avocado raw data not present (run `dvc pull`)")

    from models.evaluate import QUANTILES, aggregate
    from models.tft_model import train_and_forecast

    raw = load_raw(cfg)
    forecasts, model, info = train_and_forecast(
        cfg,
        raw,
        max_epochs=1,
        hidden_size=8,
        attention_head_size=1,
        hidden_continuous_size=4,
        batch_size=32,
    )

    assert forecasts, "TFT produced no forecasts"
    assert len(forecasts) == raw.groupby(cfg.series_id_cols).ngroups
    assert info["max_prediction_length"] == cfg.horizon
    assert model is not None

    for f in forecasts:
        assert set(f.quantiles) == set(QUANTILES)
        assert len(f.point) == cfg.horizon
        assert len(f.y_true) == len(f.point)
        for q in QUANTILES:
            assert f.quantiles[q].shape == f.point.shape
            assert np.isfinite(f.quantiles[q]).all()
        # p50 is the point forecast, and quantiles must not cross.
        np.testing.assert_allclose(f.point, f.quantiles[0.5])
        assert (f.quantiles[0.1] <= f.quantiles[0.5]).all()
        assert (f.quantiles[0.5] <= f.quantiles[0.9]).all()
        # Enough history to scale RMSSE, and a positive WRMSSE weight.
        assert len(f.train_history) > cfg.horizon
        assert f.weight > 0.0

    metrics = aggregate(forecasts)
    assert np.isfinite(metrics["wmape"]) and metrics["wmape"] >= 0.0
    assert metrics["wrmsse"] > 0.0
    assert metrics["n_series"] == float(len(forecasts))


@pytest.mark.integration
def test_tft_panel_has_contiguous_time_idx_and_leakage_free_split():
    """``build_panel`` needs no torch, but is only meaningful with the extra."""
    cfg = DatasetConfig.load("avocado")
    if not (RAW_DIR / cfg.raw_filename).exists():
        pytest.skip("avocado raw data not present (run `dvc pull`)")

    from models.tft_model import build_panel

    raw = load_raw(cfg)
    panel, known_reals, unknown_reals = build_panel(cfg, raw)

    assert "time_idx" in panel.columns
    assert panel["time_idx"].min() == 0
    # One global clock: the number of distinct time_idx equals distinct dates.
    assert panel["time_idx"].nunique() == panel[cfg.date_col].nunique()
    # No duplicate (series, step) rows — TimeSeriesDataSet rejects them.
    assert not panel.duplicated(subset=[*cfg.series_id_cols, "time_idx"]).any()
    # Calendar reals are known over the horizon; target/exogenous are not.
    assert cfg.target_col in unknown_reals
    assert cfg.target_col not in known_reals
    for col in cfg.exogenous_cols:
        assert col in unknown_reals
