"""Phase 2 integration test: the LightGBM global model trains and beats a floor.

Skips when lightgbm (optional ``[models]`` group) or the avocado data are
missing, keeping the core CI job light.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lightgbm", reason="lightgbm not installed ([models] extra)")

from config import DatasetConfig  # noqa: E402
from data.loader import RAW_DIR, load_raw  # noqa: E402


@pytest.mark.integration
def test_lightgbm_panel_forecast_is_sane():
    cfg = DatasetConfig.load("avocado")
    if not (RAW_DIR / cfg.raw_filename).exists():
        pytest.skip("avocado raw data not present (run `dvc pull`)")

    from models.evaluate import QUANTILES, aggregate
    from models.lightgbm_model import train_and_forecast

    raw = load_raw(cfg)
    forecasts, models, spec, info = train_and_forecast(cfg, raw)

    assert len(forecasts) == raw.groupby(cfg.series_id_cols).ngroups
    # every series forecast covers the full horizon for all three quantiles
    for f in forecasts:
        assert len(f.point) == cfg.horizon
        assert set(f.quantiles) == set(QUANTILES)

    metrics = aggregate(forecasts)
    assert 0.0 <= metrics["wmape"] < 0.3  # global feature model should be well under 30%
    assert metrics["wrmsse"] > 0.0
