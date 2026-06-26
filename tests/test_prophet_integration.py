"""Phase 1 integration test: the Prophet baseline trains and logs a tracked run.

Skips when prophet (optional ``[models]`` group) or the avocado data aren't
available, so the core CI job stays green without the heavy model stack.
"""

from __future__ import annotations

import pytest

pytest.importorskip("prophet", reason="prophet not installed ([models] extra)")

from config import DatasetConfig  # noqa: E402
from data.loader import RAW_DIR  # noqa: E402


@pytest.mark.integration
def test_prophet_baseline_produces_tracked_run(tmp_path, monkeypatch):
    cfg = DatasetConfig.load("avocado")
    if not (RAW_DIR / cfg.raw_filename).exists():
        pytest.skip("avocado raw data not present (run `dvc pull`)")

    # Isolate the tracking store so the test doesn't pollute the real mlruns/.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())

    from models.prophet_baseline import train

    metrics = train("avocado")
    assert 0.0 <= metrics["wmape"] < 1.0  # a sane baseline beats 100% error
    assert "bias" in metrics
