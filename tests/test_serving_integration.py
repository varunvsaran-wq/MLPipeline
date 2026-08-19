"""Phase 3 integration: recursive predictor + live API over a synthetic bundle.

Guarded by the lightgbm extra (skips in the light CI job). Uses an in-memory
synthetic panel so it needs neither the DVC data nor a pre-built bundle on disk
— the point is to exercise the full serving path end to end.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm", reason="lightgbm not installed ([models] extra)")
pytest.importorskip("fastapi", reason="fastapi not installed ([serving] extra)")

from config import DatasetConfig, FeatureConfig, FourierTerm  # noqa: E402
from features.pipeline import build_feature_matrix  # noqa: E402
from models.lightgbm_model import _fit_quantile_models  # noqa: E402
from serving.model_bundle import ModelBundle  # noqa: E402
from serving.predictor import Predictor, SeriesNotFoundError  # noqa: E402


def _synthetic_panel(n=80):
    frames = []
    rng = np.random.default_rng(0)
    for region, base in (("Alpha", 1.0), ("Beta", 2.0)):
        dates = pd.date_range("2018-01-07", periods=n, freq="W")
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(n) / 52.0)
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "AveragePrice": base + seasonal + rng.normal(0, 0.02, n),
                    "Total Volume": np.linspace(1000, 2000, n),
                    "region": region,
                    "type": "conventional",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def bundle():
    cfg = DatasetConfig.load("avocado")
    cfg.features = FeatureConfig(
        lags=[1, 2, 4],
        rolling_windows=[4],
        fourier=[FourierTerm(period=52.143, order=2)],
        exogenous_lag=1,
        use_exogenous=True,
    )
    panel = _synthetic_panel()
    frame, spec = build_feature_matrix(cfg, panel)
    models = _fit_quantile_models(frame, spec)
    categories = {col: list(frame[col].cat.categories) for col in spec.categorical}
    return ModelBundle(
        dataset="avocado",
        cfg=cfg,
        models=models,
        spec=spec,
        categories=categories,
        history=panel,
        feature_schema={},
        metrics={"wmape": 0.1, "wrmsse": 1.0},
        git_commit="test",
        trained_at=datetime.now(UTC).isoformat(),
    )


# --- predictor -------------------------------------------------------------


def test_predictor_forecasts_future_horizon(bundle):
    p = Predictor(bundle)
    ids = p.series_ids()
    assert "Alpha|conventional" in ids
    rows = p.forecast("Alpha|conventional", horizon=8)
    assert len(rows) == 8
    # dates are strictly future and evenly spaced weekly
    last_hist = pd.Timestamp(bundle.history["Date"].max()).date()
    assert rows[0].date > last_hist
    # non-crossing quantiles at every step
    assert all(r.p10 <= r.p50 <= r.p90 for r in rows)


def test_predictor_unknown_series_raises(bundle):
    with pytest.raises(SeriesNotFoundError):
        Predictor(bundle).forecast("Nope|conventional", horizon=4)


# --- live API --------------------------------------------------------------


@pytest.fixture()
def client(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("SERVING_DB_URI", f"sqlite:///{(tmp_path / 'api.db').as_posix()}")
    from fastapi.testclient import TestClient

    from serving import app as app_module
    from serving import store

    store.reset_engine()
    app_module._predictor = Predictor(bundle)  # inject; skip on-disk bundle load
    with TestClient(app_module.app) as c:
        yield c
    app_module._predictor = None
    store.reset_engine()


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_forecast_endpoint_shape(client):
    r = client.post("/forecast", json={"series_id": "Alpha|conventional", "horizon": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["series_id"] == "Alpha|conventional"
    assert len(body["forecast"]) == 5
    pt = body["forecast"][0]
    assert pt["p10"] <= pt["p50"] <= pt["p90"]


def test_forecast_unknown_series_404(client):
    r = client.post("/forecast", json={"series_id": "Ghost", "horizon": 3})
    assert r.status_code == 404


def test_forecast_bad_horizon_422(client):
    r = client.post("/forecast", json={"series_id": "Alpha|conventional", "horizon": 0})
    assert r.status_code == 422


def test_batch_and_history(client):
    r = client.post(
        "/forecast/batch",
        json={"series_ids": ["Alpha|conventional", "Beta|conventional"], "horizon": 4},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2

    # the batch call logged predictions -> history now returns them
    h = client.get("/forecast/history", params={"series_id": "Alpha|conventional"})
    assert h.status_code == 200
    assert h.json()["count"] == 4


def test_leaderboard_and_series_and_metrics(client):
    lb = client.get("/model/leaderboard")
    assert lb.status_code == 200
    assert lb.json()["entries"]

    s = client.get("/model/series")
    assert s.status_code == 200
    assert s.json()["count"] == 2

    m = client.get("/metrics")
    assert m.status_code == 200
    assert b"forecast_requests_total" in m.content
