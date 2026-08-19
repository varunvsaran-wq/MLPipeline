"""FastAPI serving layer (Phase 3).

Endpoints (auto-documented at ``/docs``):

* ``POST /forecast``          — one series: point forecast + 80% interval
* ``POST /forecast/batch``    — many series in one call
* ``GET  /forecast/history``  — past logged predictions vs actuals for a series
* ``GET  /model/leaderboard`` — models ranked by validation WMAPE
* ``GET  /model/series``      — the series ids this model can forecast
* ``GET  /health``            — liveness probe (never needs the model loaded)
* ``GET  /metrics``           — Prometheus scrape

The predictor (and its bundle) is loaded lazily on first use so the container
starts — and ``/health`` returns 200 — even before a bundle is baked/mounted.
Every served forecast is written to the prediction store, which is what later
feeds drift monitoring.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from serving import store
from serving.leaderboard import load_leaderboard
from serving.predictor import Predictor, SeriesNotFoundError
from serving.schemas import (
    BatchForecastRequest,
    BatchForecastResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HistoryRecord,
    HistoryResponse,
    LeaderboardResponse,
)
from serving.shadow import get_router

DATASET = os.environ.get("SERVING_DATASET", "avocado")

app = FastAPI(
    title="Demand Forecasting API",
    version="0.3.0",
    description="Phase 3 serving layer: quantile forecasts from the global LightGBM model.",
)

# --- Prometheus metrics ----------------------------------------------------
REQUESTS = Counter("forecast_requests_total", "Forecast requests", ["endpoint", "status"])
LATENCY = Histogram("forecast_latency_seconds", "Forecast handler latency", ["endpoint"])
FORECASTS = Counter("forecast_series_total", "Number of series forecasted (rows served)")

# --- lazy predictor --------------------------------------------------------
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Load the bundle on first use; 503 if none has been built/baked."""
    global _predictor
    if _predictor is None:
        from serving.model_bundle import load_bundle

        try:
            _predictor = Predictor(load_bundle(DATASET))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _predictor


def _forecast_one(predictor: Predictor, series_id: str, horizon: int) -> ForecastResponse:
    try:
        rows = predictor.forecast(series_id, horizon)
    except SeriesNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown series_id: {series_id}") from exc
    store.log_predictions(DATASET, series_id, predictor.model_version, rows)
    # A/B shadow evaluation (Phase 6): on a sampled fraction of traffic the
    # challenger also predicts and both forecasts are logged for comparison. The
    # call is internally fail-safe and returns nothing — the shadow's output is
    # structurally incapable of reaching the caller.
    get_router().maybe_shadow(series_id, horizon, rows, predictor.model_version)
    FORECASTS.inc()
    return ForecastResponse(
        series_id=series_id,
        model_version=predictor.model_version,
        horizon=horizon,
        generated_at=datetime.now(UTC),
        forecast=[ForecastPoint(date=r.date, p10=r.p10, p50=r.p50, p90=r.p90) for r in rows],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _predictor is not None
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_version=_predictor.model_version if loaded else None,
    )


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    start = time.perf_counter()
    predictor = get_predictor()
    try:
        resp = _forecast_one(predictor, req.series_id, req.horizon)
        REQUESTS.labels("forecast", "ok").inc()
        return resp
    except HTTPException as exc:
        REQUESTS.labels("forecast", str(exc.status_code)).inc()
        raise
    finally:
        LATENCY.labels("forecast").observe(time.perf_counter() - start)


@app.post("/forecast/batch", response_model=BatchForecastResponse)
def forecast_batch(req: BatchForecastRequest) -> BatchForecastResponse:
    start = time.perf_counter()
    predictor = get_predictor()
    try:
        results = [_forecast_one(predictor, sid, req.horizon) for sid in req.series_ids]
        REQUESTS.labels("forecast_batch", "ok").inc()
        return BatchForecastResponse(
            model_version=predictor.model_version,
            horizon=req.horizon,
            generated_at=datetime.now(UTC),
            results=results,
        )
    except HTTPException as exc:
        REQUESTS.labels("forecast_batch", str(exc.status_code)).inc()
        raise
    finally:
        LATENCY.labels("forecast_batch").observe(time.perf_counter() - start)


@app.get("/forecast/history", response_model=HistoryResponse)
def forecast_history(
    series_id: str = Query(..., description="Series id to fetch logged predictions for."),
    limit: int = Query(100, gt=0, le=1000),
) -> HistoryResponse:
    predictor = get_predictor()
    actuals = predictor.actuals(series_id)
    rows = store.fetch_history(series_id, limit=limit)
    records = [
        HistoryRecord(
            series_id=r["series_id"],
            target_date=r["target_date"],
            p10=r["p10"],
            p50=r["p50"],
            p90=r["p90"],
            model_version=r["model_version"],
            predicted_at=r["predicted_at"],
            actual=actuals.get(r["target_date"]),
        )
        for r in rows
    ]
    REQUESTS.labels("history", "ok").inc()
    return HistoryResponse(series_id=series_id, count=len(records), records=records)


@app.get("/model/leaderboard", response_model=LeaderboardResponse)
def model_leaderboard() -> LeaderboardResponse:
    # Use the served bundle's metrics as a fallback if the comparison hasn't run.
    fallback = _predictor.bundle.metrics if _predictor is not None else None
    data = load_leaderboard(DATASET, fallback_metrics=fallback)
    REQUESTS.labels("leaderboard", "ok").inc()
    return LeaderboardResponse(**data)


@app.get("/model/series")
def model_series() -> dict:
    predictor = get_predictor()
    ids = predictor.series_ids()
    return {"dataset": DATASET, "count": len(ids), "series_ids": ids}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


__all__ = ["app", "get_predictor", "DATASET"]
