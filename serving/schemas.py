"""Pydantic v2 request/response schemas for the serving API.

These are the API contract — FastAPI renders them into the OpenAPI docs at
``/docs`` and validates every request/response against them. Field constraints
(e.g. ``horizon`` bounds) are enforced here so bad input is rejected at the edge
with a 422 rather than blowing up inside the model.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

MAX_HORIZON = 104  # ~2 years of weekly steps; guards against pathological requests


class ForecastRequest(BaseModel):
    """A single-series forecast request."""

    series_id: str = Field(
        ...,
        description="Series identifier: the series-id columns joined by '|', e.g. 'TotalUS|conventional'.",
        examples=["TotalUS|conventional"],
    )
    horizon: int = Field(
        ..., gt=0, le=MAX_HORIZON, description="Number of future steps to forecast."
    )


class BatchForecastRequest(BaseModel):
    """Forecast several series in one call (shared horizon)."""

    series_ids: list[str] = Field(..., min_length=1, max_length=200)
    horizon: int = Field(..., gt=0, le=MAX_HORIZON)


class ForecastPoint(BaseModel):
    """One forecasted step: point estimate (p50) plus an 80% interval."""

    date: date
    p10: float = Field(..., description="10th-percentile (lower bound of the 80% interval).")
    p50: float = Field(..., description="Point forecast (median).")
    p90: float = Field(..., description="90th-percentile (upper bound of the 80% interval).")


class ForecastResponse(BaseModel):
    series_id: str
    model_version: str
    horizon: int
    generated_at: datetime
    forecast: list[ForecastPoint]


class BatchForecastResponse(BaseModel):
    model_version: str
    horizon: int
    generated_at: datetime
    results: list[ForecastResponse]


class HistoryRecord(BaseModel):
    """A previously logged prediction, optionally joined with the actual."""

    series_id: str
    target_date: date
    p10: float
    p50: float
    p90: float
    model_version: str
    predicted_at: datetime
    actual: float | None = Field(None, description="Observed value if known, else null.")


class HistoryResponse(BaseModel):
    series_id: str
    count: int
    records: list[HistoryRecord]


class LeaderboardEntry(BaseModel):
    model: str
    wmape: float | None = None
    wrmsse: float | None = None
    pinball_p10: float | None = None
    pinball_p50: float | None = None
    pinball_p90: float | None = None
    bias: float | None = None


class LeaderboardResponse(BaseModel):
    dataset: str
    entries: list[LeaderboardEntry]


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_version: str | None = None


__all__ = [
    "ForecastRequest",
    "BatchForecastRequest",
    "ForecastPoint",
    "ForecastResponse",
    "BatchForecastResponse",
    "HistoryRecord",
    "HistoryResponse",
    "LeaderboardEntry",
    "LeaderboardResponse",
    "HealthResponse",
    "MAX_HORIZON",
]
