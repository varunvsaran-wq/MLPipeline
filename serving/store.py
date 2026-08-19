"""Prediction log — every served forecast is persisted for later analysis.

Logging predictions is the backbone of the whole MLOps loop: ``/forecast/history``
reads from here, and Phase 5's residual/rolling-WMAPE drift signals are computed
by joining these rows against actuals as they arrive. Defaults to a local SQLite
file so the API runs with zero external dependencies; point ``SERVING_DB_URI`` at
the compose Postgres to share the tracking stack's database.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.engine import Engine

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DB = f"sqlite:///{(_PROJECT_ROOT / 'serving' / 'predictions.db').as_posix()}"

_metadata = MetaData()

predictions = Table(
    "predictions",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(64), nullable=False),
    Column("series_id", String(256), nullable=False, index=True),
    Column("target_date", Date, nullable=False),
    Column("horizon_step", Integer, nullable=False),
    Column("p10", Float, nullable=False),
    Column("p50", Float, nullable=False),
    Column("p90", Float, nullable=False),
    Column("model_version", String(128), nullable=False),
    Column("predicted_at", DateTime, nullable=False),
)

_engine: Engine | None = None


def get_engine() -> Engine:
    """Process-wide engine, created lazily against ``SERVING_DB_URI`` or SQLite."""
    global _engine
    if _engine is None:
        uri = os.environ.get("SERVING_DB_URI", _DEFAULT_DB)
        _engine = create_engine(uri, future=True)
        _metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Drop the cached engine (used by tests that swap the DB URI)."""
    global _engine
    _engine = None


def log_predictions(
    dataset: str,
    series_id: str,
    model_version: str,
    rows: list,
    predicted_at: datetime | None = None,
) -> int:
    """Persist a series' forecast rows. ``rows`` are objects with date/p10/p50/p90."""
    ts = predicted_at or datetime.now(UTC)
    payload = [
        {
            "dataset": dataset,
            "series_id": series_id,
            "target_date": r.date,
            "horizon_step": i + 1,
            "p10": float(r.p10),
            "p50": float(r.p50),
            "p90": float(r.p90),
            "model_version": model_version,
            "predicted_at": ts,
        }
        for i, r in enumerate(rows)
    ]
    if not payload:
        return 0
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(insert(predictions), payload)
    return len(payload)


def fetch_history(series_id: str, limit: int = 100) -> list[dict]:
    """Most-recent logged predictions for a series (newest first)."""
    engine = get_engine()
    stmt = (
        select(predictions)
        .where(predictions.c.series_id == series_id)
        .order_by(predictions.c.predicted_at.desc(), predictions.c.target_date.desc())
        .limit(limit)
    )
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt)]


__all__ = ["get_engine", "reset_engine", "log_predictions", "fetch_history", "predictions"]
