"""Drift and retraining event log — the audit trail behind the monitoring loop.

The dashboard needs two histories that a point-in-time drift computation cannot
give it: how a signal has behaved over the past weeks, and what happened the
last few times a retrain fired (metrics before, metrics after, whether the
candidate was promoted). Both are append-only facts, so they live in tables
rather than being recomputed.

These tables share the prediction log's database on purpose — same
:func:`serving.store.get_engine`, so ``SERVING_DB_URI`` moves drift events,
retraining events and predictions together between the local SQLite file and the
compose Postgres, and joins across them stay possible. The metric dicts are
stored as JSON text: their keys follow whatever :func:`models.metrics.evaluate`
returns for the dataset, and freezing that into columns would defeat the
dataset-agnostic design.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from serving.store import get_engine as _get_serving_engine

_metadata = MetaData()

drift_events = Table(
    "drift_events",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(64), nullable=False, index=True),
    Column("signal_name", String(128), nullable=False, index=True),
    Column("value", Float, nullable=False),
    Column("status", String(16), nullable=False),
    Column("detail", Text, nullable=False, default=""),
    Column("recorded_at", DateTime, nullable=False),
)

retraining_events = Table(
    "retraining_events",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(64), nullable=False, index=True),
    Column("triggered_by", String(128), nullable=False),
    Column("before_metrics", Text, nullable=False, default="{}"),
    Column("after_metrics", Text, nullable=False, default="{}"),
    Column("promoted", Boolean, nullable=False),
    Column("model_version", String(128), nullable=False, default=""),
    Column("notes", Text, nullable=False, default=""),
    Column("recorded_at", DateTime, nullable=False),
)

_created_for: Engine | None = None


def get_engine() -> Engine:
    """The serving engine, with the monitoring tables ensured on first use.

    Tracks *which* engine the tables were created against, so a test that swaps
    ``SERVING_DB_URI`` and calls :func:`serving.store.reset_engine` transparently
    gets a fresh schema without any extra bookkeeping.
    """
    global _created_for
    engine = _get_serving_engine()
    if _created_for is not engine:
        _metadata.create_all(engine)
        _created_for = engine
    return engine


def reset_tables() -> None:
    """Forget that the tables were created (used by tests that swap the DB URI)."""
    global _created_for
    _created_for = None


def _dumps(payload: dict[str, Any] | None) -> str:
    try:
        return json.dumps(payload or {}, default=float, sort_keys=True)
    except (TypeError, ValueError):
        return "{}"


def _loads(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def record_drift_event(
    dataset: str,
    signal_name: str,
    value: float,
    status: str,
    detail: str = "",
) -> int:
    """Append one evaluated drift signal. Returns the new row id."""
    engine = get_engine()
    row = {
        "dataset": dataset,
        "signal_name": signal_name,
        "value": float(value),
        "status": status,
        "detail": detail,
        "recorded_at": datetime.now(UTC),
    }
    with engine.begin() as conn:
        result = conn.execute(insert(drift_events), row)
    key = result.inserted_primary_key
    return int(key[0]) if key else 0


def fetch_drift_events(dataset: str | None = None, limit: int = 100) -> list[dict]:
    """Most-recent drift events (newest first), optionally for one dataset."""
    engine = get_engine()
    stmt = select(drift_events)
    if dataset is not None:
        stmt = stmt.where(drift_events.c.dataset == dataset)
    stmt = stmt.order_by(drift_events.c.recorded_at.desc(), drift_events.c.id.desc()).limit(limit)
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt)]


def record_retraining_event(
    dataset: str,
    triggered_by: str,
    before_metrics: dict,
    after_metrics: dict,
    promoted: bool,
    model_version: str = "",
    notes: str = "",
) -> int:
    """Append one retraining outcome. Returns the new row id."""
    engine = get_engine()
    row = {
        "dataset": dataset,
        "triggered_by": triggered_by,
        "before_metrics": _dumps(before_metrics),
        "after_metrics": _dumps(after_metrics),
        "promoted": bool(promoted),
        "model_version": model_version,
        "notes": notes,
        "recorded_at": datetime.now(UTC),
    }
    with engine.begin() as conn:
        result = conn.execute(insert(retraining_events), row)
    key = result.inserted_primary_key
    return int(key[0]) if key else 0


def fetch_retraining_events(dataset: str | None = None, limit: int = 5) -> list[dict]:
    """Most-recent retraining events (newest first), metric JSON decoded to dicts."""
    engine = get_engine()
    stmt = select(retraining_events)
    if dataset is not None:
        stmt = stmt.where(retraining_events.c.dataset == dataset)
    stmt = stmt.order_by(
        retraining_events.c.recorded_at.desc(), retraining_events.c.id.desc()
    ).limit(limit)
    with engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(stmt)]
    for row in rows:
        row["before_metrics"] = _loads(row.get("before_metrics"))
        row["after_metrics"] = _loads(row.get("after_metrics"))
        row["promoted"] = bool(row.get("promoted"))
    return rows


__all__ = [
    "drift_events",
    "fetch_drift_events",
    "fetch_retraining_events",
    "get_engine",
    "record_drift_event",
    "record_retraining_event",
    "reset_tables",
    "retraining_events",
]
