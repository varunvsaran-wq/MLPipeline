"""Data layer for the ops dashboard — every panel's numbers, computed here.

Streamlit apps are notoriously untestable: the rendering calls only make sense
inside a script run, so any logic that lives next to ``st.metric`` is logic that
CI can never check. The dashboard is therefore split in two, and this is the
half that matters. Each function below is pure in the sense that counts,
histograms and drift tables are derived from *arguments* — a list of prediction
rows, a reference/current pair of frames — rather than from a live database, so
the whole panel set can be exercised with synthetic inputs in milliseconds.
``dashboard/app.py`` is left as a rendering shell that fetches, calls, and draws.

Two other design constraints shaped this module:

* **The monitoring package is optional at import time.** ``monitoring.drift``,
  ``monitoring.store`` and ``models.registry`` are consumed through guarded,
  function-local imports. If they are missing (a lean container, or a partially
  built checkout) the dashboard still loads and the affected panel degrades to an
  empty table or an ``"unknown"`` status instead of taking the page down with an
  ``ImportError``.
* **Dataset-agnostic.** Column names come from :class:`config.DatasetConfig`;
  the served dataset name comes from ``SERVING_DATASET``, matching
  ``serving/app.py`` so the dashboard always describes what the API is serving.

Timestamps are normalised to UTC on the way in: SQLite hands back naive
datetimes while Postgres hands back aware ones, and the 24h/7d windows must not
depend on which backend is configured.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATASET = "avocado"

#: Fallback PSI thresholds, used only for display when ``monitoring.drift`` is
#: unavailable. The live values always come from the monitoring module.
PSI_YELLOW = 0.1
PSI_RED = 0.2

STATUS_ORDER = {"red": 0, "yellow": 1, "green": 2, "unknown": 3}

_PREDICTION_COLUMNS = [
    "id",
    "dataset",
    "series_id",
    "target_date",
    "horizon_step",
    "p10",
    "p50",
    "p90",
    "model_version",
    "predicted_at",
]


# --- small shared helpers --------------------------------------------------


def served_dataset() -> str:
    """Dataset the API is serving, from ``SERVING_DATASET`` (default avocado)."""
    return os.environ.get("SERVING_DATASET", DEFAULT_DATASET)


def _as_utc(value: Any) -> datetime | None:
    """Coerce a stored timestamp to an aware UTC datetime (or ``None``)."""
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts is pd.NaT:
        return None
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    return ts.to_pydatetime()


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return None if ts is pd.NaT else ts.date()


def _now(now: datetime | None = None) -> datetime:
    return now.astimezone(UTC) if now is not None else datetime.now(UTC)


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    """An empty frame with the promised columns, so callers can render blindly."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})


# --- panel 1: current production model -------------------------------------


def _registry_info(dataset: str) -> dict[str, Any] | None:
    """Best-effort lookup of the registered Production version (guarded).

    ``models.registry`` is built alongside this module and its exact surface is
    not fixed yet, so we probe the plausible entry points and give up quietly.
    """
    try:
        from models import registry  # type: ignore[attr-defined]
    except Exception:
        return None
    for attr in ("production_version", "get_production_version", "current_production"):
        fn = getattr(registry, attr, None)
        if callable(fn):
            try:
                info = fn(dataset)
            except Exception:
                return None
            if info is None:
                return None
            return dict(info) if isinstance(info, Mapping) else {"version": str(info)}
    return None


def production_model_info(dataset: str | None = None, bundle: Any = None) -> dict[str, Any]:
    """Version, training date and validation metrics of the served model.

    ``bundle`` may be injected (a :class:`serving.model_bundle.ModelBundle` or any
    object exposing the same attributes) which is what makes this testable; when
    omitted the served bundle is loaded from disk. A missing bundle is reported
    as ``available=False`` rather than raised, because the dashboard should still
    render its other panels when no model has been built yet.
    """
    ds = dataset or served_dataset()
    if bundle is None:
        try:
            from serving.model_bundle import load_bundle

            bundle = load_bundle(ds)
        except Exception as exc:  # missing artifact, or joblib/lightgbm absent
            return {
                "dataset": ds,
                "available": False,
                "model_version": "unknown",
                "trained_at": None,
                "metrics": {},
                "series_count": 0,
                "feature_count": 0,
                "registry": None,
                "error": str(exc),
            }

    try:
        series_count = len(bundle.series_ids())
    except Exception:
        series_count = 0
    spec = getattr(bundle, "spec", None)
    feature_count = len(getattr(spec, "all", []) or []) if spec is not None else 0

    return {
        "dataset": getattr(bundle, "dataset", ds),
        "available": True,
        "model_version": getattr(bundle, "model_version", "unknown"),
        "trained_at": getattr(bundle, "trained_at", None),
        "metrics": dict(getattr(bundle, "metrics", {}) or {}),
        "series_count": series_count,
        "feature_count": feature_count,
        "registry": _registry_info(ds),
        "error": None,
    }


def metrics_table(metrics: Mapping[str, Any]) -> pd.DataFrame:
    """Validation metrics as a two-column frame, ordered for display."""
    preferred = ["wmape", "wrmsse", "smape", "mae", "rmse", "mape", "pinball", "coverage"]
    keys = [k for k in preferred if k in metrics]
    keys += sorted(k for k in metrics if k not in keys)
    rows = [{"metric": k, "value": metrics[k]} for k in keys]
    return pd.DataFrame(rows, columns=["metric", "value"])


# --- prediction log access (the only DB-touching functions) ----------------


def fetch_prediction_rows(
    dataset: str | None = None,
    since: datetime | None = None,
    limit: int = 50_000,
) -> list[dict]:
    """Recent rows from the prediction log, newest first.

    ``serving.store.fetch_history`` is per-series; the dashboard needs a
    dataset-wide slice, so the SELECT is built here against the shared table
    object rather than by widening the serving module's API.
    """
    try:
        from sqlalchemy import select

        from serving.store import get_engine, predictions
    except Exception:
        return []

    stmt = select(predictions)
    if dataset:
        stmt = stmt.where(predictions.c.dataset == dataset)
    if since is not None:
        stmt = stmt.where(predictions.c.predicted_at >= since.replace(tzinfo=None))
    stmt = stmt.order_by(predictions.c.predicted_at.desc(), predictions.c.id.desc()).limit(limit)

    try:
        engine = get_engine()
        with engine.connect() as conn:
            return [dict(row._mapping) for row in conn.execute(stmt)]
    except Exception:
        return []


def predictions_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Normalise prediction-log rows into a typed frame with UTC timestamps."""
    records = [dict(r) for r in rows]
    if not records:
        return _empty(_PREDICTION_COLUMNS)
    frame = pd.DataFrame(records)
    for col in _PREDICTION_COLUMNS:
        if col not in frame.columns:
            frame[col] = np.nan
    frame["predicted_at"] = [_as_utc(v) for v in frame["predicted_at"]]
    frame["target_date"] = [_as_date(v) for v in frame["target_date"]]
    for col in ("p10", "p50", "p90"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


# --- panel 2: live request volume ------------------------------------------


def _request_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per logical API request (a series + a single ``predicted_at``)."""
    return frame[["series_id", "predicted_at"]].drop_duplicates()


def request_volume(
    rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Prediction and request counts over the last 24 hours and 7 days.

    A single ``/forecast`` call logs one row per horizon step, so raw row counts
    overstate traffic. Both are reported: ``predictions_*`` counts rows,
    ``requests_*`` counts distinct (series, timestamp) pairs.
    """
    frame = rows if isinstance(rows, pd.DataFrame) else predictions_frame(rows)
    ref = _now(now)
    out: dict[str, Any] = {
        "as_of": ref,
        "predictions_24h": 0,
        "predictions_7d": 0,
        "requests_24h": 0,
        "requests_7d": 0,
        "series_24h": 0,
        "total_predictions": int(len(frame)),
        "last_prediction_at": None,
    }
    if frame.empty:
        return out

    stamps = pd.Series(list(frame["predicted_at"]))
    valid = frame[stamps.notna().to_numpy()]
    if valid.empty:
        return out

    ts = pd.to_datetime(pd.Series(list(valid["predicted_at"])), utc=True)
    for label, delta in (("24h", timedelta(hours=24)), ("7d", timedelta(days=7))):
        window = valid[(ts >= pd.Timestamp(ref - delta)).to_numpy()]
        out[f"predictions_{label}"] = int(len(window))
        out[f"requests_{label}"] = int(len(_request_keys(window)))
        if label == "24h":
            out["series_24h"] = int(window["series_id"].nunique())
    out["last_prediction_at"] = _as_utc(ts.max())
    return out


def hourly_volume(
    rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    now: datetime | None = None,
    hours: int = 24,
) -> pd.DataFrame:
    """Per-hour request/prediction counts over the trailing ``hours`` window.

    Empty hours are zero-filled so the bar chart shows a continuous timeline
    rather than silently collapsing quiet periods.
    """
    frame = rows if isinstance(rows, pd.DataFrame) else predictions_frame(rows)
    ref = _now(now).replace(minute=0, second=0, microsecond=0)
    index = pd.date_range(end=pd.Timestamp(ref), periods=max(hours, 1), freq="h", tz=UTC)
    out = pd.DataFrame({"hour": index, "predictions": 0, "requests": 0})
    if frame.empty:
        return out

    stamps = pd.to_datetime(pd.Series(list(frame["predicted_at"])), utc=True, errors="coerce")
    valid = frame[stamps.notna().to_numpy()].copy()
    if valid.empty:
        return out
    valid["hour"] = stamps.dropna().dt.floor("h").to_numpy()

    preds = valid.groupby("hour").size()
    reqs = valid[["series_id", "predicted_at", "hour"]].drop_duplicates().groupby("hour").size()
    out["predictions"] = [int(preds.get(h, 0)) for h in index]
    out["requests"] = [int(reqs.get(h, 0)) for h in index]
    return out


# --- panel 3: prediction distribution --------------------------------------


def prediction_histogram(
    rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    bins: int = 20,
    value_col: str = "p50",
) -> pd.DataFrame:
    """Histogram of logged point forecasts: ``bin_start/bin_end/center/count``.

    Binning happens here (not in the chart) so the shape of the served
    distribution is something tests can assert on.
    """
    frame = rows if isinstance(rows, pd.DataFrame) else predictions_frame(rows)
    columns = ["bin_start", "bin_end", "center", "count"]
    if frame.empty or value_col not in frame.columns:
        return _empty(columns)
    values = pd.to_numeric(pd.Series(list(frame[value_col])), errors="coerce").dropna()
    if values.empty:
        return _empty(columns)

    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:  # degenerate: a single distinct value
        hi = lo + 1.0
    counts, edges = np.histogram(values.to_numpy(), bins=max(int(bins), 1), range=(lo, hi))
    return pd.DataFrame(
        {
            "bin_start": edges[:-1],
            "bin_end": edges[1:],
            "center": (edges[:-1] + edges[1:]) / 2.0,
            "count": counts.astype(int),
        }
    )


# --- panel 4: feature drift -------------------------------------------------


def monitoring_available() -> bool:
    """Whether ``monitoring.drift`` can be imported (drives the 'unknown' state)."""
    try:
        import monitoring.drift  # noqa: F401
    except Exception:
        return False
    return True


def reference_current_split(
    frame: pd.DataFrame,
    date_col: str = "ds",
    split_frac: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature matrix into a reference and a current window by date.

    Drift is only meaningful against a baseline; with no separate production
    feed, the served history's own past is the honest baseline. The cutoff is a
    date quantile (not a row quantile) so every series is split at the same
    instant and the comparison stays like-for-like.
    """
    if frame.empty or date_col not in frame.columns:
        return frame, frame
    dates = pd.to_datetime(pd.Series(list(frame[date_col])), errors="coerce")
    unique = np.sort(dates.dropna().unique())
    if len(unique) < 2:
        return frame, frame
    idx = min(max(int(len(unique) * split_frac), 1), len(unique) - 1)
    cutoff = unique[idx]
    mask = (dates < cutoff).to_numpy()
    return frame[mask], frame[~mask]


def signals_table(signals: Sequence[Any]) -> pd.DataFrame:
    """Render a list of ``DriftSignal`` objects as a sortable table."""
    columns = ["name", "value", "status", "threshold", "detail"]
    if not signals:
        return _empty(columns)
    rows = [
        {
            "name": getattr(s, "name", ""),
            "value": float(getattr(s, "value", float("nan"))),
            "status": getattr(s, "status", "unknown"),
            "threshold": float(getattr(s, "threshold", float("nan"))),
            "detail": getattr(s, "detail", ""),
        }
        for s in signals
    ]
    table = pd.DataFrame(rows, columns=columns)
    table["_order"] = [STATUS_ORDER.get(s, 3) for s in table["status"]]
    table = table.sort_values(["_order", "value"], ascending=[True, False])
    return table.drop(columns="_order").reset_index(drop=True)


def feature_drift_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: Sequence[str],
    bins: int = 10,
) -> pd.DataFrame:
    """PSI per feature with a red/yellow/green status.

    Delegates to ``monitoring.drift.feature_drift``; if monitoring is not
    installed the panel degrades to an empty table rather than failing.
    """
    columns = ["feature", "psi", "status", "threshold", "detail"]
    try:
        from monitoring.drift import feature_drift
    except Exception:
        return _empty(columns)
    usable = [f for f in features if f in reference.columns and f in current.columns]
    if not usable or reference.empty or current.empty:
        return _empty(columns)
    signals = feature_drift(reference, current, usable, bins=bins)
    table = signals_table(signals).rename(columns={"name": "feature", "value": "psi"})
    return table[columns]


def drift_thresholds() -> dict[str, float]:
    """PSI warning/alert thresholds, from monitoring when it is importable."""
    try:
        from monitoring.drift import PSI_RED, PSI_YELLOW
    except Exception:
        return {"yellow": PSI_YELLOW, "red": PSI_RED}
    return {"yellow": float(PSI_YELLOW), "red": float(PSI_RED)}


def overall_status(signals: Sequence[Any]) -> str:
    """Worst status across the supplied signals ("unknown" when there are none)."""
    try:
        from monitoring.drift import overall_status as _overall
    except Exception:
        _overall = None
    if _overall is not None and signals:
        try:
            return str(_overall(list(signals)))
        except Exception:
            pass
    statuses = [getattr(s, "status", "unknown") for s in signals]
    for level in ("red", "yellow", "green"):
        if level in statuses:
            return level
    return "unknown"


# --- panel 5: residual drift ------------------------------------------------


def residual_frame(
    rows: Iterable[Mapping[str, Any]] | pd.DataFrame,
    actuals: Mapping[str, Mapping[date, float]],
) -> pd.DataFrame:
    """Join logged predictions to observed actuals and compute residuals.

    ``actuals`` is ``{series_id: {date: value}}`` — exactly the shape
    ``serving.predictor.Predictor.actuals`` returns — so the caller can build it
    once per series and tests can pass a literal dict. Rows whose target date has
    not been observed yet are dropped: an unrealised forecast has no residual.
    """
    columns = ["series_id", "target_date", "p50", "actual", "residual", "abs_error"]
    frame = rows if isinstance(rows, pd.DataFrame) else predictions_frame(rows)
    if frame.empty:
        return _empty(columns)

    records: list[dict[str, Any]] = []
    for series_id, target_date, p50 in zip(
        frame["series_id"], frame["target_date"], frame["p50"], strict=False
    ):
        observed = actuals.get(series_id, {})
        key = _as_date(target_date)
        if key is None or key not in observed or pd.isna(p50):
            continue
        actual = float(observed[key])
        records.append(
            {
                "series_id": series_id,
                "target_date": key,
                "p50": float(p50),
                "actual": actual,
                "residual": float(p50) - actual,
                "abs_error": abs(float(p50) - actual),
            }
        )
    if not records:
        return _empty(columns)
    return pd.DataFrame(records, columns=columns).sort_values("target_date").reset_index(drop=True)


def residual_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Daily mean/std residual with observation counts, for the time-series panel."""
    columns = ["date", "mean_residual", "std_residual", "n"]
    if frame.empty:
        return _empty(columns)
    grouped = frame.groupby("target_date")["residual"]
    out = pd.DataFrame(
        {
            "date": list(grouped.groups.keys()),
            "mean_residual": grouped.mean().to_numpy(),
            "std_residual": grouped.std(ddof=0).fillna(0.0).to_numpy(),
            "n": grouped.size().to_numpy().astype(int),
        }
    )
    return out.sort_values("date").reset_index(drop=True)


def _fallback_rolling_wmape(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Local rolling WMAPE, used only when ``monitoring.drift`` is unavailable.

    Kept deliberately small and identical in shape to the monitoring version
    (columns ``date``/``wmape``) so the panel keeps working in a lean install.
    """
    daily = frame.groupby("target_date").agg(
        abs_error=("abs_error", "sum"), actual=("actual", lambda s: s.abs().sum())
    )
    daily = daily.sort_index()
    num = daily["abs_error"].rolling(window, min_periods=1).sum()
    den = daily["actual"].rolling(window, min_periods=1).sum().replace(0.0, np.nan)
    return pd.DataFrame({"date": list(daily.index), "wmape": (num / den).to_numpy()})


def rolling_wmape_series(frame: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Rolling WMAPE over the joined prediction/actual frame (``date``, ``wmape``)."""
    if frame.empty:
        return _empty(["date", "wmape"])
    try:
        from monitoring.drift import rolling_wmape

        return rolling_wmape(frame, window=window)
    except Exception:
        return _fallback_rolling_wmape(frame, window)


def residual_drift_signal(reference: pd.DataFrame, current: pd.DataFrame) -> Any | None:
    """PSI of the residual distribution, reference vs current (``None`` if absent)."""
    try:
        from monitoring.drift import residual_drift
    except Exception:
        return None
    if reference.empty or current.empty:
        return None
    try:
        return residual_drift(reference["residual"], current["residual"])
    except Exception:
        return None


def wmape_breach_signal(rolling: pd.DataFrame, baseline_wmape: float | None) -> Any | None:
    """Whether rolling WMAPE has breached the retraining tolerance (``None`` if absent)."""
    if baseline_wmape is None or rolling.empty:
        return None
    try:
        from monitoring.drift import wmape_breach
    except Exception:
        return None
    try:
        return wmape_breach(rolling, float(baseline_wmape))
    except Exception:
        return None


# --- panel 6: leaderboard ---------------------------------------------------


def leaderboard_table(
    dataset: str | None = None,
    fallback_metrics: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Registered models ranked by validation WMAPE (best first)."""
    ds = dataset or served_dataset()
    try:
        from serving.leaderboard import load_leaderboard

        payload = load_leaderboard(ds, dict(fallback_metrics) if fallback_metrics else None)
    except Exception:
        return _empty(["model"])
    entries = payload.get("entries") or []
    if not entries:
        return _empty(["model"])
    table = pd.DataFrame(entries)
    cols = ["model"] + [c for c in table.columns if c != "model"]
    return table[cols]


# --- panel 7: retraining + drift event history ------------------------------


def retraining_events_table(
    events: Sequence[Mapping[str, Any]] | None = None,
    dataset: str | None = None,
    limit: int = 5,
) -> pd.DataFrame:
    """Last ``limit`` retraining events with before/after metrics side by side.

    ``events`` can be injected (tests, replay); otherwise they are read from
    ``monitoring.store``. Metric dicts are flattened to ``wmape_before`` /
    ``wmape_after`` / ``wmape_delta`` columns because a nested dict in a
    dataframe cell is unreadable in a dashboard.
    """
    columns = [
        "created_at",
        "dataset",
        "triggered_by",
        "promoted",
        "model_version",
        "wmape_before",
        "wmape_after",
        "wmape_delta",
        "notes",
    ]
    if events is None:
        try:
            from monitoring.store import fetch_retraining_events

            events = fetch_retraining_events(dataset=dataset or served_dataset(), limit=limit)
        except Exception:
            return _empty(columns)
    rows = [dict(e) for e in list(events)[:limit]]
    if not rows:
        return _empty(columns)

    out: list[dict[str, Any]] = []
    for e in rows:
        before = _coerce_metrics(e.get("before_metrics"))
        after = _coerce_metrics(e.get("after_metrics"))
        b, a = before.get("wmape"), after.get("wmape")
        out.append(
            {
                "created_at": e.get("created_at") or e.get("recorded_at"),
                "dataset": e.get("dataset", ""),
                "triggered_by": e.get("triggered_by", ""),
                "promoted": bool(e.get("promoted", False)),
                "model_version": e.get("model_version", ""),
                "wmape_before": b,
                "wmape_after": a,
                "wmape_delta": (a - b) if (a is not None and b is not None) else None,
                "notes": e.get("notes", ""),
            }
        )
    return pd.DataFrame(out, columns=columns)


def _coerce_metrics(value: Any) -> dict[str, float]:
    """Metrics may arrive as a dict or as a JSON string, depending on the store."""
    if isinstance(value, Mapping):
        raw = dict(value)
    elif isinstance(value, str) and value.strip():
        import json

        try:
            raw = json.loads(value)
        except ValueError:
            return {}
        if not isinstance(raw, dict):
            return {}
    else:
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def drift_events_table(
    events: Sequence[Mapping[str, Any]] | None = None,
    dataset: str | None = None,
    limit: int = 100,
) -> pd.DataFrame:
    """Recorded drift events (newest first), for the history strip under the panels."""
    columns = ["created_at", "dataset", "signal_name", "value", "status", "detail"]
    if events is None:
        try:
            from monitoring.store import fetch_drift_events

            events = fetch_drift_events(dataset=dataset or served_dataset(), limit=limit)
        except Exception:
            return _empty(columns)
    rows = [dict(e) for e in list(events)[:limit]]
    if not rows:
        return _empty(columns)
    table = pd.DataFrame(rows)
    for col in columns:
        if col not in table.columns:
            table[col] = None
    return table[columns]


__all__ = [
    "DEFAULT_DATASET",
    "PSI_RED",
    "PSI_YELLOW",
    "STATUS_ORDER",
    "drift_events_table",
    "drift_thresholds",
    "feature_drift_table",
    "fetch_prediction_rows",
    "hourly_volume",
    "leaderboard_table",
    "metrics_table",
    "monitoring_available",
    "overall_status",
    "prediction_histogram",
    "predictions_frame",
    "production_model_info",
    "reference_current_split",
    "request_volume",
    "residual_drift_signal",
    "residual_frame",
    "residual_series",
    "retraining_events_table",
    "rolling_wmape_series",
    "served_dataset",
    "signals_table",
    "wmape_breach_signal",
]
