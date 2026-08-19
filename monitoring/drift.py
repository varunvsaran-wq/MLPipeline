"""Drift signals — the monitoring half of the retraining loop.

A forecasting model degrades in three distinguishable ways, and conflating them
makes the alert useless. So we compute three separate signals:

* **Input feature drift** — the lag/rolling/calendar features the model consumes
  no longer look like what it was fitted on. Measured with the Population
  Stability Index over quantile bins of a reference window, the standard retail
  practice: PSI < 0.1 is noise, 0.1-0.2 is worth watching, > 0.2 is a real shift.
* **Residual drift** — the distribution of ``p50 - actual`` moves. The inputs may
  be perfectly stable while the relationship the model learned has changed; this
  is the signal that catches that.
* **Rolling WMAPE** — the business-facing one. A 30-day rolling WMAPE that runs
  more than ``tolerance`` worse than the validation WMAPE is the retraining
  trigger, not merely an alert.

Everything here is pure pandas/numpy so the thresholds are testable without any
optional dependency. :func:`evidently_report` is a thin, guarded extra that
renders the same comparison as an Evidently HTML report when the library happens
to be installed; if its API does not match, it returns ``None`` rather than
taking the core signals down with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from models.metrics import wmape

PSI_YELLOW: float = 0.1
PSI_RED: float = 0.2

# Floor for empty bins so the log ratio stays finite. Small relative to any
# meaningful share, large enough that a single empty bin cannot dominate.
_EPS: float = 1e-6

_STATUS_ORDER: dict[str, int] = {"green": 0, "yellow": 1, "red": 2}


@dataclass
class DriftSignal:
    """One evaluated signal: what it is, how bad it is, and the verdict."""

    name: str
    value: float
    status: str
    threshold: float
    detail: str = ""


def _psi_status(value: float) -> str:
    if value > PSI_RED:
        return "red"
    if value >= PSI_YELLOW:
        return "yellow"
    return "green"


def psi(reference: ArrayLike, current: ArrayLike, bins: int = 10) -> float:
    """Population Stability Index between a reference and a current sample.

    The reference is cut into ``bins`` quantile bins (so the reference shares are
    roughly uniform and no bin is defined by an arbitrary grid), the current
    sample is bucketed with the same edges, and we sum
    ``(cur% - ref%) * ln(cur% / ref%)`` over bins. Empty bins are floored at a
    small epsilon so the result is always finite.

    Returns 0.0 when either sample is empty or the reference is constant — an
    undefined comparison should not masquerade as drift.
    """
    if bins < 2:
        raise ValueError(f"need at least 2 bins, got {bins}")
    ref = np.asarray(reference, dtype=float).ravel()
    cur = np.asarray(current, dtype=float).ravel()
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 2:
        # Constant reference: no distribution to compare against.
        return 0.0
    # Open the outer edges so out-of-range current values land in the end bins.
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_share = np.maximum(ref_counts / ref.size, _EPS)
    cur_share = np.maximum(cur_counts / cur.size, _EPS)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
    bins: int = 10,
) -> list[DriftSignal]:
    """PSI per feature, in the order given. Missing columns are skipped."""
    signals: list[DriftSignal] = []
    for name in features:
        if name not in reference.columns or name not in current.columns:
            continue
        ref_col = pd.to_numeric(reference[name], errors="coerce").to_numpy(dtype=float)
        cur_col = pd.to_numeric(current[name], errors="coerce").to_numpy(dtype=float)
        value = psi(ref_col, cur_col, bins=bins)
        signals.append(
            DriftSignal(
                name=name,
                value=value,
                status=_psi_status(value),
                threshold=PSI_RED,
                detail=f"PSI over {bins} reference quantile bins",
            )
        )
    return signals


def residual_drift(
    reference_residuals: ArrayLike,
    current_residuals: ArrayLike,
    bins: int = 10,
) -> DriftSignal:
    """PSI between two residual (``p50 - actual``) distributions.

    A shift here means the error structure has changed — a bias swing or a
    variance blow-up — even if every input feature still looks familiar.
    """
    ref = np.asarray(reference_residuals, dtype=float).ravel()
    cur = np.asarray(current_residuals, dtype=float).ravel()
    value = psi(ref, cur, bins=bins)
    ref_finite = ref[np.isfinite(ref)]
    cur_finite = cur[np.isfinite(cur)]
    ref_mean = float(np.mean(ref_finite)) if ref_finite.size else float("nan")
    cur_mean = float(np.mean(cur_finite)) if cur_finite.size else float("nan")
    return DriftSignal(
        name="residual",
        value=value,
        status=_psi_status(value),
        threshold=PSI_RED,
        detail=f"mean residual {ref_mean:.4g} -> {cur_mean:.4g}",
    )


def rolling_wmape(
    frame: pd.DataFrame,
    window: int = 30,
    date_col: str = "target_date",
    actual_col: str = "actual",
    pred_col: str = "p50",
) -> pd.DataFrame:
    """Rolling WMAPE over a trailing ``window`` of calendar days.

    One value per date on which something was scored, computed with
    :func:`models.metrics.wmape` over every row falling in the trailing window —
    a calendar window rather than a row count, so gaps in the panel don't
    silently stretch it, and volume-weighted across series by construction.
    Returns columns ``date, wmape``; windows with zero actual volume are dropped
    because WMAPE is undefined there.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    missing = {date_col, actual_col, pred_col} - set(frame.columns)
    if missing:
        raise KeyError(f"frame is missing columns: {sorted(missing)}")
    empty = pd.DataFrame(
        {"date": pd.Series(dtype="datetime64[ns]"), "wmape": pd.Series(dtype=float)}
    )
    if frame.empty:
        return empty

    work = frame[[date_col, actual_col, pred_col]].copy()
    work["date"] = pd.to_datetime(work[date_col])
    work = work.dropna(subset=["date", actual_col, pred_col]).sort_values("date")
    if work.empty:
        return empty

    dates = work["date"].to_numpy()
    actual = work[actual_col].to_numpy(dtype=float)
    pred = work[pred_col].to_numpy(dtype=float)
    span = pd.Timedelta(days=window - 1)

    out_dates: list[pd.Timestamp] = []
    out_values: list[float] = []
    for end in pd.unique(work["date"]):
        start = pd.Timestamp(end) - span
        lo = int(np.searchsorted(dates, np.datetime64(start), side="left"))
        hi = int(np.searchsorted(dates, np.datetime64(end), side="right"))
        if hi <= lo or np.abs(actual[lo:hi]).sum() == 0:
            continue
        out_dates.append(pd.Timestamp(end))
        out_values.append(wmape(actual[lo:hi], pred[lo:hi]))
    if not out_dates:
        return empty
    return pd.DataFrame({"date": pd.to_datetime(out_dates), "wmape": out_values})


def wmape_breach(
    rolling: pd.DataFrame,
    baseline_wmape: float,
    tolerance: float = 0.2,
) -> DriftSignal:
    """Compare the latest rolling WMAPE against the validation baseline.

    Red once the live error exceeds ``baseline * (1 + tolerance)`` — that is the
    retraining trigger. Yellow from halfway there, so degradation is visible
    before it fires.
    """
    limit = float(baseline_wmape) * (1.0 + float(tolerance))
    if rolling is None or rolling.empty or "wmape" not in rolling.columns:
        return DriftSignal(
            name="rolling_wmape",
            value=float("nan"),
            status="green",
            threshold=limit,
            detail="no scored actuals yet",
        )
    latest = float(rolling["wmape"].iloc[-1])
    warn = float(baseline_wmape) * (1.0 + float(tolerance) / 2.0)
    if latest > limit:
        status = "red"
    elif latest > warn:
        status = "yellow"
    else:
        status = "green"
    ratio = latest / baseline_wmape if baseline_wmape else float("nan")
    return DriftSignal(
        name="rolling_wmape",
        value=latest,
        status=status,
        threshold=limit,
        detail=(
            f"latest rolling WMAPE {latest:.4f} vs validation {baseline_wmape:.4f} "
            f"({ratio:.2f}x, trigger at {limit:.4f})"
        ),
    )


def overall_status(signals: list[DriftSignal]) -> str:
    """Worst status across signals (green < yellow < red); green when empty."""
    worst = "green"
    for signal in signals:
        if _STATUS_ORDER.get(signal.status, 0) > _STATUS_ORDER[worst]:
            worst = signal.status
    return worst


def evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> str | None:
    """Evidently data-drift HTML for the same comparison, or ``None``.

    Purely additive: the dashboard embeds it when available, and every threshold
    decision above is made without it. Any import or API mismatch returns
    ``None`` — an optional renderer must never break monitoring.
    """
    cols = [c for c in features if c in reference.columns and c in current.columns]
    if not cols:
        return None
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset

        definition = DataDefinition(numerical_columns=list(cols))
        report = Report([DataDriftPreset(num_method="psi", num_threshold=PSI_RED)])
        snapshot = report.run(
            Dataset.from_pandas(current[cols], data_definition=definition),
            Dataset.from_pandas(reference[cols], data_definition=definition),
        )
        return str(snapshot.get_html_str(as_iframe=False))
    except Exception:  # noqa: BLE001 - optional renderer, never fatal
        return None


__all__ = [
    "PSI_RED",
    "PSI_YELLOW",
    "DriftSignal",
    "evidently_report",
    "feature_drift",
    "overall_status",
    "psi",
    "residual_drift",
    "rolling_wmape",
    "wmape_breach",
]
