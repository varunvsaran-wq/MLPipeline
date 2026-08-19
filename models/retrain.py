"""Automated retraining loop — the step that closes the MLOps cycle (Phase 6).

    python -m models.retrain --dataset avocado [--force] [--dry-run] [--threshold 0.01]

Phases 1-5 built the pieces: features, model families, a promotion gate, a
prediction log and drift signals. This module is the controller that wires them
into a loop that can run unattended — drift is evaluated, and when it goes red
the platform retrains, re-evaluates on the same held-out horizon, asks the
promotion gate whether the candidate has earned Production, rebuilds the
servable bundle if it has, records the before/after metrics, and tells a human.

The design decisions worth stating plainly:

* **The trigger is separate from the action.** :func:`evaluate_drift` computes
  and persists the signals, :func:`should_retrain` is a pure rule over them, and
  :func:`retrain` performs the work. That split is what makes the expensive path
  testable: the rule can be exercised without a model, and the orchestration can
  be exercised with an injected ``trainer``.
* **Red retrains, yellow does not.** A yellow signal is a request for attention,
  not for a new model; retraining on every wobble burns compute and, worse,
  churns Production with models that are statistically indistinguishable. Only
  :func:`monitoring.drift.overall_status` == ``"red"`` fires, and ``--force``
  exists for the operator who has already decided.
* **Retraining is not promotion.** The candidate always goes through
  :func:`models.promote.run_gate`, so a retrain that produces a worse model
  leaves Production exactly where it was and the CLI exits non-zero. Closing the
  loop must not mean losing the safety rail that Phase 4 installed.
* **A missing registry degrades, it does not crash.** The local default MLflow
  store is a ``file://`` directory, which cannot host the Model Registry
  (:class:`~models.registry.RegistryUnsupportedError`). In that case we still
  retrain, still evaluate the candidate against the incumbent's metrics with the
  same pure decision rule, still record the event, and report clearly that the
  registry transition was skipped. A local developer gets the whole loop minus
  the one part that genuinely needs a database.
* **``dvc pull`` is best-effort.** Fetching the data is the first step of an
  honest retrain, but it fails without a configured remote, and a missing remote
  should not stop a retrain on data that is already on disk. The outcome is
  logged as a step, never raised, and ``--skip-dvc`` opts out entirely.

Exit codes follow the gate, so CI can branch on them: ``0`` when nothing needed
doing or the candidate is promotable, ``1`` when the gate blocked it, ``2`` on a
configuration error.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from models import notify
from models.promote import (
    DEFAULT_METRIC,
    DEFAULT_THRESHOLD,
    PromotionDecision,
    run_gate,
    should_promote,
)
from models.registry import RegistryUnsupportedError
from monitoring import store as monitoring_store
from monitoring.drift import (
    DriftSignal,
    feature_drift,
    overall_status,
    residual_drift,
    rolling_wmape,
    wmape_breach,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Registered-model name per dataset. One name per dataset (not per family) so
#: the Production pointer answers "what is live for avocado?" regardless of
#: which family last won the comparison.
MODEL_NAME_TEMPLATE = "demand-forecasting-{dataset}"

#: Trailing calendar days for the rolling-WMAPE signal.
DEFAULT_WINDOW = 30
#: Fraction worse than the validation baseline that counts as a breach.
DEFAULT_TOLERANCE = 0.2
#: Date quantile splitting the drift reference window from the current window.
DEFAULT_SPLIT_FRAC = 0.7
#: Features PSI is computed over. The full suite is ~40 columns for avocado;
#: alerting on all of them is noise, and the lag/rolling block leads the list.
DEFAULT_MAX_FEATURES = 8
#: Calendar features are excluded from PSI. The reference and current windows are
#: split *by date*, so month/week/day-of-week distributions differ by
#: construction — they would fire red on every healthy dataset and drown the
#: signals that mean something. Same for the deterministic Fourier terms.
_CALENDAR_FEATURES = frozenset(
    {"month", "weekofyear", "quarter", "dayofweek", "is_weekend", "is_holiday"}
)


@dataclass
class TrainingOutcome:
    """What one training pass produced: metrics per family and their MLflow runs.

    Kept as an explicit type (rather than a bare metrics dict) because the
    promotion gate needs the *run* behind the winning family, and because it is
    the seam tests inject at to avoid retraining real models.
    """

    metrics_by_model: dict[str, dict[str, float]] = field(default_factory=dict)
    run_ids: dict[str, str] = field(default_factory=dict)
    notes: str = ""


@dataclass
class RetrainReport:
    """Everything one retraining cycle decided, in the order it decided it."""

    dataset: str
    triggered_by: str = "manual"
    triggered: bool = False
    drift_status: str = "unknown"
    signals: list[DriftSignal] = field(default_factory=list)
    before_metrics: dict[str, float] = field(default_factory=dict)
    after_metrics: dict[str, float] = field(default_factory=dict)
    metric: str = DEFAULT_METRIC
    best_family: str | None = None
    candidate_run_id: str | None = None
    decision: PromotionDecision | None = None
    promoted: bool = False
    registry_available: bool = True
    bundle_version: str | None = None
    event_id: int | None = None
    dry_run: bool = False
    steps: list[str] = field(default_factory=list)
    notes: str = ""
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def metric_delta(self) -> float | None:
        """``after - before`` on the gate metric, or ``None`` if either is absent."""
        before = self.before_metrics.get(self.metric)
        after = self.after_metrics.get(self.metric)
        if before is None or after is None:
            return None
        return float(after) - float(before)

    def step(self, text: str) -> str:
        """Append one narrated step (the CLI prints these as they happen)."""
        self.steps.append(text)
        return text

    def metrics_diff(self) -> str:
        return notify.format_metrics_diff(self.before_metrics, self.after_metrics)

    def summary(self) -> str:
        """Human-readable report, used for the notification body and the CLI tail."""
        head = "PROMOTED" if self.promoted else ("RETRAINED" if self.triggered else "NO ACTION")
        lines = [
            f"[{head}] dataset={self.dataset} trigger={self.triggered_by} "
            f"drift={self.drift_status}{' (dry run)' if self.dry_run else ''}",
            "",
            "steps:",
            *(f"  {i + 1}. {s}" for i, s in enumerate(self.steps)),
        ]
        if self.triggered:
            lines += [
                "",
                f"best family : {self.best_family or '-'}",
                f"candidate   : run {self.candidate_run_id or '-'}"
                + (f", version {self.decision.candidate_version}" if self.decision else ""),
                "",
                "metrics diff (production -> candidate):",
                self.metrics_diff(),
            ]
        if self.decision is not None:
            lines += ["", f"gate: {self.decision.reason}"]
        if self.notes:
            lines += ["", f"notes: {self.notes}"]
        return "\n".join(lines)


# --- drift evaluation -------------------------------------------------------


def fetch_logged_predictions(dataset: str, limit: int = 50_000) -> list[dict]:
    """Rows from the serving prediction log for one dataset (oldest first).

    ``serving.store`` exposes a per-series history helper; the retrainer needs a
    dataset-wide slice, so the SELECT is built here against the shared table
    rather than widening the serving API for one caller.
    """
    from sqlalchemy import select

    from serving.store import get_engine, predictions

    stmt = (
        select(predictions)
        .where(predictions.c.dataset == dataset)
        .order_by(predictions.c.target_date.asc(), predictions.c.id.asc())
        .limit(limit)
    )
    with get_engine().connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt)]


def _split_by_date(
    frame: pd.DataFrame, date_col: str, split_frac: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cut a frame into reference/current windows at a date quantile.

    The cutoff is a quantile over *distinct dates*, not rows, so every series is
    split at the same instant and the two windows stay like-for-like.
    """
    if frame.empty or date_col not in frame.columns:
        return frame, frame
    dates = pd.to_datetime(pd.Series(list(frame[date_col])), errors="coerce")
    unique = pd.Series(dates.dropna().unique()).sort_values()
    if len(unique) < 2:
        return frame, frame
    index = min(max(int(len(unique) * split_frac), 1), len(unique) - 1)
    cutoff = unique.iloc[index]
    mask = (dates < cutoff).to_numpy()
    return frame[mask], frame[~mask]


def scored_frame(
    cfg: Any, history: pd.DataFrame, rows: Sequence[Mapping[str, Any]]
) -> pd.DataFrame:
    """Join logged predictions to observed actuals: ``target_date/actual/p50/residual``.

    Forecast rows whose target date has not been observed yet are dropped — an
    unrealised forecast has no residual and must not dilute the error signal.
    """
    from features.pipeline import series_key

    columns = ["series_id", "target_date", "p50", "actual", "residual"]
    if not len(rows):
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    hist = history.copy()
    hist[cfg.date_col] = pd.to_datetime(hist[cfg.date_col])
    hist["__sid__"] = series_key(cfg, hist).values
    actuals = {
        (sid, pd.Timestamp(dt).normalize()): float(value)
        for sid, dt, value in zip(
            hist["__sid__"], hist[cfg.date_col], hist[cfg.target_col], strict=True
        )
        if pd.notna(value)
    }

    records: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get("series_id"), pd.Timestamp(row.get("target_date")).normalize())
        actual = actuals.get(key)
        p50 = row.get("p50")
        if actual is None or p50 is None or pd.isna(p50):
            continue
        records.append(
            {
                "series_id": key[0],
                "target_date": key[1],
                "p50": float(p50),
                "actual": actual,
                "residual": float(p50) - actual,
            }
        )
    if not records:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})
    return pd.DataFrame(records, columns=columns).sort_values("target_date").reset_index(drop=True)


def drift_features(numeric: Sequence[str]) -> list[str]:
    """Numeric features worth monitoring: the data-bearing ones, calendar excluded.

    Lag, rolling and lagged-exogenous columns describe the demand process and are
    where a regime change shows up. Calendar and Fourier columns are pure
    functions of the timestamp, so a date-split reference/current comparison
    guarantees they differ — a red flag that carries no information.
    """
    return [
        name
        for name in numeric
        if name not in _CALENDAR_FEATURES and not name.startswith("fourier")
    ]


def evaluate_drift(
    dataset: str = "avocado",
    bundle: Any = None,
    rows: Sequence[Mapping[str, Any]] | None = None,
    metric: str = DEFAULT_METRIC,
    window: int = DEFAULT_WINDOW,
    tolerance: float = DEFAULT_TOLERANCE,
    split_frac: float = DEFAULT_SPLIT_FRAC,
    max_features: int = DEFAULT_MAX_FEATURES,
    record: bool = True,
) -> tuple[str, list[DriftSignal]]:
    """Compute the three drift signals for a dataset and persist them.

    ``bundle`` (anything exposing ``cfg``/``history``/``metrics``, normally a
    :class:`serving.model_bundle.ModelBundle`) is the production model under
    scrutiny: its history supplies both the feature matrix and the actuals, and
    its validation WMAPE is the baseline the rolling error is judged against.
    ``rows`` defaults to the serving prediction log.

    Returns ``(overall_status, signals)``. Recording is best-effort: an
    unreachable event store must not stop the trigger from firing.
    """
    from features.pipeline import build_feature_matrix

    if bundle is None:
        from serving.model_bundle import load_bundle

        bundle = load_bundle(dataset)
    cfg = bundle.cfg
    history = bundle.history
    signals: list[DriftSignal] = []

    frame, spec = build_feature_matrix(cfg, history)
    features = drift_features(spec.numeric)[:max_features]
    reference, current = _split_by_date(frame, "ds", split_frac)
    signals.extend(feature_drift(reference, current, features))

    prediction_rows = list(rows) if rows is not None else fetch_logged_predictions(dataset)
    scored = scored_frame(cfg, history, prediction_rows)
    if not scored.empty:
        ref_scored, cur_scored = _split_by_date(scored, "target_date", split_frac)
        signals.append(residual_drift(ref_scored["residual"], cur_scored["residual"]))
        baseline = bundle.metrics.get(metric) if getattr(bundle, "metrics", None) else None
        if baseline is not None and math.isfinite(float(baseline)) and float(baseline) > 0:
            rolling = rolling_wmape(scored, window=window)
            signals.append(wmape_breach(rolling, float(baseline), tolerance=tolerance))

    status = overall_status(signals)
    if record:
        for signal in signals:
            try:
                monitoring_store.record_drift_event(
                    dataset, signal.name, signal.value, signal.status, signal.detail
                )
            except Exception as exc:  # noqa: BLE001 - the audit log is not the trigger
                print(f"[warn] could not record drift event {signal.name}: {exc}", file=sys.stderr)
    return status, signals


def should_retrain(status_or_signals: str | Sequence[DriftSignal]) -> bool:
    """The trigger rule: retrain if and only if overall drift status is red.

    Accepts either a status string or the signal list, so callers can pass
    whichever they hold. Yellow is an alert, not a trigger — see the module
    docstring for why.
    """
    if isinstance(status_or_signals, str):
        status = status_or_signals
    else:
        status = overall_status(list(status_or_signals))
    return status == "red"


# --- data + training --------------------------------------------------------


def dvc_pull(skip: bool = False, timeout: int = 900) -> tuple[bool, str]:
    """Best-effort ``dvc pull``. Returns ``(ok, message)`` and never raises."""
    if skip:
        return True, "dvc pull skipped (--skip-dvc)"
    try:
        result = subprocess.run(
            ["dvc", "pull"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"dvc pull unavailable ({exc}); continuing with data already on disk"
    if result.returncode == 0:
        tail = (result.stdout or "").strip().splitlines()
        return True, f"dvc pull ok: {tail[-1] if tail else 'up to date'}"
    detail = ((result.stderr or result.stdout or "").strip().splitlines() or ["no output"])[-1]
    return False, f"dvc pull failed ({detail}); continuing with data already on disk"


def latest_run_id(dataset: str, family: str, tracking_uri: str | None = None) -> str | None:
    """Most recent MLflow run for one model family on one dataset, or ``None``.

    ``models.run_comparison`` tags every child run with ``model_family`` and
    ``dataset`` but returns only metrics, so this is how the retrainer recovers
    the run the promotion gate needs. Guarded: a tracking store we cannot query
    means "no candidate run", which the caller degrades on.
    """
    try:
        from mlflow.tracking import MlflowClient

        from models.run_comparison import EXPERIMENT
        from models.tracking import DEFAULT_TRACKING_URI

        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if experiment is None:
            return None
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.model_family = '{family}' and tags.dataset = '{dataset}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        return runs[0].info.run_id if runs else None
    except Exception:  # noqa: BLE001 - no run id simply means no registry gate
        return None


def run_comparison_trainer(
    dataset: str,
    max_series: int | None = None,
    with_tft: bool = False,
    tft_epochs: int = 5,
    tracking_uri: str | None = None,
) -> TrainingOutcome:
    """Default trainer: re-run the full family comparison on the fixed holdout.

    Reuses :func:`models.run_comparison.run` verbatim so the retrained models are
    evaluated exactly the way the leaderboard was — same feature pipeline, same
    held-out horizon per series, same logged metrics. Anything else would make
    the candidate's numbers incomparable with the incumbent's.
    """
    from models.run_comparison import run as run_comparison

    metrics_by_model = run_comparison(
        dataset, max_series=max_series, with_tft=with_tft, tft_epochs=tft_epochs
    )
    metrics = {
        family: {str(k): float(v) for k, v in values.items()}
        for family, values in metrics_by_model.items()
    }
    run_ids: dict[str, str] = {}
    for family in metrics:
        found = latest_run_id(dataset, family.lower(), tracking_uri=tracking_uri)
        if found:
            run_ids[family] = found
    return TrainingOutcome(metrics_by_model=metrics, run_ids=run_ids)


def best_family(
    metrics_by_model: Mapping[str, Mapping[str, float]],
    metric: str = DEFAULT_METRIC,
    higher_is_better: bool = False,
) -> str | None:
    """Family with the best value of ``metric``; ``None`` when nothing is comparable."""
    scored = [
        (family, float(values[metric]))
        for family, values in metrics_by_model.items()
        if metric in values and math.isfinite(float(values[metric]))
    ]
    if not scored:
        return None
    chooser = max if higher_is_better else min
    return chooser(scored, key=lambda item: item[1])[0]


# --- orchestration ----------------------------------------------------------


def _production_metrics(dataset: str, override: Mapping[str, float] | None) -> dict[str, float]:
    """Metrics of the model currently served, for the 'before' half of the diff."""
    if override is not None:
        return {str(k): float(v) for k, v in override.items()}
    try:
        from serving.model_bundle import load_bundle

        return {str(k): float(v) for k, v in (load_bundle(dataset).metrics or {}).items()}
    except Exception:  # noqa: BLE001 - no bundle yet is a legitimate cold start
        return {}


def retrain(
    dataset: str = "avocado",
    triggered_by: str = "manual",
    model_name: str | None = None,
    metric: str = DEFAULT_METRIC,
    threshold: float = DEFAULT_THRESHOLD,
    higher_is_better: bool = False,
    tracking_uri: str | None = None,
    skip_dvc: bool = False,
    dry_run: bool = False,
    force: bool = False,
    max_series: int | None = None,
    with_tft: bool = False,
    rebuild_bundle: bool = True,
    before_metrics: Mapping[str, float] | None = None,
    status: str | None = None,
    signals: Sequence[DriftSignal] | None = None,
    trainer: Callable[[str], TrainingOutcome] | None = None,
    notifier: Callable[[str, str], Any] | None = None,
    verbose: bool = True,
) -> RetrainReport:
    """Run one full retraining cycle and return a structured report.

    Steps, in order: evaluate drift (unless ``signals``/``status`` are supplied),
    apply the trigger rule, ``dvc pull``, retrain every family and evaluate them
    on the fixed holdout, pick the winner, run the promotion gate, rebuild the
    servable bundle on promotion, record the retraining event, notify.

    ``trainer`` is the injection seam for tests and for anyone who wants a
    different training entry point; it receives the dataset name and returns a
    :class:`TrainingOutcome`. ``dry_run`` evaluates the gate without writing to
    the registry and skips the bundle rebuild.
    """
    report = RetrainReport(
        dataset=dataset, triggered_by=triggered_by, metric=metric, dry_run=dry_run
    )

    def announce(text: str) -> None:
        report.step(text)
        if verbose:
            print(f"[retrain] {text}", flush=True)

    # 1. Drift.
    if signals is not None or status is not None:
        report.signals = list(signals or [])
        report.drift_status = status or overall_status(report.signals)
        announce(f"drift status supplied by caller: {report.drift_status}")
    else:
        try:
            report.drift_status, report.signals = evaluate_drift(
                dataset, metric=metric, window=DEFAULT_WINDOW
            )
            reds = [s.name for s in report.signals if s.status == "red"]
            announce(
                f"evaluated {len(report.signals)} drift signals -> {report.drift_status}"
                + (f" (red: {', '.join(reds)})" if reds else "")
            )
        except Exception as exc:  # noqa: BLE001 - monitoring gaps must not block --force
            report.drift_status = "unknown"
            report.notes = f"drift evaluation failed: {exc}"
            announce(f"drift evaluation failed ({exc}); status unknown")

    # 2. Trigger rule.
    report.triggered = force or should_retrain(report.drift_status)
    if not report.triggered:
        announce(f"no retrain: drift is {report.drift_status}, trigger requires red (use --force)")
        _notify(report, notifier)
        return report
    announce(
        "retraining triggered" + (" by --force" if force and report.drift_status != "red" else "")
    )

    # 3. Data.
    ok, message = dvc_pull(skip=skip_dvc)
    announce(message if ok else f"warning: {message}")

    # 4. Train + evaluate on the fixed holdout.
    report.before_metrics = _production_metrics(dataset, before_metrics)
    train = trainer or (
        lambda ds: run_comparison_trainer(
            ds, max_series=max_series, with_tft=with_tft, tracking_uri=tracking_uri
        )
    )
    outcome = train(dataset)
    if not outcome.metrics_by_model:
        report.notes = "training produced no metrics; nothing to gate"
        announce(report.notes)
        _record_event(report)
        _notify(report, notifier)
        return report
    announce(
        "retrained families: "
        + ", ".join(
            f"{family} {metric}={values.get(metric, float('nan')):.4f}"
            for family, values in outcome.metrics_by_model.items()
        )
    )

    # 5. Pick the winner.
    report.best_family = best_family(outcome.metrics_by_model, metric, higher_is_better)
    if report.best_family is None:
        report.notes = f"no family reported a finite {metric}; nothing to gate"
        announce(report.notes)
        _record_event(report)
        _notify(report, notifier)
        return report
    report.after_metrics = dict(outcome.metrics_by_model[report.best_family])
    report.candidate_run_id = outcome.run_ids.get(report.best_family)
    announce(
        f"best family: {report.best_family} "
        f"({metric}={report.after_metrics.get(metric, float('nan')):.4f}, "
        f"run={report.candidate_run_id or 'unknown'})"
    )

    # 6. Promotion gate.
    _gate(
        report,
        model_name or MODEL_NAME_TEMPLATE.format(dataset=dataset),
        threshold,
        higher_is_better,
        tracking_uri,
        dry_run,
        announce,
    )

    # 7. Rebuild the servable bundle when the candidate wins.
    wants_bundle = report.promoted or (not report.registry_available and _would_promote(report))
    if wants_bundle and rebuild_bundle and not dry_run:
        try:
            from serving.model_bundle import build_bundle

            bundle = build_bundle(dataset)
            report.bundle_version = bundle.model_version
            announce(f"rebuilt servable bundle: {bundle.model_version}")
        except Exception as exc:  # noqa: BLE001 - the registry is still the source of truth
            announce(f"warning: bundle rebuild failed ({exc}); serving still on the old artifact")
    elif wants_bundle and not rebuild_bundle:
        announce("bundle rebuild skipped (rebuild_bundle=False)")

    # 8. Audit trail + notification.
    _record_event(report)
    if report.event_id:
        announce(f"recorded retraining event #{report.event_id}")
    _notify(report, notifier)
    return report


def _would_promote(report: RetrainReport) -> bool:
    return bool(report.decision and report.decision.promote)


def _gate(
    report: RetrainReport,
    model_name: str,
    threshold: float,
    higher_is_better: bool,
    tracking_uri: str | None,
    dry_run: bool,
    announce: Callable[[str], None],
) -> None:
    """Run the promotion gate, degrading to a local comparison without a registry."""
    if report.candidate_run_id is None:
        report.registry_available = False
        announce("no MLflow run id for the candidate; falling back to a local metric comparison")
    else:
        try:
            report.decision = run_gate(
                model_name,
                run_id=report.candidate_run_id,
                metric=report.metric,
                threshold=threshold,
                higher_is_better=higher_is_better,
                tracking_uri=tracking_uri,
                dry_run=dry_run,
            )
            report.promoted = report.decision.promoted
            announce(
                f"promotion gate: {'PROMOTE' if report.decision.promote else 'BLOCK'} "
                f"- {report.decision.reason}"
            )
            if report.promoted:
                announce(
                    f"{model_name} v{report.decision.candidate_version} -> Production"
                    + (
                        f" (archived v{', v'.join(str(v) for v in report.decision.archived_versions)})"
                        if report.decision.archived_versions
                        else ""
                    )
                )
            return
        except RegistryUnsupportedError as exc:
            report.registry_available = False
            report.notes = (
                "promotion skipped: no database-backed MLflow registry "
                "(set MLFLOW_TRACKING_URI to sqlite:///mlflow.db or the compose server). "
                f"{exc}"
            )
            announce("registry unavailable; falling back to a local metric comparison")

    promote, reason = should_promote(
        report.after_metrics.get(report.metric),
        report.before_metrics.get(report.metric),
        threshold=threshold,
        higher_is_better=higher_is_better,
    )
    report.decision = PromotionDecision(
        promote=promote,
        reason=f"local comparison only (registry unavailable): {reason}",
        metric=report.metric,
        candidate_metric=report.after_metrics.get(report.metric),
        production_metric=report.before_metrics.get(report.metric),
    )
    report.promoted = False
    announce(f"local verdict: {'would promote' if promote else 'would block'} - {reason}")


def _record_event(report: RetrainReport) -> None:
    """Append the retraining event; a failed audit write is logged, not raised."""
    if report.dry_run:
        report.step("dry run: retraining event not recorded")
        return
    version = report.bundle_version or (
        report.decision.candidate_version if report.decision else None
    )
    notes = report.decision.reason if report.decision else report.notes
    try:
        report.event_id = monitoring_store.record_retraining_event(
            dataset=report.dataset,
            triggered_by=report.triggered_by,
            before_metrics=report.before_metrics,
            after_metrics=report.after_metrics,
            promoted=report.promoted,
            model_version=version or "",
            notes=notes or "",
        )
    except Exception as exc:  # noqa: BLE001 - the loop already did the real work
        report.step(f"warning: could not record retraining event ({exc})")


def _notify(report: RetrainReport, notifier: Callable[[str, str], Any] | None) -> None:
    """Send the run summary. Notification failures are swallowed by design."""
    verdict = (
        "promoted" if report.promoted else ("no action" if not report.triggered else "blocked")
    )
    title = f"[{report.dataset}] retraining {verdict} (drift {report.drift_status})"
    try:
        (notifier or notify.send)(title, report.summary())
    except Exception as exc:  # noqa: BLE001 - never lose a run to a webhook
        report.step(f"warning: notification failed ({exc})")


# --- CLI --------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drift-triggered retraining loop (Phase 6).")
    p.add_argument("--dataset", default="avocado")
    p.add_argument(
        "--triggered-by", default="cli", help="what asked for this run (cron, drift, ...)"
    )
    p.add_argument("--model-name", default=None, help=f"default: {MODEL_NAME_TEMPLATE}")
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--higher-is-better", action="store_true")
    p.add_argument("--tracking-uri", default=None, help="overrides MLFLOW_TRACKING_URI")
    p.add_argument("--skip-dvc", action="store_true", help="do not attempt `dvc pull`")
    p.add_argument(
        "--max-series", type=int, default=None, help="cap Prophet series for a quick run"
    )
    p.add_argument("--with-tft", action="store_true", help="also retrain the TFT (needs [deep])")
    p.add_argument("--no-bundle", action="store_true", help="skip rebuilding the servable bundle")
    p.add_argument("--force", action="store_true", help="retrain even if drift is not red")
    p.add_argument(
        "--dry-run", action="store_true", help="evaluate without writing to the registry"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. 0 = nothing to do or promotable, 1 = gate blocked, 2 = error."""
    args = _parse_args(argv)
    try:
        report = retrain(
            dataset=args.dataset,
            triggered_by=args.triggered_by,
            model_name=args.model_name,
            metric=args.metric,
            threshold=args.threshold,
            higher_is_better=args.higher_is_better,
            tracking_uri=args.tracking_uri,
            skip_dvc=args.skip_dvc,
            dry_run=args.dry_run,
            force=args.force,
            max_series=args.max_series,
            with_tft=args.with_tft,
            rebuild_bundle=not args.no_bundle,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print()
    print(report.summary())
    if not report.triggered:
        return 0
    return 0 if _would_promote(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_FEATURES",
    "DEFAULT_SPLIT_FRAC",
    "DEFAULT_TOLERANCE",
    "DEFAULT_WINDOW",
    "MODEL_NAME_TEMPLATE",
    "RetrainReport",
    "TrainingOutcome",
    "best_family",
    "dvc_pull",
    "evaluate_drift",
    "fetch_logged_predictions",
    "latest_run_id",
    "main",
    "retrain",
    "run_comparison_trainer",
    "scored_frame",
    "should_retrain",
]
