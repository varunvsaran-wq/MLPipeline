"""Phase 6 tests: the retraining trigger, the orchestration, and its failure modes.

Everything here is synthetic and fast. The expensive half of the loop — actually
retraining the model families — is injected through ``retrain(trainer=...)``,
which is precisely why that seam exists; what these tests pin is the part that
decides *whether* to retrain, *what* the report says, and *how the loop behaves
when the registry is missing*.

Two self-contained backends are used, both throwaway: a temp SQLite prediction /
event database via ``SERVING_DB_URI`` (the ``temp_store`` pattern from
``test_phase3.py``) and a temp SQLite MLflow tracking+registry store (the
``registry_uri`` pattern from ``test_phase4_promotion.py``). No mocks of MLflow,
no docker-compose, no real training.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config import DatasetConfig, FeatureConfig
from models import notify
from models import retrain as retrain_mod
from models.retrain import (
    RetrainReport,
    TrainingOutcome,
    best_family,
    dvc_pull,
    evaluate_drift,
    retrain,
    scored_frame,
    should_retrain,
)
from monitoring.drift import DriftSignal

MODEL_NAME = "demand-forecasting-phase6"


# --- fixtures ---------------------------------------------------------------


@pytest.fixture()
def temp_store(tmp_path: Path, monkeypatch):
    """Throwaway prediction + monitoring database."""
    from monitoring import store as monitoring_store
    from serving import store as serving_store

    monkeypatch.setenv("SERVING_DB_URI", f"sqlite:///{(tmp_path / 'phase6.db').as_posix()}")
    serving_store.reset_engine()
    monitoring_store.reset_tables()
    yield monitoring_store
    serving_store.reset_engine()
    monitoring_store.reset_tables()


@pytest.fixture()
def registry_uri(tmp_path: Path, monkeypatch) -> str:
    """Throwaway SQLite tracking+registry store (the cheapest registry-capable one)."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp_id = mlflow.create_experiment(
        "phase6-retrain", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    monkeypatch.setenv("_PHASE6_EXPERIMENT_ID", exp_id)
    return uri


def _log_run(uri: str, wmape: float) -> str:
    import os

    import mlflow

    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(experiment_id=os.environ["_PHASE6_EXPERIMENT_ID"]) as run:
        mlflow.log_metric("wmape", wmape)
        mlflow.log_text("stand-in for the logged model", "model_p50/MLmodel")
        return run.info.run_id


def _signals(*statuses: str) -> list[DriftSignal]:
    return [
        DriftSignal(name=f"s{i}", value=0.5, status=status, threshold=0.2)
        for i, status in enumerate(statuses)
    ]


def _outcome(run_id: str | None = None, lgbm: float = 0.10, prophet: float = 0.20):
    return TrainingOutcome(
        metrics_by_model={
            "LightGBM": {"wmape": lgbm, "bias": -0.01, "n_series": 4.0},
            "Prophet": {"wmape": prophet, "bias": 0.05, "n_series": 4.0},
        },
        run_ids={"LightGBM": run_id} if run_id else {},
    )


# --- the trigger rule (pure) ------------------------------------------------


def test_red_status_triggers_retraining():
    assert should_retrain("red")


def test_yellow_and_green_do_not_trigger():
    assert not should_retrain("yellow")
    assert not should_retrain("green")
    assert not should_retrain("unknown")


def test_trigger_accepts_a_signal_list_and_takes_the_worst():
    assert should_retrain(_signals("green", "yellow", "red"))
    assert not should_retrain(_signals("green", "yellow"))
    assert not should_retrain([])


# --- family selection -------------------------------------------------------


def test_best_family_is_the_lowest_wmape():
    assert best_family({"A": {"wmape": 0.2}, "B": {"wmape": 0.1}}) == "B"


def test_best_family_ignores_missing_and_non_finite_metrics():
    metrics = {"A": {"bias": 0.1}, "B": {"wmape": float("nan")}, "C": {"wmape": 0.3}}
    assert best_family(metrics) == "C"
    assert best_family({"A": {"bias": 0.1}}) is None


def test_best_family_can_maximise():
    assert best_family({"A": {"coverage": 0.8}, "B": {"coverage": 0.9}}, "coverage", True) == "B"


# --- metrics diff -----------------------------------------------------------


def test_metrics_diff_reports_direction_and_delta():
    table = notify.format_metrics_diff({"wmape": 0.20}, {"wmape": 0.10})
    assert "wmape" in table
    assert "-0.1000" in table
    assert "-50.00%" in table
    assert "better" in table


def test_metrics_diff_marks_a_regression_as_worse():
    assert "worse" in notify.format_metrics_diff({"wmape": 0.10}, {"wmape": 0.15})


def test_metrics_diff_handles_one_sided_and_empty_metrics():
    table = notify.format_metrics_diff({}, {"wmape": 0.10})
    assert "wmape" in table and "-" in table
    assert notify.format_metrics_diff({}, {}) == "(no metrics recorded)"


def test_metrics_diff_orders_wmape_first():
    table = notify.format_metrics_diff({"bias": 0.1, "wmape": 0.2}, {"bias": 0.2, "wmape": 0.1})
    body = table.splitlines()[2:]
    assert body[0].startswith("wmape")


# --- notifications ----------------------------------------------------------


def test_send_falls_back_to_stdout_when_slack_is_unset(monkeypatch):
    monkeypatch.delenv(notify.SLACK_ENV_VAR, raising=False)
    stream = io.StringIO()
    result = notify.send("title", "body", stream=stream)
    assert result.delivered and result.channel == "stdout"
    assert "title" in stream.getvalue() and "body" in stream.getvalue()


def test_send_posts_to_slack_when_configured(monkeypatch):
    posted: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def getcode(self):
            return 200

    def fake_urlopen(request, timeout=None):
        posted["url"] = request.full_url
        posted["body"] = request.data.decode("utf-8")
        return _Response()

    monkeypatch.setenv(notify.SLACK_ENV_VAR, "https://hooks.slack.test/abc")
    monkeypatch.setattr(notify.urllib.request, "urlopen", fake_urlopen)
    result = notify.send("retrained", "wmape 0.2 -> 0.1")
    assert result.delivered and result.channel == "slack"
    assert posted["url"] == "https://hooks.slack.test/abc"
    assert "wmape 0.2 -> 0.1" in str(posted["body"])


def test_slack_failure_is_reported_not_raised(monkeypatch):
    def boom(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setenv(notify.SLACK_ENV_VAR, "https://hooks.slack.test/abc")
    monkeypatch.setattr(notify.urllib.request, "urlopen", boom)
    stream = io.StringIO()
    result = notify.send("title", "body", stream=stream)
    assert not result.delivered and result.channel == "slack"
    assert "connection refused" in (result.error or "")
    assert "body" in stream.getvalue()  # the content still reaches the operator


# --- dvc step ---------------------------------------------------------------


def test_dvc_pull_can_be_skipped():
    ok, message = dvc_pull(skip=True)
    assert ok and "skipped" in message


def test_dvc_pull_failure_is_not_fatal(monkeypatch):
    def boom(*args, **kwargs):
        raise OSError("dvc not installed")

    monkeypatch.setattr(retrain_mod.subprocess, "run", boom)
    ok, message = dvc_pull()
    assert not ok and "continuing with data already on disk" in message


# --- drift evaluation over synthetic data -----------------------------------


def _synthetic_cfg() -> DatasetConfig:
    return DatasetConfig(
        name="synthetic",
        date_col="date",
        target_col="demand",
        series_id_cols=["store"],
        horizon=7,
        freq="D",
        features=FeatureConfig(lags=[1, 7], rolling_windows=[7], fourier=[], use_exogenous=False),
    )


def _synthetic_bundle(shock: float = 3.0, days: int = 120, shocked_days: int = 40):
    """A bundle-shaped object whose recent history contains a demand shock."""
    cfg = _synthetic_cfg()
    rng = np.random.default_rng(7)
    start = date(2025, 1, 1)
    records = []
    for store in ("A", "B"):
        for step in range(days):
            # ``base`` is what a healthy model predicts; ``demand`` is what
            # actually happened, which the shock detaches from it.
            base = 100.0 + (10.0 if store == "B" else 0.0) + 5.0 * np.sin(2 * np.pi * step / 7)
            demand = base + float(rng.normal(0.0, 2.0))
            if step >= days - shocked_days:
                demand *= shock
            records.append(
                {
                    "date": start + timedelta(days=step),
                    "store": store,
                    "demand": demand,
                    "base": base,
                }
            )
    history = pd.DataFrame(records)
    return SimpleNamespace(cfg=cfg, history=history, metrics={"wmape": 0.05})


def _synthetic_rows(bundle, scored_days: int = 90) -> list[dict]:
    """Production forecasts that track the pre-shock level and miss the shock."""
    history = bundle.history
    cutoff = history["date"].max() - timedelta(days=scored_days)
    return [
        {
            "series_id": row["store"],
            "target_date": row["date"],
            "p50": float(row["base"]),
            "p10": float(row["base"]) * 0.9,
            "p90": float(row["base"]) * 1.1,
        }
        for _, row in history[history["date"] > cutoff].iterrows()
    ]


def test_scored_frame_joins_actuals_and_drops_unobserved():
    bundle = _synthetic_bundle()
    rows = _synthetic_rows(bundle)[:5]
    rows.append({"series_id": "A", "target_date": date(2099, 1, 1), "p50": 1.0})
    frame = scored_frame(bundle.cfg, bundle.history, rows)
    assert len(frame) == 5
    assert set(frame.columns) >= {"target_date", "actual", "p50", "residual"}
    assert frame["residual"].abs().sum() > 0


def test_scored_frame_is_empty_without_rows():
    bundle = _synthetic_bundle()
    assert scored_frame(bundle.cfg, bundle.history, []).empty


def test_demand_shock_drives_drift_red_and_is_recorded(temp_store):
    bundle = _synthetic_bundle()
    status, signals = evaluate_drift(
        "synthetic", bundle=bundle, rows=_synthetic_rows(bundle), window=30
    )
    assert status == "red"
    names = {s.name for s in signals}
    assert "residual" in names and "rolling_wmape" in names
    assert should_retrain(signals)

    events = temp_store.fetch_drift_events("synthetic")
    assert len(events) == len(signals)
    assert any(e["status"] == "red" for e in events)


def test_stable_history_stays_green(temp_store):
    # A longer, unshocked panel: PSI over a small current window is biased upward,
    # so a fair "no drift" check needs enough points on both sides of the split.
    bundle = _synthetic_bundle(shock=1.0, days=800)
    rows = [
        {"series_id": r["series_id"], "target_date": r["target_date"], "p50": r["p50"]}
        for r in _synthetic_rows(bundle, scored_days=600)
    ]
    status, signals = evaluate_drift("synthetic", bundle=bundle, rows=rows, window=30)
    assert status in {"green", "yellow"}
    assert not should_retrain(status)


def test_drift_can_be_evaluated_without_recording(temp_store):
    bundle = _synthetic_bundle()
    evaluate_drift("synthetic", bundle=bundle, rows=_synthetic_rows(bundle), record=False)
    assert temp_store.fetch_drift_events("synthetic") == []


# --- orchestration ----------------------------------------------------------


def test_no_retrain_when_drift_is_green(temp_store):
    report = retrain(
        "synthetic",
        status="green",
        signals=_signals("green"),
        skip_dvc=True,
        notifier=lambda title, body: None,
        verbose=False,
    )
    assert not report.triggered
    assert report.after_metrics == {}
    assert temp_store.fetch_retraining_events("synthetic") == []
    assert "no retrain" in report.summary()


def test_force_retrains_a_green_dataset(temp_store):
    report = retrain(
        "synthetic",
        status="green",
        signals=_signals("green"),
        force=True,
        skip_dvc=True,
        before_metrics={"wmape": 0.30},
        trainer=lambda ds: _outcome(),
        rebuild_bundle=False,
        notifier=lambda title, body: None,
        verbose=False,
    )
    assert report.triggered and report.best_family == "LightGBM"
    assert report.after_metrics["wmape"] == pytest.approx(0.10)
    assert report.metric_delta == pytest.approx(-0.20)


def test_missing_registry_degrades_without_crashing(temp_store, monkeypatch, tmp_path):
    """A file:// tracking store cannot host the registry: retrain, record, report."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    report = retrain(
        "synthetic",
        triggered_by="drift",
        status="red",
        signals=_signals("red"),
        skip_dvc=True,
        before_metrics={"wmape": 0.30},
        trainer=lambda ds: _outcome(run_id="deadbeef"),
        rebuild_bundle=False,
        notifier=lambda title, body: None,
        verbose=False,
    )
    assert report.triggered and not report.registry_available
    assert not report.promoted
    assert report.decision is not None and report.decision.promote  # local verdict is still made
    assert "registry unavailable" in report.decision.reason
    assert "database-backed MLflow registry" in report.notes

    events = temp_store.fetch_retraining_events("synthetic")
    assert len(events) == 1
    assert events[0]["before_metrics"]["wmape"] == pytest.approx(0.30)
    assert events[0]["after_metrics"]["wmape"] == pytest.approx(0.10)
    assert events[0]["promoted"] is False


def test_missing_candidate_run_falls_back_to_a_local_comparison(temp_store):
    report = retrain(
        "synthetic",
        status="red",
        signals=_signals("red"),
        skip_dvc=True,
        before_metrics={"wmape": 0.05},
        trainer=lambda ds: _outcome(),  # no run ids at all
        rebuild_bundle=False,
        notifier=lambda title, body: None,
        verbose=False,
    )
    assert not report.registry_available
    assert report.decision is not None and not report.decision.promote  # 0.10 is worse than 0.05


def test_retrain_promotes_a_better_candidate(temp_store, registry_uri):
    """HANDOFF Phase 6 acceptance, in miniature: red drift -> retrain -> promotion."""
    from models import registry

    incumbent_run = _log_run(registry_uri, 0.30)
    incumbent = registry.register_model(MODEL_NAME, incumbent_run, tracking_uri=registry_uri)
    registry.transition(
        MODEL_NAME, incumbent.version, registry.STAGE_PRODUCTION, tracking_uri=registry_uri
    )
    candidate_run = _log_run(registry_uri, 0.10)

    report = retrain(
        "synthetic",
        triggered_by="demand-shock",
        model_name=MODEL_NAME,
        status="red",
        signals=_signals("red"),
        skip_dvc=True,
        tracking_uri=registry_uri,
        before_metrics={"wmape": 0.30},
        trainer=lambda ds: _outcome(run_id=candidate_run),
        rebuild_bundle=False,
        notifier=lambda title, body: None,
        verbose=False,
    )

    assert report.registry_available and report.promoted
    assert report.decision is not None and report.decision.promote
    production = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert production is not None and production.run_id == candidate_run
    assert (
        registry.get_version(MODEL_NAME, incumbent.version, tracking_uri=registry_uri).current_stage
        == registry.STAGE_ARCHIVED
    )

    events = temp_store.fetch_retraining_events("synthetic")
    assert len(events) == 1 and events[0]["promoted"] is True
    assert "PROMOTED" in report.summary()


def test_retrain_blocks_a_worse_candidate(temp_store, registry_uri):
    from models import registry

    incumbent_run = _log_run(registry_uri, 0.10)
    incumbent = registry.register_model(MODEL_NAME, incumbent_run, tracking_uri=registry_uri)
    registry.transition(
        MODEL_NAME, incumbent.version, registry.STAGE_PRODUCTION, tracking_uri=registry_uri
    )
    candidate_run = _log_run(registry_uri, 0.25)

    report = retrain(
        "synthetic",
        model_name=MODEL_NAME,
        status="red",
        signals=_signals("red"),
        skip_dvc=True,
        tracking_uri=registry_uri,
        before_metrics={"wmape": 0.10},
        trainer=lambda ds: _outcome(run_id=candidate_run, lgbm=0.25, prophet=0.40),
        rebuild_bundle=False,
        notifier=lambda title, body: None,
        verbose=False,
    )

    assert not report.promoted
    production = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert production is not None and production.run_id == incumbent_run
    assert temp_store.fetch_retraining_events("synthetic")[0]["promoted"] is False


def test_dry_run_writes_nothing(temp_store, registry_uri):
    from models import registry

    incumbent_run = _log_run(registry_uri, 0.30)
    incumbent = registry.register_model(MODEL_NAME, incumbent_run, tracking_uri=registry_uri)
    registry.transition(
        MODEL_NAME, incumbent.version, registry.STAGE_PRODUCTION, tracking_uri=registry_uri
    )
    candidate_run = _log_run(registry_uri, 0.05)

    report = retrain(
        "synthetic",
        model_name=MODEL_NAME,
        status="red",
        signals=_signals("red"),
        skip_dvc=True,
        dry_run=True,
        tracking_uri=registry_uri,
        before_metrics={"wmape": 0.30},
        trainer=lambda ds: _outcome(run_id=candidate_run),
        notifier=lambda title, body: None,
        verbose=False,
    )

    assert report.decision is not None and report.decision.promote
    assert not report.promoted
    production = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert production is not None and production.run_id == incumbent_run
    assert temp_store.fetch_retraining_events("synthetic") == []


def test_training_without_metrics_is_reported_not_raised(temp_store):
    report = retrain(
        "synthetic",
        status="red",
        signals=_signals("red"),
        skip_dvc=True,
        trainer=lambda ds: TrainingOutcome(),
        notifier=lambda title, body: None,
        verbose=False,
    )
    assert report.triggered and report.best_family is None
    assert "nothing to gate" in report.notes


def test_notification_failure_does_not_break_the_run(temp_store):
    def boom(title: str, body: str):
        raise RuntimeError("webhook down")

    report = retrain(
        "synthetic",
        status="red",
        signals=_signals("red"),
        skip_dvc=True,
        before_metrics={"wmape": 0.30},
        trainer=lambda ds: _outcome(),
        rebuild_bundle=False,
        notifier=boom,
        verbose=False,
    )
    assert report.triggered
    assert any("notification failed" in step for step in report.steps)


def test_report_summary_contains_the_metrics_diff():
    report = RetrainReport(dataset="synthetic", triggered=True)
    report.before_metrics = {"wmape": 0.20}
    report.after_metrics = {"wmape": 0.10}
    summary = report.summary()
    assert "metrics diff" in summary and "-50.00%" in summary


# --- CLI exit codes ---------------------------------------------------------


def _patched_main(monkeypatch, report: RetrainReport) -> int:
    monkeypatch.setattr(retrain_mod, "retrain", lambda **kwargs: report)
    return retrain_mod.main(["--dataset", "synthetic", "--skip-dvc"])


def test_cli_exits_zero_when_nothing_needs_retraining(monkeypatch):
    assert _patched_main(monkeypatch, RetrainReport(dataset="synthetic", triggered=False)) == 0


def test_cli_exits_zero_on_promotion(monkeypatch):
    from models.promote import PromotionDecision

    report = RetrainReport(dataset="synthetic", triggered=True, promoted=True)
    report.decision = PromotionDecision(promote=True, reason="better", promoted=True)
    assert _patched_main(monkeypatch, report) == 0


def test_cli_exits_nonzero_when_the_gate_blocks(monkeypatch):
    from models.promote import PromotionDecision

    report = RetrainReport(dataset="synthetic", triggered=True)
    report.decision = PromotionDecision(promote=False, reason="worse")
    assert _patched_main(monkeypatch, report) == 1


def test_cli_exits_two_on_a_configuration_error(monkeypatch):
    def boom(**kwargs):
        raise FileNotFoundError("no such dataset")

    monkeypatch.setattr(retrain_mod, "retrain", boom)
    assert retrain_mod.main(["--dataset", "nope"]) == 2
