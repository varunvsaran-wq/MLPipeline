"""Phase 4 tests: the promotion decision rule and the real registry lifecycle.

The decision tests are pure unit tests over :func:`should_promote`. The registry
tests are self-contained too: they point MLflow at a throwaway ``sqlite://`` DB in
``tmp_path``, which is the cheapest store that can actually host the Model
Registry, so the ``None → Staging → Production → Archived`` transitions exercised
here are the real ones — no mocks and no docker-compose server.

The acceptance criterion from HANDOFF Phase 4 is covered explicitly by
``test_gate_blocks_worse_candidate`` and ``test_gate_promotes_better_candidate``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from models import promote as promote_mod
from models import registry
from models.promote import DEFAULT_THRESHOLD, run_gate, should_promote

MODEL_NAME = "demand-forecasting-test"

# --- decision rule (pure, always runs) -------------------------------------


def test_promotes_when_relative_improvement_meets_threshold():
    ok, reason = should_promote(0.90, 1.00, threshold=0.01)
    assert ok
    assert "10.00%" in reason


def test_blocks_when_improvement_below_threshold():
    ok, reason = should_promote(0.995, 1.00, threshold=0.01)
    assert not ok
    assert "does not beat" in reason


def test_blocks_a_strictly_worse_candidate():
    ok, reason = should_promote(1.20, 1.00, threshold=DEFAULT_THRESHOLD)
    assert not ok
    assert "-20.00%" in reason


def test_threshold_boundary_is_inclusive():
    assert should_promote(0.99, 1.00, threshold=0.01)[0]


def test_wmape_is_lower_is_better_by_default():
    # A larger WMAPE must never promote under the default direction.
    assert not should_promote(0.30, 0.20)[0]
    assert should_promote(0.10, 0.20)[0]


def test_higher_is_better_flips_the_comparison():
    assert should_promote(0.90, 0.80, higher_is_better=True)[0]
    assert not should_promote(0.70, 0.80, higher_is_better=True)[0]


def test_no_incumbent_promotes_automatically():
    ok, reason = should_promote(5.0, None)
    assert ok
    assert "no incumbent" in reason


def test_missing_or_nonfinite_candidate_metric_fails_the_gate():
    assert not should_promote(None, 1.0)[0]
    assert not should_promote(math.nan, 1.0)[0]
    assert not should_promote(math.inf, 1.0)[0]


def test_near_zero_incumbent_falls_back_to_absolute_improvement():
    ok, reason = should_promote(-0.05, 0.0, threshold=0.01)
    assert ok
    assert "absolute comparison" in reason
    assert not should_promote(0.005, 0.0, threshold=0.01)[0]


# --- tracking URI validation ------------------------------------------------


def test_file_store_is_rejected_with_actionable_message():
    with pytest.raises(registry.RegistryUnsupportedError) as exc:
        registry.resolve_tracking_uri("file:///c:/tmp/mlruns")
    msg = str(exc.value)
    assert "sqlite" in msg and "5000" in msg


def test_db_and_server_uris_are_accepted():
    assert registry.resolve_tracking_uri("sqlite:///x.db") == "sqlite:///x.db"
    assert registry.resolve_tracking_uri("http://localhost:5000") == "http://localhost:5000"


def test_env_var_is_used_when_no_uri_passed(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///env.db")
    assert registry.resolve_tracking_uri() == "sqlite:///env.db"


# --- real registry lifecycle against a temp sqlite store --------------------


@pytest.fixture()
def registry_uri(tmp_path: Path, monkeypatch) -> str:
    """A throwaway SQLite tracking+registry store with artifacts inside tmp_path."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    exp_id = mlflow.create_experiment(
        "phase4-promotion", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    monkeypatch.setenv("_PHASE4_EXPERIMENT_ID", exp_id)
    return uri


def _log_candidate_run(uri: str, wmape: float) -> str:
    """Log a run carrying a validation WMAPE plus a stand-in model artifact."""
    import os

    import mlflow

    mlflow.set_tracking_uri(uri)
    with mlflow.start_run(experiment_id=os.environ["_PHASE4_EXPERIMENT_ID"]) as run:
        mlflow.log_metric("wmape", wmape)
        mlflow.log_metric("wrmsse", wmape * 2)
        mlflow.log_text("stand-in for the logged model", "model_p50/MLmodel")
        return run.info.run_id


def test_register_and_stage_lifecycle(registry_uri):
    run_id = _log_candidate_run(registry_uri, 0.30)
    client = registry.get_client(registry_uri)

    mv = registry.register_model(MODEL_NAME, run_id, client=client)
    assert mv.current_stage == registry.STAGE_NONE
    assert registry.get_production_version(MODEL_NAME, client=client) is None

    registry.transition(MODEL_NAME, mv.version, registry.STAGE_STAGING, client=client)
    staged = registry.get_stage_version(MODEL_NAME, registry.STAGE_STAGING, client=client)
    assert staged is not None and staged.version == mv.version

    registry.transition(MODEL_NAME, mv.version, registry.STAGE_PRODUCTION, client=client)
    prod = registry.get_production_version(MODEL_NAME, client=client)
    assert prod is not None and prod.version == mv.version
    assert registry.version_metrics(prod, client=client)["wmape"] == pytest.approx(0.30)

    registry.transition(MODEL_NAME, mv.version, registry.STAGE_ARCHIVED, client=client)
    assert registry.get_production_version(MODEL_NAME, client=client) is None


def test_transition_rejects_unknown_stage(registry_uri):
    with pytest.raises(ValueError):
        registry.transition(MODEL_NAME, 1, "Live", tracking_uri=registry_uri)


def test_first_model_promotes_automatically(registry_uri):
    run_id = _log_candidate_run(registry_uri, 0.40)
    decision = run_gate(MODEL_NAME, run_id=run_id, tracking_uri=registry_uri)
    assert decision.promote and decision.promoted
    assert decision.production_metric is None
    prod = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert prod is not None and prod.version == decision.candidate_version


def test_gate_blocks_worse_candidate(registry_uri):
    """HANDOFF acceptance: a worse candidate fails the gate and Production is untouched."""
    incumbent_run = _log_candidate_run(registry_uri, 0.20)
    run_gate(MODEL_NAME, run_id=incumbent_run, tracking_uri=registry_uri)
    incumbent = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert incumbent is not None

    worse_run = _log_candidate_run(registry_uri, 0.35)
    decision = run_gate(MODEL_NAME, run_id=worse_run, tracking_uri=registry_uri)
    assert not decision.promote and not decision.promoted

    still_prod = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert still_prod is not None and still_prod.version == incumbent.version
    # The blocked candidate is registered but parked in Staging, never Production.
    blocked = registry.get_version(
        MODEL_NAME, decision.candidate_version, tracking_uri=registry_uri
    )
    assert blocked.current_stage == registry.STAGE_STAGING

    # CI contract: the gate exits non-zero so the Actions run fails.
    code = promote_mod.main(
        ["--model-name", MODEL_NAME, "--run-id", worse_run, "--tracking-uri", registry_uri]
    )
    assert code == 1


def test_gate_promotes_better_candidate_and_archives_incumbent(registry_uri):
    """HANDOFF acceptance: a better candidate promotes and the old Production is archived."""
    incumbent_run = _log_candidate_run(registry_uri, 0.30)
    run_gate(MODEL_NAME, run_id=incumbent_run, tracking_uri=registry_uri)
    incumbent = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert incumbent is not None

    better_run = _log_candidate_run(registry_uri, 0.21)  # 30% better
    decision = run_gate(MODEL_NAME, run_id=better_run, tracking_uri=registry_uri)
    assert decision.promote and decision.promoted
    assert decision.archived_versions == [incumbent.version]

    prod = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert prod is not None and prod.version == decision.candidate_version
    old = registry.get_version(MODEL_NAME, incumbent.version, tracking_uri=registry_uri)
    assert old.current_stage == registry.STAGE_ARCHIVED

    # CI contract: a passing gate exits 0 so the deploy proceeds.
    best_run = _log_candidate_run(registry_uri, 0.10)
    code = promote_mod.main(
        ["--model-name", MODEL_NAME, "--run-id", best_run, "--tracking-uri", registry_uri]
    )
    assert code == 0
    assert (
        registry.get_version(
            MODEL_NAME, decision.candidate_version, tracking_uri=registry_uri
        ).current_stage
        == registry.STAGE_ARCHIVED
    )


def test_marginal_improvement_is_blocked_by_threshold(registry_uri):
    incumbent_run = _log_candidate_run(registry_uri, 0.30)
    run_gate(MODEL_NAME, run_id=incumbent_run, tracking_uri=registry_uri)
    marginal_run = _log_candidate_run(registry_uri, 0.2995)  # ~0.17% better
    decision = run_gate(MODEL_NAME, run_id=marginal_run, tracking_uri=registry_uri, threshold=0.01)
    assert not decision.promote


def test_dry_run_writes_nothing(registry_uri):
    incumbent_run = _log_candidate_run(registry_uri, 0.30)
    run_gate(MODEL_NAME, run_id=incumbent_run, tracking_uri=registry_uri)
    before = registry.get_versions(MODEL_NAME, tracking_uri=registry_uri)

    better_run = _log_candidate_run(registry_uri, 0.10)
    decision = run_gate(MODEL_NAME, run_id=better_run, tracking_uri=registry_uri, dry_run=True)
    assert decision.promote and not decision.promoted
    assert decision.candidate_version is None

    after = registry.get_versions(MODEL_NAME, tracking_uri=registry_uri)
    assert len(after) == len(before)
    prod = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert prod is not None and prod.run_id == incumbent_run


def test_run_gate_requires_a_candidate_identifier(registry_uri):
    with pytest.raises(ValueError):
        run_gate(MODEL_NAME, tracking_uri=registry_uri)


def test_gate_by_existing_version(registry_uri):
    incumbent_run = _log_candidate_run(registry_uri, 0.30)
    run_gate(MODEL_NAME, run_id=incumbent_run, tracking_uri=registry_uri)
    better_run = _log_candidate_run(registry_uri, 0.15)
    mv = registry.register_model(MODEL_NAME, better_run, tracking_uri=registry_uri)

    decision = run_gate(MODEL_NAME, version=mv.version, tracking_uri=registry_uri)
    assert decision.promote and decision.promoted
    prod = registry.get_production_version(MODEL_NAME, tracking_uri=registry_uri)
    assert prod is not None and prod.version == mv.version


def test_same_run_as_production_is_a_no_op(registry_uri):
    run_id = _log_candidate_run(registry_uri, 0.30)
    run_gate(MODEL_NAME, run_id=run_id, tracking_uri=registry_uri)
    decision = run_gate(MODEL_NAME, run_id=run_id, tracking_uri=registry_uri, dry_run=True)
    assert not decision.promote
    assert "already the Production model" in decision.reason
