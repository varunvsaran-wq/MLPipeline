"""MLflow Model Registry access layer for the promotion gate (Phase 4).

Tracking answers "what did this experiment score?"; the registry answers "which
artifact is *live* right now?". Phase 4 needs the second question answered
reliably from CI, so every registry call in the project goes through this module
rather than reaching for :class:`MlflowClient` ad hoc.

Three design decisions worth stating plainly:

* **A database-backed store is mandatory.** The registry tables only exist in a
  SQL store; the ``mlruns/`` file store that :mod:`models.tracking` falls back to
  cannot serve them, and MLflow's own error for that case is obscure. So
  :func:`resolve_tracking_uri` validates the URI up front and raises
  :class:`RegistryUnsupportedError` with an actionable message (start the compose
  MLflow server, or point at ``sqlite:///...``).
* **Stages, not aliases.** HANDOFF specifies the ``None → Staging → Production →
  Archived`` lifecycle, so we use the stage APIs even though MLflow 2.13+ marks
  them deprecated. The deprecation warnings are suppressed at the single call
  site in :func:`transition` instead of globally, so unrelated warnings still
  surface.
* **Metrics are read back from the source run**, never re-computed or copied onto
  the version. The run is the single source of truth for held-out validation
  scores, which keeps the gate honest: a version can only be compared on numbers
  that were actually logged when it was trained.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

from models.tracking import DEFAULT_TRACKING_URI

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps light CI imports cheap
    from mlflow.entities.model_registry import ModelVersion
    from mlflow.tracking import MlflowClient

STAGE_NONE = "None"
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"
STAGES: tuple[str, ...] = (STAGE_NONE, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED)

# Artifact path used by models/run_comparison.py when logging the LightGBM p50 model.
DEFAULT_ARTIFACT_PATH = "model_p50"

_DB_SCHEMES = ("sqlite", "postgresql", "postgres", "mysql", "mssql")
_SERVER_SCHEMES = ("http", "https")


class RegistryUnsupportedError(RuntimeError):
    """Raised when the tracking URI cannot back a Model Registry."""


def resolve_tracking_uri(tracking_uri: str | None = None) -> str:
    """Return a tracking URI that can host the Model Registry, or explain why it can't.

    Resolution order: explicit argument, ``MLFLOW_TRACKING_URI``, then the
    project default. A file store (the default) is rejected, because the registry
    needs SQL tables.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    scheme = uri.split(":", 1)[0].lower()
    if scheme in _SERVER_SCHEMES or scheme in _DB_SCHEMES:
        return uri
    raise RegistryUnsupportedError(
        f"MLflow tracking URI {uri!r} is a file store and cannot back the Model Registry. "
        "Use the compose server (MLFLOW_TRACKING_URI=http://localhost:5000, "
        "`docker compose up mlflow`) or a database store "
        "(MLFLOW_TRACKING_URI=sqlite:///mlflow.db)."
    )


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    """Build an :class:`MlflowClient` bound to a validated, registry-capable URI."""
    from mlflow.tracking import MlflowClient

    uri = resolve_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def ensure_registered_model(client: MlflowClient, name: str) -> None:
    """Create the registered model if it doesn't exist yet (idempotent)."""
    from mlflow.exceptions import MlflowException

    try:
        client.get_registered_model(name)
    except MlflowException:
        try:
            client.create_registered_model(name)
        except MlflowException:  # pragma: no cover - lost a create race, fine either way
            client.get_registered_model(name)


def register_model(
    name: str,
    run_id: str,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    client: MlflowClient | None = None,
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
) -> ModelVersion:
    """Register the model logged by ``run_id`` as a new version of ``name``.

    The registered model is created on first use. The new version lands in stage
    ``None``; promotion is a separate, gated decision (see :mod:`models.promote`).
    """
    client = client or get_client(tracking_uri)
    ensure_registered_model(client, name)
    source = f"runs:/{run_id}/{artifact_path.strip('/')}"
    return client.create_model_version(name=name, source=source, run_id=run_id, tags=tags or {})


def get_versions(name: str, client: MlflowClient | None = None, tracking_uri: str | None = None):
    """All versions of ``name``, newest version number first."""
    client = client or get_client(tracking_uri)
    versions = client.search_model_versions(f"name='{name}'")
    return sorted(versions, key=lambda v: int(v.version), reverse=True)


def get_version(
    name: str,
    version: str | int,
    client: MlflowClient | None = None,
    tracking_uri: str | None = None,
) -> ModelVersion:
    """One specific model version."""
    client = client or get_client(tracking_uri)
    return client.get_model_version(name, str(version))


def get_stage_version(
    name: str,
    stage: str,
    client: MlflowClient | None = None,
    tracking_uri: str | None = None,
) -> ModelVersion | None:
    """Highest-numbered version currently in ``stage``, or ``None`` if the stage is empty."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    client = client or get_client(tracking_uri)
    in_stage = [v for v in get_versions(name, client=client) if v.current_stage == stage]
    return in_stage[0] if in_stage else None


def get_production_version(
    name: str, client: MlflowClient | None = None, tracking_uri: str | None = None
) -> ModelVersion | None:
    """Current Production version of ``name``, or ``None`` when nothing is live yet."""
    from mlflow.exceptions import MlflowException

    client = client or get_client(tracking_uri)
    try:
        return get_stage_version(name, STAGE_PRODUCTION, client=client)
    except MlflowException:
        # Registered model doesn't exist yet — that is simply "no incumbent".
        return None


def version_metrics(
    version: ModelVersion, client: MlflowClient | None = None, tracking_uri: str | None = None
) -> dict[str, float]:
    """Validation metrics logged by the run that produced ``version``.

    Returns an empty mapping when the version has no source run (hand-registered
    artifacts), which the gate treats as "not comparable".
    """
    if not version.run_id:
        return {}
    client = client or get_client(tracking_uri)
    run = client.get_run(version.run_id)
    return {k: float(v) for k, v in run.data.metrics.items()}


def run_metrics(
    run_id: str, client: MlflowClient | None = None, tracking_uri: str | None = None
) -> dict[str, float]:
    """Metrics logged by a tracking run, by name."""
    client = client or get_client(tracking_uri)
    run = client.get_run(run_id)
    return {k: float(v) for k, v in run.data.metrics.items()}


def transition(
    name: str,
    version: str | int,
    stage: str,
    archive_existing: bool = False,
    client: MlflowClient | None = None,
    tracking_uri: str | None = None,
    description: str | None = None,
) -> ModelVersion:
    """Move a version through the lifecycle.

    ``archive_existing=True`` archives whatever else occupies ``stage`` — that is
    how a promotion retires the outgoing Production model in one atomic step.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    client = client or get_client(tracking_uri)
    with warnings.catch_warnings():
        # Stage APIs are deprecated in MLflow 2.13+, but HANDOFF specifies the
        # None -> Staging -> Production -> Archived lifecycle. Silence only here.
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        moved = client.transition_model_version_stage(
            name=name,
            version=str(version),
            stage=stage,
            archive_existing_versions=archive_existing,
        )
    if description:
        client.update_model_version(name=name, version=str(version), description=description)
    return moved


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "STAGES",
    "STAGE_ARCHIVED",
    "STAGE_NONE",
    "STAGE_PRODUCTION",
    "STAGE_STAGING",
    "RegistryUnsupportedError",
    "ensure_registered_model",
    "get_client",
    "get_production_version",
    "get_stage_version",
    "get_version",
    "get_versions",
    "register_model",
    "resolve_tracking_uri",
    "run_metrics",
    "transition",
    "version_metrics",
]
