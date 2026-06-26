"""MLflow tracking helpers shared by every trainer.

Reproducibility rule: each run is tagged with the git commit it was produced
from. The tracking URI comes from ``MLFLOW_TRACKING_URI`` so the same code logs
to the docker-compose MLflow server (``http://localhost:5000``) when it is up,
and falls back to a local ``mlruns/`` file store otherwise.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACKING_URI = (PROJECT_ROOT / "mlruns").as_uri()


def git_commit_hash() -> str:
    """Short git commit hash of the working tree, or ``'unknown'`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_is_dirty() -> bool:
    """True if there are uncommitted changes (logged so runs aren't silently irreproducible)."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_mlflow(experiment: str) -> str:
    """Point MLflow at the configured tracking server and select the experiment.

    Returns the resolved tracking URI.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


__all__ = ["git_commit_hash", "git_is_dirty", "setup_mlflow", "DEFAULT_TRACKING_URI"]
