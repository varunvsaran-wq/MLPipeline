"""Servable model bundle — the artifact the serving layer loads at runtime.

Training (``models/run_comparison.py``) evaluates models on a held-out horizon;
*serving* needs something different: a production model fit on **all** available
history plus everything required to forecast genuinely-future dates for any
series. This module builds and loads that bundle.

A bundle captures, in one joblib file:

* the three quantile LightGBM models (p10/p50/p90) fit on the full history,
* the :class:`~features.pipeline.FeatureSpec` (exact input columns/order),
* the training categories per categorical column — needed so single-series
  inference produces the *same* category codes the global model trained on
  (pandas assigns codes by the categories present, so this must be pinned),
* the raw history frame (so lag/rolling features can be recomputed at serve
  time without a DVC pull),
* the logged feature schema (the drift contract) and validation metrics,
* provenance: dataset name, git commit, trained-at timestamp.

Build one with::

    python -m serving.model_bundle --dataset avocado
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from config import DatasetConfig
from features.pipeline import FeatureSpec, series_key
from models.evaluate import QUANTILES

# NB: training-only imports (mlflow, the trainer, the data loader) are kept
# *inside* ``build_bundle`` so ``load_bundle`` — the serving hot path — pulls in
# only joblib + lightgbm, keeping the API container lean.

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "models" / "artifacts"


@dataclass
class ModelBundle:
    """Everything the serving layer needs to forecast, in one object."""

    dataset: str
    cfg: DatasetConfig
    models: dict[float, LGBMRegressor]
    spec: FeatureSpec
    categories: dict[str, list]
    history: pd.DataFrame
    feature_schema: dict
    metrics: dict
    git_commit: str
    trained_at: str

    @property
    def model_version(self) -> str:
        """Human-readable version stamp (git commit + train date)."""
        return f"{self.dataset}@{self.git_commit}-{self.trained_at[:10]}"

    def series_ids(self) -> list[str]:
        """Sorted list of series identifiers (``series_id_cols`` joined by '|')."""
        ids = series_key(self.cfg, self.history).unique().tolist()
        return sorted(ids)


def bundle_path(dataset: str) -> Path:
    return ARTIFACTS_DIR / dataset / "bundle.joblib"


def build_bundle(dataset: str = "avocado") -> ModelBundle:
    """Train the production quantile models on all history and persist a bundle."""
    from data.loader import load_raw
    from features.pipeline import build_feature_matrix, feature_schema
    from models.evaluate import aggregate
    from models.lightgbm_model import _fit_quantile_models, train_and_forecast
    from models.tracking import git_commit_hash

    cfg = DatasetConfig.load(dataset)
    raw = load_raw(cfg)

    frame, spec = build_feature_matrix(cfg, raw)
    # Production models: fit on every row that has a target (no held-out horizon).
    models = _fit_quantile_models(frame, spec)
    categories = {col: list(frame[col].cat.categories) for col in spec.categorical}
    schema = feature_schema(frame[["ds", "y", *spec.numeric]])

    # Validation metrics come from the honest held-out backtest so the leaderboard
    # endpoint reports generalisation error, not training-set fit.
    val_forecasts, _, _, _ = train_and_forecast(cfg, raw)
    metrics = {k: float(v) for k, v in aggregate(val_forecasts).items()}

    bundle = ModelBundle(
        dataset=dataset,
        cfg=cfg,
        models=models,
        spec=spec,
        categories=categories,
        history=raw,
        feature_schema=schema,
        metrics=metrics,
        git_commit=git_commit_hash(),
        trained_at=datetime.now(UTC).isoformat(),
    )

    path = bundle_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle


def load_bundle(dataset: str = "avocado") -> ModelBundle:
    """Load a previously built bundle, or raise a clear error if it's missing."""
    path = bundle_path(dataset)
    if not path.exists():
        raise FileNotFoundError(
            f"No servable bundle for '{dataset}' at {path}. "
            f"Build one with: python -m serving.model_bundle --dataset {dataset}"
        )
    return joblib.load(path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a servable model bundle (Phase 3).")
    p.add_argument("--dataset", default="avocado")
    return p.parse_args()


if __name__ == "__main__":
    # Re-import under the real package name so the pickled ``ModelBundle`` class
    # is referenced as ``serving.model_bundle.ModelBundle`` rather than
    # ``__main__.ModelBundle`` (which no other process could unpickle).
    from serving.model_bundle import build_bundle as _build_bundle
    from serving.model_bundle import bundle_path as _bundle_path

    args = _parse_args()
    b = _build_bundle(args.dataset)
    print(f"built bundle: {b.model_version}")
    print(f"  path      : {_bundle_path(args.dataset)}")
    print(f"  series    : {len(b.series_ids())}")
    print(f"  features  : {len(b.spec.all)}  (quantiles: {list(QUANTILES)})")
    print(f"  val WMAPE : {b.metrics.get('wmape'):.4f}  WRMSSE: {b.metrics.get('wrmsse'):.4f}")


__all__ = ["ModelBundle", "build_bundle", "load_bundle", "bundle_path", "ARTIFACTS_DIR"]
