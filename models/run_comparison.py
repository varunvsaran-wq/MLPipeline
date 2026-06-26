"""Phase 2 entry point: train Prophet + LightGBM on a dataset and compare them.

    python -m models.run_comparison --dataset avocado

Logs one MLflow run per model family (nested under a parent comparison run) with
all metrics — WMAPE, WRMSSE, pinball p10/p50/p90, bias — plus the LightGBM model
artifact and the feature schema. Prints and returns the leaderboard, and writes
it to ``models/leaderboard.md`` for the README.

Acceptance (HANDOFF Phase 2): an MLflow comparison table shows Prophet vs
LightGBM across all metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from config import DatasetConfig
from data.loader import load_raw
from features.pipeline import build_feature_matrix, feature_schema
from models.evaluate import aggregate
from models.tracking import git_commit_hash, git_is_dirty, setup_mlflow

EXPERIMENT = "demand-forecasting"
LEADERBOARD_PATH = Path(__file__).resolve().parent / "leaderboard.md"
_METRIC_ORDER = ["wmape", "wrmsse", "pinball_p10", "pinball_p50", "pinball_p90", "bias", "n_series"]


def _common_tags(dataset: str, family: str) -> None:
    mlflow.set_tag("git_commit", git_commit_hash())
    mlflow.set_tag("git_dirty", str(git_is_dirty()))
    mlflow.set_tag("dataset", dataset)
    mlflow.set_tag("model_family", family)


def _leaderboard_markdown(dataset: str, metrics_by_model: dict[str, dict]) -> str:
    cols = [m for m in _METRIC_ORDER if m != "n_series"]
    header = "| Model | " + " | ".join(c.upper() for c in cols) + " |"
    sep = "|---|" + "|".join(["---:"] * len(cols)) + "|"
    lines = [
        f"Leaderboard — {dataset} (validation: last horizon held out per series)",
        "",
        header,
        sep,
    ]
    for model, m in metrics_by_model.items():
        cells = [f"{m[c]:.4f}" if c in m else "—" for c in cols]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run(dataset: str = "avocado", max_series: int | None = None) -> dict[str, dict]:
    cfg = DatasetConfig.load(dataset)
    raw = load_raw(cfg)

    # Feature schema (drift contract) logged once for the dataset.
    frame, spec = build_feature_matrix(cfg, raw)
    schema = feature_schema(frame[["ds", "y", *spec.numeric]])

    # Train both families up front (heavy imports kept local).
    from models.lightgbm_model import train_and_forecast
    from models.prophet_panel import forecast_panel

    lgbm_forecasts, lgbm_models, _, lgbm_info = train_and_forecast(cfg, raw)
    prophet_forecasts = forecast_panel(cfg, raw, max_series=max_series)

    metrics_by_model = {
        "LightGBM": aggregate(lgbm_forecasts),
        "Prophet": aggregate(prophet_forecasts),
    }

    setup_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"compare-{dataset}") as parent:
        mlflow.set_tag("git_commit", git_commit_hash())
        mlflow.set_tag("dataset", dataset)
        mlflow.set_tag("phase", "2")
        mlflow.log_dict(schema, "feature_schema.json")
        mlflow.log_param("n_features", len(spec.all))
        mlflow.log_param("horizon", cfg.horizon)

        # LightGBM child run.
        with mlflow.start_run(run_name=f"lightgbm-{dataset}", nested=True):
            _common_tags(dataset, "lightgbm")
            mlflow.log_params({"n_features": len(spec.all), **lgbm_info})
            mlflow.log_metrics(metrics_by_model["LightGBM"])
            mlflow.lightgbm.log_model(lgbm_models[0.5], artifact_path="model_p50")

        # Prophet child run.
        with mlflow.start_run(run_name=f"prophet-{dataset}", nested=True):
            _common_tags(dataset, "prophet")
            mlflow.log_params({"n_series": len(prophet_forecasts), "interval_width": 0.8})
            mlflow.log_metrics(metrics_by_model["Prophet"])

        # Comparison artifact on the parent.
        table = _leaderboard_markdown(dataset, metrics_by_model)
        mlflow.log_text(table, "leaderboard.md")
        mlflow.log_dict(
            {k: {mk: float(mv) for mk, mv in v.items()} for k, v in metrics_by_model.items()},
            "comparison.json",
        )
        parent_id = parent.info.run_id

    LEADERBOARD_PATH.write_text(table, encoding="utf-8")
    print(table)
    print(f"parent_run_id={parent_id}")
    print(
        "metrics:",
        json.dumps(
            {k: {m: round(x, 4) for m, x in v.items()} for k, v in metrics_by_model.items()},
            indent=2,
        ),
    )
    return metrics_by_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prophet vs LightGBM comparison (Phase 2).")
    p.add_argument("--dataset", default="avocado")
    p.add_argument(
        "--max-series", type=int, default=None, help="cap Prophet series for a quick run"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.dataset, max_series=args.max_series)
