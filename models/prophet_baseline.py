"""Phase 1 baseline: a Prophet model on a single Avocado series, tracked in MLflow.

Run it:

    python -m models.prophet_baseline --dataset avocado

Acceptance (HANDOFF Phase 1): a tracked run appears in MLflow with WMAPE logged
and a reproducible Prophet model artifact, tagged with the git commit hash.
"""

from __future__ import annotations

import argparse
import json

import mlflow
import pandas as pd
from prophet import Prophet

from config import DatasetConfig
from data.loader import load_raw, select_series
from features.pipeline import feature_schema, temporal_train_test_split, to_prophet_frame
from models.metrics import evaluate
from models.tracking import git_commit_hash, git_is_dirty, setup_mlflow

EXPERIMENT = "demand-forecasting"


def train(dataset: str = "avocado", *, weekly_seasonality: bool = True) -> dict:
    """Train + evaluate the Prophet baseline and log everything to MLflow.

    Returns the computed metrics dict.
    """
    cfg = DatasetConfig.load(dataset)

    raw = load_raw(cfg)
    series = select_series(cfg, raw)
    frame = to_prophet_frame(cfg, series)
    train_df, test_df = temporal_train_test_split(frame, cfg.horizon)

    setup_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"prophet-{dataset}") as run:
        # Reproducibility tags.
        mlflow.set_tag("git_commit", git_commit_hash())
        mlflow.set_tag("git_dirty", str(git_is_dirty()))
        mlflow.set_tag("model_family", "prophet")
        mlflow.set_tag("dataset", dataset)

        # Params.
        mlflow.log_params(
            {
                "dataset": dataset,
                "series": json.dumps(cfg.default_series),
                "freq": cfg.freq,
                "horizon": cfg.horizon,
                "weekly_seasonality": weekly_seasonality,
                "n_train": len(train_df),
                "n_test": len(test_df),
            }
        )

        # Feature schema (drift contract).
        mlflow.log_dict(feature_schema(frame), "feature_schema.json")

        # Fit on history only.
        model = Prophet(
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        if cfg.holiday_country:
            model.add_country_holidays(country_name=cfg.holiday_country)
        model.fit(train_df)

        # Forecast the held-out horizon and score it.
        future = model.make_future_dataframe(periods=cfg.horizon, freq=cfg.freq)
        forecast = model.predict(future)
        preds = forecast.set_index("ds").loc[test_df["ds"], "yhat"].to_numpy()
        metrics = evaluate(test_df["y"].to_numpy(), preds)
        mlflow.log_metrics(metrics)

        # Persist the model artifact + a human-readable forecast vs actual table.
        mlflow.prophet.log_model(model, artifact_path="model")
        comparison = pd.DataFrame(
            {"ds": test_df["ds"], "y_true": test_df["y"].to_numpy(), "y_pred": preds}
        )
        mlflow.log_text(comparison.to_csv(index=False), "forecast_vs_actual.csv")

        print(
            f"run_id={run.info.run_id}  WMAPE={metrics['wmape']:.4f}  bias={metrics['bias']:+.4f}"
        )
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Prophet baseline and log to MLflow.")
    p.add_argument(
        "--dataset", default="avocado", help="dataset config name under config/datasets/"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.dataset)
