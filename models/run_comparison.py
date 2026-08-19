"""Comparison entry point: train the model families on a dataset and rank them.

    python -m models.run_comparison --dataset avocado
    python -m models.run_comparison --dataset avocado --with-tft --tft-epochs 5

Logs one MLflow run per model family (nested under a parent comparison run) with
all metrics — WMAPE, WRMSSE, pinball p10/p50/p90, bias — plus the model
artifacts and the feature schema. Prints and returns the leaderboard, writing it
to ``models/leaderboard.md`` (for the README) and ``models/leaderboard.json``
(which the serving layer's ``/model/leaderboard`` reads).

Prophet and LightGBM always run. The TFT (Phase 4) is **opt-in** via
``--with-tft``: it needs the heavy ``[deep]`` extra and is far slower to train,
so the default comparison stays runnable without torch installed.

Acceptance (HANDOFF Phase 2): an MLflow comparison table shows Prophet vs
LightGBM across all metrics. Phase 4 adds TFT to that same table.
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
# Machine-readable leaderboard the serving layer's /model/leaderboard reads.
LEADERBOARD_JSON = Path(__file__).resolve().parent / "leaderboard.json"
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


def run(
    dataset: str = "avocado",
    max_series: int | None = None,
    with_tft: bool = False,
    tft_epochs: int = 5,
    register_as: str | None = None,
) -> dict[str, dict]:
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

    # TFT is opt-in: it needs the [deep] extra and is far slower to train, so the
    # default comparison stays runnable without torch.
    tft_info: dict | None = None
    tft_model = None
    if with_tft:
        from models.tft_model import train_and_forecast as tft_train_and_forecast

        tft_forecasts, tft_model, tft_info = tft_train_and_forecast(cfg, raw, max_epochs=tft_epochs)
        metrics_by_model["TFT"] = aggregate(tft_forecasts)

    setup_mlflow(EXPERIMENT)
    with mlflow.start_run(run_name=f"compare-{dataset}") as parent:
        mlflow.set_tag("git_commit", git_commit_hash())
        mlflow.set_tag("dataset", dataset)
        mlflow.set_tag("phase", "2")
        mlflow.log_dict(schema, "feature_schema.json")
        mlflow.log_param("n_features", len(spec.all))
        mlflow.log_param("horizon", cfg.horizon)

        # LightGBM child run. Its run id is the promotion gate's candidate handle,
        # so we capture and surface it (see models/promote.py).
        with mlflow.start_run(run_name=f"lightgbm-{dataset}", nested=True) as lgbm_run:
            _common_tags(dataset, "lightgbm")
            mlflow.log_params({"n_features": len(spec.all), **lgbm_info})
            mlflow.log_metrics(metrics_by_model["LightGBM"])
            mlflow.lightgbm.log_model(lgbm_models[0.5], artifact_path="model_p50")
            lgbm_run_id = lgbm_run.info.run_id

        # Prophet child run.
        with mlflow.start_run(run_name=f"prophet-{dataset}", nested=True):
            _common_tags(dataset, "prophet")
            mlflow.log_params({"n_series": len(prophet_forecasts), "interval_width": 0.8})
            mlflow.log_metrics(metrics_by_model["Prophet"])

        # TFT child run (only when trained).
        if tft_info is not None:
            with mlflow.start_run(run_name=f"tft-{dataset}", nested=True):
                _common_tags(dataset, "tft")
                mlflow.log_params(tft_info)
                mlflow.log_metrics(metrics_by_model["TFT"])
                try:
                    # Aliased import: `import mlflow.pytorch` would rebind the name
                    # `mlflow` as a local and shadow the module-level import.
                    import mlflow.pytorch as mlflow_pytorch

                    mlflow_pytorch.log_model(tft_model, artifact_path="model_tft")
                except Exception as exc:  # noqa: BLE001 - artifact logging is best-effort
                    mlflow.set_tag("model_log_error", str(exc)[:250])

        # Comparison artifact on the parent.
        table = _leaderboard_markdown(dataset, metrics_by_model)
        mlflow.log_text(table, "leaderboard.md")
        mlflow.log_dict(
            {k: {mk: float(mv) for mk, mv in v.items()} for k, v in metrics_by_model.items()},
            "comparison.json",
        )
        parent_id = parent.info.run_id

    LEADERBOARD_PATH.write_text(table, encoding="utf-8")
    LEADERBOARD_JSON.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "models": {
                    k: {mk: float(mv) for mk, mv in v.items()} for k, v in metrics_by_model.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Optionally register the LightGBM run as a new model version. Registration is
    # deliberately separate from *promotion*: this only creates the candidate, the
    # gate in models/promote.py decides whether it reaches Production.
    if register_as:
        from models.registry import RegistryUnsupportedError, register_model

        try:
            mv = register_model(register_as, lgbm_run_id, artifact_path="model_p50")
            print(f"registered {register_as} version {mv.version} from run {lgbm_run_id}")
        except RegistryUnsupportedError as exc:
            # A file:// tracking store has no registry — say so instead of crashing.
            print(f"registration skipped: {exc}")

    print(table)
    print(f"parent_run_id={parent_id}")
    print(f"lightgbm_run_id={lgbm_run_id}")
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
    p.add_argument(
        "--with-tft",
        action="store_true",
        help="also train the TFT (Phase 4; needs the [deep] extra and is slow)",
    )
    p.add_argument("--tft-epochs", type=int, default=5, help="max epochs for the TFT")
    p.add_argument(
        "--register-as",
        default=None,
        metavar="MODEL_NAME",
        help="register the LightGBM run as a new version of this registered model "
        "(needs a DB-backed MLFLOW_TRACKING_URI; promotion is still gated)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        args.dataset,
        max_series=args.max_series,
        with_tft=args.with_tft,
        tft_epochs=args.tft_epochs,
        register_as=args.register_as,
    )
