# Demand Forecasting MLOps Platform

End-to-end demand/price forecasting with a production-grade MLOps wrapper. The
model is intentionally secondary — the infrastructure (data versioning,
experiment tracking, model registry, serving, drift monitoring, CI/CD, and
automated retraining) is the deliverable.

> **Status:** Phase 2 complete (full feature suite + global LightGBM vs Prophet,
> WMAPE/WRMSSE/pinball leaderboard). See the [phase plan](#phase-plan).

---

## Architecture (target)

```
            ┌──────────┐   DVC    ┌───────────┐   MLflow   ┌────────────┐
  Raw data ─┤ data/    ├─────────►│ features/ ├───────────►│ models/    │
            └──────────┘ versioned└───────────┘  tracked   └─────┬──────┘
                                                                  │ registry
                                                       ┌──────────▼─────────┐
                                                       │ MLflow Registry    │
                                                       │ None→Staging→Prod  │
                                                       └──────────┬─────────┘
                              ┌───────────────┐                   │ promote
                  requests ──►│ serving/      │◄──────────────────┘
                              │ FastAPI       │
                              └───────┬───────┘
                                      │ logged predictions
                              ┌───────▼───────┐      ┌──────────────┐
                              │ monitoring/   ├─────►│ dashboard/   │
                              │ Evidently     │drift │ Streamlit    │
                              └───────────────┘      └──────────────┘
```

## Tech stack

| Concern | Tool |
|---|---|
| Data versioning | DVC (local remote now; swap to S3/GCS via `dvc remote modify`) |
| Experiment tracking / registry | MLflow (self-hosted, Postgres backend) |
| HPO | Optuna (MLflow child runs) |
| Models | Prophet, LightGBM/XGBoost, TFT (PyTorch Forecasting) |
| Serving | FastAPI + Pydantic v2 |
| Inference format | ONNX (+ native) |
| Containerization | Multi-stage Dockerfile + docker-compose |
| Drift monitoring | Evidently AI |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Backend store | Postgres |
| Deploy target | Render / Fly.io |

## Project layout

```
config/        dataset-agnostic config (config/datasets/*.yaml) + loader
data/          DVC-tracked datasets (raw/ processed/); pointers in git, data in remote
features/      feature engineering pipeline (lags, rolling, calendar, Fourier)
models/        training + evaluation for Prophet / LightGBM / TFT
serving/        FastAPI app (Phase 3)
monitoring/    Evidently drift signals (Phase 5)
dashboard/     Streamlit ops dashboard (Phase 5)
docker/        service Dockerfiles
.github/workflows/  CI/CD pipelines
tests/         pytest unit + integration suite
```

## Quick start (local)

```bash
# 1. Python env
python -m venv .venv
source .venv/Scripts/activate          # Windows; use bin/activate on *nix
pip install -e ".[dev]"                 # add [models] [serving] etc. per phase

# 2. Bring up the tracking stack (MLflow + Postgres)
cp .env.example .env
docker compose up -d
#   MLflow UI → http://localhost:5000

# 3. Data versioning
dvc status                              # green: nothing tracked yet
dvc pull                                # once datasets are added

# 4. Tests
pytest -q
```

## Phase plan

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffold, DVC, MLflow+Postgres compose, empty pytest | ✅ done |
| 1 | Avocado + Prophet baseline, MLflow run with WMAPE | ✅ done |
| 2 | Full feature suite + LightGBM vs Prophet, WMAPE/WRMSSE/pinball | ✅ done¹ |
| 3 | FastAPI serving, Docker, CI | ⬜ |
| 4 | Registry promotion gate, TFT, ONNX | ⬜ |
| 5 | Evidently drift signals + Streamlit ops dashboard | ⬜ |
| 6 | Retraining loop, CD with rollback, A/B shadow router | ⬜ |

Each phase has acceptance criteria in `HANDOFF.md`; work stops for review at
each gate.

¹ Phase 2's feature suite + LightGBM + WMAPE/WRMSSE/pinball are built and run on
the **Avocado panel** (108 series = 54 regions × 2 types). The dataset is
swap-only — point `--dataset m5` at the M5 config once the Kaggle data is
available (`config/datasets/m5.yaml` is wired with the daily feature suite).

## Running the models

```bash
pip install -e ".[dev,models]"     # Prophet, LightGBM, etc.
dvc pull                            # fetch the DVC-tracked avocado.csv

# Phase 1: single-series Prophet baseline
python -m models.prophet_baseline --dataset avocado

# Phase 2: full feature suite + global LightGBM vs Prophet, all metrics
python -m models.run_comparison --dataset avocado
#   -> nested MLflow runs (one per model) + models/leaderboard.md
```

Runs log to a local `mlruns/` store by default. To use the docker-compose MLflow
server instead, `export MLFLOW_TRACKING_URI=http://localhost:5000` first.

**Feature suite** (`config/datasets/*.yaml` → `features:`): target lags, rolling
mean/std/min/max, calendar (month/week/quarter/weekend/holiday), Fourier
seasonality, and lagged exogenous regressors. The global LightGBM forecasts the
horizon **recursively** (each step's p50 is fed back as history) so lag/rolling
features never peek at validation actuals.

## Model leaderboard

Avocado panel, validation = last 12 weekly points held out per series (108
series). Lower is better for all columns except bias (closer to 0). WRMSSE
weights series by dollar volume — an adaptation of the official M5 hierarchy
weights for this dataset. Probabilistic forecasts are p10/p50/p90 (LightGBM
quantile models; Prophet's 80% interval).

| Model | WMAPE | WRMSSE | Pinball p10 | Pinball p50 | Pinball p90 | Bias |
|---|---:|---:|---:|---:|---:|---:|
| **LightGBM** (global) | **0.0848** | **1.0917** | **0.0296** | **0.0571** | 0.0286 | **−0.0123** |
| Prophet (per series) | 0.1580 | 2.6642 | 0.0710 | 0.1064 | 0.0390 | +0.1655 |

The feature-based global model roughly halves WMAPE and more than halves WRMSSE
versus the per-series Prophet baseline. Numbers regenerate from MLflow; see
`models/run_comparison.py` and `models/leaderboard.md`.
