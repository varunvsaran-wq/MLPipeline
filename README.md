# Demand Forecasting MLOps Platform

End-to-end demand/price forecasting with a production-grade MLOps wrapper. The
model is intentionally secondary — the infrastructure (data versioning,
experiment tracking, model registry, serving, drift monitoring, CI/CD, and
automated retraining) is the deliverable.

> **Status:** Phase 0 complete (repo scaffold + tracking stack). See the
> [phase plan](#phase-plan).

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
| 1 | Avocado + Prophet baseline, MLflow run with WMAPE | ⬜ |
| 2 | M5 migration, full feature suite, LightGBM, WMAPE/WRMSSE/pinball | ⬜ |
| 3 | FastAPI serving, Docker, CI | ⬜ |
| 4 | Registry promotion gate, TFT, ONNX | ⬜ |
| 5 | Evidently drift signals + Streamlit ops dashboard | ⬜ |
| 6 | Retraining loop, CD with rollback, A/B shadow router | ⬜ |

Each phase has acceptance criteria in `HANDOFF.md`; work stops for review at
each gate.

## Model leaderboard

_Populated from MLflow once Phase 2 lands (Prophet vs LightGBM vs TFT across
WMAPE / WRMSSE / pinball / bias)._
