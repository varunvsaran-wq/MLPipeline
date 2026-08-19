# Demand Forecasting MLOps Platform

End-to-end demand/price forecasting with a production-grade MLOps wrapper. The
model is intentionally secondary — the infrastructure (data versioning,
experiment tracking, model registry, serving, drift monitoring, CI/CD, and
automated retraining) is the deliverable.

> **Status:** All six phases complete — data versioning, tracking, a three-family
> model comparison, a served quantile-forecast API, a gated model registry, drift
> monitoring with an ops dashboard, and an automated retrain → promote → deploy
> loop with rollback and A/B shadow traffic. See the [phase plan](#phase-plan).

---

## Architecture

The system is a **closed loop**, not a pipeline: predictions feed monitoring,
monitoring triggers retraining, retraining feeds the registry, and the registry
gates what gets served. That cycle closing is what makes this MLOps rather than
a model in a notebook.

```
                    ┌──────────┐   DVC    ┌───────────┐  MLflow  ┌────────────┐
      Raw data ────►│ data/    ├─────────►│ features/ ├─────────►│ models/    │
        ▲           │ +validate│versioned │ lags/roll │ tracked  │ Prophet    │
        │           └──────────┘          │ cal/four. │          │ LightGBM   │
        │                                 └───────────┘          │ TFT        │
        │                                                        └─────┬──────┘
        │                                          register candidate  │
        │                                              ┌───────────────▼──────┐
        │                                              │ MLflow Registry      │
        │                                              │ None→Staging→Prod    │
        │                                              │      →Archived       │
        │                                              └───────────┬──────────┘
        │                          ┌───────────────────────────────┤
        │                          │ promotion gate (models/promote.py)
        │                          │ promote only if ≥1% better than Production
        │                          │           else exit 1 → CI fails, no deploy
        │                    ┌─────▼─────────┐
        │      requests ────►│ serving/      │──► p10/p50/p90 forecast
        │                    │ FastAPI       │      (shadow model sees ~10% of
        │                    └─────┬─────────┘       traffic; never served)
        │                          │ every prediction logged
        │                    ┌─────▼─────────┐        ┌──────────────┐
        │                    │ monitoring/   ├───────►│ dashboard/   │
        │                    │ PSI · residual│ drift  │ Streamlit    │
        │                    │ rolling WMAPE │ status │ ops view     │
        │                    └─────┬─────────┘        └──────────────┘
        │                          │ breach = red
        │                    ┌─────▼─────────┐
        └────────────────────┤ retrain loop  │  DVC pull → retrain all families
           new model         │ models/       │  → eval on fixed holdout
           (if it beats prod)│ retrain.py    │  → gate → promote → notify
                             └───────────────┘
```

CI/CD wraps the whole thing: `ci.yml` on every PR, `cd.yml` on merge to main
(build → push to GHCR → deploy → health-check ×3 → **rollback on failure**), and
`retrain.yml` on a cron or a drift-breach dispatch.

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
data/          DVC-tracked datasets; loader.py + validation.py (CI data gate)
features/      feature engineering pipeline (lags, rolling, calendar, Fourier)
models/        model families, evaluation, registry + promotion gate, retraining
                 metrics.py evaluate.py            — shared scoring harness
                 prophet_panel.py lightgbm_model.py tft_model.py
                 run_comparison.py                 — leaderboard entry point
                 registry.py promote.py            — registry + gated promotion
                 retrain.py notify.py              — automated retraining loop
                 onnx_export.py                    — ONNX export + latency bench
serving/       FastAPI app, model bundle, predictor, prediction log, shadow router
monitoring/    drift signals (PSI / residual / rolling WMAPE) + event store
dashboard/     Streamlit ops dashboard (data.py = pure logic, app.py = rendering)
scripts/       demo_self_heal.py — the end-to-end drift→retrain→promote demo
docker/        service Dockerfiles (mlflow, serving, dashboard)
.github/workflows/  ci.yml, cd.yml, retrain.yml
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
| 3 | FastAPI serving, Docker, CI | ✅ done |
| 4 | Registry promotion gate, TFT, ONNX | ✅ done² |
| 5 | Drift signals (PSI/residual/rolling WMAPE) + Streamlit ops dashboard | ✅ done |
| 6 | Retraining loop, CD with rollback, A/B shadow router | ✅ done³ |

Each phase has acceptance criteria in `HANDOFF.md`; work stops for review at
each gate.

¹ Phase 2's feature suite + LightGBM + WMAPE/WRMSSE/pinball are built and run on
the **Avocado panel** (108 series = 54 regions × 2 types). The dataset is
swap-only — point `--dataset m5` at the M5 config once the Kaggle data is
available (`config/datasets/m5.yaml` is wired with the daily feature suite).

² TFT is code-complete and wired into the comparison harness, but its training
path needs the heavy `[deep]` extra (torch + pytorch-forecasting) and is
unverified in this environment — it is opt-in via `--with-tft`. ONNX export
proves exact numeric parity on a synthetic panel and is ~56× faster than native
LightGBM at batch-size 1 (the recursive serving regime); on the real Avocado
model the float32 threshold precision of `TreeEnsembleRegressor` breaks parity,
which `models/onnx_export.py` detects and reports rather than shipping silently.

³ The end-to-end self-heal path (`scripts/demo_self_heal.py`) is verified: an
injected demand shock is detected (drift → red), retraining runs, and the new
model is promoted past the gate (production WMAPE 0.37 → 0.11) with the old
version archived. CD/retrain workflows are authored and YAML-validated but,
being GitHub Actions, are not executed locally.

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

## Serving API (Phase 3)

A FastAPI app serves quantile forecasts from a **model bundle** — the three
p10/p50/p90 LightGBM models trained on all history, plus the feature spec,
training categories, raw history, feature schema, and validation metrics, saved
as one artifact under `models/artifacts/<dataset>/`.

```bash
pip install -e ".[serving,models]"

# 1. Build the servable bundle (production models on all history).
python -m serving.model_bundle --dataset avocado

# 2. Run the API (docs at http://localhost:8000/docs).
uvicorn serving.app:app --reload
```

Or bring up the whole stack (MLflow + Postgres + API) in one command:

```bash
python -m serving.model_bundle --dataset avocado   # bundle is mounted into the API
docker compose up -d --build                        # API :8000, MLflow :5000
```

**Endpoints** (OpenAPI docs at `/docs`):

| Method & path | Purpose |
|---|---|
| `POST /forecast` | one series: point forecast (p50) + 80% interval (p10/p90) |
| `POST /forecast/batch` | many series in one call |
| `GET /forecast/history` | logged past predictions for a series, joined with actuals |
| `GET /model/leaderboard` | models ranked by validation WMAPE |
| `GET /model/series` | series ids this model can forecast |
| `GET /health` | liveness probe (200 even before a bundle is loaded) |
| `GET /metrics` | Prometheus scrape (request counts, latency histogram) |

Forecasts for genuinely-future dates are produced **recursively** (each step's
p50 is fed back as history before recomputing lag/rolling features), reusing the
exact training feature pipeline so serving features can't drift from training.
Every served forecast is written to a prediction log (SQLite by default, or
Postgres via `SERVING_DB_URI`) — the substrate Phase 5's drift monitoring reads.

`GET /health` returning 200 and `/docs` rendering are the Phase 3 acceptance
checks; CI additionally builds the serving image and asserts `/health` in the
running container.

## Registry, promotion gate, TFT & ONNX (Phase 4)

**Model registry + gated promotion.** `models/registry.py` wraps the MLflow
registry lifecycle (`None → Staging → Production → Archived`); `models/promote.py`
is the gate. A candidate reaches Production only if it beats the incumbent by a
threshold (default: ≥1% relative improvement on validation WMAPE, lower-is-better).

```bash
# needs a DB-backed tracking store — the compose server or a sqlite:/// URI
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m models.run_comparison --dataset avocado --register-as demand-forecasting-avocado
python -m models.promote --model-name demand-forecasting-avocado --run-id <lgbm_run_id>
#   exit 0 = promoted (or first model), exit 1 = gate blocked → CI fails, no deploy
```

The gate is dataset-agnostic and reads metrics straight off the tracked runs, so
it works identically in CI. The registry **requires** a database-backed store; a
`file://`/`mlruns` store raises a clear `RegistryUnsupportedError`.

**TFT** (`models/tft_model.py`) adds a Temporal Fusion Transformer to the same
`SeriesForecast` harness (quantile output → the same pinball/WRMSSE scoring).
Opt-in — it needs `pip install -e ".[deep]"`:

```bash
python -m models.run_comparison --dataset avocado --with-tft --tft-epochs 5
```

**ONNX export** (`models/onnx_export.py`) converts the LightGBM quantile models to
ONNX, verifies numeric parity against the native booster, and benchmarks
onnxruntime vs native latency:

```bash
python -m models.onnx_export --dataset avocado --batch-size 512 --repeats 50
```

## Drift monitoring & ops dashboard (Phase 5)

`monitoring/drift.py` computes three signals — **input feature drift** (PSI per
lag/rolling feature; red at PSI > 0.2), **residual drift** (distribution of
prediction − actual), and **rolling WMAPE** (breach when 30-day rolling error
exceeds validation WMAPE by a tolerance, the retraining trigger). Signals and
retraining events persist to the same DB as the prediction log.

The **Streamlit ops dashboard** (`dashboard/`, deliberately separate from any
prediction UI) shows the production model, request volume, prediction
histogram, per-feature PSI with red/yellow/green status, residual drift,
the leaderboard, and the last retraining events. `dashboard/data.py` holds the
pure, tested logic; `dashboard/app.py` is the thin rendering shell.

```bash
pip install -e ".[monitoring]"
streamlit run dashboard/app.py          # or: docker compose up dashboard  (→ :8501)
```

## Automated retraining, CD & A/B shadow (Phase 6)

**Retraining loop** (`models/retrain.py`): evaluate drift → (if red) `dvc pull` →
retrain all families → evaluate on the fixed holdout → promotion gate → rebuild
the servable bundle on success → record the event → notify (`models/notify.py`,
Slack via `SLACK_WEBHOOK_URL`, stdout otherwise). It degrades gracefully when no
DB-backed registry is available.

```bash
python -m models.retrain --dataset avocado --force        # run one cycle now
python scripts/demo_self_heal.py --dataset avocado        # the acceptance demo
```

The **self-heal demo** is the executable Phase 6 acceptance criterion and is
hermetic (temp DBs, temp MLflow store, a copied CSV — the real data is never
touched): it injects a demand shock, shows drift go red, retrains, and promotes
the new model end-to-end (production WMAPE ≈ 0.37 → ≈ 0.11, old version archived).

**A/B shadow router** (`serving/shadow.py`): on a sampled fraction of traffic
(`SHADOW_TRAFFIC_PCT`, default 10%, enabled by `SHADOW_ENABLED`) a challenger
model also predicts; both forecasts are logged for comparison and **the shadow
output is never served**. The call is fail-safe — a shadow error can't affect the
live response.

**CI/CD** (`.github/workflows/`): `ci.yml` (lint, data validation, tests, image
smoke test) on every PR; `cd.yml` on merge to main (verify → promotion gate →
build & push to GHCR → deploy to Fly.io → `/health` ×3 → **rollback to the
previous image tag on any failure**); `retrain.yml` on a cron or a drift-breach
`repository_dispatch`. Deploy/registry steps degrade to a clear skip when their
secrets aren't configured.

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
