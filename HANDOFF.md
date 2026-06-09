# HANDOFF.md — Demand Forecasting MLOps Platform

**Audience:** autonomous coding agent (Claude Code, Cline, etc.)
**Goal:** Build an end-to-end demand/price forecasting system with a production-grade MLOps wrapper. The model is intentionally secondary — the infrastructure (versioning, tracking, registry, serving, drift monitoring, CI/CD, retraining) is the deliverable.

---

## 0. Operating principles

- **Build the plumbing before optimizing the model.** A working skeleton (data → train → track → serve) beats a high-accuracy model with no infrastructure.
- **Every artifact must be reproducible.** Data versioned with DVC, runs tracked in MLflow, each run tagged with its git commit hash.
- **Ship in strict phase order (below).** Do not start a phase until the prior phase's acceptance criteria pass.
- **Domain-correct metrics only.** Use WMAPE / WRMSSE / pinball loss, not bare RMSE.
- After each phase, run its acceptance checks and stop for human review before continuing.

---

## 1. Tech stack (fixed — do not substitute without flagging)

| Concern | Tool |
|---|---|
| Data versioning | DVC (remote → S3 or GCS bucket) |
| Experiment tracking | MLflow (self-hosted) |
| Model registry | MLflow registry (None → Staging → Production → Archived) |
| HPO (optional) | Optuna, logged as MLflow child runs |
| Models | Prophet, LightGBM/XGBoost, TFT (PyTorch Forecasting) |
| Serving | FastAPI + Pydantic v2 |
| Inference format | ONNX (in addition to native) where applicable |
| Containerization | Multi-stage Dockerfile + docker-compose |
| Drift monitoring | Evidently AI |
| Dashboard | Streamlit (ops dashboard, separate from any prediction UI) |
| CI/CD | GitHub Actions |
| Backend store | Postgres (via docker-compose) |
| Deploy target | Render or Fly.io |

---

## 2. Dataset

Start small, then migrate for the demo.

1. **Phase 1 prototype:** Avocado prices (Kaggle) — small, fast iteration.
2. **Demo target:** M5 Forecasting / Walmart sales (Kaggle) — hierarchical, multi-store, industry-standard benchmark.

Alternatives if needed: Rossmann Store Sales, NYC Taxi demand, ENTSO-E/EIA electricity price/load.

Keep the pipeline dataset-agnostic so the Avocado → M5 migration is a config change, not a rewrite.

---

## 3. Feature engineering (the differentiator)

Do not feed raw target-only series. Build a rich feature set:

- **Lag features:** t-1, t-7, t-14, t-28.
- **Rolling stats:** 7-day mean/std/min/max; 28-day mean.
- **Calendar:** day of week, month, week of year, is_weekend, is_holiday (use the `holidays` library).
- **Fourier terms:** sin/cos encodings for weekly + yearly seasonality.
- **Lagged external regressors:** promotions, events, weather if available.

Version all features with DVC. **Log the feature schema (column names, dtypes, expected ranges) to MLflow** so schema drift can be detected at inference.

---

## 4. Models & evaluation

Train and compare three families; this comparison is the ablation story:

1. **Prophet** — interpretable baseline, native holiday handling.
2. **LightGBM / XGBoost** — tabular feature-based, usually best on M5-type data.
3. **TFT (Temporal Fusion Transformer)** via PyTorch Forecasting — depth + uncertainty estimates.

**Metrics — log all per model per run in MLflow:**

- WMAPE (Weighted Mean Absolute Percentage Error)
- WRMSSE (M5 official metric)
- Pinball loss at 10th / 50th / 90th percentiles (probabilistic forecasting)
- Forecast bias (mean signed error)

Produce a leaderboard comparison table in the README.

---

## 5. Phase plan & acceptance criteria

### Phase 0 — Repo scaffold
- Git repo, Python env (pyproject/requirements), pre-commit, project structure (`data/`, `features/`, `models/`, `serving/`, `monitoring/`, `dashboard/`, `.github/workflows/`).
- DVC initialized with remote configured.
- MLflow server running locally via docker-compose (MLflow + Postgres).
- **Accept:** `docker-compose up` starts MLflow UI; `dvc status` works; empty pytest suite runs green.

### Phase 1 — Baseline end-to-end (Avocado + Prophet)
- Load Avocado dataset, DVC-track it.
- Minimal feature pipeline + Prophet baseline.
- MLflow logs params, metrics, model artifact, git commit hash.
- **Accept:** a tracked run appears in MLflow with WMAPE logged and a reproducible model artifact.

### Phase 2 — Full features + LightGBM on M5
- Migrate config to M5 dataset.
- Implement full feature suite (lags, rolling, Fourier, calendar) + schema logging.
- Train LightGBM; implement WMAPE + WRMSSE + pinball loss.
- **Accept:** MLflow comparison table shows Prophet vs LightGBM across all metrics.

### Phase 3 — Serving layer
FastAPI endpoints (Pydantic v2 schemas, auto OpenAPI docs at `/docs`):

```
POST /forecast            — item_id + horizon → point forecast + prediction interval
POST /forecast/batch      — multiple items
GET  /forecast/history    — past predictions vs actuals for an item
GET  /model/leaderboard   — registered models ranked by validation WMAPE
GET  /health              — liveness probe
GET  /metrics             — Prometheus-format scrape
```
- Multi-stage Dockerfile (lean final image ~200MB).
- docker-compose spins up API + MLflow + Postgres in one command.
- pytest suite: unit (feature transforms, schemas), integration (load model, run 10 forecasts, assert output schema).
- GitHub Actions `ci.yml` on every PR: pytest, data validation (Great Expectations or custom range assertions), model integration test, Docker build smoke test (`/health`).
- **Accept:** `/docs` renders; CI passes on a PR; `/health` returns 200 in the built container.

### Phase 4 — Registry + promotion + TFT
- MLflow registry with None → Staging → Production → Archived flow.
- Promotion script: only promote to Production if the new model beats current Production on held-out validation by ≥ a set threshold; gate in CI (failing model → failed Actions run, no deploy).
- Add TFT model to the comparison.
- ONNX export where applicable; benchmark latency vs native via onnxruntime.
- **Accept:** running promotion against a worse model fails the gate; against a better model promotes it.

### Phase 5 — Drift monitoring + dashboard
Implement three drift signals with Evidently:

1. **Input feature drift** — rolling distribution of lag/calendar features (PSI; flag feature with PSI > 0.2).
2. **Residual drift** — distribution of (prediction − actual) over time; shift signals degradation.
3. **Rolling WMAPE** — 30-day rolling window; threshold breach (e.g. 20% worse than validation WMAPE) is the retraining trigger.

Streamlit ops dashboard (separate from prediction UI):
- Current production model version, training date, validation metrics.
- Live request volume (24h, 7d).
- Prediction distribution histogram (hourly from logged requests).
- PSI drift scores per feature with red/yellow/green status.
- Residual drift time-series; model leaderboard.
- Last 5 retraining events with before/after metrics.
- **Accept:** injecting shifted feature data turns drift status red in the dashboard.

### Phase 6 — Retraining loop + CD + A/B
- `cd.yml` on merge to main: full CI → promotion check → Docker build + push to GHCR → deploy to Render/Fly.io → post-deploy `/health` x3 → **rollback to previous pinned image tag if any check fails.**
- Retraining workflow triggered by drift breach: DVC pull → re-run features → retrain all three families → evaluate on fixed holdout → compare vs production → promote if beats threshold → integration tests on staging → promote to production, archive old → Slack/email notification with metrics diff. (May be cron + synthetic drift injection rather than live.)
- A/B shadow router: per request, roll random number; if below threshold (~10%), route to shadow model and log both predictions without serving the shadow response; compare distributions in the dashboard.
- **Accept:** a simulated demand-shock injection is detected, triggers retraining, and promotes a new model end-to-end; failed deploy auto-rolls-back.

---

## 6. Final deliverables
- README as system-design doc: architecture diagram, local-run instructions, deploy instructions, description of each CI/CD job.
- MLflow metric leaderboard table in README.
- 2–3 minute demo video showing the drift event → retrain → self-heal sequence.
- All infra reproducible via `docker-compose up`.

---

## 7. Definition of done
A reviewer can clone the repo, run one command to bring up the stack, open the ops dashboard, inject synthetic drift, and watch the system detect it, retrain, evaluate against production, and promote a new model — with rollback safety on deploy and a shadow A/B path for validation.
