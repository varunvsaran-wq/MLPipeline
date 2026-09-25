# Demo runbook

How to run the demo, and a 2–3 minute script for recording it. The story is
the closed loop: a model serves traffic, the world changes, monitoring catches
it, and the platform retrains and promotes a better model on its own.

## 1. Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\activate                       # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,models,serving,monitoring]"
dvc pull                                     # or place avocado.csv in data/raw/ (see below)
```

Python 3.11 is required (`pyproject.toml` pins `>=3.11,<3.12`).

**Data on a fresh machine:** the DVC remote is a local folder next to the repo,
so `dvc pull` only works on the machine that pushed it. Anywhere else, download
`avocado.csv` from <https://www.kaggle.com/datasets/neuromusic/avocado-prices>
into `data/raw/`. The file's md5 should match `data/raw/avocado.csv.dvc`.

## 2. Start it

```powershell
python scripts/run_demo.py --fresh
```

This builds anything missing (the model bundle, the leaderboard), starts the
API on :8000 and the dashboard on :8501, backfills 26 weeks of past forecasts,
and sends 80 live forecast requests. When it prints **DEMO IS UP**, open
<http://localhost:8501>. Press Ctrl+C to stop both servers.

`--fresh` clears the local prediction log, so the only retraining events on
screen are the ones you make during the demo. Leave it off to keep history.
Server logs go to `.demo-logs/`.

For a dry run in a terminal, with no UI:
`python scripts/demo_self_heal.py` (about 40 seconds; exits 0 when a model is promoted).

## 3. Recording script (about 2.5 minutes)

| Time | Show | Say |
|---|---|---|
| 0:00 | README architecture diagram | "Demand forecasting, but the point is the platform around the model. It's a closed loop: serve, monitor, retrain, gate, promote." |
| 0:20 | <http://localhost:8000/docs>, then run `POST /forecast` | "A FastAPI service returns p10/p50/p90 quantile forecasts from a global LightGBM model. Every prediction is logged." |
| 0:40 | Dashboard panels 1–3 | "The ops dashboard: which model is in production and its validation metrics, live request volume, and the distribution of forecasts." |
| 0:55 | Panel 6 (leaderboard) | "Three model families were compared on the same holdout. LightGBM roughly halves Prophet's WMAPE." |
| 1:05 | Sidebar: drag **Inject synthetic feature drift** to about 1.5×; panel 4 goes red | "Input drift is tracked with PSI per feature. Shift the inputs and it crosses 0.2 and goes red." Then reset the slider to 1.0. |
| 1:20 | Panel 5 | "Residual drift and a 30-step rolling WMAPE, computed from logged predictions joined to actuals. Green: the model is doing what it did in validation." |
| 1:30 | Sidebar: **Demand shock 1.6×** → click **Inject shock → retrain → promote** | "Now a real regime change. Demand jumps 60% over the last 40 weeks." |
| 1:35 | The live log streaming in the page | "Drift goes red on rolling WMAPE, residuals and features. That triggers a retrain of every family, evaluation on a fixed holdout, and the promotion gate: the candidate has to beat production by at least 1%." |
| 2:10 | Status turns green: *promoted a new model*; scroll to panel 7 | "Production WMAPE under the shock was 0.37. The retrained model scores 0.11. It's promoted, the old version is archived, and the event is on the audit trail." |
| 2:25 | Optional: set the shock to 1.0 and run again | "No shock means no drift and no retrain. The loop only fires when it should." |
| 2:35 | `.github/workflows/cd.yml` | "In CI/CD the same gate blocks bad models, and a failed post-deploy health check rolls back to the previous image." |

## What is real and what is simulated

Worth being upfront about this if anyone asks:

- **Real:** the drift math (PSI, residual shift, rolling WMAPE), the retraining
  of every family, holdout evaluation, the promotion gate, and MLflow registry
  transitions.
- **Simulated:** the demand shock. It is applied to a *temporary copy* of the
  data, the retrain uses a throwaway MLflow store, and the served bundle is not
  replaced. The demo can be run repeatedly without damaging anything, which is
  also why the "Production model" panel doesn't change after it runs.
- **Backfilled:** the residual panel's history is the production model replayed
  over the last 26 weeks, because a fresh API has only forecast the future.
- **Not run locally:** the CD and retrain GitHub Actions workflows. Their deploy
  steps need Fly.io and GHCR secrets.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `missing: prophet, ...` | `pip install -e ".[dev,models,serving,monitoring]"` inside the venv |
| `raw data missing` | `dvc pull`, or download the CSV (see Setup) |
| Port already in use | `python scripts/run_demo.py --api-port 8001 --dashboard-port 8502` |
| Self-heal button says "Unavailable here" | You're on the Docker dashboard, which leaves out the training stack. Use `run_demo.py`. |
| Dashboard panel 2 is empty | `python scripts/generate_traffic.py --requests 100`, then click **Refresh data** |
