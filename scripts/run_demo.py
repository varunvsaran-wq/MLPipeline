"""One command to bring the demo up locally: API + ops dashboard, warmed with traffic.

    python scripts/run_demo.py [--dataset avocado] [--fresh] [--no-traffic]

What it does, in order, skipping anything already done:

1. checks the Python deps the demo needs and says exactly what to install;
2. makes sure the raw dataset is on disk (``dvc pull`` if not);
3. builds the servable model bundle if ``models/artifacts/<dataset>/`` is empty;
4. runs the model comparison once if ``models/leaderboard.json`` is missing;
5. starts the FastAPI server (:8000) and the Streamlit dashboard (:8501);
6. backfills 26 weeks of past forecasts (so residual panels have actuals) and
   sends a burst of live forecast traffic, so every dashboard panel has data;
7. prints the demo walkthrough and waits; Ctrl+C stops both servers.

``--fresh`` deletes the local prediction log first, so the dashboard starts
empty and the only retraining events on screen are the ones made during the demo.
It never touches a database set through ``SERVING_DB_URI``.

This runs everything natively (no Docker) because the in-dashboard self-heal
button needs the training stack, which the slim dashboard image leaves out.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED = {
    "lightgbm": "models",
    "prophet": "models",
    "mlflow": "core",
    "fastapi": "serving",
    "uvicorn": "serving",
    "streamlit": "monitoring",
    "evidently": "monitoring",
}


def step(text: str) -> None:
    print(f"\n==> {text}", flush=True)


def check_deps() -> bool:
    missing = [name for name in REQUIRED if importlib.util.find_spec(name) is None]
    if not missing:
        print("    all demo dependencies installed")
        return True
    print(f"    missing: {', '.join(missing)}")
    print('    install with:  pip install -e ".[dev,models,serving,monitoring]"')
    return False


def ensure_data(dataset: str) -> bool:
    from config import DatasetConfig
    from data import loader

    raw = loader.RAW_DIR / DatasetConfig.load(dataset).raw_filename
    if raw.exists():
        print(f"    found {raw.relative_to(PROJECT_ROOT)}")
        return True
    print("    raw data missing; trying `dvc pull` ...")
    dvc = shutil.which("dvc")
    if dvc:
        subprocess.run([dvc, "pull"], cwd=PROJECT_ROOT, check=False)
    if raw.exists():
        return True
    print(
        f"    still missing {raw}.\n"
        "    The DVC remote is a local folder, so a fresh clone can't pull it. Download\n"
        "    'avocado.csv' from https://www.kaggle.com/datasets/neuromusic/avocado-prices\n"
        f"    and save it as {raw}"
    )
    return False


def run_module(*args: str) -> bool:
    result = subprocess.run([sys.executable, "-m", *args], cwd=PROJECT_ROOT, check=False)
    return result.returncode == 0


def ensure_bundle(dataset: str) -> bool:
    from serving.model_bundle import bundle_path

    path = bundle_path(dataset)
    if path.exists():
        print(f"    found {path.relative_to(PROJECT_ROOT)}")
        return True
    print("    building the servable bundle (trains the production models, ~1 min) ...")
    return run_module("serving.model_bundle", "--dataset", dataset)


def ensure_leaderboard(dataset: str) -> None:
    from serving.leaderboard import LEADERBOARD_JSON

    if LEADERBOARD_JSON.exists():
        print(f"    found {LEADERBOARD_JSON.relative_to(PROJECT_ROOT)}")
        return
    print("    running the Prophet vs LightGBM comparison (a few minutes) ...")
    if not run_module("models.run_comparison", "--dataset", dataset):
        print("    comparison failed; the leaderboard will show the served model only")


def backfill_history(dataset: str, weeks: int) -> int:
    """Log the production model's one-step-ahead forecasts for the last ``weeks`` periods.

    A freshly started API has only forecast the future, so nothing in the log
    has an actual yet and the residual / rolling-WMAPE panels would be blank.
    This replays what the model would have predicted each week, stamped as
    issued one period before its target date, so the live-volume panels (last
    24h / 7d) are unaffected. Skipped if the log already has rows for ``dataset``.
    """
    from types import SimpleNamespace

    import pandas as pd
    from sqlalchemy import func, select

    from features.pipeline import build_feature_matrix
    from serving import store
    from serving.model_bundle import load_bundle

    with store.get_engine().connect() as conn:
        existing = conn.execute(
            select(func.count())
            .select_from(store.predictions)
            .where(store.predictions.c.dataset == dataset)
        ).scalar_one()
    if existing:
        print(f"    prediction log already has {existing} rows; skipping backfill")
        return 0

    bundle = load_bundle(dataset)
    frame, _ = build_feature_matrix(bundle.cfg, bundle.history)
    for col, cats in bundle.categories.items():
        frame[col] = pd.Categorical(frame[col].astype(str), categories=[str(c) for c in cats])
    dates = pd.Series(frame["ds"].unique()).sort_values()
    recent = frame[frame["ds"] >= dates.iloc[-min(weeks, len(dates))]].dropna(subset=["y"])
    recent = recent.copy()
    x = recent[bundle.spec.all]
    for quantile, column in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
        recent[column] = bundle.models[quantile].predict(x)

    step_back = pd.Timedelta(days=7) if len(dates) < 2 else dates.iloc[-1] - dates.iloc[-2]
    written = 0
    for r in recent.itertuples():
        row = SimpleNamespace(
            date=pd.Timestamp(r.ds).date(),
            p10=min(r.p10, r.p50),
            p50=r.p50,
            p90=max(r.p90, r.p50),
        )
        issued = (pd.Timestamp(r.ds) - step_back).to_pydatetime()
        written += store.log_predictions(
            dataset, str(r.series_id), bundle.model_version, [row], predicted_at=issued
        )
    return written


def wait_http(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(1)
    return False


def start(cmd: list[str], env: dict[str, str], log: Path) -> subprocess.Popen:
    handle = log.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=handle)


def walkthrough(api: str, dash: str) -> None:
    print(
        f"""
------------------------------------------------------------------------------
 DEMO IS UP
   Ops dashboard : {dash}
   API docs      : {api}/docs
   MLflow runs   : `mlflow ui` in another terminal -> http://localhost:5000

 Suggested walkthrough (see DEMO.md for the full script):
   1. Dashboard panels 1-3: production model, live traffic, forecast histogram.
   2. Sidebar "Inject synthetic feature drift": drag to ~1.5x -> PSI turns red.
   3. Sidebar "Self-heal demo" -> click "Inject shock -> retrain -> promote".
      Watch drift go red, all families retrain, the gate promote the winner.
   4. Panel 7 (retraining history) now shows the run with WMAPE before/after.
   5. Optional: set the shock to ~1.0 and rerun -> drift stays green, no retrain.

 More traffic any time:  python scripts/generate_traffic.py --requests 100
 Press Ctrl+C to stop both servers.
------------------------------------------------------------------------------""",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bring up the local demo (API + dashboard).")
    p.add_argument("--dataset", default="avocado")
    p.add_argument("--api-port", type=int, default=8000)
    p.add_argument("--dashboard-port", type=int, default=8501)
    p.add_argument("--fresh", action="store_true", help="wipe the local prediction log first")
    p.add_argument("--no-traffic", action="store_true", help="skip the warm-up traffic")
    p.add_argument("--backfill-weeks", type=int, default=26, help="0 disables the backfill")
    p.add_argument("--requests", type=int, default=80, help="warm-up API calls")
    p.add_argument("--skip-comparison", action="store_true", help="don't build the leaderboard")
    args = p.parse_args(argv)

    step("checking dependencies")
    if not check_deps():
        return 2
    step("checking data")
    if not ensure_data(args.dataset):
        return 2
    step("checking the model bundle")
    if not ensure_bundle(args.dataset):
        print("    bundle build failed")
        return 2
    if not args.skip_comparison:
        step("checking the leaderboard")
        ensure_leaderboard(args.dataset)

    env = {**os.environ, "SERVING_DATASET": args.dataset, "PYTHONIOENCODING": "utf-8"}
    if args.fresh:
        step("resetting the local prediction log")
        if "SERVING_DB_URI" in os.environ:
            print("    SERVING_DB_URI is set; leaving that database alone")
        else:
            db = PROJECT_ROOT / "serving" / "predictions.db"
            db.unlink(missing_ok=True)
            print(f"    removed {db.relative_to(PROJECT_ROOT)}")

    if args.backfill_weeks > 0:
        step(f"backfilling {args.backfill_weeks} weeks of past forecasts")
        print(f"    {backfill_history(args.dataset, args.backfill_weeks)} rows written")

    logs = PROJECT_ROOT / ".demo-logs"
    logs.mkdir(exist_ok=True)
    api_url = f"http://localhost:{args.api_port}"
    dash_url = f"http://localhost:{args.dashboard_port}"

    step("starting the API and the dashboard")
    procs = [
        start(
            [sys.executable, "-m", "uvicorn", "serving.app:app", "--port", str(args.api_port)],
            env,
            logs / "api.log",
        ),
        start(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard/app.py",
                "--server.port",
                str(args.dashboard_port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            env,
            logs / "dashboard.log",
        ),
    ]
    try:
        if not wait_http(f"{api_url}/health", 60):
            print(f"    API did not come up; see {logs / 'api.log'}")
            return 1
        print(f"    API ready at {api_url}")
        if not wait_http(f"{dash_url}/_stcore/health", 90):
            print(f"    dashboard did not come up; see {logs / 'dashboard.log'}")
            return 1
        print(f"    dashboard ready at {dash_url}")

        if not args.no_traffic:
            step(f"sending {args.requests} warm-up forecast requests")
            from scripts.generate_traffic import generate

            n = generate(api_url, args.requests, quiet=True, seed=7)
            print(f"    {n} series forecasts logged")

        walkthrough(api_url, dash_url)
        while all(proc.poll() is None for proc in procs):
            time.sleep(1)
        print(f"a server exited unexpectedly; logs are in {logs}")
        return 1
    except KeyboardInterrupt:
        print("\nstopping ...")
        return 0
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
