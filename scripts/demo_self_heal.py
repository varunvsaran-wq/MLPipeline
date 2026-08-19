"""Self-healing demo: a demand shock is detected, retrains the platform, promotes (Phase 6).

    python scripts/demo_self_heal.py [--dataset avocado] [--shock 1.6] [--weeks 40] [--keep]

This is the executable form of the Phase 6 acceptance criterion — *a simulated
demand-shock injection is detected, triggers retraining, and promotes a new
model end-to-end* — and it narrates every step it takes.

**Nothing real is modified.** The demo is deliberately hermetic:

* the raw CSV is copied to a temp directory and the shock is applied to the
  copy; ``data.loader.RAW_DIR`` is pointed at that copy for the duration, so the
  DVC-tracked dataset on disk is never touched;
* the prediction log, drift events and retraining events go to a throwaway
  SQLite file via ``SERVING_DB_URI``, not to ``serving/predictions.db``;
* MLflow tracking *and* the Model Registry point at a throwaway
  ``sqlite:///`` store in the same temp directory — which is also what lets the
  promotion step run for real here, since a ``file://`` store cannot host a
  registry;
* the servable bundle is **not** rebuilt (``rebuild_bundle=False``) so
  ``models/artifacts/`` keeps its production artifact, and the leaderboard files
  that ``run_comparison`` rewrites are restored on the way out.

The narrative in seven steps:

1. Copy the dataset and multiply the target by ``--shock`` over the last
   ``--weeks`` periods — a sudden regime change no lag feature anticipates.
2. Replay what the *current production model* forecast for that window from
   pre-shock features, and log those forecasts exactly as the serving API would
   have. Building them from pre-shock history is the point: a model fed
   already-shocked lags would absorb the shock for free and no error would ever
   surface.
3. Evaluate drift over the logged predictions and the shocked feature matrix.
   Feature PSI, residual PSI and rolling WMAPE are all computed for real.
4. Seed the registry with the incumbent as Production, carrying the rolling
   WMAPE it is *actually* delivering under the shock — the honest number for the
   gate to beat, since its old validation score no longer describes the world.
5. Apply the trigger rule: red retrains.
6. Retrain every family on the shocked data, evaluate on the fixed holdout,
   pick the winner, and put it through the promotion gate.
7. Show the registry transition, the metrics diff and the recorded event.

Exit code 0 means a new model was promoted (acceptance met), 1 means the gate
blocked it, 2 means the demo could not run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # allow `python scripts/demo_self_heal.py`
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SHOCK = 1.6
DEFAULT_SHOCK_WEEKS = 40
DEFAULT_SCORED_WEEKS = 80
# Prophet is fit per series; three is enough to prove the comparison still runs
# while keeping the demo to well under a minute.
DEFAULT_MAX_SERIES = 3


def banner(number: int, title: str) -> None:
    print()
    print(f"=== STEP {number}: {title} ".ljust(78, "="))


def say(text: str = "") -> None:
    print(f"    {text}" if text else "")


# --- step 1: the shock ------------------------------------------------------


def inject_shock(
    cfg: Any, workdir: Path, shock: float, weeks: int
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Copy the raw CSV into ``workdir`` and scale the target over the last ``weeks``.

    Returns ``(pre_shock, shocked, shock_start)``. The shocked copy is what every
    later step reads, via a patched ``data.loader.RAW_DIR``; the untouched frame
    is kept because the production model's forecasts were issued *before* the
    shock and must be reconstructed from the world as it looked then.
    """
    from data import loader

    source = loader.RAW_DIR / cfg.raw_filename
    if not source.exists():
        raise FileNotFoundError(f"raw data missing at {source}; run `dvc pull` first")

    raw_dir = workdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, raw_dir / cfg.raw_filename)

    pre_shock = pd.read_csv(raw_dir / cfg.raw_filename)
    pre_shock[cfg.date_col] = pd.to_datetime(pre_shock[cfg.date_col])
    pre_shock = pre_shock.sort_values(cfg.date_col).reset_index(drop=True)

    dates = pd.Series(pre_shock[cfg.date_col].unique()).sort_values()
    shock_start = dates.iloc[-min(weeks, len(dates))]
    shocked = pre_shock.copy()
    mask = shocked[cfg.date_col] >= shock_start
    shocked.loc[mask, cfg.target_col] = shocked.loc[mask, cfg.target_col] * shock
    shocked.to_csv(raw_dir / cfg.raw_filename, index=False)

    loader.RAW_DIR = raw_dir  # every downstream load_raw() now reads the copy
    return pre_shock, shocked, shock_start


# --- step 2: what production would have served ------------------------------


def log_production_forecasts(
    dataset: str,
    bundle: Any,
    pre_shock: pd.DataFrame,
    scored_weeks: int,
) -> int:
    """Fill the prediction log with what production forecast *before* the shock.

    This stands in for weeks of ``/forecast`` traffic. The features are built
    from the pre-shock history on purpose: those forecasts were issued when the
    old regime was the only thing anyone had seen. Scoring the model on
    already-shocked lag features would let it absorb the shock for free and no
    error would ever surface — which is precisely the mistake that makes drift
    demos lie. The actuals it is judged against are the shocked ones, supplied
    later by the bundle history.

    Returns the number of forecast rows written.
    """
    from features.pipeline import build_feature_matrix
    from serving.store import log_predictions

    cfg = bundle.cfg
    frame, _ = build_feature_matrix(cfg, pre_shock)
    for col, categories in bundle.categories.items():
        frame[col] = pd.Categorical(frame[col].astype(str), categories=[str(c) for c in categories])

    dates = pd.Series(frame["ds"].unique()).sort_values()
    window_start = dates.iloc[-min(scored_weeks, len(dates))]
    recent = frame[frame["ds"] >= window_start].dropna(subset=["y"]).copy()

    x = recent[bundle.spec.all]
    for quantile, column in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
        recent[column] = bundle.models[quantile].predict(x)

    now = datetime.now(UTC)
    for series_id, group in recent.groupby("series_id", sort=False):
        rows = [
            SimpleNamespace(
                date=pd.Timestamp(r.ds).date(),
                p10=float(min(r.p10, r.p50)),
                p50=float(r.p50),
                p90=float(max(r.p90, r.p50)),
            )
            for r in group.itertuples()
        ]
        log_predictions(dataset, str(series_id), bundle.model_version, rows, predicted_at=now)

    return int(len(recent))


# --- step 4: the incumbent in the registry ----------------------------------


def seed_registry(model_name: str, live_wmape: float, experiment_id: str) -> str:
    """Register the incumbent as Production with the WMAPE it currently delivers."""
    import mlflow

    from models.registry import STAGE_PRODUCTION, register_model, transition

    with mlflow.start_run(experiment_id=experiment_id, run_name="incumbent-production") as run:
        mlflow.set_tag("model_family", "incumbent")
        mlflow.log_metric("wmape", live_wmape)
        mlflow.log_text("stand-in for the deployed artifact", "model_p50/MLmodel")
        run_id = run.info.run_id
    version = register_model(model_name, run_id).version
    transition(model_name, version, STAGE_PRODUCTION, description="incumbent at demo start")
    return version


# --- plumbing ---------------------------------------------------------------


def _snapshot_leaderboards() -> dict[Path, str | None]:
    from models.run_comparison import LEADERBOARD_JSON, LEADERBOARD_PATH

    return {
        path: (path.read_text(encoding="utf-8") if path.exists() else None)
        for path in (LEADERBOARD_PATH, LEADERBOARD_JSON)
    }


def _restore_leaderboards(snapshot: dict[Path, str | None]) -> None:
    for path, content in snapshot.items():
        if content is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(content, encoding="utf-8")


def _signal_table(signals: list[Any]) -> str:
    rows = [("signal", "value", "status", "threshold")]
    rows += [(s.name, f"{s.value:.4f}", s.status, f"{s.threshold:.4f}") for s in signals]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 6 self-healing demo (drift -> retrain -> promote)."
    )
    p.add_argument("--dataset", default="avocado")
    p.add_argument("--shock", type=float, default=DEFAULT_SHOCK, help="target multiplier")
    p.add_argument("--weeks", type=int, default=DEFAULT_SHOCK_WEEKS, help="periods to shock")
    p.add_argument(
        "--scored-weeks",
        type=int,
        default=DEFAULT_SCORED_WEEKS,
        help="periods of production forecasts to log (must exceed --weeks)",
    )
    p.add_argument("--max-series", type=int, default=DEFAULT_MAX_SERIES, help="Prophet series cap")
    p.add_argument("--keep", action="store_true", help="keep the temp workspace for inspection")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix=f"self-heal-{args.dataset}-"))

    # Everything stateful is redirected into the temp workspace *before* the
    # project modules that read these settings are imported.
    os.environ["SERVING_DB_URI"] = f"sqlite:///{(workdir / 'demo.db').as_posix()}"
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(workdir / 'mlflow.db').as_posix()}"

    import mlflow

    from config import DatasetConfig
    from models.retrain import MODEL_NAME_TEMPLATE, evaluate_drift, retrain, should_retrain
    from models.run_comparison import EXPERIMENT
    from monitoring import store as monitoring_store
    from serving import store as serving_store

    serving_store.reset_engine()
    monitoring_store.reset_tables()
    leaderboards = _snapshot_leaderboards()

    print("SELF-HEALING DEMO - drift detection -> retraining -> promotion")
    print(f"workspace     : {workdir}")
    print(f"prediction db : {os.environ['SERVING_DB_URI']}")
    print(f"mlflow store  : {os.environ['MLFLOW_TRACKING_URI']}")
    say("(the real dataset, bundle and databases are left untouched)")

    try:
        cfg = DatasetConfig.load(args.dataset)
        model_name = MODEL_NAME_TEMPLATE.format(dataset=args.dataset)
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        experiment_id = mlflow.create_experiment(
            EXPERIMENT, artifact_location=(workdir / "artifacts").as_uri()
        )

        # --- 1 ------------------------------------------------------------
        banner(1, "inject a synthetic demand shock")
        pre_shock, shocked, shock_start = inject_shock(cfg, workdir, args.shock, args.weeks)
        say(f"copied {cfg.raw_filename} to {workdir / 'raw'} and scaled {cfg.target_col}")
        say(
            f"x{args.shock:g} from {pd.Timestamp(shock_start).date()} onward "
            f"({args.weeks} periods, {len(shocked)} rows total)"
        )

        # --- 2 ------------------------------------------------------------
        banner(2, "replay what the production model forecast before the shock")
        from serving.model_bundle import load_bundle

        bundle = load_bundle(args.dataset)
        baseline_wmape = float(bundle.metrics.get("wmape", float("nan")))
        say(f"production bundle : {bundle.model_version}")
        say(f"validation WMAPE  : {baseline_wmape:.4f} (measured before the shock)")
        logged = log_production_forecasts(args.dataset, bundle, pre_shock, args.scored_weeks)
        say(f"logged {logged} forecasts covering the last {args.scored_weeks} periods")
        # The bundle carries the history it was trained on; point it at the
        # shocked world so those forecasts are judged against what happened.
        bundle.history = shocked

        # --- 3 ------------------------------------------------------------
        banner(3, "evaluate drift")
        status, signals = evaluate_drift(args.dataset, bundle=bundle)
        print()
        for line in _signal_table(signals).splitlines():
            say(line)
        print()
        say(f"overall drift status: {status.upper()}")
        for signal in signals:
            if signal.status == "red":
                say(f"  red: {signal.name} - {signal.detail}")
        say(f"{len(monitoring_store.fetch_drift_events(args.dataset))} drift events recorded")

        live_wmape = next(
            (s.value for s in signals if s.name == "rolling_wmape" and s.value == s.value),
            baseline_wmape,
        )

        # --- 4 ------------------------------------------------------------
        banner(4, "record the incumbent as Production")
        version = seed_registry(model_name, live_wmape, experiment_id)
        say(f"{model_name} v{version} -> Production, wmape={live_wmape:.4f}")
        say(f"(that is the rolling WMAPE it is delivering now, {live_wmape / baseline_wmape:.1f}x")
        say(" its pre-shock validation score - the honest number for a candidate to beat)")

        # --- 5 ------------------------------------------------------------
        banner(5, "apply the trigger rule")
        triggered = should_retrain(status)
        say(f"should_retrain({status!r}) -> {triggered}")
        if not triggered:
            say("drift is not red; the loop would stop here. Nothing was promoted.")
            return 1

        # --- 6 ------------------------------------------------------------
        banner(6, "retrain, evaluate on the fixed holdout, and run the promotion gate")
        report = retrain(
            dataset=args.dataset,
            triggered_by="demand-shock-demo",
            status=status,
            signals=signals,
            skip_dvc=True,
            max_series=args.max_series,
            rebuild_bundle=False,  # never overwrite the real servable artifact
            before_metrics={"wmape": live_wmape},
        )

        # --- 7 ------------------------------------------------------------
        banner(7, "the outcome")
        from models.registry import get_production_version

        production = get_production_version(model_name)
        print()
        for line in report.summary().splitlines():
            say(line)
        print()
        say(
            f"registry: {model_name} Production is now v{production.version if production else '-'} "
            f"(was v{version})"
        )
        events = monitoring_store.fetch_retraining_events(args.dataset, limit=1)
        if events:
            event = events[0]
            say(
                f"retraining event #{event['id']}: promoted={event['promoted']}, "
                f"wmape {event['before_metrics'].get('wmape'):.4f} -> "
                f"{event['after_metrics'].get('wmape'):.4f}"
            )
        print()
        if report.promoted:
            say("ACCEPTANCE MET: shock detected -> retrained -> new model promoted end-to-end.")
        else:
            say("Gate blocked the candidate; Production is unchanged.")
        return 0 if report.promoted else 1

    except Exception as exc:  # noqa: BLE001 - a demo reports its own failure
        print(f"\n[ERROR] demo could not complete: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        _restore_leaderboards(leaderboards)
        serving_store.reset_engine()
        monitoring_store.reset_tables()
        if args.keep:
            print(f"\nworkspace kept at {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["inject_shock", "log_production_forecasts", "main", "seed_registry"]
