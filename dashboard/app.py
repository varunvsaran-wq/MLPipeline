"""Streamlit ops dashboard — the operator's view of the running system.

This is deliberately *not* a prediction UI. Nobody requests a forecast here; the
page answers the questions an on-call engineer asks about a deployed model: what
is serving, how much traffic is it taking, has the input distribution moved, is
the error growing, and when did we last retrain.

The file is a rendering shell on purpose. Every number comes from
:mod:`dashboard.data`, which is pure and unit-tested — Streamlit script bodies
cannot be meaningfully asserted on, so nothing that could be wrong is allowed to
live here. What remains is layout, caching, the drift-injection control used
to demonstrate the Phase 5 acceptance criterion (shifted feature data must turn
the drift status red), and a button that runs the Phase 6 self-heal demo
(``scripts/demo_self_heal.py --record``) and streams its narration here.

Run it with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dashboard import data as dd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_SCRIPT = PROJECT_ROOT / "scripts" / "demo_self_heal.py"
REFRESH_SECONDS = 60
STATUS_COLOR = {"green": "#1a9850", "yellow": "#e6a700", "red": "#d73027", "unknown": "#888888"}
STATUS_EMOJI = {"green": "🟢", "yellow": "🟡", "red": "🔴", "unknown": "⚪"}


# --- cached loaders ---------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _bundle(dataset: str) -> Any | None:
    try:
        from serving.model_bundle import load_bundle

        return load_bundle(dataset)
    except Exception:
        return None


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def _prediction_rows(dataset: str, _nonce: int) -> pd.DataFrame:
    return dd.predictions_frame(dd.fetch_prediction_rows(dataset))


@st.cache_data(ttl=600, show_spinner=False)
def _feature_matrix(dataset: str, _nonce: int) -> tuple[pd.DataFrame, list[str]]:
    """Feature matrix rebuilt from the bundle's history (the drift reference)."""
    bundle = _bundle(dataset)
    if bundle is None:
        return pd.DataFrame(), []
    try:
        from features.pipeline import build_feature_matrix

        frame, spec = build_feature_matrix(bundle.cfg, bundle.history)
        return frame, list(spec.numeric)
    except Exception:
        return pd.DataFrame(), []


@st.cache_data(ttl=REFRESH_SECONDS, show_spinner=False)
def _residuals(dataset: str, series_ids: tuple[str, ...], _nonce: int) -> pd.DataFrame:
    bundle = _bundle(dataset)
    rows = dd.fetch_prediction_rows(dataset)
    if bundle is None or not rows:
        return dd.residual_frame(rows, {})
    try:
        from serving.predictor import Predictor

        predictor = Predictor(bundle)
        actuals = {sid: predictor.actuals(sid) for sid in series_ids}
    except Exception:
        actuals = {}
    return dd.residual_frame(rows, actuals)


def _status_badge(status: str, label: str) -> str:
    color = STATUS_COLOR.get(status, STATUS_COLOR["unknown"])
    return (
        f"<span style='background:{color};color:#fff;padding:2px 10px;"
        f"border-radius:10px;font-size:0.85rem'>{label}: {status.upper()}</span>"
    )


# --- panels -----------------------------------------------------------------


def render_header(dataset: str) -> None:
    st.title("Demand Forecasting — Ops Dashboard")
    st.caption(
        f"dataset `{dataset}` · refreshed {datetime.now(UTC):%Y-%m-%d %H:%M:%S} UTC · "
        "monitoring package "
        + ("available" if dd.monitoring_available() else "not installed (drift panels degraded)")
    )


def render_production_model(info: dict) -> None:
    st.subheader("1 · Production model")
    if not info["available"]:
        st.warning(f"No servable bundle: {info['error']}")
        return
    cols = st.columns(4)
    cols[0].metric("Version", info["model_version"])
    cols[1].metric("Trained at", str(info["trained_at"])[:19] or "—")
    cols[2].metric("Series", info["series_count"])
    cols[3].metric("Features", info["feature_count"])
    registry = info.get("registry")
    if registry:
        st.caption(f"Registry (Production): {registry['name']} v{registry['version']}")
    metrics = dd.metrics_table(info["metrics"])
    if metrics.empty:
        st.info("No validation metrics recorded in the bundle.")
    else:
        st.dataframe(metrics, hide_index=True, width="stretch")


def render_volume(frame: pd.DataFrame, now: datetime) -> None:
    st.subheader("2 · Live request volume")
    volume = dd.request_volume(frame, now=now)
    cols = st.columns(5)
    cols[0].metric("Requests (24h)", volume["requests_24h"])
    cols[1].metric("Requests (7d)", volume["requests_7d"])
    cols[2].metric("Predictions (24h)", volume["predictions_24h"])
    cols[3].metric("Predictions (7d)", volume["predictions_7d"])
    cols[4].metric("Series seen (24h)", volume["series_24h"])
    hourly = dd.hourly_volume(frame, now=now, hours=24)
    st.bar_chart(hourly.set_index("hour")[["requests", "predictions"]])
    if volume["last_prediction_at"] is None:
        st.info("No predictions logged yet — call the API to populate this panel.")


def render_histogram(frame: pd.DataFrame, now: datetime) -> None:
    st.subheader("3 · Prediction distribution")
    hours = st.slider("Window (hours)", 1, 168, 24, key="hist_hours")
    recent = frame
    if not frame.empty:
        cutoff = now - timedelta(hours=hours)
        keep = [ts is not None and ts >= cutoff for ts in frame["predicted_at"]]
        recent = frame[keep]
    hist = dd.prediction_histogram(recent, bins=20)
    if hist.empty:
        st.info("No logged forecasts in this window.")
        return
    st.bar_chart(hist.set_index("center")["count"])


def render_feature_drift(dataset: str, nonce: int, drift_scale: float) -> list:
    st.subheader("4 · Feature drift (PSI)")
    frame, numeric = _feature_matrix(dataset, nonce)
    if frame.empty or not numeric:
        st.info("Feature matrix unavailable (no bundle, or feature build failed).")
        return []
    reference, current = dd.reference_current_split(frame, date_col="ds")
    if drift_scale != 1.0:
        current = current.copy()
        for col in numeric:
            current[col] = current[col] * drift_scale
        st.warning(f"Synthetic drift injected: numeric features scaled by {drift_scale:.2f}×")

    features = st.multiselect("Features", numeric, default=numeric[:12], key="drift_features")
    table = dd.feature_drift_table(reference, current, features)
    if table.empty:
        st.info("No drift signals — the monitoring package may not be installed yet.")
        return []
    thresholds = dd.drift_thresholds()
    st.caption(f"PSI thresholds — yellow ≥ {thresholds['yellow']}, red ≥ {thresholds['red']}")
    display = table.copy()
    display["status"] = [f"{STATUS_EMOJI.get(s, '⚪')} {s}" for s in display["status"]]
    st.dataframe(display, hide_index=True, width="stretch")
    st.bar_chart(table.set_index("feature")["psi"])
    return list(table.itertuples(index=False))


def render_residual_drift(dataset: str, frame: pd.DataFrame, nonce: int, baseline: float | None):
    st.subheader("5 · Residual drift")
    series_ids = (
        tuple(sorted(str(s) for s in frame["series_id"].unique())) if not frame.empty else ()
    )
    residuals = _residuals(dataset, series_ids, nonce)
    if residuals.empty:
        st.info("No logged prediction has a realised actual yet — nothing to score.")
        return None, None
    series = dd.residual_series(residuals)
    st.line_chart(series.set_index("date")[["mean_residual"]])

    rolling = dd.rolling_wmape_series(residuals, window=30)
    if not rolling.empty:
        st.caption("Rolling WMAPE (30-step window)")
        st.line_chart(rolling.set_index("date")[["wmape"]])

    reference, current = dd.reference_current_split(residuals, date_col="target_date")
    resid_signal = dd.residual_drift_signal(reference, current)
    breach_signal = dd.wmape_breach_signal(rolling, baseline)
    badges = [
        _status_badge(getattr(s, "status", "unknown"), getattr(s, "name", "signal"))
        for s in (resid_signal, breach_signal)
        if s is not None
    ]
    if badges:
        st.markdown(" ".join(badges), unsafe_allow_html=True)
    return resid_signal, breach_signal


def render_leaderboard(dataset: str, fallback_metrics: dict) -> None:
    st.subheader("6 · Model leaderboard")
    table = dd.leaderboard_table(dataset, fallback_metrics)
    if table.empty:
        st.info("No leaderboard entries yet — run models/run_comparison.py.")
        return
    st.dataframe(table, hide_index=True, width="stretch")


def render_retraining(dataset: str) -> None:
    st.subheader("7 · Retraining history (last 5)")
    table = dd.retraining_events_table(dataset=dataset, limit=5)
    if table.empty:
        st.info("No retraining events recorded yet.")
    else:
        st.dataframe(table, hide_index=True, width="stretch")

    events = dd.drift_events_table(dataset=dataset, limit=50)
    if not events.empty:
        with st.expander("Recorded drift events"):
            st.dataframe(events, hide_index=True, width="stretch")


# --- self-heal demo ---------------------------------------------------------


def self_heal_blocker(dataset: str) -> str | None:
    """Why the self-heal demo can't run from here, or ``None`` if it can.

    The demo retrains every model family, so it needs the training stack and the
    raw data — both absent from the lean dashboard image by design.
    """
    missing = [m for m in ("mlflow", "prophet", "lightgbm") if importlib.util.find_spec(m) is None]
    if missing:
        return f"training deps not installed here ({', '.join(missing)})"
    try:
        from config import DatasetConfig
        from data import loader

        raw = loader.RAW_DIR / DatasetConfig.load(dataset).raw_filename
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        return f"dataset config unavailable: {exc}"
    if not raw.exists():
        return f"raw data missing at {raw} (run `dvc pull`)"
    return None


def run_self_heal(dataset: str, shock: float) -> int:
    """Run the self-heal demo in a subprocess, streaming its narration into the page."""
    cmd = [
        sys.executable,
        str(DEMO_SCRIPT),
        "--dataset",
        dataset,
        "--shock",
        f"{shock:g}",
        "--record",
    ]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    lines: list[str] = []
    with st.status(f"Self-heal demo: injecting a x{shock:g} demand shock...", expanded=True) as box:
        output = st.empty()
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            # Library warnings are noise in a live demo; the narration is what matters.
            if "Warning" in line or line.lstrip().startswith("warnings.warn"):
                continue
            lines.append(line.rstrip())
            output.code("\n".join(lines[-40:]), language=None)
        code = proc.wait()
        labels = {0: ("promoted a new model", "complete"), 1: ("gate blocked", "complete")}
        label, state = labels.get(code, ("failed", "error"))
        box.update(label=f"Self-heal demo: {label} (exit {code})", state=state, expanded=True)
    return code


def render_self_heal_controls(dataset: str) -> None:
    st.header("Self-heal demo")
    blocker = self_heal_blocker(dataset)
    if blocker:
        st.caption(
            f"Unavailable here: {blocker}. Run it from a full local install instead: "
            "`python scripts/demo_self_heal.py --record`"
        )
        return
    shock = st.slider(
        "Demand shock (×)",
        0.5,
        2.5,
        1.6,
        0.1,
        help="Multiplies the target over the last 40 weeks of a copy of the data. "
        "Near 1.0 the drift stays green and nothing retrains.",
    )
    if st.button("Inject shock → retrain → promote", type="primary"):
        st.session_state["self_heal"] = {"shock": shock}


# --- entry point ------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Forecasting Ops Dashboard", page_icon="📈", layout="wide")
    dataset = dd.served_dataset()

    with st.sidebar:
        st.header("Controls")
        dataset = st.text_input("Dataset", value=dataset)
        drift_scale = st.slider(
            "Inject synthetic feature drift (×)",
            0.5,
            3.0,
            1.0,
            0.1,
            help="Scales the current window's numeric features — used to prove drift turns red.",
        )
        nonce = 0
        if st.button("Refresh data"):
            st.cache_data.clear()
            nonce = 1
        st.divider()
        render_self_heal_controls(dataset)

    pending = st.session_state.pop("self_heal", None)
    if pending:
        run_self_heal(dataset, pending["shock"])
        st.cache_data.clear()
        st.info("Retraining history below now includes this run (panel 7).")

    info = dd.production_model_info(dataset, bundle=_bundle(dataset))
    now = datetime.now(UTC)
    frame = _prediction_rows(dataset, nonce)

    render_header(dataset)
    render_production_model(info)
    st.divider()
    render_volume(frame, now)
    st.divider()
    render_histogram(frame, now)
    st.divider()
    signals = render_feature_drift(dataset, nonce, drift_scale)
    st.divider()
    baseline = info["metrics"].get("wmape") if info["available"] else None
    resid_signal, breach_signal = render_residual_drift(dataset, frame, nonce, baseline)
    st.divider()
    render_leaderboard(dataset, info["metrics"] if info["available"] else {})
    st.divider()
    render_retraining(dataset)

    all_signals = [s for s in (resid_signal, breach_signal) if s is not None]
    all_signals += [s for s in signals if getattr(s, "status", None) is not None]
    st.sidebar.markdown(
        _status_badge(dd.overall_status(all_signals), "Overall"), unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()


__all__ = ["main"]
