"""Phase 5 unit tests: the ops dashboard's data layer.

The Streamlit script itself cannot be asserted on, which is exactly why all the
logic lives in ``dashboard/data.py``. These tests feed synthetic prediction rows,
feature frames and event dicts straight into those functions — no database, no
bundle, no monitoring package — so they run in the core CI job in milliseconds
and stay green while ``monitoring/`` is still being written.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from dashboard import data as dd

NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def _rows(n: int = 6, start: datetime = NOW, series: str = "A|conv") -> list[dict]:
    """n logged prediction rows, one hour apart going backwards."""
    return [
        {
            "id": i + 1,
            "dataset": "avocado",
            "series_id": series,
            "target_date": date(2026, 1, 4) + timedelta(days=i),
            "horizon_step": 1,
            "p10": 1.0 + i,
            "p50": 1.5 + i,
            "p90": 2.0 + i,
            "model_version": "v1",
            "predicted_at": start - timedelta(hours=i),
        }
        for i in range(n)
    ]


# --- normalisation ---------------------------------------------------------


def test_predictions_frame_empty_has_contract_columns():
    frame = dd.predictions_frame([])
    assert frame.empty
    for col in ("series_id", "target_date", "p50", "predicted_at"):
        assert col in frame.columns


def test_predictions_frame_normalises_naive_timestamps_to_utc():
    rows = _rows(1)
    rows[0]["predicted_at"] = datetime(2026, 7, 22, 12, 0)  # naive, as SQLite returns
    frame = dd.predictions_frame(rows)
    assert frame["predicted_at"].iloc[0] == NOW
    assert frame["target_date"].iloc[0] == date(2026, 1, 4)


def test_served_dataset_reads_env(monkeypatch):
    monkeypatch.delenv("SERVING_DATASET", raising=False)
    assert dd.served_dataset() == "avocado"
    monkeypatch.setenv("SERVING_DATASET", "m5")
    assert dd.served_dataset() == "m5"


# --- panel 2: request volume ------------------------------------------------


def test_request_volume_windows_24h_and_7d():
    rows = _rows(3)  # now, -1h, -2h
    rows += [
        {**r, "id": 100 + i, "predicted_at": NOW - timedelta(days=3)}
        for i, r in enumerate(_rows(2))
    ]
    rows += [
        {**r, "id": 200 + i, "predicted_at": NOW - timedelta(days=30)}
        for i, r in enumerate(_rows(4))
    ]
    vol = dd.request_volume(rows, now=NOW)
    assert vol["predictions_24h"] == 3
    assert vol["predictions_7d"] == 5
    assert vol["total_predictions"] == 9
    assert vol["series_24h"] == 1
    assert vol["last_prediction_at"] == NOW


def test_request_volume_counts_requests_not_rows():
    # one API call -> three horizon rows sharing a timestamp
    rows = [
        {**r, "id": i, "horizon_step": i + 1, "predicted_at": NOW} for i, r in enumerate(_rows(3))
    ]
    vol = dd.request_volume(rows, now=NOW)
    assert vol["predictions_24h"] == 3
    assert vol["requests_24h"] == 1


def test_request_volume_empty():
    vol = dd.request_volume([], now=NOW)
    assert vol["predictions_24h"] == 0
    assert vol["last_prediction_at"] is None


def test_hourly_volume_zero_fills_quiet_hours():
    hourly = dd.hourly_volume(_rows(3), now=NOW, hours=24)
    assert len(hourly) == 24
    assert hourly["predictions"].sum() == 3
    assert (hourly["predictions"] == 0).sum() == 21
    assert hourly["hour"].is_monotonic_increasing


def test_hourly_volume_empty_still_returns_grid():
    hourly = dd.hourly_volume([], now=NOW, hours=6)
    assert len(hourly) == 6
    assert hourly["predictions"].sum() == 0


# --- panel 3: histogram -----------------------------------------------------


def test_prediction_histogram_bins_and_counts():
    hist = dd.prediction_histogram(_rows(10), bins=5)
    assert len(hist) == 5
    assert hist["count"].sum() == 10
    assert (hist["bin_end"] > hist["bin_start"]).all()


def test_prediction_histogram_handles_single_value():
    rows = [{**r, "p50": 3.0} for r in _rows(4)]
    hist = dd.prediction_histogram(rows, bins=4)
    assert hist["count"].sum() == 4


def test_prediction_histogram_empty():
    assert dd.prediction_histogram([]).empty


# --- panel 4: feature drift -------------------------------------------------


def _feature_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=20, freq="D")
    return pd.DataFrame({"ds": dates, "lag_1": range(20), "month": [1] * 20})


def test_reference_current_split_is_date_ordered():
    reference, current = dd.reference_current_split(_feature_frame(), split_frac=0.7)
    assert len(reference) == 14
    assert len(current) == 6
    assert reference["ds"].max() < current["ds"].min()


def test_reference_current_split_degenerate_frame():
    frame = pd.DataFrame({"ds": [pd.Timestamp("2026-01-01")], "lag_1": [1]})
    reference, current = dd.reference_current_split(frame)
    assert len(reference) == len(current) == 1


def test_signals_table_sorts_red_first():
    signals = [
        SimpleNamespace(name="a", value=0.01, status="green", threshold=0.2, detail=""),
        SimpleNamespace(name="b", value=0.35, status="red", threshold=0.2, detail="shifted"),
        SimpleNamespace(name="c", value=0.15, status="yellow", threshold=0.2, detail=""),
    ]
    table = dd.signals_table(signals)
    assert list(table["name"]) == ["b", "c", "a"]
    assert table["detail"].iloc[0] == "shifted"


def test_signals_table_empty():
    assert dd.signals_table([]).empty


def test_feature_drift_table_degrades_without_monitoring():
    """Missing features (or a missing monitoring package) must not raise."""
    frame = _feature_frame()
    table = dd.feature_drift_table(frame, frame, ["not_a_column"])
    assert table.empty
    assert list(table.columns) == ["feature", "psi", "status", "threshold", "detail"]


def test_feature_drift_table_flags_shifted_features_when_monitoring_present():
    pytest.importorskip("monitoring.drift")
    reference = pd.DataFrame({"lag_1": list(range(200))})
    current = pd.DataFrame({"lag_1": [v + 500 for v in range(200)]})
    table = dd.feature_drift_table(reference, current, ["lag_1"])
    assert table.iloc[0]["status"] == "red"


def test_drift_thresholds_shape():
    thresholds = dd.drift_thresholds()
    assert thresholds["yellow"] < thresholds["red"]


def test_overall_status_picks_worst_and_unknown_when_empty():
    signals = [
        SimpleNamespace(name="a", value=0.0, status="green", threshold=0.2),
        SimpleNamespace(name="b", value=0.3, status="red", threshold=0.2),
    ]
    assert dd.overall_status(signals) == "red"
    assert dd.overall_status([]) == "unknown"


# --- panel 5: residual drift ------------------------------------------------


def test_residual_frame_joins_actuals_and_drops_unrealised():
    rows = _rows(4)
    actuals = {"A|conv": {date(2026, 1, 4): 1.0, date(2026, 1, 5): 2.0}}
    frame = dd.residual_frame(rows, actuals)
    assert len(frame) == 2  # the other two target dates have no actual yet
    assert frame["residual"].iloc[0] == pytest.approx(0.5)
    assert (frame["abs_error"] >= 0).all()


def test_residual_frame_without_actuals_is_empty_but_typed():
    frame = dd.residual_frame(_rows(3), {})
    assert frame.empty
    assert "residual" in frame.columns


def test_residual_series_aggregates_per_date():
    rows = _rows(2) + [{**r, "id": 50 + i, "series_id": "B|org"} for i, r in enumerate(_rows(2))]
    actuals = {
        "A|conv": {date(2026, 1, 4): 1.0, date(2026, 1, 5): 2.0},
        "B|org": {date(2026, 1, 4): 2.0, date(2026, 1, 5): 3.0},
    }
    series = dd.residual_series(dd.residual_frame(rows, actuals))
    assert list(series["date"]) == [date(2026, 1, 4), date(2026, 1, 5)]
    assert list(series["n"]) == [2, 2]
    assert series["mean_residual"].iloc[0] == pytest.approx(0.0)


def test_residual_series_empty():
    assert dd.residual_series(dd.residual_frame([], {})).empty


def test_rolling_wmape_series_is_finite_and_positive():
    rows = _rows(5)
    actuals = {"A|conv": {date(2026, 1, 4) + timedelta(days=i): 10.0 for i in range(5)}}
    rolling = dd.rolling_wmape_series(dd.residual_frame(rows, actuals), window=3)
    assert not rolling.empty
    assert {"date", "wmape"} <= set(rolling.columns)
    assert (rolling["wmape"].dropna() >= 0).all()


def test_rolling_wmape_series_empty_input():
    assert dd.rolling_wmape_series(dd.residual_frame([], {})).empty


def test_residual_and_breach_signals_tolerate_missing_inputs():
    empty = dd.residual_frame([], {})
    assert dd.residual_drift_signal(empty, empty) is None
    assert dd.wmape_breach_signal(pd.DataFrame(), 0.1) is None
    assert dd.wmape_breach_signal(pd.DataFrame({"date": [1], "wmape": [0.2]}), None) is None


# --- panel 1 / 6 / 7 --------------------------------------------------------


def test_production_model_info_from_injected_bundle():
    bundle = SimpleNamespace(
        dataset="avocado",
        model_version="avocado@abc123-2026-07-01",
        trained_at="2026-07-01T00:00:00+00:00",
        metrics={"wmape": 0.12, "wrmsse": 0.9},
        spec=SimpleNamespace(all=["lag_1", "month"]),
        series_ids=lambda: ["A|conv", "B|org"],
    )
    info = dd.production_model_info("avocado", bundle=bundle)
    assert info["available"] is True
    assert info["series_count"] == 2
    assert info["feature_count"] == 2
    assert info["metrics"]["wmape"] == 0.12


def test_production_model_info_missing_bundle_degrades(monkeypatch):
    monkeypatch.setenv("SERVING_DATASET", "does-not-exist")
    info = dd.production_model_info("does-not-exist")
    assert info["available"] is False
    assert info["model_version"] == "unknown"
    assert info["error"]


def test_metrics_table_orders_wmape_first():
    table = dd.metrics_table({"rmse": 2.0, "wmape": 0.1})
    assert list(table["metric"]) == ["wmape", "rmse"]


def test_leaderboard_table_falls_back_to_bundle_metrics():
    table = dd.leaderboard_table("synthetic-dataset", {"wmape": 0.2})
    assert not table.empty
    assert table.columns[0] == "model"


def test_retraining_events_table_flattens_metrics():
    events = [
        {
            "created_at": "2026-07-01T00:00:00Z",
            "dataset": "avocado",
            "triggered_by": "psi_breach",
            "before_metrics": {"wmape": 0.20},
            "after_metrics": '{"wmape": 0.15}',  # JSON string, as some stores return
            "promoted": True,
            "model_version": "v2",
            "notes": "auto",
        }
    ]
    table = dd.retraining_events_table(events)
    assert len(table) == 1
    assert table["wmape_before"].iloc[0] == pytest.approx(0.20)
    assert table["wmape_after"].iloc[0] == pytest.approx(0.15)
    assert table["wmape_delta"].iloc[0] == pytest.approx(-0.05)
    assert bool(table["promoted"].iloc[0]) is True


def test_retraining_events_table_limits_to_five():
    events = [{"dataset": "avocado", "triggered_by": "cron"} for _ in range(9)]
    assert len(dd.retraining_events_table(events, limit=5)) == 5


def test_retraining_events_table_empty_is_typed():
    table = dd.retraining_events_table([])
    assert table.empty
    assert "wmape_after" in table.columns


def test_drift_events_table_fills_missing_columns():
    table = dd.drift_events_table([{"signal_name": "psi:lag_1", "status": "red"}])
    assert list(table.columns) == [
        "created_at",
        "dataset",
        "signal_name",
        "value",
        "status",
        "detail",
    ]
    assert table["status"].iloc[0] == "red"


# --- DB-backed path (uses a temp SQLite file, like tests/test_phase3.py) ----


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    store = pytest.importorskip("serving.store")
    monkeypatch.setenv("SERVING_DB_URI", f"sqlite:///{(tmp_path / 'dash.db').as_posix()}")
    store.reset_engine()
    yield store
    store.reset_engine()


def test_fetch_prediction_rows_reads_the_log(temp_store):
    rows = [
        SimpleNamespace(date=date(2026, 1, 7), p10=1.0, p50=1.2, p90=1.4),
        SimpleNamespace(date=date(2026, 1, 14), p10=1.1, p50=1.3, p90=1.5),
    ]
    temp_store.log_predictions("avocado", "A|conv", "v1", rows)
    temp_store.log_predictions("other", "Z|x", "v1", rows)

    fetched = dd.fetch_prediction_rows("avocado")
    assert len(fetched) == 2
    assert {r["dataset"] for r in fetched} == {"avocado"}

    volume = dd.request_volume(fetched)
    assert volume["predictions_24h"] == 2
    assert volume["requests_24h"] == 1


def test_fetch_prediction_rows_honours_since(temp_store):
    rows = [SimpleNamespace(date=date(2026, 1, 7), p10=1.0, p50=1.2, p90=1.4)]
    temp_store.log_predictions("avocado", "A|conv", "v1", rows)
    future = datetime.now(UTC) + timedelta(days=1)
    assert dd.fetch_prediction_rows("avocado", since=future) == []


# --- the Streamlit shell ----------------------------------------------------


def test_app_module_imports():
    """The rendering shell must at least import (no top-level Streamlit calls)."""
    pytest.importorskip("streamlit")
    import dashboard.app as app

    assert callable(app.main)
