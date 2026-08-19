"""Phase 5 unit tests: PSI, the three drift signals, and the event store.

Everything is synthetic and dependency-light so this runs in the core CI job.
The acceptance test for the phase is :func:`test_injected_feature_shift_turns_status_red`
— shifted feature data must take the overall drift status to red. The Evidently
renderer is exercised separately and skips when the library is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import (
    PSI_RED,
    DriftSignal,
    evidently_report,
    feature_drift,
    overall_status,
    psi,
    residual_drift,
    rolling_wmape,
    wmape_breach,
)

RNG = np.random.default_rng(20260722)

# --- PSI -------------------------------------------------------------------


def test_psi_identical_distributions_is_near_zero():
    sample = RNG.normal(size=5000)
    assert psi(sample, sample.copy()) == pytest.approx(0.0, abs=1e-9)


def test_psi_same_population_different_draws_is_small():
    ref = RNG.normal(loc=10.0, scale=2.0, size=5000)
    cur = RNG.normal(loc=10.0, scale=2.0, size=5000)
    assert psi(ref, cur) < 0.1


def test_psi_strong_shift_exceeds_red_threshold():
    ref = RNG.normal(loc=0.0, scale=1.0, size=5000)
    cur = RNG.normal(loc=3.0, scale=1.0, size=5000)
    assert psi(ref, cur) > PSI_RED


def test_psi_is_finite_when_bins_are_empty():
    # Disjoint supports: every current value lands in one end bin.
    ref = RNG.normal(size=2000)
    cur = RNG.normal(loc=50.0, size=2000)
    value = psi(ref, cur)
    assert np.isfinite(value)
    assert value > PSI_RED


def test_psi_is_symmetric_and_non_negative():
    ref = RNG.normal(size=3000)
    cur = RNG.normal(loc=0.7, size=3000)
    assert psi(ref, cur) >= 0.0
    assert psi(ref, cur) == pytest.approx(psi(cur, ref), rel=0.35)


def test_psi_handles_degenerate_and_empty_inputs():
    assert psi(np.ones(100), np.ones(100)) == 0.0
    assert psi([], [1.0, 2.0]) == 0.0
    assert psi([1.0, 2.0], []) == 0.0
    with pytest.raises(ValueError):
        psi([1.0, 2.0], [1.0, 2.0], bins=1)


def test_psi_ignores_nans():
    ref = np.concatenate([RNG.normal(size=1000), np.full(50, np.nan)])
    cur = np.concatenate([RNG.normal(size=1000), np.full(50, np.nan)])
    assert np.isfinite(psi(ref, cur))


# --- feature drift ---------------------------------------------------------


def _reference_frame(n: int = 2000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lag_1": RNG.normal(loc=100.0, scale=10.0, size=n),
            "roll_mean_7": RNG.normal(loc=100.0, scale=5.0, size=n),
            "fourier_sin_1": RNG.uniform(-1.0, 1.0, size=n),
        }
    )


def test_feature_drift_stable_data_is_green():
    signals = feature_drift(_reference_frame(), _reference_frame(), ["lag_1", "roll_mean_7"])
    assert [s.name for s in signals] == ["lag_1", "roll_mean_7"]
    assert all(s.status == "green" for s in signals)
    assert overall_status(signals) == "green"


def test_feature_drift_flags_only_the_shifted_feature():
    ref = _reference_frame()
    cur = _reference_frame()
    cur["lag_1"] = cur["lag_1"] + 40.0  # 4 sigma shift
    signals = feature_drift(ref, cur, ["lag_1", "roll_mean_7", "fourier_sin_1"])
    by_name = {s.name: s for s in signals}
    assert by_name["lag_1"].status == "red"
    assert by_name["lag_1"].value > PSI_RED
    assert by_name["roll_mean_7"].status == "green"
    assert by_name["fourier_sin_1"].status == "green"


def test_feature_drift_skips_missing_columns():
    ref = _reference_frame()
    signals = feature_drift(ref, ref, ["lag_1", "not_a_column"])
    assert [s.name for s in signals] == ["lag_1"]


def test_feature_drift_yellow_band():
    ref = pd.DataFrame({"x": RNG.normal(size=20000)})
    # Tuned to land between the yellow and red PSI thresholds.
    cur = pd.DataFrame({"x": RNG.normal(loc=0.4, size=20000)})
    signal = feature_drift(ref, cur, ["x"])[0]
    assert 0.1 <= signal.value <= 0.2
    assert signal.status == "yellow"


# --- residual drift --------------------------------------------------------


def test_residual_drift_stable_is_green():
    ref = RNG.normal(loc=0.0, scale=2.0, size=3000)
    cur = RNG.normal(loc=0.0, scale=2.0, size=3000)
    signal = residual_drift(ref, cur)
    assert signal.name == "residual"
    assert signal.status == "green"


def test_residual_drift_detects_bias_swing():
    ref = RNG.normal(loc=0.0, scale=2.0, size=3000)
    cur = RNG.normal(loc=6.0, scale=2.0, size=3000)
    signal = residual_drift(ref, cur)
    assert signal.value > PSI_RED
    assert signal.status == "red"
    assert "->" in signal.detail


def test_residual_drift_detects_variance_blowup():
    ref = RNG.normal(loc=0.0, scale=1.0, size=4000)
    cur = RNG.normal(loc=0.0, scale=6.0, size=4000)
    assert residual_drift(ref, cur).status == "red"


# --- rolling WMAPE ---------------------------------------------------------


def _log_frame(days: int, error: float, start: str = "2026-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=days, freq="D")
    actual = np.full(days, 100.0)
    return pd.DataFrame({"target_date": dates, "actual": actual, "p50": actual * (1.0 + error)})


def test_rolling_wmape_matches_the_known_error():
    rolling = rolling_wmape(_log_frame(60, error=0.10), window=30)
    assert list(rolling.columns) == ["date", "wmape"]
    assert len(rolling) == 60
    assert rolling["wmape"].iloc[-1] == pytest.approx(0.10)


def test_rolling_wmape_window_is_trailing_calendar_days():
    good = _log_frame(30, error=0.10, start="2026-01-01")
    bad = _log_frame(30, error=0.50, start="2026-01-31")
    rolling = rolling_wmape(pd.concat([good, bad], ignore_index=True), window=30)
    # The last window sees only the degraded stretch.
    assert rolling["wmape"].iloc[-1] == pytest.approx(0.50)
    # Mid-way the window straddles both regimes.
    mid = rolling[rolling["date"] == pd.Timestamp("2026-02-05")]["wmape"].iloc[0]
    assert 0.10 < mid < 0.50


def test_rolling_wmape_aggregates_multiple_series_per_day():
    frame = pd.DataFrame(
        {
            "target_date": ["2026-01-01", "2026-01-01"],
            "actual": [100.0, 900.0],
            "p50": [110.0, 900.0],
        }
    )
    rolling = rolling_wmape(frame, window=30)
    assert len(rolling) == 1
    assert rolling["wmape"].iloc[0] == pytest.approx(10.0 / 1000.0)


def test_rolling_wmape_empty_and_bad_input():
    empty = rolling_wmape(pd.DataFrame(columns=["target_date", "actual", "p50"]), window=30)
    assert empty.empty
    assert list(empty.columns) == ["date", "wmape"]
    with pytest.raises(KeyError):
        rolling_wmape(pd.DataFrame({"target_date": []}), window=30)


def test_rolling_wmape_custom_column_names():
    frame = _log_frame(5, error=0.2).rename(
        columns={"target_date": "d", "actual": "y", "p50": "yhat"}
    )
    rolling = rolling_wmape(frame, window=30, date_col="d", actual_col="y", pred_col="yhat")
    assert rolling["wmape"].iloc[-1] == pytest.approx(0.2)


# --- WMAPE breach (the retraining trigger) ---------------------------------


def test_wmape_breach_within_tolerance_is_green():
    rolling = rolling_wmape(_log_frame(40, error=0.105), window=30)
    signal = wmape_breach(rolling, baseline_wmape=0.10, tolerance=0.2)
    assert signal.name == "rolling_wmape"
    assert signal.status == "green"
    assert signal.threshold == pytest.approx(0.12)


def test_wmape_breach_beyond_tolerance_is_red():
    rolling = rolling_wmape(_log_frame(40, error=0.15), window=30)
    signal = wmape_breach(rolling, baseline_wmape=0.10, tolerance=0.2)
    assert signal.status == "red"
    assert signal.value == pytest.approx(0.15)
    assert "trigger at" in signal.detail


def test_wmape_breach_warns_before_it_fires():
    rolling = rolling_wmape(_log_frame(40, error=0.115), window=30)
    assert wmape_breach(rolling, baseline_wmape=0.10, tolerance=0.2).status == "yellow"


def test_wmape_breach_without_actuals_is_green_not_an_alert():
    signal = wmape_breach(pd.DataFrame(columns=["date", "wmape"]), baseline_wmape=0.10)
    assert signal.status == "green"
    assert np.isnan(signal.value)


# --- overall status --------------------------------------------------------


def test_overall_status_takes_the_worst():
    def sig(status: str) -> DriftSignal:
        return DriftSignal(name=status, value=0.0, status=status, threshold=PSI_RED)

    assert overall_status([]) == "green"
    assert overall_status([sig("green"), sig("green")]) == "green"
    assert overall_status([sig("green"), sig("yellow")]) == "yellow"
    assert overall_status([sig("yellow"), sig("red"), sig("green")]) == "red"


# --- acceptance criterion --------------------------------------------------


def test_injected_feature_shift_turns_status_red():
    """Phase 5 acceptance: injecting shifted feature data turns drift status red."""
    features = ["lag_1", "roll_mean_7", "fourier_sin_1"]
    reference = _reference_frame(3000)

    baseline = _reference_frame(3000)
    assert overall_status(feature_drift(reference, baseline, features)) == "green"

    shifted = baseline.copy()
    shifted["lag_1"] = shifted["lag_1"] * 1.5 + 60.0
    shifted["roll_mean_7"] = shifted["roll_mean_7"] + 30.0
    signals = feature_drift(reference, shifted, features)
    assert overall_status(signals) == "red"
    assert any(s.status == "red" and s.value > PSI_RED for s in signals)


# --- Evidently (optional) --------------------------------------------------


def test_evidently_report_renders_html_when_available():
    pytest.importorskip("evidently")
    ref = _reference_frame(300)
    cur = _reference_frame(300)
    cur["lag_1"] += 40.0
    html = evidently_report(ref, cur, ["lag_1", "roll_mean_7"])
    if html is None:
        pytest.skip("installed Evidently API does not match the guarded call")
    assert isinstance(html, str)
    assert "<" in html and len(html) > 500


def test_evidently_report_returns_none_without_usable_columns():
    ref = _reference_frame(50)
    assert evidently_report(ref, ref, ["nope"]) is None


# --- event store -----------------------------------------------------------


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    from monitoring import store as monitoring_store
    from serving import store as serving_store

    monkeypatch.setenv("SERVING_DB_URI", f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    serving_store.reset_engine()
    monitoring_store.reset_tables()
    yield monitoring_store
    serving_store.reset_engine()
    monitoring_store.reset_tables()


def test_drift_event_roundtrip(temp_store):
    row_id = temp_store.record_drift_event("avocado", "lag_1", 0.31, "red", "PSI over 10 bins")
    assert row_id > 0
    events = temp_store.fetch_drift_events("avocado")
    assert len(events) == 1
    event = events[0]
    assert event["signal_name"] == "lag_1"
    assert event["value"] == pytest.approx(0.31)
    assert event["status"] == "red"
    assert event["detail"] == "PSI over 10 bins"
    assert event["recorded_at"] is not None


def test_drift_events_filter_and_limit(temp_store):
    for i in range(6):
        temp_store.record_drift_event("avocado", f"f{i}", 0.05 * i, "green")
    temp_store.record_drift_event("m5", "f0", 0.4, "red")
    assert len(temp_store.fetch_drift_events("avocado")) == 6
    assert len(temp_store.fetch_drift_events("m5")) == 1
    assert len(temp_store.fetch_drift_events()) == 7
    assert len(temp_store.fetch_drift_events("avocado", limit=2)) == 2


def test_drift_events_shares_the_prediction_database(temp_store):
    from datetime import date
    from types import SimpleNamespace

    from serving import store as serving_store

    serving_store.log_predictions(
        "avocado",
        "A|conv",
        "v1",
        [SimpleNamespace(date=date(2026, 1, 7), p10=1.0, p50=1.2, p90=1.4)],
    )
    temp_store.record_drift_event("avocado", "residual", 0.05, "green")
    assert temp_store.get_engine() is serving_store.get_engine()
    assert len(serving_store.fetch_history("A|conv")) == 1
    assert len(temp_store.fetch_drift_events("avocado")) == 1


def test_retraining_event_roundtrip(temp_store):
    row_id = temp_store.record_retraining_event(
        dataset="avocado",
        triggered_by="rolling_wmape",
        before_metrics={"wmape": 0.181, "bias": -0.4},
        after_metrics={"wmape": 0.142, "bias": 0.1},
        promoted=True,
        model_version="lightgbm-2026-07-22",
        notes="30d WMAPE 21% worse than validation",
    )
    assert row_id > 0
    events = temp_store.fetch_retraining_events("avocado")
    assert len(events) == 1
    event = events[0]
    assert event["triggered_by"] == "rolling_wmape"
    assert event["before_metrics"] == {"wmape": 0.181, "bias": -0.4}
    assert event["after_metrics"]["wmape"] == pytest.approx(0.142)
    assert event["promoted"] is True
    assert event["model_version"] == "lightgbm-2026-07-22"


def test_retraining_events_newest_first_and_limited(temp_store):
    for i in range(7):
        temp_store.record_retraining_event(
            "avocado", "manual", {"wmape": 0.2}, {"wmape": 0.2 - i / 100}, promoted=i % 2 == 0
        )
    events = temp_store.fetch_retraining_events("avocado")
    assert len(events) == 5  # default limit
    assert events[0]["id"] > events[-1]["id"]
    assert len(temp_store.fetch_retraining_events(limit=10)) == 7


def test_retraining_event_tolerates_unserialisable_metrics(temp_store):
    temp_store.record_retraining_event(
        "avocado", "drift", {"wmape": np.float64(0.2)}, {"obj": object()}, promoted=False
    )
    event = temp_store.fetch_retraining_events("avocado")[0]
    assert event["before_metrics"]["wmape"] == pytest.approx(0.2)
    assert event["after_metrics"] == {}
    assert event["promoted"] is False
