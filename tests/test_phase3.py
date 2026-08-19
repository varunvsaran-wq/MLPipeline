"""Phase 3 unit tests: API schemas, prediction store, leaderboard, validation.

These are dependency-light (no lightgbm / no data), so they run in the core CI
job. The predictor + live-API tests live in ``test_serving_integration.py`` and
self-skip when the model extra isn't installed.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from config import DatasetConfig
from data.validation import validate_frame
from serving.leaderboard import _sorted_entries, load_leaderboard
from serving.schemas import BatchForecastRequest, ForecastRequest

# --- schemas ---------------------------------------------------------------


def test_forecast_request_rejects_bad_horizon():
    with pytest.raises(ValueError):
        ForecastRequest(series_id="A", horizon=0)
    with pytest.raises(ValueError):
        ForecastRequest(series_id="A", horizon=9999)


def test_forecast_request_accepts_valid():
    req = ForecastRequest(series_id="TotalUS|conventional", horizon=6)
    assert req.horizon == 6


def test_batch_request_requires_at_least_one_series():
    with pytest.raises(ValueError):
        BatchForecastRequest(series_ids=[], horizon=4)


# --- prediction store ------------------------------------------------------


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    from serving import store

    monkeypatch.setenv("SERVING_DB_URI", f"sqlite:///{(tmp_path / 'p.db').as_posix()}")
    store.reset_engine()
    yield store
    store.reset_engine()


def test_store_log_and_fetch_roundtrip(temp_store):
    rows = [
        SimpleNamespace(date=date(2026, 1, 7), p10=1.0, p50=1.2, p90=1.4),
        SimpleNamespace(date=date(2026, 1, 14), p10=1.1, p50=1.3, p90=1.5),
    ]
    n = temp_store.log_predictions("avocado", "A|conv", "v1", rows)
    assert n == 2
    fetched = temp_store.fetch_history("A|conv", limit=10)
    assert len(fetched) == 2
    assert {r["horizon_step"] for r in fetched} == {1, 2}
    assert fetched[0]["model_version"] == "v1"


def test_store_isolates_by_series(temp_store):
    row = [SimpleNamespace(date=date(2026, 1, 7), p10=1.0, p50=1.2, p90=1.4)]
    temp_store.log_predictions("avocado", "A", "v1", row)
    temp_store.log_predictions("avocado", "B", "v1", row)
    assert len(temp_store.fetch_history("A")) == 1
    assert len(temp_store.fetch_history("B")) == 1


# --- leaderboard -----------------------------------------------------------


def test_leaderboard_sorts_by_wmape_ascending():
    entries = _sorted_entries({"Prophet": {"wmape": 0.16}, "LightGBM": {"wmape": 0.08}})
    assert [e["model"] for e in entries] == ["LightGBM", "Prophet"]


def test_leaderboard_fallback_uses_bundle_metrics(tmp_path, monkeypatch):
    # No leaderboard.json on disk -> single-row fallback from the served metrics.
    monkeypatch.setattr("serving.leaderboard.LEADERBOARD_JSON", tmp_path / "nope.json")
    data = load_leaderboard("avocado", fallback_metrics={"wmape": 0.09, "wrmsse": 1.1})
    assert data["dataset"] == "avocado"
    assert data["entries"] and data["entries"][0]["wmape"] == 0.09


# --- data validation -------------------------------------------------------


def _good_panel(n=40):
    frames = []
    for region in ("A", "B"):
        dates = pd.date_range("2019-01-06", periods=n, freq="W")
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "AveragePrice": np.linspace(1.0, 2.0, n),
                    "region": region,
                    "type": "conventional",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _cfg():
    return DatasetConfig.load("avocado")


def test_validation_passes_clean_data():
    assert validate_frame(_cfg(), _good_panel()) == []


def test_validation_flags_negative_target():
    df = _good_panel()
    df.loc[0, "AveragePrice"] = -5.0
    problems = validate_frame(_cfg(), df)
    assert any("negative" in p for p in problems)


def test_validation_flags_duplicates_and_nulls():
    df = _good_panel()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate (series, date)
    df.loc[1, "AveragePrice"] = np.nan
    problems = validate_frame(_cfg(), df)
    assert any("duplicate" in p for p in problems)
    assert any("null" in p for p in problems)


def test_validation_flags_missing_column():
    df = _good_panel().drop(columns=["AveragePrice"])
    problems = validate_frame(_cfg(), df)
    assert any("missing required column" in p for p in problems)


def test_validation_flags_series_shorter_than_horizon():
    cfg = _cfg()
    short = _good_panel(n=cfg.horizon)  # exactly horizon -> not enough to hold out
    problems = validate_frame(cfg, short)
    assert any("horizon" in p for p in problems)
