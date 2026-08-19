"""Phase 4: ONNX export of the LightGBM quantile forecaster.

Runs against a small synthetic panel with real quantile models rather than the
DVC dataset, so the suite stays fast and self-contained while still exercising
the genuine converter. The panel is deliberately well-separated numerically —
that is the regime in which :func:`models.onnx_export.float32_threshold_risk`
reports no risk and parity is therefore expected to hold tightly. A separate
test pins the diagnostic itself on a matrix that *is* unsafe, which is the
limitation documented in the module docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm", reason="lightgbm not installed ([models] extra)")
pytest.importorskip("onnx", reason="onnx not installed ([deep] extra)")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed ([deep] extra)")
pytest.importorskip("onnxmltools", reason="onnxmltools not installed ([deep] extra)")

from config import DatasetConfig, FeatureConfig, FourierTerm  # noqa: E402
from features.pipeline import build_feature_matrix  # noqa: E402
from models.evaluate import QUANTILES  # noqa: E402
from models.lightgbm_model import _fit_quantile_models  # noqa: E402
from models.onnx_export import (  # noqa: E402
    INPUT_NAME,
    PARITY_TOLERANCE,
    benchmark_latency,
    categorical_split_count,
    check_parity,
    convert_model,
    export_models,
    float32_threshold_risk,
    format_report,
    make_session,
    native_predict,
    numeric_matrix,
    onnx_predict,
)


def _synthetic_panel(n: int = 80) -> pd.DataFrame:
    """Two weekly series with distinct levels — the same shape as avocado."""
    rng = np.random.default_rng(0)
    frames = []
    for region, base in (("Alpha", 1.0), ("Beta", 2.0)):
        dates = pd.date_range("2018-01-07", periods=n, freq="W")
        seasonal = 0.2 * np.sin(2 * np.pi * np.arange(n) / 52.0)
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "AveragePrice": base + seasonal + rng.normal(0, 0.02, n),
                    "Total Volume": np.linspace(1000, 2000, n),
                    "region": region,
                    "type": "conventional",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def fitted():
    cfg = DatasetConfig.load("avocado")
    cfg.features = FeatureConfig(
        lags=[1, 2, 4],
        rolling_windows=[4],
        fourier=[FourierTerm(period=52.143, order=2)],
        exogenous_lag=1,
        use_exogenous=True,
    )
    panel = _synthetic_panel()
    frame, spec = build_feature_matrix(cfg, panel)
    models = _fit_quantile_models(frame, spec)
    categories = {col: list(frame[col].cat.categories) for col in spec.categorical}
    matrix = numeric_matrix(frame, spec, categories=categories)
    return cfg, frame, spec, models, categories, matrix


# --- feature matrix --------------------------------------------------------


def test_numeric_matrix_shape_and_codes(fitted):
    _, frame, spec, _, categories, matrix = fitted
    assert matrix.shape == (len(frame), len(spec.all))
    assert matrix.dtype == np.float64
    # categorical columns land in the trailing positions as integer codes
    for col in spec.categorical:
        col_values = matrix[:, spec.all.index(col)]
        expected = np.arange(len(categories[col]), dtype=float)
        assert set(np.unique(col_values)).issubset(set(expected))


def test_numeric_matrix_reproduces_native_prediction_exactly(fitted):
    """The code matrix is not an approximation of the pandas frame input."""
    _, frame, spec, models, _, matrix = fitted
    from_frame = models[0.5].predict(frame[spec.all])
    from_codes = native_predict(models[0.5], matrix)
    assert np.max(np.abs(from_frame - from_codes)) == 0.0


def test_numeric_matrix_pins_unknown_categories_to_minus_one(fitted):
    _, frame, spec, _, _, _ = fitted
    other = frame.copy()
    other["region"] = "Unseen"
    matrix = numeric_matrix(other, spec, categories={"region": ["Alpha", "Beta"]})
    assert np.all(matrix[:, spec.all.index("region")] == -1)


# --- conversion + parity ---------------------------------------------------


def test_categorical_splits_are_present_in_the_graph(fitted):
    """The identity columns really are trained as categorical splits."""
    _, _, spec, models, _, _ = fitted
    onnx_model = convert_model(models[0.5], len(spec.all))
    assert categorical_split_count(onnx_model) > 0


@pytest.mark.parametrize("precision", ["double", "float"])
def test_onnx_matches_native_within_tolerance(fitted, precision):
    _, _, spec, models, _, matrix = fitted
    dtype = np.float64 if precision == "double" else np.float32
    batch = np.ascontiguousarray(matrix, dtype=dtype)
    for q in QUANTILES:
        onnx_model = convert_model(models[q], len(spec.all), precision=precision)
        report = check_parity(models[q], make_session(onnx_model), batch, quantile=q)
        assert report.n_rows == len(batch)
        assert report.passed, f"q={q} {precision}: max_abs_diff={report.max_abs_diff:.3e}"
        assert report.max_abs_diff <= PARITY_TOLERANCE
        assert report.n_exceeding == 0


def test_session_input_name_and_output_shape(fitted):
    _, _, spec, models, _, matrix = fitted
    session = make_session(convert_model(models[0.5], len(spec.all)))
    assert [i.name for i in session.get_inputs()] == [INPUT_NAME]
    out = onnx_predict(session, matrix[:10])
    assert out.shape == (10,)


def test_nan_rows_round_trip(fitted):
    """Lag/rolling warm-up rows contain NaN; LightGBM's missing-value routing
    must survive the export, not just the dense rows."""
    _, _, spec, models, _, matrix = fitted
    nan_rows = matrix[np.isnan(matrix).any(axis=1)]
    assert len(nan_rows) > 0
    session = make_session(convert_model(models[0.5], len(spec.all)))
    diff = np.abs(onnx_predict(session, nan_rows) - native_predict(models[0.5], nan_rows))
    assert diff.max() <= PARITY_TOLERANCE


# --- float32 threshold diagnostic ------------------------------------------


def test_risk_diagnostic_is_clean_for_the_synthetic_panel(fitted):
    _, _, spec, _, _, matrix = fitted
    assert float32_threshold_risk(matrix, spec.all) == {}


def test_risk_diagnostic_flags_large_magnitude_and_near_duplicate_columns():
    n = 200
    safe = np.linspace(0.0, 10.0, n)
    huge = 6.0e7 + np.arange(n) * 0.01  # one float32 ulp here is 4.0
    dense = 1.0 + np.arange(n) * 1e-16  # values closer than float32 resolution
    matrix = np.column_stack([safe, huge, dense])
    risky = float32_threshold_risk(matrix, ["safe", "huge", "dense"])
    assert set(risky) == {"huge", "dense"}
    assert all(ratio < 1.0 for ratio in risky.values())


# --- export bundle ---------------------------------------------------------


def test_export_models_writes_files_and_passes_parity(fitted, tmp_path):
    _, _, spec, models, _, matrix = fitted
    export = export_models(models, spec, matrix, dataset="synthetic", output_dir=tmp_path)
    assert set(export.paths) == set(models)
    for path in export.paths.values():
        assert path.exists() and path.stat().st_size > 0
    assert export.passed
    assert export.risky_features == {}
    assert all(count > 0 for count in export.categorical_splits.values())

    payload = export.as_dict()
    assert payload["passed"] is True
    assert payload["n_features"] == len(spec.all)
    assert payload["categorical"] == spec.categorical


def test_export_rejects_mismatched_matrix_width(fitted, tmp_path):
    _, _, spec, models, _, matrix = fitted
    with pytest.raises(ValueError, match="columns"):
        export_models(models, spec, matrix[:, :-1], dataset="bad", output_dir=tmp_path)


def test_exported_file_reloads_and_still_matches(fitted, tmp_path):
    _, _, spec, models, _, matrix = fitted
    export = export_models(models, spec, matrix, dataset="synthetic", output_dir=tmp_path)
    session = make_session(export.paths[0.5])  # load from disk, not from memory
    diff = np.abs(onnx_predict(session, matrix) - native_predict(models[0.5], matrix))
    assert diff.max() <= PARITY_TOLERANCE


# --- benchmark -------------------------------------------------------------


def test_benchmark_returns_structured_stats(fitted, tmp_path):
    _, _, spec, models, _, matrix = fitted
    export = export_models(models, spec, matrix, dataset="synthetic", output_dir=tmp_path)
    sessions = {q: make_session(p) for q, p in export.paths.items()}
    bench = benchmark_latency(models, sessions, matrix[:64], repeats=5, warmup=1)

    assert bench["batch_size"] == 64
    assert bench["n_features"] == len(spec.all)
    assert bench["n_models"] == len(models)
    assert bench["repeats"] == 5
    for engine in ("native", "onnx"):
        stats = bench[engine]
        assert set(stats) == {"mean_ms", "p50_ms", "p95_ms", "min_ms"}
        assert all(v > 0 for v in stats.values())
        assert stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"]
    assert bench["speedup_p50"] > 0
    assert bench["per_row_onnx_us"] > 0


def test_format_report_renders_without_mlflow(fitted, tmp_path):
    _, _, spec, models, _, matrix = fitted
    export = export_models(models, spec, matrix, dataset="synthetic", output_dir=tmp_path)
    sessions = {q: make_session(p) for q, p in export.paths.items()}
    bench = benchmark_latency(models, sessions, matrix[:64], repeats=3, warmup=1)
    text = format_report(
        {
            "dataset": "synthetic",
            "model_version": "test",
            "export": export.as_dict(),
            "benchmark": bench,
        }
    )
    assert "parity vs native booster" in text
    assert "PASS" in text
    assert "speedup (onnx vs native)" in text


# --- real bundle (opt-in) --------------------------------------------------


@pytest.mark.integration
def test_real_bundle_export_and_benchmark(tmp_path):
    """End-to-end against the built avocado bundle, if one exists.

    Parity is *not* asserted here: the avocado feature matrix contains columns
    the risk diagnostic flags (see the module docstring), so the export is known
    to diverge on a small fraction of rows. What this test pins is that the
    pipeline runs and that the diagnostic explains any failure it sees.
    """
    from models.onnx_export import export_and_benchmark
    from serving.model_bundle import bundle_path

    if not bundle_path("avocado").exists():
        pytest.skip("no built avocado bundle on disk")

    result = export_and_benchmark(
        "avocado", batch_size=256, repeats=3, warmup=1, output_dir=tmp_path
    )
    export = result["export"]
    assert export["n_features"] > 0
    assert result["benchmark"]["batch_size"] == 256
    if not export["passed"]:
        assert export["risky_features"], "parity failed with no risky feature to explain it"
