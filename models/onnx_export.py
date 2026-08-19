"""ONNX export for the LightGBM quantile forecaster, plus a latency benchmark.

Phase 4 asks for "ONNX export where applicable; benchmark latency vs native via
onnxruntime". This module does exactly that for the global LightGBM forecaster:
it converts each quantile booster (p10/p50/p90) to an ``ai.onnx.ml``
``TreeEnsembleRegressor``, *verifies numerical parity* against the native
booster, and times both engines on a realistic batch.

Why bother at all: the serving container currently carries lightgbm purely to
call ``Booster.predict``. An ONNX graph is a frozen, dependency-light artifact
that any runtime can execute, which makes it the natural hand-off point to a
non-Python inference tier. The value only holds if the exported graph returns
the *same numbers*, so parity is treated as part of the export, not as an
afterthought — :func:`export_models` refuses to report success unless
:func:`check_parity` passes.

Design decisions
----------------

**Input representation.** ONNX takes one dense numeric tensor, whereas the
project's models are fit on a pandas frame in which the series-identity columns
(``cfg.series_id_cols``, e.g. ``region``/``type``) are ``category`` dtype passed
through ``categorical_feature=``. The export therefore materialises the feature
matrix as :func:`numeric_matrix`: columns in ``FeatureSpec.all`` order, with the
categorical columns replaced by their integer category codes. This is not a
lossy shortcut — LightGBM stores categorical splits against exactly those codes,
and ``Booster.predict`` on the code matrix reproduces the frame-based prediction
bit-for-bit (verified: max abs diff 0.0 on the full avocado panel). The codes
must be pinned to the *training* categories, so :func:`numeric_matrix` accepts
the ``categories`` mapping carried by :class:`~serving.model_bundle.ModelBundle`.

**Categorical splits do survive the round-trip.** Contrary to the usual warning,
``onnxmltools`` does handle them: each LightGBM categorical split (a bitset over
category codes) is expanded into a chain of ``BRANCH_EQ`` nodes that onnxruntime
evaluates correctly. The avocado p50 model's 2057 categorical splits become
24 271 ``BRANCH_EQ`` nodes alongside 22 743 ``BRANCH_LEQ`` numeric nodes, and
they route identically to the booster. The export is therefore *not* restricted
to a numeric-only variant — see :func:`categorical_split_count`. The cost is
graph size: the exported p50 graph is ~2.7 MB.

**Where parity genuinely breaks: float32 thresholds.** The real constraint is
unrelated to categorical features. ``TreeEnsembleRegressor`` stores split
thresholds in the ``nodes_values`` attribute, which is a *float32* list, and the
converter versions pinned here (onnxmltools 1.16 / skl2onnx 1.20) do not emit
the opset-3 ``nodes_values_as_tensor`` alternative. Any feature whose distinct
observed values sit closer together than the float32 ulp at that feature's scale
can therefore have two neighbouring values collapse onto the same side of a
threshold, sending a row to a different leaf. The error is discrete (a wrong
leaf), not a rounding drift: most rows are exact and a few are visibly off.

On the real avocado bundle this bites, and :func:`float32_threshold_risk` names
the culprits exactly:

* ``Total_Volume_lag1`` reaches 6.25e7, where one float32 ulp is 4.0 while
  adjacent observed values differ by 0.01;
* every ``roll_mean_*``/``roll_std_*`` column contains near-duplicate windows
  whose values differ by ~1e-16.

Measured on the 18 249-row avocado panel (p50 model, 400 trees, double input):
mean abs diff 6.4e-5, max abs diff 8.5e-3, with 1313/18249 rows above 1e-5. That
is not a tolerance to paper over — it is a genuine, feature-scale-dependent
limitation of the conversion. It disappears entirely when no feature carries
sub-ulp value gaps: on a well-separated panel the same pipeline reproduces the
booster to ~1e-7.

The honest contract is therefore: **run the risk diagnostic, and only trust the
exported graph for feature matrices it reports as clean.** :func:`export_models`
surfaces the diagnostic alongside the parity report so the caller can see which
columns, if any, put the export at risk. Fixing it properly needs a converter
that writes double thresholds; rescaling or rounding the offending features
before training is the practical workaround.

**What the benchmark actually shows.** The answer depends entirely on batch
size, which is why the CLI takes ``--batch-size``. On the avocado bundle
(3 x 400 trees, 40 features, 20-core CPU, 100 reps, all three quantiles per
timed unit):

===========  =================  ================  =========
batch rows   native p50 (ms)    onnx p50 (ms)     speedup
===========  =================  ================  =========
1            29.34              0.53              55.7x
64           57.96              36.41             1.59x
512          59.18              152.21            0.39x
===========  =================  ================  =========

Two effects cross over. At batch 1 LightGBM is dominated by OpenMP thread-pool
setup (``n_jobs=-1`` fans a single row across every core), which onnxruntime
simply does not pay — this is the regime the ``/forecast`` endpoint lives in,
one row per recursive step, and ONNX wins it by a wide margin. At large batches
that fixed cost amortises and the expanded ``BRANCH_EQ`` chains dominate
instead, so the native booster wins. Export is worth it for low-latency
single-row serving, not for bulk scoring.

CLI::

    python -m models.onnx_export --dataset avocado --repeats 50 --batch-size 512
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from features.pipeline import FeatureSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lightgbm import LGBMRegressor

# Heavy/optional imports (onnx, onnxruntime, onnxmltools, lightgbm, mlflow) are
# deliberately kept inside functions so importing this module stays cheap and
# the light CI job — which installs neither the [deep] nor [models] extra —
# can still collect it.

#: Default place to drop exported graphs: alongside the servable bundles.
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"

#: Parity gate. Tight enough to catch a mis-routed leaf (which moves a
#: prediction by ~1e-3 or more), loose enough to absorb float32 accumulation.
PARITY_TOLERANCE = 1e-5

#: Single input tensor name used by every exported graph.
INPUT_NAME = "input"

Precision = Literal["float", "double"]


@dataclass
class ParityReport:
    """Numerical agreement between one native booster and its ONNX graph."""

    quantile: float
    n_rows: int
    tolerance: float
    max_abs_diff: float
    mean_abs_diff: float
    p99_abs_diff: float
    n_exceeding: int

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= self.tolerance

    def as_dict(self) -> dict[str, Any]:
        return {
            "quantile": self.quantile,
            "n_rows": self.n_rows,
            "tolerance": self.tolerance,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "p99_abs_diff": self.p99_abs_diff,
            "n_exceeding": self.n_exceeding,
            "passed": self.passed,
        }


@dataclass
class OnnxExport:
    """Result of exporting a full set of quantile models."""

    dataset: str
    precision: Precision
    feature_names: list[str]
    categorical: list[str]
    paths: dict[float, Path]
    parity: dict[float, ParityReport]
    categorical_splits: dict[float, int] = field(default_factory=dict)
    risky_features: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when every quantile model round-trips inside the tolerance."""
        return bool(self.parity) and all(r.passed for r in self.parity.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "precision": self.precision,
            "n_features": len(self.feature_names),
            "categorical": list(self.categorical),
            "paths": {q: str(p) for q, p in self.paths.items()},
            "parity": {q: r.as_dict() for q, r in self.parity.items()},
            "categorical_splits": dict(self.categorical_splits),
            "risky_features": dict(self.risky_features),
            "passed": self.passed,
        }


# --- feature matrix --------------------------------------------------------


def numeric_matrix(
    frame: pd.DataFrame,
    spec: FeatureSpec,
    *,
    categories: Mapping[str, Sequence[Any]] | None = None,
    dtype: str = "float64",
) -> np.ndarray:
    """Render the model inputs as one dense numeric array in ``spec.all`` order.

    Categorical columns become their integer category codes — the same values
    LightGBM splits on internally. ``categories`` pins the code assignment to the
    training vocabulary (a single-series frame would otherwise re-index and hand
    the model the wrong codes); pass ``ModelBundle.categories`` at serve time.
    Unknown levels become ``-1``, which LightGBM/ONNX treat as "not in the
    left-hand category set", matching pandas' own missing-category code.
    """
    out = frame.loc[:, spec.all].copy()
    for col in spec.categorical:
        series = out[col]
        if categories is not None and col in categories:
            series = pd.Categorical(
                series.astype(str), categories=[str(c) for c in categories[col]]
            )
            out[col] = series.codes
        elif isinstance(series.dtype, pd.CategoricalDtype):
            out[col] = series.cat.codes
        else:
            out[col] = pd.Categorical(series).codes
    return out.to_numpy(dtype=dtype)


def float32_threshold_risk(matrix: np.ndarray, feature_names: Sequence[str]) -> dict[str, float]:
    """Flag features whose values are too finely spaced for float32 thresholds.

    ONNX stores split thresholds as float32. If a column's two closest distinct
    values differ by less than one float32 ulp at that column's scale, a
    threshold between them cannot be represented and rows will route to the
    wrong leaf. Returns ``{feature: gap / ulp}`` for the offending columns only,
    so a ratio below ``1.0`` is the warning and an empty dict means the matrix is
    safe to export.
    """
    risky: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        col = matrix[:, i]
        values = np.unique(col[np.isfinite(col)])
        if values.size < 2:
            continue
        gap = float(np.min(np.diff(values)))
        ulp = float(np.spacing(np.float32(np.max(np.abs(values)))))
        if ulp > 0.0 and gap <= ulp:
            risky[name] = gap / ulp
    return risky


# --- conversion ------------------------------------------------------------


def _tensor_type(precision: Precision, n_features: int):
    from onnxmltools.convert.common.data_types import DoubleTensorType, FloatTensorType

    cls = DoubleTensorType if precision == "double" else FloatTensorType
    return cls([None, n_features])


def _numpy_dtype(precision: Precision) -> str:
    return "float64" if precision == "double" else "float32"


def convert_model(model: LGBMRegressor, n_features: int, *, precision: Precision = "double"):
    """Convert one fitted :class:`~lightgbm.LGBMRegressor` to an ONNX model proto.

    The underlying ``Booster`` is converted rather than the sklearn wrapper: the
    wrapper refuses a plain numpy matrix once it has been fit with
    ``categorical_feature=`` ("train and valid dataset categorical_feature do not
    match"), while the booster happily consumes the code matrix that ONNX needs.
    """
    from onnxmltools import convert_lightgbm

    return convert_lightgbm(
        model.booster_,
        initial_types=[(INPUT_NAME, _tensor_type(precision, n_features))],
        name=f"lightgbm_quantile_{n_features}f",
    )


def categorical_split_count(onnx_model) -> int:
    """Number of ``BRANCH_EQ`` (categorical) branches in an exported graph."""
    total = 0
    for node in onnx_model.graph.node:
        if node.op_type != "TreeEnsembleRegressor":
            continue
        for attr in node.attribute:
            if attr.name == "nodes_modes":
                total += sum(1 for m in attr.strings if m == b"BRANCH_EQ")
    return total


def make_session(onnx_model_or_path: Any):
    """Build a CPU onnxruntime session from a model proto, bytes, or a path."""
    import onnxruntime as ort

    if isinstance(onnx_model_or_path, (str, Path)):
        source: Any = str(onnx_model_or_path)
    elif isinstance(onnx_model_or_path, bytes):
        source = onnx_model_or_path
    else:
        source = onnx_model_or_path.SerializeToString()
    return ort.InferenceSession(source, providers=["CPUExecutionProvider"])


def onnx_predict(session, matrix: np.ndarray) -> np.ndarray:
    """Run one forward pass and flatten the ``(n, 1)`` regressor output."""
    return np.asarray(session.run(None, {INPUT_NAME: matrix})[0]).ravel()


def native_predict(model: LGBMRegressor, matrix: np.ndarray) -> np.ndarray:
    """Native reference prediction, taken from the booster on the code matrix."""
    return np.asarray(model.booster_.predict(matrix)).ravel()


# --- parity ----------------------------------------------------------------


def check_parity(
    model: LGBMRegressor,
    session,
    matrix: np.ndarray,
    *,
    quantile: float = 0.5,
    tolerance: float = PARITY_TOLERANCE,
) -> ParityReport:
    """Compare native and ONNX predictions row by row.

    Reports the whole error distribution rather than a single number: a
    mis-routed leaf shows up as a large ``max_abs_diff`` with a negligible
    ``mean_abs_diff``, which is precisely the signature the float32-threshold
    limitation produces.
    """
    native = native_predict(model, matrix)
    exported = onnx_predict(session, matrix)
    diff = np.abs(exported - native)
    return ParityReport(
        quantile=quantile,
        n_rows=int(matrix.shape[0]),
        tolerance=tolerance,
        max_abs_diff=float(diff.max()) if diff.size else 0.0,
        mean_abs_diff=float(diff.mean()) if diff.size else 0.0,
        p99_abs_diff=float(np.percentile(diff, 99)) if diff.size else 0.0,
        n_exceeding=int((diff > tolerance).sum()),
    )


# --- export ----------------------------------------------------------------


def export_models(
    models: Mapping[float, LGBMRegressor],
    spec: FeatureSpec,
    matrix: np.ndarray,
    *,
    dataset: str = "avocado",
    output_dir: Path | None = None,
    precision: Precision = "double",
    tolerance: float = PARITY_TOLERANCE,
) -> OnnxExport:
    """Convert every quantile model, write ``<q>.onnx``, and verify parity.

    ``matrix`` is the parity/validation batch produced by :func:`numeric_matrix`;
    it must already be in ``spec.all`` order. It is cast to the dtype implied by
    ``precision`` so the native reference and the graph see identical inputs.
    """
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR / dataset / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    batch = np.ascontiguousarray(matrix, dtype=_numpy_dtype(precision))
    n_features = batch.shape[1]
    if n_features != len(spec.all):
        raise ValueError(f"matrix has {n_features} columns but spec declares {len(spec.all)}")

    paths: dict[float, Path] = {}
    parity: dict[float, ParityReport] = {}
    cat_splits: dict[float, int] = {}
    for q, model in models.items():
        onnx_model = convert_model(model, n_features, precision=precision)
        path = out_dir / f"q{str(q).replace('.', '')}.onnx"
        path.write_bytes(onnx_model.SerializeToString())
        paths[q] = path
        cat_splits[q] = categorical_split_count(onnx_model)
        parity[q] = check_parity(
            model, make_session(onnx_model), batch, quantile=q, tolerance=tolerance
        )

    return OnnxExport(
        dataset=dataset,
        precision=precision,
        feature_names=list(spec.all),
        categorical=list(spec.categorical),
        paths=paths,
        parity=parity,
        categorical_splits=cat_splits,
        risky_features=float32_threshold_risk(batch, spec.all),
    )


# --- benchmark -------------------------------------------------------------


def _latency_stats(samples: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
    }


def benchmark_latency(
    models: Mapping[float, LGBMRegressor],
    sessions: Mapping[float, Any],
    matrix: np.ndarray,
    *,
    repeats: int = 50,
    warmup: int = 5,
    precision: Precision = "double",
) -> dict[str, Any]:
    """Time native LightGBM against onnxruntime on the same batch.

    One *timed unit* is a full probabilistic forecast — all three quantile models
    scored over the batch — because that is what the serving layer actually does
    per request. The native side calls ``Booster.predict`` rather than the
    sklearn wrapper so the comparison is engine versus engine, not engine versus
    pandas validation overhead. Both engines default to their own thread pools;
    the numbers are therefore whole-process throughput, not single-core cost.
    """
    import time

    batch = np.ascontiguousarray(matrix, dtype=_numpy_dtype(precision))
    quantiles = sorted(models)

    def run_native() -> None:
        for q in quantiles:
            models[q].booster_.predict(batch)

    def run_onnx() -> None:
        for q in quantiles:
            sessions[q].run(None, {INPUT_NAME: batch})

    for _ in range(max(0, warmup)):
        run_native()
        run_onnx()

    native_samples: list[float] = []
    onnx_samples: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_native()
        native_samples.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        run_onnx()
        onnx_samples.append((time.perf_counter() - t0) * 1000.0)

    native = _latency_stats(native_samples)
    onnx = _latency_stats(onnx_samples)
    return {
        "batch_size": int(batch.shape[0]),
        "n_features": int(batch.shape[1]),
        "n_models": len(quantiles),
        "quantiles": quantiles,
        "repeats": int(repeats),
        "warmup": int(warmup),
        "precision": precision,
        "native": native,
        "onnx": onnx,
        "speedup_mean": native["mean_ms"] / onnx["mean_ms"] if onnx["mean_ms"] else float("nan"),
        "speedup_p50": native["p50_ms"] / onnx["p50_ms"] if onnx["p50_ms"] else float("nan"),
        "per_row_native_us": native["mean_ms"] * 1000.0 / max(1, int(batch.shape[0])),
        "per_row_onnx_us": onnx["mean_ms"] * 1000.0 / max(1, int(batch.shape[0])),
    }


# --- orchestration ---------------------------------------------------------


def export_and_benchmark(
    dataset: str = "avocado",
    *,
    batch_size: int = 512,
    repeats: int = 50,
    warmup: int = 5,
    precision: Precision = "double",
    output_dir: Path | None = None,
    tolerance: float = PARITY_TOLERANCE,
    log_mlflow: bool = False,
) -> dict[str, Any]:
    """Load the servable bundle, export it, verify parity, and benchmark it.

    Everything is driven off :class:`~serving.model_bundle.ModelBundle` and
    :class:`~config.DatasetConfig`, so no column name is hard-coded here.
    """
    from features.pipeline import build_feature_matrix
    from serving.model_bundle import load_bundle

    bundle = load_bundle(dataset)
    frame, spec = build_feature_matrix(bundle.cfg, bundle.history)
    matrix = numeric_matrix(frame, spec, categories=bundle.categories)

    # A realistic serving batch: the most recent rows of the panel.
    batch = matrix[-batch_size:] if batch_size and batch_size < len(matrix) else matrix

    export = export_models(
        bundle.models,
        spec,
        matrix,
        dataset=dataset,
        output_dir=output_dir,
        precision=precision,
        tolerance=tolerance,
    )
    sessions = {q: make_session(p) for q, p in export.paths.items()}
    bench = benchmark_latency(
        bundle.models,
        sessions,
        batch,
        repeats=repeats,
        warmup=warmup,
        precision=precision,
    )

    result: dict[str, Any] = {
        "dataset": dataset,
        "model_version": bundle.model_version,
        "export": export.as_dict(),
        "benchmark": bench,
    }
    if log_mlflow:
        result["mlflow_run_id"] = log_export_to_mlflow(export, bench)
    return result


def log_export_to_mlflow(
    export: OnnxExport,
    benchmark: dict[str, Any],
    *,
    experiment: str = "onnx-export",
) -> str | None:
    """Best-effort MLflow logging of the parity report, timings, and artifacts.

    Deliberately lazy and non-fatal: an offline run (no tracking server, mlflow
    not installed) must not fail the export, so any failure returns ``None``.
    """
    try:
        import mlflow

        from models.tracking import git_commit_hash, setup_mlflow

        setup_mlflow(experiment)
        with mlflow.start_run(run_name=f"onnx-{export.dataset}") as run:
            mlflow.set_tag("git_commit", git_commit_hash())
            mlflow.log_params(
                {
                    "dataset": export.dataset,
                    "precision": export.precision,
                    "n_features": len(export.feature_names),
                    "batch_size": benchmark["batch_size"],
                    "repeats": benchmark["repeats"],
                }
            )
            for engine in ("native", "onnx"):
                for stat, value in benchmark[engine].items():
                    mlflow.log_metric(f"{engine}_{stat}", float(value))
            mlflow.log_metric("speedup_p50", float(benchmark["speedup_p50"]))
            for q, report in export.parity.items():
                mlflow.log_metric(f"parity_max_abs_diff_q{q}", report.max_abs_diff)
                mlflow.log_metric(f"parity_n_exceeding_q{q}", report.n_exceeding)
            mlflow.log_dict(export.as_dict(), "onnx_export.json")
            for path in export.paths.values():
                mlflow.log_artifact(str(path), artifact_path="onnx")
            return run.info.run_id
    except Exception:  # noqa: BLE001 - tracking must never break the export
        return None


def format_report(result: dict[str, Any]) -> str:
    """Render :func:`export_and_benchmark` output as a readable console table."""
    export = result["export"]
    bench = result["benchmark"]
    lines = [
        f"ONNX export — dataset={result['dataset']}  model={result['model_version']}",
        f"  precision : {export['precision']}   features: {export['n_features']}"
        f"   categorical: {', '.join(export['categorical']) or '-'}",
        "",
        "parity vs native booster",
        f"  {'quantile':>9} {'rows':>7} {'max_abs':>11} {'mean_abs':>11} "
        f"{'p99_abs':>11} {'>tol':>6} {'cat_splits':>11}  ok",
    ]
    for q in sorted(export["parity"], key=float):
        r = export["parity"][q]
        lines.append(
            f"  {float(q):>9.2f} {r['n_rows']:>7d} {r['max_abs_diff']:>11.3e} "
            f"{r['mean_abs_diff']:>11.3e} {r['p99_abs_diff']:>11.3e} "
            f"{r['n_exceeding']:>6d} {export['categorical_splits'][q]:>11d}"
            f"  {'PASS' if r['passed'] else 'FAIL'}"
        )
    risky = export["risky_features"]
    if risky:
        lines += [
            "",
            "float32 threshold risk (gap/ulp < 1 -> splits cannot be represented)",
        ]
        for name, ratio in sorted(risky.items(), key=lambda kv: kv[1]):
            lines.append(f"  {name:<28} {ratio:>10.3e}")
    else:
        lines += ["", "float32 threshold risk: none — every feature is safely separated"]

    lines += [
        "",
        f"latency — batch={bench['batch_size']} rows x {bench['n_features']} features, "
        f"{bench['n_models']} quantile models, {bench['repeats']} reps",
        f"  {'engine':<10} {'mean_ms':>10} {'p50_ms':>10} {'p95_ms':>10} {'min_ms':>10}",
    ]
    for engine in ("native", "onnx"):
        s = bench[engine]
        lines.append(
            f"  {engine:<10} {s['mean_ms']:>10.3f} {s['p50_ms']:>10.3f} "
            f"{s['p95_ms']:>10.3f} {s['min_ms']:>10.3f}"
        )
    lines += [
        f"  speedup (onnx vs native): {bench['speedup_p50']:.2f}x on p50, "
        f"{bench['speedup_mean']:.2f}x on mean",
        f"  per-row: native {bench['per_row_native_us']:.3f} us, "
        f"onnx {bench['per_row_onnx_us']:.3f} us",
    ]
    if result.get("mlflow_run_id"):
        lines.append(f"  mlflow run: {result['mlflow_run_id']}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LightGBM quantile models to ONNX (Phase 4).")
    p.add_argument("--dataset", default="avocado")
    p.add_argument("--batch-size", type=int, default=512, help="rows in the benchmark batch")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--precision", choices=("double", "float"), default="double")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--tolerance", type=float, default=PARITY_TOLERANCE)
    p.add_argument("--mlflow", action="store_true", help="log the run to MLflow (best effort)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    result = export_and_benchmark(
        args.dataset,
        batch_size=args.batch_size,
        repeats=args.repeats,
        warmup=args.warmup,
        precision=args.precision,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        tolerance=args.tolerance,
        log_mlflow=args.mlflow,
    )
    print(format_report(result))
    return 0 if result["export"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ParityReport",
    "OnnxExport",
    "Precision",
    "numeric_matrix",
    "float32_threshold_risk",
    "convert_model",
    "categorical_split_count",
    "make_session",
    "onnx_predict",
    "native_predict",
    "check_parity",
    "export_models",
    "benchmark_latency",
    "export_and_benchmark",
    "log_export_to_mlflow",
    "format_report",
    "main",
    "DEFAULT_OUTPUT_DIR",
    "PARITY_TOLERANCE",
    "INPUT_NAME",
]
