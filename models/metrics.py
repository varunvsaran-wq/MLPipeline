"""Domain-correct forecasting metrics.

Per the project's operating principles we report WMAPE / bias (and, from Phase 2,
WRMSSE and pinball loss) rather than bare RMSE. Phase 1 needs WMAPE as the
headline metric; the rest land alongside LightGBM.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_arrays(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    if yt.size == 0:
        raise ValueError("empty arrays")
    return yt, yp


def wmape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted Mean Absolute Percentage Error = sum|e| / sum|y|.

    Volume-weighted and well-defined when individual actuals are zero, which is
    why it is preferred over plain MAPE for demand data.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    denom = np.abs(yt).sum()
    if denom == 0:
        raise ValueError("WMAPE undefined: sum(|y_true|) == 0")
    return float(np.abs(yt - yp).sum() / denom)


def forecast_bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean signed error (pred - actual). Positive = over-forecasting."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(yp - yt))


def pinball_loss(y_true: ArrayLike, y_pred: ArrayLike, quantile: float) -> float:
    """Quantile (pinball) loss at ``quantile`` in (0, 1) for probabilistic forecasts."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    yt, yp = _as_arrays(y_true, y_pred)
    diff = yt - yp
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def rmsse(y_true: ArrayLike, y_pred: ArrayLike, train_history: ArrayLike) -> float:
    """Root Mean Squared Scaled Error for one series (the per-series term of WRMSSE).

    The scale is the mean squared 1-step naive error over the training history,
    which makes the error comparable across series of different magnitudes.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    hist = np.asarray(train_history, dtype=float)
    if hist.size < 2:
        raise ValueError("need >= 2 training points to scale RMSSE")
    scale = np.mean(np.diff(hist) ** 2)
    if scale == 0:
        return float("nan")  # flat series: scale undefined, drop from the average
    return float(np.sqrt(np.mean((yt - yp) ** 2) / scale))


def wrmsse(per_series: list[tuple]) -> float:
    """Weighted RMSSE across series (the official M5 metric form).

    ``per_series`` is a list of ``(y_true, y_pred, train_history, weight)``.
    Weights are normalised internally; series with an undefined scale are
    skipped. For non-M5 data the weights are an adaptation (e.g. dollar volume),
    not the official M5 hierarchy weights.
    """
    scores: list[float] = []
    weights: list[float] = []
    for y_true, y_pred, hist, w in per_series:
        s = rmsse(y_true, y_pred, hist)
        if np.isnan(s):
            continue
        scores.append(s)
        weights.append(float(w))
    if not scores:
        raise ValueError("no series with a defined RMSSE scale")
    wv = np.asarray(weights)
    total = wv.sum()
    if total == 0:
        wv = np.ones_like(wv)
        total = wv.sum()
    return float(np.dot(np.asarray(scores), wv) / total)


def evaluate(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    quantile_preds: dict[float, ArrayLike] | None = None,
) -> dict[str, float]:
    """Forecast metric bundle logged to MLflow each run.

    Always includes WMAPE + bias. When ``quantile_preds`` (e.g. for the 10/50/90
    percentiles) is given, adds pinball loss per quantile. WRMSSE is computed
    separately (it needs per-series training history) via :func:`wrmsse`.
    """
    out = {
        "wmape": wmape(y_true, y_pred),
        "bias": forecast_bias(y_true, y_pred),
    }
    if quantile_preds:
        for q, qp in quantile_preds.items():
            out[f"pinball_p{int(round(q * 100)):02d}"] = pinball_loss(y_true, qp, q)
    return out


__all__ = ["wmape", "forecast_bias", "pinball_loss", "rmsse", "wrmsse", "evaluate"]
