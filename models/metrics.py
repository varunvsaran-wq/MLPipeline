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


def evaluate(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Point-forecast metric bundle logged to MLflow each run."""
    return {
        "wmape": wmape(y_true, y_pred),
        "bias": forecast_bias(y_true, y_pred),
    }


__all__ = ["wmape", "forecast_bias", "pinball_loss", "evaluate"]
