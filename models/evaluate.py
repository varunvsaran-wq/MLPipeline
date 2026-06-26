"""Shared evaluation harness so every model family is scored identically.

Each trainer returns a list of :class:`SeriesForecast` (one per series over the
held-out horizon); :func:`aggregate` pools them into the metric bundle —
WMAPE / bias / pinball pooled across series, and WRMSSE weighted across them.
This is what makes the Prophet-vs-LightGBM leaderboard an apples-to-apples
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.metrics import evaluate, wrmsse

QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class SeriesForecast:
    """One series' validation forecast plus what's needed to score it."""

    series_id: str
    y_true: np.ndarray
    point: np.ndarray  # point forecast (the p50 quantile)
    quantiles: dict[float, np.ndarray]  # {0.1: .., 0.5: .., 0.9: ..}
    train_history: np.ndarray  # training-period actuals (for RMSSE scaling)
    weight: float  # series importance (e.g. dollar volume) for WRMSSE


def aggregate(forecasts: list[SeriesForecast]) -> dict[str, float]:
    """Pool per-series forecasts into the logged metric bundle."""
    if not forecasts:
        raise ValueError("no forecasts to aggregate")
    y_true = np.concatenate([f.y_true for f in forecasts])
    point = np.concatenate([f.point for f in forecasts])
    quantile_preds = {q: np.concatenate([f.quantiles[q] for f in forecasts]) for q in QUANTILES}
    metrics = evaluate(y_true, point, quantile_preds)
    metrics["wrmsse"] = wrmsse([(f.y_true, f.point, f.train_history, f.weight) for f in forecasts])
    metrics["n_series"] = float(len(forecasts))
    return metrics


__all__ = ["SeriesForecast", "aggregate", "QUANTILES"]
