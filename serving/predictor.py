"""Inference engine: recursively forecast future dates for one series.

The training-time forecaster in ``models/lightgbm_model.py`` predicts a *held-out*
horizon that already exists in the data. Serving is different: the requested
dates lie in the future, so we synthesise them, then walk the horizon one step at
a time — predicting p10/p50/p90, feeding the p50 back in as the realised value so
the next step's lag/rolling features are populated. This reuses
:func:`build_feature_matrix`, so serving features can never drift from training
features.

Two serving-time assumptions worth stating plainly:

* **Exogenous regressors are unknown in the future**, so we carry the last
  observed value forward (naive persistence) before computing their lagged
  features. A real deployment would pass known-ahead regressors instead.
* **Category codes are pinned** to the training categories captured in the
  bundle, because a single-series frame would otherwise re-index its categorical
  columns and feed LightGBM the wrong codes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from features.pipeline import build_feature_matrix, series_key
from models.evaluate import QUANTILES
from serving.model_bundle import ModelBundle


@dataclass
class ForecastRow:
    date: date
    p10: float
    p50: float
    p90: float


class SeriesNotFoundError(KeyError):
    """Raised when a requested series id isn't present in the bundle history."""


class Predictor:
    """Loads a :class:`ModelBundle` and answers forecast requests."""

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.cfg = bundle.cfg
        self._history = bundle.history.copy()
        self._history[self.cfg.date_col] = pd.to_datetime(self._history[self.cfg.date_col])
        self._history["__sid__"] = series_key(self.cfg, self._history).values

    @property
    def model_version(self) -> str:
        return self.bundle.model_version

    def series_ids(self) -> list[str]:
        return sorted(self._history["__sid__"].unique().tolist())

    def actuals(self, series_id: str) -> dict[date, float]:
        """Observed target values for a series, keyed by date (for history joins)."""
        g = self._history[self._history["__sid__"] == series_id]
        return {
            pd.Timestamp(d).date(): float(v)
            for d, v in zip(g[self.cfg.date_col], g[self.cfg.target_col], strict=True)
        }

    def _future_dates(self, last: pd.Timestamp, horizon: int) -> list[pd.Timestamp]:
        offset = pd.tseries.frequencies.to_offset(self.cfg.freq)
        out, cur = [], last
        for _ in range(horizon):
            cur = cur + offset
            out.append(cur)
        return out

    def _pin_categories(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Force categorical columns onto the bundle's training categories."""
        for col, cats in self.bundle.categories.items():
            frame[col] = pd.Categorical(frame[col].astype(str), categories=[str(c) for c in cats])
        return frame

    def forecast(self, series_id: str, horizon: int) -> list[ForecastRow]:
        """Recursively forecast ``horizon`` future steps for ``series_id``."""
        cfg = self.cfg
        hist = self._history[self._history["__sid__"] == series_id]
        if hist.empty:
            raise SeriesNotFoundError(series_id)
        hist = hist.sort_values(cfg.date_col)

        last_date = hist[cfg.date_col].max()
        future_dates = self._future_dates(last_date, horizon)

        # Build the working frame: real history + blank future rows for this series.
        base_cols = [cfg.date_col, cfg.target_col, *cfg.series_id_cols, *cfg.exogenous_cols]
        base_cols = [c for c in dict.fromkeys(base_cols) if c in hist.columns]
        work = hist[base_cols].copy()

        id_values = {c: hist.iloc[-1][c] for c in cfg.series_id_cols}
        exog_fill = {c: hist.iloc[-1][c] for c in cfg.exogenous_cols if c in hist.columns}
        future = pd.DataFrame(
            {
                cfg.date_col: future_dates,
                cfg.target_col: [np.nan] * horizon,
                **{c: [v] * horizon for c, v in id_values.items()},
                **{c: [v] * horizon for c, v in exog_fill.items()},
            }
        )
        work = pd.concat([work, future[base_cols]], ignore_index=True)

        rows: list[ForecastRow] = []
        for dt in future_dates:
            fm, _ = build_feature_matrix(cfg, work)
            fm = self._pin_categories(fm)
            cur = fm[fm["ds"] == dt]
            x = cur[self.bundle.spec.all]
            qpred = np.array([self.bundle.models[q].predict(x)[0] for q in QUANTILES])
            qpred = np.sort(qpred)  # enforce non-crossing quantiles
            qmap = dict(zip(QUANTILES, qpred, strict=True))
            rows.append(
                ForecastRow(
                    date=pd.Timestamp(dt).date(),
                    p10=float(qmap[0.1]),
                    p50=float(qmap[0.5]),
                    p90=float(qmap[0.9]),
                )
            )
            # feed p50 back so the next step's lag/rolling features are populated
            work.loc[work[cfg.date_col] == dt, cfg.target_col] = float(qmap[0.5])

        return rows


__all__ = ["Predictor", "ForecastRow", "SeriesNotFoundError"]
