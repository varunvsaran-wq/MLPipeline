"""Temporal Fusion Transformer over the full panel (Phase 4).

The third model family in the leaderboard, and the one that fills the gap the
first two leave. Prophet is *local* (one fit per series, no cross-learning);
LightGBM is *global* but tabular, so multi-step forecasts have to be produced
recursively, feeding each step's p50 back in as pseudo-history. The TFT is
global *and* natively multi-horizon: a single forward pass emits the whole
``cfg.horizon`` at once, so there is no error-feedback loop and no recursive
drift.

It is also probabilistic by construction. Training against
``QuantileLoss(quantiles=[0.1, 0.5, 0.9])`` means the p10 / p50 / p90 the
harness scores come straight out of the network's own output head rather than
from three separately fitted models (LightGBM) or a parametric interval
(Prophet). p50 is the point forecast, matching the other two families.

Design notes:

* **Leakage.** The held-out window is the final ``cfg.horizon`` timestamps of
  the panel, chosen by a single global ``time_idx`` cutoff — the same
  convention as :func:`features.pipeline.panel_train_test_split` and
  :mod:`models.lightgbm_model`, so the three leaderboard rows are scored on
  identical rows. The training ``TimeSeriesDataSet`` is built from data at or
  before the cutoff only; the validation set reuses its fitted encoders and
  normalisers via ``from_dataset(..., predict=True)``, which yields exactly one
  decoder window per series ending at the last observation.
* **Known vs unknown.** Calendar features are derived from the date alone, so
  they are legitimately *known* over the forecast window. The target and any
  ``cfg.exogenous_cols`` are *unknown* future reals — the network may only read
  them inside the encoder. This mirrors the ``exogenous_lag`` discipline in the
  tabular pipeline.
* **Import cost.** ``torch`` / ``pytorch-forecasting`` live in the optional
  ``[deep]`` extra and cost seconds and gigabytes to import. Every deep import
  is therefore made inside a function, exactly as :mod:`models.run_comparison`
  keeps its heavy model imports local: importing this module is free and safe
  in an environment that has no torch at all.
* **Defaults.** Small hidden size, few epochs, CPU accelerator. The point of
  this run is a comparable leaderboard entry on a laptop, not a tuned model;
  every knob is a keyword argument so a real training run can scale them up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import DatasetConfig
from features.pipeline import series_key
from models.evaluate import QUANTILES, SeriesForecast

# Defaults deliberately sized for a CPU smoke run; override per call.
DEFAULT_MAX_EPOCHS = 5
DEFAULT_HIDDEN_SIZE = 16
DEFAULT_ATTENTION_HEAD_SIZE = 2
DEFAULT_HIDDEN_CONTINUOUS_SIZE = 8
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 0.03
DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42

# Calendar columns derived purely from the date, hence known over the horizon.
_CALENDAR_COLS = ("month", "weekofyear", "quarter", "dayofweek", "is_weekend")


def _series_weight(train_rows: pd.DataFrame, cfg: DatasetConfig) -> float:
    """WRMSSE weight: dollar-style volume if an exogenous volume col exists, else |target|.

    Identical convention to :mod:`models.lightgbm_model` and
    :mod:`models.prophet_panel` — the weights must match across families or the
    WRMSSE column stops being comparable.
    """
    if cfg.exogenous_cols and cfg.exogenous_cols[0] in train_rows:
        return float(train_rows[cfg.exogenous_cols[0]].abs().sum())
    return float(train_rows[cfg.target_col].abs().sum())


def _group_cols(cfg: DatasetConfig) -> list[str]:
    """Columns identifying a series, falling back to a synthetic single group."""
    return list(cfg.series_id_cols) or ["series_id"]


def _join_ids(frame: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    """Rebuild the canonical ``"A|B"`` series id from the group columns."""
    return frame[group_cols].astype(str).agg("|".join, axis=1)


def build_panel(cfg: DatasetConfig, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Shape the raw panel into the frame ``TimeSeriesDataSet`` expects.

    Adds a contiguous integer ``time_idx`` derived from the *global* date
    ordering (so all series share one clock and one cutoff), the canonical
    ``series_id``, and the date-derived calendar reals. Returns the frame plus
    the known / unknown real column lists.
    """
    df = raw.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df["series_id"] = series_key(cfg, df).values
    df = df.dropna(subset=[cfg.target_col])
    df = df.sort_values(["series_id", cfg.date_col]).reset_index(drop=True)

    # One global clock: rank of the date among all observed timestamps.
    dates = np.sort(df[cfg.date_col].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    df["time_idx"] = df[cfg.date_col].map(date_to_idx).astype("int64")

    group_cols = _group_cols(cfg)
    for col in group_cols:
        df[col] = df[col].astype(str)

    # TimeSeriesDataSet requires at most one row per (group, time_idx).
    df = df.drop_duplicates(subset=[*group_cols, "time_idx"], keep="first").reset_index(drop=True)

    dt = df[cfg.date_col].dt
    df["month"] = dt.month.astype("float32")
    df["weekofyear"] = dt.isocalendar().week.astype("float32")
    df["quarter"] = dt.quarter.astype("float32")
    df["dayofweek"] = dt.dayofweek.astype("float32")
    df["is_weekend"] = (dt.dayofweek >= 5).astype("float32")

    df[cfg.target_col] = df[cfg.target_col].astype("float32")
    known_reals = ["time_idx", *_CALENDAR_COLS]

    unknown_reals = [cfg.target_col]
    for col in cfg.exogenous_cols:
        if col in df.columns and col != cfg.target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df[col] = df.groupby("series_id", sort=False)[col].transform(
                lambda s: s.ffill().bfill()
            )
            df[col] = df[col].fillna(0.0)
            unknown_reals.append(col)

    return df, known_reals, unknown_reals


def _encoder_length(panel: pd.DataFrame, group_cols: list[str], horizon: int) -> int:
    """Longest lookback every series can actually supply after the cutoff."""
    shortest = int(panel.groupby(group_cols, sort=False).size().min())
    # Want a few horizons of context; cap at what the shortest series affords.
    return max(1, min(4 * horizon, shortest - horizon))


def _make_datasets(
    cfg: DatasetConfig,
    panel: pd.DataFrame,
    known_reals: list[str],
    unknown_reals: list[str],
    max_encoder_length: int,
) -> tuple[Any, Any, int]:
    """Build the train / validation ``TimeSeriesDataSet`` pair around one cutoff."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    group_cols = _group_cols(cfg)
    # Global cutoff: everything after it is the held-out horizon, for every series.
    cutoff = int(panel["time_idx"].max()) - cfg.horizon

    training = TimeSeriesDataSet(
        panel[panel["time_idx"] <= cutoff],
        time_idx="time_idx",
        target=cfg.target_col,
        group_ids=group_cols,
        min_encoder_length=max(1, max_encoder_length // 2),
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=cfg.horizon,
        static_categoricals=group_cols,
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        target_normalizer=GroupNormalizer(groups=group_cols),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )
    # predict=True -> exactly one decoder window per series, the final `horizon`
    # steps; encoders/normalisers are inherited from `training` (fit on train only).
    validation = TimeSeriesDataSet.from_dataset(
        training, panel, predict=True, stop_randomization=True
    )
    return training, validation, cutoff


def _lightning() -> Any:
    """Return the Lightning namespace, tolerating both packagings."""
    try:
        import lightning.pytorch as pl
    except ImportError:  # pragma: no cover - depends on the installed extra
        import pytorch_lightning as pl
    return pl


def train_and_forecast(
    cfg: DatasetConfig,
    raw: pd.DataFrame,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    attention_head_size: int = DEFAULT_ATTENTION_HEAD_SIZE,
    hidden_continuous_size: int = DEFAULT_HIDDEN_CONTINUOUS_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_encoder_length: int | None = None,
    accelerator: str = "cpu",
    seed: int = DEFAULT_SEED,
) -> tuple[list[SeriesForecast], Any, dict[str, Any]]:
    """Train one global TFT and forecast the held-out horizon for every series.

    Returns the per-series forecasts in the shared :class:`SeriesForecast`
    contract, the trained ``TemporalFusionTransformer``, and a dict of params
    worth logging to MLflow.

    All torch / pytorch-forecasting imports happen inside this call, so the
    module stays importable without the ``[deep]`` extra installed.
    """
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss

    pl = _lightning()
    pl.seed_everything(seed, workers=True)

    panel, known_reals, unknown_reals = build_panel(cfg, raw)
    group_cols = _group_cols(cfg)
    if max_encoder_length is None:
        max_encoder_length = _encoder_length(panel, group_cols, cfg.horizon)

    training, validation, cutoff = _make_datasets(
        cfg, panel, known_reals, unknown_reals, max_encoder_length
    )
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size * 4, num_workers=0)

    quantiles = list(QUANTILES)
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        loss=QuantileLoss(quantiles=quantiles),
        output_size=len(quantiles),
        optimizer="adam",
        log_interval=-1,
        reduce_on_plateau_patience=2,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        gradient_clip_val=0.1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    forecasts = _predict_forecasts(cfg, model, val_loader, panel, cutoff, accelerator)
    info = {
        "max_epochs": max_epochs,
        "hidden_size": hidden_size,
        "attention_head_size": attention_head_size,
        "hidden_continuous_size": hidden_continuous_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_encoder_length": max_encoder_length,
        "max_prediction_length": cfg.horizon,
        "n_series": len(forecasts),
        "n_known_reals": len(known_reals),
        "n_unknown_reals": len(unknown_reals),
        "quantiles": ",".join(str(q) for q in quantiles),
    }
    return forecasts, model, info


def _predict_forecasts(
    cfg: DatasetConfig,
    model: Any,
    val_loader: Any,
    panel: pd.DataFrame,
    cutoff: int,
    accelerator: str,
) -> list[SeriesForecast]:
    """Run the validation pass and reassemble it into :class:`SeriesForecast`s."""
    prediction = model.predict(
        val_loader,
        mode="quantiles",
        return_index=True,
        trainer_kwargs={
            "accelerator": accelerator,
            "devices": 1,
            "logger": False,
            "enable_progress_bar": False,
        },
    )
    output = prediction.output
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    # (n_series, horizon, n_quantiles); enforce non-crossing quantiles per step.
    preds = np.sort(np.asarray(output, dtype=float), axis=-1)

    index = prediction.index.reset_index(drop=True)
    group_cols = _group_cols(cfg)
    pred_ids = _join_ids(index, group_cols).to_numpy()
    # `index.time_idx` is the first decoder step of each window.
    first_idx = index["time_idx"].to_numpy(dtype=int)

    actuals = {
        (sid, int(tidx)): float(val)
        for sid, tidx, val in zip(
            panel["series_id"], panel["time_idx"], panel[cfg.target_col], strict=True
        )
    }
    train_panel = panel[panel["time_idx"] <= cutoff]
    train_groups = dict(list(train_panel.groupby("series_id", sort=False)))

    forecasts: list[SeriesForecast] = []
    for row, sid in enumerate(pred_ids):
        # Decoder steps are contiguous in time_idx from the window's first step.
        # A series missing a timestamp simply contributes fewer scored points.
        keep = [k for k in range(cfg.horizon) if (sid, int(first_idx[row]) + k) in actuals]
        if not keep:
            continue
        y_true = np.array([actuals[(sid, int(first_idx[row]) + k)] for k in keep], dtype=float)
        quantiles = {q: preds[row, keep, i].astype(float) for i, q in enumerate(QUANTILES)}
        train_rows = train_groups.get(sid)
        if train_rows is None or train_rows.empty:
            continue
        forecasts.append(
            SeriesForecast(
                series_id=str(sid),
                y_true=y_true,
                point=quantiles[0.5],
                quantiles=quantiles,
                train_history=train_rows[cfg.target_col].to_numpy(dtype=float),
                weight=_series_weight(train_rows, cfg),
            )
        )
    return forecasts


__all__ = [
    "train_and_forecast",
    "build_panel",
    "DEFAULT_MAX_EPOCHS",
    "DEFAULT_HIDDEN_SIZE",
    "DEFAULT_BATCH_SIZE",
]
