"""Dataset-agnostic configuration.

A dataset is described entirely by a YAML file under ``config/datasets/``.
Switching from Avocado (Phase 1) to M5 (Phase 2) is meant to be a one-line
config change, not a code rewrite, so every pipeline stage reads its settings
from a :class:`DatasetConfig` rather than hard-coding column names.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent
DATASETS_DIR = CONFIG_DIR / "datasets"


class FourierTerm(BaseModel):
    """One seasonal Fourier component: ``period`` base-frequency steps, ``order`` harmonics."""

    period: float
    order: int = 3


class FeatureConfig(BaseModel):
    """Tabular feature suite controls (Phase 2). All units are base-freq steps.

    Defaults are daily-oriented (M5); the avocado config overrides with weekly
    lags/windows so the same pipeline serves both datasets.
    """

    lags: list[int] = Field(default_factory=lambda: [1, 7, 14, 28])
    rolling_windows: list[int] = Field(default_factory=lambda: [7, 28])
    fourier: list[FourierTerm] = Field(default_factory=list)
    # Lag (in base-freq steps) applied to exogenous regressors so only
    # information available at forecast time leaks into features.
    exogenous_lag: int = 1
    use_exogenous: bool = True


class DatasetConfig(BaseModel):
    """Schema for a single dataset definition."""

    name: str
    # Column roles — the pipeline only ever refers to these, never literals.
    date_col: str
    target_col: str
    # Columns that identify a single series (e.g. ["store_id", "item_id"]).
    series_id_cols: list[str] = Field(default_factory=list)
    # Optional exogenous regressors known at/ before forecast time.
    exogenous_cols: list[str] = Field(default_factory=list)
    # Forecast horizon in periods (days for Avocado/M5).
    horizon: int = 28
    # Calendar frequency, pandas offset alias.
    freq: str = "D"
    # ISO country code for the `holidays` library.
    holiday_country: str | None = None
    # Raw file under data/raw/ (defaults to "<name>.csv" if unset).
    raw_file: str | None = None
    # Default single series to use for the univariate baseline (Phase 1),
    # as {series_id_col: value}. Empty means the data is already univariate.
    default_series: dict[str, str] = Field(default_factory=dict)
    # Tabular feature suite controls (Phase 2 / LightGBM).
    features: FeatureConfig = Field(default_factory=FeatureConfig)

    @property
    def raw_filename(self) -> str:
        """Resolved raw filename, defaulting to ``<name>.csv``."""
        return self.raw_file or f"{self.name}.csv"

    @classmethod
    def load(cls, name: str) -> DatasetConfig:
        """Load ``config/datasets/<name>.yaml``."""
        path = DATASETS_DIR / f"{name}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in DATASETS_DIR.glob("*.yaml"))
            raise FileNotFoundError(f"No dataset config '{name}'. Available: {available}")
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return cls(**data)


__all__ = ["DatasetConfig", "FeatureConfig", "FourierTerm", "CONFIG_DIR", "DATASETS_DIR"]
