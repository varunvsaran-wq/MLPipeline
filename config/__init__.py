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


__all__ = ["DatasetConfig", "CONFIG_DIR", "DATASETS_DIR"]
