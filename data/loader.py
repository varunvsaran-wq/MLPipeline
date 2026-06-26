"""Dataset-agnostic raw data loading and series selection.

Every pipeline stage reads through these helpers so that switching datasets is
a config change (see :class:`config.DatasetConfig`), never a code edit. The raw
files themselves are DVC-tracked; only the pointers live in git, so call
``dvc pull`` before running if ``data/raw/`` is empty.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import DatasetConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def load_raw(cfg: DatasetConfig) -> pd.DataFrame:
    """Load the raw CSV for ``cfg``, parse the date column, and sort by date."""
    path = RAW_DIR / cfg.raw_filename
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data for '{cfg.name}' not found at {path}. "
            "Run `dvc pull` (or fetch the dataset) first."
        )
    df = pd.read_csv(path)
    if cfg.date_col not in df.columns:
        raise KeyError(
            f"Expected date column '{cfg.date_col}' in {path.name}; got columns {list(df.columns)}"
        )
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    return df.sort_values(cfg.date_col).reset_index(drop=True)


def select_series(
    cfg: DatasetConfig,
    df: pd.DataFrame,
    selectors: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Filter ``df`` to a single series and return a tidy ``[date, target]`` frame.

    ``selectors`` maps series-id columns to the value to keep
    (e.g. ``{"region": "TotalUS", "type": "conventional"}``). When omitted, the
    dataset's ``default_series`` is used. The result is one row per period,
    aggregated by mean if the selection is still not unique per date.
    """
    selectors = selectors if selectors is not None else dict(cfg.default_series)
    sub = df
    for col, value in selectors.items():
        if col not in sub.columns:
            raise KeyError(f"Series-id column '{col}' not in data; have {list(sub.columns)}")
        sub = sub[sub[col] == value]
    if sub.empty:
        raise ValueError(f"No rows for series {selectors!r} in dataset '{cfg.name}'.")

    out = (
        sub.groupby(cfg.date_col, as_index=False)[cfg.target_col]
        .mean()
        .sort_values(cfg.date_col)
        .reset_index(drop=True)
    )
    return out


__all__ = ["load_raw", "select_series", "PROJECT_ROOT", "RAW_DIR"]
