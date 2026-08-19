"""Data validation — cheap, dependency-light range/schema assertions.

Runs in CI (and can gate retraining later) so a malformed or corrupted dataset
fails fast with a clear message instead of surfacing as a bizarre model result.
These are custom assertions rather than a Great Expectations suite to keep the
default CI job light; the checks are dataset-agnostic, driven by
:class:`config.DatasetConfig`.

CLI::

    python -m data.validation --dataset avocado    # exit 1 on any failure
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from config import DatasetConfig
from data.loader import load_raw


class ValidationError(Exception):
    """Raised when a dataset fails validation (aggregates all failures)."""


def validate_frame(cfg: DatasetConfig, df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems (empty means the data is clean)."""
    problems: list[str] = []

    # Required columns present.
    required = [cfg.date_col, cfg.target_col, *cfg.series_id_cols]
    for col in required:
        if col not in df.columns:
            problems.append(f"missing required column: {col!r}")
    if problems:  # can't check further without the core columns
        return problems

    # Dates parse and there are no nulls in key columns.
    if df[cfg.date_col].isna().any():
        problems.append(f"{cfg.date_col!r} has unparseable/null dates")
    if df[cfg.target_col].isna().any():
        n = int(df[cfg.target_col].isna().sum())
        problems.append(f"{cfg.target_col!r} has {n} null value(s)")

    # Target is numeric, finite, and non-negative (demand/price can't be < 0).
    target = pd.to_numeric(df[cfg.target_col], errors="coerce")
    if target.isna().any() and not df[cfg.target_col].isna().any():
        problems.append(f"{cfg.target_col!r} has non-numeric values")
    finite = target.replace([np.inf, -np.inf], np.nan).dropna()
    if not finite.empty and (finite < 0).any():
        problems.append(f"{cfg.target_col!r} has negative values (min={finite.min():.4g})")

    # Each series must have enough history to hold out the validation horizon.
    if cfg.series_id_cols:
        counts = df.groupby(cfg.series_id_cols).size()
        too_short = counts[counts <= cfg.horizon]
        if not too_short.empty:
            problems.append(
                f"{len(too_short)} series have <= horizon ({cfg.horizon}) points; "
                f"smallest has {int(counts.min())}"
            )

    # No duplicate (series, date) rows — would corrupt lag features.
    dup_keys = [*cfg.series_id_cols, cfg.date_col]
    n_dups = int(df.duplicated(subset=dup_keys).sum())
    if n_dups:
        problems.append(f"{n_dups} duplicate (series, date) row(s)")

    return problems


def validate_dataset(dataset: str) -> list[str]:
    cfg = DatasetConfig.load(dataset)
    df = load_raw(cfg)
    return validate_frame(cfg, df)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a dataset's raw data (Phase 3 CI gate).")
    p.add_argument("--dataset", default="avocado")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        issues = validate_dataset(args.dataset)
    except FileNotFoundError as exc:
        print(f"SKIP: {exc}")
        sys.exit(0)  # absent DVC data isn't a validation failure in CI
    if issues:
        print(f"FAIL: {args.dataset} has {len(issues)} problem(s):")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)
    print(f"OK: {args.dataset} passed all data validation checks")


__all__ = ["validate_frame", "validate_dataset", "ValidationError"]
