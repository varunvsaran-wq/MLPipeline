"""Phase 0 smoke tests.

These intentionally only exercise the scaffold: the test suite must run green
before any modelling code exists. Later phases replace/extend these with real
unit (feature transforms, schemas) and integration tests.
"""

from __future__ import annotations

from config import DatasetConfig


def test_suite_runs():
    """The empty suite must be collectable and green (Phase 0 acceptance)."""
    assert True


def test_dataset_configs_load():
    """Both shipped dataset configs parse into a valid DatasetConfig."""
    for name in ("avocado", "m5"):
        cfg = DatasetConfig.load(name)
        assert cfg.name == name
        assert cfg.date_col
        assert cfg.target_col
        assert cfg.horizon > 0


def test_unknown_dataset_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        DatasetConfig.load("does-not-exist")
