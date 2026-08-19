"""Model leaderboard source for ``GET /model/leaderboard``.

The handoff wants "registered models ranked by validation WMAPE". The MLflow
registry isn't populated until Phase 4, so for now the leaderboard is sourced,
in order of preference, from:

1. ``models/leaderboard.json`` — written by ``models/run_comparison.py`` (the
   full Prophet-vs-LightGBM comparison across every metric), then
2. the served bundle's own validation metrics (a single-row fallback so the
   endpoint still works in a fresh container that only has a baked bundle).

Entries are always sorted by WMAPE ascending (best first).
"""

from __future__ import annotations

import json
from pathlib import Path

LEADERBOARD_JSON = Path(__file__).resolve().parents[1] / "models" / "leaderboard.json"


def _sorted_entries(models: dict[str, dict]) -> list[dict]:
    rows = [{"model": name, **metrics} for name, metrics in models.items()]
    rows.sort(key=lambda r: r.get("wmape", float("inf")))
    return rows


def load_leaderboard(dataset: str, fallback_metrics: dict | None = None) -> dict:
    """Return ``{"dataset", "entries": [...]}`` ranked by validation WMAPE."""
    if LEADERBOARD_JSON.exists():
        data = json.loads(LEADERBOARD_JSON.read_text(encoding="utf-8"))
        if data.get("models"):
            return {
                "dataset": data.get("dataset", dataset),
                "entries": _sorted_entries(data["models"]),
            }
    if fallback_metrics:
        return {
            "dataset": dataset,
            "entries": _sorted_entries({"LightGBM (served)": fallback_metrics}),
        }
    return {"dataset": dataset, "entries": []}


__all__ = ["load_leaderboard", "LEADERBOARD_JSON"]
