"""A/B shadow router — score a challenger model on live traffic, serve nothing.

The safest way to learn whether a freshly retrained model is actually better is
to let it see *real* requests. The unsafe part is letting it answer them. This
module splits those two things apart: on a sampled fraction of requests
(``SHADOW_TRAFFIC_PCT``, ~10% by default) the challenger bundle forecasts the
same series/horizon the champion just did, and both sets of numbers are written
to :data:`shadow_predictions` for offline comparison. The dashboard then reads
that table to compare the two prediction distributions before anyone promotes
anything.

Two invariants hold this design up, and both are deliberately enforced here
rather than left to the caller:

* **The shadow output is never served.** :meth:`ShadowRouter.maybe_shadow`
  returns ``None``. There is no code path by which a challenger prediction can
  reach the API response — the caller already has its champion rows and gets
  nothing back to substitute in.
* **The shadow can never break a live request.** Everything inside
  ``maybe_shadow`` runs under a blanket ``except Exception`` that logs and
  swallows: a missing challenger bundle, a raising model, a DB hiccup. A
  monitoring feature that can 500 the serving path is worse than no monitoring
  feature. A failed bundle load is remembered so we don't pay for it again on
  every subsequent request.

Sampling uses an *injectable* :class:`random.Random` so tests (and anyone
reproducing a routing decision) can pin it; the module-level default instance is
only a convenience. The challenger bundle is loaded lazily on the first sampled
request, so importing this module costs nothing and an API that never shadows
never pays for a second model in memory.

Comparisons live in their own table on the *shared* serving engine — same
:func:`serving.store.get_engine`, so ``SERVING_DB_URI`` moves predictions, drift
events and shadow comparisons together and joins across them stay possible.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from serving.store import get_engine as _get_serving_engine

if TYPE_CHECKING:  # pragma: no cover - typing only
    from serving.predictor import ForecastRow

logger = logging.getLogger(__name__)

_metadata = MetaData()

shadow_predictions = Table(
    "shadow_predictions",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset", String(64), nullable=False, index=True),
    Column("series_id", String(256), nullable=False, index=True),
    Column("target_date", Date, nullable=False),
    Column("horizon_step", Integer, nullable=False),
    # p10/p90 are kept for both models on purpose: a challenger can match the
    # champion's median while being far better or worse calibrated, and interval
    # width is exactly the signal that shows it.
    Column("primary_p10", Float, nullable=False),
    Column("primary_p50", Float, nullable=False),
    Column("primary_p90", Float, nullable=False),
    Column("shadow_p10", Float, nullable=False),
    Column("shadow_p50", Float, nullable=False),
    Column("shadow_p90", Float, nullable=False),
    Column("primary_model_version", String(128), nullable=False, default=""),
    Column("shadow_model_version", String(128), nullable=False, default=""),
    Column("predicted_at", DateTime, nullable=False),
)

_created_for: Engine | None = None


def get_engine() -> Engine:
    """The serving engine, with the shadow table ensured on first use.

    Tracks *which* engine the table was created against, so a test that swaps
    ``SERVING_DB_URI`` and calls :func:`serving.store.reset_engine` transparently
    gets a fresh schema without extra bookkeeping.
    """
    global _created_for
    engine = _get_serving_engine()
    if _created_for is not engine:
        _metadata.create_all(engine)
        _created_for = engine
    return engine


def reset_tables() -> None:
    """Forget that the table was created (used by tests that swap the DB URI)."""
    global _created_for
    _created_for = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("shadow: ignoring non-numeric %s=%r", name, raw)
        return default


@dataclass(frozen=True)
class ShadowConfig:
    """Routing configuration, resolved from the environment.

    ``enabled`` defaults to *off*: shadowing doubles inference cost, so it is an
    opt-in an operator turns on for the window in which a challenger is being
    evaluated.
    """

    enabled: bool = False
    traffic_pct: float = 0.1
    dataset: str = "avocado"
    bundle: str | None = None

    @classmethod
    def from_env(cls) -> ShadowConfig:
        """Build a config from ``SHADOW_*`` (falling back to ``SERVING_DATASET``)."""
        dataset = os.environ.get("SHADOW_DATASET") or os.environ.get("SERVING_DATASET", "avocado")
        bundle = os.environ.get("SHADOW_BUNDLE") or None
        pct = _env_float("SHADOW_TRAFFIC_PCT", 0.1)
        return cls(
            enabled=_env_flag("SHADOW_ENABLED", False),
            traffic_pct=min(max(pct, 0.0), 1.0),
            dataset=dataset,
            bundle=bundle,
        )


class _ForecastingModel(Protocol):
    """The slice of :class:`serving.predictor.Predictor` the router needs."""

    model_version: str

    def forecast(self, series_id: str, horizon: int) -> list[Any]: ...


class ShadowRouter:
    """Samples live requests and scores the challenger bundle beside the champion."""

    def __init__(
        self,
        config: ShadowConfig | None = None,
        predictor: _ForecastingModel | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or ShadowConfig.from_env()
        self._predictor = predictor
        self._rng = rng or random.Random()
        self._load_failed = False

    # --- routing -----------------------------------------------------------

    def should_shadow(self, rng: random.Random | None = None) -> bool:
        """Roll for this request. ``rng`` overrides the router's own generator."""
        if not self.config.enabled:
            return False
        pct = self.config.traffic_pct
        if pct <= 0.0:
            return False
        if pct >= 1.0:
            return True
        return (rng or self._rng).random() < pct

    # --- challenger --------------------------------------------------------

    def _load_predictor(self) -> _ForecastingModel | None:
        """Load the challenger lazily; remember failure so we retry at most once."""
        if self._predictor is not None:
            return self._predictor
        if self._load_failed:
            return None
        # Imported here (not at module scope) so merely importing the router
        # doesn't drag lightgbm/pandas into a process that never shadows.
        import joblib

        from serving.model_bundle import load_bundle
        from serving.predictor import Predictor

        bundle = (
            joblib.load(self.config.bundle)
            if self.config.bundle
            else load_bundle(self.config.dataset)
        )
        self._predictor = Predictor(bundle)
        return self._predictor

    # --- entry point -------------------------------------------------------

    def maybe_shadow(
        self,
        series_id: str,
        horizon: int,
        primary_rows: list[ForecastRow],
        primary_model_version: str = "",
    ) -> None:
        """Maybe score the challenger and log both forecasts. Never returns output.

        Fail-safe by construction: any exception — sampling, bundle load,
        inference, persistence — is logged and swallowed, because the caller has
        already produced a valid champion response that must be served.
        """
        try:
            if not primary_rows or not self.should_shadow():
                return
            try:
                predictor = self._load_predictor()
            except Exception:
                self._load_failed = True
                logger.warning("shadow: challenger bundle unavailable; disabling", exc_info=True)
                return
            if predictor is None:
                return
            shadow_rows = predictor.forecast(series_id, horizon)
            log_shadow_comparison(
                dataset=self.config.dataset,
                series_id=series_id,
                primary_rows=primary_rows,
                shadow_rows=shadow_rows,
                primary_model_version=primary_model_version,
                shadow_model_version=getattr(predictor, "model_version", ""),
            )
        except Exception:  # noqa: BLE001 - the whole point is that nothing escapes
            logger.warning("shadow: comparison failed for %s (ignored)", series_id, exc_info=True)


_router: ShadowRouter | None = None


def get_router() -> ShadowRouter:
    """Process-wide router, built lazily from the environment on first use."""
    global _router
    if _router is None:
        _router = ShadowRouter()
    return _router


def reset_router() -> None:
    """Drop the cached router (used by tests and by config reloads)."""
    global _router
    _router = None


# --- persistence -----------------------------------------------------------


def log_shadow_comparison(
    dataset: str,
    series_id: str,
    primary_rows: list[ForecastRow],
    shadow_rows: list[ForecastRow],
    primary_model_version: str = "",
    shadow_model_version: str = "",
    predicted_at: datetime | None = None,
) -> int:
    """Persist paired champion/challenger rows. Returns the number of rows written.

    Rows are paired by target date, so a challenger that emits a different number
    of steps simply contributes fewer comparisons instead of misaligning them.
    """
    ts = predicted_at or datetime.now(UTC)
    shadow_by_date = {r.date: r for r in shadow_rows}
    payload = []
    for i, primary in enumerate(primary_rows):
        shadow = shadow_by_date.get(primary.date)
        if shadow is None:
            continue
        payload.append(
            {
                "dataset": dataset,
                "series_id": series_id,
                "target_date": primary.date,
                "horizon_step": i + 1,
                "primary_p10": float(primary.p10),
                "primary_p50": float(primary.p50),
                "primary_p90": float(primary.p90),
                "shadow_p10": float(shadow.p10),
                "shadow_p50": float(shadow.p50),
                "shadow_p90": float(shadow.p90),
                "primary_model_version": primary_model_version,
                "shadow_model_version": shadow_model_version,
                "predicted_at": ts,
            }
        )
    if not payload:
        return 0
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(insert(shadow_predictions), payload)
    return len(payload)


def fetch_shadow_comparisons(dataset: str | None = None, limit: int = 500) -> list[dict]:
    """Most-recent shadow comparisons (newest first), optionally for one dataset."""
    engine = get_engine()
    stmt = select(shadow_predictions)
    if dataset is not None:
        stmt = stmt.where(shadow_predictions.c.dataset == dataset)
    stmt = stmt.order_by(
        shadow_predictions.c.predicted_at.desc(), shadow_predictions.c.id.desc()
    ).limit(limit)
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt)]


def shadow_divergence(dataset: str | None = None, limit: int = 500) -> dict[str, float | int]:
    """Summarise how far the challenger's medians sit from the champion's.

    Mean *absolute* divergence answers "how different are they at all", mean
    *signed* divergence answers "is the challenger biased high or low", and the
    relative form makes the number comparable across datasets with different
    target scales. ``n == 0`` means nothing has been shadowed yet.
    """
    rows = fetch_shadow_comparisons(dataset=dataset, limit=limit)
    if not rows:
        return {
            "n": 0,
            "mean_abs_divergence": 0.0,
            "mean_signed_divergence": 0.0,
            "mean_abs_pct_divergence": 0.0,
        }
    diffs = [float(r["shadow_p50"]) - float(r["primary_p50"]) for r in rows]
    denom = sum(abs(float(r["primary_p50"])) for r in rows)
    total_abs = sum(abs(d) for d in diffs)
    return {
        "n": len(diffs),
        "mean_abs_divergence": total_abs / len(diffs),
        "mean_signed_divergence": sum(diffs) / len(diffs),
        "mean_abs_pct_divergence": (total_abs / denom) if denom else 0.0,
    }


__all__ = [
    "ShadowConfig",
    "ShadowRouter",
    "fetch_shadow_comparisons",
    "get_engine",
    "get_router",
    "log_shadow_comparison",
    "reset_router",
    "reset_tables",
    "shadow_divergence",
    "shadow_predictions",
]
