"""Phase 6 unit tests: the A/B shadow router.

The core suite here is dependency-light (fake predictors, temp SQLite), so it
runs in the light CI job. The invariants under test are the ones the design
promises: the shadow output never reaches the caller, a broken challenger never
breaks a live request, routing is reproducible under a seeded RNG, and both
forecasts land in the comparison table. One end-to-end test over a real
:class:`ModelBundle` self-skips when lightgbm isn't installed.
"""

from __future__ import annotations

import random
from datetime import date
from types import SimpleNamespace

import pytest

from serving import shadow
from serving.shadow import ShadowConfig, ShadowRouter


def _rows(n: int = 3, base: float = 1.0) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            date=date(2026, 1, 7 + 7 * i),
            p10=base + i - 0.1,
            p50=base + i,
            p90=base + i + 0.1,
        )
        for i in range(n)
    ]


class _FakePredictor:
    model_version = "challenger@abc-2026-07-22"

    def __init__(self, offset: float = 0.5) -> None:
        self.offset = offset
        self.calls: list[tuple[str, int]] = []

    def forecast(self, series_id: str, horizon: int) -> list[SimpleNamespace]:
        self.calls.append((series_id, horizon))
        return _rows(horizon, base=1.0 + self.offset)


class _BoomPredictor:
    model_version = "boom"

    def forecast(self, series_id: str, horizon: int) -> list[SimpleNamespace]:
        raise RuntimeError("challenger exploded")


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    from serving import store

    monkeypatch.setenv("SERVING_DB_URI", f"sqlite:///{(tmp_path / 'shadow.db').as_posix()}")
    store.reset_engine()
    shadow.reset_tables()
    shadow.reset_router()
    yield store
    store.reset_engine()
    shadow.reset_tables()
    shadow.reset_router()


def _cfg(**kw) -> ShadowConfig:
    base = {"enabled": True, "traffic_pct": 1.0, "dataset": "avocado", "bundle": None}
    base.update(kw)
    return ShadowConfig(**base)


# --- config ----------------------------------------------------------------


def test_config_defaults_are_off_at_ten_percent(monkeypatch):
    for var in ("SHADOW_ENABLED", "SHADOW_TRAFFIC_PCT", "SHADOW_DATASET", "SHADOW_BUNDLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("SERVING_DATASET", raising=False)
    cfg = ShadowConfig.from_env()
    assert cfg.enabled is False
    assert cfg.traffic_pct == pytest.approx(0.1)
    assert cfg.dataset == "avocado"
    assert cfg.bundle is None


def test_config_reads_env_and_clamps(monkeypatch):
    monkeypatch.setenv("SHADOW_ENABLED", "true")
    monkeypatch.setenv("SHADOW_TRAFFIC_PCT", "5")
    monkeypatch.setenv("SHADOW_DATASET", "retail")
    monkeypatch.setenv("SHADOW_BUNDLE", "/tmp/b.joblib")
    cfg = ShadowConfig.from_env()
    assert cfg.enabled is True
    assert cfg.traffic_pct == 1.0  # clamped into [0, 1]
    assert cfg.dataset == "retail"
    assert cfg.bundle == "/tmp/b.joblib"


def test_config_ignores_garbage_pct(monkeypatch):
    monkeypatch.setenv("SHADOW_TRAFFIC_PCT", "not-a-number")
    assert ShadowConfig.from_env().traffic_pct == pytest.approx(0.1)


# --- routing invariants ----------------------------------------------------


def test_disabled_never_shadows():
    router = ShadowRouter(config=_cfg(enabled=False), predictor=_FakePredictor())
    assert not any(router.should_shadow(random.Random(i)) for i in range(50))


def test_zero_pct_never_shadows():
    router = ShadowRouter(config=_cfg(traffic_pct=0.0), predictor=_FakePredictor())
    assert not any(router.should_shadow(random.Random(i)) for i in range(50))


def test_full_pct_always_shadows():
    router = ShadowRouter(config=_cfg(traffic_pct=1.0), predictor=_FakePredictor())
    assert all(router.should_shadow(random.Random(i)) for i in range(50))


def test_seeded_rng_is_reproducible():
    router = ShadowRouter(config=_cfg(traffic_pct=0.1), predictor=_FakePredictor())
    rng_a, rng_b = random.Random(7), random.Random(7)
    first = [router.should_shadow(rng_a) for _ in range(200)]
    second = [router.should_shadow(rng_b) for _ in range(200)]
    assert first == second
    assert any(first) and not all(first)  # 10% actually samples both ways
    assert 5 <= sum(first) <= 40  # roughly 10% of 200


def test_injected_rng_beats_router_rng():
    # the router's own generator would produce one sequence; passing an explicit
    # rng must override it, which is what makes routing testable at all
    router = ShadowRouter(config=_cfg(traffic_pct=0.5), rng=random.Random(0))
    expected = [random.Random(1).random() < 0.5]
    assert [router.should_shadow(random.Random(1)) for _ in range(4)] == expected * 4


# --- maybe_shadow ----------------------------------------------------------


def test_maybe_shadow_returns_none_and_logs_both(temp_store):
    fake = _FakePredictor(offset=0.5)
    router = ShadowRouter(config=_cfg(traffic_pct=1.0), predictor=fake)
    primary = _rows(3)

    assert router.maybe_shadow("A|conv", 3, primary, "champion@v1") is None
    assert fake.calls == [("A|conv", 3)]

    rows = shadow.fetch_shadow_comparisons(dataset="avocado")
    assert len(rows) == 3
    row = rows[0]
    assert row["series_id"] == "A|conv"
    assert row["primary_model_version"] == "champion@v1"
    assert row["shadow_model_version"] == fake.model_version
    # both predictions are present and differ by the fake's offset
    assert row["shadow_p50"] - row["primary_p50"] == pytest.approx(0.5)
    assert row["primary_p10"] < row["primary_p50"] < row["primary_p90"]
    assert row["shadow_p10"] < row["shadow_p50"] < row["shadow_p90"]


def test_maybe_shadow_does_not_mutate_primary_rows(temp_store):
    router = ShadowRouter(config=_cfg(traffic_pct=1.0), predictor=_FakePredictor())
    primary = _rows(3)
    before = [(r.date, r.p10, r.p50, r.p90) for r in primary]
    router.maybe_shadow("A|conv", 3, primary, "champion@v1")
    assert [(r.date, r.p10, r.p50, r.p90) for r in primary] == before


def test_maybe_shadow_skips_when_not_sampled(temp_store):
    fake = _FakePredictor()
    router = ShadowRouter(config=_cfg(traffic_pct=0.0), predictor=fake)
    router.maybe_shadow("A|conv", 3, _rows(3), "champion@v1")
    assert fake.calls == []
    assert shadow.fetch_shadow_comparisons() == []


def test_raising_shadow_model_does_not_propagate(temp_store):
    router = ShadowRouter(config=_cfg(traffic_pct=1.0), predictor=_BoomPredictor())
    assert router.maybe_shadow("A|conv", 3, _rows(3), "champion@v1") is None
    assert shadow.fetch_shadow_comparisons() == []


def test_missing_bundle_does_not_propagate_and_disables(temp_store, tmp_path):
    router = ShadowRouter(config=_cfg(traffic_pct=1.0, bundle=str(tmp_path / "nope.joblib")))
    assert router.maybe_shadow("A|conv", 3, _rows(3), "champion@v1") is None
    assert router._load_failed is True
    # second call short-circuits without another (expensive) load attempt
    assert router.maybe_shadow("A|conv", 3, _rows(3), "champion@v1") is None
    assert shadow.fetch_shadow_comparisons() == []


def test_empty_primary_rows_are_a_no_op(temp_store):
    fake = _FakePredictor()
    router = ShadowRouter(config=_cfg(traffic_pct=1.0), predictor=fake)
    router.maybe_shadow("A|conv", 3, [], "champion@v1")
    assert fake.calls == []


def test_mismatched_dates_are_paired_not_misaligned(temp_store):
    primary = _rows(3)
    shadow_rows = _rows(3, base=2.0)
    shadow_rows[1].date = date(2030, 1, 1)  # challenger emitted a date we didn't
    n = shadow.log_shadow_comparison("avocado", "A|conv", primary, shadow_rows, "v1", "v2")
    assert n == 2
    assert {r["target_date"] for r in shadow.fetch_shadow_comparisons()} == {
        primary[0].date,
        primary[2].date,
    }


# --- aggregate helper ------------------------------------------------------


def test_shadow_divergence_summarises(temp_store):
    router = ShadowRouter(config=_cfg(traffic_pct=1.0), predictor=_FakePredictor(offset=0.5))
    router.maybe_shadow("A|conv", 4, _rows(4), "champion@v1")
    agg = shadow.shadow_divergence(dataset="avocado")
    assert agg["n"] == 4
    assert agg["mean_abs_divergence"] == pytest.approx(0.5)
    assert agg["mean_signed_divergence"] == pytest.approx(0.5)
    assert agg["mean_abs_pct_divergence"] > 0


def test_shadow_divergence_empty(temp_store):
    agg = shadow.shadow_divergence(dataset="avocado")
    assert agg["n"] == 0
    assert agg["mean_abs_divergence"] == 0.0


def test_get_router_is_a_lazy_singleton(temp_store):
    shadow.reset_router()
    first = shadow.get_router()
    assert shadow.get_router() is first
    shadow.reset_router()
    assert shadow.get_router() is not first


# --- end-to-end over a real bundle -----------------------------------------


def test_router_with_real_bundle(temp_store):
    pytest.importorskip("lightgbm", reason="lightgbm not installed ([models] extra)")

    import numpy as np
    import pandas as pd

    from config import DatasetConfig, FeatureConfig, FourierTerm
    from features.pipeline import build_feature_matrix
    from models.lightgbm_model import _fit_quantile_models
    from serving.model_bundle import ModelBundle
    from serving.predictor import Predictor

    cfg = DatasetConfig.load("avocado")
    cfg.features = FeatureConfig(
        lags=[1, 2],
        rolling_windows=[4],
        fourier=[FourierTerm(period=52.143, order=1)],
        exogenous_lag=1,
        use_exogenous=True,
    )
    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range("2018-01-07", periods=n, freq="W")
    panel = pd.DataFrame(
        {
            "Date": dates,
            "AveragePrice": 1.0 + 0.2 * np.sin(np.arange(n) / 8.0) + rng.normal(0, 0.02, n),
            "Total Volume": np.linspace(1000, 2000, n),
            "region": "Alpha",
            "type": "conventional",
        }
    )
    frame, spec = build_feature_matrix(cfg, panel)
    bundle = ModelBundle(
        dataset="avocado",
        cfg=cfg,
        models=_fit_quantile_models(frame, spec),
        spec=spec,
        categories={c: list(frame[c].cat.categories) for c in spec.categorical},
        history=panel,
        feature_schema={},
        metrics={},
        git_commit="test",
        trained_at="2026-07-22T00:00:00+00:00",
    )
    predictor = Predictor(bundle)
    primary = predictor.forecast("Alpha|conventional", 4)

    router = ShadowRouter(config=_cfg(traffic_pct=1.0), predictor=predictor)
    assert router.maybe_shadow("Alpha|conventional", 4, primary, predictor.model_version) is None

    rows = shadow.fetch_shadow_comparisons(dataset="avocado")
    assert len(rows) == 4
    # same model on both sides -> identical medians, zero divergence
    assert shadow.shadow_divergence(dataset="avocado")["mean_abs_divergence"] == pytest.approx(0.0)
