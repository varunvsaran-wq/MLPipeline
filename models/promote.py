"""Promotion gate: a candidate only reaches Production if it earns it (Phase 4).

    python -m models.promote --model-name demand-forecasting-lgbm --run-id <run>

The gate exists because "the newest model wins" is how forecasting systems
quietly regress. Instead we compare the candidate's held-out validation metric
against the metric logged by the run behind the *current Production version*, and
require a material improvement before touching the registry. CI calls this
script, so a failing gate exits non-zero and the deploy never happens.

Threshold semantics — deliberately explicit, because this is the part people get
wrong:

* The default metric is **WMAPE, which is lower-is-better**. ``--higher-is-better``
  flips the comparison for metrics like coverage or R².
* The threshold is a **relative** improvement fraction against the incumbent:
  ``improvement = (production - candidate) / |production|`` for lower-is-better
  metrics (and the mirror image otherwise). The default ``0.01`` therefore means
  "at least 1% better than Production", not "1 WMAPE point better". Relative is
  the sane default because it stays meaningful across datasets whose error scales
  differ by orders of magnitude.
* If the incumbent's metric is ~0 the relative form is undefined, so the gate
  falls back to requiring an absolute improvement of at least ``threshold``.
* **The first model promotes automatically.** With no incumbent there is nothing
  to beat, and refusing to promote would mean the platform can never bootstrap.
  That case exits 0. Conversely, re-gating the run that is *already* Production
  is a no-op and exits 1: nothing was promoted, so nothing should deploy.

:func:`should_promote` is pure and side-effect free so the decision rule can be
unit-tested without MLflow; only :func:`run_gate` touches the registry.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field

from models.registry import (
    DEFAULT_ARTIFACT_PATH,
    STAGE_PRODUCTION,
    STAGE_STAGING,
    RegistryUnsupportedError,
    get_client,
    get_production_version,
    register_model,
    run_metrics,
    transition,
    version_metrics,
)

DEFAULT_METRIC = "wmape"
DEFAULT_THRESHOLD = 0.01  # 1% relative improvement over Production
_NEAR_ZERO = 1e-12


@dataclass
class PromotionDecision:
    """Outcome of one gate evaluation — everything needed to explain the verdict."""

    promote: bool
    reason: str
    metric: str = DEFAULT_METRIC
    candidate_metric: float | None = None
    production_metric: float | None = None
    candidate_version: str | None = None
    production_version: str | None = None
    promoted: bool = False
    archived_versions: list[str] = field(default_factory=list)


def should_promote(
    candidate_metric: float | None,
    production_metric: float | None,
    threshold: float = DEFAULT_THRESHOLD,
    higher_is_better: bool = False,
) -> tuple[bool, str]:
    """Decide whether ``candidate_metric`` beats ``production_metric`` by ``threshold``.

    ``production_metric=None`` means there is no incumbent, which promotes. A
    missing or non-finite candidate metric always fails the gate — an unmeasured
    model is not a better model.
    """
    if candidate_metric is None or not math.isfinite(candidate_metric):
        return False, "candidate has no finite validation metric; cannot evaluate the gate"
    if production_metric is None:
        return True, (
            f"no incumbent Production model; promoting candidate ({candidate_metric:.6g}) "
            "as the first version"
        )
    if not math.isfinite(production_metric):
        return True, (
            "incumbent Production metric is not finite; promoting candidate "
            f"({candidate_metric:.6g}) by default"
        )

    raw_gain = (
        candidate_metric - production_metric
        if higher_is_better
        else production_metric - candidate_metric
    )
    direction = "higher-is-better" if higher_is_better else "lower-is-better"

    if abs(production_metric) < _NEAR_ZERO:
        ok = raw_gain >= threshold
        verb = "beats" if ok else "does not beat"
        return ok, (
            f"incumbent metric is ~0, falling back to absolute comparison: candidate "
            f"{candidate_metric:.6g} {verb} production {production_metric:.6g} by "
            f"{raw_gain:.6g} (required >= {threshold:.6g}, {direction})"
        )

    improvement = raw_gain / abs(production_metric)
    ok = improvement >= threshold
    verb = "beats" if ok else "does not beat"
    return ok, (
        f"candidate {candidate_metric:.6g} {verb} production {production_metric:.6g}: "
        f"{improvement * 100:.2f}% relative improvement (required >= {threshold * 100:.2f}%, "
        f"{direction})"
    )


def run_gate(
    model_name: str,
    run_id: str | None = None,
    version: str | int | None = None,
    metric: str = DEFAULT_METRIC,
    threshold: float = DEFAULT_THRESHOLD,
    higher_is_better: bool = False,
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    tracking_uri: str | None = None,
    dry_run: bool = False,
    stage_candidate: bool = True,
) -> PromotionDecision:
    """Evaluate the gate for one candidate and, if it passes, promote it.

    Exactly one of ``run_id`` / ``version`` identifies the candidate. A ``run_id``
    is registered as a new version (and parked in Staging) unless ``dry_run`` is
    set, in which case nothing is written and the candidate metric is read
    straight off the run.

    On promotion the candidate moves to Production with
    ``archive_existing=True``, so the outgoing Production version is archived in
    the same transition.
    """
    if not run_id and version is None:
        raise ValueError("provide either run_id or version to identify the candidate")

    client = get_client(tracking_uri)

    incumbent = get_production_version(model_name, client=client)
    production_metric = None
    if incumbent is not None:
        production_metric = version_metrics(incumbent, client=client).get(metric)

    candidate_version: str | None = None
    if version is not None:
        candidate_mv = client.get_model_version(model_name, str(version))
        candidate_version = candidate_mv.version
        candidate_metric = version_metrics(candidate_mv, client=client).get(metric)
        candidate_run = candidate_mv.run_id
    else:
        candidate_run = run_id
        candidate_metric = run_metrics(str(run_id), client=client).get(metric)

    # Re-running the gate on the live model must not register a duplicate version.
    if incumbent is not None and candidate_run and incumbent.run_id == candidate_run:
        return PromotionDecision(
            promote=False,
            reason=f"candidate run {candidate_run} is already the Production model; nothing to do",
            metric=metric,
            candidate_metric=candidate_metric,
            production_metric=production_metric,
            candidate_version=candidate_version or incumbent.version,
            production_version=incumbent.version,
        )

    if version is None and not dry_run:
        candidate_mv = register_model(
            model_name, str(run_id), artifact_path=artifact_path, client=client
        )
        candidate_version = candidate_mv.version
        if stage_candidate:
            transition(model_name, candidate_version, STAGE_STAGING, client=client)

    promote, reason = should_promote(
        candidate_metric, production_metric, threshold=threshold, higher_is_better=higher_is_better
    )
    decision = PromotionDecision(
        promote=promote,
        reason=reason,
        metric=metric,
        candidate_metric=candidate_metric,
        production_metric=production_metric,
        candidate_version=candidate_version,
        production_version=incumbent.version if incumbent else None,
    )

    if promote and not dry_run and candidate_version is not None:
        transition(
            model_name,
            candidate_version,
            STAGE_PRODUCTION,
            archive_existing=True,
            client=client,
            description=reason,
        )
        decision.promoted = True
        if incumbent is not None:
            decision.archived_versions = [incumbent.version]
    return decision


def format_verdict(model_name: str, decision: PromotionDecision, dry_run: bool) -> str:
    """Human-readable summary printed by CI."""
    head = "PROMOTE" if decision.promote else "BLOCK"
    lines = [
        f"[{head}] {model_name} ({decision.metric}{' — dry run' if dry_run else ''})",
        f"  candidate : v{decision.candidate_version or '?'} = {decision.candidate_metric}",
        f"  production: v{decision.production_version or '-'} = {decision.production_metric}",
        f"  reason    : {decision.reason}",
    ]
    if decision.promoted:
        lines.append(f"  action    : v{decision.candidate_version} -> Production")
        if decision.archived_versions:
            lines.append("  archived  : " + ", ".join(f"v{v}" for v in decision.archived_versions))
    elif decision.promote and dry_run:
        lines.append("  action    : none (dry run)")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLflow promotion gate (Phase 4).")
    p.add_argument("--model-name", required=True, help="registered model name")
    p.add_argument("--run-id", default=None, help="MLflow run that produced the candidate")
    p.add_argument("--version", default=None, help="existing registered version to evaluate")
    p.add_argument("--metric", default=DEFAULT_METRIC, help="validation metric to gate on")
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="required relative improvement over Production (0.01 = 1%%)",
    )
    p.add_argument(
        "--higher-is-better",
        action="store_true",
        help="metric improves upward (default assumes lower-is-better, e.g. WMAPE)",
    )
    p.add_argument("--artifact-path", default=DEFAULT_ARTIFACT_PATH)
    p.add_argument("--tracking-uri", default=None, help="overrides MLFLOW_TRACKING_URI")
    p.add_argument(
        "--no-staging",
        action="store_true",
        help="skip the intermediate Staging transition for newly registered candidates",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="evaluate the gate without writing to the registry"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. 0 = promoted (or nothing to beat), 1 = gate failed, 2 = misconfigured."""
    args = _parse_args(argv)
    try:
        decision = run_gate(
            model_name=args.model_name,
            run_id=args.run_id,
            version=args.version,
            metric=args.metric,
            threshold=args.threshold,
            higher_is_better=args.higher_is_better,
            artifact_path=args.artifact_path,
            tracking_uri=args.tracking_uri,
            dry_run=args.dry_run,
            stage_candidate=not args.no_staging,
        )
    except (RegistryUnsupportedError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(format_verdict(args.model_name, decision, args.dry_run))
    return 0 if decision.promote else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_METRIC",
    "DEFAULT_THRESHOLD",
    "PromotionDecision",
    "format_verdict",
    "main",
    "run_gate",
    "should_promote",
]
