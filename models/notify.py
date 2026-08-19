"""Outbound notifications for the retraining loop (Phase 6).

A retrain that nobody hears about is indistinguishable from a retrain that never
happened, so :mod:`models.retrain` ends every run — promoted or blocked — by
sending a summary here. The module is deliberately tiny and has three
properties that matter more than features:

* **No new dependency.** Slack incoming webhooks accept a plain JSON POST, which
  :mod:`urllib.request` can do; pulling in ``requests`` or an email SDK for one
  HTTP call would be a poor trade in a container we keep lean.
* **A sane default when nothing is configured.** With ``SLACK_WEBHOOK_URL``
  unset the message is printed to stdout. That is the common local case, and it
  keeps the CLI's step-by-step log complete instead of silently dropping the
  most interesting part on a developer machine.
* **Never fatal.** Every failure path — bad URL, timeout, non-2xx response — is
  caught and reported in the returned :class:`NotificationResult`. A flaky
  webhook must not fail a retraining run that has already promoted a model; the
  authoritative record of what happened is the retraining event row, not the
  Slack message.

:func:`format_metrics_diff` is here rather than in the retrainer because "what
changed between the old model and the new one" is the payload of the
notification, and rendering it is pure string work that tests can pin exactly.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import IO

SLACK_ENV_VAR = "SLACK_WEBHOOK_URL"
DEFAULT_TIMEOUT = 5.0

#: Metrics shown first in a diff, in the order a forecasting reviewer reads them.
_METRIC_ORDER = (
    "wmape",
    "wrmsse",
    "pinball_p10",
    "pinball_p50",
    "pinball_p90",
    "bias",
    "n_series",
)

#: Metrics whose *increase* is an improvement. Everything else (WMAPE, WRMSSE,
#: pinball losses) is lower-is-better, which is the forecasting norm.
_HIGHER_IS_BETTER = frozenset({"coverage", "r2", "accuracy"})

_MISSING = "-"


@dataclass
class NotificationResult:
    """Where a notification went, and whether it got there."""

    delivered: bool
    channel: str  # "slack" | "stdout" | "none"
    error: str | None = None


def slack_webhook_url(explicit: str | None = None) -> str | None:
    """The configured webhook: explicit argument, then ``SLACK_WEBHOOK_URL``."""
    url = explicit or os.environ.get(SLACK_ENV_VAR, "")
    url = url.strip()
    return url or None


def _to_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out


def _numeric(metrics: Mapping[str, object] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        number = _to_float(value)
        if number is not None:
            out[str(key)] = number
    return out


def _ordered_keys(before: Mapping[str, float], after: Mapping[str, float]) -> list[str]:
    known = [k for k in _METRIC_ORDER if k in before or k in after]
    rest = sorted((set(before) | set(after)) - set(known))
    return known + rest


def _verdict(key: str, delta: float, higher_is_better: Iterable[str]) -> str:
    if delta == 0.0:
        return "same"
    improved = delta > 0 if key in set(higher_is_better) else delta < 0
    return "better" if improved else "worse"


def format_metrics_diff(
    before: Mapping[str, object] | None,
    after: Mapping[str, object] | None,
    keys: Iterable[str] | None = None,
    higher_is_better: Iterable[str] = _HIGHER_IS_BETTER,
) -> str:
    """Render a before/after metrics table with deltas and a better/worse verdict.

    Values that are missing on one side are shown as ``-`` and carry no delta:
    the incumbent may simply not have logged a metric the candidate reports, and
    inventing a comparison there would be worse than admitting the gap. Percent
    change is relative to the *before* value and omitted when that value is zero.
    """
    b = _numeric(before)
    a = _numeric(after)
    selected = list(keys) if keys is not None else _ordered_keys(b, a)
    if not selected:
        return "(no metrics recorded)"

    header = ("metric", "before", "after", "delta", "change", "")
    rows: list[tuple[str, str, str, str, str, str]] = [header]
    for key in selected:
        bv, av = b.get(key), a.get(key)
        if bv is None or av is None:
            rows.append(
                (
                    key,
                    f"{bv:.4f}" if bv is not None else _MISSING,
                    f"{av:.4f}" if av is not None else _MISSING,
                    _MISSING,
                    _MISSING,
                    _MISSING,
                )
            )
            continue
        delta = av - bv
        pct = f"{delta / abs(bv) * 100:+.2f}%" if bv != 0 else _MISSING
        rows.append(
            (
                key,
                f"{bv:.4f}",
                f"{av:.4f}",
                f"{delta:+.4f}",
                pct,
                _verdict(key, delta, higher_is_better),
            )
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_message(title: str, body: str) -> str:
    """One plain-text block: a title line, a rule, then the body."""
    title = title.strip()
    rule = "=" * max(len(title), 8)
    return f"{title}\n{rule}\n{body.rstrip()}"


def _post_slack(url: str, text: str, timeout: float) -> None:
    """POST ``{"text": ...}`` to a Slack incoming webhook. Raises on failure."""
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - URL comes from operator config
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        status = getattr(response, "status", None) or response.getcode()
        if not 200 <= int(status) < 300:
            raise urllib.error.HTTPError(url, int(status), "unexpected status", None, None)


def send(
    title: str,
    body: str,
    webhook_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    stream: IO[str] | None = None,
) -> NotificationResult:
    """Send one notification; fall back to ``stream`` (stdout) when Slack is unset.

    Returns a :class:`NotificationResult` and never raises: callers are in the
    middle of a retraining run and must not lose it to a notification problem.
    A Slack failure still prints the message locally, so the content survives
    even when delivery does not.
    """
    message = format_message(title, body)
    url = slack_webhook_url(webhook_url)
    out = stream if stream is not None else sys.stdout

    if url is None:
        try:
            print(message, file=out)
        except Exception as exc:  # noqa: BLE001 - a broken stream is still not fatal
            return NotificationResult(delivered=False, channel="none", error=str(exc))
        return NotificationResult(delivered=True, channel="stdout")

    try:
        _post_slack(url, message, timeout)
    except Exception as exc:  # noqa: BLE001 - any transport error degrades to stdout
        try:
            print(message, file=out)
        except Exception:  # noqa: BLE001 - nothing left to try
            pass
        return NotificationResult(delivered=False, channel="slack", error=str(exc))
    return NotificationResult(delivered=True, channel="slack")


__all__ = [
    "DEFAULT_TIMEOUT",
    "SLACK_ENV_VAR",
    "NotificationResult",
    "format_message",
    "format_metrics_diff",
    "send",
    "slack_webhook_url",
]
