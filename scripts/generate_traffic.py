"""Send realistic forecast traffic to a running API so the dashboard has something to show.

    python scripts/generate_traffic.py [--url http://localhost:8000] [--requests 60]

The ops dashboard's volume, histogram and prediction-log panels are empty until
somebody calls ``/forecast``. This script plays that somebody: it asks the API
which series it can forecast, then fires a mix of single and batch requests at
random series and horizons. Every one of them lands in the prediction log exactly
as real traffic would (and, with ``SHADOW_ENABLED=1``, feeds the shadow router).

Standard library only, so it runs from any environment that can reach the API.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://localhost:8000"


def _call(url: str, payload: dict | None = None, timeout: float = 60.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_api(base_url: str, timeout: float = 60.0) -> bool:
    """Poll ``/health`` until it answers 200 or ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _call(f"{base_url}/health", timeout=5).get("status") == "ok":
                return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1)
    return False


def generate(
    base_url: str = DEFAULT_URL,
    requests: int = 60,
    max_horizon: int = 12,
    batch_every: int = 5,
    delay: float = 0.0,
    seed: int | None = None,
    quiet: bool = False,
) -> int:
    """Fire ``requests`` forecast calls; returns the number of series forecasted."""
    rng = random.Random(seed)
    series = _call(f"{base_url}/model/series")["series_ids"]
    if not series:
        raise RuntimeError("the API reports no forecastable series")

    forecasted = 0
    for i in range(1, requests + 1):
        horizon = rng.randint(1, max_horizon)
        if batch_every and i % batch_every == 0:
            ids = rng.sample(series, k=min(len(series), rng.randint(2, 6)))
            _call(f"{base_url}/forecast/batch", {"series_ids": ids, "horizon": horizon})
            forecasted += len(ids)
            label = f"batch x{len(ids)}"
        else:
            sid = rng.choice(series)
            _call(f"{base_url}/forecast", {"series_id": sid, "horizon": horizon})
            forecasted += 1
            label = sid
        if not quiet:
            print(f"  [{i:>3}/{requests}] h={horizon:<2} {label}")
        if delay:
            time.sleep(delay)
    return forecasted


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Send demo forecast traffic to the API.")
    p.add_argument("--url", default=DEFAULT_URL, help="API base URL")
    p.add_argument("--requests", type=int, default=60, help="number of API calls")
    p.add_argument("--max-horizon", type=int, default=12)
    p.add_argument("--delay", type=float, default=0.0, help="seconds between calls")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    base = args.url.rstrip("/")
    if not wait_for_api(base, timeout=10):
        print(f"API not reachable at {base} - is it running?", file=sys.stderr)
        return 2
    try:
        n = generate(base, args.requests, args.max_horizon, delay=args.delay, seed=args.seed)
    except urllib.error.HTTPError as exc:
        print(f"API error {exc.code}: {exc.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    print(f"sent {args.requests} requests covering {n} series forecasts to {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
