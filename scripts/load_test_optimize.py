#!/usr/bin/env python3
import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    p = max(0.0, min(1.0, p))
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = p * (len(ordered) - 1)
    low = int(idx)
    high = min(len(ordered) - 1, low + 1)
    frac = idx - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


class IdempotencyCounter:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._count = 0

    def next(self) -> str:
        with self._lock:
            self._count += 1
            return f"{self.prefix}-{self._count}-{uuid.uuid4().hex[:8]}"


def build_request(
    url: str,
    payload_raw: bytes,
    session_cookie: str,
    idempotency_key: str | None,
) -> urllib.request.Request:
    request = urllib.request.Request(url, method="POST", data=payload_raw)
    request.add_header("Content-Type", "application/json")
    request.add_header("Cookie", session_cookie)
    if idempotency_key:
        request.add_header("Idempotency-Key", idempotency_key)
    return request


def run_one(
    *,
    url: str,
    payload_raw: bytes,
    timeout: float,
    session_cookie: str,
    idempotency: IdempotencyCounter | None,
) -> dict[str, Any]:
    started = time.perf_counter()
    status = 0
    error = None
    idempotency_key = idempotency.next() if idempotency else None

    try:
        request = build_request(url, payload_raw, session_cookie, idempotency_key)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.getcode())
            _ = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        error = f"http_{exc.code}"
    except Exception as exc:
        error = str(exc)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {"status": status, "elapsed_ms": elapsed_ms, "error": error}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concurrent load test for POST /optimize or POST /optimize/async.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--endpoint", default="/optimize/async", help="Target endpoint")
    parser.add_argument("--payload-file", required=True, help="JSON payload file for optimize request")
    parser.add_argument(
        "--session-cookie",
        required=True,
        help="Cookie header value, e.g. 'spotify_opt_sid=...'",
    )
    parser.add_argument("--requests", type=int, default=60, help="Total request count")
    parser.add_argument("--concurrency", type=int, default=6, help="Concurrent workers")
    parser.add_argument("--timeout-seconds", type=float, default=40.0, help="Per-request timeout")
    parser.add_argument(
        "--idempotency-prefix",
        default="loadtest",
        help="Idempotency key prefix; empty value disables Idempotency-Key header",
    )
    parser.add_argument(
        "--max-fail-rate",
        type=float,
        default=0.05,
        help="Exit non-zero if failed request ratio exceeds this threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"{args.base_url.rstrip('/')}/{args.endpoint.lstrip('/')}"

    with open(args.payload_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload_raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    idempotency = None
    if args.idempotency_prefix.strip():
        idempotency = IdempotencyCounter(args.idempotency_prefix.strip())

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [
            pool.submit(
                run_one,
                url=url,
                payload_raw=payload_raw,
                timeout=args.timeout_seconds,
                session_cookie=args.session_cookie,
                idempotency=idempotency,
            )
            for _ in range(max(1, args.requests))
        ]
        for future in as_completed(futures):
            results.append(future.result())
    total_ms = (time.perf_counter() - started) * 1000.0

    latencies = [float(item["elapsed_ms"]) for item in results]
    failures = [item for item in results if item.get("error") or int(item.get("status", 0)) >= 400]
    status_counts: dict[int, int] = {}
    for item in results:
        status = int(item.get("status") or 0)
        status_counts[status] = status_counts.get(status, 0) + 1

    fail_rate = len(failures) / max(1, len(results))
    rps = (len(results) * 1000.0) / max(1.0, total_ms)

    print(f"target={url}")
    print(f"requests={len(results)} concurrency={max(1, args.concurrency)} total_ms={total_ms:.1f} rps={rps:.2f}")
    print(
        "latency_ms "
        f"mean={statistics.mean(latencies):.2f} "
        f"p50={percentile(latencies, 0.50):.2f} "
        f"p95={percentile(latencies, 0.95):.2f} "
        f"p99={percentile(latencies, 0.99):.2f}"
    )
    print(f"status_counts={dict(sorted(status_counts.items()))}")
    print(f"failures={len(failures)} fail_rate={fail_rate:.4f}")

    if failures:
        sample = failures[:5]
        print("failure_samples:")
        for item in sample:
            print(f"  status={item.get('status')} error={item.get('error')}")

    if fail_rate > max(0.0, float(args.max_fail_rate)):
        print(
            f"FAIL: fail_rate {fail_rate:.4f} exceeds max_fail_rate {float(args.max_fail_rate):.4f}",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
