#!/usr/bin/env python3
import argparse
import json
import time
import urllib.error
import urllib.request
import uuid


def request_json(
    method: str,
    url: str,
    *,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict]:
    raw = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, method=method.upper(), data=raw, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.getcode())
            body = resp.read().decode("utf-8")
            return status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        parsed = {}
        if body:
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = {"raw": body}
        return int(exc.code), parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for async optimize flow.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--payload-file", required=True, help="Optimize request payload JSON file")
    parser.add_argument(
        "--session-cookie",
        required=True,
        help="Cookie header value, e.g. 'spotify_opt_sid=...'",
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0, help="Status poll interval")
    parser.add_argument("--timeout-seconds", type=float, default=600.0, help="End-to-end timeout")
    parser.add_argument(
        "--check-ready",
        action="store_true",
        help="Call /ready before starting the optimize run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_url.rstrip("/")
    headers = {
        "Cookie": args.session_cookie,
        "Idempotency-Key": f"smoke-{uuid.uuid4().hex}",
    }

    if args.check_ready:
        ready_status, ready_payload = request_json("GET", f"{base}/ready", headers=headers, timeout=20.0)
        if ready_status != 200:
            print(f"FAIL: /ready status={ready_status} body={ready_payload}")
            return 1
        print("ready_check=ok")

    with open(args.payload_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    status, queued = request_json(
        "POST",
        f"{base}/optimize/async",
        payload=payload,
        headers=headers,
        timeout=30.0,
    )
    if status != 200:
        print(f"FAIL: queue request status={status} body={queued}")
        return 1

    run_id = queued.get("run_id")
    if not run_id:
        print(f"FAIL: missing run_id in queue response: {queued}")
        return 1
    print(f"queued_run_id={run_id}")

    deadline = time.time() + max(5.0, args.timeout_seconds)
    final_status = None
    final_payload = {}
    while time.time() < deadline:
        poll_status, poll_payload = request_json(
            "GET",
            f"{base}/optimize/{run_id}",
            headers=headers,
            timeout=20.0,
        )
        if poll_status != 200:
            print(f"FAIL: status polling failed status={poll_status} body={poll_payload}")
            return 1
        state = str(poll_payload.get("status") or "").lower()
        progress = int(poll_payload.get("progress") or 0)
        print(f"status={state} progress={progress}")
        if state in {"completed", "failed"}:
            final_status = state
            final_payload = poll_payload
            break
        time.sleep(max(0.2, args.poll_interval_seconds))

    if final_status is None:
        print(f"FAIL: run {run_id} timed out after {args.timeout_seconds:.1f}s")
        return 1
    if final_status != "completed":
        print(f"FAIL: run {run_id} ended in status={final_status} payload={final_payload}")
        return 1

    result = final_payload.get("result") or {}
    playlist_id = result.get("playlist_id")
    transition_score = result.get("transition_score")
    print(f"completed_run_id={run_id} playlist_id={playlist_id} transition_score={transition_score}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
