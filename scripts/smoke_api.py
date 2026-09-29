#!/usr/bin/env python3
"""End-to-end API smoke check used by the one-shot `verify` compose service.

Hits the health endpoint and then analyzes one recoverable chime fragment
and one unsolvable fragment.  Exits 0 only when every assertion holds.

Usage: SMOKE_BASE_URL=http://web:8000 python scripts/smoke_api.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("SMOKE_BASE_URL", "http://web:8000").rstrip("/")
TIMEOUT = 10

SOLVABLE = {
    "peaks": [
        {"frequency": 200.4, "tolerance": 1.5},
        {"frequency": 299.6, "tolerance": 1.5},
        {"frequency": 401.1, "tolerance": 1.5},
        {"frequency": 555.2, "tolerance": 1.5},
        {"frequency": 600.3, "tolerance": 1.5},
        {"frequency": 799.7, "tolerance": 1.5},
        {"frequency": 900.5, "tolerance": 1.5},
    ],
    "f0_min": 80,
    "f0_max": 130,
    "max_harmonic": 12,
    "max_rejected": 1,
}

UNSOLVABLE = {
    "peaks": [
        {"frequency": 200.0, "tolerance": 2.0},
        {"frequency": 256.0, "tolerance": 2.0},
        {"frequency": 300.0, "tolerance": 2.0},
        {"frequency": 384.0, "tolerance": 2.0},
        {"frequency": 512.0, "tolerance": 2.0},
        {"frequency": 600.0, "tolerance": 2.0},
    ],
    "f0_min": 90,
    "f0_max": 130,
    "max_harmonic": 12,
    "max_rejected": 1,
}


def get(path: str) -> dict:
    with urllib.request.urlopen(BASE_URL + path, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def post(path: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def main() -> int:
    failures: list[str] = []

    def check(cond: bool, message: str) -> None:
        print(("PASS" if cond else "FAIL") + " - " + message)
        if not cond:
            failures.append(message)

    # 1. Health endpoint.
    health = get("/health")
    check(health.get("status") == "ok", f"GET /health -> {health}")

    # 2. Recoverable fragment.
    status, data = post("/api/analyze", SOLVABLE)
    check(status == 200, f"solvable analyze HTTP {status}")
    if status == 200:
        check(data["feasible"] is True, "solvable fragment is feasible")
        check(99.0 <= (data.get("f0") or 0) <= 101.0,
              f"solvable f0 ~ 100 Hz (got {data.get('f0')})")
        check(data["adopted_count"] == 6,
              f"solvable adopts 6 peaks (got {data.get('adopted_count')})")
        check(data["rejected_count"] == 1,
              f"solvable rejects 1 peak (got {data.get('rejected_count')})")
        hs = [p["harmonic"] for p in data["peaks"] if p["adopted"]]
        check(hs == sorted(set(hs)),
              f"adopted harmonics strictly increasing/unique: {hs}")
        joint = all(
            abs(p["frequency"] - p["harmonic"] * data["f0"])
            <= p["tolerance"] + 1e-9
            for p in data["peaks"] if p["adopted"]
        )
        check(joint, "single f0 lies inside every adopted peak's tolerance")

    # 3. Unsolvable fragment.
    status, data = post("/api/analyze", UNSOLVABLE)
    check(status == 200, f"unsolvable analyze HTTP {status}")
    if status == 200:
        check(data["feasible"] is False, "unsolvable fragment reports infeasible")
        check(data.get("f0") is None, "no fundamental reported when infeasible")
        check(data.get("first_unexplainable_index") == 3,
              f"first unexplainable peak index is 3 / 384 Hz "
              f"(got {data.get('first_unexplainable_index')})")
        freqs = [p["frequency"] for p in data["peaks"]]
        check(freqs == [200.0, 256.0, 300.0, 384.0, 512.0, 600.0],
              "original peak values preserved on no-solution result")
        check(all(not p["adopted"] for p in data["peaks"]),
              "no peak silently adopted when infeasible")

    if failures:
        print(f"\nSMOKE FAILED: {len(failures)} assertion(s) failed")
        return 1
    print("\nSMOKE OK: health, recoverable and unsolvable fragments verified")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # network errors etc.
        print(f"SMOKE ERROR: {type(exc).__name__}: {exc}")
        sys.exit(1)
