"""Compose 一次性 verify 服务：等待健康端点 → 代码单元测试 → API 冒烟。

对一份"可复原"残音（含杂峰）与一份"无解"残音分别冒烟，校验：
  * 健康端点 200；
  * 可复原：feasible=true、采用数、严格递增序号、统一基频容差；
  * 无解：feasible=false、原始峰值完整保留、最早证据峰定位正确。

全部通过退出码 0，任一失败非零退出。用法：
    python verify.py [--base-url http://web:8000] [--wait-seconds 60]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


RECOVERABLE = {
    "frequencies": [300.0, 400.5, 499.2, 600.3, 668.0, 800.8, 901.5, 1002.0],
    "tolerances": [1.5, 1.5, 1.5, 1.5, 0.8, 1.5, 1.5, 1.5],
    "f0_min": 95, "f0_max": 105, "max_harmonic": 12, "max_rejected": 2,
}

UNSOLVABLE = {
    # 300/400/500/600 共享 f0≈100；668 与联合交集互不相容；
    # 846 在候选区间内无归属；只允许剔除 1 条 → 无解。
    "frequencies": [300, 400, 500, 600, 668, 846],
    "tolerances": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
    "f0_min": 95, "f0_max": 105, "max_harmonic": 12, "max_rejected": 1,
}


def _request(base, path, method="GET", payload=None, timeout=10):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base.rstrip("/") + path, data=data,
                                 headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def wait_health(base, seconds):
    deadline = time.time() + seconds
    last_err = None
    while time.time() < deadline:
        try:
            status, body = _request(base, "/healthz", timeout=3)
            if status == 200 and body.get("status") == "ok":
                return True
        except (urllib.error.URLError, ConnectionError, OSError, json.JSONDecodeError) as exc:
            last_err = exc
        time.sleep(1.0)
    print(f"[verify] ✗ 健康端点在 {seconds}s 内未就绪：{last_err}")
    return False


def run_unit_tests():
    print("[verify] === 代码单元测试（联合算法 + 暴力对拍）===")
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()


def check(label, ok, detail=""):
    mark = "✓" if ok else "✗"
    print(f"[verify] {mark} {label}" + (f" —— {detail}" if detail else ""))
    return bool(ok)


def smoke_recoverable(base):
    print("[verify] === API 冒烟：可复原残音（含 1 条杂峰）===")
    try:
        status, body = _request(base, "/api/solve", "POST", RECOVERABLE)
    except Exception as exc:  # noqa: BLE001
        return check("可复原请求", False, str(exc))
    ok = True
    ok &= check("HTTP 200", status == 200, f"实际 {status}")
    ok &= check("feasible=true", body.get("feasible") is True)
    sol = body.get("solution", {})
    peaks = sol.get("peaks", [])
    ok &= check("采用 7 条、剔除 1 条",
                sol.get("adopted_count") == 7 and sol.get("rejected_count") == 1,
                f"实际 {sol.get('adopted_count')}/{sol.get('rejected_count')}")
    harmonics = [h for h in sol.get("harmonics", []) if h]
    ok &= check("泛音序号严格递增不重复",
                bool(harmonics) and harmonics == sorted(set(harmonics)),
                str(harmonics))
    f0 = sol.get("f0")
    tol_ok = f0 is not None and all(
        abs(p["frequency"] - p["harmonic"] * f0) <= p["tolerance"] + 1e-9
        for p in peaks if p["adopted"]
    )
    ok &= check("同一基频同时落入所有采用峰容差", tol_ok, f"f0={f0}")
    ok &= check("第 5 条 668 被剔除", sol.get("harmonics", [None] * 8)[4] is None)
    ev = body.get("evidence")
    ok &= check("证据指向第 5 条杂峰", bool(ev) and ev.get("index") == 4,
                f"evidence={ev.get('index') if ev else None}")
    return ok


def smoke_unsolvable(base):
    print("[verify] === API 冒烟：无解残音（页面必须保留原始峰值）===")
    try:
        status, body = _request(base, "/api/solve", "POST", UNSOLVABLE)
    except Exception as exc:  # noqa: BLE001
        return check("无解药请求", False, str(exc))
    ok = True
    ok &= check("HTTP 200", status == 200, f"实际 {status}")
    ok &= check("feasible=false（无共同基频）", body.get("feasible") is False)
    sol = body.get("solution", {})
    peaks = sol.get("peaks", [])
    # 原始峰值完整保留，不被部分方案覆盖
    retained = [p["frequency"] for p in peaks]
    ok &= check("原始峰值完整保留且顺序不变",
                retained == UNSOLVABLE["frequencies"], str(retained))
    ok &= check("无解时不给出 f0", sol.get("f0") is None)
    ev = body.get("evidence")
    ok &= check("指出按峰值顺序最早无法纳入的证据峰（第 5 条 668）",
                bool(ev) and ev.get("index") == 4,
                f"evidence={ev.get('index') if ev else None}")
    if ev:
        ok &= check("最大联合子集 4 < 要求的 5 条",
                    ev.get("optimal_count") == 4 and ev.get("required_count") == 5,
                    f"{ev.get('optimal_count')} vs {ev.get('required_count')}")
        ok &= check("强制纳入证据峰时联合采用数下降",
                    ev.get("max_count_with_peak", 99) < 4,
                    str(ev.get("max_count_with_peak")))
    return ok


def smoke_static_page(base):
    try:
        req = urllib.request.Request(base.rstrip("/") + "/", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            ok = resp.status == 200 and "联合基频" in body
            return check("前端页面 GET / 返回 200 且含应用内容", ok,
                         f"status={resp.status}, {len(body)} bytes")
    except Exception as exc:  # noqa: BLE001
        return check("前端页面 GET /", False, str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url",
                        default="http://web:8000",
                        help="被测服务地址（默认 compose 服务名 http://web:8000）")
    parser.add_argument("--wait-seconds", type=int, default=90)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    print(f"[verify] 目标服务：{args.base_url}")
    if not wait_health(args.base_url, args.wait_seconds):
        return 1
    print("[verify] ✓ 健康端点就绪")

    results = []
    if not args.skip_tests:
        results.append(run_unit_tests())
    results.append(smoke_static_page(args.base_url))
    results.append(smoke_recoverable(args.base_url))
    results.append(smoke_unsolvable(args.base_url))

    if all(results):
        print("[verify] ✅ 全部通过：构建产物、代码测试、健康端点与两份残音冒烟均成功。")
        return 0
    print("[verify] ❌ 存在失败项，详见上文。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
