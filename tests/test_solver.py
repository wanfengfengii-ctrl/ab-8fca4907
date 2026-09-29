"""solver 的单元测试与随机对拍（仅依赖标准库 unittest）。

随机用例用独立的暴力枚举（全部剔除/采用 × 严格递增序号组合）做对照，
确保联合搜索结果与穷举最优一致，而不是逐峰就近拼接的结果。
"""

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.solver import (  # noqa: E402
    EPS_FEAS,
    Solution,
    ValidationError,
    _better,
    _evaluate,
    solve,
    validate_input,
)


def assert_well_formed(testcase, result, expect_feasible=None):
    sol = result.solution
    hs = [h for h in sol.assignment if h]
    # 严格递增且不重复
    testcase.assertEqual(hs, sorted(set(hs)))
    testcase.assertTrue(all(h >= 1 for h in hs))
    n = len(result.frequencies)
    testcase.assertEqual(len(sol.peaks), n)
    rejected = [i for i, h in enumerate(sol.assignment) if not h]
    if result.feasible:
        testcase.assertLessEqual(len(rejected), result.max_rejected)
    if expect_feasible is not None:
        testcase.assertIs(result.feasible, expect_feasible)
    if result.feasible:
        # 同一 f0 必须同时落入每条采用峰容差
        for i, h in enumerate(sol.assignment):
            if h:
                testcase.assertLessEqual(
                    abs(result.frequencies[i] - h * sol.f0),
                    result.tolerances[i] + EPS_FEAS * max(1.0, result.frequencies[i]),
                )
        testcase.assertGreaterEqual(sol.f0, result.f0_min - 1e-9)
        testcase.assertLessEqual(sol.f0, result.f0_max + 1e-9)
    # 原始峰值始终保留
    testcase.assertEqual([p.frequency for p in sol.peaks], result.frequencies)
    testcase.assertEqual([p.tolerance for p in sol.peaks], result.tolerances)
    return sol


def brute_force(freq, tol, flo, fhi, H, R):
    """独立穷举：所有 (剔除/采用, 严格递增 h) 组合，按题面四级标准选优。

    采用数必须满足剔除预算（≥ n-R）；否则返回 None（无解）。
    """
    n = len(freq)
    required = max(1, n - R)
    best = {"sol": None}

    def rec(i, used_rej, prev_h, assignment):
        if i == n:
            if len(assignment) - assignment.count(0) < required:
                return
            sol = _evaluate(assignment, freq, tol, flo, fhi)
            if sol is not None:
                cur = best["sol"]
                if cur is None:
                    best["sol"] = sol
                elif sol.adopted_count != cur.adopted_count:
                    if sol.adopted_count > cur.adopted_count:
                        best["sol"] = sol
                elif _better(sol, cur):
                    best["sol"] = sol
            return
        for h in range(prev_h + 1, H + 1):
            assignment.append(h)
            rec(i + 1, used_rej, h, assignment)
            assignment.pop()
        if used_rej < R:
            assignment.append(0)
            rec(i + 1, used_rej + 1, prev_h, assignment)
            assignment.pop()

    rec(0, 0, 0, [])
    return best["sol"]


class FixedCases(unittest.TestCase):

    def test_recoverable_sample_joint_bell_plus_noise(self):
        freq = [300.0, 400.5, 499.2, 600.3, 668.0, 800.8, 901.5, 1002.0]
        tol = [1.5] * 8
        tol[4] = 0.8
        r = solve(freq, tol, 95, 105, 12, 2)
        sol = assert_well_formed(self, r, True)
        self.assertEqual(sol.adopted_count, 7)
        self.assertEqual(sol.assignment[:5], [3, 4, 5, 6, 0])
        self.assertEqual(sol.assignment[5:], [8, 9, 10])
        self.assertLess(abs(sol.f0 - 100.2), 0.2)
        self.assertIsNotNone(r.evidence)
        self.assertEqual(r.evidence.index, 4)
        self.assertEqual(r.evidence.optimal_count, 7)
        self.assertLess(r.evidence.max_count_with_peak, 7)

    def test_unsolvable_fragment_retains_peaks_and_reports_earliest(self):
        # 300/400/500/600 共同支持 f0≈100；668 的基频带与联合交集互不相容；
        # 846 在候选区间内无任何泛音归属。R=1 时要求至少采用 5 条，
        # 但最大联合子集只有 4 条 → 无解；页面保留全部原始峰值。
        freq = [300.0, 400.0, 500.0, 600.0, 668.0, 846.0]
        tol = [1.5] * 6
        r = solve(freq, tol, 95, 105, 12, 1)
        sol = assert_well_formed(self, r, False)
        self.assertEqual(sol.adopted_count, 0)
        self.assertTrue(math.isnan(sol.f0))
        self.assertEqual([p.frequency for p in sol.peaks], freq)
        # 证据基于不受剔除预算限制的最大联合子集：K*=4，最早冲突峰是第 5 条
        self.assertIsNotNone(r.evidence)
        self.assertEqual(r.evidence.index, 4)
        self.assertEqual(r.evidence.optimal_count, 4)
        self.assertEqual(r.evidence.required_count, 5)
        self.assertLess(r.evidence.max_count_with_peak, 4)
        self.assertEqual(r.evidence.excluded_indices[:2], [4, 5])
        self.assertTrue(any(b["harmonic"] == 7 for b in r.evidence.valid_bands))

    def test_completely_infeasible_all_peaks_outside_range(self):
        freq = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        tol = [0.1] * 6
        r = solve(freq, tol, 400, 500, 12, 3)
        assert_well_formed(self, r, False)
        self.assertEqual(r.solution.adopted_count, 0)
        self.assertIsNotNone(r.evidence)
        self.assertEqual(r.evidence.index, 0)
        self.assertEqual(r.evidence.valid_bands, [])
        self.assertTrue(math.isnan(r.solution.f0))
        d = r.to_dict()
        self.assertIs(d["feasible"], False)
        self.assertIsNone(d["solution"]["f0"])
        self.assertEqual([p["frequency"] for p in d["solution"]["peaks"]], freq)

    def test_greedy_nearest_harmonic_trap(self):
        # 1160 逐峰就近会选 h=12（≈96.7）而堵死 1300；联合搜索改派 h=11，
        # 与其余峰共享 f0≈104，6 条全采用。
        freq = [100.0, 200.0, 300.0, 415.0, 1160.0, 1300.0]
        tol = [5.0, 10.0, 15.0, 15.0, 60.0, 60.0]
        r = solve(freq, tol, 95, 105, 12, 0)
        sol = assert_well_formed(self, r, True)
        self.assertEqual(sol.adopted_count, 6)
        self.assertEqual(sol.assignment, [1, 2, 3, 4, 11, 12])
        self.assertGreaterEqual(sol.f0, 103.3 - 1e-9)
        self.assertLessEqual(sol.f0, 105.0 + 1e-9)

    def test_tie_break_lexicographic_order(self):
        a = _evaluate([1, 2], [100.0, 200.0], [0.5, 0.5], 99.0, 101.0)
        b = _evaluate([1, 2], [100.0, 200.0], [0.5, 0.5], 99.0, 101.0)
        self.assertFalse(_better(b, a) or _better(a, b))
        s1 = Solution(100.0, (99.5, 100.5), [1, 2], [], 0.0, 0.0)
        s2 = Solution(100.0, (99.5, 100.5), [1, 3], [], 0.0, 0.0)
        self.assertTrue(_better(s1, s2))
        s3 = Solution(100.0, (99.5, 100.5), [1, 0], [], 0.0, 0.0)
        self.assertTrue(_better(s1, s3))

    def test_validation_rules(self):
        good = ([100, 200, 300, 400, 500, 600], [1] * 6, 90, 110, 8, 1)
        validate_input(*good)
        bad_cases = [
            ([1, 2, 3, 4, 5], [1] * 5, 1, 2, 3, 0),          # 少于 6 条
            ([1, 2, 3, 4, 5, 6], [1] * 6, 10, 5, 3, 0),       # 区间倒置
            ([1, 2, 3, 3, 5, 6], [1] * 6, 1, 9, 3, 0),        # 非严格递增
            ([1, 2, 3, 4, 5, 6], [1] * 6, 1, 9, 0, 0),        # H 越界
            ([1, 2, 3, 4, 5, 6], [1] * 6, 1, 9, 3, -1),       # R 越界
            (["a", 2, 3, 4, 5, 6], [1] * 6, 1, 9, 3, 0),      # 非数字
        ]
        for case in bad_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValidationError):
                    validate_input(*case)

    def test_all_rejected_budget_still_maximizes(self):
        freq = [100.0, 200.0, 201.0, 300.0, 400.0, 500.0]
        tol = [0.5] * 6
        r = solve(freq, tol, 99, 101, 6, 5)
        sol = assert_well_formed(self, r, True)
        self.assertEqual(sol.adopted_count, 5)
        self.assertEqual(sol.assignment, [1, 2, 0, 3, 4, 5])

    def test_large_wide_tolerance_is_fast_and_joint(self):
        # 18 峰、H=40、10% 宽容差：每条峰多个可行序号，是枚举/兜底压力用例。
        f0_true = 107.0
        hs = list(range(1, 19))
        # 小抖动 + 保证严格递增，10% 容差仍给每条峰多个可行序号
        freq = []
        for i, h in enumerate(hs):
            f = h * f0_true + (3 if i % 2 else -2)
            if i and f <= freq[-1]:
                f = freq[-1] + 1.0
            freq.append(f)
        tol = [0.10 * f for f in freq]
        import time
        t0 = time.time()
        r = solve(freq, tol, 50, 200, 40, 9)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 8.0)
        sol = assert_well_formed(self, r, True)
        self.assertEqual(sol.adopted_count, 18)
        # 序号严格递增 1..18，统一基频同时满足所有采用峰
        self.assertEqual(sol.assignment, hs)

    def test_dp_fallback_returns_valid_joint_solution(self):
        from app.solver import _dp_fallback
        freq = [h * 100.0 for h in range(1, 10)]
        tight = [0.07 * f for f in freq]
        assigns = _dp_fallback(freq, tight, 90, 110, 12, 4, 9)
        self.assertTrue(assigns)
        for a in assigns:
            hs = [h for h in a if h]
            self.assertEqual(hs, sorted(set(hs)))
            sol = _evaluate(list(a), freq, tight, 90, 110)
            self.assertIsNotNone(sol)


class RandomBruteForce(unittest.TestCase):

    def test_matches_brute_force(self):
        for seed in range(50):
            with self.subTest(seed=seed):
                rng = random.Random(seed)
                n = rng.randint(6, 8)
                H = rng.randint(n, n + 2)
                f0_true = rng.uniform(90, 110)
                hs = sorted(rng.sample(range(1, H + 1), n))
                freq = sorted(h * f0_true + rng.uniform(-3, 3) for h in hs)
                for i in range(1, n):
                    if freq[i] <= freq[i - 1]:
                        freq[i] = freq[i - 1] + rng.uniform(0.5, 2.0)
                tol = [rng.uniform(0.3, 4.0) for _ in range(n)]
                R = rng.randint(0, 2)

                r = solve(freq, tol, 80, 120, H, R)
                bf = brute_force(freq, tol, 80, 120, H, R)
                assert_well_formed(self, r)
                if bf is None:
                    self.assertFalse(r.feasible)
                    self.assertTrue(math.isnan(r.solution.f0))
                else:
                    self.assertTrue(r.feasible)
                    self.assertEqual(r.solution.adopted_count, bf.adopted_count)
                    self.assertEqual(r.solution.assignment, bf.assignment)
                    self.assertLess(abs(r.solution.f0 - bf.f0), 1e-7)
                    self.assertLess(abs(r.solution.max_rel_error - bf.max_rel_error), 1e-9)
                    self.assertLess(abs(r.solution.sse - bf.sse), 1e-7)
                    if r.evidence:
                        self.assertEqual(r.evidence.index, r.evidence.excluded_indices[0])
                        self.assertEqual(r.evidence.optimal_count, bf.adopted_count)

    def test_evidence_force_counts_consistent(self):
        rng = random.Random(2024)
        for _ in range(20):
            n = rng.randint(6, 9)
            H = rng.randint(n, 12)
            f0_true = rng.uniform(95, 105)
            hs = sorted(rng.sample(range(1, H + 1), n))
            freq = sorted(h * f0_true + rng.uniform(-6, 6) for h in hs)
            for i in range(1, n):
                if freq[i] <= freq[i - 1]:
                    freq[i] = freq[i - 1] + 0.5
            tol = [rng.uniform(0.2, 2.0) for _ in range(n)]
            r = solve(freq, tol, 90, 110, H, rng.randint(0, 2))
            assert_well_formed(self, r)
            if r.evidence:
                ev = r.evidence
                self.assertLess(ev.max_count_with_peak, ev.optimal_count)
                self.assertEqual(ev.index, ev.excluded_indices[0])
                for j in range(ev.index):
                    self.assertNotIn(j, ev.excluded_indices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
