"""编钟残音的联合基频识别。

本模块**不**做"逐峰就近除以最近泛音序号再拼接"的事。对每条采用峰必须
给出一个泛音序号 h（正整数，随录入顺序严格递增且不重复），并要求存在
**同一个**基频 f0 使 |f_i - h_i f0| <= tol_i 同时成立：

    f0 属于 [(f_i - tol_i)/h_i, (f_i + tol_i)/h_i] 与候选区间的交集。

算法（联合，非逐峰）：

1. 端点扫描计数：归属关系只在端点 (f_i±tol_i)/h 处变化。在每个端点上
   用"严格递增槽位贪心" O(n) 求该 f0 下最多能联合采用多少峰（可跳过
   任意峰、每峰可行 h 是连续整数段，贪心选最小可行槽为经典最优），并
   据此判定剔除预算下的最大联合采用数 K 与无约束 K*。闭区间端点处的
   可行归属集是相邻开区间内归属集的超集，故只扫端点不会漏最优。

2. 二分最小最大相对误差 e*：判定"容差收紧为 min(tol_i, e·f_i) 后，
   预算内是否仍能联合采用 K 条"。

3. 在 e* 收紧容差下，于全部归属端点精确枚举 K 峰联合分配（真实紧容差
   数据中每条峰只有唯一可行序号，枚举量为 1），逐份精确求最优统一
   基频，按  采用数 → 最大相对误差 → 误差平方和 → 录入顺序最早序号
   四级标准选优。枚举超预算（仅极端宽容差对抗输入）时，改用固定点
   加权最小二乘动态规划扫描端点并沿 WLS 顶点下降取候选，保证有界。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

EPS_FEAS = 1e-10  # 闭区间归属的浮点容差（相对）
EPS_REL = 1e-10   # 最大相对误差比较容差
EPS_SSE = 1e-10   # 误差平方和比较容差

MAX_PEAKS = 18
MIN_PEAKS = 6
MAX_HARMONIC = 40

BISECT_STEPS = 55
ENUM_CAP = 200_000       # 精确枚举的总分配数上限；超过即启用 DP 兜底
DESCENT_ROUNDS = 8       # DP 兜底时 WLS 顶点下降轮数


class ValidationError(ValueError):
    """输入数据不合法。"""


@dataclass
class PeakResult:
    index: int
    frequency: float
    tolerance: float
    adopted: bool
    harmonic: Optional[int] = None
    predicted: Optional[float] = None
    residual: Optional[float] = None      # f_i - h_i * f0（带符号）
    abs_error: Optional[float] = None
    rel_error: Optional[float] = None     # abs_error / f_i

    def to_dict(self) -> dict:
        within = None
        if self.adopted:
            within = self.abs_error <= self.tolerance + EPS_FEAS * max(1.0, abs(self.frequency))
        return {
            "index": self.index,
            "frequency": self.frequency,
            "tolerance": self.tolerance,
            "adopted": self.adopted,
            "harmonic": self.harmonic,
            "predicted": self.predicted,
            "residual": self.residual,
            "abs_error": self.abs_error,
            "rel_error": self.rel_error,
            "within_tolerance": within,
        }


@dataclass
class Evidence:
    index: int                   # 最早无法纳入任何最优联合解释的峰（0 起）
    frequency: float
    tolerance: float
    optimal_count: int           # 最大联合采用数（无解时为无剔除约束的 K*）
    required_count: int          # 剔除预算要求至少采用的峰数 n - R
    max_count_with_peak: int     # 强制容纳该峰时最多能联合采用多少峰
    excluded_indices: list[int]
    valid_bands: list[dict]      # 该峰在各 h 下与候选区间相交的基频带

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "frequency": self.frequency,
            "tolerance": self.tolerance,
            "optimal_count": self.optimal_count,
            "required_count": self.required_count,
            "max_count_with_peak": self.max_count_with_peak,
            "excluded_indices": self.excluded_indices,
            "valid_bands": self.valid_bands,
        }


@dataclass
class Solution:
    f0: float
    f0_interval: tuple[float, float]
    assignment: list[int]        # 每条峰：h（采用）或 0（剔除）
    peaks: list[PeakResult]
    max_rel_error: float
    sse: float

    @property
    def adopted_count(self) -> int:
        return sum(1 for h in self.assignment if h)

    @property
    def rejected_indices(self) -> list[int]:
        return [i for i, h in enumerate(self.assignment) if not h]

    def order_key(self) -> tuple:
        # 录入顺序逐峰比较：采用条目 (0, h) 早于剔除条目 (1,)。
        return tuple((0, h) if h else (1,) for h in self.assignment)


@dataclass
class SolveResult:
    frequencies: list[float]
    tolerances: list[float]
    f0_min: float
    f0_max: float
    max_harmonic: int
    max_rejected: int
    solution: Solution
    evidence: Optional[Evidence]
    feasible: bool = True

    @property
    def fully_explained(self) -> bool:
        return self.feasible and self.solution.adopted_count == len(self.frequencies)

    def to_dict(self) -> dict:
        s = self.solution

        def num(v):
            return v if isinstance(v, (int, float)) and math.isfinite(v) else None

        return {
            "status": "ok",
            "feasible": self.feasible,
            "input": {
                "frequencies": self.frequencies,
                "tolerances": self.tolerances,
                "f0_min": self.f0_min,
                "f0_max": self.f0_max,
                "max_harmonic": self.max_harmonic,
                "max_rejected": self.max_rejected,
            },
            "solution": {
                "f0": num(s.f0),
                "f0_interval": list(s.f0_interval) if self.feasible else None,
                "adopted_count": s.adopted_count,
                "rejected_count": len(self.frequencies) - s.adopted_count,
                "max_rel_error": num(s.max_rel_error),
                "sse": num(s.sse),
                "peaks": [p.to_dict() for p in s.peaks],
                "adopted_indices": [i for i, h in enumerate(s.assignment) if h],
                "rejected_indices": s.rejected_indices,
                "harmonics": [h if h else None for h in s.assignment],
            },
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "fully_explained": self.fully_explained,
        }


def validate_input(frequencies, tolerances, f0_min, f0_max, max_harmonic, max_rejected):
    """校验并归一化页面录入，返回 (freq, tol, flo, fhi, H, R)。"""

    def as_float(value, name):
        try:
            v = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} 必须是十进制数")
        if not math.isfinite(v):
            raise ValidationError(f"{name} 必须是有限数")
        return v

    def as_int(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise ValidationError(f"{name} 必须是整数")
        try:
            text = value.strip() if isinstance(value, str) else value
            iv = int(text)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} 必须是整数")
        if float(iv) != float(value):
            raise ValidationError(f"{name} 必须是整数")
        return iv

    if not isinstance(frequencies, (list, tuple)) or not isinstance(tolerances, (list, tuple)):
        raise ValidationError("frequencies 与 tolerances 必须是数组")
    n = len(frequencies)
    if not MIN_PEAKS <= n <= MAX_PEAKS:
        raise ValidationError(f"峰值条数必须在 {MIN_PEAKS} 至 {MAX_PEAKS} 之间，当前 {n} 条")
    if len(tolerances) != n:
        raise ValidationError("允许偏差条数必须与峰值条数相同")

    freq = [as_float(v, f"第 {i + 1} 条峰值频率") for i, v in enumerate(frequencies)]
    tol = [as_float(v, f"第 {i + 1} 条允许偏差") for i, v in enumerate(tolerances)]

    for i, f in enumerate(freq):
        if f <= 0:
            raise ValidationError(f"第 {i + 1} 条峰值频率必须为正数")
        if i and freq[i - 1] >= f:
            raise ValidationError(
                f"峰值必须按频率严格递增：第 {i} 条 {freq[i - 1]} >= 第 {i + 1} 条 {f}"
            )
    for i, t in enumerate(tol):
        if t < 0:
            raise ValidationError(f"第 {i + 1} 条允许偏差不能为负数")

    flo = as_float(f0_min, "候选基频下界")
    fhi = as_float(f0_max, "候选基频上界")
    if flo <= 0:
        raise ValidationError("候选基频下界必须为正数")
    if fhi < flo:
        raise ValidationError("候选基频上界不得小于下界")

    H = as_int(max_harmonic, "最高泛音序号")
    if not 1 <= H <= MAX_HARMONIC:
        raise ValidationError(f"最高泛音序号必须在 1 至 {MAX_HARMONIC} 之间")
    R = as_int(max_rejected, "最多可剔除杂峰数")
    if not 0 <= R <= n:
        raise ValidationError("最多可剔除杂峰数必须在 0 至峰值条数之间")

    return freq, tol, flo, fhi, H, R


# ---------------------------------------------------------------- 单份方案评估

def _feasible_scale(lo: float, hi: float) -> float:
    return max(1.0, abs(lo), abs(hi))


def _evaluate(assignment, freq, tol, flo, fhi):
    """对一份完整联合方案求最优统一基频并构造 Solution（不可行返回 None）。"""
    adopted = [i for i, h in enumerate(assignment) if h]
    if not adopted:
        return None

    lo, hi = flo, fhi
    for i in adopted:
        h = assignment[i]
        lo = max(lo, (freq[i] - tol[i]) / h)
        hi = min(hi, (freq[i] + tol[i]) / h)
    if lo > hi + EPS_FEAS * _feasible_scale(lo, hi):
        return None

    # 最大相对误差可解析化：令 c_i = h_i/f_i，则
    #   M(f) = max_i |c_i f - 1|
    #        = max( c_max f - 1, 1 - c_min f )
    # （正臂在 c 最大处取到，负臂在 c 最小处取到。）两条仿射线的上包络
    # 是凸 V 形，无约束交点 f* = 2/(c_max+c_min)，与交集 [lo,hi] 投影即可。
    c_min = min(assignment[i] / freq[i] for i in adopted)
    c_max = max(assignment[i] / freq[i] for i in adopted)

    def max_rel(f0):
        return max(c_max * f0 - 1.0, 1.0 - c_min * f0)

    f_star = 2.0 / (c_max + c_min)
    m = max_rel(min(max(f_star, lo), hi))

    # 保持 M=m 的基频闭区间 [A, B]：
    # 1 - c_min f <= m → f >= (1-m)/c_min；
    # c_max f - 1 <= m → f <= (1+m)/c_max。
    A = max(lo, (1.0 - m) / c_min)
    B = min(hi, (1.0 + m) / c_max)
    if A > B:  # 浮点保护
        A = B = min(max((A + B) / 2.0, lo), hi)

    # 在最大相对误差最优的前提下，SSE 最小点为加权最小二乘解
    # f_wls = Σ h_i f_i / Σ h_i² 在 [A, B] 上的投影。
    num = sum(assignment[i] * freq[i] for i in adopted)
    den = sum(assignment[i] ** 2 for i in adopted)
    f0 = min(max(num / den, A), B)

    peaks = []
    sse = 0.0
    worst = 0.0
    for i, f in enumerate(freq):
        h = assignment[i]
        if h:
            predicted = h * f0
            residual = f - predicted
            abs_err = abs(residual)
            rel_err = abs_err / f
            sse += residual * residual
            worst = max(worst, rel_err)
            peaks.append(PeakResult(i, f, tol[i], True, h, predicted, residual, abs_err, rel_err))
        else:
            peaks.append(PeakResult(i, f, tol[i], False))

    return Solution(f0=f0, f0_interval=(lo, hi), assignment=list(assignment),
                    peaks=peaks, max_rel_error=worst, sse=sse)


def _better(candidate: Solution, best: Optional[Solution]) -> bool:
    if best is None:
        return True
    # 调用方只在采用数相同（都为 K）的方案间比较。
    if candidate.max_rel_error < best.max_rel_error - EPS_REL:
        return True
    if candidate.max_rel_error > best.max_rel_error + EPS_REL:
        return False
    sse_eps = EPS_SSE * max(1.0, candidate.sse, best.sse)
    if candidate.sse < best.sse - sse_eps:
        return True
    if candidate.sse > best.sse + sse_eps:
        return False
    return candidate.order_key() < best.order_key()


# ---------------------------------------------------------------- 端点扫描计数

def _h_segments(freq, tol, f0, H):
    """固定 f0 下每条峰的可行泛音序号整数段 [(a, b), ...]，不可行为 None。"""
    segs = []
    eps = EPS_FEAS * max(1.0, f0)
    for f, t in zip(freq, tol):
        a = math.ceil((f - t) / f0 - eps)
        b = math.floor((f + t) / f0 + eps)
        a = max(a, 1)
        b = min(b, H)
        segs.append((a, b) if a <= b else None)
    return segs


def _greedy_assign(segments, force=-1):
    """严格递增槽位贪心：按录入顺序为每条峰选最小可行 h，可跳过峰。

    返回 (采用数, 分配列表)；force=j 时该峰必须采用，否则返回 None。
    连续整数段上"取最小可行槽"为最长严格递增匹配的经典最优策略。
    """
    assignment = [0] * len(segments)
    prev_h = 0
    count = 0
    for i, seg in enumerate(segments):
        if seg is None:
            if i == force:
                return None
            continue
        a, b = seg
        h = max(a, prev_h + 1)
        if h <= b:
            assignment[i] = h
            prev_h = h
            count += 1
        elif i == force:
            return None
    return count, assignment


def _scan_points(freq, tol, flo, fhi, H):
    """生成 f0 轴上的全部归属端点（含候选区间端点）。

    归属关系只在端点 (f_i±tol_i)/h 处变化；闭区间端点处的可行归属集
    是相邻开区间内归属集的超集，故最大联合采用数在端点处即可取得，
    无需另取区间中点。
    """
    points = {flo, fhi}
    for f, t in zip(freq, tol):
        for h in range(1, H + 1):
            low = (f - t) / h
            high = (f + t) / h
            if high >= flo and low <= fhi:
                if flo <= low <= fhi:
                    points.add(low)
                if flo <= high <= fhi:
                    points.add(high)
    return sorted(points)


def _max_adoption(freq, tol, flo, fhi, H, reject_budget, force=-1):
    """端点扫描求（强制采用 force 峰时）预算内与无约束的最大联合采用数。

    返回 (budget_count, free_count)；force 峰无处可采用时两者均为 -1。
    """
    budget_count = -1
    free_count = -1
    for f0 in _scan_points(freq, tol, flo, fhi, H):
        segs = _h_segments(freq, tol, f0, H)
        got = _greedy_assign(segs, force)
        if got is None:
            continue
        count, _ = got
        if count > free_count:
            free_count = count
        if len(freq) - count <= reject_budget and count > budget_count:
            budget_count = count
    return budget_count, free_count


# ------------------------------------------------ 收紧容差下的候选分配枚举

def _enumerate_at_point(segs, n, H, R, K, found, counter):
    """在一个固定 f0 的归属段上精确枚举全部 K 峰、剔除 ≤ R 的递增分配。"""

    def dfs(i, prev_h, used_rej, acc):
        adopted = len(acc) - used_rej
        remaining = n - i
        if adopted + remaining < K or used_rej > R:
            return
        if counter[0] >= ENUM_CAP:
            return
        if i == n:
            if adopted == K:
                found.add(tuple(acc))
                counter[0] += 1
            return
        seg = segs[i]
        if seg is not None:
            a, b = seg
            for h in range(max(a, prev_h + 1), b + 1):
                dfs(i + 1, h, used_rej, acc + [h])
                if counter[0] >= ENUM_CAP:
                    return
        if used_rej < R:
            dfs(i + 1, prev_h, used_rej + 1, acc + [0])

    dfs(0, 0, 0, [])


def _fixed_point_dp(freq, tight, f0, H, R, K):
    """DP 兜底：固定 f0 下最小化 Σ(f_i - h_i f0)² 的 K 峰递增分配。

    已剔除数 = i - k 由处理位置 i 与采用数 k 决定，状态 dp[k][h]；
    采用转移对 ph < h 取前缀最小值，每峰 O(K·H)。SSE 并列时按题目
    字典序（采用且序号小者更早；剔除映射为 H+1 排在最后）。
    """
    n = len(freq)
    REJ = H + 1
    neg = None
    dp = [[neg] * (H + 1) for _ in range(K + 1)]
    dp[0][0] = (0.0, (), ())
    segs = _h_segments(freq, tight, f0, H)

    for i, seg in enumerate(segs):
        nxt = [[neg] * (H + 1) for _ in range(K + 1)]

        def relax(k, h, cost, order, assign):
            old = nxt[k][h]
            if old is None or cost < old[0] - EPS_SSE * max(1.0, cost):
                nxt[k][h] = (cost, order, assign)
            elif abs(cost - old[0]) <= EPS_SSE * max(1.0, cost) and order < old[1]:
                nxt[k][h] = (cost, order, assign)

        for k in range(0, min(K, i) + 1):
            prefix = [neg] * (H + 1)
            run = neg
            for h in range(H + 1):
                cell = dp[k][h]
                if cell is not None and (run is None or (cell[0], cell[1]) < (run[0], run[1])):
                    run = cell
                prefix[h] = run

            # 采用：处理完 i 后剔除数 = i - k。
            if seg is not None and k < K and i - k <= R:
                a, b = seg
                f = freq[i]
                for h in range(a, b + 1):
                    pre = prefix[h - 1]
                    if pre is None:
                        continue
                    resid = f - h * f0
                    relax(k + 1, h, pre[0] + resid * resid,
                          pre[1] + (h,), pre[2] + (h,))

            # 剔除：处理完 i 后剔除数 = i + 1 - k。
            if i + 1 - k <= R:
                for ph in range(H + 1):
                    cell = dp[k][ph]
                    if cell is not None:
                        relax(k, ph, cell[0], cell[1] + (REJ,), cell[2] + (0,))
        dp = nxt

    winner = None
    for h in range(H + 1):
        cell = dp[K][h]
        if cell is not None and (winner is None or (cell[0], cell[1]) < (winner[0], winner[1])):
            winner = cell
    return list(winner[2]) if winner else None


def _wls_vertex(assignment, freq):
    """一份分配的加权最小二乘基频 Σhf/Σh²。"""
    num = sum(h * freq[i] for i, h in enumerate(assignment) if h)
    den = sum(h * h for h in assignment if h)
    return num / den if den else None


def _dp_fallback(freq, tight, flo, fhi, H, R, K):
    """极端宽容差下的有界兜底：端点 + 原子区间种子点跑固定点 DP，
    并沿每份分配的 WLS 顶点反复下降，收集候选分配。"""
    ordered = _scan_points(freq, tight, flo, fhi, H)
    seeds = list(ordered)
    for a in range(len(ordered) - 1):
        x1, x2 = ordered[a], ordered[a + 1]
        if x2 - x1 > EPS_FEAS * max(1.0, abs(x1), abs(x2)):
            seeds.append(x1 + (x2 - x1) / 3.0)
            seeds.append(x1 + 2.0 * (x2 - x1) / 3.0)

    found: set[tuple] = set()
    pending = [min(max(p, flo), fhi) for p in seeds]
    seen: set = set()
    for _ in range(DESCENT_ROUNDS):
        nxt_pending = []
        for p in pending:
            key = round(p, 12)
            if key in seen:
                continue
            seen.add(key)
            assign = _fixed_point_dp(freq, tight, p, H, R, K)
            if assign is None:
                continue
            t = tuple(assign)
            if t not in found:
                found.add(t)
                v = _wls_vertex(assign, freq)
                if v is not None and flo - 1e-9 <= v <= fhi + 1e-9:
                    nxt_pending.append(min(max(v, flo), fhi))
        pending = nxt_pending
        if not pending:
            break
    return found


# ---------------------------------------------------------------- 主求解

def solve(frequencies, tolerances, f0_min, f0_max, max_harmonic, max_rejected) -> SolveResult:
    freq, tol, flo, fhi, H, R = validate_input(
        frequencies, tolerances, f0_min, f0_max, max_harmonic, max_rejected
    )
    n = len(freq)
    required = max(1, n - R)

    # 阶段一：端点扫描求预算内最大联合采用数 K 与无约束 K*。
    K, K_free = _max_adoption(freq, tol, flo, fhi, H, R)
    K = max(K, 0)
    K_free = max(K_free, 0)
    feasible = K >= required

    best = None
    if feasible:
        # 阶段二 a：二分最小最大相对误差 e*。
        def keeps_k(e):
            tight = [min(t, e * f) for f, t in zip(freq, tol)]
            bc, _ = _max_adoption(freq, tight, flo, fhi, H, R)
            return max(bc, 0) >= K

        e_hi = max(t / f for f, t in zip(freq, tol))
        if not keeps_k(e_hi):
            e_hi = 1.0  # 数值保护
        e_lo = 0.0
        for _ in range(BISECT_STEPS):
            mid = (e_lo + e_hi) / 2.0
            if keeps_k(mid):
                e_hi = mid
            else:
                e_lo = mid
        e_star = e_hi * (1.0 + 1e-9) + 1e-12

        # 阶段二 b：收紧容差下枚举 K 峰联合分配。
        tight = [min(t, e_star * f) + EPS_FEAS * f for f, t in zip(freq, tol)]
        candidates: set[tuple] = set()
        counter = [0]
        for f0 in _scan_points(freq, tight, flo, fhi, H):
            segs = _h_segments(freq, tight, f0, H)
            count, _ = _greedy_assign(segs)
            if count < K or n - count > R:
                continue
            _enumerate_at_point(segs, n, H, R, K, candidates, counter)
            if counter[0] >= ENUM_CAP:
                break

        # 极端宽容差导致精确枚举爆炸时，用有界 DP 兜底补足候选。
        if counter[0] >= ENUM_CAP or not candidates:
            candidates |= _dp_fallback(freq, tight, flo, fhi, H, R, K)

        # 阶段二 c：逐份精确评估（真实容差、最优统一基频）并四级选优。
        for assignment in candidates:
            sol = _evaluate(list(assignment), freq, tol, flo, fhi)
            if sol is not None and sol.adopted_count == K and _better(sol, best):
                best = sol

    if best is None:
        # 预算内无联合解：不呈现超出剔除预算的部分方案，全部原始峰值保留。
        blank = [PeakResult(i, freq[i], tol[i], False) for i in range(n)]
        best = Solution(f0=float("nan"), f0_interval=(flo, fhi),
                        assignment=[0] * n, peaks=blank,
                        max_rel_error=float("nan"), sse=float("nan"))

    # 证据：可行时按预算口径，无解时按无剔除约束口径 K*——这样无解药
    # 也能指向真正的冲突杂峰，而不是最早的正常峰。
    budget_for_evidence = R if feasible else n
    base_count = K if feasible else K_free
    excluded = []
    force_counts: dict[int, int] = {}
    if base_count == 0:
        for j in range(n):
            force_counts[j] = 0
            excluded.append(j)
    else:
        for j in range(n):
            bc, fc = _max_adoption(freq, tol, flo, fhi, H, budget_for_evidence, force=j)
            cnt = max(bc if feasible else fc, 0)
            force_counts[j] = cnt
            if cnt < base_count:
                excluded.append(j)

    evidence = None
    if excluded:
        earliest = excluded[0]
        evidence = Evidence(
            index=earliest,
            frequency=freq[earliest],
            tolerance=tol[earliest],
            optimal_count=base_count,
            required_count=required,
            max_count_with_peak=force_counts[earliest],
            excluded_indices=excluded,
            valid_bands=_valid_bands(freq[earliest], tol[earliest], flo, fhi, H),
        )

    return SolveResult(freq, tol, flo, fhi, H, R, best, evidence, feasible)


def _valid_bands(f, t, flo, fhi, H):
    """该峰在每个 h 下与候选基频闭区间相交的基频带。"""
    bands = []
    for h in range(1, H + 1):
        lo = max(flo, (f - t) / h)
        hi = min(fhi, (f + t) / h)
        if lo <= hi + EPS_FEAS * max(1.0, abs(lo), abs(hi)):
            bands.append({"harmonic": h, "f0_low": lo, "f0_high": hi})
    return bands
