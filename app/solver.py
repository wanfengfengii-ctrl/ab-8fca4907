"""Joint fundamental-frequency assignment for chime-bell partial peaks.

Given measured peaks (frequencies with per-peak tolerances), a candidate
fundamental interval [f0_min, f0_max], a highest allowed harmonic number and a
maximum number of peaks that may be discarded as environmental noise, find a
strictly increasing, non-repeating harmonic assignment such that a *single*
fundamental f0 lies inside every adopted peak's tolerance simultaneously.

Selection order (lexicographic):
  1. maximize the number of adopted peaks
  2. minimize the maximum relative error over adopted peaks
  3. minimize the sum of squared relative errors
  4. earliest scheme in entry order: lexicographically smallest harmonic
     tuple of adopted peaks, then lexicographically smallest adopted-index
     tuple

This is a joint optimization: harmonic numbers are never assigned peak by
peak with a nearest-neighbor rule and then stitched together afterwards.  A
branch-and-bound DFS jointly chooses the rejected peaks and the harmonic
tuple, carrying the intersection of every adopted peak's feasible f0 windows.
For each complete feasible tuple the optimal f0 is picked from a finite set
of breakpoints of the piecewise-quadratic error functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


def _eps_at(x: float) -> float:
    return 1e-12 * max(1.0, abs(x))


def _nonempty(lo: float, hi: float) -> bool:
    return lo <= hi + _eps_at(max(abs(lo), abs(hi)))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


@dataclass(frozen=True)
class SolveRequest:
    frequencies: tuple[float, ...]
    tolerances: tuple[float, ...]
    f0_min: float
    f0_max: float
    max_harmonic: int
    max_rejected: int


@dataclass
class PeakResult:
    index: int  # 0-based entry order
    frequency: float
    tolerance: float
    adopted: bool
    harmonic: Optional[int]
    expected: Optional[float]  # n * f0
    abs_error: Optional[float]  # |f - n*f0|
    rel_error: Optional[float]  # |f - n*f0| / f


@dataclass
class SolveResult:
    feasible: bool
    f0: Optional[float]
    adopted: list[PeakResult] = field(default_factory=list)
    rejected: list[PeakResult] = field(default_factory=list)
    max_relative_error: Optional[float] = None
    sum_squared_relative_error: Optional[float] = None
    # For infeasible instances: first peak (in entry order) that cannot belong
    # to any joint explanation respecting the rejection budget.
    first_unexplainable_index: Optional[int] = None
    reason: Optional[str] = None


def _harmonic_window(req: SolveRequest, i: int, n: int) -> tuple[float, float]:
    """f0 interval for which n*f0 lies inside peak i's tolerance band."""
    f = req.frequencies[i]
    t = req.tolerances[i]
    return ((f - t) / n, (f + t) / n)


def _build_windows(
    req: SolveRequest,
) -> list[list[tuple[int, float, float]]]:
    """Per peak: feasible (harmonic, f0_low, f0_high) triples in harmonic
    order, clipped to the candidate fundamental interval."""
    windows: list[list[tuple[int, float, float]]] = [
        [] for _ in range(len(req.frequencies))
    ]
    for i in range(len(req.frequencies)):
        for n in range(1, req.max_harmonic + 1):
            lo, hi = _harmonic_window(req, i, n)
            lo = max(lo, req.f0_min)
            hi = min(hi, req.f0_max)
            if _nonempty(lo, hi):
                windows[i].append((n, lo, hi))
    return windows


def solve(req: SolveRequest) -> SolveResult:
    n_peaks = len(req.frequencies)
    windows = _build_windows(req)
    best = _Best(n_peaks)
    min_adopt = n_peaks - req.max_rejected

    # Endpoint identities for memoization: every intersection bound is either
    # a global candidate interval end or one of the harmonic-window endpoints.
    lo_ids: dict[float, int] = {req.f0_min: 0}
    hi_ids: dict[float, int] = {req.f0_max: 0}
    for row in windows:
        for _, lo, hi in row:
            lo_ids.setdefault(lo, len(lo_ids))
            hi_ids.setdefault(hi, len(hi_ids))
    seen: set[tuple[int, int, int, int, int]] = set()

    def dfs(
        i: int,
        last_h: int,
        lo: float,
        hi: float,
        adopted: list[tuple[int, int]],
        adopted_count: int,
        rejected_count: int,
    ) -> None:
        if rejected_count > req.max_rejected:
            return
        target = best.count if best.assignment is not None else min_adopt
        remaining = n_peaks - i
        need = target - adopted_count  # further adoptions required
        # Even adopting every remaining peak cannot reach the current target.
        if need > remaining:
            return
        key = (
            i,
            last_h,
            rejected_count,
            lo_ids[lo],
            hi_ids[hi],
        )
        if key in seen:
            return
        seen.add(key)
        # Forward feasibility: count remaining peaks that still offer some
        # harmonic above last_h compatible with the current f0 intersection.
        capable = 0
        for j in range(i, n_peaks):
            for n, wlo, whi in windows[j]:
                if n > last_h and _nonempty(max(lo, wlo), min(hi, whi)):
                    capable += 1
                    break
        if need > capable:
            return

        if i == n_peaks:
            if adopted_count < min_adopt:
                return
            f0, max_rel, sse = _best_f0(req, adopted, lo, hi)
            best.update(_Candidate(adopted_count, max_rel, sse, list(adopted), f0))
            return

        # Adopt first (finds high-cardinality solutions quickly), then reject.
        # If peak i is adopted, leave need-1 higher harmonics for later peaks.
        max_n = req.max_harmonic - max(0, need - 1)
        for n, wlo, whi in windows[i]:
            if n <= last_h or n > max_n:
                continue
            nlo, nhi = max(lo, wlo), min(hi, whi)
            if _nonempty(nlo, nhi):
                adopted.append((i, n))
                dfs(i + 1, n, nlo, nhi, adopted, adopted_count + 1, rejected_count)
                adopted.pop()

        if rejected_count < req.max_rejected:
            dfs(i + 1, last_h, lo, hi, adopted, adopted_count, rejected_count + 1)

    dfs(0, 0, req.f0_min, req.f0_max, [], 0, 0)

    if best.assignment is None:
        return _infeasible(req, windows)
    return _build_result(req, best.solution)


@dataclass(frozen=True)
class _Candidate:
    count: int
    max_rel: float
    sse: float
    assignment: list[tuple[int, int]]
    f0: float


class _Best:
    """Keeps the lexicographically best candidate under the four rules."""

    __slots__ = ("n_peaks", "count", "max_rel", "sse", "assignment", "f0")

    def __init__(self, n_peaks: int) -> None:
        self.n_peaks = n_peaks
        self.count: int = -1
        self.max_rel: float = float("inf")
        self.sse: float = float("inf")
        self.assignment: Optional[list[tuple[int, int]]] = None
        self.f0: float = 0.0

    @property
    def solution(self) -> Optional[_Candidate]:
        if self.assignment is None:
            return None
        return _Candidate(
            self.count, self.max_rel, self.sse, self.assignment, self.f0
        )

    def update(self, cand: _Candidate) -> None:
        if self.assignment is None:
            better = True
        elif cand.count != self.count:
            better = cand.count > self.count
        elif not _close(cand.max_rel, self.max_rel):
            better = cand.max_rel < self.max_rel
        elif not _close(cand.sse, self.sse):
            better = cand.sse < self.sse
        else:
            # Rule 4 — earliest scheme in entry order: scan peaks in entry
            # order; at the first peak where schemes differ, adopting it with a
            # smaller harmonic beats adopting it with a larger one, which beats
            # rejecting it. Encoded as (flag, n) with flag 0 = adopted.
            better = _decision_key(cand.assignment, self.n_peaks) < _decision_key(
                self.assignment, self.n_peaks
            )
        if better:
            self.count = cand.count
            self.max_rel = cand.max_rel
            self.sse = cand.sse
            self.assignment = list(cand.assignment)
            self.f0 = cand.f0


def _decision_key(assignment: list[tuple[int, int]], n_peaks: int) -> list[tuple[int, int]]:
    chosen = dict(assignment)
    return [
        (0, chosen[i]) if i in chosen else (1, 0) for i in range(n_peaks)
    ]


def _best_f0(
    req: SolveRequest,
    adopted: list[tuple[int, int]],
    lo: float,
    hi: float,
) -> tuple[float, float, float]:
    """Pick f0 in the feasible intersection minimizing (max relative error,
    then SSE).

    For peak i with harmonic n, rel_i(x) = |x - c_i| / c_i where c_i = f_i/n:
    a piecewise-linear V.  The maximum of the V's is convex piecewise linear;
    its minimum on [lo, hi] is attained at an endpoint, a kink c_i, or at the
    crossing of a right arm and a left arm, x = 2 c_i c_j/(c_i + c_j)
    (harmonic mean).  SSE = sum (x-c_i)^2/c_i^2 is a smooth convex quadratic
    with the single stationary point x* = (sum 1/c_i)/(sum 1/c_i^2).  The set
    on which the max error attains its minimum is a closed interval whose ends
    are themselves breakpoints, so evaluating this finite candidate set yields
    the exact lexicographic optimum of (max error, SSE).
    """
    cs = [req.frequencies[i] / n for i, n in adopted]

    points: set[float] = {lo, hi}
    for c in cs:
        if lo - _eps_at(c) <= c <= hi + _eps_at(c):
            points.add(min(max(c, lo), hi))
    for p in range(len(cs)):
        for q in range(p + 1, len(cs)):
            a, b = cs[p], cs[q]
            x = 2.0 * a * b / (a + b)  # crossing of opposing V arms
            if lo - _eps_at(x) <= x <= hi + _eps_at(x):
                points.add(x)

    # Unique SSE stationary point; harmless to evaluate even if outside the
    # optimal max-error interval (its max error will then be larger).
    num = sum(1.0 / c for c in cs)
    den = sum(1.0 / (c * c) for c in cs)
    if den > 0.0:
        points.add(min(max(num / den, lo), hi))

    best_x = lo
    best_key: Optional[tuple[float, float]] = None
    for x in sorted(points):
        x = min(max(x, lo), hi)
        mr = 0.0
        sse = 0.0
        for i, n in adopted:
            f = req.frequencies[i]
            r = abs(f - n * x) / f
            mr = max(mr, r)
            sse += r * r
        key = (mr, sse)
        if best_key is None or key < best_key:
            best_key = key
            best_x = x
    assert best_key is not None
    return best_x, best_key[0], best_key[1]


def _build_result(req: SolveRequest, cand: _Candidate) -> SolveResult:
    assignment = dict(cand.assignment)
    adopted: list[PeakResult] = []
    rejected: list[PeakResult] = []
    for i, f in enumerate(req.frequencies):
        t = req.tolerances[i]
        if i in assignment:
            h = assignment[i]
            expected = h * cand.f0
            abs_err = abs(f - expected)
            rel_err = abs_err / f
            adopted.append(
                PeakResult(i, f, t, True, h, expected, abs_err, rel_err)
            )
        else:
            rejected.append(
                PeakResult(i, f, t, False, None, None, None, None)
            )
    return SolveResult(
        feasible=True,
        f0=cand.f0,
        adopted=adopted,
        rejected=rejected,
        max_relative_error=cand.max_rel,
        sum_squared_relative_error=cand.sse,
    )


def _infeasible(
    req: SolveRequest, windows: list[list[tuple[int, float, float]]]
) -> SolveResult:
    """Report the earliest peak that cannot enter any joint explanation.

    Evidence rule: scan peaks in entry (frequency) order and find the smallest
    k such that no joint assignment exists over the prefix peaks 0..k which
    adopts peak k while rejecting at most ``max_rejected`` of the prefix.
    Earlier prefixes are all jointly explainable, so k is precisely the first
    point where a common fundamental breaks down.  This is still a joint
    check over every harmonic choice of every earlier peak — not a per-peak
    nearest-neighbor guess.
    """
    first: Optional[int] = None
    for k in range(len(req.frequencies)):
        if not _prefix_can_adopt(req, windows, k):
            first = k
            break
    return SolveResult(
        feasible=False,
        f0=None,
        rejected=[
            PeakResult(
                i, req.frequencies[i], req.tolerances[i], False, None, None, None, None
            )
            for i in range(len(req.frequencies))
        ],
        first_unexplainable_index=first,
        reason="no_common_fundamental",
    )


def _prefix_can_adopt(
    req: SolveRequest,
    windows: list[list[tuple[int, float, float]]],
    k: int,
) -> bool:
    """Does some joint assignment over peaks 0..k adopt peak k within the
    global rejection budget (strictly increasing harmonics, one common f0)?"""
    found = False
    seen: set[tuple[int, int, int, float, float, int]] = set()

    def dfs(i: int, last_h: int, lo: float, hi: float, rejects: int, ok: int) -> None:
        nonlocal found
        if found or rejects > req.max_rejected:
            return
        if i > k and not ok:
            return
        key = (i, last_h, rejects, lo, hi, ok)
        if key in seen:
            return
        seen.add(key)
        if i == k + 1:
            if ok:
                found = True
            return

        for n, wlo, whi in windows[i]:
            if n <= last_h:
                continue
            nlo, nhi = max(lo, wlo), min(hi, whi)
            if _nonempty(nlo, nhi):
                dfs(i + 1, n, nlo, nhi, rejects, ok or (1 if i == k else 0))

        if rejects < req.max_rejected:
            dfs(i + 1, last_h, lo, hi, rejects + 1, ok)

    dfs(0, 0, req.f0_min, req.f0_max, 0, 0)
    return found
