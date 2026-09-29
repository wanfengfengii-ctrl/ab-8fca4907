"""Unit tests for the joint solver."""

from __future__ import annotations

import math
import random

import pytest

from app.solver import SolveRequest, solve


def check_consistency(result, req: SolveRequest) -> None:
    """Invariants every feasible result must satisfy."""
    assert result.feasible
    assert result.f0 is not None
    assert req.f0_min - 1e-9 <= result.f0 <= req.f0_max + 1e-9
    harmonics = [p.harmonic for p in result.adopted]
    assert harmonics == sorted(set(harmonics))
    assert all(1 <= h <= req.max_harmonic for h in harmonics)
    assert len(result.rejected) <= req.max_rejected
    assert len(result.adopted) + len(result.rejected) == len(req.frequencies)
    adopted_idx = {p.index for p in result.adopted}
    for i, (f, t) in enumerate(zip(req.frequencies, req.tolerances)):
        if i in adopted_idx:
            h = next(p.harmonic for p in result.adopted if p.index == i)
            assert abs(f - h * result.f0) <= t + 1e-9
    assert result.adopted  # never empty on success


def make(freqs, tols=None, f0_min=1.0, f0_max=1000.0, H=12, rej=1):
    if tols is None:
        tols = (0.5,) * len(freqs)
    elif isinstance(tols, (int, float)):
        tols = (float(tols),) * len(freqs)
    return SolveRequest(
        tuple(freqs), tuple(tols), float(f0_min), float(f0_max), H, rej
    )


def test_exact_harmonic_series():
    # Perfect 100 Hz series, harmonics 1..8.
    req = make([100 * i for i in range(1, 9)], 0.01, 90, 110, 12, 0)
    r = solve(req)
    check_consistency(r, req)
    assert [p.harmonic for p in r.adopted] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert r.f0 == pytest.approx(100.0, abs=1e-9)
    assert r.max_relative_error == pytest.approx(0.0, abs=1e-12)


def test_noise_peak_is_rejected_and_fundamental_recovered():
    # 100 Hz series 2,3,4,6,8,9 plus a 555.2 Hz environmental spike.
    req = make(
        [200.4, 299.6, 401.1, 555.2, 600.3, 799.7, 900.5],
        1.5, 80, 130, 12, 1,
    )
    r = solve(req)
    check_consistency(r, req)
    assert len(r.adopted) == 6
    assert [p.index for p in r.rejected] == [3]
    assert [p.harmonic for p in r.adopted] == [2, 3, 4, 6, 8, 9]
    assert r.f0 == pytest.approx(100.03, abs=0.05)


def test_strictly_increasing_unique_harmonics():
    req = make([150.0, 300.0, 450.0, 600.0, 750.0, 900.0], 0.2, 140, 160, 10, 0)
    r = solve(req)
    hs = [p.harmonic for p in r.adopted]
    assert all(hs[i] < hs[i + 1] for i in range(len(hs) - 1))


def test_adopted_count_outranks_accuracy():
    # A single peak can be fit exactly, but a 6-peak joint fit with small
    # nonzero error must win because adopted count has priority.
    req = make(
        [100.0, 200.1, 300.2, 399.8, 500.3, 600.1],
        1.0, 1, 1000, 12, 5,
    )
    r = solve(req)
    check_consistency(r, req)
    assert len(r.adopted) == 6  # not the 1-peak "zero error" temptation
    assert r.max_relative_error > 0


def test_unsolvable_preserves_peaks_and_points_to_first_evidence():
    # Harmonics of two incompatible bells (~100 Hz and ~128 Hz) interleaved;
    # one rejection cannot reconcile the two fundamentals.
    req = make([200.0, 256.0, 300.0, 384.0, 512.0, 600.0], 2.0, 90, 130, 12, 1)
    r = solve(req)
    assert not r.feasible
    assert r.f0 is None
    # Original peaks are all retained, none are silently adopted.
    assert [p.frequency for p in r.rejected] == list(req.frequencies)
    assert r.adopted == []
    # 384 Hz (index 3) is the first peak no prefix joint explanation includes.
    assert r.first_unexplainable_index == 3


def test_unsolvable_evidence_prefix_property():
    # The reported k must split: every earlier prefix is explainable, k is not.
    from app.solver import _build_windows, _prefix_can_adopt

    req = make(
        [100.0, 200.0, 350.0, 400.0, 650.0, 800.0], 1.0, 90, 110, 12, 1
    )
    r = solve(req)
    assert not r.feasible
    k = r.first_unexplainable_index
    assert k is not None
    windows = _build_windows(req)
    for j in range(k):
        assert _prefix_can_adopt(req, windows, j)
    assert not _prefix_can_adopt(req, windows, k)


def test_rejection_budget_respected():
    # Three mutually incompatible groups; zero rejections allowed -> infeasible.
    req = make([100.0, 200.0, 303.0, 400.0, 505.0, 600.0], 0.2, 90, 110, 12, 0)
    r = solve(req)
    assert not r.feasible
    # With enough budget the remaining peaks become explainable.
    req2 = SolveRequest(req.frequencies, req.tolerances, 90.0, 110.0, 12, 2)
    r2 = solve(req2)
    assert r2.feasible
    check_consistency(r2, req2)
    assert len(r2.rejected) <= 2


def test_tolerance_binding_common_f0():
    # Peaks that each look like an integer multiple of *different* nearby
    # fundamentals must not pass: the shared f0 is what matters.
    req = make([100.0, 201.0, 299.0, 402.0, 499.0, 601.0], 0.05, 1, 1000, 12, 0)
    r = solve(req)
    # Tight tolerances forbid any single f0 covering all six.
    assert not r.feasible


def test_tie_break_earliest_harmonic_scheme():
    # The same peaks admit three exact interpretations: f0=100 (6,12,...),
    # f0=200 (3,6,...) or f0=300 (2,4,...).  With identical count/error, the
    # smallest harmonic tuple in entry order must win -> f0 = 300.
    req = make([600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0], 0.01,
               50, 400, 40, 0)
    r = solve(req)
    check_consistency(r, req)
    assert r.f0 == pytest.approx(300.0, abs=1e-9)
    assert [p.harmonic for p in r.adopted] == [2, 4, 6, 8, 10, 12]


def test_narrow_f0_interval_can_rule_out_assignments():
    # Peaks fit f0=100 but candidate interval only contains 120..130.
    req = make([200.0, 300.0, 400.0, 500.0, 600.0, 700.0], 1.0, 120, 130, 12, 0)
    r = solve(req)
    assert not r.feasible


def test_solver_matches_brute_force_randomized():
    """Cross-check all four tie-break rules against exhaustive enumeration."""
    from app.solver import (
        _best_f0,
        _harmonic_window,
        _nonempty,
    )
    import itertools

    def decision_key(adopted, n_peaks):
        chosen = dict(adopted)
        return [(0, chosen[i]) if i in chosen else (1, 0)
                for i in range(n_peaks)]

    def brute(req):
        n_peaks = len(req.frequencies)
        best = None
        opts = [[None] + list(range(1, req.max_harmonic + 1))
                for _ in req.frequencies]
        for choice in itertools.product(*opts):
            adopted = [(i, n) for i, n in enumerate(choice) if n is not None]
            if len(req.frequencies) - len(adopted) > req.max_rejected:
                continue
            ns = [n for _, n in adopted]
            if len(set(ns)) != len(ns) or ns != sorted(ns):
                continue
            lo, hi = req.f0_min, req.f0_max
            for i, n in adopted:
                a, b = _harmonic_window(req, i, n)
                lo, hi = max(lo, a), min(hi, b)
            if not _nonempty(lo, hi):
                continue
            f0, mr, sse = _best_f0(req, adopted, lo, hi)
            cand = (len(adopted), mr, sse, decision_key(adopted, n_peaks),
                    [n for _, n in adopted], [i for i, _ in adopted], f0)
            if best is None:
                best = cand
            elif cand[0] != best[0]:
                if cand[0] > best[0]:
                    best = cand
            elif not math.isclose(cand[1], best[1], abs_tol=1e-9):
                if cand[1] < best[1]:
                    best = cand
            elif not math.isclose(cand[2], best[2], abs_tol=1e-9):
                if cand[2] < best[2]:
                    best = cand
            elif cand[3] < best[3]:
                best = cand
        return best

    random.seed(1234)
    for _ in range(80):
        n = random.randint(3, 6)
        H = random.randint(n, 7)
        f0t = random.uniform(40, 200)
        harms = sorted(random.sample(range(1, H + 1), n))
        freqs = sorted(max(harms[k] * f0t + random.uniform(-0.5, 0.5), 1.0)
                       for k in range(n))
        tols = tuple(round(random.uniform(0.02, 1.0), 3) for _ in range(n))
        req = SolveRequest(
            tuple(round(f, 2) for f in freqs), tols,
            round(f0t * 0.75, 2), round(f0t * 1.25, 2), H,
            random.randint(0, 2),
        )
        r = solve(req)
        b = brute(req)
        if b is None:
            assert not r.feasible
            continue
        assert math.isclose(r.max_relative_error, b[1], abs_tol=1e-9)
        assert math.isclose(r.sum_squared_relative_error, b[2], abs_tol=1e-9)
        assert len(r.adopted) == b[0]
        assert [p.harmonic for p in r.adopted] == b[4]
        assert [p.index for p in r.adopted] == b[5]
