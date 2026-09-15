"""Solver-free unit tests: backends, interval soundness, boundary bound, certificate levels."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from discopt.calphad import INTERVAL, NUM, CertificateLevel, Interval, RegularSolution, certify, tangent_distance
from discopt.calphad.equilibrium import boundary_lower_bound
from discopt.calphad.plot import lower_convex_hull


def test_num_xlogx_at_zero():
    assert NUM.xlogx(0.0) == 0.0
    np.testing.assert_allclose(NUM.xlogx(np.array([0.0, 0.5, 1.0])), [0.0, 0.5 * math.log(0.5), 0.0])


@pytest.mark.parametrize("lo,hi", [(0.0, 1e-9), (0.0, 0.2), (0.0, 1 / math.e), (0.1, 0.9), (0.5, 1.0), (0.0, 1.0)])
def test_interval_xlogx_encloses_samples(lo, hi):
    iv = INTERVAL.xlogx(Interval(lo, hi))
    t = np.linspace(lo, hi, 20001)
    v = NUM.xlogx(t)
    assert iv.lo <= v.min() and v.max() <= iv.hi
    assert iv.lo >= v.min() - 1e-6  # tight: the extremes are attained


def test_interval_xlogx_rejects_negative():
    with pytest.raises(ValueError):
        INTERVAL.xlogx(Interval(-0.1, 0.5))


def test_boundary_lower_bound_is_valid():
    phase = RegularSolution("LIQ", omega=3.0, g=(-0.3, 0.2))
    mu = np.array([-0.4, 0.1])
    floor = 1e-6
    lb = boundary_lower_bound(phase, mu, floor)
    ys = np.concatenate([np.linspace(0, floor, 1001), np.linspace(1 - floor, 1, 1001)])
    assert lb <= tangent_distance(phase, [ys], mu, NUM).min()
    assert lb >= tangent_distance(phase, [ys], mu, NUM).min() - 1e-5


def test_lower_convex_hull():
    x = np.linspace(0, 1, 101)
    g = -np.sin(np.pi * x) * 0 + (x - 0.5) ** 4 - 0.1 * (x - 0.5) ** 2
    hx, hg = lower_convex_hull(x, g)
    assert hx[0] == 0 and hx[-1] == 1
    assert np.all(np.diff(np.diff(hg) / np.diff(hx)) >= -1e-12)


def _fake(lbs, ubs, *, converged=True, residual=0.0):
    R = SimpleNamespace(
        converged=converged,
        status="optimal" if converged else "iteration_limit",
        residual=residual,
        primal_infeasibility=0.0,
    )
    ph = lambda i: SimpleNamespace(name=f"P{i}", atoms_min=1.0)
    P = [
        SimpleNamespace(phase=ph(i), lb=lb, ub=ub, destabilizing=lambda eps, ub=ub: ub < -eps)
        for i, (lb, ub) in enumerate(zip(lbs, ubs))
    ]
    return R, P


@pytest.mark.parametrize(
    "lbs,ubs,kw,level",
    [
        ([-1e-6], [0.0], {"converged": False}, CertificateLevel.FAILED),
        ([], [], {}, CertificateLevel.LOCAL_EQUILIBRIUM),
        ([-1.0], [-0.1], {}, CertificateLevel.LOCAL_EQUILIBRIUM),
        ([None], [0.0], {}, CertificateLevel.STABILITY_SEARCHED),
        ([-1e-6, None], [0.0, 0.0], {}, CertificateLevel.GLOBAL_SEARCH_INCOMPLETE),
        ([-1e-3], [0.0], {}, CertificateLevel.GLOBAL_SEARCH_INCOMPLETE),
        ([-1e-6, 0.2], [0.0, 0.3], {}, CertificateLevel.CERTIFIED_TO_TOLERANCE),
        ([-1e-12], [0.0], {"residual": 1e-12}, CertificateLevel.GLOBALLY_CERTIFIED),
        ([-1e-12], [0.0], {"residual": 1e-6}, CertificateLevel.CERTIFIED_TO_TOLERANCE),
    ],
)
def test_certificate_levels(lbs, ubs, kw, level):
    R, P = _fake(lbs, ubs, **kw)
    cert = certify(R, P, b=[0.5, 0.5], eps=1e-5, T=1000.0)
    assert cert.level == level
    if level >= CertificateLevel.CERTIFIED_TO_TOLERANCE:
        assert cert.gap_bound == pytest.approx(kw.get("residual", 0.0) + max(0.0, -min(lbs)))
        assert cert.eps_J_per_mol == pytest.approx(1e-5 * 8.314462618 * 1000.0)
