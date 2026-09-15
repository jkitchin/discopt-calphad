"""MVP 1: a binary regular solution with a miscibility gap, one test per step (architecture.md §8)."""

import discopt.modeling as dm
import numpy as np
import pytest
from conftest import EPS

from discopt.calphad import (
    EXPR,
    NUM,
    CertificateLevel,
    CompositionSet,
    RegularSolution,
    equilibrium,
    is_duplicate,
    tangent_distance,
)
from discopt.calphad.equilibrium.pricing import price_phase
from discopt.calphad.solvers import model_evaluator


def test_step1_gibbs_energy_is_nonconvex(gap_phase):
    """G'' < 0 inside the spinodal, computed from the expression lowered to discopt."""
    m = dm.Model("g")
    x = m.continuous("x", lb=1e-9, ub=1 - 1e-9)
    m.minimize(gap_phase.gibbs([x], EXPR))
    ev = model_evaluator(m)
    for xv in (0.05, 0.2, 0.5, 0.8, 0.95):
        pt = np.array([xv])
        assert ev.evaluate_objective(pt) == pytest.approx(gap_phase.gibbs([xv], NUM), abs=1e-14)
        assert ev.evaluate_hessian(pt)[0, 0] == pytest.approx(gap_phase.d2gibbs(xv), rel=1e-10)
    s1, s2 = gap_phase.spinodal()
    assert ev.evaluate_hessian(np.array([0.5]))[0, 0] < 0
    assert ev.evaluate_hessian(np.array([0.5 * s1]))[0, 0] > 0
    assert gap_phase.d2gibbs(s1) == pytest.approx(0.0, abs=1e-10)


def test_step2_single_set_is_kkt_point_but_unstable(gap_result):
    first = gap_result.history[0]
    R = first.restricted
    assert R.converged
    assert len(R.sets) == 1 and R.sets[0].y[0] == pytest.approx(0.5, abs=1e-8)
    assert R.residual < 1e-8 and R.tangency_residual < 1e-8 and R.primal_infeasibility < 1e-10
    (p,) = first.pricing
    assert p.ub < -EPS


def test_step3_global_pricing_brackets_dense_grid(gap_result, gap_phase):
    (staged,) = gap_result.history[0].pricing
    ys = np.linspace(0, 1, 400_001)
    psi = tangent_distance(gap_phase, [ys], staged.mu, NUM)
    grid_min = psi.min()
    # inside the loop, pricing stops once the sign is decided: a valid but loose bracket
    assert staged.destabilizing(EPS) and staged.ub >= grid_min - 1e-12
    assert staged.lb is None or staged.lb <= grid_min + 1e-12
    # with eps=None, the gap closes fully
    p = price_phase(gap_phase, staged.mu)
    assert p.lb is not None and p.gap_certified
    assert p.lb <= grid_min + 1e-12
    assert p.ub == pytest.approx(grid_min, abs=1e-8)
    assert p.ub - p.lb < 1e-5
    # the minimizer lies in a destabilizing basin outside the spinodal region
    s1, s2 = gap_phase.spinodal()
    assert not (s1 < p.y[0] < s2)
    assert min(abs(p.y[0] - xb) for xb in gap_phase.binodal()) < 1e-4


def test_step4_new_set_added_and_duplicates_rejected(gap_result, gap_phase):
    first = gap_result.history[0]
    assert len(first.added) == 1 and not first.rejected
    existing = [CompositionSet(gap_phase, [0.5], 1.0)]
    assert is_duplicate(existing, gap_phase, [0.5 + 1e-6], tol=1e-4)
    assert not is_duplicate(existing, gap_phase, first.added[0].y, tol=1e-4)
    other = RegularSolution("OTHER", omega=3.0)
    assert not is_duplicate(existing, other, [0.5], tol=1e-4)


def test_step5_resolve_converges_with_two_sets(gap_result):
    R = gap_result.history[-1].restricted
    assert R.converged
    assert len(gap_result.composition_sets) == 2
    assert all(cs.amount > 1e-3 for cs in gap_result.composition_sets)
    assert R.residual < 1e-8 and R.primal_infeasibility < 1e-10


def test_step6_binodal_common_tangent_and_lever_rule(gap_result, gap_phase):
    xa, xb = gap_phase.binodal()
    sets = sorted(gap_result.composition_sets, key=lambda cs: cs.x[0])
    assert sets[0].x[0] == pytest.approx(xa, abs=1e-6)
    assert sets[1].x[0] == pytest.approx(xb, abs=1e-6)
    # common tangent: G'(xa) = mu_A - mu_B, G(xa) = mu_A xa + mu_B (1 - xa)
    om = gap_phase.omega
    dG = np.log(xa) - np.log(1 - xa) + om * (1 - 2 * xa)
    mu_B = gap_phase.gibbs([xa], NUM) - xa * dG
    mu_A = mu_B + dG
    np.testing.assert_allclose(gap_result.mu, [mu_A, mu_B], atol=1e-6)
    # lever rule at overall x = 0.5
    total = sum(cs.amount for cs in sets)
    assert total == pytest.approx(1.0, abs=1e-10)
    assert sets[0].amount == pytest.approx((xb - 0.5) / (xb - xa), abs=1e-6)


def test_step7_certificate(gap_result, convex_result):
    cert = gap_result.certificate
    assert all(lb is not None and lb >= -EPS for lb in cert.lower_bounds.values())
    assert cert.level == CertificateLevel.CERTIFIED_TO_TOLERANCE
    assert cert.gap_bound is not None and cert.gap_bound <= cert.n_total_max * EPS
    assert gap_result.n_iterations == 2

    c2 = convex_result.certificate
    assert c2.level == CertificateLevel.CERTIFIED_TO_TOLERANCE
    assert convex_result.n_iterations == 1
    (cs,) = convex_result.composition_sets
    assert cs.x[0] == pytest.approx(0.5, abs=1e-8)


@pytest.mark.slow
def test_asymmetric_endmembers_off_center():
    """Linear g terms leave the binodal unchanged; the lever rule holds at x = 0.3."""
    phase = RegularSolution("LIQ", omega=3.0, g=(-0.3, 0.2))
    res = equilibrium(phase, x=0.3, eps=EPS)
    xa, xb = RegularSolution("REF", omega=3.0).binodal()
    sets = sorted(res.composition_sets, key=lambda cs: cs.x[0])
    assert res.certificate.level == CertificateLevel.CERTIFIED_TO_TOLERANCE
    assert [cs.x[0] for cs in sets] == pytest.approx([xa, xb], abs=1e-6)
    assert sets[0].amount == pytest.approx((xb - 0.3) / (xb - xa), abs=1e-6)
