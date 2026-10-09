"""MVP 2: several phases, temperature dependence, phase appearance and disappearance."""

import numpy as np
import pytest
from conftest import EPS

from discopt.calphad import (
    R_GAS,
    BinarySolution,
    CertificateLevel,
    CompositionSet,
    StoichiometricPhase,
    TPoly,
    equilibrium,
)
from discopt.calphad.plot import gx_curves, lower_convex_hull

# Ideal liquid and ideal solid with equal entropies of fusion: an analytic lens.
# A melts at 1000 K and B at 1500 K; each has dS_fus = 10 J/(mol K).
LIQ = BinarySolution("LIQUID", g=(TPoly(10000.0, -10.0), TPoly(15000.0, -10.0)))
FCC = BinarySolution("FCC_A1")


def lens(T):
    """Analytic tie-line (x_A in the liquid, x_A in the solid) for the ideal lens."""
    KA = np.exp(LIQ.g[0](T) / (R_GAS * T))  # x_A^S / x_A^L
    KB = np.exp(LIQ.g[1](T) / (R_GAS * T))
    xl = (1 - KB) / (KA - KB)
    return xl, KA * xl


@pytest.fixture(scope="module")
def lens_result():
    return equilibrium([LIQ, FCC], x=0.5, T=1250.0, eps=EPS)


def test_lens_tie_line(lens_result):
    res = lens_result
    xl, xs = lens(1250.0)
    assert xs < 0.5 < xl
    assert res.certificate.level == CertificateLevel.CERTIFIED_TO_TOLERANCE
    assert res.phase_names == ["FCC_A1", "LIQUID"]
    by = {cs.phase.name: cs for cs in res.composition_sets}
    assert by["LIQUID"].x[0] == pytest.approx(xl, abs=1e-6)
    assert by["FCC_A1"].x[0] == pytest.approx(xs, abs=1e-6)
    assert by["LIQUID"].amount == pytest.approx((0.5 - xs) / (xl - xs), abs=1e-6)
    # mu_i / RT = g_i^S / RT + ln x_i^S, where g^S = 0
    np.testing.assert_allclose(res.mu, [np.log(xs), np.log(1 - xs)], atol=1e-6)


@pytest.mark.parametrize(
    "T,start,stable",
    [(900.0, LIQ, "FCC_A1"), (1600.0, FCC, "LIQUID")],
    ids=["solid-appears-liquid-vanishes", "liquid-appears-solid-vanishes"],
)
def test_phase_appearance_and_disappearance(T, start, stable):
    res = equilibrium([LIQ, FCC], x=0.5, T=T, eps=EPS, initial_sets=[CompositionSet(start, [0.5], 1.0)])
    assert res.certificate.level == CertificateLevel.CERTIFIED_TO_TOLERANCE
    assert res.phase_names == [stable]
    assert any(start.name in [s.phase.name for s in rec.removed] for rec in res.history)
    (cs,) = res.composition_sets
    assert cs.x[0] == pytest.approx(0.5, abs=1e-8)


def test_initial_set_picks_lowest_energy_phase():
    res = equilibrium([LIQ, FCC], x=0.5, T=900.0, eps=EPS)
    assert res.n_iterations == 1 and res.phase_names == ["FCC_A1"]
    assert res.history[0].restricted.sets[0].phase is FCC


# Three phases at T = 1000 K: a liquid with a small negative L0, an FCC with a
# miscibility gap, and a line compound AB.
T3 = 1000.0
LIQ3 = BinarySolution("LIQUID", g=(TPoly(10000.0, -10.0), TPoly(15000.0, -10.0)), L=(-5000.0,))
FCC3 = BinarySolution("FCC_A1", L=(TPoly(20000.0), 3000.0))
AB = StoichiometricPhase("AB", ("A", "B"), (0.5, 0.5), g=TPoly(-5000.0))
PHASES3 = [LIQ3, FCC3, AB]


def hull_reference(phases, T, x):
    """G/RT per atom on the lower convex hull at x, and the phases at the ends of that hull segment."""
    xs, gs, names = [], [], []
    for ph in phases:
        px, pg = gx_curves(ph, T, n=200_001)
        xs.append(px)
        gs.append(pg)
        names += [ph.name] * px.size
    xs, gs, names = np.concatenate(xs), np.concatenate(gs), np.array(names)
    hx, hg = lower_convex_hull(xs, gs)
    g_at = float(np.interp(x, hx, hg))
    i = int(np.searchsorted(hx, x))
    ends = {hx[max(i - 1, 0)], hx[min(i, hx.size - 1)]}
    present = set()
    for e in ends:
        on_hull = np.isclose(xs, e, atol=0) & np.isclose(gs, hg[hx == e][0], rtol=0, atol=1e-12)
        present |= set(names[on_hull])
    return g_at, present


@pytest.mark.parametrize("x", [0.3, pytest.param(0.8, marks=pytest.mark.slow)])
def test_three_phase_system_matches_convex_hull(x):
    res = equilibrium(PHASES3, x=x, T=T3, eps=EPS)
    g_ref, present = hull_reference(PHASES3, T3, x)
    assert res.certificate.level == CertificateLevel.CERTIFIED_TO_TOLERANCE, res.summary()
    assert res.G == pytest.approx(g_ref, abs=1e-7)
    assert set(res.phase_names) == present


# ---------------------------------------------------------------------------- a compound at its own composition


def test_compounds_only_solve_as_a_linear_program():
    """One compound at its composition leaves μ free along one direction; two compounds fix it."""
    from discopt.calphad import CompositionSet, StoichiometricPhase, solve_restricted

    ab = StoichiometricPhase("AB", ("A", "B"), (1.0, 1.0), g=-4000.0)
    ab2 = StoichiometricPhase("AB2", ("A", "B"), (1.0, 2.0), g=-5000.0)
    r = solve_restricted([CompositionSet(ab, [])], [0.5, 0.5], T=1000.0)
    assert r.converged and r.mu_directions is not None and r.mu_directions.shape == (2, 1)
    assert r.sets[0].amount == pytest.approx(0.5)
    assert r.mu @ [1.0, 1.0] == pytest.approx(float(ab.gibbs([], T=1000.0)))  # on the compound's tangent
    r2 = solve_restricted([CompositionSet(ab, []), CompositionSet(ab2, [])], [0.45, 0.55], T=1000.0)
    assert r2.converged and r2.mu_directions is None and r2.residual < 1e-10


@pytest.mark.parametrize("x", [0.375, 0.75])
def test_equilibrium_at_exactly_a_compound_composition(x):
    """Al3Ni5 (x_Al = 0.375) and Al3Ni (0.75) at 1000 K. The solve used to fail: restricted solve status: error."""
    from discopt.calphad import databases

    db = databases.load("alni_dupin")
    phases = list(db.phase_models(("AL", "NI")).values())
    res = equilibrium(phases, x=x, T=1000.0, certify="none")
    assert res.certificate.level.name == "STABILITY_SEARCHED"
    assert res.phase_names == ["AL3NI5" if x == 0.375 else "AL3NI1"]


def test_lone_compound_is_certified_with_a_centred_mu():
    """CuMg2 at its own composition: μ is chosen in the middle of its stability range, and certifies."""
    from discopt.calphad import databases

    db = databases.load("cumg_nims")
    res = equilibrium(list(db.phase_models(("CU", "MG")).values()), x=1 / 3, T=700.0, eps=1e-5)
    assert res.phase_names == ["CUMG2"]
    assert res.certificate.level.name in ("CERTIFIED_TO_TOLERANCE", "GLOBALLY_CERTIFIED")
    # every other phase sits clearly above the tangent, not just touching it
    others = [p for p in res.history[-1].pricing if p.phase.name != "CUMG2"]
    assert min(p.ub for p in others) > 1e-3
