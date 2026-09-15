"""Unit tests for the MVP 2 phase models and the multiphase loop helpers."""

import math

import discopt.modeling as dm
import numpy as np
import pytest

from discopt.calphad import (
    R_GAS,
    BinarySolution,
    CompositionSet,
    Interval,
    StoichiometricPhase,
    TPoly,
    initial_set,
    is_duplicate,
    price_phase,
    tangent_distance,
)
from discopt.calphad.ops import EXPR, INTERVAL, NUM
from discopt.calphad.solvers.local import model_evaluator

RK = BinarySolution("FCC", g=(TPoly(1000.0, -2.0), TPoly(-500.0, 1.0, 0.5)), L=(TPoly(12000.0, -3.0), 4000.0, -1500.0))
AB2 = StoichiometricPhase("AB2", ("A", "B"), (1.0, 2.0), g=TPoly(-30000.0, 5.0))


def rk_reference(x, T):
    RT = R_GAS * T
    xb = 1 - x
    g = x * RK.g[0](T) + xb * RK.g[1](T) + RT * (x * math.log(x) + xb * math.log(xb))
    L = [RK.L[0](T), RK.L[1], RK.L[2]]
    g += x * xb * sum(Lk * (x - xb) ** k for k, Lk in enumerate(L))
    return g / RT


def test_tpoly():
    assert TPoly(1.0, 2.0, 3.0, 4.0, 5.0)(10.0) == pytest.approx(1 + 20 + 30 * math.log(10) + 400 + 0.5)


@pytest.mark.parametrize("x", [0.013, 0.4, 0.97])
def test_redlich_kister_num_matches_reference(x):
    assert RK.gibbs([x], NUM, 900.0) == pytest.approx(rk_reference(x, 900.0), rel=1e-13)


def test_redlich_kister_expression_through_evaluator():
    T, x0 = 900.0, 0.37
    m = dm.Model("rk")
    y = m.continuous("y", lb=1e-9, ub=1 - 1e-9)
    m.minimize(RK.gibbs([y], EXPR, T))
    ev = model_evaluator(m)
    assert float(ev.evaluate_objective(np.array([x0]))) == pytest.approx(rk_reference(x0, T), rel=1e-12)
    h = 1e-4
    fd = (rk_reference(x0 + h, T) - 2 * rk_reference(x0, T) + rk_reference(x0 - h, T)) / h**2
    H = np.asarray(ev.evaluate_hessian(np.array([x0])), dtype=float)
    assert float(H.reshape(-1)[0]) == pytest.approx(fd, rel=1e-5)


@pytest.mark.parametrize("box", [(0.0, 1e-9), (0.2, 0.3), (1 - 1e-9, 1.0)])
def test_redlich_kister_interval_encloses(box):
    T = 900.0
    enc = RK.gibbs([Interval(*box)], INTERVAL, T)
    for x in np.linspace(max(box[0], 1e-300), box[1], 7):
        v = RK.gibbs([x], NUM, T)
        assert enc.lo <= v <= enc.hi


def test_requires_temperature():
    with pytest.raises(ValueError):
        RK.gibbs([0.5], NUM)
    with pytest.raises(ValueError):
        AB2.gibbs()


def test_stoichiometric_phase_protocol():
    assert AB2.n_y == 0 and AB2.y_bounds == [] and AB2.atoms_min == 3.0
    assert AB2.amounts() == (1.0, 2.0)
    assert AB2.gibbs((), NUM, 800.0) == pytest.approx((-30000.0 + 4000.0) / (R_GAS * 800.0))
    assert AB2.initial_y([1.0, 2.0]).size == 0
    assert AB2.initial_y([0.5, 0.5]) is None
    with pytest.raises(ValueError):
        StoichiometricPhase("bad", ("A", "B"), (1.0,))


def test_compound_pricing_is_exact():
    mu = np.array([-2.0, -1.5])
    p = price_phase(AB2, mu, T=800.0)
    expected = AB2.gibbs((), NUM, 800.0) - (1.0 * -2.0 + 2.0 * -1.5)
    assert p.status == "exact" and p.node_count == 0
    assert p.lb == p.ub == pytest.approx(expected)
    assert p.ub == pytest.approx(float(tangent_distance(AB2, (), mu, NUM, 800.0)))


def test_dedup_with_compound_and_solution():
    sets = [CompositionSet(AB2, [], 1.0), CompositionSet(RK, [0.3], 1.0)]
    assert is_duplicate(sets, AB2, np.empty(0), 1e-4)
    assert not is_duplicate(sets[1:], AB2, np.empty(0), 1e-4)
    assert is_duplicate(sets, RK, [0.30001], 1e-4)
    assert not is_duplicate(sets, RK, [0.31], 1e-4)


def test_initial_set_lowest_energy_and_errors():
    other = BinarySolution("HIGH", g=(50000.0, 50000.0))
    cs = initial_set([other, RK, AB2], np.array([0.4, 0.6]), 900.0)
    assert cs.phase is RK and cs.amount == pytest.approx(1.0)
    cs = initial_set([AB2], np.array([1.0, 2.0]), 900.0)
    assert cs.phase is AB2 and cs.amount == pytest.approx(1.0)
    with pytest.raises(ValueError):
        initial_set([AB2], np.array([0.5, 0.5]), 900.0)
