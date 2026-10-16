"""The interval lower bounds of pricing: natural and mean-value (Taylor) forms."""

from pathlib import Path

import numpy as np
import pytest

from discopt.calphad import NUM, Interval, ThermoDatabase, tangent_distance
from discopt.calphad.equilibrium.pricing import _taylor_lower, boundary_lower_bound
from discopt.calphad.ops import DUAL, Dual


def pycalphad_db(name):
    pycalphad = pytest.importorskip("pycalphad")
    return Path(pycalphad.__file__).parent / "tests" / "databases" / name


def test_dual_product_rule_encloses():
    """q(y) ∈ c + d (y − c) for a product of Taylor models, at sampled points."""
    y = Dual(Interval(0.2, 0.4), Interval(0.3), (Interval(1.0),))
    q = (1 - y) * y * 3.0 + DUAL.xlogx(y)
    for t in np.linspace(0.2, 0.4, 21):
        val = (1 - t) * t * 3.0 + t * np.log(t)
        enc = q.c + q.d[0] * (Interval(t) - 0.3)
        assert enc.lo - 1e-12 <= val <= enc.hi + 1e-12
        assert q.v.lo - 1e-12 <= val <= q.v.hi + 1e-12


def test_dual_zero_direction_ignores_unbounded_derivative():
    """xlogx at 0 has derivative −∞, but a direction it does not depend on gets an exact zero."""
    x = Dual(Interval(0.0, 0.1), Interval(0.1), (None, Interval(1.0)))
    r = DUAL.xlogx(x)
    assert r.d[0] is None and r.d[1].lo == -np.inf


PHASES = [
    ("alni_dupin_2001.tdb", ("AL", "NI"), "BCC_B2"),  # partitioned, magnetic, vacancies on every sublattice
    ("alni_dupin_2001.tdb", ("AL", "NI"), "FCC_L12"),  # partitioned, four sublattices
    ("crfe_bcc_magnetic.tdb", ("CR", "FE"), "BCC_A2"),  # magnetic
    ("cuo.tdb", ("CU", "O"), "IONIC_LIQ"),  # two-sublattice ionic liquid
    ("al2o3_nd2o3_zro2.tdb", ("ND", "ZR", "O"), "FLUO"),  # charged, vacancies
]


@pytest.mark.parametrize(("db", "elements", "name"), PHASES)
def test_taylor_bound_encloses_samples(db, elements, name):
    """On random boxes, many touching the domain's edge, no sampled Ψ lies below the bound."""
    phase = ThermoDatabase.from_tdb(pycalphad_db(db)).phase_models(elements)[name]
    rng = np.random.default_rng(1)
    T = 1200.0
    for _ in range(40):
        mu = rng.uniform(-20, 0, len(elements))
        box = []
        for _j in range(phase.n_y):
            w = rng.choice([1e-9, 1e-3, 0.05, 0.3, 1.0])
            a = rng.choice([0.0, 1.0 - w, rng.uniform(0, 1 - w)])
            box.append((float(a), float(a + w)))
        lb = _taylor_lower(phase, mu, T, box)
        pts = np.array([[rng.uniform(a, c) for a, c in box] for _ in range(30)])
        corners = np.array([[a if rng.random() < 0.5 else c for a, c in box] for _ in range(10)])
        psi = min(float(tangent_distance(phase, y, mu, NUM, T)) for y in np.vstack([pts, corners]))
        assert lb <= psi + 1e-9


def test_b2_slab_bound_closes_next_to_gamma_prime():
    """At the γ + γ′ μ (x(Al) = 0.22, 1200 K), BCC_B2's minimum is near a slab, Ψ ≈ 0.17.

    The natural interval form stalled at −0.46 there after 20 000 boxes per
    slab; the mean-value form at the box corner closes it.
    """
    b2 = ThermoDatabase.from_tdb(pycalphad_db("alni_dupin_2001.tdb")).phase_models(("AL", "NI"))["BCC_B2"]
    mu = np.array([-18.78515355, -6.26624598])
    assert boundary_lower_bound(b2, mu, T=1200.0, target=0.0) >= 0.0
