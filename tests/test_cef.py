"""MVP 4: compound energy formalism phases (pycalphad as test-only oracle)."""

from itertools import product
from pathlib import Path

import numpy as np
import pytest

from discopt.calphad import ThermoDatabase, equilibrium
from discopt.calphad.database import UnsupportedModelError
from discopt.calphad.equilibrium.common import R_GAS, tangent_distance
from discopt.calphad.equilibrium.pricing import price_phase
from discopt.calphad.models import BinarySolution, CEFPhase, CEFTerm
from discopt.calphad.ops import INTERVAL, NUM, Interval
from discopt.calphad.plot import gx_curves

PYCALPHAD_R = 8.3145  # pycalphad.variables.R


def pycalphad_db(name):
    pycalphad = pytest.importorskip("pycalphad")
    return Path(pycalphad.__file__).parent / "tests" / "databases" / name


def _ideal(phase, fractions):
    return sum(a * sum(v * np.log(v) for v in d.values() if v > 0) for a, d in zip(phase.site_ratios, fractions))


# ---------------------------------------------------------------------------- model mechanics


def test_stick_breaking_round_trip_and_amounts():
    ph = CEFPhase("X", ("A", "B"), (1.0, 3.0), (("A", "B", "VA"), ("A", "B")))
    assert ph.n_y == 3 and ph.atoms_min == 3.0
    rng = np.random.default_rng(1)
    for _ in range(20):
        u = rng.uniform(size=3)
        fr = ph.site_fractions(u)
        assert all(sum(d.values()) == pytest.approx(1.0, abs=1e-14) for d in fr)
        np.testing.assert_allclose(ph.variables_from_site_fractions(fr), u, atol=1e-12)
        a = ph.amounts(u)
        assert a[0] == pytest.approx(fr[0]["A"] + 3 * fr[1]["A"])
        assert a[1] == pytest.approx(fr[0]["B"] + 3 * fr[1]["B"])
    # a vacancy on every sublattice: restricted to min_occupancy of the most atoms (issue #6)
    v = CEFPhase("V", ("A", "B"), (1.0, 1.0), (("A", "VA"), ("B", "VA")))
    assert v.vacancy_limited and v.atoms_max == 2.0 and v.atoms_min == 1.0


def _recip_phase():
    terms = (
        CEFTerm((("A",), ("A",)), 0, -1000.0),
        CEFTerm((("A",), ("B",)), 0, lambda t: -8000.0 + 2.0 * t),
        CEFTerm((("B",), ("A",)), 0, 3000.0),
        CEFTerm((("B",), ("B",)), 0, -2000.0),
        CEFTerm((("A", "B"), ("A",)), 0, 5000.0),
        CEFTerm((("A", "B"), ("*",)), 1, -1500.0),
        CEFTerm((("A",), ("A", "B")), 2, 700.0),
        CEFTerm((("A", "B"), ("A", "B")), 0, 4000.0),
        CEFTerm((("A", "B"), ("A", "B")), 1, -900.0),
        CEFTerm((("A", "B"), ("A", "B")), 2, 1100.0),
    )
    return CEFPhase("R", ("A", "B"), (1.0, 2.0), (("A", "B"), ("A", "B")), terms)


def test_reciprocal_gibbs_by_hand():
    ph, t = _recip_phase(), 900.0
    for u in ([0.3, 0.8], [0.55, 0.1], [0.97, 0.5]):
        a1, a2 = u
        b1, b2 = 1 - a1, 1 - a2
        G = a1 * a2 * -1000 + a1 * b2 * (-8000 + 2 * t) + b1 * a2 * 3000 + b1 * b2 * -2000
        G += a1 * b1 * a2 * 5000
        G += a1 * b1 * (a1 - b1) * -1500  # wildcard: Σ y on sublattice 2 is 1
        G += a1 * a2 * b2 * (a2 - b2) ** 2 * 700
        # reciprocal: order 1 multiplies the second mixing sublattice's difference, order 2 the first's
        G += a1 * b1 * a2 * b2 * (4000 - 900 * (a2 - b2) + 1100 * (a1 - b1))
        G += R_GAS * t * (a1 * np.log(a1) + b1 * np.log(b1) + 2 * (a2 * np.log(a2) + b2 * np.log(b2)))
        assert ph.gibbs(u, T=t) * R_GAS * t == pytest.approx(G, rel=1e-13)


def test_one_sublattice_cef_equals_binary_solution():
    g, L = (lambda t: -3000 - t, 1000.0), (-12000.0, 2500.0, 800.0)
    bs = BinarySolution("S", ("A", "B"), g=g, L=L, sites=2.0, fixed=(1.0, 0.0))
    terms = [CEFTerm((("A",), ("A",)), 0, g[0]), CEFTerm((("A",), ("B",)), 0, g[1])]
    terms += [CEFTerm((("A",), ("A", "B")), k, Lk) for k, Lk in enumerate(L)]
    cef = CEFPhase("S", ("A", "B"), (1.0, 2.0), (("A",), ("A", "B")), tuple(terms))
    for y in (0.02, 0.4, 0.9):
        assert cef.gibbs([y], T=750.0) == pytest.approx(bs.gibbs([y], T=750.0), rel=1e-13)
        assert cef.amounts([y]) == pytest.approx(bs.amounts([y]))


def test_ternary_interaction_muggianu():
    terms = [CEFTerm(((c,),), 0, 0.0) for c in "ABC"] + [
        CEFTerm((("A", "B", "C"),), k, v) for k, v in enumerate((6000.0, -2000.0, 900.0))
    ]
    ph = CEFPhase("T3", ("A", "B", "C"), (1.0,), (("A", "B", "C"),), tuple(terms))
    t, fr = 1000.0, [0.2, 0.5, 0.3]
    u = ph.variables_from_site_fractions([fr])
    G = np.prod(fr) * (6000 * fr[0] - 2000 * fr[1] + 900 * fr[2]) + R_GAS * t * sum(v * np.log(v) for v in fr)
    assert ph.gibbs(u, T=t) * R_GAS * t == pytest.approx(G, rel=1e-13)


def test_database_implicit_ternary_orders():
    db = ThermoDatabase.from_string(
        """
        ELEMENT A X 0 0 0 ! ELEMENT B X 0 0 0 ! ELEMENT C X 0 0 0 !
        PHASE LIQ % 1 1 ! CONST LIQ : A,B,C : !
        PARA G(LIQ,A;0) 298.15 0; 3000 N ! PARA G(LIQ,B;0) 298.15 0; 3000 N ! PARA G(LIQ,C;0) 298.15 0; 3000 N !
        PARA G(LIQ,A,B,C;0) 298.15 6000; 3000 N !
        """
    )
    ph = db.phase_model("LIQ", ("A", "B", "C"))
    assert sorted(t.order for t in ph.terms if len(t.constituents[0]) == 3) == [0, 1, 2]
    u = ph.variables_from_site_fractions([[0.2, 0.5, 0.3]])
    excess = ph.gibbs(u, T=1000.0) * R_GAS * 1000.0 - R_GAS * 1000.0 * _ideal(ph, ph.site_fractions(u))
    assert excess == pytest.approx(0.2 * 0.5 * 0.3 * 6000.0, rel=1e-12)  # L0 = L1 = L2 and Σ v = 1


def test_interval_encloses_numeric_values():
    ph, t = _recip_phase(), 600.0
    rng = np.random.default_rng(3)
    mu = np.array([-2.0, -1.5])
    for _ in range(30):
        lo = rng.uniform(0, 0.9, size=2)
        hi = lo + rng.uniform(0, 0.1, size=2)
        enc = tangent_distance(ph, [Interval(a, b) for a, b in zip(lo, hi)], mu, INTERVAL, t)
        for w in product(*zip(lo, hi)):
            v = float(tangent_distance(ph, np.array(w), mu, NUM, t))
            assert enc.lo <= v <= enc.hi


def test_gx_curves_for_cef_phase():
    ph = _recip_phase()
    x, g = gx_curves(ph, T=800.0, n=200, grid_points=40_000)
    assert np.all(np.diff(x) >= 0) and 0.0 <= x[0] < 0.01 and 0.99 < x[-1] <= 1.0
    ys = np.random.default_rng(0).uniform(size=(500, 2))
    for y in ys:  # no sampled point lies far below the returned lower envelope
        a = np.array(ph.amounts(y))
        xi, gi = a[0] / a.sum(), ph.gibbs(y, T=800.0) / a.sum()
        assert gi >= np.interp(xi, x, g) - 0.05


# ---------------------------------------------------------------------------- pycalphad oracle


CEF_CASES = [
    ("cumg.tdb", ("CU", "MG"), "CU2MG"),  # wildcard interactions
    ("cumg_parameters.tdb", ("CU", "MG"), "LAVES_C15"),
    ("alfe.tdb", ("AL", "FE"), "AL13FE4"),  # VA on a mixing sublattice
    ("Al-Mg_Zhong.tdb", ("AL", "MG"), "ALMG_GAMMA"),  # two mixing sublattices plus a singleton
    ("alni_dupin_2001.tdb", ("AL", "NI"), "AL3NI2"),  # VA and wildcards
    ("nbre_liu.tdb", ("NB", "RE"), "CHI_RENB"),
    ("crtiv_ghosh.tdb", ("CR", "TI"), "LAVES_C14"),
    ("Al-Fe_sundman2009.tdb", ("AL", "FE"), "AL8FE5_D82"),
    ("alcrni.tdb", ("AL", "CR", "NI"), "LIQUID"),  # ternary substitutional with explicit L0/L1/L2 ternary terms
    ("CrFeNb_Jacob2016.tdb", ("CR", "FE", "NB"), "LIQUID"),
    ("crtiv_ghosh.tdb", ("CR", "TI", "V"), "LAVES_C15"),  # two three-constituent sublattices
]


@pytest.mark.parametrize("fname,elements,name", CEF_CASES)
@pytest.mark.parametrize("t", [500.0, 933.47, 1400.0])
def test_cef_gibbs_matches_pycalphad(fname, elements, name, t):
    path = pycalphad_db(fname)  # skips when pycalphad is absent
    from pycalphad import Database, calculate

    model = ThermoDatabase.from_tdb(path).phase_model(name, elements)
    assert isinstance(model, CEFPhase)
    rng = np.random.default_rng(7)
    fractions = [[dict(zip(c, rng.dirichlet(np.ones(len(c))))) for c in model.constituents] for _ in range(6)]
    fractions.append([{c: (1.0 if i == 0 else 0.0) for i, c in enumerate(cons)} for cons in model.constituents])
    points = [[d[c] for cons, d in zip(model.constituents, fr) for c in cons] for fr in fractions]
    ref = calculate(
        Database(str(path)), [*elements, "VA"], name, T=t, P=101325, N=1, points=np.array(points), output="GM"
    )
    for fr, gm in zip(fractions, np.asarray(ref.GM).ravel()):
        u = model.variables_from_site_fractions(fr)
        atoms = float(np.sum(model.amounts(u)))
        ours = model.gibbs(u, T=t) * R_GAS * t / atoms
        ours += (PYCALPHAD_R - R_GAS) * t * _ideal(model, model.site_fractions(u)) / atoms
        assert ours == pytest.approx(gm, rel=1e-9, abs=1e-6), fr


def test_unsupported_cef_structure_raises():
    db = ThermoDatabase.from_string(
        """
        ELEMENT A X 0 0 0 ! ELEMENT B X 0 0 0 !
        PHASE P % 2 1 1 ! CONST P : A,B : A,B : !
        PARA G(P,A:A;0) 298.15 0; 3000 N ! PARA G(P,A:B;0) 298.15 0; 3000 N !
        PARA G(P,B:A;0) 298.15 0; 3000 N ! PARA G(P,B:B;0) 298.15 0; 3000 N !
        PARA G(P,A,B:A,B;3) 298.15 1; 3000 N !
        """
    )
    with pytest.raises(UnsupportedModelError, match="order > 2"):
        db.phase_model("P", ("A", "B"))


# ---------------------------------------------------------------------------- solver


@pytest.fixture(scope="module")
def cumg():
    return ThermoDatabase.from_tdb(pycalphad_db("cumg.tdb"))


def test_pricing_cef_phase_bound_is_valid(cumg):
    t = 700.0
    phases = cumg.phase_models(("CU", "MG"))
    cu2mg = phases["CU2MG"]
    assert cu2mg.n_y == 2
    # μ from the pure FCC Cu and HCP Mg references
    mu = np.array([phases["FCC_A1"].gibbs([1.0], T=t), phases["HCP_A3"].gibbs([0.0], T=t)])
    r = price_phase(cu2mg, mu, T=t, eps=1e-5)
    u = [v.ravel() for v in np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401))]
    grid = float(np.min(tangent_distance(cu2mg, u, mu, NUM, t)))
    assert r.lb is not None and r.lb <= grid + 1e-9
    assert r.ub == pytest.approx(grid, abs=1e-2) and r.ub <= grid + 1e-9
    assert r.destabilizing(1e-5)  # Cu2Mg is stable against the pure elements: decided in the first stage
    assert r.stages == 1


def test_staged_pricing_certifies_a_clearly_unstable_phase(monkeypatch):
    import importlib

    pricing = importlib.import_module("discopt.calphad.equilibrium.pricing")

    # min Ψ ≈ 2.743: interval branch and bound (run first for a phase this small) decides it alone
    r = price_phase(_recip_phase(), np.array([-2.0, -1.5]), T=700.0, eps=1e-5)
    assert r.certified(1e-5) and r.bound_source == "interval"
    assert r.ub == pytest.approx(2.7433, abs=1e-3)
    # without it, a full gap closure takes minutes, but LB ≥ −eps is certified in the first discopt stage
    monkeypatch.setattr(pricing, "INTERVAL_MAX_VARS", 0)
    r = price_phase(_recip_phase(), np.array([-2.0, -1.5]), T=700.0, eps=1e-5)
    assert r.certified(1e-5) and r.bound_source == "callback" and r.stages == 1
    assert r.ub == pytest.approx(2.7433, abs=1e-3)


def test_new_set_is_not_dropped_before_it_can_grow(cumg):
    # Cu2Mg destabilizes Cu-rich FCC here. Started at N ≈ 0, R used to move its site fractions to
    # where Ψ ≫ 0 and drop it, so the loop added and removed it until max_iter.
    phases = cumg.phase_models(("CU", "MG"))
    res = equilibrium(list(phases.values()), x=0.95, T=700.0, certify="none")
    assert res.phase_names == ["CU2MG", "FCC_A1"]
    assert res.n_iterations <= 3 and not any(rec.removed for rec in res.history)
    x_cu = {cs.phase.name: cs.x[0] for cs in res.composition_sets}
    assert x_cu["FCC_A1"] == pytest.approx(0.9638, abs=1e-3)  # pycalphad 0.11.2
    assert x_cu["CU2MG"] == pytest.approx(0.6675, abs=1e-3)
    assert res.G * R_GAS * 700.0 == pytest.approx(-30115.04, abs=0.5)


@pytest.mark.slow
@pytest.mark.parametrize("x_cu,T", [(0.75, 700.0), (0.62, 900.0)])
def test_cumg_equilibrium_with_cef_phase_matches_pycalphad(cumg, x_cu, T):
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    phases = cumg.phase_models(("CU", "MG"))
    res = equilibrium(list(phases.values()), x=x_cu, T=T)
    print(res.summary())
    assert res.certificate.level.name == "CERTIFIED_TO_TOLERANCE"
    assert "CU2MG" in res.phase_names

    dbf = Database(str(pycalphad_db("cumg.tdb")))
    eq = pyeq(dbf, ["CU", "MG", "VA"], sorted(phases), {v.X("CU"): x_cu, v.T: T, v.P: 101325, v.N: 1})
    ref_phases = sorted({str(p) for p in np.asarray(eq.Phase).ravel() if str(p)})
    assert res.phase_names == ref_phases
    gm = float(np.asarray(eq.GM).ravel()[0])
    mu_ref = np.asarray(eq.MU).ravel()
    atoms = float(res.b.sum())
    assert res.G * R_GAS * T / atoms == pytest.approx(gm, abs=0.5)
    np.testing.assert_allclose(res.mu * R_GAS * T, mu_ref, atol=2.0)
