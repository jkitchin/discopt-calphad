"""Ionic models, issue #6: species, charged CEF phases, and the two-sublattice ionic liquid."""

from pathlib import Path

import numpy as np
import pytest

from discopt.calphad import ThermoDatabase, equilibrium
from discopt.calphad.database import Species
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.equilibrium.restricted import CompositionSet, _implied_neutrality
from discopt.calphad.models import CEFPhase
from discopt.calphad.models.ionic import IonicLiquidPhase
from discopt.calphad.ops import NUM

PYCALPHAD_R = 8.3145


def pycalphad_db(name):
    pycalphad = pytest.importorskip("pycalphad")
    return Path(pycalphad.__file__).parent / "tests" / "databases" / name


def _xlx(d):
    return sum(v * np.log(v) for v in d.values() if v > 0)


def _ideal(m, u):
    """The terms of G/RT that carry R: the configurational entropy."""
    sf = m.site_fractions(u)
    if isinstance(m, IonicLiquidPhase):
        P, Q = m.ratios(sf)
        return P * _xlx(sf[0]) + Q * _xlx(sf[1])
    return sum(a * _xlx(d) for a, d in zip(m.site_ratios, sf))


def test_species_composition_and_charge():
    els = ["AL", "O", "CU", "Y", "C", "U"]
    assert Species("CU+1", "CU1/+1").composition(els) == ({"CU": 1.0}, 1.0)
    assert Species("O-2", "O1/-2").composition(els) == ({"O": 1.0}, -2.0)
    assert Species("ALO3/2", "AL1O1.5").composition(els) == ({"AL": 1.0, "O": 1.5}, 0.0)
    assert Species("YO3/2", "O1.5Y1").composition(els) == ({"O": 1.5, "Y": 1.0}, 0.0)
    assert Species("CUO", "CU1O1").composition(els) == ({"CU": 1.0, "O": 1.0}, 0.0)  # Cu, not C and U
    with pytest.raises(ValueError, match="cannot read"):
        Species("X", "QQ2").composition(els)


def test_undefined_r_is_the_gas_constant():
    """cuo.tdb writes R*T*LN(1E-05*P) in the gas parameter, with no FUNCTION R."""
    db = ThermoDatabase.from_tdb(pycalphad_db("cuo.tdb"))
    gas = db.phase_model("GAS", ("CU", "O"))
    assert gas.stoichiometry == (0.0, 2.0)  # one O2 per formula unit
    g = float(gas.gibbs([], NUM, 1000.0)) * R_GAS * 1000.0
    ref = 2 * db.evaluate("GHSEROO", 1000.0) + R_GAS * 1000.0 * np.log(1e-5 * 101325)
    assert g == pytest.approx(ref, rel=1e-12)


@pytest.fixture(scope="module")
def cuo():
    return ThermoDatabase.from_tdb(pycalphad_db("cuo.tdb"))


def test_ionic_liquid_site_ratios_and_neutrality(cuo):
    liq = cuo.phase_model("IONIC_LIQ", ("CU", "O"))
    assert isinstance(liq, IonicLiquidPhase) and liq.n_y == 3 and not liq.charged
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = rng.uniform(0, 1, 3)
        sf = liq.site_fractions(u)
        P, Q = liq.ratios(sf)
        assert Q == pytest.approx(sf[0]["CU+1"] + 2 * sf[0]["CU+2"] + 3 * sf[0]["CU+3"])
        assert P == pytest.approx(2 * sf[1]["O-2"] + Q * sf[1]["VA"])
        # neutral for every y: cation charge P·Q equals anion charge Q·2 y_O
        assert P * Q - Q * 2 * sf[1]["O-2"] == pytest.approx(Q * Q * sf[1]["VA"])
        cu, o = liq.amounts(u)
        assert cu == pytest.approx(P) and o == pytest.approx(Q * sf[1]["O-2"])


def test_ionic_liquid_atoms_min_is_at_a_vertex(cuo):
    """Atoms per formula unit are not linear in y, but their minimum is still at a vertex."""
    liq = cuo.phase_model("IONIC_LIQ", ("CU", "O"))
    pts = np.random.default_rng(1).uniform(0, 1, size=(20000, 3))
    atoms = np.array([sum(liq.amounts(u)) for u in pts])
    assert atoms.min() >= liq.atoms_min - 1e-12
    assert liq.atoms_min == pytest.approx(1.0)  # Cu+1:Va, metallic copper


def test_neutral_species_without_vacancy_follows_sundman():
    """(ND+3, ZR+4)(O-2, ALO3/2): the neutral endmember is multiplied by Q. pycalphad 0.11.2 raises here."""
    db = ThermoDatabase.from_tdb(pycalphad_db("al2o3_nd2o3_zro2.tdb"))
    liq = db.phase_model("I_LIQUID", ("AL", "ND", "ZR", "O"))
    T = 2500.0
    u = liq.variables_from_site_fractions([{"ND+3": 0.4, "ZR+4": 0.6}, {"ALO3/2": 0.3, "O-2": 0.7}])
    sf = liq.site_fractions(u)
    P, Q = liq.ratios(sf)
    assert Q == pytest.approx(0.4 * 3 + 0.6 * 4) and P == pytest.approx(0.7 * 2)

    def g(label):
        p = next(p for p in db.phase_parameters("I_LIQUID") if p.label == label)
        return db.evaluate(p.function, T)

    expected = (
        0.4 * 0.7 * g("G(I_LIQUID,ND+3:O-2;0)")
        + 0.6 * 0.7 * g("G(I_LIQUID,ZR+4:O-2;0)")
        + Q * 0.3 * g("G(I_LIQUID,ALO3/2;0)")
    )
    interactions = [p for p in db.phase_parameters("I_LIQUID") if max(len(c) for c in p.constituents) > 1]
    for p in interactions:
        cons = p.constituents
        if cons == (("ND+3", "ZR+4"), ("O-2",)):
            expected += 0.4 * 0.6 * 0.7 * (0.4 - 0.6) ** p.order * db.evaluate(p.function, T)
        elif cons == (("ND+3",), ("ALO3/2", "O-2")):
            expected += 0.4 * 0.3 * 0.7 * (0.7 - 0.3) ** p.order * db.evaluate(p.function, T)  # O-2 first
        elif cons == (("ZR+4",), ("ALO3/2", "O-2")):
            expected += 0.6 * 0.3 * 0.7 * (0.7 - 0.3) ** p.order * db.evaluate(p.function, T)
        elif cons == (("ND+3", "ZR+4"), ("ALO3/2", "O-2")) and p.order == 0:  # reciprocal, anions present
            expected += 0.4 * 0.6 * 0.3 * 0.7 * db.evaluate(p.function, T)
        else:
            pytest.skip(f"no hand formula for {p.label}")
    expected = expected / (R_GAS * T) + P * _xlx(sf[0]) + Q * _xlx(sf[1])
    assert float(liq.gibbs(u, NUM, T)) == pytest.approx(expected, rel=1e-12)


def test_charged_cef_phase_needs_neutrality():
    db = ThermoDatabase.from_tdb(pycalphad_db("al2o3_nd2o3_zro2.tdb"))
    fluo = db.phase_model("FLUO", ("ND", "ZR", "O"))
    assert isinstance(fluo, CEFPhase) and fluo.charged
    # (ND+3, ZR+4)_2 (O-2, VA)_4: neutral when 2(3 y_Nd + 4 y_Zr) = 8 y_O
    u = fluo.variables_from_site_fractions([{"ND+3": 0.5, "ZR+4": 0.5}, {"O-2": 0.875, "VA": 0.125}])
    assert float(fluo.charge(u)) == pytest.approx(0.0, abs=1e-12)
    u2 = fluo.variables_from_site_fractions([{"ND+3": 0.5, "ZR+4": 0.5}, {"O-2": 0.5, "VA": 0.5}])
    assert float(fluo.charge(u2)) == pytest.approx(3.0)
    b = np.array([0.3, 0.7, 0.3 * 1.5 + 0.7 * 2])
    y0 = fluo.initial_y(b)
    assert abs(float(fluo.domain_equalities(y0)[0])) < 1e-9
    a = np.array(fluo.amounts(y0))
    assert a / a.sum() == pytest.approx(b / b.sum())


def test_redundant_neutrality_is_detected():
    """With fixed valences and a neutral b, one neutrality row follows from the balance rows."""
    db = ThermoDatabase.from_tdb(pycalphad_db("al2o3_nd2o3_zro2.tdb"))
    fluo = db.phase_model("FLUO", ("ND", "ZR", "O"))
    pyro = db.phase_model("PYRO", ("ND", "ZR", "O"))
    y_f = fluo.initial_y(np.array([0.5, 0.5, 1.75]))
    y_p = pyro.initial_y(np.array([0.5, 0.5, 1.75]))
    sets = [CompositionSet(fluo, y_f, 0.5), CompositionSet(pyro, y_p, 0.1)]
    els = ("ND", "ZR", "O")
    assert _implied_neutrality(sets, np.array([0.5, 0.5, 1.75]), els, [True, True]) == 0
    assert _implied_neutrality(sets, np.array([0.5, 0.5, 1.75]), els, [False, True]) == 1  # held sets have no row
    assert _implied_neutrality(sets, np.array([0.5, 0.5, 2.0]), els, [True, True]) is None  # b not neutral


# ---------------------------------------------------------------------------- pycalphad oracle

CASES = [
    ("cuo.tdb", ("CU", "O"), "IONIC_LIQ"),
    ("femns.tdb", ("FE", "MN", "S"), "IONIC_LIQ"),  # (Fe+2, Mn+2)(S-2, Va, S): a neutral S as well
    ("al2o3_nd2o3_zro2.tdb", ("ND", "ZR", "O"), "I_LIQUID"),
    ("al2o3_nd2o3_zro2.tdb", ("AL", "ND", "ZR", "O"), "FLUO"),
    ("al2o3_nd2o3_zro2.tdb", ("ND", "ZR", "O"), "PYRO"),
    ("al2o3_nd2o3_zro2.tdb", ("ND", "ZR", "O"), "ND2O3_A"),
    ("zrlayalo.tdb", ("LA", "Y", "ZR", "O"), "FLUORITE"),
]


@pytest.mark.parametrize("fname,elements,name", CASES)
@pytest.mark.parametrize("t", [1500.0, 2500.0])
def test_ionic_gibbs_matches_pycalphad(fname, elements, name, t):
    path = pycalphad_db(fname)
    from pycalphad import Database, calculate

    model = ThermoDatabase.from_tdb(path).phase_model(name, elements)
    rng = np.random.default_rng(5)
    fractions = [[dict(zip(c, rng.dirichlet(np.ones(len(c))))) for c in model.constituents] for _ in range(8)]
    points = [[d[c] for cons, d in zip(model.constituents, fr) for c in cons] for fr in fractions]
    ref = calculate(Database(str(path)), [*elements, "VA"], name, T=t, P=101325, N=1, points=np.array(points), output="GM")
    for fr, gm in zip(fractions, np.asarray(ref.GM).ravel()):
        u = model.variables_from_site_fractions(fr)
        atoms = float(np.sum(model.amounts(u)))
        ours = (float(model.gibbs(u, NUM, t)) * R_GAS * t + (PYCALPHAD_R - R_GAS) * t * _ideal(model, u)) / atoms
        assert ours == pytest.approx(gm, rel=1e-9, abs=1e-6), fr


@pytest.mark.parametrize(
    "xo,t",
    [(0.05, 1500.0), (0.4, 1000.0), (0.55, 1000.0), (0.3, 1400.0), (0.02, 1200.0), (0.34, 1600.0)],
)
def test_cu_o_matches_pycalphad(cuo, xo, t):
    """Cu–O (Schramm et al. 2005): ionic liquid, Cu2O, CuO, FCC and O2 gas."""
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    models = cuo.phase_models(("CU", "O"))
    res = equilibrium(list(models.values()), x=1 - xo, T=t, certify="none")
    ours = sorted((cs.phase.name, round(float(cs.x[1]), 3)) for cs in res.composition_sets if cs.amount > 1e-8)
    path = pycalphad_db("cuo.tdb")
    eq = pyeq(Database(str(path)), ["CU", "O", "VA"], sorted(models), {v.X("O"): xo, v.T: t, v.P: 101325, v.N: 1})
    names = [str(p) for p in np.squeeze(eq.Phase.values) if p]
    xs = np.atleast_1d(np.squeeze(eq.X.sel(component="O").values))[: len(names)]
    assert ours == sorted((n, round(float(c), 3)) for n, c in zip(names, xs))


def test_cu_o_liquid_gap_below_pycalphad(cuo):
    """x(O) = 0.2, 1500 K: two liquids, 46 J/mol below the Cu2O + liquid that pycalphad 0.11.2 reports.

    Both codes give the same G for the ionic liquid (test above), so the lower
    state is the better minimum of the same model. Cu2O is above our tangent.
    """
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    models = cuo.phase_models(("CU", "O"))
    T = 1500.0
    res = equilibrium(list(models.values()), x=0.8, T=T, certify="none")
    assert sorted(cs.phase.name for cs in res.composition_sets) == ["IONIC_LIQ", "IONIC_LIQ"]
    eq = pyeq(Database(str(pycalphad_db("cuo.tdb"))), ["CU", "O", "VA"], sorted(models), {v.X("O"): 0.2, v.T: T, v.P: 101325, v.N: 1})
    assert sorted(str(p) for p in np.squeeze(eq.Phase.values) if p) == ["CU2O", "IONIC_LIQ"]
    ours_j = res.G * R_GAS * T
    theirs_j = float(eq.GM.squeeze())
    assert ours_j < theirs_j - 30.0
    cu2o = models["CU2O"]
    assert float(cu2o.gibbs([], NUM, T)) - res.mu @ np.array(cu2o.amounts([])) > 1e-3


@pytest.mark.parametrize("a,t", [(0.05, 2000.0), (0.1, 1500.0), (0.5, 1800.0), (0.3, 1800.0), (0.9, 1800.0), (0.02, 1200.0)])
def test_zro2_nd2o3_matches_pycalphad(a, t):
    """The ZrO2–NdO1.5 section of Fabrichnaya and Seifert's assessment: charged CEF phases with neutrality."""
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    path = pycalphad_db("al2o3_nd2o3_zro2.tdb")
    els = ("ND", "ZR", "O")
    models = ThermoDatabase.from_tdb(path).phase_models(els)
    b = np.array([a, 1 - a, 1.5 * a + 2 * (1 - a)])
    res = equilibrium(list(models.values()), b=b, T=t, certify="none")
    ours = sorted(
        (cs.phase.name, round(float(cs.x[0] / (cs.x[0] + cs.x[1])), 3)) for cs in res.composition_sets if cs.amount > 1e-8
    )
    x = b / b.sum()
    eq = pyeq(Database(str(path)), [*els, "VA"], sorted(models), {v.X("ND"): x[0], v.X("ZR"): x[1], v.T: t, v.P: 101325, v.N: 1})
    names = [str(p) for p in np.squeeze(eq.Phase.values) if p]
    X = np.atleast_2d(np.squeeze(eq.X.values))[: len(names)]  # components sorted: ND, O, ZR
    assert ours == sorted((n, round(float(c[0] / (c[0] + c[2])), 3)) for n, c in zip(names, X))
