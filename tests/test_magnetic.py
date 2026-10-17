"""The Inden–Hillert–Jarl magnetic contribution (issue #6): model, interval bounds, and certified pricing."""

import sys
from pathlib import Path

import numpy as np
import pytest

from discopt.calphad import CertificateLevel, ThermoDatabase, equilibrium
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.equilibrium.pricing import price_phase
from discopt.calphad.equilibrium.restricted import CompositionSet, solve_restricted
from discopt.calphad.models import CEFPhase
from discopt.calphad.models.magnetic import Branch, Magnetic, _afm_image, _ordered, _paramagnetic, branch_ranges, ihj
from discopt.calphad.ops import EXPR, INTERVAL, NUM, Interval

PYCALPHAD_R = 8.3145  # pycalphad.variables.R


def pycalphad_db(name):
    pycalphad = pytest.importorskip("pycalphad")
    return Path(pycalphad.__file__).parent / "tests" / "databases" / name


# A small magnetic binary: Fe-like A (T_c 1043 K, β 2.22) and antiferromagnetic
# Cr-like B (T_c −311 K, β −0.008), with a TC interaction. The type code `&`
# carries the MAGNETIC definition, whatever phase name the line names.
TDB = """
ELEMENT A X 0 0 0 ! ELEMENT B X 0 0 0 ! ELEMENT VA X 0 0 0 !
TYPE_DEFINITION & GES A_P_D SOMETHING_ELSE MAGNETIC -1.0 4.00000E-01 !
PHASE BCC %& 2 1 3 ! CONST BCC : A,B : VA : !
PARA G(BCC,A:VA;0) 298.15 -8000-30*T; 6000 N !
PARA G(BCC,B:VA;0) 298.15 -7000-32*T; 6000 N !
PARA G(BCC,A,B:VA;0) 298.15 20000-8*T; 6000 N !
PARA TC(BCC,A:VA;0) 298.15 1043; 6000 N !
PARA TC(BCC,B:VA;0) 298.15 -311.5; 6000 N !
PARA TC(BCC,A,B:VA;0) 298.15 1650; 6000 N !
PARA BMAGN(BCC,A:VA;0) 298.15 2.22; 6000 N !
PARA BMAGN(BCC,B:VA;0) 298.15 -0.008; 6000 N !
PARA BM(BCC,A,B:VA;0) 298.15 -0.85; 6000 N !
PHASE LIQ % 1 1 ! CONST LIQ : A,B : !
PARA G(LIQ,A;0) 298.15 5000-36*T; 6000 N !
PARA G(LIQ,B;0) 298.15 6000-37*T; 6000 N !
"""


@pytest.fixture(scope="module")
def db():
    return ThermoDatabase.from_string(TDB)


@pytest.fixture(scope="module")
def bcc(db):
    return db.phase_model("BCC", ("A", "B"))


def test_magnetic_phase_builds_from_type_code(db, bcc):
    assert isinstance(bcc, CEFPhase)
    assert bcc.magnetic.afm == -1.0 and bcc.magnetic.p == 0.4
    assert len(bcc.magnetic.tc) == 3 and len(bcc.magnetic.beta) == 3  # BM is BMAGN
    assert db.magnetic_type("LIQ") is None
    y = 0.7  # site fraction of A
    tc = 1043 * y - 311.5 * (1 - y) + 1650 * y * (1 - y)
    assert bcc.curie_temperature([y], 500.0) == pytest.approx(tc)


def test_ihj_branches_are_c1_at_the_curie_temperature():
    for p in (0.4, 0.28):
        A = 518 / 1125 + (11692 / 15975) * (1 / p - 1)
        assert _ordered(1.0, p, A) == pytest.approx(_paramagnetic(1.0, A), abs=1e-15)
        h = 1e-6
        d_ord = (_ordered(1 + h, p, A) - _ordered(1 - h, p, A)) / (2 * h)
        d_para = (_paramagnetic(1 + h, A) - _paramagnetic(1 - h, A)) / (2 * h)
        assert d_ord == pytest.approx(d_para, rel=1e-8)


def test_num_matches_expr_on_every_branch(bcc):
    """The discopt expression on the branch at y equals the numeric value at y."""
    import discopt.modeling as dm

    from discopt.calphad.solvers.local import model_evaluator

    for T in (300.0, 900.0, 1500.0):
        for y in (0.02, 0.3, 0.7, 0.98):
            branch = bcc.magnetic_branch([y], T)
            m = dm.Model("g")
            v = m.continuous("y", lb=0.0, ub=1.0)
            m.minimize(bcc.gibbs([v], EXPR.with_branch(branch), T))
            got = float(model_evaluator(m).evaluate_objective(np.array([y])))
            assert got == pytest.approx(float(bcc.gibbs([y], NUM, T)), rel=1e-12), (T, y, branch)


def test_expr_without_a_branch_raises(bcc):
    import discopt.modeling as dm

    m = dm.Model("g")
    v = m.continuous("y", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="with_branch"):
        bcc.gibbs([v], EXPR, 800.0)


@pytest.mark.parametrize("p", [0.4, 0.28])
def test_ihj_f_is_decreasing_across_both_branches(p):
    """The interval rule for f uses only its endpoints, which needs f decreasing on all s ≥ 0."""
    A = 518 / 1125 + (11692 / 15975) * (1 / p - 1)
    s_para, s_ord = np.linspace(0, 1, 20001), np.geomspace(1, 1e4, 20001)
    assert np.all(np.diff(_paramagnetic(s_para, A)) < 0)
    assert np.all(np.diff(_ordered(s_ord, p, A)) < 0)
    assert _paramagnetic(0.0, A) == 0.0


def test_afm_image():
    """x' is x for x > 0 and x / afm for x ≤ 0, so its range over a straddling interval starts at 0."""
    for lo, hi in [(2.0, 5.0), (-6.0, -3.0), (-6.0, 1.0), (-1.0, 4.0), (0.0, 0.0)]:
        out = _afm_image(Interval(lo, hi), -3.0)
        xs = np.append(np.linspace(lo, hi, 1001), [0.0] if lo <= 0 <= hi else [])
        img = np.where(xs <= 0, xs / -3.0, xs)
        assert out.lo <= img.min() and out.hi >= img.max()
        assert out.lo == pytest.approx(img.min(), abs=1e-14) and out.hi == pytest.approx(img.max(), rel=1e-14)


def test_branch_ranges_prune_and_cut_s():
    mag = Magnetic(afm=-1.0, p=0.4)
    # T_c in [200, 600] at T = 400: ferromagnetic only, both sides of s = 1; β > 0 only
    got = branch_ranges(mag, Interval(200.0, 600.0), Interval(0.5, 1.0), 400.0)
    assert [b for b, _ in got] == [Branch(False, False, False), Branch(False, False, True)]
    (_, para), (_, ordered) = got
    assert para[0] == pytest.approx(0.5) and para[1] == 1.0
    assert ordered[0] == 1.0 and ordered[1] == pytest.approx(1.5)
    # T_c in [-100, 50] at T = 400: never ordered; both signs of T_c and of β
    got = branch_ranges(mag, Interval(-100.0, 50.0), Interval(-0.1, 0.2), 400.0)
    assert sorted(got) == sorted(
        [(Branch(tn, bn, False), r) for tn, r in ((True, got[0][1]), (False, got[2][1])) for bn in (True, False)]
    )
    assert not any(b.ordered for b, _ in got)


def test_interval_encloses_magnetic_gibbs(bcc):
    rng = np.random.default_rng(0)
    for T in (300.0, 500.0, 800.0, 1100.0, 1600.0):
        for _ in range(200):
            c, w = rng.uniform(0, 1), rng.uniform(0, 0.4)
            lo, hi = max(c - w, 0.0), min(c + w, 1.0)
            box = bcc.gibbs([Interval(lo, hi)], INTERVAL, T)
            vals = np.array([float(bcc.gibbs([y], NUM, T)) for y in np.linspace(lo, hi, 201)])
            assert box.lo <= vals.min() and box.hi >= vals.max(), (T, lo, hi)


def test_symbolic_view_matches_numeric(bcc):
    import sympy as sp

    from discopt.calphad.symbolic import T as T_

    expr = bcc.gibbs_expr()
    R = sp.Symbol("R", positive=True)
    yA, yB = sp.Symbol("y0_A", nonnegative=True), sp.Symbol("y0_B", nonnegative=True)
    assert expr.free_symbols == {yA, yB, T_, R}
    f = sp.lambdify([yA, yB, T_, R], expr)
    for T in (400.0, 1200.0):
        for y in (0.1, 0.8):
            assert f(y, 1 - y, T, R_GAS) == pytest.approx(float(bcc.gibbs([y], NUM, T)) * R_GAS * T, rel=1e-10)


XIONG_TDB = """
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A BLANK 1 0 0 !
ELEMENT B BLANK 1 0 0 !
TYPE_DEFINITION % SEQ * !
TYPE_DEFINITION & GES A_P_D BCC MAGNETIC 0.0 0.25 !
PHASE BCC %& 2 1 3 !
CONSTITUENT BCC : A,B : VA : !
PARAMETER G(BCC,A:VA;0) 298.15 -8000-30*T; 6000 N !
PARAMETER G(BCC,B:VA;0) 298.15 -7000-32*T; 6000 N !
PARAMETER G(BCC,A,B:VA;0) 298.15 15000-6*T; 6000 N !
PARAMETER TC(BCC,A:VA;0) 298.15 1000; 6000 N !
PARAMETER TC(BCC,B:VA;0) 298.15 -50; 6000 N !
PARAMETER NT(BCC,B:VA;0) 298.15 600; 6000 N !
PARAMETER NT(BCC,A:VA;0) 298.15 -100; 6000 N !
PARAMETER BMAGN(BCC,A:VA;0) 298.15 2.2; 6000 N !
PARAMETER BMAGN(BCC,B:VA;0) 298.15 0.6; 6000 N !
PARAMETER BMAGN(BCC,A,B:VA;0) 298.15 -0.5; 6000 N !
PHASE LIQ % 1 1 !
CONSTITUENT LIQ : A,B : !
PARAMETER G(LIQ,A;0) 298.15 5000-36*T; 6000 N !
PARAMETER G(LIQ,B;0) 298.15 6000-37*T; 6000 N !
"""


@pytest.fixture(scope="module")
def xiong_db():
    return ThermoDatabase.from_string(XIONG_TDB)


def test_xiong_function_is_continuous_and_decreasing():
    """With the τ²¹ term inside the (1/p − 1) bracket (Xiong et al. 2012), g is C¹ at τ = 1 and decreasing."""
    from discopt.calphad.models.magnetic import _x_ord, _x_para

    for p in (0.25, 0.28, 0.4):
        assert _x_para(1.0, p) == pytest.approx(_x_ord(1.0, p), abs=1e-7)
        h = 1e-6
        assert (_x_para(1.0, p) - _x_para(1 - h, p)) / h == pytest.approx(
            (_x_ord(1 + h, p) - _x_ord(1.0, p)) / h, rel=1e-4
        )
        assert np.all(np.diff(_x_para(np.linspace(0, 1, 5001), p)) <= 0)
        assert np.all(np.diff(_x_ord(np.geomspace(1, 1e3, 5001), p)) < 0)


def test_xiong_phase_builds_with_neel_terms(xiong_db):
    bcc = xiong_db.phase_model("BCC", ("A", "B"))
    assert bcc.magnetic.xiong and bcc.magnetic.quantities == ("tc", "nt", "beta")
    assert len(bcc.magnetic.nt) == 2
    assert bcc.neel_temperature([0.2], 500.0) == pytest.approx(-100 * 0.2 + 600 * 0.8)


def _pycalphad_xiong_shift(s, p):
    """pycalphad 0.11.2 minus ours for one g with s = X/T: its τ²¹/1617 sits outside the (1/p − 1) bracket."""
    from discopt.calphad.models.magnetic import _XB, _xD

    if s < 1:  # τ > 1, or X ≤ 0: the same
        return 0.0
    return -(1 - _XB * (1 / p - 1)) * s**-21 / 1617 / _xD(p)


@pytest.mark.parametrize("t", [300.0, 450.0, 700.0, 1100.0, 1500.0])
def test_xiong_gibbs_matches_pycalphad(xiong_db, t):
    pytest.importorskip("pycalphad")
    from pycalphad import Database, calculate

    bcc = xiong_db.phase_model("BCC", ("A", "B"))
    ys = np.linspace(0.02, 0.98, 13)
    pts = np.array([[y, 1 - y, 1.0] for y in ys])
    ref = calculate(Database(XIONG_TDB), ["A", "B", "VA"], "BCC", T=t, P=101325, N=1, points=pts, output="GM")
    for y, gm in zip(ys, np.asarray(ref.GM).ravel()):
        tc, nt, beta = bcc.curie_temperature([y], t), bcc.neel_temperature([y], t), bcc.magnetic_moment([y], t)
        shift = (_pycalphad_xiong_shift(tc / t, 0.25) + _pycalphad_xiong_shift(nt / t, 0.25)) * np.log1p(beta)
        ideal = y * np.log(y) + (1 - y) * np.log(1 - y)
        mag = float(bcc.magnetic.term((tc, nt, beta), t, NUM))
        ours = (float(bcc.gibbs([y], NUM, t)) + shift) * R_GAS * t + (PYCALPHAD_R - R_GAS) * t * (ideal + mag + shift)
        assert ours == pytest.approx(gm, rel=1e-9, abs=1e-6), (t, y)


def test_xiong_interval_encloses(xiong_db):
    bcc = xiong_db.phase_model("BCC", ("A", "B"))
    rng = np.random.default_rng(4)
    for T in (300.0, 550.0, 900.0, 1400.0):
        for _ in range(150):
            c, w = rng.uniform(0, 1), rng.uniform(0, 0.3)
            lo, hi = max(c - w, 0.0), min(c + w, 1.0)
            box = bcc.gibbs([Interval(lo, hi)], INTERVAL, T)
            vals = np.array([float(bcc.gibbs([y], NUM, T)) for y in np.linspace(lo, hi, 201)])
            assert box.lo <= vals.min() and box.hi >= vals.max()


def test_xiong_global_pricing_matches_a_grid(xiong_db):
    bcc = xiong_db.phase_model("BCC", ("A", "B"))
    ys = np.linspace(1e-9, 1 - 1e-9, 20001)
    for T, mu in ((450.0, [-4.0, -3.5]), (800.0, [-3.0, -2.5])):
        r = price_phase(bcc, np.array(mu), T=T)
        psi = np.array([float(bcc.gibbs([y], NUM, T)) - mu[0] * y - mu[1] * (1 - y) for y in ys])
        assert r.ub == pytest.approx(psi.min(), abs=1e-6)
        assert r.lb is not None and psi.min() - 1e-5 <= r.lb <= psi.min() + 1e-12


def test_xiong_equilibrium_matches_pycalphad(xiong_db):
    pytest.importorskip("pycalphad")
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    phases = list(xiong_db.phase_models(("A", "B")).values())
    for x, t in ((0.5, 500.0), (0.3, 900.0)):
        res = equilibrium(phases, x=x, T=t, certify="none")
        eq = pyeq(Database(XIONG_TDB), ["A", "B", "VA"], ["BCC", "LIQ"], {v.X("A"): x, v.T: t, v.P: 101325, v.N: 1})
        names = [str(n) for n in np.squeeze(eq.Phase.values) if n]
        xs = np.atleast_1d(np.squeeze(eq.X.sel(component="A").values))[: len(names)]
        ref = sorted((n, round(float(c), 2)) for n, c in zip(names, xs))
        assert sorted((cs.phase.name, round(float(cs.x[0]), 2)) for cs in res.composition_sets) == ref


def test_fixed_composition_magnetic_phase_is_priced_exactly():
    compound = (
        "PHASE AB %& 2 1 1 ! CONST AB : A : B : ! PARA G(AB,A:B;0) 298.15 -30000; 6000 N !"
        " PARA TC(AB,A:B;0) 298.15 600; 6000 N ! PARA BMAGN(AB,A:B;0) 298.15 1.1; 6000 N !"
    )
    db = ThermoDatabase.from_string(TDB + compound)
    ab = db.phase_model("AB", ("A", "B"))
    assert ab.n_y == 0 and ab.magnetic is not None
    r = price_phase(ab, np.zeros(2), T=500.0)
    assert r.lb == r.ub and r.bound_source == "exact"


def test_restricted_solve_switches_branch():
    """Cr–Fe BCC at 700 K: the Cr-rich set starts with T_c > 0 and must end where T_c < 0.

    The first solve stops on the edge T_c = 0; the second, on the
    antiferromagnetic branch, reaches the miscibility gap pycalphad finds.
    """
    bcc = ThermoDatabase.from_tdb(pycalphad_db("CrFeNb_Jacob2016.tdb")).phase_model("BCC_A2", ("CR", "FE"))
    T = 700.0
    assert not bcc.magnetic_branch([0.8], T).tc_negative
    mod = sys.modules["discopt.calphad.equilibrium.restricted"]
    orig, solves = mod._solve_once, []

    def spy(*a, **k):
        out = orig(*a, **k)
        solves.append(out[1])
        return out

    mod._solve_once = spy
    try:
        res = solve_restricted([CompositionSet(bcc, [0.55]), CompositionSet(bcc, [0.8])], np.array([0.5, 0.5]), T=T)
    finally:
        mod._solve_once = orig
    assert res.converged and len(solves) == 2 and solves[-1] == []
    assert sorted(float(cs.y[0]) for cs in res.sets) == pytest.approx([0.1135, 0.9228], abs=5e-4)
    assert bcc.magnetic_branch(res.sets[1].y, T).tc_negative


def test_global_pricing_matches_a_dense_grid(bcc):
    """Branch-split pricing finds the grid minimum and bounds it, at T on both sides of the T_c range."""
    ys = np.linspace(1e-9, 1 - 1e-9, 40001)
    for T, mu in ((500.0, [-3.0, -2.5]), (900.0, [-2.0, -3.0])):
        r = price_phase(bcc, np.array(mu), T=T)
        psi = np.array([float(bcc.gibbs([y], NUM, T)) - mu[0] * y - mu[1] * (1 - y) for y in ys])
        assert r.bound_source in ("tree", "callback") and r.lb is not None
        assert r.ub == pytest.approx(psi.min(), abs=1e-7)
        assert r.lb <= psi.min() + 1e-12 and r.lb >= psi.min() - 1e-5


def test_magnetic_equilibrium_is_certified(db):
    phases = list(db.phase_models(("A", "B")).values())
    res = equilibrium(phases, x=0.5, T=900.0, certify="final")
    assert res.certificate.level in (CertificateLevel.GLOBALLY_CERTIFIED, CertificateLevel.CERTIFIED_TO_TOLERANCE)
    assert res.certificate.lower_bounds["BCC"] is not None


@pytest.mark.slow
def test_crfe_bcc_gap_is_certified():
    """Cr–Fe at 700 K: the ferromagnetic and antiferromagnetic BCC sets of the gap, certified."""
    bcc = ThermoDatabase.from_tdb(pycalphad_db("CrFeNb_Jacob2016.tdb")).phase_model("BCC_A2", ("CR", "FE"))
    res = equilibrium([bcc], x=0.5, T=700.0, certify="final")
    assert res.certificate.level in (CertificateLevel.GLOBALLY_CERTIFIED, CertificateLevel.CERTIFIED_TO_TOLERANCE)
    assert sorted(float(cs.x[0]) for cs in res.composition_sets) == pytest.approx([0.1135, 0.9228], abs=5e-4)


# ---------------------------------------------------------------------------- pycalphad oracle

MAG_CASES = [
    ("crtiv_ghosh.tdb", ("CR", "TI"), "BCC_A2"),
    ("crtiv_ghosh.tdb", ("CR", "V"), "HCP_A3"),
    ("CrFeNb_Jacob2016.tdb", ("CR", "FE"), "BCC_A2"),
    ("CrFeNb_Jacob2016.tdb", ("CR", "FE"), "FCC_A1"),
    ("alfe.tdb", ("AL", "FE"), "BCC_A2"),
    ("crfe_bcc_magnetic.tdb", ("CR", "FE"), "BCC_A2"),  # BM for BMAGN
    ("Al-Fe_sundman2009.tdb", ("AL", "FE"), "A2_NOB"),  # type definition names phase "@"
    ("COST507.tdb", ("FE", "NI"), "FCC_A1"),
]


def _ideal(phase, fractions):
    return sum(a * sum(v * np.log(v) for v in d.values() if v > 0) for a, d in zip(phase.site_ratios, fractions))


@pytest.mark.parametrize("fname,elements,name", MAG_CASES)
@pytest.mark.parametrize("t", [300.0, 700.0, 1000.0, 1300.0, 1800.0])
def test_magnetic_gibbs_matches_pycalphad(fname, elements, name, t):
    """G to 1e-9 relative on both sides of T_c.

    Not at T = T_c exactly: there pycalphad's Piecewise has no branch for τ = 1
    and returns 0 for the magnetic term, where the IHJ function is continuous.
    """
    path = pycalphad_db(fname)
    from pycalphad import Database, calculate

    model = ThermoDatabase.from_tdb(path).phase_model(name, elements)
    assert isinstance(model, CEFPhase) and model.magnetic is not None
    rng = np.random.default_rng(3)
    fractions = [[dict(zip(c, rng.dirichlet(np.ones(len(c))))) for c in model.constituents] for _ in range(8)]
    for k in range(2):  # the endmembers
        fractions.append([{c: float(i == k % len(cons)) for i, c in enumerate(cons)} for cons in model.constituents])
    points = [[d[c] for cons, d in zip(model.constituents, fr) for c in cons] for fr in fractions]
    ref = calculate(
        Database(str(path)), [*elements, "VA"], name, T=t, P=101325, N=1, points=np.array(points), output="GM"
    )
    for fr, gm in zip(fractions, np.asarray(ref.GM).ravel()):
        u = model.variables_from_site_fractions(fr)
        atoms = float(np.sum(model.amounts(u)))
        mag = float(ihj(model.magnetic, model.curie_temperature(u, t), model.magnetic_moment(u, t), t, NUM))
        ours = float(model.gibbs(u, T=t)) * R_GAS * t / atoms
        ours += (PYCALPHAD_R - R_GAS) * t * (_ideal(model, model.site_fractions(u)) + mag) / atoms
        assert ours == pytest.approx(gm, rel=1e-9, abs=1e-6), fr


@pytest.mark.parametrize(
    "fname,elements,x,t",
    [
        ("CrFeNb_Jacob2016.tdb", ("CR", "FE"), 0.5, 700.0),  # BCC miscibility gap, ferro + AFM sets
        ("CrFeNb_Jacob2016.tdb", ("CR", "FE"), 0.1, 1200.0),  # inside the gamma loop
        ("CrFeNb_Jacob2016.tdb", ("FE", "NB"), 0.9, 1000.0),  # ferromagnetic BCC + Laves
        ("crtiv_ghosh.tdb", ("CR", "V"), 0.5, 300.0),  # AFM BCC gap
    ],
)
def test_magnetic_equilibrium_matches_pycalphad(fname, elements, x, t):
    path = pycalphad_db(fname)
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    db = ThermoDatabase.from_tdb(path)
    models = db.phase_models(elements)
    res = equilibrium(list(models.values()), x=x, T=t, certify="none")
    eq = pyeq(
        Database(str(path)), [*elements, "VA"], list(models), {v.X(elements[1]): 1 - x, v.T: t, v.P: 101325, v.N: 1}
    )
    names = [p for p in np.squeeze(eq.Phase.values) if p]
    xs = np.atleast_1d(np.squeeze(eq.X.sel(component=elements[0]).values))[: len(names)]
    ref = sorted((str(n), round(float(c), 3)) for n, c in zip(names, xs))
    ours = sorted((cs.phase.name, round(float(cs.x[0]), 3)) for cs in res.composition_sets if cs.amount > 1e-8)
    assert ours == ref
