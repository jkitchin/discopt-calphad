"""Order–disorder (partitioned) phases, issue #6: G_dis(x) + G_ord(y) − G_ord(x)."""

from pathlib import Path

import numpy as np
import pytest

from discopt.calphad import ThermoDatabase, UnsupportedModelError, equilibrium
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.equilibrium.loop import is_duplicate
from discopt.calphad.equilibrium.restricted import CompositionSet
from discopt.calphad.models.magnetic import ihj
from discopt.calphad.models.partitioned import PartitionedPhase
from discopt.calphad.ops import INTERVAL, NUM, Interval

PYCALPHAD_R = 8.3145


def pycalphad_db(name):
    pycalphad = pytest.importorskip("pycalphad")
    return Path(pycalphad.__file__).parent / "tests" / "databases" / name


# A small L1₂-type phase: an ordered part on (0.75, 0.25) with its own ordering energies, over a
# disordered A1 parent with a magnetic A.
TDB = """
ELEMENT A X 0 0 0 ! ELEMENT B X 0 0 0 ! ELEMENT VA X 0 0 0 !
TYPE_DEFINITION % SEQ * !
TYPE_DEFINITION ) GES A_P_D A1 MAGNETIC -3.0 0.28 !
PHASE A1 %) 2 1 1 ! CONST A1 : A,B : VA : !
PARA G(A1,A:VA;0) 298.15 -1000-30*T; 6000 N !
PARA G(A1,B:VA;0) 298.15 -2000-31*T; 6000 N !
PARA G(A1,A,B:VA;0) 298.15 -40000+5*T; 6000 N !
PARA G(A1,A,B:VA;1) 298.15 6000; 6000 N !
PARA TC(A1,A:VA;0) 298.15 600; 6000 N !
PARA BMAGN(A1,A:VA;0) 298.15 0.6; 6000 N !
TYPE_DEFINITION * GES A_P_D L12 DIS_PART A1 ,,, !
PHASE L12 %* 3 0.75 0.25 1 ! CONST L12 : A,B : A,B : VA : !
PARA G(L12,A:B:VA;0) 298.15 -6000+T; 6000 N !
PARA G(L12,B:A:VA;0) 298.15 -2000; 6000 N !
PARA G(L12,A,B:A:VA;0) 298.15 1500; 6000 N !
PARA G(L12,A:A,B:VA;0) 298.15 900; 6000 N !
"""


@pytest.fixture(scope="module")
def db():
    return ThermoDatabase.from_string(TDB)


@pytest.fixture(scope="module")
def l12(db):
    return db.phase_model("L12", ("A", "B"))


def _parent_g(phase, x, T):
    """G/RT per ordered formula unit of the disordered part at composition x(A)."""
    d = phase.disordered
    u = d.variables_from_site_fractions([{"A": x, "B": 1 - x}, {"VA": 1.0}])
    return float(d.gibbs(u, NUM, T)) * phase.scale


def test_builds_a_partitioned_phase(db, l12):
    assert isinstance(l12, PartitionedPhase)
    assert l12.substitutional == (0, 1) and l12.interstitial == (2,)
    assert l12.scale == pytest.approx(1.0)
    assert l12.magnetic is l12.disordered.magnetic and l12.magnetic.afm == -3.0
    assert db.disordered_part("L12") == ("A1", False)
    assert db.disordered_part("A1") is None


def test_disordered_state_is_the_parent(l12):
    """With equal fractions on both substitutional sublattices, G is the parent's G exactly."""
    for T in (400.0, 900.0, 1500.0):
        for x in (0.05, 0.3, 0.7):
            u = l12.variables_from_site_fractions([{"A": x, "B": 1 - x}] * 2 + [{"VA": 1.0}])
            assert float(l12.gibbs(u, NUM, T)) == pytest.approx(_parent_g(l12, x, T), rel=1e-13, abs=1e-13)
            assert l12.ordering_energy(u, T) == pytest.approx(0.0, abs=1e-9)
            assert l12.degree_of_order(u) == pytest.approx(0.0, abs=1e-15)


def test_ordered_state_matches_the_formula(l12):
    """At an ordered point, G = λ G_dis(x) + G_ord(y) − G_ord(x), entropy from the ordered sublattices."""
    T = 800.0
    y1, y2 = 0.95, 0.15  # A on the 0.75 and 0.25 sublattices
    u = l12.variables_from_site_fractions([{"A": y1, "B": 1 - y1}, {"A": y2, "B": 1 - y2}, {"VA": 1.0}])
    x = 0.75 * y1 + 0.25 * y2
    RT = R_GAS * T

    def g_ord(a, b):  # the ordered part's energy terms, J per formula unit
        e = (-6000 + T) * a * (1 - b) + (-2000) * (1 - a) * b
        return e + 1500 * a * (1 - a) * b + 900 * a * b * (1 - b)

    def xlx(v):
        return v * np.log(v)

    s_ord = 0.75 * (xlx(y1) + xlx(1 - y1)) + 0.25 * (xlx(y2) + xlx(1 - y2))
    d = l12.disordered
    ud = d.variables_from_site_fractions([{"A": x, "B": 1 - x}, {"VA": 1.0}])
    g_dis_no_ideal = float(d.gibbs(ud, NUM, T)) - (xlx(x) + xlx(1 - x))
    expected = g_dis_no_ideal + (g_ord(y1, y2) - g_ord(x, x)) / RT + s_ord
    assert float(l12.gibbs(u, NUM, T)) == pytest.approx(expected, rel=1e-12)


def test_interval_encloses_partitioned_gibbs(l12):
    rng = np.random.default_rng(1)
    for T in (500.0, 1200.0):
        for _ in range(100):
            c, w = rng.uniform(0, 1, 2), rng.uniform(0, 0.2, 2)
            lo, hi = np.clip(c - w, 0, 1), np.clip(c + w, 0, 1)
            box = l12.gibbs([Interval(a, b) for a, b in zip(lo, hi)], INTERVAL, T)
            pts = lo + rng.uniform(0, 1, (200, 2)) * (hi - lo)
            vals = np.array([float(l12.gibbs(p, NUM, T)) for p in pts])
            assert box.lo <= vals.min() and box.hi >= vals.max()


def test_b2_symmetry_and_duplicates():
    """B2: the two 0.5 sublattices are equivalent, so the variants (y1, y2) and (y2, y1) are one state."""
    b2 = ThermoDatabase.from_tdb(pycalphad_db("alfe.tdb")).phase_model("B2_BCC", ("AL", "FE"))
    assert isinstance(b2, PartitionedPhase) and len(b2.symmetry) == 2
    y = np.array([0.8, 0.2])
    (_, swapped) = b2.variants(y)
    assert swapped == pytest.approx([0.2, 0.8])
    for T in (600.0, 1200.0):
        assert float(b2.gibbs(y, NUM, T)) == pytest.approx(float(b2.gibbs(swapped, NUM, T)), rel=1e-12)
    assert is_duplicate([CompositionSet(b2, y)], b2, swapped, 1e-6)
    # the symmetry constraint keeps one variant: the first constituent descends along the class
    (g,) = b2.symmetry_constraints(y)
    assert g == pytest.approx(0.6)


def test_fcc_4sl_permutation_option():
    """CoV's FCC_4SL uses the F option: each parameter stands for all its permutations (24 symmetries)."""
    fcc = ThermoDatabase.from_tdb(pycalphad_db("CoV-20Wan.tdb")).phase_model("FCC_4SL", ("CO", "V"))
    assert len(fcc.symmetry) == 24
    assert len(fcc.symmetry_constraints(np.zeros(4))) == 3


def test_unsupported_variants_say_why():
    alfe = ThermoDatabase.from_tdb(pycalphad_db("Al-Fe_sundman2009.tdb"))
    with pytest.raises(UnsupportedModelError, match="magnetic parameters on the ordered part"):
        alfe.phase_model("BCC_4SL", ("AL", "FE"))


# ---------------------------------------------------------------------------- pycalphad oracle

CASES = [
    ("alni_dupin_2001.tdb", ("AL", "NI"), "FCC_L12"),
    ("alcrni.tdb", ("AL", "NI"), "L12_FCC"),
    ("alfe.tdb", ("AL", "FE"), "B2_BCC"),
    ("COST507.tdb", ("AL", "NI"), "BCC_B2"),
    ("COST507.tdb", ("FE", "NI"), "BCC_B2"),
    ("CoV-20Wan.tdb", ("CO", "V"), "FCC_4SL"),  # 4 sublattices, F option
    ("CoV-20Wan.tdb", ("CO", "V"), "SIGMA_D8B"),  # NEVER_DISORDER
    ("alni_dupin_2001.tdb", ("AL", "NI"), "BCC_B2"),  # vacancies on both substitutional sublattices
    ("alcrni.tdb", ("AL", "NI"), "B2"),  # vacancies, and no interstitial sublattice
]


@pytest.mark.parametrize("fname,elements,name", CASES)
@pytest.mark.parametrize("t", [500.0, 1000.0, 1500.0])
def test_partitioned_gibbs_matches_pycalphad(fname, elements, name, t):
    path = pycalphad_db(fname)
    from pycalphad import Database, calculate

    model = ThermoDatabase.from_tdb(path).phase_model(name, elements)
    assert isinstance(model, PartitionedPhase)
    rng = np.random.default_rng(7)
    fractions = [[dict(zip(c, rng.dirichlet(np.ones(len(c))))) for c in model.constituents] for _ in range(8)]
    # a disordered point: the same fractions on every substitutional sublattice
    cons0 = model.constituents[model.substitutional[0]]
    x = dict(zip(cons0, rng.dirichlet(np.ones(len(cons0)))))
    fractions.append([dict(x) if s in model.substitutional else {c: 1.0 for c in cons} for s, cons in enumerate(model.constituents)])
    points = [[d[c] for cons, d in zip(model.constituents, fr) for c in cons] for fr in fractions]
    ref = calculate(Database(str(path)), [*elements, "VA"], name, T=t, P=101325, N=1, points=np.array(points), output="GM")
    for fr, gm in zip(fractions, np.asarray(ref.GM).ravel()):
        u = model.variables_from_site_fractions(fr)
        atoms = float(np.sum(model.amounts(u)))
        ours = float(model.gibbs(u, T=t)) * R_GAS * t / atoms
        # pycalphad's R is 8.3145: rescale the terms that carry R (entropy and magnetism)
        ideal = float(model.ordered.ideal_at(model.site_fractions(u), NUM))
        mag = 0.0
        if model.magnetic is not None:
            tc, beta = model.curie_temperature(u, t), model.magnetic_moment(u, t)
            mag = model.scale * float(ihj(model.magnetic, tc, beta, t, NUM))
        ours += (PYCALPHAD_R - R_GAS) * t * (ideal + mag) / atoms
        assert ours == pytest.approx(gm, rel=1e-9, abs=1e-6), fr


def _label(cs):
    """Phase name, with a disordered set of a partitioned phase named after its parent."""
    ph = cs.phase
    if isinstance(ph, PartitionedPhase) and ph.degree_of_order(cs.y) < 1e-3:
        return ph.disordered.name
    return ph.name


@pytest.mark.parametrize("x,t", [(0.22, 1200.0), (0.15, 1200.0), (0.25, 1000.0), (0.10, 900.0)])
def test_alni_gamma_gamma_prime_matches_pycalphad(x, t):
    """Dupin's Al–Ni: γ (FCC_A1) and γ′ (FCC_L12) with the native partitioned model."""
    path = pycalphad_db("alni_dupin_2001.tdb")
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    models = ThermoDatabase.from_tdb(path).phase_models(("AL", "NI"))
    assert "FCC_L12" in models
    res = equilibrium(list(models.values()), x=x, T=t, certify="none")
    ours = sorted((_label(cs), round(float(cs.x[0]), 3)) for cs in res.composition_sets if cs.amount > 1e-8)
    eq = pyeq(Database(str(path)), ["AL", "NI", "VA"], sorted(models), {v.X("AL"): x, v.T: t, v.P: 101325, v.N: 1})
    names = [str(p) for p in np.squeeze(eq.Phase.values) if p]
    xs = np.atleast_1d(np.squeeze(eq.X.sel(component="AL").values))[: len(names)]
    ys = np.squeeze(eq.Y.values)[: len(names)]
    ref = []
    for n, c, y in zip(names, xs, np.atleast_2d(ys)):
        if n == "FCC_L12" and abs(y[0] - y[2]) < 1e-3:  # pycalphad's disordered L12 is γ
            n = "FCC_A1"
        ref.append((n, round(float(c), 3)))
    assert ours == sorted(ref)


def test_disordered_set_merges_into_its_parent():
    """A disordered FCC_L12 set next to an FCC_A1 set at the same composition is one state."""
    from discopt.calphad.equilibrium.loop import _merge_coincident

    models = ThermoDatabase.from_tdb(pycalphad_db("alni_dupin_2001.tdb")).phase_models(("AL", "NI"))
    a1, l12 = models["FCC_A1"], models["FCC_L12"]
    for sets in (
        [CompositionSet(l12, np.array([0.1, 0.1]), 0.2), CompositionSet(a1, np.array([0.1]), 0.5)],
        [CompositionSet(a1, np.array([0.1]), 0.5), CompositionSet(l12, np.array([0.1, 0.1]), 0.2)],
    ):
        (merged,) = _merge_coincident(sets, 1e-4)
        assert merged.phase is a1 and merged.amount == pytest.approx(0.7)  # one atom per formula unit in both
    ordered = [CompositionSet(a1, np.array([0.1]), 0.5), CompositionSet(l12, np.array([0.0, 0.4]), 0.2)]
    assert len(_merge_coincident(ordered, 1e-4)) == 2


@pytest.mark.slow
def test_alni_gamma_gamma_prime_is_certified():
    """γ + γ′ at 1200 K, certified with global pricing of the partitioned FCC_L12 (about 9 minutes)."""
    from discopt.calphad import CertificateLevel

    models = ThermoDatabase.from_tdb(pycalphad_db("alni_dupin_2001.tdb")).phase_models(("AL", "NI"))
    res = equilibrium(list(models.values()), x=0.22, T=1200.0, certify="none")
    cert = res.certify()
    assert cert.level in (CertificateLevel.GLOBALLY_CERTIFIED, CertificateLevel.CERTIFIED_TO_TOLERANCE)
    assert sorted((_label(cs), round(float(cs.x[0]), 3)) for cs in res.composition_sets) == [
        ("FCC_A1", 0.147),
        ("FCC_L12", 0.23),
    ]


# ---------------------------------------------------------------------------- vacancies on every sublattice

VAC_TDB = """
ELEMENT A X 0 0 0 ! ELEMENT B X 0 0 0 ! ELEMENT VA X 0 0 0 !
PHASE V % 2 1 3 ! CONST V : A,B,VA : VA : !
PARA G(V,A:VA;0) 298.15 -10000; 6000 N !
PARA G(V,B:VA;0) 298.15 -12000; 6000 N !
PARA G(V,VA:VA;0) 298.15 0; 6000 N !
PARA G(V,A,VA:VA;0) 298.15 20000; 6000 N !
PARA G(V,B,VA:VA;0) 298.15 25000; 6000 N !
PARA G(V,A,B:VA;0) 298.15 -3000; 6000 N !
"""


def test_vacancy_limited_phase_domain():
    """With VA on every sublattice, atoms_min is min_occupancy · atoms_max, and the domain says so."""
    v = ThermoDatabase.from_string(VAC_TDB).phase_model("V", ("A", "B"))
    assert v.vacancy_limited and v.atoms_max == 1.0 and v.atoms_min == 0.5
    u_full = v.variables_from_site_fractions([{"A": 0.5, "B": 0.5, "VA": 0.0}, {"VA": 1.0}])
    u_empty = v.variables_from_site_fractions([{"A": 0.2, "B": 0.1, "VA": 0.7}, {"VA": 1.0}])
    (g_full,) = v.domain_constraints(u_full)
    (g_empty,) = v.domain_constraints(u_empty)
    assert g_full == pytest.approx(0.5) and g_empty == pytest.approx(-0.2)
    db = ThermoDatabase.from_string(VAC_TDB)
    db.min_occupancy = 0.9
    assert db.phase_model("V", ("A", "B")).atoms_min == pytest.approx(0.9)


def test_vacancy_limited_pricing_covers_the_domain_only():
    """Without the domain, a dilute 'vacuum' state makes Ψ per atom unbounded; within it, pricing matches a grid."""
    from discopt.calphad.equilibrium.pricing import price_phase

    v = ThermoDatabase.from_string(VAC_TDB).phase_model("V", ("A", "B"))
    T, mu = 1000.0, np.array([-1.5, -1.6])
    rng = np.random.default_rng(3)
    pts = rng.uniform(0, 1, size=(200000, 2))
    psi = np.array([float(v.gibbs(p, NUM, T)) - mu @ np.array(v.amounts(p)) for p in pts[:20000]])
    inside = np.array([float(v.domain_constraints(p)[0]) >= 0 for p in pts[:20000]])
    r = price_phase(v, mu, T=T)
    assert r.lb is not None and r.lb <= psi[inside].min() + 1e-9
    assert r.ub == pytest.approx(psi[inside].min(), abs=2e-3)
    assert float(v.domain_constraints(r.y)[0]) >= -1e-9
    # the dilute corner, outside the domain, is far below: per atom it is unbounded
    dilute = v.variables_from_site_fractions([{"A": 1e-8, "B": 0.0, "VA": 1 - 1e-8}, {"VA": 1.0}])
    per_atom = (float(v.gibbs(dilute, NUM, T)) - mu @ np.array(v.amounts(dilute))) / 1e-8
    assert per_atom < -10


@pytest.mark.parametrize("x,t", [(0.5, 1200.0), (0.3, 1200.0), (0.58, 1200.0), (0.55, 1000.0)])
def test_alni_beta_with_vacancies_matches_pycalphad(x, t):
    """Dupin's Al–Ni with every phase, including B2 with constitutional vacancies on both sublattices."""
    path = pycalphad_db("alni_dupin_2001.tdb")
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    models = ThermoDatabase.from_tdb(path).phase_models(("AL", "NI"))
    assert {"BCC_A2", "BCC_B2"} <= set(models) and models["BCC_B2"].vacancy_limited
    res = equilibrium(list(models.values()), x=x, T=t, certify="none")
    assert any("min_occupancy" in m for m in res.certificate.messages)
    ours = sorted((_label(cs), round(float(cs.x[0]), 3)) for cs in res.composition_sets if cs.amount > 1e-8)
    eq = pyeq(Database(str(path)), ["AL", "NI", "VA"], sorted(models), {v.X("AL"): x, v.T: t, v.P: 101325, v.N: 1})
    names = [str(p) for p in np.squeeze(eq.Phase.values) if p]
    xs = np.atleast_1d(np.squeeze(eq.X.sel(component="AL").values))[: len(names)]
    ys = np.atleast_2d(np.squeeze(eq.Y.values))[: len(names)]
    ref = []
    for n, c, y in zip(names, xs, ys):
        if n == "BCC_B2" and np.max(np.abs(y[0:3] - y[3:6])) < 1e-3:
            n = "BCC_A2"
        if n == "FCC_L12" and abs(y[0] - y[2]) < 1e-3:
            n = "FCC_A1"
        ref.append((n, round(float(c), 3)))
    assert ours == sorted(ref)
