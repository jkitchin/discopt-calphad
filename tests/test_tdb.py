"""MVP 3: TDB import/export, symbolic functions, database phase models (pycalphad as test-only oracle)."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from discopt.calphad import BinarySolution, StoichiometricPhase
from discopt.calphad.database import PhaseNotApplicable, ThermoDatabase, UnsupportedModelError
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models import CEFPhase
from discopt.calphad.symbolic import (
    PiecewiseT,
    T,
    TDBSyntaxError,
    function_ref,
    parse_expression,
    parse_piecewise,
    tdb_str,
)

DATA = Path(__file__).parent / "data"
SYNTH = DATA / "ab_synthetic.tdb"


def pycalphad_db(name):
    pycalphad = pytest.importorskip("pycalphad")
    return Path(pycalphad.__file__).parent / "tests" / "databases" / name


# ---------------------------------------------------------------------------- expressions


def test_parse_expression_structure():
    e = parse_expression("-7976.15+137.07*T-24.37*T*LN(T)-1.88E-3*T**2+74092*T**(-1)+GHSERAL#")
    ref = -7976.15 + 137.07 * T - 24.37 * T * sp.log(T) - 1.88e-3 * T**2 + 74092 / T + function_ref("GHSERAL")
    assert sp.simplify(e - ref) == 0
    assert parse_expression("-T**2") == -(T**2)  # ** binds tighter than unary minus
    assert parse_expression("2**-1") == sp.Rational(1, 2)
    assert parse_expression("1.5D+03") == 1500.0


@pytest.mark.parametrize("bad", ["", "1+", "(1+T", "T $ 2", "LN(T"])
def test_parse_expression_errors(bad):
    with pytest.raises(TDBSyntaxError):
        parse_expression(bad)


def test_piecewise_ranges_and_errors():
    pw = parse_piecewise(" 298.15 +T;  900 Y  2*T; 3000 N REF ")
    assert pw.breaks == (298.15, 900.0, 3000.0) and pw.reference == "REF"
    assert pw.select(899.9) == T and pw.select(900.0) == 2 * T  # [lo, hi), as in pycalphad
    assert pw.select(3000.0) == 2 * T
    with pytest.raises(ValueError, match="outside"):
        pw.select(3500.0)
    assert pw.select(3500.0, extrapolate=True) == 2 * T
    assert parse_piecewise("298.15 1; ,, N").t_max == np.inf
    assert parse_piecewise("298.15 1;,,N 01DUP").reference == "01DUP"
    # tolerated on the last range only: missing N, missing upper T, missing lower T
    assert parse_piecewise("298.15 0.0; 6000.00   01DUP") == PiecewiseT((298.15, 6000.0), (sp.Float(0.0),), "01DUP")
    assert parse_piecewise(" +GALLIQ#;  N ") == PiecewiseT((0.0, np.inf), (function_ref("GALLIQ"),))
    assert parse_piecewise("298.15 -8000+T; 900 N").exprs[0] == -8000 + T
    for bad in ["298.15 T", "298.15 T; 900 Y", "298.15 T; 900 N 2*T; 1000 N", "298.15 T; 900 2*T; 1000 N"]:
        with pytest.raises(TDBSyntaxError):
            parse_piecewise(bad)


def test_tdb_str_round_trip():
    e = parse_expression("-11276.24+223.048446*T-38.5844296*T*LN(T)+.018531982*T**2-5.764227E-06*T**3+74092*T**(-1)")
    e2 = e + 3 * function_ref("GHSERAL") - sp.exp(sp.Float(0.01) * T)
    assert parse_expression(tdb_str(e2)) == e2
    pw = parse_piecewise("298.15 GHSERA#+1; 900 Y 2*T; 3000 N R1")
    assert parse_piecewise(pw.to_tdb()) == pw


# ---------------------------------------------------------------------------- database


@pytest.fixture(scope="module")
def synth():
    return ThermoDatabase.from_tdb(SYNTH)


def test_synthetic_contents(synth):
    assert {"A", "B", "C", "VA", "/-"} <= set(synth.elements)
    assert synth.elements["A"].mass == 10.0
    assert synth.species["AB2"].formula == "A1B2"
    assert set(synth.functions) == {"GHSERA", "GHSERB", "GALIQ", "GBLIQ"}
    liq = synth.phases["LIQUID"]
    assert liq.hints == ("L",) and liq.constituents == (("A", "B"),)
    assert synth.phases["FCC_A1"].constituents == (("A", "B"), ("VA",))  # '%' marks stripped
    assert synth.phases["AB2"].site_ratios == (1.0, 2.0)
    labels = {p.label for p in synth.parameters}
    assert "G(LIQUID,A,B;0)" in labels  # L( is stored as G(
    assert "G(FCC_A1,A,B:*;1)" in labels
    assert len(synth.amendments("FCC_A1")) == 1 and "DIS_PART" in synth.amendments("ORD_PH")[0]


def test_function_evaluation_and_expansion(synth):
    t = 1000.0  # second range of GHSERA
    ghsera = -11000 + 220 * t - 38 * t * np.log(t) + 1.8e-2 * t**2 - 5.7e-6 * t**3
    assert synth.evaluate("GHSERA", t) == pytest.approx(ghsera, rel=1e-14)
    assert synth.evaluate("GALIQ", t) == pytest.approx(10000 - 10 * t + ghsera, rel=1e-14)
    assert synth.evaluate("GALIQ", 500.0) == pytest.approx(float(synth.expand("GALIQ").subs(T, 500.0)), rel=1e-12)
    assert synth.functions["GBLIQ"].exprs[0].has(function_ref("GHSERB"))  # reference without '#'
    with pytest.raises(ValueError, match="outside"):
        synth.evaluate("GHSERA", 200.0)
    assert np.isfinite(synth.evaluate("GHSERA", 200.0, extrapolate=True))


def test_pressure_dependent_function():
    db = ThermoDatabase.from_string("FUNCTION DP 298.15 P-101325; 6000 N ! FUNCTION G2 298.15 1+DP#*T; 6000 N !")
    assert db.evaluate("G2", 500.0) == 1.0
    db.pressure = 201325.0
    assert db.evaluate("G2", 500.0) == 1.0 + 1e5 * 500.0


def test_undefined_and_recursive_functions():
    db = ThermoDatabase.from_string("FUNCTION F1 298.15 F2#+1; 3000 N ! FUNCTION F2 298.15 F1#; 3000 N !")
    with pytest.raises(ValueError, match="recursive"):
        db.evaluate("F1", 500.0)
    db = ThermoDatabase.from_string("FUNCTION F1 298.15 NOPE#+1; 3000 N !")
    with pytest.raises(KeyError, match="NOPE"):
        db.evaluate("F1", 500.0)


def test_unknown_and_ambiguous_commands():
    with pytest.raises(TDBSyntaxError, match="unsupported"):
        ThermoDatabase.from_string("ASSESSED_SPECIES_BOGUS X !")
    with pytest.raises(TDBSyntaxError, match="ambiguous"):
        ThermoDatabase.from_string("DEF X !")
    with pytest.raises(TDBSyntaxError, match="undefined PHASE"):
        ThermoDatabase.from_string("CONSTITUENT FOO :A: !")
    with pytest.raises(TDBSyntaxError, match="site ratios"):
        ThermoDatabase.from_string("PHASE FOO % 2 1 !")


def test_fortran_exponent_without_e():
    # A fixed-width number with the 'E' left out. The ELEMENT line here is
    # COST507's yttrium verbatim, and that one field used to lose the whole
    # 27-element database.
    db = ThermoDatabase.from_string("ELEMENT Y HCP_A3 8.89059+01 5.9664E+03 4.4434E+01 ! PHASE RATIO % 1 1.5+01 !")
    y = db.elements["Y"]
    assert (y.mass, y.h298, y.s298) == pytest.approx((88.9059, 5966.4, 44.434))
    assert db.phases["RATIO"].site_ratios == (15.0,)
    # The repair must not reach inside an expression, where those same characters
    # are addition: 116000+58*T has to keep meaning what it says.
    db2 = ThermoDatabase.from_string("FUNCTION FZ 298.15 116000+58*T; 6000 N !")
    assert db2.evaluate("FZ", 1000.0) == pytest.approx(174000.0)


def test_skipped_commands_are_collected_not_fatal():
    # COST507 has a '!' typed where a '1' belongs. Because '!' ends a command,
    # that one keystroke truncates a PARAMETER and leaves the rest of its number
    # as an orphan command -- and strict parsing loses the other 2,584 with it.
    text = (
        " PHASE LIQUID % 1 1.0 ! CONST LIQUID :A,B: !"
        " PARA G(LIQUID,A;0) 298.15 -1000; 3000 N !"
        " PARA L(LIQUID,A,B;0) 298.15 !.0E-4; 6000 N !"
        " PARA G(LIQUID,B;0) 298.15 -2000; 3000 N !"
    )
    with pytest.raises(TDBSyntaxError):
        ThermoDatabase.from_string(text)
    skipped = []
    db = ThermoDatabase.from_string(text, skipped=skipped)
    assert [s["command"][:14] for s in skipped] == ["PARA L(LIQUID,", ".0E-4; 6000 N"]
    assert "T_low" in skipped[0]["reason"] and "unsupported" in skipped[1]["reason"]
    # the commands on either side of the corruption still landed
    assert {p.label for p in db.parameters} == {"G(LIQUID,A;0)", "G(LIQUID,B;0)"}
    assert "LIQUID" in db.phase_models(("A", "B"))


def test_round_trip_synthetic(synth):
    text = synth.to_tdb()
    db2 = ThermoDatabase.from_string(text)
    assert db2.functions == synth.functions
    assert db2.parameters == synth.parameters
    assert db2.elements == synth.elements and db2.species == synth.species
    assert {k: vars(v) for k, v in db2.phases.items()} == {k: vars(v) for k, v in synth.phases.items()}
    assert db2.to_tdb() == text


def test_add_tdb_edit(synth):
    db = ThermoDatabase.from_tdb(SYNTH)
    before = db.phase_model("LIQUID", "AB").g[0](1000.0)
    db.add_tdb("PARAMETER G(LIQUID,A;0) 298.15 +GALIQ#+100; 3000 N !")
    # two terms now match the endmember: TDB parameters with the same label add, so warn by value
    assert db.phase_model("LIQUID", "AB").g[0](1000.0) == pytest.approx(2 * before + 100)


# ---------------------------------------------------------------------------- phase models


def test_phase_model_kinds_and_errors(synth):
    liq = synth.phase_model("LIQUID", ("A", "B"))
    assert isinstance(liq, BinarySolution) and liq.sites == 1.0 and len(liq.L) == 2
    ab2 = synth.phase_model("AB2", ("A", "B"))
    assert isinstance(ab2, StoichiometricPhase) and ab2.stoichiometry == (1.0, 2.0)
    hcp = synth.phase_model("HCP_A3", ("A", "B"))
    assert isinstance(hcp, StoichiometricPhase) and hcp.stoichiometry == (0.0, 1.0)
    sigma = synth.phase_model("SIGMA", ("A", "B"))
    assert isinstance(sigma, CEFPhase) and sigma.n_y == 2 and sigma.atoms_min == 4.0
    # TC with no MAGNETIC type definition for the phase's type code: ignored, as pycalphad ignores it
    bcc = synth.phase_model("BCC_MAG", ("A", "B"))
    assert isinstance(bcc, BinarySolution)
    with pytest.raises(UnsupportedModelError, match="interstitial"):  # FCC_A1 has a VA sublattice, ORD_PH none
        synth.phase_model("ORD_PH", ("A", "B"))
    with pytest.raises(PhaseNotApplicable):
        synth.phase_model("GRAPHITE", ("A", "B"))
    liq3 = synth.phase_model("LIQUID", ("A", "B", "C"))  # a ternary system: the CEF form, with no C on the lattice
    assert isinstance(liq3, CEFPhase) and liq3.amounts([0.4]) == (0.4, 0.6, 0.0)
    assert set(synth.phase_models(("A", "B"))) == {"LIQUID", "FCC_A1", "AB2", "HCP_A3", "SIGMA", "BCC_MAG"}
    with pytest.raises(UnsupportedModelError):
        synth.phase_models(("A", "B"), phases=["LIQUID", "ORD_PH"])


def test_missing_endmember_is_not_applicable():
    # A multicomponent file parameterizes a phase only for the subsystems it was
    # assessed in, so the constituent test passes and then an endmember has no G:
    # COST507's ALLI does exactly that for Al-Mg. The phase is not applicable to
    # this system rather than the database being broken, so `phase_models` has to
    # skip it -- raised as a bare ValueError it escaped and cost the whole solve.
    db = ThermoDatabase.from_string(
        " PHASE LIQUID % 1 1.0 ! CONST LIQUID :A,B: !"
        " PARA G(LIQUID,A;0) 298.15 -1000; 3000 N !"
        " PARA G(LIQUID,B;0) 298.15 -2000; 3000 N !"
        " PHASE HALF % 1 1.0 ! CONST HALF :A,B: !"
        " PARA G(HALF,A;0) 298.15 -3000; 3000 N !"
    )
    assert set(db.phase_models(("A", "B"))) == {"LIQUID"}
    assert "no G parameter for endmember B" in db.applicability(("A", "B"))["HALF"]
    with pytest.raises(PhaseNotApplicable, match="no G parameter"):
        db.phase_model("HALF", ("A", "B"))
    with pytest.raises(PhaseNotApplicable):  # naming it explicitly still fails
        db.phase_models(("A", "B"), phases=["HALF"])
    assert issubclass(PhaseNotApplicable, ValueError)  # callers catching ValueError are unaffected


def test_element_order_flips_odd_rk_terms(synth):
    t = 1100.0
    ab = synth.phase_model("FCC_A1", ("A", "B"))
    ba = synth.phase_model("FCC_A1", ("B", "A"))
    for y in (0.1, 0.37, 0.8):
        assert ab.gibbs([y], T=t) == pytest.approx(ba.gibbs([1 - y], T=t), rel=1e-13)


def test_liquid_gibbs_by_hand(synth):
    t, y = 1200.0, 0.3
    liq = synth.phase_model("LIQUID", ("A", "B"))
    ga, gb = synth.evaluate("GALIQ", t), synth.evaluate("GBLIQ", t)
    L0, L1 = -5000 + 1.5 * t, 2500.0
    G = y * ga + (1 - y) * gb + R_GAS * t * (y * np.log(y) + (1 - y) * np.log(1 - y))
    G += y * (1 - y) * (L0 + L1 * (2 * y - 1))
    assert liq.gibbs([y], T=t) * R_GAS * t == pytest.approx(G, rel=1e-13)


def test_symbolic_parameter_view(synth):
    liq = synth.phase_model("LIQUID", ("A", "B"))
    assert liq.g[0].expr().has(function_ref("GALIQ"))
    assert not liq.g[0].expr(expand=True).has(function_ref("GHSERA"))
    assert repr(liq.L[1]) == "G(LIQUID,A,B;1)"


def test_sites_and_fixed_amounts():
    ph = BinarySolution("X", ("A", "B"), g=(0.0, 0.0), sites=2.0, fixed=(0.0, 1.0))
    assert ph.atoms_min == 3.0
    assert ph.amounts([0.25]) == (0.5, 2.5)
    assert ph.initial_y([1.0, 2.0]) == pytest.approx([0.5])  # x_A = 1/3: y = (1 - 0)/2
    assert ph.initial_y([0.9, 0.1]) is None  # needs more A than the sublattice holds


# ---------------------------------------------------------------------------- pycalphad oracle


def _pycalphad_gm(path, comps, phase, points, t):
    from pycalphad import Database, calculate

    dbf = Database(str(path))
    res = calculate(dbf, comps, phase, T=t, P=101325, N=1, points=np.atleast_2d(points), output="GM")
    return np.asarray(res.GM).ravel()


def _points(db, name, elements, y):
    """pycalphad site-fraction points (sorted constituents, per sublattice) for our y."""
    ph = db.phases[name]
    allowed = set(elements) | {"VA"}
    subl = [sorted(c for c in s if c in allowed) for s in ph.constituents]
    row = []
    for s in subl:
        if len(s) == 1:
            row.append(1.0)
        else:
            ya = y if elements[0] == s[0] else 1 - y
            row += [ya, 1 - ya]
    return row


PYCALPHAD_R = 8.3145  # pycalphad.variables.R

CASES = [
    ("alzn_mey.tdb", ("AL", "ZN"), ["LIQUID", "FCC_A1", "HCP_A3"]),
    ("cumg.tdb", ("CU", "MG"), ["LIQUID", "FCC_A1", "HCP_A3", "CUMG2"]),
    ("alfe.tdb", ("AL", "FE"), ["LIQUID", "HCP_A3", "AL2FE", "AL5FE2", "AL5FE4"]),
]


@pytest.mark.parametrize("fname,elements,phases", CASES)
@pytest.mark.parametrize("t", [400.0, 692.68, 700.0, 933.47, 1500.0])  # includes breakpoints
def test_gibbs_matches_pycalphad(fname, elements, phases, t):
    path = pycalphad_db(fname)
    db = ThermoDatabase.from_tdb(path)
    comps = [*elements, "VA"]
    for name in phases:
        model = db.phase_model(name, elements)
        ys = [()] if model.n_y == 0 else [(y,) for y in (0.01, 0.3, 0.5, 0.77, 0.99)]
        pts = [_points(db, name, elements, y[0] if y else None) for y in ys]
        gm = _pycalphad_gm(path, comps, name, pts, t)
        for y, ref in zip(ys, gm):
            atoms = float(np.sum(model.amounts(np.asarray(y))))
            ours = model.gibbs(np.asarray(y), T=t) * R_GAS * t / atoms
            if y:  # pycalphad uses R = 8.3145 in the ideal mixing term; we use the CODATA value
                ideal = model.sites * (y[0] * np.log(y[0]) + (1 - y[0]) * np.log(1 - y[0]))
                ours += (PYCALPHAD_R - R_GAS) * t * ideal / atoms
            assert ours == pytest.approx(ref, rel=1e-10, abs=1e-6), (name, y)


@pytest.mark.parametrize(
    "fname",
    [
        "alzn_mey.tdb",
        "cumg.tdb",
        "alfe.tdb",
        "alnipt.tdb",
        "crfe_bcc_magnetic.tdb",
        "alcrni.tdb",
        "alcocrni.tdb",
        "crtiv_ghosh.tdb",
        "COST507.tdb",
        "cumg_parameters.tdb",
        "femn.tdb",
    ],
)
def test_round_trip_real_databases(fname):
    db = ThermoDatabase.from_tdb(pycalphad_db(fname))
    db2 = ThermoDatabase.from_string(db.to_tdb())
    # the writer groups parameters by phase, so compare as multisets
    assert db2.functions == db.functions and Counter(db2.parameters) == Counter(db.parameters)
    for name in list(db.functions)[:20]:
        try:
            v = db.evaluate(name, 1000.0)
        except (KeyError, ValueError):
            continue
        assert db2.evaluate(name, 1000.0) == v


@pytest.mark.slow
def test_alzn_equilibrium_matches_pycalphad():
    """Al–Zn at 600 K, x_Zn = 0.5: FCC miscibility-gap / HCP two-phase region."""
    path = pycalphad_db("alzn_mey.tdb")  # skips when pycalphad is absent
    from pycalphad import Database
    from pycalphad import equilibrium as pyeq
    from pycalphad import variables as v

    from discopt.calphad import CertificateLevel, equilibrium

    db = ThermoDatabase.from_tdb(path)
    phases = list(db.phase_models(("AL", "ZN")).values())
    t, xzn = 600.0, 0.5
    res = equilibrium(phases, x=1 - xzn, T=t)
    assert res.certificate.level >= CertificateLevel.CERTIFIED_TO_TOLERANCE

    eq = pyeq(
        Database(str(path)),
        ["AL", "ZN", "VA"],
        ["LIQUID", "FCC_A1", "HCP_A3"],
        {v.X("ZN"): xzn, v.T: t, v.P: 101325, v.N: 1},
    )
    gm_ref = float(np.squeeze(eq.GM.values))
    mu_ref = np.squeeze(eq.MU.values)
    # tolerances cover the R difference (≈ 4e-5 J/mol/K times the mixing entropy) and pycalphad's own convergence
    assert res.G * R_GAS * t / np.sum(res.b) == pytest.approx(gm_ref, abs=0.1)
    assert res.mu * R_GAS * t == pytest.approx(mu_ref, abs=1.0)
    names = sorted(set(str(p) for p in np.squeeze(eq.Phase.values) if str(p)))
    assert sorted(set(res.phase_names)) == names


def test_coincident_composition_sets_are_merged():
    """Al–Zn at x_Al = 0.05, 500 K: two HCP_A3 sets used to converge onto one composition and split its amount."""
    from discopt.calphad import databases, equilibrium

    pycalphad_db("alzn_mey.tdb")  # the registry reads pycalphad's copy; skip rather than download
    phases = list(databases.load("alzn_mey").phase_models(("AL", "ZN")).values())
    res = equilibrium(phases, x=0.05, T=500.0, certify="none")
    assert sorted(cs.phase.name for cs in res.composition_sets) == ["FCC_A1", "HCP_A3"]
    assert sum(cs.amount * cs.atoms for cs in res.composition_sets) == pytest.approx(1.0)
