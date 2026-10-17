"""Einstein (THETA) and two-state (GD) contributions: model, bounds, and a certified equilibrium."""

from pathlib import Path

import numpy as np
import pytest

from discopt.calphad import CertificateLevel, ThermoDatabase, equilibrium
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models.lattice import einstein, two_state
from discopt.calphad.ops import DUAL, EXPR, INTERVAL, NUM, Dual, Interval

PYCALPHAD_R = 8.3145  # pycalphad.variables.R


def pycalphad_db(name):
    pycalphad = pytest.importorskip("pycalphad")
    return Path(pycalphad.__file__).parent / "tests" / "databases" / name


def femn_liquid():
    return ThermoDatabase.from_tdb(pycalphad_db("femn.tdb")).phase_model("LIQUID", ("FE", "MN"))


def test_femn_liquid_is_supported():
    db = ThermoDatabase.from_tdb(pycalphad_db("femn.tdb"))
    assert db.applicability(("FE", "MN")) == {"LIQUID": "ok"}
    liq = db.phase_model("LIQUID", ("FE", "MN"))
    assert len(liq.einstein) == 2 and len(liq.two_state) == 2 and liq.magnetic.xiong


@pytest.mark.parametrize("T", [300.0, 800.0, 1200.0, 1800.0, 2500.0])
def test_femn_liquid_matches_pycalphad(T):
    """G of Fe–Mn liquid (Xiong magnetism, Einstein, two-state) against pycalphad.calculate."""
    from pycalphad import Database, calculate

    liq = femn_liquid()
    pdb = Database(str(pycalphad_db("femn.tdb")))
    rng = np.random.default_rng(int(T))
    for yfe in rng.uniform(0.001, 0.999, 6):
        ref = calculate(pdb, ["FE", "MN", "VA"], "LIQUID", T=T, P=101325, N=1, points=np.array([[yfe, 1 - yfe, 1.0]]))
        u = liq.variables_from_site_fractions([{"FE": yfe, "MN": 1 - yfe}, {"VA": 1.0}])
        Y = liq._Y(np.asarray(u))
        ours = float(liq.gibbs(u, NUM, T)) * R_GAS * T
        # pycalphad's R is 8.3145: rescale every term that carries R
        ideal = float(liq.ideal_at(Y, NUM))
        mag = float(liq.magnetic_at(Y, T, NUM))
        ein = float(einstein(liq._sum(liq.einstein, Y, T), T, NUM))
        gd = float(liq._sum(liq.two_state, Y, T))
        ours += (PYCALPHAD_R - R_GAS) * T * (ideal + mag + ein)
        ours += -PYCALPHAD_R * T * np.logaddexp(0, -gd / (PYCALPHAD_R * T)) - R_GAS * T * float(two_state(gd, T, NUM))
        assert ours == pytest.approx(float(ref.GM.values.squeeze()), rel=1e-9)


@pytest.mark.parametrize(("f", "args"), [(einstein, (np.log(245.0), 1.0)), (two_state, (42000.0, 30000.0))])
@pytest.mark.parametrize("T", [150.0, 1000.0, 3000.0])
def test_interval_and_dual_rules_enclose(f, args, T):
    """Interval and Dual rules enclose sampled values and the Taylor model holds."""
    mid, half = args
    rng = np.random.default_rng(3)
    for w in (1e-6, 0.1 * half, half):
        a = mid + rng.uniform(-half, half)
        x = Interval(a, a + w)
        iv = f(x, T, INTERVAL)
        c = a + 0.3 * w
        dual = f(Dual(x, Interval(c), (Interval(1.0),)), T, DUAL)
        for t in np.linspace(a, a + w, 25):
            val = float(f(t, T, NUM))
            assert iv.lo <= val <= iv.hi
            taylor = dual.c + dual.d[0] * (Interval(t) - c)
            assert taylor.lo - 1e-12 * abs(val) <= val <= taylor.hi + 1e-12 * abs(val)


def test_expr_matches_num():
    """The discopt expression evaluates to the NUM value."""
    import discopt.modeling as dm

    for f, x in ((einstein, np.log(245.0)), (two_state, 42000.0)):
        m = dm.Model("t")
        v = m.continuous("v", lb=x, ub=x)
        expr = f(v, 1100.0, EXPR)
        m.minimize(expr)
        r = m.solve()
        assert r.objective == pytest.approx(float(f(x, 1100.0, NUM)), rel=1e-7)


def test_gibbs_expr_has_the_new_terms():
    import sympy as sp

    from discopt.calphad.symbolic import T as T_

    db = ThermoDatabase.from_tdb(pycalphad_db("femn.tdb"))
    liq = db.phase_model("LIQUID", ("FE", "MN"))
    expr = liq.gibbs_expr()
    # named database functions (TELIQMN) stay applied functions of T: give them their values
    calls = expr.atoms(sp.core.function.AppliedUndef)
    expr = expr.xreplace({c: db.evaluate(c.func.__name__, 1500.0) for c in calls})
    yfe = 0.4
    u = liq.variables_from_site_fractions([{"FE": yfe, "MN": 1 - yfe}, {"VA": 1.0}])
    sub = {sp.Symbol("y0_FE", nonnegative=True): yfe, sp.Symbol("y0_MN", nonnegative=True): 1 - yfe}
    sub[sp.Symbol("R", positive=True)] = R_GAS
    sub[T_] = 1500.0
    assert float(expr.xreplace(sub).evalf()) == pytest.approx(
        float(liq.gibbs(u, NUM, 1500.0)) * R_GAS * 1500.0, rel=1e-9
    )


def test_femn_liquid_equilibrium_is_certified():
    """One liquid, Fe–Mn at 1500 K: certified, and the same single phase as pycalphad."""
    liq = femn_liquid()
    res = equilibrium([liq], x=0.4, T=1500.0, certify="none")
    cert = res.certify()
    assert cert.level in (CertificateLevel.GLOBALLY_CERTIFIED, CertificateLevel.CERTIFIED_TO_TOLERANCE)
    assert len(res.composition_sets) == 1
    assert float(res.composition_sets[0].x[0]) == pytest.approx(0.4, abs=1e-6)
