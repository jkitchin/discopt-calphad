"""Fitting CALPHAD parameters to data (issue #2)."""

import numpy as np
import pytest

from discopt.calphad import databases
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.fit import Dataset, FreeParameter, Observation, fit

EL = ("AL", "ZN")
TRUE = {
    "L(FCC_A1,AL,ZN;0)": (7297.5, 0.47512),
    "L(FCC_A1,AL,ZN;1)": (6612.9, -4.5911),
    "L(LIQUID,AL,ZN;0)": (10465.5, -3.39259),
}


@pytest.fixture(scope="module")
def db():
    return databases.load("alzn_mey")


def _lm(db, free=()):
    from discopt.calphad.fit import _LinearModel

    return _LinearModel(db, EL, [FreeParameter.parse(f) for f in free])


def test_free_parameter_forms():
    fp = FreeParameter.parse("L(FCC_A1,AL,ZN;0)", "a + b*T + c*T*ln(T)")
    assert fp.names == ("L(FCC_A1,AL,ZN;0).a", "L(FCC_A1,AL,ZN;0).b", "L(FCC_A1,AL,ZN;0).c")
    with pytest.raises(ValueError, match="linear"):
        FreeParameter.parse("L(FCC_A1,AL,ZN;0)", "a*exp(b*T)")


def test_probed_gibbs_matches_the_phase_model(db):
    lm = _lm(db)
    fcc = db.phase_model("FCC_A1", EL)
    for t, y in [(500.0, 0.3), (800.0, 0.85)]:
        g = lm.phase("FCC_A1").F("G", [y], t)[0]
        assert g == pytest.approx(float(fcc.gibbs([y], T=t)) * R_GAS * t, rel=1e-10)


def test_enthalpy_is_the_gibbs_helmholtz_derivative(db):
    lm = _lm(db)
    ph = lm.phase("LIQUID")
    t, h, y = 900.0, 1e-3, [0.4]
    g_over_t = lambda tt: ph.F("G", y, tt)[0] / tt  # noqa: E731
    dgt = (g_over_t(t + h) - g_over_t(t - h)) / (2 * h)
    assert ph.F("H", y, t)[0] == pytest.approx(-t * t * dgt, rel=1e-6)
    assert ph.F("S", y, t)[0] == pytest.approx((ph.F("H", y, t)[0] - ph.F("G", y, t)[0]) / t, rel=1e-9)


def test_mu_is_the_tangent(db):
    lm = _lm(db)
    ph = lm.phase("FCC_A1")
    t, y = 600.0, 0.3
    mu = ph.mu(y, t)[:, 0]
    assert mu @ [y, 1 - y] == pytest.approx(ph.F("G", [y], t)[0], rel=1e-10)
    h = 1e-5
    slope = (ph.F("G", [y + h], t)[0] - ph.F("G", [y - h], t)[0]) / (2 * h)
    assert mu[0] - mu[1] == pytest.approx(slope, rel=1e-6)


def _synthetic(db, rng=None, noise=0.0):
    """Mixing enthalpies, activities and tie-lines computed from the database itself."""
    lm = _lm(db)
    ds = Dataset()
    for kind, phase, t in [("HM_MIX", "LIQUID", 1000.0), ("HM_MIX", "FCC_A1", 650.0), ("SM_MIX", "FCC_A1", 650.0)]:
        for x in np.linspace(0.1, 0.9, 9):
            v = lm.phase(phase).thermo(kind, x, t)[0]
            s = 20.0 if kind == "HM_MIX" else 0.05
            ds.add_thermo(kind, phase, t, x, v + (rng.normal(0, s) if noise else 0.0), s)
    for x in (0.2, 0.5, 0.8):
        ph = lm.phase("LIQUID")
        a = np.exp((ph.mu(x, 1000.0)[1, 0] - ph.pure_mu("ZN", 1000.0)[0]) / (R_GAS * 1000.0))
        ds.add_activity("ZN", "LIQUID", 1000.0, x, a * (1 + rng.normal(0, 0.01)) if noise else a, 0.01 * a)
    return ds


def test_noise_free_recovery(db):
    ds = _synthetic(db)
    res = fit(db, ds, EL, list(TRUE))
    assert res.rank == 6 and res.chi2 < 1e-12
    for label, (a, b) in TRUE.items():
        assert res.parameters[f"{label}.a"] == pytest.approx(a, rel=1e-6)
        assert res.parameters[f"{label}.b"] == pytest.approx(b, rel=1e-5, abs=1e-6)
    # the fitted database reproduces the original model
    fcc0, fcc1 = db.phase_model("FCC_A1", EL), res.database.phase_model("FCC_A1", EL)
    assert fcc1.gibbs([0.37], T=640.0) == pytest.approx(fcc0.gibbs([0.37], T=640.0), rel=1e-9)
    assert "fitted by discopt.calphad.fit" in res.to_tdb()


def test_noisy_recovery_within_uncertainty(db):
    rng = np.random.default_rng(0)
    res = fit(db, _synthetic(db, rng, noise=1.0), EL, list(TRUE))
    for label, (a, b) in TRUE.items():
        for name, true in ((f"{label}.a", a), (f"{label}.b", b)):
            k = res.names.index(name)
            assert abs(res.values[k] - true) < 4 * res.std[k], name
    assert 0.3 < res.chi2 / res.dof < 3.0
    assert "HM_MIX" in res.summary()


def test_tie_line_data_fixes_a_parameter(db):
    """A liquid/fcc tie-line computed from the database recovers the liquid L0 constant."""
    from discopt.calphad import equilibrium

    t = 700.0
    phases = db.phase_models(EL, ["LIQUID", "FCC_A1"])
    res = equilibrium(list(phases.values()), x=0.35, T=t, eps=1e-5)
    assert sorted(res.phase_names) == ["FCC_A1", "LIQUID"]
    pairs = [(cs.phase.name, float(cs.x[0])) for cs in res.composition_sets]
    ds = Dataset().add_equilibrium(t, pairs, sigma=1.0)
    free = {"L(LIQUID,AL,ZN;0)": "a - 3.39259*T"}
    fr = fit(db, ds, EL, free)
    assert fr.parameters["L(LIQUID,AL,ZN;0).a"] == pytest.approx(10465.5, abs=5.0)


def test_bounds_and_rank_warning(db):
    ds = Dataset().add_thermo("HM_MIX", "LIQUID", 1000.0, 0.5, -1000.0, 10.0)
    with pytest.warns(UserWarning, match="determine only 1 of 2"):
        fit(db, ds, EL, ["L(LIQUID,AL,ZN;0)"])
    res = fit(db, ds, EL, {"L(LIQUID,AL,ZN;0)": "a"}, bounds={"L(LIQUID,AL,ZN;0).a": (0, None)})
    assert res.parameters["L(LIQUID,AL,ZN;0).a"] == pytest.approx(0.0, abs=1e-6)
    assert res.active_bounds == ["L(LIQUID,AL,ZN;0).a"]


def test_dataset_round_trip(tmp_path):
    ds = Dataset([Observation("hm_mix", 1000.0, -500.0, 10.0, phase="liquid", x=0.5)])
    ds.add_equilibrium(700.0, {"LIQUID": 0.3, "FCC_A1": 0.6})
    ds.to_json(tmp_path / "d.json")
    back = Dataset.from_json(tmp_path / "d.json")
    assert back.observations == ds.observations
    (tmp_path / "d.csv").write_text("kind,phase,T,x,value,sigma\nHM_MIX,LIQUID,1000,0.5,-500,10\n")
    assert Dataset.from_csv(tmp_path / "d.csv").observations[0] == ds.observations[0]


# ---------------------------------------------------------------------------- multi-sublattice μ, one-sided ZPF, ESPEI

CUMG = ("CU", "MG")
#: Cu2MG's CU:MG endmember keeps its heat-capacity terms; only a is free
CU2MG_FORM = "a + 364.73085*T - 69.276417*T*ln(T) - 0.00051925*T**2 - 5.65953e-6*T**3 + 143502/T"
CUMG_FREE = {"L(LIQUID,CU,MG;0)": "a + 4.75612*T", "G(CU2MG,CU:MG;0)": CU2MG_FORM}
CUMG_START = {"L(LIQUID,CU,MG;0).a": -30000.0, "G(CU2MG,CU:MG;0).a": -50000.0}


@pytest.fixture(scope="module")
def cumg():
    return databases.load("cumg_nims")


@pytest.fixture(scope="module")
def cumg_lines(cumg):
    """Two-phase equilibria of the assessment: LIQUID + CU2MG at 1000 and 900 K, FCC + LIQUID at 1000 K."""
    from discopt.calphad import equilibrium

    phases = list(cumg.phase_models(CUMG).values())
    out = []
    for T, x in [(1000.0, 0.75), (900.0, 0.62), (1000.0, 0.9)]:
        res = equilibrium(phases, x=x, T=T, certify="none")
        out.append((T, [(cs.phase.name, float(cs.x[0])) for cs in res.composition_sets]))
    assert all(len(sets) == 2 for _, sets in out)
    return out


def test_mu_of_a_two_sublattice_phase_is_the_tangent_of_its_internal_equilibrium(cumg):
    from discopt.calphad.fit import _LinearModel

    lm = _LinearModel(cumg, CUMG, [])
    ph = lm.phase("CU2MG")
    assert ph.model.n_y == 2
    t, x = 900.0, 0.66
    mu = ph.mu(x, t)[:, 0]

    def gstar(xx):
        """Minimum of G per atom at fixed x along its 1-D feasible set: x = (2 u0 + u1)/3."""
        from scipy.optimize import minimize_scalar

        lo, hi = max(0.0, (3 * xx - 1) / 2), min(1.0, 3 * xx / 2)
        g = lambda u0: float(ph.model.gibbs([u0, 3 * xx - 2 * u0], T=t)) * R_GAS * t / 3  # noqa: E731
        r = minimize_scalar(g, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
        return min(r.fun, g(lo), g(hi))

    assert mu @ [x, 1 - x] == pytest.approx(gstar(x), abs=1e-3)  # J/mol
    h = 1e-5  # G* curves sharply near Cu2MG's stoichiometry: a fine 5-point stencil for the reference
    slope = (gstar(x - 2 * h) - 8 * gstar(x - h) + 8 * gstar(x + h) - gstar(x + 2 * h)) / (12 * h)
    assert mu[0] - mu[1] == pytest.approx(slope, rel=1e-5)


def test_tie_lines_with_a_cef_phase_recover_parameters(cumg, cumg_lines):
    ds = Dataset()
    for T, sets in cumg_lines:
        ds.add_equilibrium(T, sets, sigma=100.0)
    res = fit(cumg, ds, CUMG, CUMG_FREE, initial=CUMG_START)
    assert res.converged and res.iterations > 1  # Cu2MG's site fractions depend on the parameters
    assert res.parameters["L(LIQUID,CU,MG;0).a"] == pytest.approx(-36984.0, abs=0.05)
    assert res.parameters["G(CU2MG,CU:MG;0).a"] == pytest.approx(-54690.99, abs=0.05)


def test_one_sided_boundary_points_recover_parameters(cumg, cumg_lines):
    """Each tie-line given one side at a time: the other phase must touch the tangent (driving force 0)."""
    ds = Dataset()
    for T, sets in cumg_lines:
        for k in range(2):
            ds.add_equilibrium(T, [(n, x if i == k else None) for i, (n, x) in enumerate(sets)], sigma=100.0)
    res = fit(cumg, ds, CUMG, CUMG_FREE, initial=CUMG_START)
    assert res.converged and res.chi2 < 1e-6
    assert res.parameters["L(LIQUID,CU,MG;0).a"] == pytest.approx(-36984.0, abs=0.05)
    assert res.parameters["G(CU2MG,CU:MG;0).a"] == pytest.approx(-54690.99, abs=0.05)


def test_linear_data_need_one_solve(db):
    res = fit(db, _synthetic(db), EL, list(TRUE))
    assert res.iterations == 1 and res.converged


def test_zpf_without_anything_to_fix_mu_raises(cumg):
    ds = Dataset().add_equilibrium(800.0, [("LIQUID", None), ("CUMG2", None)])
    with pytest.raises(ValueError, match="μ needs a measured composition"):
        fit(cumg, ds, CUMG, CUMG_FREE)


def test_espei_reader_on_the_cu_mg_data(cumg):
    from pathlib import Path

    from discopt.calphad.io.espei import read_espei

    root = Path(__file__).parents[1] / "examples" / "data" / "espei-cu-mg"
    ds, skipped = read_espei(root, cumg, CUMG, phase_map={"LAVES_C15": "CU2MG"})
    kinds = {k: sum(o.kind == k for o in ds) for k in ("ZPF", "HM_MIX", "HM_FORM", "ACR")}
    assert kinds == {"ZPF": 238, "HM_MIX": 55, "HM_FORM": 15, "ACR": 9}
    reasons = " ".join(r for _, r in skipped)
    assert "BCC_A2" in reasons and "does not fit HCP_A3" in reasons
    # Bocklund's Laves mixing enthalpies mix one sublattice at a time: read, and usable by the fit
    laves = [o for o in ds if o.kind == "HM_MIX" and o.phase == "CU2MG"]
    assert len(laves) == 4
    # Garg (1973): a_Mg in the liquid at 1200 K, x_Cu = 0.9 -> x (first element, Cu) = 0.9, σ(ln a) = 500/RT
    acr = next(o for o in ds if o.kind == "ACR" and o.x == pytest.approx(0.9))
    assert acr.element == "MG" and acr.value == pytest.approx(0.0057)
    assert acr.sigma / acr.value == pytest.approx(500.0 / (R_GAS * 1200.0))
    assert acr.source == "garg1973thermodynamic"
    # ZPF compositions are given as x_Mg: converted to x_Cu, and a compound's composition is dropped
    zpf = next(o for o in ds if o.kind == "ZPF" and o.source == "Jones1931")
    assert zpf.phases[0] == ("LIQUID", pytest.approx(1 - 0.999988522)) and zpf.phases[1] == ("HCP_A3", None)
    # Zhou (2007) Laves formation enthalpies: one observation per sublattice configuration
    zhou = [o for o in ds if o.source == "Zhou2007"]
    assert [o.site_fractions for o in zhou][0] == ({"CU": 1.0}, {"MG": 1.0})
    assert [o.value for o in zhou if o.phase == "CU2MG"] == [-15720.0, 34720.0, 7000.0, 15500.0]


# ---------------------------------------------------------------------------- magnetic phases

ALNI = ("AL", "NI")


@pytest.fixture(scope="module")
def alni():
    return databases.load("alni_dupin")


def test_magnetic_phase_rows_include_the_fixed_magnetic_term(alni):
    """FCC_A1 is magnetic in Al-Ni: G matches the model, H is the Gibbs-Helmholtz derivative, also at 298.15 K."""
    from discopt.calphad.fit import _LinearModel

    fcc = _LinearModel(alni, ALNI, []).phase("FCC_A1")
    model = alni.phase_model("FCC_A1", ALNI)
    assert model.magnetic is not None
    for t, x in [(600.0, 0.1), (298.15, 0.0)]:
        y = fcc.y_from_site_fractions([{"AL": x, "NI": 1 - x}, {"VA": 1.0}])
        assert fcc.F("G", y, t)[0] == pytest.approx(float(model.gibbs(y, T=t)) * R_GAS * t, rel=1e-12)
    y, t, h = fcc.y_from_site_fractions([{"AL": 0.1, "NI": 0.9}, {"VA": 1.0}]), 600.0, 1e-3
    g_over_t = lambda tt: fcc.F("G", y, tt)[0] / tt  # noqa: E731
    assert fcc.F("H", y, t)[0] == pytest.approx(-t * t * (g_over_t(t + h) - g_over_t(t - h)) / (2 * h), rel=1e-7)
    assert np.isfinite(fcc.F("H", fcc.y_from_site_fractions([{"NI": 1.0}, {"VA": 1.0}]), 298.15)[0])


def test_fit_through_a_magnetic_reference_and_a_magnetic_phase(alni):
    """A compound's formation enthalpy refers to magnetic FCC Ni; FCC itself is magnetic. Both used to raise."""
    from discopt.calphad.fit import _LinearModel

    lm = _LinearModel(alni, ALNI, [])
    fcc = lm.phase("FCC_A1")
    ds = Dataset().add_formation(
        "AL3NI1", lm.phase("AL3NI1").formation(np.empty(0), 298.15)[0], 100.0, endmember="AL:NI"
    )
    for x in (0.1, 0.2):
        y = fcc.y_from_site_fractions([{"AL": x, "NI": 1 - x}, {"VA": 1.0}])
        ds.add_thermo("HM_MIX", "FCC_A1", 800.0, x, fcc.thermo("HM_MIX", None, 800.0, y)[0], 50.0)
    free = {
        "G(AL3NI1,AL:NI;0)": "a + 12.29913*T + 0.75*GHSERAL + 0.25*GHSERNI",
        "L(FCC_A1,AL,NI:VA;0)": "a + 16.212965*T",
        "L(FCC_A1,AL,NI:VA;1)": "a - 34.914168*T",
    }
    res = fit(alni, ds, ALNI, free)
    assert res.parameters["G(AL3NI1,AL:NI;0).a"] == pytest.approx(-48483.73, abs=1e-3)
    assert res.parameters["L(FCC_A1,AL,NI:VA;0).a"] == pytest.approx(-162407.75, abs=1e-3)
    assert res.parameters["L(FCC_A1,AL,NI:VA;1).a"] == pytest.approx(73417.798, abs=1e-3)
    with pytest.raises(NotImplementedError, match="TC and BMAGN cannot be fitted"):
        fit(alni, ds, ALNI, {"TC(FCC_A1,NI:VA;0)": "a"})


def test_predict_and_enthalpy_form():
    from discopt.calphad.fit import enthalpy_form, predict

    alcu = databases.load("alcuy_zhang")
    assert enthalpy_form(alcu, "G(ALCU_THETA,AL:CU;0)") == "a + 6.75*T+2*GHSERAL#+GHSERCU#"
    assert enthalpy_form(alcu, "L(FCC_A1,AL,CU:VA;1)") == "a - 2*T"
    assert enthalpy_form(alcu, "L(FCC_A1,AL,CU:VA;2)") == "a"
    ds = Dataset().add_formation("ALCU_THETA", 0.0, 1.0, endmember="AL:CU")
    ds.add_thermo("HM_MIX", "FCC_A1", 298.15, 0.5, 0.0, 1.0)
    # θ: -47406/3 J/mol of atoms; FCC: x(1-x) L0 at 298 K, H of (-53520 + 2T) is -53520
    assert predict(alcu, ds, ("AL", "CU")) == pytest.approx([-47406 / 3, 0.25 * -53520], abs=1.0)
    # an enthalpy-only fit with that form recovers the constant and keeps the assessed entropy term
    theta = Dataset(ds.observations[:1])
    res = fit(alcu, theta, ("AL", "CU"), {"G(ALCU_THETA,AL:CU;0)": enthalpy_form(alcu, "G(ALCU_THETA,AL:CU;0)")})
    assert res.parameters["G(ALCU_THETA,AL:CU;0).a"] == pytest.approx(0.0, abs=1e-3)  # the datum said ΔH_f = 0


def test_mixing_on_one_sublattice_of_a_cef_phase(cumg):
    """HM_MIX on Cu2Mg with one sublattice mixed: relative to the two endmembers of that sublattice."""
    from discopt.calphad.fit import _LinearModel, predict

    ph = _LinearModel(cumg, CUMG, []).phase("CU2MG")
    sf = ({"CU": 0.5, "MG": 0.5}, {"MG": 1.0})
    obs = Observation("HM_MIX", 298.15, 0.0, 1.0, phase="CU2MG", site_fractions=sf)
    h = lambda f: ph.F("H", ph.y_from_site_fractions(f), 298.15)[0] / 3  # noqa: E731
    expected = h(sf) - 0.5 * h(({"CU": 1.0}, {"MG": 1.0})) - 0.5 * h(({"MG": 1.0}, {"MG": 1.0}))
    assert predict(cumg, [obs], CUMG)[0] == pytest.approx(expected, rel=1e-12)
    # the CU,MG:* interaction is per formula unit (y_Cu y_Mg L0), and a formula unit has 3 atoms
    assert expected == pytest.approx(0.25 * 6599.45 / 3, rel=1e-6)
    with pytest.raises(ValueError, match="exactly one mixed sublattice"):
        bad = Observation(
            "HM_MIX", 298.15, 0.0, 1.0, phase="CU2MG", site_fractions=({"CU": 0.5, "MG": 0.5}, {"CU": 0.5, "MG": 0.5})
        )
        predict(cumg, [bad], CUMG)
