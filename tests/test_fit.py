"""Fitting CALPHAD parameters to data (issue #2)."""

from pathlib import Path

import numpy as np
import pytest

from discopt.calphad import ThermoDatabase
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.fit import Dataset, FreeParameter, Observation, fit

TDB = Path(__file__).parents[1] / "examples" / "data" / "alzn_mey.tdb"
EL = ("AL", "ZN")
TRUE = {
    "L(FCC_A1,AL,ZN;0)": (7297.5, 0.47512),
    "L(FCC_A1,AL,ZN;1)": (6612.9, -4.5911),
    "L(LIQUID,AL,ZN;0)": (10465.5, -3.39259),
}


@pytest.fixture(scope="module")
def db():
    return ThermoDatabase.from_tdb(TDB)


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
