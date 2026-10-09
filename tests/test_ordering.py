"""Plain CEF stand-ins for order-disorder phases (issue #14)."""

import numpy as np
import pytest

from discopt.calphad import R_GAS, ThermoDatabase, databases
from discopt.calphad.ordering import stand_in

EL = ("AL", "NI")


@pytest.fixture(scope="module")
def alni():
    return databases.load("alni_dupin")


@pytest.mark.parametrize("ordered", ["FCC_L12", "BCC_B2"])
def test_stand_in_is_exact_and_leaves_the_database_alone(alni, ordered):
    si = stand_in(alni, ordered, EL)
    assert si.disordered == {"FCC_L12": "FCC_A1", "BCC_B2": "BCC_A2"}[ordered]
    assert max(si.max_error.values()) < 1e-6  # J/mol of atoms: L1₂ and B2 project exactly
    assert any("magnetic" in d for d in si.dropped)
    if ordered == "BCC_B2":
        assert any("vacancies" in d for d in si.dropped)
    db = si.database
    assert db.applicability(EL)[si.name] == "ok"
    # the caller's phases are untouched: FCC_A1 is still magnetic, BCC_A2 still has vacancies
    assert db.phase_model("FCC_A1", EL).magnetic is not None
    assert db.phases["BCC_A2"].constituents == alni.phases["BCC_A2"].constituents
    assert len(alni.parameters) < len(db.parameters) and si.name not in alni.phases
    # TDB round trip
    back = ThermoDatabase.from_string(db.to_tdb())
    sf = [{"AL": 0.3, "NI": 0.7}, {"AL": 0.8, "NI": 0.2}, {"VA": 1.0}]
    m1, m2 = db.phase_model(si.name, EL), back.phase_model(si.name, EL)
    u = m1.variables_from_site_fractions(sf)
    assert float(m2.gibbs(u, T=900.0)) == pytest.approx(float(m1.gibbs(u, T=900.0)), rel=1e-10)


@pytest.mark.parametrize("ordered", ["FCC_L12", "BCC_B2"])
def test_stand_in_matches_pycalphads_partitioned_model(alni, ordered):
    """Against pycalphad's own order-disorder model, magnetism and volume left out (they are dropped)."""
    pycalphad = pytest.importorskip("pycalphad")

    class NoMag(pycalphad.Model):
        contributions = [c for c in pycalphad.Model.contributions if c[0] not in ("mag", "vol")]

    pdb = pycalphad.Database(str(databases.fetch("alni_dupin")))
    si = stand_in(alni, ordered, EL)
    model = si.database.phase_model(si.name, EL)
    for T in (600.0, 1200.0):
        for y0, y1 in [(0.1, 0.9), (0.25, 0.25), (0.6, 0.05), (0.9, 0.7)]:
            sf = [{"AL": y0, "NI": 1 - y0}, {"AL": y1, "NI": 1 - y1}, {"VA": 1.0}]
            pts = [y0, 1 - y0, y1, 1 - y1, 1.0] if ordered == "FCC_L12" else [y0, 1 - y0, 0, y1, 1 - y1, 0, 1.0]
            r = pycalphad.calculate(
                pdb,
                ["AL", "NI", "VA"],
                ordered,
                T=T,
                P=101325,
                N=1,
                points=np.array([pts]),
                model={ordered: NoMag},
                output="GM",
            )
            u = model.variables_from_site_fractions(sf)
            atoms = float(sum(model.amounts(u)))
            mine = float(model.gibbs(u, T=T)) * R_GAS * T / atoms
            smix = -sum(a * sum(v * np.log(v) for v in d.values() if v > 0) for a, d in zip(model.site_ratios, sf))
            mine += (R_GAS - 8.3145) * T * smix / atoms  # pycalphad's R
            assert mine == pytest.approx(float(r.GM.values.squeeze()), abs=1e-6)


def test_a_phase_without_a_disordered_part_is_refused(alni):
    with pytest.raises(ValueError, match="no DISORDERED_PART"):
        stand_in(alni, "AL3NI2", EL)
