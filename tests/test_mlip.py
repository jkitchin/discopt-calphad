"""CALPHAD data from ASE calculators (issue #5, phase 1): EMT and mock calculators only."""

import dataclasses

import numpy as np
import pytest

pytest.importorskip("ase")
pytest.importorskip("icet")

from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402

from discopt.calphad import databases, equilibrium, mlip  # noqa: E402
from discopt.calphad.fit import Dataset, Observation, fit  # noqa: E402
from discopt.calphad.mlip import EV_TO_J_PER_MOL as F  # noqa: E402

EL = ("AL", "CU")
THETA = {"ALCU_THETA": ("C16", ["8h"], ["4a"])}
FAST_SQS = mlip.SQSOptions(n_steps=200)
#: regular-solution energies of the mock calculator, eV/atom
L0, L1 = -0.12, 0.05


class RegularSolution(Calculator):
    """E/atom = x e_Al + (1 − x) e_Cu + x(1 − x)(L0 + L1 (2x − 1)), with x the Al fraction; no forces."""

    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        s = self.atoms.get_chemical_symbols()
        n, x = len(s), s.count("Al") / len(s)
        e = -3.0 * x - 2.0 * (1 - x) + x * (1 - x) * (L0 + L1 * (2 * x - 1))
        self.results = {"energy": n * e, "forces": np.zeros((n, 3)), "stress": np.zeros(6)}


class SizeDependent(RegularSolution):
    """The mock energy plus a finite-size term c/N, so SQS in different supercells disagree."""

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        s = self.atoms.get_chemical_symbols()
        x = s.count("Al") / len(s)
        if 0 < x < 1:
            self.results["energy"] += 0.4 * x  # per cell: 0.4 x / N per atom


class PullCopper(RegularSolution):
    """The mock energy plus a constant force on the Cu atoms of a compound: relaxing it leaves the prototype."""

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        cu = np.array([s == "Cu" for s in self.atoms.get_chemical_symbols()])
        if cu.all() or not cu.any():
            return
        f = np.zeros((len(cu), 3))
        f[cu, 0] = 0.5
        self.results["forces"] = f
        self.results["energy"] -= 0.5 * float(self.atoms.positions[cu, 0].sum())


@pytest.fixture(scope="module")
def db():
    return databases.load("alcuy_zhang")


# ---------------------------------------------------------------------------- prototypes


def test_prototypes_have_the_right_sites():
    counts = {
        "A1": 4,
        "A2": 2,
        "A3": 2,
        "A4": 8,
        "B2": 2,
        "L1_2": 4,
        "D0_19": 8,
        "C15": 24,
        "C14": 12,
        "C36": 24,
        "C16": 12,
        "D8_b": 30,
    }
    for name, n in counts.items():
        assert sum(mlip.PROTOTYPES[name].multiplicities().values()) == n, name
    assert mlip.PHASES["LAVES_C15"].site_counts() == [16, 8]
    assert mlip.PHASES["SIGMA"].site_counts() == [10, 4, 16]


def test_phase_structure_builds_endmembers_and_checks_ratios(db):
    l12 = mlip.PHASES["L12_FCC"].build(("CU", "AU", "VA"), volume_per_atom=12.0)
    assert l12.get_chemical_formula() == "AuCu3" and l12.get_volume() == pytest.approx(48.0)
    b2 = mlip.PHASES["B2_BCC"].build(("AL", "VA", "VA"))  # a vacancy sublattice leaves its sites empty
    assert len(b2) == 1
    theta = mlip.phase_structure(*THETA["ALCU_THETA"])
    theta.check_ratios(db.phases["ALCU_THETA"].site_ratios, name="ALCU_THETA")
    with pytest.raises(ValueError, match="do not match"):
        mlip.phase_structure("C15", ["16d"], ["8a"]).check_ratios((1.0, 1.0), name="X")
    with pytest.raises(KeyError, match="no structure registered"):
        mlip.phase_structure("ALCU_THETA")


# ---------------------------------------------------------------------------- relaxation


def test_symmetric_relaxation_stays_on_its_prototype():
    model = mlip.from_ase(EMT(), name="EMT")
    for phase in ("FCC_A1", "BCC_A2", "HCP_A3"):
        atoms = mlip.PHASES[phase].build(("CU", "VA"), mlip.atomic_volume("Cu"))
        rel = model.relax([atoms])[0]
        assert rel.converged and not rel.left_prototype, (phase, rel.reason)
    # a c/a change keeps the hexagonal symmetry, so it is not a departure
    hcp = mlip.PHASES["HCP_A3"].build(("CU", "VA"), mlip.atomic_volume("Cu"))
    hcp.set_cell(hcp.cell * [1, 1, 1.12], scale_atoms=True)
    assert not model.relax([hcp], mlip.RelaxPolicy(mode="full"))[0].left_prototype


def test_bcc_collapse_is_flagged():
    model = mlip.from_ase(EMT(), name="EMT")
    bcc = mlip.PHASES["BCC_A2"].build(("CU", "VA"), mlip.atomic_volume("Cu")).repeat(2)
    bcc.rattle(0.03, seed=1)
    rel = model.relax([bcc], mlip.RelaxPolicy(mode="full", steps=300))[0]
    assert rel.left_prototype and "strain" in rel.reason
    fcc = model.relax([mlip.PHASES["FCC_A1"].build(("CU", "VA"), mlip.atomic_volume("Cu"))])[0]
    assert rel.energy_per_atom == pytest.approx(fcc.energy_per_atom, abs=2e-3)  # it became close-packed


def test_a_structure_that_runs_away_is_left_out_of_the_data(db):
    model = mlip.from_ase(PullCopper(), name="pull")
    with pytest.warns(UserWarning, match="left its prototype"):
        res = mlip.compute(
            model,
            db,
            EL,
            compounds={"ALCU_THETA": ["AL:CU"]},
            structures=THETA,
            sigma=1.0,
            relax=mlip.RelaxPolicy(mode="full", cell=False, steps=20, optimizer="FIRE"),
        )
    assert len(res.dataset) == 0
    assert res.flagged and res.flagged[0][0] == "ALCU_THETA AL:CU"


def test_relaxations_are_cached(tmp_path):
    atoms = mlip.PHASES["FCC_A1"].build(("AL", "VA"), mlip.atomic_volume("Al"))
    model = mlip.from_ase(EMT(), name="EMT", cache=tmp_path)
    first = model.relax([atoms])[0]
    assert not first.cached and len(list(tmp_path.glob("*.json"))) == 1
    again = mlip.from_ase(EMT(), name="EMT", cache=tmp_path).relax([atoms])[0]
    assert again.cached and again.energy == first.energy
    # another source label or other settings are other entries
    assert not mlip.from_ase(EMT(), name="EMT-2", cache=tmp_path).relax([atoms])[0].cached
    assert not model.relax([atoms], mlip.RelaxPolicy(fmax=0.001))[0].cached


# ---------------------------------------------------------------------------- data and fits


def _mock_result(db, **kwargs):
    model = mlip.from_ase(RegularSolution(), name="mock")
    return mlip.compute(
        model,
        db,
        EL,
        compounds={"ALCU_THETA": ["AL:CU"]},
        structures=THETA,
        mixing={"FCC_A1": [0.25, 0.5, 0.75]},
        sqs=FAST_SQS,
        **kwargs,
    )


def test_mock_calculator_gives_an_exact_fit(db):
    res = _mock_result(db, sigma=1.0)
    assert {o.source for o in res.dataset} == {"mock"}
    assert all(o.note == mlip.APPROXIMATION for o in res.dataset)
    free = {
        "L(FCC_A1,AL,CU:VA;0)": "a",
        "L(FCC_A1,AL,CU:VA;1)": "a",
        "L(FCC_A1,AL,CU:VA;2)": "a",
        "G(ALCU_THETA,AL:CU;0)": "a + 6.75*T + 2*GHSERAL + GHSERCU",
    }
    fr = fit(db, res.dataset, EL, free)
    p = fr.parameters
    assert p["L(FCC_A1,AL,CU:VA;0).a"] == pytest.approx(L0 * F, rel=1e-9)
    assert p["L(FCC_A1,AL,CU:VA;1).a"] == pytest.approx(L1 * F, rel=1e-9)
    assert p["L(FCC_A1,AL,CU:VA;2).a"] == pytest.approx(0.0, abs=1e-6)
    assert p["G(ALCU_THETA,AL:CU;0).a"] == pytest.approx(3 * (2 / 9) * (L0 + L1 / 3) * F, rel=1e-9)


def test_sigma_rules(db):
    # a single structure has no spread
    with pytest.raises(ValueError, match="single structure.*sigma="):
        _mock_result(db)
    # the mock's SQS energies agree exactly, so they have no spread either
    with pytest.raises(ValueError, match="zero spread.*sigma="):
        _mock_result(db, sigma={"ALCU_THETA": 1.0})
    # a model uncertainty is a spread; a given sigma (kJ/mol) overrides it
    model = mlip.from_ase(RegularSolution(), name="mock", uncertainty=lambda atoms: 0.01)
    res = mlip.compute(model, db, EL, compounds={"ALCU_THETA": ["AL:CU"]}, structures=THETA)
    expected = np.sqrt(0.01**2 + (2 / 3 * 0.01) ** 2 + (1 / 3 * 0.01) ** 2) * F
    assert res.dataset.observations[0].sigma == pytest.approx(expected)
    res = mlip.compute(model, db, EL, compounds={"ALCU_THETA": ["AL:CU"]}, structures=THETA, sigma=3.0)
    assert res.dataset.observations[0].sigma == 3000.0
    # a user model_sigma (kJ/mol) is enough for a single structure
    res = _mock_result(db, model_sigma=1.5)
    assert {o.sigma for o in res.dataset} == {1500.0}  # the mock's SQS spread is zero, so only the model part


def test_pooled_spread(db):
    """Two SQS per composition: one σ per phase with Σ(n_i − 1) degrees of freedom, unless pool=False."""
    model = mlip.from_ase(SizeDependent(), name="size")
    kw = dict(mixing={"FCC_A1": [0.25, 0.5, 0.75]}, sqs=FAST_SQS)
    with pytest.warns(UserWarning, match="only the configurational"):
        alone = mlip.compute(model, db, EL, pool=False, **kw)
    per = np.array([o.sigma for o in alone.dataset])
    assert len(set(per)) == 3 and [r["sigma_dof"] for r in alone.rows] == [1, 1, 1]
    with pytest.warns(UserWarning, match="only the configurational"):
        pooled = mlip.compute(model, db, EL, **kw)
    sig = np.array([o.sigma for o in pooled.dataset])
    assert np.allclose(sig, np.sqrt(np.mean(per**2)))  # equal weights: each composition has one dof
    assert [r["sigma_dof"] for r in pooled.rows] == [3, 3, 3]
    # the model part adds in quadrature, and silences the warning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        both = mlip.compute(model, db, EL, model_sigma={"FCC_A1": 1.0}, **kw)
    assert np.allclose([o.sigma for o in both.dataset], np.sqrt(sig**2 + 1000.0**2))
    assert "dof" in both.table()


def test_emt_pipeline_end_to_end(db):
    """Structures -> EMT relaxations (SQS included) -> Dataset -> fit -> equilibrium."""
    model = mlip.from_ase(EMT(), name="EMT")
    res = mlip.compute(
        model,
        db,
        EL,
        compounds={"ALCU_THETA": ["AL:CU"]},
        structures=THETA,
        mixing={"FCC_A1": [0.25, 0.5, 0.75]},
        model_sigma=2.0,
        sqs=FAST_SQS,
    )
    assert not res.flagged
    mix = [o for o in res.dataset if o.kind == "HM_MIX"]
    # σ: the pooled spread of two SQS per composition, and the 2 kJ/mol model part, in quadrature
    assert len(mix) == 3 and len({o.sigma for o in mix}) == 1 and mix[0].sigma > 2000.0
    assert [r["structures"] for r in res.rows if r["kind"] == "HM_MIX"] == [2, 2, 2]
    # mixed sources: two hand-entered "experimental" points
    data = Dataset(res.dataset.observations)
    data.add_thermo("HM_MIX", "FCC_A1", 298.15, [0.3, 0.7], [7000.0, 6000.0], 500.0)
    for o in data.observations[-2:]:
        o.source = "experiment"
    start = dataclasses.replace(db, parameters=[p for p in db.parameters if p.label != "G(FCC_A1,AL,CU:VA;2)"])
    free = {
        "L(FCC_A1,AL,CU:VA;0)": "a",
        "L(FCC_A1,AL,CU:VA;1)": "a",
        "G(ALCU_THETA,AL:CU;0)": "a + 6.75*T + 2*GHSERAL + GHSERCU",
    }
    fr = fit(start, data, EL, free)
    assert fr.rank == 3
    assert "$ data sources: EMT (4), experiment (2)" in fr.to_tdb()
    phases = list(fr.database.phase_models(EL).values())
    eq = equilibrium(phases, x=0.9, T=700.0, certify="none")
    assert eq.composition_sets


def test_only_sqs_in_phase_one(db):
    model = mlip.from_ase(RegularSolution(), name="mock")
    with pytest.raises(NotImplementedError, match="'sqs' only"):
        mlip.compute(model, db, EL, mixing={"FCC_A1": [0.5]}, method="random", sigma=1.0)


# ---------------------------------------------------------------------------- fit extensions


def test_formation_data_on_a_cef_endmember(db):
    """AL2Y is a two-sublattice CEF phase in Al-Cu; its CU:AL endmember is reached by site fractions."""
    from discopt.calphad.fit import _LinearModel

    lm = _LinearModel(db, EL, [])
    ph = lm.phase("AL2Y")
    assert ph.model.n_y == 2
    true = -12000.0  # J/mol of atoms
    label = "G(AL2Y,CU:AL;0)"
    data = Dataset().add_formation("AL2Y", true, 10.0, endmember="CU:AL", source="test")
    # (CU)0.66667(AL)0.33333: one atom per formula unit
    fr = fit(db, data, EL, {label: "a + 0.66667*GHSERCU + 0.33333*GHSERAL"})
    fitted = _LinearModel(fr.database, EL, []).phase("AL2Y")
    y = fitted.y_from_site_fractions([{"CU": 1}, {"AL": 1}])
    assert fitted.formation(y, 298.15)[0] == pytest.approx(true, abs=1e-6)
    assert fr.parameters[f"{label}.a"] == pytest.approx(true, rel=1e-6)


def test_cef_endmembers_from_a_calculator(db):
    """AL2Y is a C15 Laves phase with (Al,Cu) on both sublattices: both mixed endmembers, then an exact refit."""
    model = mlip.from_ase(RegularSolution(), name="mock")
    res = mlip.compute(
        model, db, EL, compounds=["AL2Y"], structures={"AL2Y": mlip.PHASES["LAVES_C15"]}, model_sigma=1.0
    )
    rows = {r["config"]: r for r in res.rows}
    assert sorted(rows) == ["AL:CU", "CU:AL"]  # the pure endmembers AL:AL and CU:CU are not compounds
    for config, x in (("AL:CU", 2 / 3), ("CU:AL", 1 / 3)):
        assert rows[config]["x"] == pytest.approx(x)
        assert rows[config]["value"] * 1000 == pytest.approx(x * (1 - x) * (L0 + L1 * (2 * x - 1)) * F, rel=1e-9)
    # (AL,CU)0.66667(AL,CU)0.33333 holds one atom per formula unit
    free = {
        "G(AL2Y,AL:CU;0)": "a + 0.66667*GHSERAL + 0.33333*GHSERCU",
        "G(AL2Y,CU:AL;0)": "a + 0.66667*GHSERCU + 0.33333*GHSERAL",
    }
    fr = fit(db, res.dataset, EL, free)
    assert fr.chi2 < 1e-12
    for label, config in (("G(AL2Y,AL:CU;0)", "AL:CU"), ("G(AL2Y,CU:AL;0)", "CU:AL")):
        assert fr.parameters[f"{label}.a"] == pytest.approx(rows[config]["value"] * 1000, rel=1e-6)


def test_site_fraction_observations_round_trip(tmp_path):
    with pytest.raises(ValueError, match="sum to 1"):
        Observation("HM_FORM", 298.15, -1.0, 1.0, phase="X", site_fractions=({"AL": 0.5},))
    with pytest.raises(ValueError, match="give x or site_fractions"):
        Observation("HM_MIX", 298.15, -1.0, 1.0, phase="FCC_A1")
    ds = Dataset().add_formation("ALCU_THETA", -15000.0, 2000.0, endmember="AL:CU", source="uma", note="0 K")
    ds.to_json(tmp_path / "d.json")
    back = Dataset.from_json(tmp_path / "d.json")
    assert back.observations == ds.observations
    assert back.observations[0].site_fractions == ({"AL": 1.0}, {"CU": 1.0})
    assert back.sources() == {"uma": 1}


def test_a_structure_from_plain_atoms_matches_the_registered_prototype(db):
    """θ-Al2Cu given as ase.Atoms plus the atoms of each sublattice: the same data as the C16 prototype."""
    c16 = mlip.phase_structure(*THETA["ALCU_THETA"])
    atoms = c16.build(("AL", "CU"), volume_per_atom=15.0)
    wyckoff = atoms.get_array("wyckoff")
    sublattices = [[i for i, w in enumerate(wyckoff) if w == "8h"], [i for i, w in enumerate(wyckoff) if w == "4a"]]
    custom = mlip.PhaseStructure.from_atoms(atoms, sublattices, name="theta")
    assert custom.site_counts() == [8, 4]
    model = mlip.from_ase(EMT(), name="EMT")
    kw = dict(compounds={"ALCU_THETA": ["AL:CU"]}, sigma=1.0)
    ref = mlip.compute(model, db, EL, structures=THETA, **kw)
    mine = mlip.compute(model, db, EL, structures={"ALCU_THETA": (atoms, sublattices)}, **kw)
    assert mine.dataset.observations[0].value == pytest.approx(ref.dataset.observations[0].value, abs=1e-6)
    with pytest.raises(ValueError, match="every atom"):
        mlip.PhaseStructure.from_atoms(atoms, [sublattices[0], []])
    with pytest.raises(ValueError, match="already on another"):
        mlip.PhaseStructure.from_atoms(atoms, [sublattices[0], sublattices[0] + sublattices[1]])


def test_new_prototypes_match_their_aflow_cifs():
    """Each prototype from the AFLOW library rebuilds its CIF's atom count and has no overlapping atoms."""
    from ase.neighborlist import neighbor_list

    counts = {
        "D0_20": 16,
        "D0_11": 16,
        "D5_13": 5,
        "D0_22": 8,
        "D0_23": 16,
        "D0_3": 16,
        "L1_0": 2,
        "B19": 4,
        "C11_b": 6,
        "B8_1": 4,
        "Ga3Pt5": 16,
    }
    for name, n in counts.items():
        p = mlip.PROTOTYPES[name]
        atoms = p.build({k: "Cu" for k in p.sites}, volume_per_atom=12.0)
        assert len(atoms) == n, name
        assert neighbor_list("d", atoms, 4.0).min() > 1.9, name


def test_structure_from_a_cif_file():
    from pathlib import Path

    cif = Path(__file__).parent / "data" / "al3ni_D020_aflow.cif"
    by_element = mlip.PhaseStructure.from_cif(cif, [["Al"], ["Ni"]])
    by_label = mlip.PhaseStructure.from_cif(cif, [["Al1", "Al2"], ["Ni3"]])
    registered = mlip.phase_structure("D0_20", ["4c1", "8d"], ["4c2"])
    assert by_element.site_counts() == by_label.site_counts() == registered.site_counts() == [12, 4]
    assert by_element.build(("AL", "NI")).get_chemical_formula() == "Al12Ni4"
    with pytest.raises(ValueError, match="no site or element"):
        mlip.PhaseStructure.from_cif(cif, [["Al"], ["Cu"]])
    with pytest.raises(ValueError, match="every atom"):
        mlip.PhaseStructure.from_cif(cif, [["Al1"], ["Ni"]])


def test_sqs_mixing_on_a_sublattice_that_also_allows_vacancies():
    """(AL,CU,VA)(CU): SQS mix Al and Cu at zero vacancy fraction, and the fit uses the data exactly."""
    from discopt.calphad import ThermoDatabase

    db = ThermoDatabase.from_string(
        """ELEMENT AL FCC_A1 26.98 4577.3 28.322 !
        ELEMENT CU FCC_A1 63.546 5004.1 33.15 !
        ELEMENT VA VACUUM 0 0 0 !
        PHASE FCC_A1 % 2 1 1 ! CONSTITUENT FCC_A1 :AL,CU:VA: !
        PARAMETER G(FCC_A1,AL:VA;0) 298.15 0; 6000 N !
        PARAMETER G(FCC_A1,CU:VA;0) 298.15 0; 6000 N !
        PHASE BX % 2 1 1 ! CONSTITUENT BX :AL,CU,VA:CU: !
        PARAMETER G(BX,AL:CU;0) 298.15 -1000; 6000 N !
        PARAMETER G(BX,CU:CU;0) 298.15 0; 6000 N !
        PARAMETER G(BX,VA:CU;0) 298.15 30000; 6000 N !"""
    )
    model = mlip.from_ase(RegularSolution(), name="mock")
    res = mlip.compute(
        model,
        db,
        EL,
        mixing={"BX": [0.25, 0.5, 0.75]},
        structures={"BX": ("B2", ["1a"], ["1b"])},
        sigma=1.0,
        sqs=FAST_SQS,
    )
    mix = [o for o in res.dataset if o.kind == "HM_MIX"]
    assert len(mix) == 3 and all("VA" not in o.site_fractions[0] or o.site_fractions[0]["VA"] == 0 for o in mix)
    fr = fit(db, res.dataset, EL, {"L(BX,AL,CU:CU;0)": "a", "L(BX,AL,CU:CU;1)": "a"})
    assert fr.chi2 < 1e-12  # the mock's energy is cubic in y, which L0 and L1 represent exactly
