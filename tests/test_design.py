"""Design of experiments for a fit (issue #10)."""

import dataclasses

import numpy as np
import pytest

from discopt.calphad import databases, equilibrium
from discopt.calphad.design import candidate_rows, design_measurements, rank_candidates
from discopt.calphad.fit import Dataset, Observation, fit, predict

EL = ("AL", "ZN")
FREE = {"L(FCC_A1,AL,ZN;0)": "a + b*T", "L(FCC_A1,AL,ZN;1)": "a + b*T", "L(LIQUID,AL,ZN;0)": "a + b*T"}


@pytest.fixture(scope="module")
def setup():
    return build()


def build():
    truth = databases.load("alzn_mey")
    dropped = {"G(FCC_A1,AL,ZN;0)", "G(FCC_A1,AL,ZN;1)", "G(LIQUID,AL,ZN;0)"}
    start = dataclasses.replace(truth, parameters=[p for p in truth.parameters if p.label not in dropped])

    def realize(obs):
        """Noise-free values from the assessment; an activity's sigma is 1% of it."""
        out = []
        for o in obs:
            if o.kind == "ZPF":
                out.append(o)
                continue
            v = float(predict(truth, [o], EL)[0])
            out.append(dataclasses.replace(o, value=v, sigma=0.01 * v if o.kind == "ACR" else o.sigma))
        return out

    base = realize(
        [Observation("HM_MIX", 1000.0, 0, 30.0, phase="LIQUID", x=x) for x in (0.3, 0.5, 0.7)]
        + [Observation("SM_MIX", 1000.0, 0, 0.3, phase="LIQUID", x=0.5)]
        + [Observation("HM_MIX", 650.0, 0, 30.0, phase="FCC_A1", x=x) for x in (0.3, 0.6)]
        + [Observation("SM_MIX", 650.0, 0, 0.3, phase="FCC_A1", x=x) for x in (0.3, 0.7)]
    )
    cands = []
    for T in (650.0, 800.0, 1000.0):
        phase = "FCC_A1" if T < 700 else "LIQUID"
        for x in (0.1, 0.3, 0.5, 0.7, 0.9):
            cands.append(Observation("HM_MIX", T, 0, 30.0, phase=phase, x=x))
            cands.append(Observation("SM_MIX", T, 0, 0.3, phase=phase, x=x))
    cands += realize([Observation("ACR", 1000.0, 1.0, 0.01, phase="LIQUID", x=x, element="ZN") for x in (0.2, 0.8)])
    phases = list(truth.phase_models(EL, ["LIQUID", "FCC_A1"]).values())
    r = equilibrium(phases, x=0.35, T=700.0, certify="none")
    cands.append(
        Observation("ZPF", 700.0, 0, 200.0, phases=tuple((cs.phase.name, float(cs.x[0])) for cs in r.composition_sets))
    )
    res = fit(start, Dataset(base), EL, FREE)
    assert res.rank == 6
    return start, base, cands, res, realize


def test_ranking_is_sorted_and_reports_reductions(setup):
    _, _, cands, res, _ = setup
    ranked = rank_candidates(res, cands)
    assert len(ranked) == len(cands)
    assert [r["relative"] for r in ranked] == sorted((r["relative"] for r in ranked), reverse=True)
    assert all(np.all(r["variance_after"] <= r["variance_before"] * (1 + 1e-9)) for r in ranked)
    assert ranked[0]["observation"].kind == "ZPF"  # a tie-line informs every coefficient at once
    assert [J.shape[0] for J in candidate_rows(res, [cands[-1]])] == [2]  # two rows: μ equality for both elements
    one = rank_candidates(res, cands, parameters=["L(LIQUID,AL,ZN;0).b"])
    assert one[0]["variance_before"].shape == (1,)


def test_chosen_measurements_do_what_was_predicted(setup):
    """Refit with the chosen (noise-free) measurements: the variances are exactly the predicted ones."""
    start, base, cands, res, realize = setup
    plan = design_measurements(res, cands, 3, criterion="A")
    assert len(set(plan["chosen"])) == 3
    refit = fit(start, Dataset(base + realize(plan["observations"])), EL, FREE)
    assert refit.std**2 == pytest.approx(plan["variance_after"], rel=1e-6)
    # better than chance: three random candidates leave much more variance
    rng = np.random.default_rng(0)
    random = fit(start, Dataset(base + realize([cands[k] for k in rng.choice(len(cands), 3, replace=False)])), EL, FREE)
    assert np.sum(refit.std**2) < 0.5 * np.sum(random.std**2)
    d = design_measurements(res, cands, 2, criterion="D")
    assert len(d["chosen"]) == 2


def test_undetermined_coefficients_rank_the_determining_measurement_first(setup):
    start, base, cands, _, _ = setup
    with pytest.warns(UserWarning, match="determine only"):
        weak = fit(start, Dataset([o for o in base if o.kind == "HM_MIX"]), EL, FREE)  # no entropy data
    # H does not depend on b in "a + b*T": for the liquid's b, only entropy, activity or tie-line data can help
    ranked = rank_candidates(weak, cands, parameters=["L(LIQUID,AL,ZN;0).b"])
    assert ranked[0]["observation"].kind in ("SM_MIX", "ZPF", "ACR") and ranked[0]["relative"] > 0.99
    assert all(r["relative"] < 1e-6 for r in ranked if r["observation"].kind == "HM_MIX")


def test_exchange_method(setup):
    _, _, cands, res, _ = setup
    with pytest.raises(ValueError, match="one row each"):
        design_measurements(res, cands, 2, method="exchange")
    pytest.importorskip("discopt.doe")
    singles = [c for c in cands if c.kind != "ZPF"]
    plan = design_measurements(res, singles, 3, method="exchange", criterion="D")
    assert len(plan["chosen"]) == 3
