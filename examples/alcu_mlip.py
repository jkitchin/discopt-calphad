"""Al–Cu parameters from a machine-learned potential (issue #5, phase 1).

A calculator produces data; the fit turns the data into CALPHAD parameters:

1. Al and Cu are relaxed in their reference (FCC) structures.
2. θ-Al2Cu (C16) is relaxed. Its formation enthalpy is taken relative to those references.
3. FCC mixing enthalpies come from two SQS per composition (icet), relaxed.
   Their spread is σ.
4. The values are compared with the Zhang assessment (``alcuy_zhang``), in kJ/mol.
5. The FCC interaction parameters L0, L1 and L2 and the θ formation constant are refitted to the
   calculated data alone. Only the enthalpy part is fitted: a 0 K energy says
   nothing about entropy, so each form keeps the assessment's T coefficient
   (L0 = a + 2T, L1 = a − 2T, θ: a + 6.75T + 2 GHSERAL + GHSERCU).
6. The Al-rich side of the T–x diagram is mapped for the fitted and the assessed
   database.

The certificate of any equilibrium computed from the fit covers the fitted
model only. The calculator's error enters only through σ; the comparison in
step 4 is the measure of it.

Run with UMA (fairchem, weights from Hugging Face):

    PYTHONPATH=src python examples/alcu_mlip.py --model uma

or with EMT, which needs nothing beyond ASE (and is a poor model of Al–Cu):

    uv run python examples/alcu_mlip.py --model emt

Relaxations and SQS are cached in the user cache, so a second run is fast.
Writes examples/alcu_mlip_<model>.png.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import databases, equilibrium, mlip
from discopt.calphad.equilibrium.certificate import CertificateLevel
from discopt.calphad.fit import _LinearModel, fit

HERE = Path(__file__).parent
EL = ("AL", "CU")
T0 = 298.15
GRID = np.linspace(0.1, 0.9, 9)  # site fraction of Al on the FCC lattice
FREE = {
    "L(FCC_A1,AL,CU:VA;0)": "a + 2*T",
    "L(FCC_A1,AL,CU:VA;1)": "a - 2*T",
    "L(FCC_A1,AL,CU:VA;2)": "a",
    "G(ALCU_THETA,AL:CU;0)": "a + 6.75*T + 2*GHSERAL + GHSERCU",
}


def calculator(name):
    if name == "emt":
        from ase.calculators.emt import EMT

        return EMT(), "EMT"
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cpu")
    return FAIRChemCalculator(predictor, task_name="omat"), "uma-s-1p1/omat"


def tie_lines(db, temps, xs):
    """Two-phase equilibria on a T–x grid, each tie-line kept once (local solves, as in alzn_diagram_fit.py)."""
    phases = list(db.phase_models(EL).values())
    out = {}
    for T in temps:
        found = {}
        for x in xs:
            res = equilibrium(phases, x=float(x), T=float(T), certify="none")
            if res.certificate.level < CertificateLevel.STABILITY_SEARCHED:
                continue
            sets = sorted((cs.phase.name, float(cs.x[0])) for cs in res.composition_sets)
            if len(sets) == 2 and abs(sets[0][1] - sets[1][1]) > 1e-3:
                found.setdefault(tuple((n, round(v, 3)) for n, v in sets), tuple(sets))
        out[float(T)] = list(found.values())
    return out


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=("uma", "emt"), default="uma")
args = parser.parse_args()

calc, label = calculator(args.model)
model = mlip.from_ase(calc, name=label, cache=True)
assessed = databases.load("alcuy_zhang")

t = time.time()
result = mlip.compute(
    model,
    assessed,
    EL,
    compounds={"ALCU_THETA": ["AL:CU"]},
    structures={"ALCU_THETA": ("C16", ["8h"], ["4a"])},
    mixing={"FCC_A1": GRID},
    # the calculator's own error, in kJ/mol: the low end of the 1-3 kJ/mol typical of
    # universal MLIPs. θ comes from one structure, so this is its whole σ; the SQS
    # values add their spread, pooled over the FCC compositions.
    model_sigma=2.0,
    sqs=mlip.SQSOptions(count=2, max_atoms=32, n_steps=3000),
)
print(f"{label}: {len(result.relaxations)} relaxations in {time.time() - t:.0f} s\n")
print(result.table())

# ---- the calculator against the assessment, kJ/mol
ref = _LinearModel(assessed, EL, [])
print(f"\n{'':22s} {label:>16s} {'assessment':>11s} {'error':>8s}   (kJ/mol)")
errors = []
for o in result.dataset:
    ph = ref.phase(o.phase)
    y = ph.y_from_site_fractions(o.site_fractions)
    a = (ph.formation(y, T0) if o.kind == "HM_FORM" else ph.thermo(o.kind, None, T0, y))[0] / 1000
    what = f"{o.kind} {o.phase} x={o.x:.2f}" if o.kind == "HM_MIX" else f"{o.kind} {o.phase}"
    errors.append(o.value / 1000 - a)
    print(f"{what:22s} {o.value / 1000:16.2f} {a:11.2f} {errors[-1]:8.2f}")
print(f"RMS error {np.sqrt(np.mean(np.square(errors))):.2f} kJ/mol, mean {np.mean(errors):.2f} kJ/mol")

# ---- fit the enthalpy parts to the calculated data alone
dropped = {"G" + k[1:] if k.startswith("L") else k for k in FREE}
start = dataclasses.replace(assessed, parameters=[p for p in assessed.parameters if p.label not in dropped])
fitted = fit(start, result.dataset, EL, FREE)
print("\n" + fitted.summary())
print("assessment:", {p.label: str(p.function.exprs[0]) for p in assessed.parameters if p.label in dropped})

# ---- figure: data against the assessment, and the Al-rich diagram
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
xs = np.linspace(0.02, 0.98, 97)
fit_lm = _LinearModel(fitted.database, EL, [])
ax1.plot(xs, [ref.phase("FCC_A1").thermo("HM_MIX", x, T0)[0] / 1000 for x in xs], color="0.5", lw=3, label="assessment")
ax1.plot(xs, [fit_lm.phase("FCC_A1").thermo("HM_MIX", x, T0)[0] / 1000 for x in xs], color="C3", label="fit")
mix = [o for o in result.dataset if o.kind == "HM_MIX"]
ax1.errorbar(
    [o.x for o in mix],
    [o.value / 1000 for o in mix],
    [2 * o.sigma / 1000 for o in mix],
    fmt="o",
    color="k",
    mfc="none",
    label=f"{label} SQS (±2σ)",
)
theta = next(o for o in result.dataset if o.kind == "HM_FORM")
th_ref = ref.phase("ALCU_THETA")
ax1.plot(
    [2 / 3],
    [th_ref.formation(th_ref.y_from_site_fractions(theta.site_fractions), T0)[0] / 1000],
    "s",
    color="0.5",
    ms=9,
    label="θ assessment",
)
ax1.plot([2 / 3], [theta.value / 1000], "D", color="k", mfc="none", ms=8, label=f"θ {label}")
ax1.axhline(0, color="k", lw=0.6)
ax1.set_xlabel("x(AL)")
ax1.set_ylabel("ΔH at 298 K (kJ/mol of atoms)")
ax1.set_title("FCC mixing and θ formation enthalpies", fontsize=10)
ax1.legend(fontsize=8)

temps = np.arange(500.0, 951.0, 30.0)
xgrid = np.linspace(0.6, 0.995, 17)
for db, style, name in [
    (assessed, dict(color="0.6", ms=5), "assessment"),
    (fitted.database, dict(color="C3", ms=2.5), f"fit to {label}"),
]:
    t = time.time()
    for T, lines in tie_lines(db, temps, xgrid).items():
        for (_, xa), (_, xb) in lines:
            ax2.plot([xa, xb], [T, T], "o", label=name, **style)
            name = None
    print(f"diagram: {time.time() - t:.0f} s")
ax2.set_xlim(0.6, 1.0)
ax2.set_xlabel("x(AL)")
ax2.set_ylabel("T (K)")
ax2.set_title("Al-rich side: two-phase tie-line ends", fontsize=10)
ax2.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(HERE / f"alcu_mlip_{args.model}.png", dpi=150)
print("wrote", HERE / f"alcu_mlip_{args.model}.png")
