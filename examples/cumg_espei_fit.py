"""Fit Cu–Mg parameters to real data, and compare with the published assessment (issue #2).

The data are the Cu–Mg datasets curated for ESPEI (PhasesResearchLab/ESPEI-datasets,
CC-BY 4.0, vendored in examples/data/espei-cu-mg). They are:

- liquid mixing enthalpies (Batalin 1987, Sommer 1983);
- liquid Mg activities (Garg 1973);
- formation enthalpies of Cu2Mg and CuMg2, from calorimetry (King 1964,
  Feufel 1995) and DFT;
- DFT mixing enthalpies of FCC;
- 238 phase-boundary points from 1907–1978. Every one of them gives only one
  phase's composition.

The fit starts from the NIMS assessment (``cumg_nims``) and refits ten
coefficients:

- the liquid L0 and L1;
- the FCC L0;
- the Cu2Mg CU:MG endmember (enthalpy and entropy; the heat-capacity terms are
  kept);
- the CuMg2 compound.

A phase-boundary point enters as a driving force: the phase whose composition
was not measured must touch the tangent of the one that was. Cu2Mg is a
two-sublattice Laves phase, so its chemical potentials come from its internal
equilibrium. Both depend on the parameters, and the fit iterates linear solves.
σ is ESPEI's default for each kind of data.

Writes examples/cumg_espei_fit.png: the T–x diagram of the fit and of the
assessment over the data, and the liquid mixing enthalpy.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import databases, equilibrium
from discopt.calphad.equilibrium.certificate import CertificateLevel
from discopt.calphad.fit import Dataset, FreeParameter, _LinearModel, fit

HERE = Path(__file__).parent
EL = ("CU", "MG")
CP = "- 69.276417*T*ln(T) - 0.00051925*T**2 - 5.65953e-6*T**3 + 143502/T"  # the assessment's Cu2Mg heat capacity
FREE = {
    "L(LIQUID,CU,MG;0)": "a + b*T",
    "L(LIQUID,CU,MG;1)": "a + b*T",
    "L(FCC_A1,CU,MG:VA;0)": "a + b*T",
    "G(CU2MG,CU:MG;0)": "a + b*T " + CP,
    "G(CUMG2,CU:MG;0)": "a + b*T + GHSERCU + 2*GHSERMG",
}

assessed = databases.load("cumg_nims")
data = Dataset.from_espei(HERE / "data" / "espei-cu-mg", assessed, EL, phase_map={"LAVES_C15": "CU2MG"})
kinds = {k: sum(o.kind == k for o in data) for k in ("ZPF", "HM_MIX", "HM_FORM", "ACR")}
print(f"{len(data)} observations {kinds}; not used:")
for name, reason in sorted(set(data.skipped)):
    print(f"  {name}: {reason}")

t = time.time()
res = fit(assessed, data, EL, FREE, verbose=True)
print(f"\n{res.summary()}\n({time.time() - t:.0f} s)")

# the assessment's own values of the same coefficients (its functions have these forms)
start = _LinearModel(assessed, EL, [FreeParameter.parse(k, v, functions=assessed.functions) for k, v in FREE.items()])
print(f"\n{'coefficient':28s} {'fit':>12s} {'± σ':>9s} {'assessment':>12s}")
for name, v, s, a in zip(res.names, res.values, res.std, start.initial_theta()):
    print(f"{name:28s} {v:12.2f} {s:9.2f} {a:12.2f}")
print(f"chi2/dof = {res.chi2 / res.dof:.2f}: σ, ESPEI's defaults, are smaller than the scatter between sources")


def tie_lines(db, temps, xs):
    """Two-phase equilibria on a T–x grid, each tie-line kept once (local solves)."""
    phases = list(db.phase_models(EL).values())
    out = []
    for T in temps:
        found = {}
        for x in xs:
            r = equilibrium(phases, x=float(x), T=float(T), certify="none")
            if r.certificate.level < CertificateLevel.STABILITY_SEARCHED:
                continue
            sets = sorted((cs.phase.name, float(cs.x[0])) for cs in r.composition_sets)
            if len(sets) == 2 and abs(sets[0][1] - sets[1][1]) > 1e-3:
                found.setdefault(tuple((n, round(v, 3)) for n, v in sets), sets)
        out += [(float(T), s) for s in found.values()]
    return out


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
temps, xs = np.arange(650.0, 1381.0, 30.0), np.linspace(0.02, 0.98, 33)
for db, style, label in [
    (assessed, dict(color="0.6", ms=5), "assessment (NIMS)"),
    (res.database, dict(color="C3", ms=2.5), "fit"),
]:
    t = time.time()
    for T, sets in tie_lines(db, temps, xs):
        ax1.plot([1 - x for _, x in sets], [T, T], "o", label=label, **style)
        label = None
    print(f"diagram: {time.time() - t:.0f} s")
label = "boundary data (measured side)"
for o in data:
    if o.kind == "ZPF":
        for _, x in o.phases:
            if x is not None:
                ax1.plot(1 - x, o.T, "x", color="k", ms=4, mew=0.8, label=label)
                label = None
ax1.set_xlim(0, 1)
ax1.set_xlabel("x(MG)")
ax1.set_ylabel("T (K)")
ax1.set_title("Cu–Mg: tie-line ends and phase-boundary data", fontsize=10)
ax1.legend(fontsize=8, loc="lower center")

liq = [o for o in data if o.kind == "HM_MIX" and o.phase == "LIQUID"]
xx = np.linspace(0.02, 0.98, 97)
fit_lm, ref_lm = _LinearModel(res.database, EL, []), _LinearModel(assessed, EL, [])
for lm, style, label in [(ref_lm, dict(color="0.6", lw=3), "assessment"), (fit_lm, dict(color="C3"), "fit")]:
    ax2.plot(1 - xx, [lm.phase("LIQUID").thermo("HM_MIX", x, 1100.0)[0] / 1000 for x in xx], label=label, **style)
for src in sorted({o.source for o in liq}):
    pts = [o for o in liq if o.source == src]
    ax2.plot(
        [1 - o.x if o.x is not None else 1 - o.site_fractions[0]["CU"] for o in pts],
        [o.value / 1000 for o in pts],
        "o",
        mfc="none",
        label=f"{src} ({pts[0].T:g} K)",
    )
ax2.set_xlabel("x(MG)")
ax2.set_ylabel("ΔH_mix of the liquid (kJ/mol of atoms)")
ax2.set_title("liquid mixing enthalpy (curves at 1100 K)", fontsize=10)
ax2.legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "cumg_espei_fit.png", dpi=150)
print("wrote", HERE / "cumg_espei_fit.png")
