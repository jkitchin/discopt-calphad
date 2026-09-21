"""Fit Al–Zn interaction parameters to data (issue #2).

The "experimental" data here are synthetic: mixing enthalpies, Zn activities and
one liquid/FCC tie-line, generated from the Mey assessment with noise added.
Starting from a database with those parameters set to zero, the fit recovers
them within their uncertainties. It then checks the fitted database with a
certified equilibrium at the tie-line. Writes examples/alzn_fit.png.
"""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import R_GAS, databases
from discopt.calphad.fit import Dataset, _LinearModel, fit

HERE = Path(__file__).parent
EL = ("AL", "ZN")
FREE = {
    "L(FCC_A1,AL,ZN;0)": "a + b*T",
    "L(FCC_A1,AL,ZN;1)": "a + b*T",
    "L(LIQUID,AL,ZN;0)": "a + b*T",
}

truth = databases.load("alzn_mey")
rng = np.random.default_rng(1)
lm = _LinearModel(truth, EL, [])  # evaluates the true model
data = Dataset()
for kind, phase, T, sigma in [
    ("HM_MIX", "LIQUID", 1000.0, 30.0),
    ("HM_MIX", "FCC_A1", 650.0, 30.0),
    ("SM_MIX", "FCC_A1", 650.0, 0.05),
]:
    x = np.linspace(0.1, 0.9, 9)
    v = [lm.phase(phase).thermo(kind, xx, T)[0] for xx in x]
    data.add_thermo(kind, phase, T, x, np.array(v) + rng.normal(0, sigma, x.size), sigma, citation="synthetic")
liq = lm.phase("LIQUID")
for x in (0.2, 0.4, 0.6, 0.8):
    a = np.exp((liq.mu(x, 1000.0)[1, 0] - liq.pure_mu("ZN", 1000.0)[0]) / (R_GAS * 1000.0))
    data.add_activity("ZN", "LIQUID", 1000.0, x, a * (1 + rng.normal(0, 0.02)), 0.02 * a, citation="synthetic")
data.add_equilibrium(700.0, {"LIQUID": 0.2119, "FCC_A1": 0.4983}, sigma=200.0, citation="synthetic")
data.to_json(HERE / "data" / "alzn_synthetic.json")

# start from a database without the fitted parameters: the fit adds them back
dropped = {"G(FCC_A1,AL,ZN;0)", "G(FCC_A1,AL,ZN;1)", "G(LIQUID,AL,ZN;0)"}
start = dataclasses.replace(truth, parameters=[p for p in truth.parameters if p.label not in dropped])
res = fit(start, data, EL, FREE)
print(res.summary())
print("\ncorrelation\n", np.array2string(res.correlation, precision=2, suppress_small=True))

for check in res.check_equilibria():
    print(
        f"\nT = {check['T']} K, x_AL = {check['x']:.3f}: observed {check['observed']}, "
        f"computed {check['computed']} ({check['certificate']})"
    )

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fit_lm = _LinearModel(res.database, EL, [])
xs = np.linspace(0.02, 0.98, 97)
for (kind, phase, T), color in zip([("HM_MIX", "LIQUID", 1000.0), ("HM_MIX", "FCC_A1", 650.0)], ("C0", "C1")):
    obs = [o for o in data if o.kind == kind and o.phase == phase]
    ax1.errorbar(
        [o.x for o in obs],
        [o.value for o in obs],
        [o.sigma for o in obs],
        fmt="o",
        color=color,
        label=f"{phase} {T:g} K data",
    )
    ax1.plot(xs, [fit_lm.phase(phase).thermo(kind, x, T)[0] for x in xs], color=color, label="fit")
ax1.set_xlabel("x(AL)")
ax1.set_ylabel("ΔH_mix (J/mol)")
ax1.legend(fontsize=8)
res.plot_residuals(ax2)
fig.tight_layout()
fig.savefig(HERE / "alzn_fit.png", dpi=150)
print("wrote", HERE / "alzn_fit.png")
