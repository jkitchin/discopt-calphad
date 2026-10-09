"""Errors in variables: are the fitted uncertainties honest? (issue #8)

Exact Al–Zn tie-lines from the Mey assessment (the FCC miscibility gap and
LIQUID + FCC) get composition noise σ_x = 0.005, many times over. Each noisy
dataset is fitted twice for the constants of the FCC and liquid L0:

- the ordinary fit, with σ_x converted to J/mol through the dilute-side
  ideal-solution slope RT·σ_x/(x(1−x)), as examples/alzn_diagram_fit.py does;
- the errors-in-variables fit (``errors_in_variables=True``), with the
  compositions as unknowns.

An honest fit has a mean χ²/dof near 1, an rms z-score near 1, and the truth
within 2σ about 95% of the time. Measured with 40 seeds: errors in variables
gave 0.94, 0.85 and 97%; the ordinary fit gave 0.04, 0.29 and 100%, so its σ
are about 3.5 times too large.

    uv run python benchmarks/eiv_coverage.py [number of seeds, default 20]
"""

import sys
import time
import warnings

import numpy as np

from discopt.calphad import R_GAS, databases, equilibrium
from discopt.calphad.fit import Dataset, fit

warnings.simplefilter("ignore")
db = databases.load("alzn_mey")
EL = ("AL", "ZN")
phases = list(db.phase_models(EL, ["LIQUID", "FCC_A1"]).values())
lines = []
for T, xs in [
    (560.0, [0.6]),
    (580.0, [0.6]),
    (600.0, [0.62]),
    (620.0, [0.65]),
    (700.0, [0.3, 0.45]),
    (750.0, [0.4, 0.55]),
    (800.0, [0.55, 0.7]),
]:
    for x in xs:
        r = equilibrium(phases, x=x, T=T, certify="none")
        sets = [(cs.phase.name, float(cs.x[0])) for cs in r.composition_sets]
        if len(sets) == 2 and not any(T == t and sorted(sets) == sorted(s) for t, s in lines):
            lines.append((T, sets))
print(len(lines), "tie-lines:", [(T, [(n, round(x, 3)) for n, x in s]) for T, s in lines])
TRUE = {"L(FCC_A1,AL,ZN;0).a": 7297.5, "L(LIQUID,AL,ZN;0).a": 10465.5}
FREE = {"L(FCC_A1,AL,ZN;0)": "a + 0.47512*T", "L(LIQUID,AL,ZN;0)": "a - 3.39259*T"}
SX = 0.005
res = {"ordinary": [], "eiv": []}
t0 = time.time()
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 20):
    rng = np.random.default_rng(seed)
    ord_ds, eiv_ds = Dataset(), Dataset()
    for T, sets in lines:
        noisy = [(n, float(np.clip(x + rng.normal(0, SX), 1e-4, 1 - 1e-4))) for n, x in sets]
        dilute = min(x * (1 - x) for _, x in noisy)
        ord_ds.add_equilibrium(T, noisy, sigma=R_GAS * T * SX / dilute)
        eiv_ds.add_equilibrium(T, noisy, sigma=1.0, sigma_x=SX)
    for name, ds, kw in [("ordinary", ord_ds, {}), ("eiv", eiv_ds, {"errors_in_variables": True})]:
        f = fit(db, ds, EL, FREE, **kw)
        z = [(f.parameters[k] - v) / f.std[f.names.index(k)] for k, v in TRUE.items()]
        res[name].append((f.chi2 / f.dof, z))
for name, rows in res.items():
    chi = np.array([c for c, _ in rows])
    z = np.array([zz for _, zz in rows])
    print(
        f"{name:9s} mean chi2/dof {chi.mean():.2f}  within 2σ {np.mean(np.abs(z) < 2):.2f}"
        f"  rms z {np.sqrt(np.mean(z**2)):.2f}"
    )
print(f"{time.time() - t0:.0f}s")
