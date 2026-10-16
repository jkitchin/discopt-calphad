"""The ZrO₂–Nd₂O₃ section, with charged sublattice phases (issue #6).

Fabrichnaya and Seifert describe the oxide solid solutions with charged species
in the compound energy formalism:

- fluorite (cubic ZrO₂), tetragonal and monoclinic ZrO₂: (Nd³⁺, Zr⁴⁺)₂(O²⁻, Va)₄;
- pyrochlore Nd₂Zr₂O₇: five sublattices, with vacancies on the oxygen sites;
- the A, H and X forms of Nd₂O₃: (Nd³⁺, Zr⁴⁺)₂(O²⁻)₃(O²⁻, Va);
- the liquid: an ionic liquid (Nd³⁺, Zr⁴⁺)(O²⁻).

A configuration is physical only if it is neutral, Σ_s a_s Σ_c q_c y_s,c = 0. That
is an extra equality on each charged phase's site fractions: in the restricted
solve, in every pricing problem, and in the boundary-slab bound, which drops boxes
whose charge interval excludes 0. With fixed valences (Nd³⁺, Zr⁴⁺, O²⁻) and a
neutral overall composition, the phases' neutrality rows sum to what the element
balance already fixes, so the restricted solve drops one of them.

The section is the line from ZrO₂ to NdO₁.₅: per mole of cations, Nd = a,
Zr = 1 − a, O = 1.5a + 2(1 − a). This script maps it (1200–2800 K) with local
pricing, maps it with pycalphad on the same database, compares both maps, and
plots the result.

Reference: O. Fabrichnaya and H. J. Seifert, *Assessment of thermodynamic
functions in the ZrO₂–Nd₂O₃–Al₂O₃ system*, Calphad 32 (2008) 142–151 (TDB file by
T. Abe, K. Hashimoto and Y. Sawada, NIMS, 2013).

Run with ``uv run python examples/zro2_nd2o3.py`` (about 10 minutes). It writes
examples/zro2_nd2o3.png.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import ThermoDatabase, equilibrium
from discopt.calphad.equilibrium.certificate import CertificateLevel

HERE = Path(__file__).parent
EL = ("ND", "ZR", "O")
TEMPS = np.arange(1200.0, 2801.0, 100.0)
AS = np.linspace(0.02, 0.98, 25)  # cation fraction of Nd


def tdb_path():
    import pycalphad

    return Path(pycalphad.__file__).parent / "tests" / "databases" / "al2o3_nd2o3_zro2.tdb"


db = ThermoDatabase.from_tdb(tdb_path())
models = db.phase_models(EL)
phases = list(models.values())
print("phases:", sorted(models))
NAMES = {"FLUO": "F", "TETR": "T", "MONO": "M", "PYRO": "P", "ND2O3_A": "A", "ND2O3_H": "H", "ND2O3_X": "X", "I_LIQUID": "L"}


def b_of(a):
    return np.array([a, 1 - a, 1.5 * a + 2 * (1 - a)])


def two_phase(sets):
    sets = sorted(sets)
    return (tuple(n for n, _ in sets), sets) if len(sets) == 2 else None


t = time.time()
ours, failed = {}, 0
for T in TEMPS:
    for a in AS:
        r = equilibrium(phases, b=b_of(a), T=float(T), certify="none")
        if r.certificate.level < CertificateLevel.STABILITY_SEARCHED:
            failed += 1
            continue
        sets = [
            (NAMES[cs.phase.name], round(float(cs.x[0] / (cs.x[0] + cs.x[1])), 4))
            for cs in r.composition_sets
            if cs.amount > 1e-8
        ]
        ours[(float(T), float(a))] = sorted(sets)
print(f"discopt-calphad: {time.time() - t:.0f} s, {failed} of {len(TEMPS) * len(AS)} points failed")

from pycalphad import Database  # noqa: E402
from pycalphad import equilibrium as peq  # noqa: E402
from pycalphad import variables as v  # noqa: E402

pdb = Database(str(tdb_path()))
t = time.time()
ref = {}
for T in TEMPS:
    for a in AS:
        x = b_of(a) / b_of(a).sum()
        eq = peq(pdb, [*EL, "VA"], sorted(models), {v.X("ND"): x[0], v.X("ZR"): x[1], v.T: float(T), v.P: 101325, v.N: 1})
        names = [str(p) for p in np.squeeze(eq.Phase.values) if p]
        X = np.atleast_2d(np.squeeze(eq.X.values))[: len(names)]  # components sorted: ND, O, ZR
        ref[(float(T), float(a))] = sorted((NAMES[n], round(float(c[0] / (c[0] + c[2])), 4)) for n, c in zip(names, X))
print(f"pycalphad: {time.time() - t:.0f} s")

same = 0
for key, sets in ours.items():
    rs = ref[key]
    if [n for n, _ in sets] == [n for n, _ in rs] and np.allclose([c for _, c in sets], [c for _, c in rs], atol=2e-3):
        same += 1
    else:
        print(f"  {key[0]:6.0f} K a={key[1]:.2f}: ours {sets}  pycalphad {rs}")
print(f"same phases and compositions at {same} of {len(ours)} points")

fig, ax = plt.subplots(figsize=(7.5, 5))
colors = {"F": "C0", "T": "C1", "M": "C2", "P": "C3", "A": "C4", "H": "C5", "X": "C6", "L": "C7"}
for tl, style in [(ref, dict(color="0.75", ms=7)), (ours, dict(ms=3))]:
    for (T, _), sets in tl.items():
        if len(sets) == 2:
            for n, c in sets:
                kw = dict(style)
                kw.setdefault("color", colors[n])
                ax.plot([c], [T], "o", **kw)
ax.plot([], [], "o", color="0.75", label="pycalphad")
labels = {"F": "fluorite", "T": "tetragonal", "M": "monoclinic", "P": "pyrochlore", "A": "Nd2O3 A", "H": "Nd2O3 H",
          "X": "Nd2O3 X", "L": "liquid"}
for n, c in colors.items():
    ax.plot([], [], "o", color=c, ms=3, label=labels[n])
ax.set_xlabel("Nd / (Nd + Zr)")
ax.set_ylabel("T (K)")
ax.set_xlim(0, 1)
ax.set_title("ZrO2–NdO1.5 (Fabrichnaya and Seifert 2008): tie-line ends", fontsize=10)
ax.legend(fontsize=7, loc="lower center", ncol=3)
fig.tight_layout()
fig.savefig(HERE / "zro2_nd2o3.png", dpi=150)
print("wrote", HERE / "zro2_nd2o3.png")
