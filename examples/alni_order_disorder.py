"""Al–Ni with its order–disorder phases: γ′ (L1₂) and β (B2, with vacancies), issue #6.

Dupin, Ansara and Sundman's assessment describes two ordered phases as
partitioned phases over a disordered parent:

- γ′ (Ni₃Al, L1₂): FCC_L12 over FCC_A1 (γ);
- β (NiAl, B2): BCC_B2 over BCC_A2, with constitutional vacancies on both
  sublattices.

Both have G = G_dis(x) + G_ord(y) − G_ord(x). y are the ordered site fractions
and x their site average. With equal fractions on every sublattice, the
ordering term vanishes and the phase is its parent exactly, so order appears as
a second well in site-fraction space. A grid over x alone cannot see it.
Pricing over y finds it as a new composition set of the same phase.

β can hold a vacancy on every sublattice. Read literally, the model then has
no global minimum: a dilute fraction of atoms in an empty lattice gains
unbounded entropy per atom. Such a phase is restricted to formula units at
least half occupied (``min_occupancy``), far beyond the few percent of
constitutional vacancies in Al-rich β. Certificates say so.

This script

1. maps the diagram (900–1700 K) with local pricing;
2. maps it with pycalphad, on the same database, and compares the phase
   boundaries;
3. plots the diagram next to β's vacancy fraction across its single-phase
   range at 1200 K. On the Al-rich side of NiAl, β takes up the excess Al by
   leaving Ni sites empty.

The disordered state of a partitioned phase has its parent's G exactly.
pycalphad reports such a set under the ordered name, so both maps name a set by
its degree of order, max |y_s − x|.

Reference: N. Dupin, I. Ansara and B. Sundman, *Thermodynamic re-assessment of
the ternary system Al–Cr–Ni*, Calphad 25 (2001) 279–298 (registry name
``alni_dupin``).

Run with ``uv run python examples/alni_order_disorder.py``. It takes about 30
minutes and writes examples/alni_order_disorder.png.
"""

import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import databases, equilibrium
from discopt.calphad.equilibrium.certificate import CertificateLevel
from discopt.calphad.models.partitioned import PartitionedPhase

HERE = Path(__file__).parent
EL = ("AL", "NI")
TEMPS = np.arange(900.0, 1701.0, 100.0)
XS = np.linspace(0.02, 0.98, 49)  # x(Al)
ORDERED = 1e-3  # a set with max |y_s − x| above this is ordered

db = databases.load("alni_dupin")
models = db.phase_models(EL)
phases = list(models.values())
print("phases:", sorted(models), "| skipped:", sorted(set(db.applicability(EL)) - set(models)))


def label(name, order):
    if name in ("FCC_L12", "FCC_A1"):
        return "γ′" if order > ORDERED else "γ"
    if name in ("BCC_B2", "BCC_A2"):
        return "β" if order > ORDERED else "A2"
    return {"LIQUID": "L", "AL3NI5": "Al3Ni5", "AL3NI2": "Al3Ni2", "AL3NI1": "Al3Ni"}.get(name, name)


def two_phase(sets):
    sets = sorted(sets)
    return (tuple(n for n, _ in sets), sets) if len(sets) == 2 else None


def discopt_map():
    out, failed = defaultdict(list), 0
    for T in TEMPS:
        for x in XS:
            r = equilibrium(phases, x=float(x), T=float(T), certify="none")
            if r.certificate.level < CertificateLevel.STABILITY_SEARCHED:
                failed += 1
                continue
            sets = []
            for cs in r.composition_sets:
                if cs.amount <= 1e-8:
                    continue
                order = cs.phase.degree_of_order(cs.y) if isinstance(cs.phase, PartitionedPhase) else 0.0
                sets.append((label(cs.phase.name, order), round(float(cs.x[0]), 6)))
            tp = two_phase(sets)
            if tp:
                out[(float(T), tp[0])].append(tp[1])
    return out, failed


def pycalphad_map():
    from pycalphad import Database
    from pycalphad import equilibrium as peq
    from pycalphad import variables as v

    pdb = Database(str(databases.fetch("alni_dupin")))
    out = defaultdict(list)
    for T in TEMPS:
        eq = peq(pdb, [*EL, "VA"], sorted(models), {v.X("AL"): XS, v.T: float(T), v.P: 101325, v.N: 1})
        ph, xs, ys = eq.Phase.values.squeeze(), eq.X.values.squeeze(), eq.Y.values.squeeze()
        for i in range(len(XS)):
            sets = []
            for k, name in enumerate(ph[i]):
                if not name:
                    continue
                y = ys[i][k]
                order = (
                    abs(y[0] - y[2])
                    if name == "FCC_L12"
                    else np.max(np.abs(y[0:3] - y[3:6]))
                    if name == "BCC_B2"
                    else 0.0
                )
                sets.append((label(str(name), order), round(float(xs[i][k][0]), 6)))
            tp = two_phase(sets)
            if tp:
                out[(float(T), tp[0])].append(tp[1])
    return out


t = time.time()
ours, failed = discopt_map()
print(f"discopt-calphad: {time.time() - t:.0f} s, {failed} of {len(TEMPS) * len(XS)} points failed")
t = time.time()
ref = pycalphad_map()
print(f"pycalphad: {time.time() - t:.0f} s")

print("\nphase boundaries in both maps (mean over the tie-lines of a region at that T):")
diffs = []
for key in sorted(set(ours) & set(ref)):
    a = np.mean([[x for _, x in s] for s in ours[key]], axis=0)
    b = np.mean([[x for _, x in s] for s in ref[key]], axis=0)
    diffs.append(float(np.max(np.abs(a - b))))
    print(f"  {key[0]:6.0f} K {' + '.join(key[1]):14s} ours {a.round(4)}  pycalphad {b.round(4)}  |Δx| {diffs[-1]:.5f}")
print(f"largest boundary difference: {max(diffs):.5f} in x(Al)")
print("regions only in discopt-calphad:", sorted(set(ours) - set(ref)))
print("regions only in pycalphad:", sorted(set(ref) - set(ours)))

# vacancies in β across its single-phase range at 1200 K
T_VAC = 1200.0
beta = models["BCC_B2"]
xv, vac = [], []
for x in np.linspace(0.36, 0.58, 23):
    r = equilibrium(phases, x=float(x), T=T_VAC, certify="none")
    for cs in r.composition_sets:
        if cs.phase is beta and cs.amount > 0.99 * sum(c.amount for c in r.composition_sets):
            sf = beta.site_fractions(cs.y)
            xv.append(float(cs.x[0]))
            vac.append(0.5 * (sf[0]["VA"] + sf[1]["VA"]))
print(f"β at {T_VAC:.0f} K: vacancy fraction {min(vac):.2e} to {max(vac):.2e} for x(Al) {min(xv):.3f} to {max(xv):.3f}")

# ------------------------------------------------------------------ figure
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [3, 2]})
colors = {"γ": "C0", "γ′": "C2", "β": "C1", "L": "C3", "Al3Ni5": "C4", "Al3Ni2": "C5", "Al3Ni": "C6", "A2": "C7"}
for tl, style, name in [(ref, dict(color="0.65", ms=7), "pycalphad"), (ours, dict(ms=3), "discopt-calphad")]:
    for (T, _), sets in tl.items():
        for s in sets:
            for n, x in s:
                kw = dict(style)
                kw.setdefault("color", colors.get(n, "k"))
                ax.plot([x], [T], "o", label=name, **kw)
                name = None
for n, c in colors.items():
    ax.plot([], [], "o", color=c, label=n)
ax.set_xlabel("x(Al)")
ax.set_ylabel("T (K)")
ax.set_xlim(0, 1)
ax.set_title("Al–Ni (Dupin et al. 2001): tie-line ends", fontsize=10)
ax.legend(fontsize=7, loc="upper right", ncol=2)
ax2.semilogy(xv, vac, "C1o-", ms=3)
ax2.set_xlabel("x(Al)")
ax2.set_ylabel("vacancy site fraction in β")
ax2.set_title(f"β (B2) constitutional vacancies at {T_VAC:.0f} K", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "alni_order_disorder.png", dpi=150)
print("wrote", HERE / "alni_order_disorder.png")
