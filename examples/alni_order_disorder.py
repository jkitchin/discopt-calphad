"""Ni-rich Al–Ni: γ and γ′ with the native order–disorder model (issue #6).

γ′ (Ni₃Al, L1₂) is FCC_L12 in Dupin, Ansara and Sundman's assessment, a
partitioned phase over the disordered FCC_A1 (γ):

    G = G_dis(x) + G_ord(y) − G_ord(x)

y are the site fractions on the 0.75 and 0.25 sublattices, and x is their
average. With equal fractions on both, the ordering term vanishes and FCC_L12
is γ exactly, so ordering appears as a second well in site-fraction space. A
grid over x alone cannot see it. Pricing over y finds it as a new composition
set of the same phase.

This script

1. maps the Ni-rich side (x(Al) ≤ 0.30, 900–1700 K) with local pricing;
2. maps it with pycalphad, on the same database and phase list, and compares
   the phase boundaries;
3. plots the diagram next to the degree of order of γ′ along its composition
   range at 1200 K.

The disordered state of FCC_L12 has FCC_A1's G exactly. pycalphad reports such
a set as FCC_L12, so both maps name a set by its degree of order, max |y_s − x|.

BCC_A2 and BCC_B2 allow vacancies on their substitutional sublattices, which
the solver does not support yet. They are left out of both maps. In the full
assessment, β (B2) bounds γ′ on the Al-rich side, above x(Al) ≈ 0.27.

Reference: N. Dupin, I. Ansara and B. Sundman, *Thermodynamic re-assessment of
the ternary system Al–Cr–Ni*, Calphad 25 (2001) 279–298 (registry name
``alni_dupin``).

Run with ``uv run python examples/alni_order_disorder.py``. It takes about 5
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
XS = np.linspace(0.02, 0.30, 29)  # x(Al)
ORDERED = 1e-3  # a set with max |y_s − x| above this is γ′

db = databases.load("alni_dupin")
models = db.phase_models(EL)
phases = list(models.values())
print("phases:", sorted(models), "| skipped:", sorted(set(db.applicability(EL)) - set(models)))


def label(name, order):
    if name in ("FCC_L12", "FCC_A1"):
        return "γ′" if order > ORDERED else "γ"
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
                order = abs(ys[i][k][0] - ys[i][k][2]) if name == "FCC_L12" else 0.0
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
    print(f"  {key[0]:6.0f} K {' + '.join(key[1]):10s} ours {a.round(4)}  pycalphad {b.round(4)}  |Δx| {diffs[-1]:.5f}")
print(f"largest boundary difference: {max(diffs):.5f} in x(Al)")
print("regions only in discopt-calphad:", sorted(set(ours) - set(ref)))
print("regions only in pycalphad:", sorted(set(ref) - set(ours)))

# degree of order of γ′ across its single-phase range at 1200 K
T_ORDER = 1200.0
gp = models["FCC_L12"]
xo, do = [], []
for x in np.linspace(0.20, 0.30, 21):
    r = equilibrium(phases, x=float(x), T=T_ORDER, certify="none")
    for cs in r.composition_sets:
        if cs.phase is gp and cs.amount > 0.5:
            xo.append(float(cs.x[0]))
            do.append(gp.degree_of_order(cs.y))

# ------------------------------------------------------------------ figure
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), gridspec_kw={"width_ratios": [3, 2]})
colors = {"γ": "C0", "γ′": "C2", "L": "C3"}
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
ax.set_xlim(0, 0.32)
ax.set_title("Ni-rich Al–Ni (Dupin et al. 2001): tie-line ends", fontsize=10)
ax.legend(fontsize=7, loc="upper left")
ax2.plot(xo, do, "C2o-", ms=3)
ax2.set_xlabel("x(Al)")
ax2.set_ylabel("max |y_s − x|")
ax2.set_title(f"γ′ degree of order at {T_ORDER:.0f} K", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "alni_order_disorder.png", dpi=150)
print("wrote", HERE / "alni_order_disorder.png")
