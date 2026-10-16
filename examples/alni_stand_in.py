"""Al–Ni with plain stand-ins for its order–disorder phases (issue #14).

Dupin, Ansara and Sundman's Al–Ni assessment (``alni_dupin``) describes γ′
(L1₂, FCC_L12) and β (B2, BCC_B2) as partitioned phases: an ordered part on
top of a disordered parent. The solver models γ′ natively (issue #6, see
examples/alni_order_disorder.py). β allows vacancies on its substitutional
sublattices, which the native model does not support yet, so it is unavailable.
:func:`discopt.calphad.ordering.stand_in` rewrites it as a plain CEF phase on
the ordered sublattices. Without magnetism and constitutional vacancies, the
stand-in reproduces the partitioned G exactly; pycalphad agrees to 1e-10 J/mol.

This script maps the Ni–Al diagram twice:

- discopt-calphad, with native γ′ and a stand-in for β (magnetism and B2
  vacancies dropped from it, but kept in the other phases);
- pycalphad, with the full assessment.

It reports how far apart the phase boundaries are, and writes
examples/alni_stand_in.png.
"""

import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import databases, equilibrium
from discopt.calphad.equilibrium.certificate import CertificateLevel
from discopt.calphad.ordering import stand_in

HERE = Path(__file__).parent
EL = ("AL", "NI")
TEMPS = np.arange(900.0, 1701.0, 100.0)
XS = np.linspace(0.02, 0.98, 25)  # x(Al)

db = databases.load("alni_dupin")
for ordered in ("BCC_B2",):
    si = stand_in(db, ordered, EL)
    print(si.summary())
    db = si.database
phases = list(db.phase_models(EL).values())
print("phases:", sorted(p.name for p in phases))


def label(name, site_fractions):
    """γ, γ′, β, ... with ordered and disordered sets of the fcc/bcc phases told apart."""
    if name in ("FCC_L12_SI", "FCC_L12", "BCC_B2_SI", "BCC_B2"):
        ordered = abs(site_fractions[0] - site_fractions[1]) > 0.02
        if name.startswith("FCC"):
            return "γ′" if ordered else "γ"
        return "β" if ordered else "A2"
    return {"FCC_A1": "γ", "LIQUID": "L", "AL3NI1": "Al3Ni", "AL3NI2": "Al3Ni2", "AL3NI5": "Al3Ni5"}.get(name, name)


def discopt_tie_lines():
    out, failed = defaultdict(list), 0
    for T in TEMPS:
        for x in XS:
            r = equilibrium(phases, x=float(x), T=float(T), certify="none")
            if r.certificate.level < CertificateLevel.STABILITY_SEARCHED:
                failed += 1
                continue
            sets = []
            for cs in r.composition_sets:
                if hasattr(cs.phase, "site_fractions") and cs.phase.n_y > 1:
                    sf = cs.phase.site_fractions(cs.y)
                    y = [sf[0].get("AL", 0.0), sf[1].get("AL", 0.0)]
                else:
                    y = [0.0, 0.0]
                sets.append((label(cs.phase.name, y), float(cs.x[0])))
            if len(sets) == 2:
                out[(float(T), tuple(sorted(n for n, _ in sets)))].append(sorted(sets))
    return out, failed


def pycalphad_tie_lines():
    from pycalphad import Database
    from pycalphad import equilibrium as peq
    from pycalphad import variables as v

    pdb = Database(str(databases.fetch("alni_dupin")))
    names = ["LIQUID", "FCC_A1", "FCC_L12", "BCC_A2", "BCC_B2", "AL3NI1", "AL3NI2", "AL3NI5"]
    out = defaultdict(list)
    for T in TEMPS:
        eq = peq(pdb, ["AL", "NI", "VA"], names, {v.X("AL"): XS, v.T: float(T), v.P: 101325, v.N: 1})
        for i in range(len(XS)):
            ph = eq.Phase.values.squeeze()[i]
            xs = eq.X.values.squeeze()[i]
            ys = eq.Y.values.squeeze()[i]
            sets = []
            for k, name in enumerate(ph):
                if not name:
                    continue
                y = (
                    [ys[k][0], ys[k][2]]
                    if name in ("FCC_L12",)
                    else [ys[k][0], ys[k][3]]
                    if name == "BCC_B2"
                    else [0, 0]
                )
                sets.append((label(name, y), float(xs[k][0])))
            if len(sets) == 2:
                out[(float(T), tuple(sorted(n for n, _ in sets)))].append(sorted(sets))
    return out


t = time.time()
ours, failed = discopt_tie_lines()
print(f"discopt-calphad: {time.time() - t:.0f} s, {failed} of {len(TEMPS) * len(XS)} points failed")
t = time.time()
ref = pycalphad_tie_lines()
print(f"pycalphad: {time.time() - t:.0f} s")

print("\nphase boundaries in both maps (each two-phase region, mean over its tie-lines at that T):")
diffs = []
for key in sorted(set(ours) & set(ref)):
    a = np.mean([[x for _, x in s] for s in ours[key]], axis=0)
    b = np.mean([[x for _, x in s] for s in ref[key]], axis=0)
    diffs.append(float(np.max(np.abs(a - b))))
    print(
        f"  {key[0]:6.0f} K {' + '.join(key[1]):16s} discopt {a.round(3)}  pycalphad {b.round(3)}  |Δx| {diffs[-1]:.3f}"
    )
only_ours = sorted({k[1] for k in set(ours) - set(ref)})
only_ref = sorted({k[1] for k in set(ref) - set(ours)})
print(f"\nlargest boundary difference: {max(diffs):.3f} in x(Al); median {np.median(diffs):.3f}")
print("regions only in discopt-calphad:", only_ours, "\nregions only in pycalphad:", only_ref)

fig, ax = plt.subplots(figsize=(7, 5))
for tl, style, name in [
    (ref, dict(color="0.6", ms=6), "pycalphad, full model"),
    (ours, dict(color="C3", ms=3), "discopt-calphad, stand-ins"),
]:
    for (T, _), sets in tl.items():
        for s in sets:
            ax.plot([x for _, x in s], [T, T], "o", label=name, **style)
            name = None
ax.set_xlabel("x(Al)")
ax.set_ylabel("T (K)")
ax.set_xlim(0, 1)
ax.set_title("Ni–Al: two-phase tie-line ends", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "alni_stand_in.png", dpi=150)
print("wrote", HERE / "alni_stand_in.png")
