"""Cr–Fe with magnetic BCC: a certified phase diagram (issue #6).

The BCC phase of Cr–Fe carries the Inden–Hillert–Jarl magnetic term: Fe is
ferromagnetic (T_C = 1043 K, β = 2.22 μ_B) and Cr antiferromagnetic
(T_N = 311.5 K, entered as T_C = −311.5 and β = −0.008 with the BCC AFM factor −1).
The term shapes the diagram in three places. It produces the γ loop on the Fe
side. The low-temperature BCC miscibility gap pairs a ferromagnetic Fe-rich set
with an antiferromagnetic Cr-rich one. It also shifts the σ-phase boundaries.

The magnetic term is piecewise in the site fractions: in the signs of T_C and β,
and in whether T is above or below T_C(x). Pricing therefore splits each
magnetic phase into the branches an interval evaluation cannot rule out and
bounds each one globally. The certificates below cover the full model.

This script

1. maps the diagram on a T–x grid with local pricing (``certify="none"``);
2. certifies one point per temperature with global pricing (``res.certify()``);
3. maps the same grid with pycalphad, on the same database, and compares the
   phase boundaries;
4. plots the diagram next to the Curie/Néel temperature T_C(x) of BCC and its
   magnetic Gibbs energy.

References

- Database: ``crfenb_jacob``, the Cr–Fe–Nb TDB file by A. Jacob (2016). Its
  Cr–Fe BCC parameters, magnetic ones included, are from J.-O. Andersson and
  B. Sundman, *Thermodynamic properties of the Cr–Fe system*, Calphad 11 (1987)
  83–92. Its unary data are SGTE: A. T. Dinsdale, Calphad 15 (1991) 317–425.
- Magnetic model: G. Inden, Proc. CALPHAD V, Düsseldorf (1976) III.4; M. Hillert
  and M. Jarl, *A model for alloying effects in ferromagnetic metals*, Calphad 2
  (1978) 227–238.

Run with ``uv run python examples/crfe_magnetic.py``. It takes about 9 minutes
and writes examples/crfe_magnetic.png.
"""

import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import databases, equilibrium
from discopt.calphad.equilibrium.certificate import CertificateLevel
from discopt.calphad.models.magnetic import ihj
from discopt.calphad.ops import NUM

HERE = Path(__file__).parent
EL = ("CR", "FE")
TEMPS = np.arange(600.0, 1801.0, 100.0)
XS = np.linspace(0.02, 0.98, 25)  # x(Cr)
CERTIFY_X = 0.5

db = databases.load("crfenb_jacob")
models = db.phase_models(EL)
phases = list(models.values())
print("phases:", sorted(models), "| magnetic:", sorted(n for n, p in models.items() if getattr(p, "magnetic", None)))


def two_phase(sets):
    """(sorted phase names, sorted (name, x(Cr)) pairs) for a two-phase state, or None."""
    sets = sorted((n, round(float(x), 6)) for n, x in sets)
    return (tuple(n for n, _ in sets), sets) if len(sets) == 2 else None


def discopt_map():
    out, failed, results = defaultdict(list), 0, {}
    for T in TEMPS:
        for x in XS:
            r = equilibrium(phases, x=float(x), T=float(T), certify="none")
            results[(float(T), float(x))] = r
            if r.certificate.level < CertificateLevel.STABILITY_SEARCHED:
                failed += 1
                continue
            tp = two_phase([(cs.phase.name, cs.x[0]) for cs in r.composition_sets if cs.amount > 1e-8])
            if tp:
                out[(float(T), tp[0])].append(tp[1])
    return out, failed, results


def pycalphad_map():
    from pycalphad import Database
    from pycalphad import equilibrium as peq
    from pycalphad import variables as v

    pdb = Database(str(databases.fetch("crfenb_jacob")))
    out = defaultdict(list)
    for T in TEMPS:
        eq = peq(pdb, [*EL, "VA"], sorted(models), {v.X("CR"): XS, v.T: float(T), v.P: 101325, v.N: 1})
        ph, xs = eq.Phase.values.squeeze(), eq.X.values.squeeze()
        for i in range(len(XS)):
            tp = two_phase([(n, xs[i][k][0]) for k, n in enumerate(ph[i]) if n])
            if tp:
                out[(float(T), tp[0])].append(tp[1])
    return out


t = time.time()
ours, failed, results = discopt_map()
print(f"discopt-calphad map: {time.time() - t:.0f} s, {failed} of {len(TEMPS) * len(XS)} points failed")

print(f"\nglobal certificates at x(Cr) = {CERTIFY_X}:")
certified = []
t = time.time()
for T in TEMPS:
    x = XS[np.argmin(np.abs(XS - CERTIFY_X))]
    r = results[(float(T), float(x))]
    cert = r.certify()
    sets = ", ".join(f"{cs.phase.name} {float(cs.x[0]):.4f}" for cs in r.composition_sets)
    lbs = [lb for lb in cert.lower_bounds.values() if lb is not None]
    worst = f"{min(lbs):+.1e}" if lbs else "none"
    print(f"  {T:6.0f} K  {cert.level.name:24s} worst LB {worst}  [{sets}]")
    certified.append((float(T), [float(cs.x[0]) for cs in r.composition_sets], cert.level))
print(f"  ({time.time() - t:.0f} s)")

t = time.time()
ref = pycalphad_map()
print(f"\npycalphad map: {time.time() - t:.0f} s")

print("\nphase boundaries in both maps (mean over the tie-lines of a region at that T):")
diffs = []
for key in sorted(set(ours) & set(ref)):
    a = np.mean([[x for _, x in s] for s in ours[key]], axis=0)
    b = np.mean([[x for _, x in s] for s in ref[key]], axis=0)
    diffs.append(float(np.max(np.abs(a - b))))
    print(f"  {key[0]:6.0f} K {' + '.join(key[1]):18s} ours {a.round(3)}  pycalphad {b.round(3)}  |Δx| {diffs[-1]:.4f}")
print(f"largest boundary difference: {max(diffs):.4f} in x(Cr)")
print("regions only in discopt-calphad:", sorted({k for k in set(ours) - set(ref)}))
print("regions only in pycalphad:", sorted({k for k in set(ref) - set(ours)}))

# ------------------------------------------------------------------ figure
fig, (ax, axm) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [3, 2]})
colors = {"BCC_A2": "C0", "FCC_A1": "C1", "SIGMA": "C2", "LIQUID": "C3"}
for tl, style, label in [(ref, dict(color="0.65", ms=7), "pycalphad"), (ours, dict(ms=3), "discopt-calphad")]:
    for (T, _), sets in tl.items():
        for s in sets:
            for name, x in s:
                kw = dict(style)
                kw.setdefault("color", colors.get(name, "k"))
                ax.plot([x], [T], "o", label=label, **kw)
                label = None
for T, xs, level in certified:
    ok = level >= CertificateLevel.CERTIFIED_TO_TOLERANCE
    ax.plot(
        xs, [T] * len(xs), "s", mfc="none", mec="k" if ok else "r", ms=9, label="certified" if T == TEMPS[0] else None
    )
for name, c in colors.items():
    ax.plot([], [], "o", color=c, label=name)
ax.set_xlabel("x(Cr)")
ax.set_ylabel("T (K)")
ax.set_xlim(0, 1)
ax.set_title("Cr–Fe: tie-line ends (squares: globally certified)", fontsize=10)
ax.legend(fontsize=7, loc="upper right", ncol=2)

bcc = models["BCC_A2"]
xg = np.linspace(0, 1, 401)
yv = [bcc.variables_from_site_fractions([{"CR": x, "FE": 1 - x}, {"VA": 1.0}]) for x in xg]
tc = np.array([bcc.curie_temperature(u, 300.0) for u in yv])
axm.plot(xg, np.where(tc > 0, tc, tc / bcc.magnetic.afm), "k-", label="T_C (ferro), T_N (AFM)")
axm.set_xlabel("x(Cr)")
axm.set_ylabel("T_C, T_N (K)")
axg = axm.twinx()
for T, c in ((600.0, "C0"), (1000.0, "C1"), (1400.0, "C3")):
    g = [
        float(ihj(bcc.magnetic, bcc.curie_temperature(u, T), bcc.magnetic_moment(u, T), T, NUM)) * 8.314462618 * T
        for u in yv
    ]
    axg.plot(xg, np.array(g) / 1000, "--", color=c, label=f"G_mag, {T:.0f} K")
axg.set_ylabel("G_mag of BCC (kJ/mol)")
axm.set_title("BCC magnetism (Andersson–Sundman 1987)", fontsize=10)
h1, l1 = axm.get_legend_handles_labels()
h2, l2 = axg.get_legend_handles_labels()
axm.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig(HERE / "crfe_magnetic.png", dpi=150)
print("wrote", HERE / "crfe_magnetic.png")
