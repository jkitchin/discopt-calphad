"""Pure iron: what the magnetic term does, checked against the SGTE data (issue #6).

Without its magnetic contribution, BCC iron would not be the stable phase at
room temperature in the SGTE description. The Inden–Hillert–Jarl term
RT ln(β + 1) f(T/T_C), with T_C = 1043 K and β = 2.22 μ_B for BCC and an
antiferromagnetic FCC (T_N = 67 K, β = 0.7 μ_B, AFM factor −3), lowers G of BCC
below T_C. That gives the α → γ → δ sequence and the λ peak of the heat
capacity at the Curie temperature.

This script

1. finds the BCC/FCC and BCC/liquid crossings of G and compares them with the
   transition temperatures Dinsdale tabulates for the SGTE unary data
   (α → γ 1184.81 K, γ → δ 1667.47 K, melting 1811 K);
2. repeats step 1 with the magnetic term removed;
3. computes C_p of BCC iron, −T ∂²G/∂T², through the λ peak at T_C;
4. checks G and C_p against pycalphad on the same database.

References

- Unary data: A. T. Dinsdale, *SGTE data for pure elements*, Calphad 15 (1991)
  317–425, read from the ``crfenb_jacob`` TDB file (A. Jacob, 2016).
- Magnetic model: G. Inden, Proc. CALPHAD V, Düsseldorf (1976) III.4; M. Hillert
  and M. Jarl, *A model for alloying effects in ferromagnetic metals*, Calphad 2
  (1978) 227–238.

Phase models need two elements, so pure Fe is the Fe end of the Cr–Fe models.
With 0 ln 0 = 0, that is exact. Run with ``uv run python examples/fe_magnetic.py``
(about 10 s). It writes examples/fe_magnetic.png.
"""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from discopt.calphad import databases
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.ops import NUM

HERE = Path(__file__).parent
SGTE = {"alpha -> gamma": 1184.81, "gamma -> delta": 1667.47, "melting": 1811.00}

db = databases.load("crfenb_jacob")
models = db.phase_models(("CR", "FE"))
bcc, fcc, liq = models["BCC_A2"], models["FCC_A1"], models["LIQUID"]
print(f"BCC: T_C(Fe) = {bcc.curie_temperature(bcc.variables_from_site_fractions([[0, 1], [1]]), 300.0):.0f} K")


def pure_fe(phase):
    """Site-fraction variables for pure Fe in ``phase``."""
    if not hasattr(phase, "constituents"):  # a BinarySolution: y is the fraction of its first element
        return np.array([float(phase.elements[0] == "FE")])
    fr = [{c: float(c == "FE") if "FE" in cons else 1.0 for c in cons} for cons in phase.constituents]
    return phase.variables_from_site_fractions(fr)


def G(phase, T):
    """G of pure Fe in ``phase``, J/mol."""
    return float(phase.gibbs(pure_fe(phase), NUM, T)) * R_GAS * T


def crossings(b, f, l):
    """The α→γ, γ→δ and melting temperatures for BCC ``b``, FCC ``f`` and liquid ``l`` (None if absent)."""
    out = {}
    for name, (p, q, lo, hi) in {
        "alpha -> gamma": (b, f, 900.0, 1400.0),
        "gamma -> delta": (b, f, 1400.0, 1780.0),
        "melting": (b, l, 1700.0, 1900.0),
    }.items():
        d = lambda T: G(p, T) - G(q, T)  # noqa: E731
        out[name] = brentq(d, lo, hi, xtol=1e-6) if d(lo) * d(hi) < 0 else None
    return out


print("\ntransition temperatures (K):")
with_mag = crossings(bcc, fcc, liq)
no_mag = crossings(dataclasses.replace(bcc, magnetic=None), dataclasses.replace(fcc, magnetic=None), liq)
for name, ref in SGTE.items():
    m, n = with_mag[name], no_mag[name]
    n_s = "none in range" if n is None else f"{n:8.2f}"
    print(f"  {name:15s} SGTE {ref:8.2f}   with magnetism {m:8.2f} (Δ {m - ref:+.2f})   without: {n_s}")
T_fine = np.linspace(300, 1400, 1101)
dg = np.array([G(dataclasses.replace(bcc, magnetic=None), T) - G(dataclasses.replace(fcc, magnetic=None), T) for T in T_fine])
print(f"  without magnetism, FCC is below BCC from {T_fine[np.argmax(dg > 0)]:.0f} K up (G_bcc - G_fcc > 0)")


def cp(phase, T, h=0.05):
    return -T * (G(phase, T + h) - 2 * G(phase, T) + G(phase, T - h)) / h**2


T_cp = np.linspace(400.0, 1600.0, 1201)
cp_bcc = np.array([cp(bcc, T) for T in T_cp])
peak = T_cp[np.argmax(cp_bcc)]
print(f"\nC_p of BCC Fe peaks at {peak:.0f} K (T_C = 1043 K), {cp_bcc.max():.1f} J/(mol K)")

# ------------------------------------------------------------------ pycalphad check
try:
    from pycalphad import Database, calculate

    pdb = Database(str(databases.fetch("crfenb_jacob")))
    worst_g = worst_cp = 0.0
    for name, ph in (("BCC_A2", bcc), ("FCC_A1", fcc)):
        Ts = np.array([500.0, 900.0, 1100.0, 1300.0, 1700.0])
        ref = calculate(pdb, ["FE", "VA"], name, T=Ts, P=101325, N=1, output="GM")
        ref_cp = calculate(pdb, ["FE", "VA"], name, T=Ts, P=101325, N=1, output="CPM")
        for T, g, c in zip(Ts, np.asarray(ref.GM).ravel(), np.asarray(ref_cp.CPM).ravel()):
            # pycalphad's R is 8.3145; the magnetic term carries R, so scale it for the comparison
            mag = G(ph, T) - G(dataclasses.replace(ph, magnetic=None), T)
            worst_g = max(worst_g, abs(G(ph, T) + mag * (8.3145 / R_GAS - 1) - g) / abs(g))
            worst_cp = max(worst_cp, abs(cp(ph, T) - c) / abs(c))
    print(f"pycalphad: G agrees to {worst_g:.1e} relative, C_p to {worst_cp:.1e} (finite differences here)")
except ImportError:
    print("pycalphad not installed; skipping the cross-check")

# ------------------------------------------------------------------ figure
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
T = np.linspace(300, 1900, 801)
nb, nf = dataclasses.replace(bcc, magnetic=None), dataclasses.replace(fcc, magnetic=None)
a1.plot(T, [(G(fcc, t) - G(bcc, t)) / 1000 for t in T], "C0-", label="G(FCC) − G(BCC), IHJ")
a1.plot(T, [(G(nf, t) - G(nb, t)) / 1000 for t in T], "C0--", label="the same, no magnetism")
a1.plot(T, [(G(liq, t) - G(bcc, t)) / 1000 for t in T], "C3-", label="G(liquid) − G(BCC)")
a1.axhline(0, color="k", lw=0.6)
for name, ref in SGTE.items():
    a1.axvline(ref, color="0.6", ls=":", lw=1)
a1.set_xlabel("T (K)")
a1.set_ylabel("ΔG (kJ/mol)")
a1.set_ylim(-1.2, 3)
a1.set_title("Pure Fe; dotted: SGTE transitions (Dinsdale 1991)", fontsize=9)
a1.legend(fontsize=7)
a2.plot(T_cp, cp_bcc, "C0-", label="BCC, IHJ")
a2.plot(T_cp, [cp(nb, t) for t in T_cp], "C0--", label="BCC, no magnetism")
a2.axvline(1043, color="0.6", ls=":", lw=1)
a2.set_xlabel("T (K)")
a2.set_ylabel("C_p (J/(mol K))")
a2.set_title("λ peak at the Curie temperature", fontsize=9)
a2.legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "fe_magnetic.png", dpi=150)
print("wrote", HERE / "fe_magnetic.png")
