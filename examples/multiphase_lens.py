"""MVP 2 example: several phases, T-dependent energies, phase appearance and disappearance.

Left: the ideal LIQUID/FCC lens. Each tie-line comes from a certified equilibrium
and is drawn over the analytic solidus and liquidus. Squares mark the
single-phase states above and below the lens.
Right: G–x at 1000 K for a liquid, an FCC with a miscibility gap, and a compound AB,
with the certified equilibrium at x_A = 0.3.

Run with: uv run python examples/multiphase_lens.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import R_GAS, BinarySolution, StoichiometricPhase, TPoly, equilibrium
from discopt.calphad.plot import plot_gx

LIQ = BinarySolution("LIQUID", g=(TPoly(10000.0, -10.0), TPoly(15000.0, -10.0)))
FCC = BinarySolution("FCC_A1")


def analytic_lens(T):
    KA = np.exp(LIQ.g[0](T) / (R_GAS * T))
    KB = np.exp(LIQ.g[1](T) / (R_GAS * T))
    xl = (1 - KB) / (KA - KB)
    return xl, KA * xl


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))

    Ts = np.linspace(1001.0, 1499.0, 300)
    xl, xs = analytic_lens(Ts)
    a1.plot(xl, Ts, "C0", label="liquidus (analytic)")
    a1.plot(xs, Ts, "C1", label="solidus (analytic)")
    # Inside the lens the bulk composition only sets the phase fractions, so the analytic
    # midpoint is used. Outside the lens (950 K, 1550 K) one phase must be left.
    for T in (950.0, 1100.0, 1200.0, 1300.0, 1400.0, 1550.0):
        x = 0.5 if T < 1000 or T > 1500 else float(np.mean(analytic_lens(T)))
        res = equilibrium([LIQ, FCC], x=x, T=T, eps=1e-5)
        print(res.summary(), "\n")
        xsets = sorted(cs.x[0] for cs in res.composition_sets)
        if len(xsets) == 2:
            a1.plot(xsets, [T, T], "k-o", ms=4, lw=1)
        else:
            a1.plot(x, T, "ks", ms=5)
            a1.annotate(res.phase_names[0], (x, T), xytext=(6, -3), textcoords="offset points", fontsize=8)
    a1.plot([], [], "k-o", ms=4, label="certified tie-lines")
    a1.set_xlabel("x(A)")
    a1.set_ylabel("T / K")
    a1.set_title("Ideal LIQUID/FCC lens")
    a1.legend(fontsize=8)

    T3 = 1000.0
    phases = [
        BinarySolution("LIQUID", g=(TPoly(10000.0, -10.0), TPoly(15000.0, -10.0)), L=(-5000.0,)),
        BinarySolution("FCC_A1", L=(20000.0, 3000.0)),
        StoichiometricPhase("AB", ("A", "B"), (0.5, 0.5), g=-5000.0),
    ]
    res = equilibrium(phases, x=0.3, T=T3, eps=1e-5)
    print(res.summary())
    plot_gx(phases, res, ax=a2)
    a2.set_title(f"T = {T3:g} K, x(A) = 0.3: {res.certificate.level.name}")

    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
