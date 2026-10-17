"""Cu–O with the two-sublattice ionic liquid (issue #6).

Schramm, Behr, Löser and Wetzig describe liquid Cu–O with one ionic liquid,
(Cu⁺, Cu²⁺, Cu³⁺)_P (O²⁻, Va)_Q. Metallic copper is Cu⁺:Va, and the oxide melt is
Cu⁺:O²⁻ (with Cu²⁺ and Cu³⁺ toward CuO). The site ratios follow from charge
neutrality, Q = Σ q_C y_C and P = 2 y_O + Q y_Va, so one phase spans the metal
and the oxide melt, with a miscibility gap between them. The other phases are
FCC Cu (with dissolved O), the line compounds Cu₂O and CuO, and O₂ gas at 1 atm.

This script

1. maps the diagram (1100–1700 K, x(O) ≤ 0.6) with local pricing;
2. maps it with pycalphad, on the same database, and compares both maps point
   by point. Where they differ, it compares G: both codes give the same G for
   every phase, so the lower state is the better minimum of the same model;
3. finds the eutectic (L ↔ FCC + Cu₂O) by bisection, and the monotectic
   (L₂ ↔ L₁ + Cu₂O, the oxide-rich liquid decomposing on cooling) as the
   temperature where Cu₂O touches the common tangent of the two liquids. That
   tangency is smooth in T, so it does not depend on local pricing finding the
   second liquid near the invariant;
4. plots the diagram, marking the points where pycalphad misses the liquid
   miscibility gap.

Reference: L. Schramm, G. Behr, W. Löser and K. Wetzig, *Thermodynamic
reassessment of the Cu–O phase diagram*, J. Phase Equilib. Diffus. 26 (2005)
605–612 (TDB file by K. Hashimoto and T. Abe, NIMS, 2008). Ionic liquid model:
B. Sundman, *Modification of the two-sublattice model for liquids*, Calphad 15
(1991) 109–119.

Run with ``uv run python examples/cuo_ionic.py`` (about 5 minutes). It writes
examples/cuo_ionic.png.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import ThermoDatabase, equilibrium
from discopt.calphad.equilibrium.certificate import CertificateLevel
from discopt.calphad.equilibrium.common import R_GAS

HERE = Path(__file__).parent
EL = ("CU", "O")
TEMPS = np.arange(1100.0, 1701.0, 50.0)
XO = np.linspace(0.01, 0.6, 30)


def tdb_path():
    import pycalphad

    return Path(pycalphad.__file__).parent / "tests" / "databases" / "cuo.tdb"


db = ThermoDatabase.from_tdb(tdb_path())
models = db.phase_models(EL)
phases = list(models.values())
print("phases:", sorted(models))


def ours(xo, T):
    r = equilibrium(phases, x=1 - float(xo), T=float(T), certify="none")
    if r.certificate.level < CertificateLevel.STABILITY_SEARCHED:
        return None, None
    sets = sorted((cs.phase.name, round(float(cs.x[1]), 4)) for cs in r.composition_sets if cs.amount > 1e-8)
    return sets, r.G * R_GAS * T


def pycalphad_map():
    from pycalphad import Database
    from pycalphad import equilibrium as peq
    from pycalphad import variables as v

    pdb = Database(str(tdb_path()))
    out = {}
    for T in TEMPS:
        eq = peq(pdb, [*EL, "VA"], sorted(models), {v.X("O"): XO, v.T: float(T), v.P: 101325, v.N: 1})
        ph, xs, gm = eq.Phase.values.squeeze(), eq.X.values.squeeze(), eq.GM.values.squeeze()
        for i, xo in enumerate(XO):
            sets = sorted((str(n), round(float(xs[i][k][1]), 4)) for k, n in enumerate(ph[i]) if n)
            out[(float(T), float(xo))] = (sets, float(gm[i]))
    return out


t = time.time()
mine, failed = {}, 0
for T in TEMPS:
    for xo in XO:
        sets, g = ours(xo, T)
        if sets is None:
            failed += 1
            continue
        mine[(float(T), float(xo))] = (sets, g)
print(f"discopt-calphad: {time.time() - t:.0f} s, {failed} of {len(TEMPS) * len(XO)} points failed")
t = time.time()
ref = pycalphad_map()
print(f"pycalphad: {time.time() - t:.0f} s")

same = differ_lower = differ_higher = py_failed = 0
gap_missed = []
for key, (sets, g) in mine.items():
    rsets, rg = ref[key]
    if not rsets or not np.isfinite(rg):
        py_failed += 1  # pycalphad returned no phases (NaN) at this point
        continue
    names, rnames = [n for n, _ in sets], [n for n, _ in rsets]
    if names == rnames and np.allclose([x for _, x in sets], [x for _, x in rsets], atol=2e-3):
        same += 1
        continue
    # pycalphad's GM is per mole of atoms with R = 8.3145; the difference matters at 1e-5 RT only
    if g < rg - 1.0:
        differ_lower += 1
        if names.count("IONIC_LIQ") == 2:
            gap_missed.append(key)
        print(f"  {key[0]:6.0f} K x(O)={key[1]:.3f}: ours {sets} is {rg - g:.1f} J/mol below pycalphad's {rsets}")
    else:
        differ_higher += 1
        print(f"  {key[0]:6.0f} K x(O)={key[1]:.3f}: ours {sets} vs pycalphad {rsets}, ΔG = {g - rg:+.2f} J/mol")
print(
    f"\nsame state at {same} points; lower G than pycalphad at {differ_lower}; not lower at {differ_higher}; "
    f"pycalphad found no state at {py_failed}"
)
print(f"pycalphad misses the liquid miscibility gap at {len(gap_missed)} points")


def bisect(xo, test, lo, hi, tol=0.05):
    """The temperature in [lo, hi] where test(assemblage) turns True, by bisection."""
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        sets, _ = ours(xo, mid)
        if sets is None:  # a failed point: nudge it and try once more
            sets, _ = ours(xo, mid + 0.01 * tol)
        if sets is None:
            raise RuntimeError(f"equilibrium failed at x(O) = {xo}, T = {mid}")
        if test([n for n, _ in sets]):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


t_eut = bisect(0.005, lambda n: "IONIC_LIQ" in n, 1300.0, 1400.0)


def cu2o_above_liquid_gap(T):
    """Ψ of Cu2O at the μ of the two-liquid tie line (the liquid alone, at x(O) = 0.2)."""
    from discopt.calphad.ops import NUM

    r = equilibrium([models["IONIC_LIQ"]], x=0.8, T=float(T), certify="none")
    cu2o = models["CU2O"]
    return float(cu2o.gibbs([], NUM, T)) - float(r.mu @ np.array(cu2o.amounts([]))), r


from scipy.optimize import brentq  # noqa: E402

t_mono = brentq(lambda T: cu2o_above_liquid_gap(T)[0], 1400.0, 1550.0, xtol=0.01)
liquids = sorted(float(cs.x[1]) for cs in cu2o_above_liquid_gap(t_mono)[1].composition_sets)
print(f"\neutectic L -> FCC + Cu2O: {t_eut:.1f} K ({t_eut - 273.15:.1f} °C)")
print(
    f"monotectic L2 -> L1 + Cu2O: {t_mono:.1f} K ({t_mono - 273.15:.1f} °C), "
    f"L1 at x(O) = {liquids[0]:.3f}, L2 at {liquids[1]:.3f}"
)

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(7.5, 5))
colors = {"IONIC_LIQ": "C3", "FCC_A1": "C0", "CU2O": "C2", "CUO": "C1", "GAS": "C7"}
for (T, _), (sets, _) in ref.items():
    if len(sets) == 2:
        for _name, x in sets:
            ax.plot([x], [T], "o", color="0.75", ms=7)
for (T, _), (sets, _) in mine.items():
    if len(sets) == 2:
        for n, x in sets:
            ax.plot([x], [T], "o", color=colors.get(n, "k"), ms=3)
for T, xo in gap_missed:
    ax.plot([xo], [T], "x", color="k", ms=6)
ax.axhline(t_eut, color="0.5", lw=0.6, ls=":")
ax.axhline(t_mono, color="0.5", lw=0.6, ls=":")
ax.plot([], [], "o", color="0.75", label="pycalphad")
for n, c in colors.items():
    ax.plot([], [], "o", color=c, ms=3, label=n)
ax.plot([], [], "x", color="k", label="pycalphad misses the liquid gap")
ax.set_xlabel("x(O)")
ax.set_ylabel("T (K)")
ax.set_xlim(0, 0.62)
ax.set_title("Cu–O (Schramm et al. 2005): tie-line ends; dotted: eutectic, monotectic", fontsize=9)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "cuo_ionic.png", dpi=150)
print("wrote", HERE / "cuo_ionic.png")
