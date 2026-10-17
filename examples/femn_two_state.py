"""Fe–Mn liquid with the Einstein and two-state contributions (issue #6).

Third-generation CALPHAD descriptions model a liquid down to 0 K. Its Gibbs energy
is a sum of physically motivated parts:

- an Einstein solid for the harmonic vibrations, G_E = 1.5 R θ_E + 3 R T ln(1 −
  e^(−θ_E/T)). The TDB stores ln θ_E (THETA), so composition dependence is
  Redlich–Kister in ln θ_E;
- the two-state model for the liquid–amorphous transition, G_2st = −R T ln(1 +
  e^(−ΔG_d/RT)). Each atom is solid-like (amorphous) or liquid-like, ΔG_d (GD) is
  the energy of the change, and the fraction of liquid-like atoms is
  1 / (1 + e^(ΔG_d/RT)). Below the glass transition the liquid becomes an
  amorphous solid, and its heat capacity falls to the Einstein value;
- magnetism (here the Xiong model, with TC and NT) and a polynomial.

This script uses pycalphad's test database femn.tdb ("for testing purposes
only", one LIQUID phase with no interaction parameters). It

1. splits G of pure liquid Fe into those parts, 20–2500 K, and computes the heat
   capacity C_p = −T ∂²G/∂T² and the fraction of liquid-like atoms;
2. compares G with pycalphad.calculate across compositions and temperatures;
3. computes Fe–Mn liquid equilibria, certifies them, and compares the assemblage
   with pycalphad.equilibrium;
4. plots the decomposition, C_p and the liquid-like fraction.

C_p rises to about 3R as the Einstein vibrations switch on, then climbs through a
broad hump between 1000 and 2000 K as atoms turn liquid-like (0.035 of them at
1000 K, 0.28 at the melting point, 1811 K). The sharp peak at 200 K is the
magnetic transition: this test file gives liquid Fe T_C = 200 K.

References:
- J. Ågren, *Thermodynamics of supercooled liquids and their glass transition*,
  Phys. Chem. Liq. 18 (1988) 123–139 (the two-state model).
- J. Ågren, B. Cheynet, M. T. Clavaguera-Mora, K. Hack, J. Hertz, F. Sommer and
  U. Kattner, *Workshop on thermodynamic models and data for pure elements and
  other endmembers of solutions*, Calphad 19 (1995) 449–480 (the Einstein and
  two-state terms for third-generation unary descriptions).
- Q. Chen and B. Sundman, *Modeling of thermodynamic properties for Bcc, Fcc,
  liquid, and amorphous iron*, J. Phase Equilib. 22 (2001) 631–644 (a
  third-generation description of Fe in this form).
- W. Xiong, Q. Chen, P. A. Korzhavyi and M. Selleby, *An improved magnetic model
  for thermodynamic modeling*, Calphad 39 (2012) 11–20.

Run with ``uv run python examples/femn_two_state.py`` (about a minute). It writes
examples/femn_two_state.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from discopt.calphad import CertificateLevel, ThermoDatabase, equilibrium
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models.lattice import einstein, two_state
from discopt.calphad.ops import NUM

HERE = Path(__file__).parent
PYCALPHAD_R = 8.3145  # pycalphad.variables.R; ours is the CODATA value


def tdb_path():
    import pycalphad

    return Path(pycalphad.__file__).parent / "tests" / "databases" / "femn.tdb"


db = ThermoDatabase.from_tdb(tdb_path())
liq = db.phase_model("LIQUID", ("FE", "MN"))


def u_of(yfe):
    return liq.variables_from_site_fractions([{"FE": yfe, "MN": 1 - yfe}, {"VA": 1.0}])


def parts(yfe, T):
    """G and its parts, J/mol, at site fraction ``yfe`` of Fe."""
    Y = liq._Y(np.asarray(u_of(yfe)))
    RT = R_GAS * T
    ein = float(einstein(liq._sum(liq.einstein, Y, T), T, NUM)) * RT
    gd = float(liq._sum(liq.two_state, Y, T))
    tst = float(two_state(gd, T, NUM)) * RT
    mag = float(liq.magnetic_at(Y, T, NUM)) * RT
    total = float(liq.gibbs(u_of(yfe), NUM, T)) * RT
    return {
        "total": total,
        "Einstein": ein,
        "two-state": tst,
        "magnetic": mag,
        "rest": total - ein - tst - mag,
        "gd": gd,
    }


# ------------------------------------------------------------------ 1. pure liquid Fe
TS = np.linspace(20.0, 2500.0, 497)
P = [parts(1.0, T) for T in TS]
h = 0.5
cp = np.array(
    [-T * (parts(1.0, T + h)["total"] - 2 * parts(1.0, T)["total"] + parts(1.0, T - h)["total"]) / h**2 for T in TS]
)
liquid_like = np.array([1 / (1 + np.exp(p["gd"] / (R_GAS * T))) for p, T in zip(P, TS)])
T_MELT = 1811.0  # Fe
for T in (500.0, 1000.0, T_MELT, 2500.0):
    print(f"pure liquid Fe at {T:.0f} K: {liquid_like[np.searchsorted(TS, T)]:.3f} of the atoms are liquid-like")
print(f"C_p at 20 K: {cp[0]:.2f} J/mol/K; at 300 K: {cp[np.searchsorted(TS, 300)]:.2f}; at 2000 K: {cp[-101]:.2f}")

# ------------------------------------------------------------------ 2. G against pycalphad
from pycalphad import Database, calculate  # noqa: E402
from pycalphad import equilibrium as peq  # noqa: E402
from pycalphad import variables as v  # noqa: E402

pdb = Database(str(tdb_path()))
worst = 0.0
for T in (300.0, 1000.0, 1800.0, 2500.0):
    for yfe in np.linspace(0.05, 0.95, 7):
        ref = float(
            calculate(
                pdb, ["FE", "MN", "VA"], "LIQUID", T=T, P=101325, N=1, points=np.array([[yfe, 1 - yfe, 1.0]])
            ).GM.values.squeeze()
        )
        p = parts(yfe, T)
        Y = liq._Y(np.asarray(u_of(yfe)))
        ideal = float(liq.ideal_at(Y, NUM))
        # every term that carries R, at pycalphad's R
        ours = p["total"] + (PYCALPHAD_R - R_GAS) * T * (ideal + (p["Einstein"] + p["magnetic"]) / (R_GAS * T))
        ours += -PYCALPHAD_R * T * np.logaddexp(0, -p["gd"] / (PYCALPHAD_R * T)) - p["two-state"]
        worst = max(worst, abs(ours - ref) / abs(ref))
print(f"G against pycalphad.calculate: largest relative difference {worst:.1e} (28 points)")

# ------------------------------------------------------------------ 3. equilibria
agree = certified = 0
XS, TE = (0.1, 0.3, 0.5, 0.7, 0.9), (600.0, 1200.0, 1800.0)
for T in TE:
    for x in XS:
        r = equilibrium([liq], x=x, T=T, certify="none")
        c = r.certify()
        certified += c.level >= CertificateLevel.CERTIFIED_TO_TOLERANCE
        eq = peq(pdb, ["FE", "MN", "VA"], ["LIQUID"], {v.X("FE"): x, v.T: T, v.P: 101325, v.N: 1})
        n_ref = sum(1 for n in np.squeeze(eq.Phase.values) if n)
        agree += n_ref == len(r.composition_sets)
print(f"equilibria: {certified} of {len(XS) * len(TE)} certified; same number of liquids as pycalphad at {agree}")

# ------------------------------------------------------------------ 4. figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
ax = axes[0]
for k in ("Einstein", "two-state", "magnetic"):
    ax.plot(TS, [p[k] / 1000 for p in P], label=k)
ax.set_xlabel("T (K)")
ax.set_ylabel("contribution to G (kJ/mol)")
ax.set_title("pure liquid Fe: parts of G", fontsize=10)
ax.legend(fontsize=8)
ax = axes[1]
ax.plot(TS, cp, color="C3")
ax.axhline(3 * R_GAS, color="0.6", lw=0.8, ls=":", label="3R (Dulong–Petit)")
ax.set_xlabel("T (K)")
ax.set_ylabel("C_p (J/mol/K)")
ax.set_ylim(0, max(60, float(np.nanmax(cp[TS > 100])) * 1.1))
ax.set_title("liquid/amorphous Fe heat capacity", fontsize=10)
ax.legend(fontsize=8)
ax = axes[2]
ax.plot(TS, liquid_like, color="C2")
ax.axvline(T_MELT, color="0.6", lw=0.8, ls=":", label="Fe melting point")
ax.legend(fontsize=8)
ax.set_xlabel("T (K)")
ax.set_ylabel("fraction of liquid-like atoms")
ax.set_title("two-state model (Ågren 1988)", fontsize=10)
fig.tight_layout()
fig.savefig(HERE / "femn_two_state.png", dpi=150)
print("wrote", HERE / "femn_two_state.png")
