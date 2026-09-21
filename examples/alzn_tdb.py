"""MVP 3: certified Al–Zn equilibria from a TDB database.

Reads the an Mey (1993) Al–Zn assessment (``alzn_mey`` in the database registry,
downloaded on first use), shows
the symbolic parameter view, builds the phase models, and computes certified
equilibria at three two-phase states:

- 580 K, x_Al = 0.6: inside the FCC miscibility gap (two FCC_A1 sets);
- 500 K, x_Al = 0.5: below the eutectoid, FCC_A1 + HCP_A3;
- 670 K, x_Al = 0.05: above the eutectic, LIQUID + HCP_A3.

Each state gets a G–x panel with the common tangent. Run with
``uv run python examples/alzn_tdb.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from discopt.calphad import databases, equilibrium
from discopt.calphad.plot import plot_gx

HERE = Path(__file__).parent
db = databases.load("alzn_mey")

print("phases usable in Al–Zn:")
for name, status in db.applicability(("AL", "ZN")).items():
    print(f"  {name:8s} {status}")

phases = db.phase_models(("AL", "ZN"))
liq = phases["LIQUID"]
print("\nsymbolic view (named functions kept):")
print("  G(LIQUID,AL;0) =", liq.g[0].expr())
for k, Lk in enumerate(liq.L):
    print(f"  L{k}(LIQUID)    =", Lk.expr(), "   <-", repr(Lk))
print("\nfirst TDB export lines:")
print("\n".join(db.to_tdb().splitlines()[2:6]))

states = [(580.0, 0.6), (500.0, 0.5), (670.0, 0.05)]  # (T, x_Al)
fig, axes = plt.subplots(1, len(states), figsize=(15, 4.4))
for ax, (T, x_al) in zip(axes, states):
    res = equilibrium(list(phases.values()), x=x_al, T=T)
    print()
    print(res.summary())
    plot_gx(list(phases.values()), res, T=T, ax=ax)
    ax.set_xlabel("x_Al")
    sets = " + ".join(cs.phase.name for cs in res.composition_sets)  # a miscibility gap shows the phase twice
    ax.set_title(f"Al–Zn, T = {T:g} K, x_Al = {x_al:g}\n{sets}  [{res.certificate.level.name}]", fontsize=9)
fig.tight_layout()
out = HERE / "alzn_tdb.png"
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
