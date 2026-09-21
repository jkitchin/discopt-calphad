"""MVP 4: certified Cu–Mg equilibria with a two-sublattice CEF phase.

Reads the NIMS Cu–Mg assessment (``cumg_nims`` in the database registry). CU2MG, the
Laves phase (CU,MG)2(CU,MG)1, has mixing on both sublattices, so it is built as
a ``CEFPhase``. The script prints its symbolic Gibbs energy, then computes
certified equilibria at three states:

- 700 K, x_Cu = 0.75: FCC_A1 + CU2MG;
- 700 K, x_Cu = 0.50: CU2MG + CUMG2;
- 900 K, x_Cu = 0.62: CU2MG + LIQUID.

Each state gets a G–x panel. The CU2MG curve is the lowest G per x over a grid
of site fractions. Run with ``uv run python examples/cumg_cef.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from discopt.calphad import databases, equilibrium
from discopt.calphad.plot import plot_gx

HERE = Path(__file__).parent
db = databases.load("cumg_nims")

phases = db.phase_models(("CU", "MG"))
cu2mg = phases["CU2MG"]
print(f"{cu2mg.name}: site ratios {cu2mg.site_ratios}, constituents {cu2mg.constituents}, n_y = {cu2mg.n_y}")
print("terms:", ", ".join(t.label for t in cu2mg.terms))
print("\nsymbolic G (per formula unit, named functions kept):")
print(" ", cu2mg.gibbs_expr())

states = [(700.0, 0.75), (700.0, 0.5), (900.0, 0.62)]  # (T, x_Cu)
fig, axes = plt.subplots(1, len(states), figsize=(15, 4.4))
for ax, (T, x_cu) in zip(axes, states):
    res = equilibrium(list(phases.values()), x=x_cu, T=T)
    print()
    print(res.summary())
    for cs in res.composition_sets:
        if cs.phase is cu2mg:
            print("  CU2MG site fractions:", cu2mg.site_fractions(cs.y))
    plot_gx(list(phases.values()), res, T=T, ax=ax)
    ax.set_xlabel("x_Cu")
    sets = " + ".join(cs.phase.name for cs in res.composition_sets)
    ax.set_title(f"Cu–Mg, T = {T:g} K, x_Cu = {x_cu:g}\n{sets}  [{res.certificate.level.name}]", fontsize=9)
fig.tight_layout()
out = HERE / "cumg_cef.png"
fig.savefig(out, dpi=130)
print(f"\nwrote {out}")
