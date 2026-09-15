"""Binary regular solution: a certified miscibility gap (omega = 3) and a convex control (omega = 1.5).

Run with ``uv run python examples/regular_solution_gap.py``. Writes ``regular_solution_gap.png`` next to this file.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from discopt.calphad import RegularSolution, equilibrium  # noqa: E402
from discopt.calphad.plot import plot_gx  # noqa: E402

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, omega in zip(axes, (3.0, 1.5)):
    phase = RegularSolution("LIQ", omega=omega)
    res = equilibrium(phase, x=0.5, eps=1e-5)
    print(f"omega = {omega}\n{res.summary()}\n")
    plot_gx(phase, res, ax=ax)
    ax.set_title(f"ω = {omega}: {res.certificate.level.name}", fontsize=10)

fig.tight_layout()
out = Path(__file__).with_suffix(".png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
