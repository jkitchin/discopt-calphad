# discopt-calphad

Certified CALPHAD phase equilibrium as a discopt plugin (`import discopt.calphad`).

**Method.** A restricted multiphase Gibbs-energy minimization over the active
composition sets is solved locally. Its elemental-balance multipliers are the
chemical potentials μ. For every phase, discopt's global solver then minimizes
the tangent-plane distance Ψ(y) = G(y) − μ·A(y):

- a negative upper bound adds a new composition set (this is how miscibility
  gaps are found);
- a lower bound ≥ −ε for every phase certifies the result to within ε·RT
  per mole of formula units.

**Status.** Pre-alpha. MVP 1 is a binary regular solution with a miscibility
gap. See `docs/architecture.md` for the design and
`docs/symbolic-diagrams-fitting.md` for the planned symbolic model, phase
diagram, and fitting layers.

```python
from discopt.calphad import RegularSolution, equilibrium

phase = RegularSolution("LIQ", omega=3.0)   # G/RT, omega = Omega/RT
res = equilibrium(phase, x=0.5, eps=1e-5)
print(res.certificate, [cs.x for cs in res.composition_sets], res.mu)
```

## Development

```bash
uv sync
uv run pytest
uv run python examples/regular_solution_gap.py   # writes examples/regular_solution_gap.png
```
