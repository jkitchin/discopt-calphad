# The phase-model protocol

`equilibrium`, pricing and certification work with any object that has these
members. The built-in models are `RegularSolution`, `BinarySolution`,
`StoichiometricPhase` and `CEFPhase`. A custom model only has to follow the
protocol; no base class is needed.

| member | meaning |
|---|---|
| `name: str` | phase name, used in results and certificates |
| `elements: tuple[str, ...]` | the system's elements, identical for every phase in one solve |
| `n_y: int` | number of independent site-fraction variables (0 for a compound) |
| `y_bounds: list[(lo, hi)]` | box for each variable; the domain must be a box |
| `atoms_min: float` | fewest atoms in one formula unit over the box, > 0; bounds the total amount in the certificate |
| `gibbs(y, ops=NUM, T=None)` | **G/RT per formula unit**, written once with arithmetic and `ops.xlogx` |
| `amounts(y)` | tuple of moles of each element in one formula unit |
| `initial_y(b)` | variables whose composition matches the element amounts `b` in ratio, or `None` if the phase cannot |

`gibbs` is evaluated on three backends, which is what makes certification
possible:

- `NUM`: floats or numpy arrays. It must broadcast when `y` is a list of
  arrays, because pricing samples it vectorized.
- `EXPR`: `y` is a list of discopt variables. The result is a discopt
  expression that POUNCE and the global solver use.
- `INTERVAL`: `y` is a list of `Interval`s. The result must enclose the range
  over the box; it bounds the thin boundary slabs.

Only `+`, `-`, `*`, numbers, and `ops.xlogx(v)` (v ln v, with 0 ln 0 = 0) work
on all three. Write `ops.xlogx(v)`, never `v * log(v)`: discopt lowers it to its
`entropy` node, which has exact convex relaxations.

```python
import numpy as np

from discopt.calphad import equilibrium
from discopt.calphad.ops import NUM


class Subregular:
    """(A,B)1 with G/RT = xlnx terms + x(1-x)(w0 + w1(2x-1)), in RT units."""

    def __init__(self, name, w0, w1, elements=("A", "B")):
        self.name, self.w0, self.w1, self.elements = name, w0, w1, elements

    n_y = 1
    y_bounds = [(0.0, 1.0)]
    atoms_min = 1.0

    def gibbs(self, y, ops=NUM, T=None):
        x = y[0]
        return ops.xlogx(x) + ops.xlogx(1 - x) + x * (1 - x) * (self.w0 + self.w1 * (2 * x - 1))

    def amounts(self, y):
        return (y[0], 1 - y[0])

    def initial_y(self, b):
        b = np.asarray(b, dtype=float)
        return np.array([b[0] / b.sum()])


res = equilibrium(Subregular("LIQ", 3.0, 0.8), x=0.5, certify="none")
print(res.summary())
```

```python
# slow
print(equilibrium(Subregular("LIQ", 3.0, 0.8), x=0.5).certificate.level.name)  # CERTIFIED_TO_TOLERANCE
```

Rules that keep the certificate valid:

- `gibbs` must be the same function on every backend. A branch on the value of
  `y` (`if x > 0.5`) cannot be lowered. A piecewise model needs the branch
  machinery described for the magnetic term in `docs/architecture.md` §9.18.
- `amounts` must be linear in the site fractions. The amount bound
  `atoms_min` assumes it.
- Every phase in one solve must have the same `elements` tuple. A phase
  without some element has zero amounts for it.
- `T` is passed through. A model in RT units can ignore it, as
  `RegularSolution` does.
