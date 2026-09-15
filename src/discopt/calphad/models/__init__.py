"""Phase models.

Every phase model follows the same protocol:

- ``name`` and ``elements``;
- ``n_y`` and ``y_bounds`` for the independent site-fraction variables;
- ``gibbs(y, ops)``: G/RT per formula unit, written once for every ``ops`` backend;
- ``amounts(y)``: moles of each element per formula unit;
- ``atoms_min``: the smallest number of atoms per formula unit;
- ``initial_y(b)``: a starting composition that satisfies the balance.
"""

from discopt.calphad.models.regular import RegularSolution

__all__ = ["RegularSolution"]
