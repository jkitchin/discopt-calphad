"""Phase models.

Every phase model follows the same protocol:

- ``name`` and ``elements``;
- ``n_y`` and ``y_bounds`` for the independent site-fraction variables (``n_y`` may be 0);
- ``gibbs(y, ops, T)``: G/RT per formula unit, written once for every ``ops`` backend;
- ``amounts(y)``: moles of each element per formula unit;
- ``atoms_min``: the smallest number of atoms per formula unit;
- ``initial_y(b)``: a composition that satisfies the balance with one set, or None.
"""

from discopt.calphad.models.cef import CEFPhase, CEFTerm
from discopt.calphad.models.regular import RegularSolution
from discopt.calphad.models.solution import BinarySolution
from discopt.calphad.models.stoichiometric import StoichiometricPhase
from discopt.calphad.models.tfunc import TPoly, coef

__all__ = ["BinarySolution", "CEFPhase", "CEFTerm", "RegularSolution", "StoichiometricPhase", "TPoly", "coef"]
