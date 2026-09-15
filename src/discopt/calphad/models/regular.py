"""Binary regular solution: the MVP 1 phase model.

In units of RT, per formula unit, with x the mole fraction of the first element:

    G/RT = x g_A + (1 - x) g_B + x ln x + (1 - x) ln(1 - x) + omega x (1 - x)

Take omega = Omega/RT. With g_A = g_B, a miscibility gap opens once omega > 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from discopt.calphad.ops import NUM

__all__ = ["RegularSolution"]


@dataclass(frozen=True)
class RegularSolution:
    """A single-sublattice binary solution phase ``(A,B)1`` with one interaction term.

    The phase has one independent site fraction, ``y = [x_A]``. Each formula unit
    holds exactly one atom, so ``amounts(y) = (x_A, 1 - x_A)``.
    """

    name: str
    omega: float
    g: tuple[float, float] = (0.0, 0.0)
    elements: tuple[str, str] = ("A", "B")

    # -- the phase-model protocol used by equilibrium/ ---------------------------
    @property
    def n_y(self) -> int:
        return 1

    @property
    def y_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)]

    #: smallest number of atoms in one formula unit, used to bound total moles
    atoms_min: float = 1.0

    def gibbs(self, y, ops=NUM):
        """G/RT per formula unit, evaluated with the given ``ops`` backend."""
        x = y[0]
        gA, gB = self.g
        return gA * x + gB * (1 - x) + ops.xlogx(x) + ops.xlogx(1 - x) + self.omega * x * (1 - x)

    def amounts(self, y):
        """Moles of each element in one formula unit."""
        x = y[0]
        return (x, 1 - x)

    def initial_y(self, b) -> np.ndarray:
        """Site fractions for which a single composition set satisfies the balance ``b``."""
        b = np.asarray(b, dtype=float)
        return np.array([b[0] / b.sum()])

    # -- analytic references (tests and plots only) ------------------------------
    def d2gibbs(self, x):
        """Second derivative of G/RT with respect to x."""
        x = np.asarray(x, dtype=float)
        return 1.0 / x + 1.0 / (1.0 - x) - 2.0 * self.omega

    def spinodal(self) -> tuple[float, float] | None:
        """Compositions where G'' = 0, or None if G is convex."""
        if self.omega <= 2.0:
            return None
        r = math.sqrt(0.25 - 1.0 / (2.0 * self.omega))
        return (0.5 - r, 0.5 + r)

    def binodal(self) -> tuple[float, float] | None:
        """Analytic miscibility-gap compositions (symmetric case, g_A = g_B), or None."""
        if self.g[0] != self.g[1]:
            raise NotImplementedError("analytic binodal only for g_A == g_B")
        if self.omega <= 2.0:
            return None
        lo = self.spinodal()[0]
        x1 = brentq(lambda x: math.log(x / (1 - x)) - self.omega * (2 * x - 1), 1e-300, lo, xtol=1e-15)
        return (x1, 1.0 - x1)
