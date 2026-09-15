"""Binary substitutional solution with a Redlich–Kister excess term."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models.tfunc import coef
from discopt.calphad.ops import NUM

__all__ = ["BinarySolution"]


@dataclass(frozen=True)
class BinarySolution:
    """A single-sublattice binary solution ``(A,B)1``; energies are in J/mol.

    With x_A = y[0] and x_B = 1 − x_A:

        G = x_A g_A(T) + x_B g_B(T) + RT (x_A ln x_A + x_B ln x_B)
            + x_A x_B Σ_k L_k(T) (x_A − x_B)^k

    ``gibbs`` returns G/RT. Each ``g`` and ``L`` entry is either a number or a
    callable of T, such as :class:`~discopt.calphad.models.TPoly`.
    """

    name: str
    elements: tuple[str, str] = ("A", "B")
    g: tuple = (0.0, 0.0)
    L: tuple = ()
    atoms_min: float = 1.0

    @property
    def n_y(self) -> int:
        return 1

    @property
    def y_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)]

    def gibbs(self, y, ops=NUM, T=None):
        if T is None:
            raise ValueError(f"{self.name}: temperature T is required")
        RT = R_GAS * T
        xa = y[0]
        xb = 1 - xa
        ga, gb = (coef(c, T) / RT for c in self.g)
        G = ga * xa + gb * xb + ops.xlogx(xa) + ops.xlogx(xb)
        if self.L:
            d = xa - xb
            rk, dk = 0.0, 1.0
            for k, Lk in enumerate(self.L):
                if k:
                    dk = dk * d
                rk = rk + (coef(Lk, T) / RT) * dk
            G = G + xa * xb * rk
        return G

    def amounts(self, y):
        xa = y[0]
        return (xa, 1 - xa)

    def initial_y(self, b) -> np.ndarray:
        b = np.asarray(b, dtype=float)
        return np.array([b[0] / b.sum()])
