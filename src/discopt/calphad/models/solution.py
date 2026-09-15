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
    """A binary solution that mixes A and B on one sublattice; energies are in J per formula unit.

    The formula unit is ``(A,B)_sites`` plus fixed amounts ``fixed = (n_A, n_B)``
    from singleton sublattices (VA contributes nothing). With y = y[0] the site
    fraction of A on the mixing sublattice:

        G = y g_A(T) + (1 − y) g_B(T) + sites·RT (y ln y + (1 − y) ln(1 − y))
            + y (1 − y) Σ_k L_k(T) (2y − 1)^k

    Here g_A and g_B are the endmember energies, and ``gibbs`` returns G/RT.
    Each ``g`` and ``L`` entry is a number or a callable of T, such as
    :class:`~discopt.calphad.models.TPoly` or a database value. With the
    defaults (``sites=1``, no fixed amounts), y is the mole fraction x_A.
    """

    name: str
    elements: tuple[str, str] = ("A", "B")
    g: tuple = (0.0, 0.0)
    L: tuple = ()
    sites: float = 1.0
    fixed: tuple[float, float] = (0.0, 0.0)

    @property
    def atoms_min(self) -> float:
        return float(self.sites + self.fixed[0] + self.fixed[1])

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
        G = ga * xa + gb * xb + self.sites * (ops.xlogx(xa) + ops.xlogx(xb))
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
        ya = y[0]
        return (self.sites * ya + self.fixed[0], self.sites * (1 - ya) + self.fixed[1])

    def initial_y(self, b) -> np.ndarray | None:
        """The site fraction at which one formula unit matches ``b`` in ratio, or None if there is none."""
        b = np.asarray(b, dtype=float)
        y = (b[0] / b.sum() * self.atoms_min - self.fixed[0]) / self.sites
        return np.array([y]) if 0.0 < y < 1.0 else None
