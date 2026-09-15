"""Stoichiometric (line) compound: no internal degrees of freedom."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models.tfunc import coef

__all__ = ["StoichiometricPhase"]


@dataclass(frozen=True)
class StoichiometricPhase:
    """A compound with fixed element amounts per formula unit.

    ``g`` is the Gibbs energy of one formula unit in J (a number or a callable of T).
    Because the phase has no site fractions (``n_y = 0``), pricing it is an exact
    evaluation and needs no global solve.
    """

    name: str
    elements: tuple[str, ...]
    stoichiometry: tuple[float, ...]
    g: object = 0.0

    def __post_init__(self):
        if len(self.stoichiometry) != len(self.elements):
            raise ValueError("stoichiometry needs one entry per element")
        if min(self.stoichiometry) < 0 or sum(self.stoichiometry) <= 0:
            raise ValueError("stoichiometry must be nonnegative with a positive total")

    @property
    def n_y(self) -> int:
        return 0

    @property
    def y_bounds(self) -> list[tuple[float, float]]:
        return []

    @property
    def atoms_min(self) -> float:
        return float(sum(self.stoichiometry))

    def gibbs(self, y=(), ops=None, T=None):
        if T is None and callable(self.g):
            raise ValueError(f"{self.name}: temperature T is required")
        return coef(self.g, T) / (R_GAS * T) if T is not None else float(self.g)

    def amounts(self, y=()):
        return tuple(float(s) for s in self.stoichiometry)

    def initial_y(self, b):
        """An empty site-fraction vector if one formula unit matches ``b`` in ratio, otherwise None."""
        b = np.asarray(b, dtype=float)
        s = np.asarray(self.stoichiometry, dtype=float)
        return np.empty(0) if np.allclose(b / b.sum(), s / s.sum(), rtol=0, atol=1e-12) else None
