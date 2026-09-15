"""Temperature-dependent coefficients."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["TPoly", "coef"]


@dataclass(frozen=True)
class TPoly:
    """``a + b T + c T ln T + d T² + e / T``: the usual CALPHAD G(T) form over one range, in J/mol."""

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 0.0

    def __call__(self, T):
        """Evaluate at T, which may be a float or a numpy array."""
        tlnt = self.c * T * (np.log(T) if isinstance(T, np.ndarray) else math.log(T)) if self.c else 0.0
        return self.a + self.b * T + tlnt + self.d * T * T + self.e / T


def coef(c, T: float | None) -> float:
    """The value of a coefficient at temperature T; ``c`` is a constant or a callable of T."""
    if callable(c):
        if T is None:
            raise ValueError("a temperature-dependent coefficient needs T")
        return float(c(T))
    return float(c)
