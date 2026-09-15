"""Pieces shared by the restricted solve, pricing, and certification."""

from __future__ import annotations

import numpy as np

from discopt.calphad.ops import NUM

__all__ = ["DOMAIN_FLOOR", "inner_bounds", "tangent_distance", "amount_matrix"]

#: Distance kept between site-fraction variables and their bounds in every
#: discopt model. With discopt 0.8.0, an ``entropy`` argument whose lower bound is
#: exactly 0 leaves the global solve without a valid bound (discopt#1242).
#: Pricing bounds the excluded boundary slabs separately, with interval
#: arithmetic, so the certificate still covers the whole domain.
DOMAIN_FLOOR = 1e-9


def inner_bounds(phase, floor: float = DOMAIN_FLOOR) -> list[tuple[float, float]]:
    """The phase's site-fraction box, shrunk by ``floor`` on every side."""
    return [(lo + floor, hi - floor) for lo, hi in phase.y_bounds]


def tangent_distance(phase, y, mu, ops=NUM):
    """Ψ(y) = G(y) − μ·A(y), in units of RT per formula unit."""
    return phase.gibbs(y, ops) - sum(float(m) * a for m, a in zip(mu, phase.amounts(y)))


def amount_matrix(sets) -> np.ndarray:
    """E × K matrix whose column k gives the element amounts per formula unit of set k."""
    return np.array([[float(a) for a in s.phase.amounts(s.y)] for s in sets], dtype=float).T
