"""Einstein and two-state (liquid–amorphous) contributions, on every backend.

Both follow pycalphad 0.11.2 (``Model.einstein_energy``, ``Model.twostate_energy``)
and are per formula unit, in units of RT:

- **Einstein.** THETA parameters hold ln θ_E; their Redlich–Kister sum L gives
  θ = e^L, and

      G_E / RT = 1.5 θ/T + 3 ln(1 − e^(−θ/T)).

  It is increasing in L: dG_E/dL / RT = 1.5 z coth(z/2) with z = θ/T, which is
  positive and increasing in z.
- **Two-state.** GD parameters hold the energy ΔG_d (J/mol) of the
  amorphous-to-liquid transition; with their sum g,

      G_2st / RT = −ln(1 + e^(−g/RT)).

  It is increasing in g, with derivative 1 / (RT (1 + e^(g/RT))), positive and
  decreasing in g.

Monotone in their argument, both have exact interval rules: the endpoints.
Their derivatives are monotone too, which gives the ``DUAL`` rule.
"""

from __future__ import annotations

import math

import numpy as np

from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.ops import Dual, Interval

__all__ = ["einstein", "two_state"]

_PAD = 1e-14  # relative pad on exp/log compositions: libm is not correctly rounded


def _padded(lo: float, hi: float) -> Interval:
    return Interval(lo - _PAD * max(1.0, abs(lo)), hi + _PAD * max(1.0, abs(hi)))


def _einstein_num(L, T):
    z = np.exp(L) / T
    return 1.5 * z + 3.0 * np.log(-np.expm1(-z))


def _einstein_slope(L: float, T: float) -> float:
    z = math.exp(L) / T
    return 1.5 * z / math.tanh(0.5 * z)


def _two_state_num(g, T):
    return -np.logaddexp(0.0, -np.asarray(g, dtype=float) / (R_GAS * T))


def _two_state_slope(g: float, T: float) -> float:
    RT = R_GAS * T
    return 1.0 / (RT * (1.0 + math.exp(min(g / RT, 700.0))))


def _monotone(x, T, f, slope, increasing_slope: bool):
    """Interval or :class:`Dual` rule for an increasing f whose slope is monotone in x."""
    if isinstance(x, Dual):
        v, c = _monotone(x.v, T, f, slope, increasing_slope), _monotone(x.c, T, f, slope, increasing_slope)
        a, b = slope(x.v.lo, T), slope(x.v.hi, T)
        g = _padded(*((a, b) if increasing_slope else (b, a)))
        return Dual(v, c, tuple(None if d is None else g * d for d in x.d))
    x = Interval._coerce(x)
    return _padded(float(f(x.lo, T)), float(f(x.hi, T)))


def einstein(lntheta, T: float, ops):
    """G_E / RT per formula unit for the sum ``lntheta`` of the THETA parameters, on backend ``ops``."""
    if ops.name == "num":
        out = _einstein_num(np.asarray(lntheta, dtype=float), T)
        return out[()] if np.ndim(out) == 0 else out
    if ops.name == "expr":
        import discopt.modeling as dm

        theta_t = dm.exp(lntheta) * (1.0 / T)
        return 1.5 * theta_t + 3.0 * dm.log(1 - dm.exp(-theta_t))
    if ops.name in ("interval", "dual"):
        return _monotone(lntheta, T, _einstein_num, _einstein_slope, increasing_slope=True)
    raise NotImplementedError(f"no Einstein contribution for backend {ops.name!r}")


def two_state(gd, T: float, ops):
    """G_2st / RT per formula unit for the sum ``gd`` (J/mol) of the GD parameters, on backend ``ops``."""
    if ops.name == "num":
        out = _two_state_num(gd, T)
        return out[()] if np.ndim(out) == 0 else out
    if ops.name == "expr":
        import discopt.modeling as dm

        return -dm.log1p(dm.exp(gd * (-1.0 / (R_GAS * T))))
    if ops.name in ("interval", "dual"):
        return _monotone(gd, T, _two_state_num, _two_state_slope, increasing_slope=False)
    raise NotImplementedError(f"no two-state contribution for backend {ops.name!r}")
