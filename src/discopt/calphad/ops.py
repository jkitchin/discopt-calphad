"""Backends for evaluating one Gibbs-energy expression three ways.

A phase model writes its Gibbs energy once, as ``gibbs(y, ops)``, using only
arithmetic and ``ops.xlogx``. Which backend it runs on decides the result:

- ``NUM``: floats or numpy arrays, with ``0 ln 0 = 0``.
- ``EXPR``: a discopt expression. ``x ln x`` becomes discopt's ``entropy``
  node, which has dedicated McCormick and interval rules.
- ``INTERVAL``: an enclosure of the range over a box, rounded outward. Pricing
  uses it to bound the thin slabs next to the domain boundary that are left
  out of the global solve (see ``pricing.py``).

The plan is to replace this with the SymPy term layer described in
``docs/symbolic-diagrams-fitting.md``. This module is the MVP stand-in.
"""

from __future__ import annotations

import math

import numpy as np
from discopt.modeling.core import FunctionCall

__all__ = ["NUM", "EXPR", "INTERVAL", "Interval"]

_INV_E = math.exp(-1.0)


class _Num:
    name = "num"

    @staticmethod
    def xlogx(x):
        x = np.asarray(x, dtype=float)
        safe = np.where(x > 0, x, 1.0)
        out = np.where(x > 0, x * np.log(safe), 0.0)
        return out[()] if out.ndim == 0 else out


class _Expr:
    name = "expr"

    @staticmethod
    def xlogx(x):
        return FunctionCall("entropy", x)


def _down(v: float) -> float:
    return math.nextafter(v, -math.inf)


def _up(v: float) -> float:
    return math.nextafter(v, math.inf)


class Interval:
    """A closed interval ``[lo, hi]`` with outward-rounded arithmetic.

    Each operation pushes its endpoints out by one ulp. That is enough for
    ``+``, ``-`` and ``*`` under round-to-nearest. ``xlogx`` gets a larger
    relative pad, because libm's ``log`` is not guaranteed to be correctly
    rounded.
    """

    __slots__ = ("lo", "hi")

    def __init__(self, lo: float, hi: float | None = None):
        hi = lo if hi is None else hi
        if not lo <= hi:
            raise ValueError(f"empty interval [{lo}, {hi}]")
        self.lo, self.hi = float(lo), float(hi)

    @staticmethod
    def _coerce(o) -> Interval:
        return o if isinstance(o, Interval) else Interval(float(o))

    def __add__(self, o):
        o = self._coerce(o)
        return Interval(_down(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, o):
        return self + (-self._coerce(o))

    def __rsub__(self, o):
        return self._coerce(o) + (-self)

    def __mul__(self, o):
        o = self._coerce(o)
        p = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(_down(min(p)), _up(max(p)))

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"Interval({self.lo!r}, {self.hi!r})"


class _Interval:
    name = "interval"

    @staticmethod
    def xlogx(x) -> Interval:
        x = Interval._coerce(x)
        if x.lo < 0:
            # outward rounding of e.g. 1 - [1 - d, 1] can reach just below 0; the
            # argument is a site fraction, so this is rounding noise, not a real value
            if x.lo < -1e-12:
                raise ValueError(f"xlogx is undefined below 0 (got {x})")
            x = Interval(0.0, max(x.hi, 0.0))

        def f(t: float) -> float:
            return t * math.log(t) if t > 0 else 0.0

        fl, fu = f(x.lo), f(x.hi)
        lo = -_INV_E if x.lo <= _INV_E <= x.hi else min(fl, fu)
        hi = max(fl, fu)
        pad = 4e-16 * max(1.0, abs(lo), abs(hi))
        return Interval(lo - pad, hi + pad)


NUM = _Num()
EXPR = _Expr()
INTERVAL = _Interval()
