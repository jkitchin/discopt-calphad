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

__all__ = ["NUM", "EXPR", "INTERVAL", "DUAL", "Dual", "Interval"]

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

    def __init__(self, branch=None, s=None):
        #: the piece of a piecewise model to lower (see :mod:`discopt.calphad.models.magnetic`)
        self.branch = branch
        #: an auxiliary discopt variable standing for T_c'/T on that piece, or None
        self.s = s

    def with_branch(self, branch, s=None) -> _Expr:
        """This backend, lowering a piecewise model on ``branch`` (with ``s`` for T_c'/T if given)."""
        return _Expr(branch, s)

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
        if isinstance(o, Dual):
            return NotImplemented
        o = self._coerce(o)
        return Interval(_down(self.lo + o.lo), _up(self.hi + o.hi))

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        return self + (-self._coerce(o))

    def __rsub__(self, o):
        return self._coerce(o) + (-self)

    def __mul__(self, o):
        if isinstance(o, Dual):
            return NotImplemented
        o = self._coerce(o)
        # endpoint products take 0 · ∞ = 0, so an unbounded factor times [0, w] stays well defined
        p = [0.0 if a == 0 or b == 0 else a * b for a in (self.lo, self.hi) for b in (o.lo, o.hi)]
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


def _log(x: Interval) -> Interval:
    """An enclosure of ln over ``x`` (−∞ at 0), padded for libm's rounding."""
    lo = math.log(x.lo) if x.lo > 0 else -math.inf
    hi = math.log(x.hi) if x.hi > 0 else -math.inf
    pad = lambda v: 4e-16 * max(1.0, abs(v)) if math.isfinite(v) else 0.0  # noqa: E731
    return Interval(lo - pad(lo), hi + pad(hi))


class Dual:
    """A first-order Taylor model of a quantity q over a box, for mean-value bounds.

    For an expansion point c in the box, three enclosures: ``v`` ⊇ q(box),
    ``c`` ⊇ q(c), and gradients ``d`` with q(y) ∈ c + Σ_k d_k (y_k − c_k) for
    every y in the box. ``d`` has one entry per expanded direction, and
    ``None`` is an exact zero: a term that does not depend on a direction adds
    nothing to it, even when its own derivative is unbounded (ln 0 in
    ``xlogx``). A quantity known only by its range is ``Dual.const(range)``.
    """

    __slots__ = ("v", "c", "d")

    def __init__(self, v: Interval, c: Interval, d: tuple):
        self.v, self.c, self.d = v, c, d

    @staticmethod
    def const(v, n: int) -> Dual:
        v = Interval._coerce(v)
        return Dual(v, v, (None,) * n)

    def _other(self, o) -> Dual:
        return o if isinstance(o, Dual) else Dual.const(o, len(self.d))

    def __add__(self, o):
        o = self._other(o)
        return Dual(self.v + o.v, self.c + o.c, tuple(_dsum(a, b) for a, b in zip(self.d, o.d)))

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.c, tuple(None if a is None else -a for a in self.d))

    def __sub__(self, o):
        return self + (-self._other(o))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        # q r − q(c) r(c) = (q − q(c)) r + q(c) (r − r(c))
        o = self._other(o)
        d = tuple(_dsum(None if a is None else a * o.v, None if b is None else b * self.c) for a, b in zip(self.d, o.d))
        return Dual(self.v * o.v, self.c * o.c, d)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f"Dual({self.v!r}, {self.c!r}, {self.d!r})"


def _dsum(a, b):
    return b if a is None else a if b is None else a + b


def _nonneg(x: Interval) -> Interval:
    # outward rounding of a site fraction can reach just below 0 (see _Interval.xlogx)
    return x if x.lo >= 0 else Interval(0.0, max(x.hi, 0.0))


class _DualInterval:
    """``INTERVAL`` carried as first-order Taylor models: values are :class:`Dual`."""

    name = "dual"

    @staticmethod
    def xlogx(x):
        if not isinstance(x, Dual):
            return _Interval.xlogx(x)  # a constant
        g = _log(_nonneg(x.v)) + 1.0  # d(x ln x)/dx over the range of x
        return Dual(_Interval.xlogx(x.v), _Interval.xlogx(x.c), tuple(None if a is None else g * a for a in x.d))


NUM = _Num()
EXPR = _Expr()
INTERVAL = _Interval()
DUAL = _DualInterval()
