"""Magnetic ordering contribution: the Inden–Hillert–Jarl (IHJ) model.

Per formula unit, as pycalphad and Thermo-Calc apply it:

    G_mag / RT = ln(β' + 1) · f(τ),    τ = T / T_c'

T_c(y) and β(y) are Redlich–Kister sums of the phase's ``TC`` and ``BMAGN``
parameters. They use the same site-fraction products as the Gibbs-energy
terms. A negative T_c or β is antiferromagnetic. It is divided by the AFM factor
from ``TYPE_DEFINITION ... MAGNETIC <afm> <p>`` (−1 for BCC, −3 for FCC/HCP),
giving the primed values. With A = 518/1125 + (11692/15975)(1/p − 1):

    τ < 1:  f = 1 − [79/(140 p) τ⁻¹ + (474/497)(1/p − 1)(τ³/6 + τ⁹/135 + τ¹⁵/600)] / A
    τ ≥ 1:  f = −(τ⁻⁵/10 + τ⁻¹⁵/315 + τ⁻²⁵/1500) / A

Everything here is written in s = 1/τ = T_c'/T. Above T_c, f is then a
polynomial in s on [0, 1], so a T_c near zero is not a singularity. Below T_c
(s > 1), the negative powers of s are bounded.

The model is piecewise in y, in three places: the sign of T_c, the sign of β,
and whether s is above or below 1. A :data:`Branch` names one piece. A discopt
expression is smooth only within one piece, so the ``EXPR`` backend needs the
branch (``EXPR.with_branch``). The local restricted solve holds each
composition set inside its branch with :func:`branch_constraints`, and switches
branch when a constraint becomes active. The numeric backend picks the branch
per point.

**Bounds.** f is continuous and strictly decreasing in s on all of s ≥ 0: on
the paramagnetic side every term falls, the two branches have the same slope at
s = 1, and on the ordered side the slope only grows more negative, since the
τ-series shrinks as s grows. So the range of f over an s-interval is
[f(s_hi), f(s_lo)], whichever branches the interval meets, and f ≤ f(0) = 0.
The primed values T_c' and β' are nonnegative, piecewise-linear images of T_c
and β, so their ranges follow from the endpoints and from 0 when the interval
contains it. ln(1 + β') is increasing. The ``INTERVAL`` backend uses these
rules, so an s-interval that straddles 1 needs no special treatment.

**Global pricing** splits the domain into the branches an interval evaluation
cannot rule out. Each branch is solved with its sign constraints, and with an
auxiliary variable s = T_c'/T bounded to [0, 1] or [1, s_max]
(``EXPR.with_branch(branch, s=...)``), so the negative powers on the ordered
side never see s near 0. The pieces are closed and the model is continuous
across their edges, so the smallest of their bounds bounds the phase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = ["Branch", "Magnetic", "branch_constraints", "branch_of", "branch_ranges", "ihj"]


class Branch(NamedTuple):
    """One smooth piece of the IHJ model."""

    #: T_c ≤ 0 (antiferromagnetic: divided by the AFM factor)
    tc_negative: bool
    #: β ≤ 0 (divided by the AFM factor)
    beta_negative: bool
    #: T_c' > T, i.e. τ < 1: the magnetically ordered side
    ordered: bool


@dataclass(frozen=True)
class Magnetic:
    """The magnetic description of a phase: AFM factor, structure factor, and the TC/BMAGN terms.

    ``tc`` and ``beta`` are tuples of :class:`~discopt.calphad.models.cef.CEFTerm`.
    Their values are in K (TC) and Bohr magnetons (BMAGN), not in J.
    """

    afm: float
    p: float
    tc: tuple = ()
    beta: tuple = ()

    def __post_init__(self):
        if self.afm == 0:
            raise NotImplementedError("the Xiong magnetic model (AFM factor 0) is not supported yet")
        if not 0 < self.p < 1:
            raise ValueError(f"magnetic structure factor p must be in (0, 1), got {self.p}")

    @property
    def A(self) -> float:
        return 518 / 1125 + (11692 / 15975) * (1 / self.p - 1)


def branch_of(tc: float, beta: float, T: float, afm: float) -> Branch:
    """The branch at numeric T_c and β. The boundaries follow pycalphad: T_c ≤ 0 and β ≤ 0 are antiferromagnetic."""
    tc_neg, beta_neg = tc <= 0, beta <= 0
    tc_eff = tc / afm if tc_neg else tc
    return Branch(bool(tc_neg), bool(beta_neg), bool(tc_eff > T))


def _paramagnetic(s, A):
    s5 = s**5
    s15 = s5 * s5 * s5
    return -(s5 * (1 / 10) + s15 * (1 / 315) + s15 * s5 * s5 * (1 / 1500)) * (1 / A)


def _ordered(s, p, A):
    # τ³ = s⁻³, τ⁹ = s⁻⁹, τ¹⁵ = s⁻¹⁵
    t3 = 1 / (s * s * s)
    t9 = t3 * t3 * t3
    t15 = t9 * t3 * t3
    series = t3 * (1 / 6) + t9 * (1 / 135) + t15 * (1 / 600)
    return 1 - ((79 / (140 * p)) * s + (474 / 497) * (1 / p - 1) * series) * (1 / A)


def _f(s: float, mag: Magnetic) -> float:
    """f at one numeric s ≥ 0."""
    return _ordered(s, mag.p, mag.A) if s > 1 else _paramagnetic(s, mag.A)


def _inv(v: float):
    """An interval holding 1 / v exactly."""
    from discopt.calphad.ops import Interval

    q = 1 / v
    return Interval(math.nextafter(q, -math.inf), math.nextafter(q, math.inf))


def _afm_image(x, afm: float):
    """The range of x' (x, or x / afm where x ≤ 0) over the interval x. It is always ≥ 0."""
    from discopt.calphad.ops import Interval

    x = Interval._coerce(x)
    if x.lo > 0:
        return x
    neg = Interval(x.lo, min(x.hi, 0.0)) * _inv(afm)
    return Interval(max(neg.lo, 0.0), max(neg.hi, x.hi))


def _ihj_interval(mag: Magnetic, tc, beta, T: float):
    """An enclosure of G_mag / RT over the intervals T_c and β (see the module docstring)."""
    from discopt.calphad.ops import Interval

    s = _afm_image(tc, mag.afm) * _inv(T)
    b = _afm_image(beta, mag.afm)
    s_lo = max(s.lo, 0.0)
    f_lo, f_hi = _f(s.hi, mag), _f(s_lo, mag)
    l_lo, l_hi = math.log1p(b.lo), math.log1p(b.hi)
    # libm's pow and log1p are not correctly rounded: pad by a few ulps of the value
    pad = 1e-14
    f = Interval(f_lo - pad * max(1.0, abs(f_lo)), min(f_hi + pad * max(1.0, abs(f_hi)), 0.0))
    ln = Interval(max(l_lo - pad * max(1.0, l_lo), 0.0), l_hi + pad * max(1.0, l_hi))
    return ln * f


def branch_ranges(mag: Magnetic, tc, beta, T: float) -> list[tuple[Branch, tuple[float, float]]]:
    """The branches that intervals T_c and β may reach, each with an enclosure of s = T_c'/T on it.

    A branch left out cannot occur anywhere the intervals hold. The s-range of
    a kept branch is already cut to its side of 1.
    """
    from discopt.calphad.ops import Interval

    tc, beta = Interval._coerce(tc), Interval._coerce(beta)
    signs = []
    if tc.lo <= 0:
        signs.append((True, _afm_image(Interval(tc.lo, min(tc.hi, 0.0)), mag.afm)))
    if tc.hi > 0:
        signs.append((False, Interval(max(tc.lo, 0.0), tc.hi)))
    betas = [neg for neg, ok in ((True, beta.lo <= 0), (False, beta.hi > 0)) if ok]
    out = []
    for tc_neg, eff in signs:
        s = eff * _inv(T)
        s_lo = max(s.lo, 0.0)
        for ordered in (False, True):
            if ordered and s.hi <= 1:
                continue
            if not ordered and s_lo > 1:
                continue
            rng = (max(s_lo, 1.0), s.hi) if ordered else (s_lo, min(s.hi, 1.0))
            out += [(Branch(tc_neg, b_neg, ordered), rng) for b_neg in betas]
    return out


def ihj(mag: Magnetic, tc, beta, T: float, ops, branch: Branch | None = None):
    """G_mag / RT for the given T_c and β, on backend ``ops``.

    ``NUM`` chooses the branch per element (array inputs broadcast). ``EXPR``
    uses ``branch``, or ``ops.branch`` if ``branch`` is None, and the auxiliary
    variable ``ops.s`` for T_c'/T when it is set. ``INTERVAL`` encloses every
    branch at once.
    """
    if ops.name == "num":
        tc = np.asarray(tc, dtype=float)
        beta = np.asarray(beta, dtype=float)
        tc_eff = np.where(tc <= 0, tc / mag.afm, tc)
        beta_eff = np.where(beta <= 0, beta / mag.afm, beta)
        s = tc_eff / T
        with np.errstate(all="ignore"):
            s_ord = np.where(s > 1, s, 2.0)  # a safe value where the ordered formula is not used
            f = np.where(s > 1, _ordered(s_ord, mag.p, mag.A), _paramagnetic(s, mag.A))
        out = np.log1p(beta_eff) * f
        return out[()] if out.ndim == 0 else out
    if ops.name == "expr":
        import discopt.modeling as dm

        branch = branch if branch is not None else getattr(ops, "branch", None)
        if branch is None:
            raise ValueError(
                "a magnetic phase is piecewise in its site fractions; build its discopt expression "
                "with EXPR.with_branch(phase.magnetic_branch(y, T))"
            )
        tc_eff = tc * (1 / mag.afm) if branch.tc_negative else tc
        beta_eff = beta * (1 / mag.afm) if branch.beta_negative else beta
        s = getattr(ops, "s", None)
        if s is None:
            s = tc_eff * (1 / T)
        f = _ordered(s, mag.p, mag.A) if branch.ordered else _paramagnetic(s, mag.A)
        return dm.log(1 + beta_eff) * f
    if ops.name == "interval":
        return _ihj_interval(mag, tc, beta, T)
    raise NotImplementedError(f"no magnetic contribution for backend {ops.name!r}")


def branch_constraints(mag: Magnetic, tc, beta, T: float, branch: Branch) -> list:
    """Expressions that are ≥ 0 exactly on ``branch``, each scaled to order one.

    The list always has three entries: the sign of T_c, the sign of β, and the
    ordered side. An entry is None when the phase has no parameters for it, so
    the entry is constant. The restricted solve adds each remaining entry as
    ``expr >= 0``. At the solution, an active constraint (value zero) means the
    set has reached the edge of its piece.
    """
    scale = 1 / T
    tc_eff = tc * (1 / mag.afm) if branch.tc_negative else tc
    return [
        ((-tc if branch.tc_negative else tc) * scale) if mag.tc else None,
        (-beta if branch.beta_negative else beta) if mag.beta else None,
        (((tc_eff - T) if branch.ordered else (T - tc_eff)) * scale) if mag.tc else None,
    ]
