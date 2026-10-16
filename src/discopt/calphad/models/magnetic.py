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

__all__ = ["Branch", "Magnetic", "XBranch", "branch_constraints", "branch_of", "branch_ranges", "ihj", "xiong"]


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
    #: NT terms (Néel temperature), used by the Xiong model only
    nt: tuple = ()

    def __post_init__(self):
        if not 0 < self.p < 1:
            raise ValueError(f"magnetic structure factor p must be in (0, 1), got {self.p}")
        if self.nt and not self.xiong:
            raise ValueError("NT parameters belong to the Xiong model (AFM factor 0)")

    @property
    def A(self) -> float:
        return 518 / 1125 + (11692 / 15975) * (1 / self.p - 1)

    @property
    def xiong(self) -> bool:
        """True for the Xiong model (AFM factor 0), False for Inden–Hillert–Jarl."""
        return self.afm == 0

    @property
    def quantities(self) -> tuple[str, ...]:
        """The composition-dependent quantities the model needs, in the order :meth:`term` takes them."""
        return ("tc", "nt", "beta") if self.xiong else ("tc", "beta")

    # ------------------------------------------------------------------ the protocol used by phase models

    def term(self, q, T: float, ops, branch=None):
        """G_mag / RT for the quantities ``q`` (see :attr:`quantities`) on backend ``ops``."""
        if self.xiong:
            return xiong(self, *q, T, ops, branch)
        return ihj(self, *q, T, ops, branch)

    def branch_at(self, q, T: float):
        """The smooth piece at numeric quantities ``q``."""
        if self.xiong:
            return XBranch(_xstate(q[0], T), _xstate(q[1], T))
        return branch_of(q[0], q[1], T, self.afm)

    def constraints(self, q, T: float, branch) -> list:
        """Expressions ≥ 0 exactly on ``branch`` (None for a constant entry); see :meth:`neighbor`."""
        if self.xiong:
            return _xiong_constraints(self, q[0], q[1], T, branch)
        return branch_constraints(self, q[0], q[1], T, branch)

    def neighbor(self, branch, i: int):
        """The branch across the edge where constraint ``i`` of :meth:`constraints` is active."""
        if self.xiong:
            return _xiong_neighbor(branch, i)
        return branch._replace(**{branch._fields[i]: not branch[i]})

    def ranges(self, q, T: float) -> list:
        """``[(branch, [(s_lo, s_hi), ...]), ...]`` for interval quantities ``q``: the branches they may reach.

        Each range bounds one auxiliary variable s = X'/T on that branch, in the
        order :meth:`pricing_rows` and the ``EXPR`` backend take them.
        """
        if self.xiong:
            return _xiong_ranges(self, q[0], q[1], T)
        return [(b, [r]) for b, r in branch_ranges(self, q[0], q[1], T)]

    def pricing_rows(self, q, T: float, branch, s) -> tuple[list, list]:
        """``(equalities, inequalities)`` holding ``q`` on ``branch``, given the auxiliary variables ``s``.

        ``s`` has one entry per range of :meth:`ranges` (None for a range of
        zero width). Each s is tied to its X'/T by an equality, and its bounds
        replace the branch's side constraint.
        """
        if self.xiong:
            return _xiong_rows(self, q[0], q[1], T, branch, s)
        tc, beta = q
        rows = branch_constraints(self, tc, beta, T, branch)
        s0 = s[0] if s else None
        if s0 is None:
            return [], [g for g in rows if g is not None]
        tc_eff = tc * (1 / self.afm) if branch.tc_negative else tc
        return [s0 - tc_eff * (1 / T)], [g for g in rows[:2] if g is not None]


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
        if isinstance(s, (list, tuple)):
            s = s[0] if s else None
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


# ---------------------------------------------------------------------------- the Xiong model


class XBranch(NamedTuple):
    """One smooth piece of the Xiong model: the state of T_C and of T_N.

    A state is 0 for X ≤ 0 (no contribution), 1 for 0 < X ≤ T (τ ≥ 1), and 2
    for X > T (τ < 1, the ordered side).
    """

    tc: int
    nt: int


#: Xiong et al., Calphad 39 (2012) 11–20, eqs. 4–6
_XA, _XB = 0.38438376, 0.63570895


def _xstate(x: float, T: float) -> int:
    return 0 if x <= 0 else (1 if x <= T else 2)


def _xD(p: float) -> float:
    return 0.33471979 + 0.49649686 * (1 / p - 1)


def _x_para(s, p):
    """g on 0 ≤ s ≤ 1 (τ ≥ 1): a polynomial in s = X/T."""
    s7 = s**7
    s21 = s7 * s7 * s7
    return -(s7 * (1 / 21) + s21 * (1 / 630) + s21 * s7 * s7 * (1 / 2975) + s21 * s21 * s7 * (1 / 8232)) * (1 / _xD(p))


def _x_ord(s, p):
    """g on s ≥ 1 (τ ≤ 1). The τ²¹/1617 term is inside the (1/p − 1) bracket, as in Xiong et al.

    pycalphad 0.11.2 adds it outside the bracket, which makes g discontinuous
    at τ = 1 unless 1/p − 1 = 1/0.63570895. Here g and its slope are continuous
    at s = 1 for every p.
    """
    t3 = 1 / (s * s * s)
    t9 = t3 * t3 * t3
    t15 = t9 * t3 * t3
    t21 = t15 * t3 * t3
    series = t3 * (1 / 6) + t9 * (1 / 135) + t15 * (1 / 600) + t21 * (1 / 1617)
    return 1 - ((_XA / p) * s + _XB * (1 / p - 1) * series) * (1 / _xD(p))


def _xg_num(s, p):
    s = np.asarray(s, dtype=float)
    with np.errstate(all="ignore"):
        so = np.where(s > 1, s, 2.0)
        out = np.where(s <= 0, 0.0, np.where(s > 1, _x_ord(so, p), _x_para(np.clip(s, 0, 1), p)))
    return out


def _xg_scalar(s: float, p: float) -> float:
    return 0.0 if s <= 0 else (_x_ord(s, p) if s > 1 else _x_para(s, p))


def xiong(mag: Magnetic, tc, nt, beta, T: float, ops, branch=None):
    """G_mag / RT = ln(β + 1) [g(T_C/T) + g(T_N/T)] (Xiong et al. 2012), on backend ``ops``.

    g is 0 where its temperature is ≤ 0. There is no AFM factor: β enters as
    it is and must stay above −1. ``EXPR`` needs a :class:`XBranch` and takes the
    auxiliary variables ``ops.s`` (one per positive state, T_C first).
    """
    p = mag.p
    if ops.name == "num":
        sc = np.asarray(tc, dtype=float) / T
        sn = np.asarray(nt, dtype=float) / T
        out = np.log1p(np.asarray(beta, dtype=float)) * (_xg_num(sc, p) + _xg_num(sn, p))
        return out[()] if np.ndim(out) == 0 else out
    if ops.name == "expr":
        import discopt.modeling as dm

        branch = branch if branch is not None else getattr(ops, "branch", None)
        if branch is None:
            raise ValueError("a Xiong magnetic phase is piecewise; build its expression with EXPR.with_branch")
        aux = list(getattr(ops, "s", None) or [])
        g = 0.0
        for x, state in ((tc, branch.tc), (nt, branch.nt)):
            if state == 0:
                continue
            s = aux.pop(0) if aux else None
            if s is None:
                s = x * (1 / T)
            g = g + (_x_para(s, p) if state == 1 else _x_ord(s, p))
        return dm.log(1 + beta) * g
    if ops.name == "interval":
        from discopt.calphad.ops import Interval

        pad = 1e-14
        total = Interval(0.0)
        for x in (tc, nt):
            s = Interval._coerce(x) * _inv(T)
            lo, hi = _xg_scalar(max(s.hi, 0.0), p), _xg_scalar(max(s.lo, 0.0), p)
            total = total + Interval(lo - pad * max(1.0, abs(lo)), min(hi + pad * max(1.0, abs(hi)), 0.0))
        b = Interval._coerce(beta)
        if b.lo <= -1:
            raise ValueError(f"Xiong model: β = {b} reaches −1, where ln(β + 1) is undefined")
        l_lo, l_hi = math.log1p(b.lo), math.log1p(b.hi)
        ln = Interval(l_lo - pad * max(1.0, abs(l_lo)), l_hi + pad * max(1.0, abs(l_hi)))
        return ln * total
    raise NotImplementedError(f"no magnetic contribution for backend {ops.name!r}")


def _xiong_constraints(mag, tc, nt, T, branch) -> list:
    """Four entries, [T_C low edge, T_C high edge, T_N low edge, T_N high edge], each ≥ 0 on ``branch``."""
    out = []
    for x, state, has in ((tc, branch.tc, bool(mag.tc)), (nt, branch.nt, bool(mag.nt))):
        if not has:
            out += [None, None]
        elif state == 0:
            out += [None, -x * (1 / T)]  # X ≤ 0; its edge X = 0 leads to state 1
        elif state == 1:
            out += [x * (1 / T), (T - x) * (1 / T)]
        else:
            out += [(x - T) * (1 / T), None]  # X ≥ T; its edge X = T leads to state 1
    return out


def _xiong_neighbor(branch: XBranch, i: int) -> XBranch:
    field, high = ("tc", "nt")[i // 2], bool(i % 2)
    state = getattr(branch, field)
    nxt = {0: 1, 1: 2 if high else 0, 2: 1}[state]
    return branch._replace(**{field: nxt})


def _xiong_states(x, T: float):
    """[(state, (s_lo, s_hi) or None), ...] that the interval quantity ``x`` may reach."""
    from discopt.calphad.ops import Interval

    x = Interval._coerce(x)
    s = x * _inv(T)
    out = []
    if x.lo <= 0:
        out.append((0, None))
    if x.hi > 0 and s.lo <= 1:
        out.append((1, (max(s.lo, 0.0), min(s.hi, 1.0))))
    if s.hi > 1:
        out.append((2, (max(s.lo, 1.0), s.hi)))
    return out


def _xiong_ranges(mag, tc, nt, T) -> list:
    out = []
    for st_c, r_c in _xiong_states(tc, T):
        for st_n, r_n in _xiong_states(nt, T):
            out.append((XBranch(st_c, st_n), [r for r in (r_c, r_n) if r is not None]))
    return out


def _xiong_rows(mag, tc, nt, T, branch, s) -> tuple[list, list]:
    aux = list(s or [])
    eqs, ineqs = [], []
    for x, state, has in ((tc, branch.tc, bool(mag.tc)), (nt, branch.nt, bool(mag.nt))):
        if state == 0:
            if has:
                ineqs.append(-x * (1 / T))
            continue
        sv = aux.pop(0) if aux else None
        if sv is None:  # no auxiliary variable: hold X on its side directly
            ineqs += [g for g in ([x * (1 / T), (T - x) * (1 / T)] if state == 1 else [(x - T) * (1 / T)])]
        else:
            eqs.append(sv - x * (1 / T))
    return eqs, ineqs
