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
per point. Interval bounds, and with them global pricing, are not implemented
yet (issue #6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = ["Branch", "Magnetic", "branch_constraints", "branch_of", "ihj"]


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


def ihj(mag: Magnetic, tc, beta, T: float, ops, branch: Branch | None = None):
    """G_mag / RT for the given T_c and β, on backend ``ops``.

    ``NUM`` chooses the branch per element (array inputs broadcast). ``EXPR``
    uses ``branch``, or ``ops.branch`` if ``branch`` is None.
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
        s = tc_eff * (1 / T)
        f = _ordered(s, mag.p, mag.A) if branch.ordered else _paramagnetic(s, mag.A)
        return dm.log(1 + beta_eff) * f
    raise NotImplementedError(
        "interval bounds for the magnetic contribution are not implemented yet (issue #6), "
        "so a magnetic phase cannot be priced globally"
    )


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
