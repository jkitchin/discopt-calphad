"""Compound energy formalism (CEF): mixing on any number of sublattices.

A phase has sublattices s with site ratios a_s and constituents c (elements or
VA). With y_{s,c} the site fraction of c on sublattice s, the Gibbs energy per
formula unit is

    G = Σ_terms  P(y) · V(T)  +  RT Σ_s a_s Σ_c y_{s,c} ln y_{s,c}

Each database parameter is one term, and pycalphad's conventions apply. P(y)
is the product, over sublattices, of the site fractions the parameter lists;
a wildcard ``*`` contributes Σ_c y_{s,c} = 1. Beyond that product:

- **Binary interaction of order k on one mixing sublattice:** also a factor
  (y_first − y_second)^k, in sorted constituent order.
- **Reciprocal parameter (two mixing sublattices):** order 1 adds
  (y_first − y_second) on the second mixing sublattice, and order 2 adds it on
  the first.
- **Ternary interaction:** order k adds the Muggianu-corrected fraction of the
  k-th constituent.

**Site-fraction variables.** Each sublattice with m > 1 constituents has
m − 1 independent variables u ∈ [0, 1]^(m−1), mapped onto the simplex by stick
breaking:

    y_1 = u_1,   y_j = u_j Π_{i<j} (1 − u_i),   y_m = Π_{i<m} (1 − u_i)

With m = 2 this is simply y = (u, 1 − u). The domain is therefore a box, which
fits the solver's phase protocol (box bounds, interval bounds on the boundary
slabs) without linear constraints. For a product of factors,
x ln x = Σ_i f_i ln f_i Π_{j≠i} f_j. Every ``xlogx`` call therefore acts on a
single u or 1 − u, where discopt's ``entropy`` rules are tightest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models.tfunc import coef
from discopt.calphad.ops import NUM

__all__ = ["CEFTerm", "CEFPhase"]

VACANCY = "VA"


@dataclass(frozen=True)
class CEFTerm:
    """One parameter: ``constituents`` is sorted within each sublattice (``("*",)`` is a wildcard).

    ``value`` is in J per formula unit: a number or a callable of T, such as a
    :class:`~discopt.calphad.database.DatabaseValue`.
    """

    constituents: tuple[tuple[str, ...], ...]
    order: int = 0
    value: object = 0.0

    @property
    def label(self) -> str:
        return ":".join(",".join(s) for s in self.constituents) + f";{self.order}"


def _prod(factors):
    out = None
    for f in factors:
        out = f if out is None else out * f
    return 1.0 if out is None else out


def _xlogx_prod(factors, ops):
    """x ln x for x = Π factors, written as Σ_i xlogx(f_i) Π_{j≠i} f_j."""
    total = 0.0
    for i, f in enumerate(factors):
        rest = factors[:i] + factors[i + 1:]
        term = ops.xlogx(f)
        total = total + (term * _prod(rest) if rest else term)
    return total


@dataclass(frozen=True)
class CEFPhase:
    """A multi-sublattice phase in the compound energy formalism (see the module docstring)."""

    name: str
    elements: tuple[str, ...]
    site_ratios: tuple[float, ...]
    #: sorted constituents per sublattice; each is an element of ``elements`` or VA
    constituents: tuple[tuple[str, ...], ...]
    terms: tuple[CEFTerm, ...] = ()
    _layout: tuple = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if len(self.site_ratios) != len(self.constituents):
            raise ValueError(f"{self.name}: one site ratio per sublattice is required")
        allowed = set(self.elements) | {VACANCY}
        for s, cons in enumerate(self.constituents):
            if not cons or len(set(cons)) != len(cons) or not set(cons) <= allowed:
                raise ValueError(f"{self.name}: bad constituents {cons} on sublattice {s}")
            if tuple(sorted(cons)) != tuple(cons):
                raise ValueError(f"{self.name}: constituents on sublattice {s} must be sorted")
        for t in self.terms:
            self._check_term(t)
        layout, i = [], 0
        for cons in self.constituents:
            layout.append(tuple(range(i, i + len(cons) - 1)))
            i += len(cons) - 1
        object.__setattr__(self, "_layout", tuple(layout))
        if self.atoms_min <= 0:
            raise NotImplementedError(
                f"{self.name}: an all-vacancy configuration has no atoms, so the certificate's amount bound is unbounded"
            )

    def _check_term(self, t: CEFTerm) -> None:
        if len(t.constituents) != len(self.constituents):
            raise ValueError(f"{self.name}: term {t.label} has the wrong number of sublattices")
        for s, comps in enumerate(t.constituents):
            if comps != ("*",) and not set(comps) <= set(self.constituents[s]):
                raise ValueError(f"{self.name}: term {t.label} lists a constituent not on sublattice {s}")
        dof = [len(c) for c in t.constituents]
        mixing = [s for s, d in enumerate(dof) if d > 1]
        if max(dof) > 3:
            raise NotImplementedError(f"{self.name}: interactions of more than three constituents ({t.label})")
        if len(mixing) > 1 and max(dof) > 2:
            raise NotImplementedError(f"{self.name}: higher-order reciprocal parameter {t.label}")
        if t.order and len(mixing) > 2:
            raise NotImplementedError(f"{self.name}: order > 0 with mixing on more than two sublattices ({t.label})")
        if t.order > 2 and (len(mixing) == 2 or max(dof) == 3):
            raise NotImplementedError(f"{self.name}: reciprocal or ternary parameter order > 2 ({t.label})")

    # ------------------------------------------------------------------ protocol

    @property
    def n_y(self) -> int:
        return sum(len(c) - 1 for c in self.constituents)

    @property
    def y_bounds(self) -> list[tuple[float, float]]:
        return [(0.0, 1.0)] * self.n_y

    @property
    def atoms_min(self) -> float:
        """Fewest atoms per formula unit: each sublattice that can hold VA may be empty."""
        return float(sum(a for a, cons in zip(self.site_ratios, self.constituents) if VACANCY not in cons))

    def _factors(self, y):
        """Per sublattice, the stick-breaking factors of each constituent's site fraction."""
        out = []
        for cons, idx in zip(self.constituents, self._layout):
            d, rest = {}, []
            for j, c in enumerate(cons):
                if j < len(idx):
                    d[c] = rest + [y[idx[j]]]
                    rest = rest + [1 - y[idx[j]]]
                else:
                    d[c] = rest
            out.append(d)
        return out

    def gibbs(self, y, ops=NUM, T=None):
        if T is None:
            raise ValueError(f"{self.name}: temperature T is required")
        RT = R_GAS * T
        F = self._factors(y)
        Y = [{c: _prod(f) for c, f in d.items()} for d in F]
        G = 0.0
        for t in self.terms:
            v = coef(t.value, T) / RT
            if v != 0.0:
                G = G + v * self._term_factor(t, Y)
        for a, d in zip(self.site_ratios, F):
            for f in d.values():
                if f:
                    G = G + a * _xlogx_prod(f, ops)
        return G

    def _term_factor(self, t: CEFTerm, Y):
        mixing = [s for s, c in enumerate(t.constituents) if len(c) > 1]
        parts = []
        for s, comps in enumerate(t.constituents):
            if comps == ("*",):
                continue
            ys = [Y[s][c] for c in comps]
            parts += [v for v in ys if not isinstance(v, float) or v != 1.0]
            if len(comps) == 2 and t.order > 0:
                diff = ys[0] - ys[1]
                if len(mixing) == 1:
                    parts += [diff] * t.order
                elif s == mixing[-t.order]:
                    parts.append(diff)
            elif len(comps) == 3:
                parts.append(ys[t.order] + (1 - (ys[0] + ys[1] + ys[2])) * (1.0 / 3.0))
        return _prod(parts)

    def amounts(self, y):
        Y = [{c: _prod(f) for c, f in d.items()} for d in self._factors(y)]
        out = []
        for e in self.elements:
            total = 0.0
            for a, d in zip(self.site_ratios, Y):
                if e in d:
                    total = total + a * d[e]
            out.append(total)
        return tuple(out)

    def initial_y(self, b) -> np.ndarray | None:
        """Interior site-fraction variables whose composition matches ``b`` in ratio, or None.

        A bounded least-squares fit from the centre of the box, then from a few
        fixed pseudo-random starts.
        """
        from scipy.optimize import least_squares

        b = np.asarray(b, dtype=float)
        xb = b / b.sum()

        def xres(u):
            a = np.array([float(v) for v in self.amounts(u)])
            return a / a.sum() - xb

        if self.n_y == 0:
            return np.empty(0) if np.max(np.abs(xres(np.empty(0)))) < 1e-12 else None
        rng = np.random.default_rng(0)
        for u0 in [np.full(self.n_y, 0.5), *rng.uniform(0.05, 0.95, size=(7, self.n_y))]:
            sol = least_squares(xres, u0, bounds=(1e-6, 1 - 1e-6), xtol=1e-15, ftol=1e-15, gtol=1e-15)
            if np.max(np.abs(sol.fun)) < 1e-9:
                return sol.x
        return None

    # ------------------------------------------------------------------ site fractions

    def site_fractions(self, y) -> list[dict[str, float]]:
        """The site fractions y_{s,c} (per sublattice, a dict by constituent) for variables ``y``."""
        return [{c: float(_prod(f)) for c, f in d.items()} for d in self._factors(np.asarray(y, float))]

    def variables_from_site_fractions(self, fractions) -> np.ndarray:
        """Inverse of :meth:`site_fractions`: stick-breaking variables from per-sublattice site fractions.

        ``fractions`` is a list with one dict (by constituent) or one sequence
        (in sorted constituent order) per sublattice.
        """
        u = []
        for cons, fr in zip(self.constituents, fractions):
            vals = [fr[c] for c in cons] if isinstance(fr, dict) else list(fr)
            if len(vals) != len(cons) or not math.isclose(sum(vals), 1.0, abs_tol=1e-9):
                raise ValueError(f"{self.name}: site fractions {vals} do not sum to 1 over {cons}")
            left = 1.0
            for v in vals[:-1]:
                u.append(min(max(v / left, 0.0), 1.0) if left > 0 else 0.0)
                left -= v
        return np.array(u)

    def endmembers(self) -> list[tuple[str, ...]]:
        """Every endmember: one constituent per sublattice."""
        return list(product(*self.constituents))

    # ------------------------------------------------------------------ symbolic view

    def gibbs_expr(self):
        """G in J per formula unit as a SymPy expression in symbols ``y<s>_<constituent>`` and ``T``.

        A term whose value has an ``expr()`` method (a database value) keeps its
        symbolic form; a plain number is inserted as is; any other callable of T
        becomes a symbol ``V_<label>``.
        """
        import sympy as sp

        from discopt.calphad.symbolic import T as T_

        Y = [
            {c: (sp.Integer(1) if len(cons) == 1 else sp.Symbol(f"y{s}_{c}", nonnegative=True)) for c in cons}
            for s, cons in enumerate(self.constituents)
        ]
        G = sp.Integer(0)
        for t in self.terms:
            if hasattr(t.value, "expr"):
                v = t.value.expr()
            elif callable(t.value):
                v = sp.Symbol(f"V_{t.label}")
            else:
                v = sp.sympify(t.value)
            G += self._term_factor(t, Y) * v
        R = sp.Symbol("R", positive=True)
        for a, d in zip(self.site_ratios, Y):
            G += sp.nsimplify(a) * R * T_ * sum(yc * sp.log(yc) for yc in d.values() if yc != 1)
        return G
