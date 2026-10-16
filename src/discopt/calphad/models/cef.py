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

**Vacancies.** A vacancy on a sublattice makes the number of atoms per formula
unit depend on y. When every sublattice can hold one (constitutional vacancies
in B2, for example), a formula unit could even be empty, and the model has no
global minimum. With G(VA:…:VA) = 0, as in most databases, a dilute fraction δ
of atoms in an otherwise empty lattice has G per atom ≈ G_atom + L + RT ln δ,
which goes to −∞: an unbounded "vacuum phase" that only sampling resolution
hides. The phase is therefore restricted to formula units holding at least
``min_occupancy`` (default 0.5) of :attr:`atoms_max` in atoms, through
:meth:`domain_constraints`. Constitutional vacancies in real phases are far
below that. A certificate covers that domain, and the bound on the total
amount of formula units, Σb/atoms_min, is finite again.

**Magnetic ordering.** With ``magnetic`` set, G also gets the Inden–Hillert–Jarl
term RT ln(β' + 1) f(T/T_c'). T_c(y) and β(y) are built from TC and BMAGN terms
with the same products as above. See :mod:`discopt.calphad.models.magnetic`.
The term is piecewise in y, so a discopt expression of it needs a branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models.magnetic import Magnetic
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
        rest = factors[:i] + factors[i + 1 :]
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
    #: the magnetic ordering contribution, if the phase has one
    magnetic: Magnetic | None = None
    #: with a vacancy on every sublattice, the fewest atoms a formula unit may hold, as a fraction of
    #: :attr:`atoms_max` (see "Vacancies" in the module docstring)
    min_occupancy: float = 0.5
    #: constituents that are not elements: ``(name, ((element, amount), ...), charge)`` each
    species: tuple = ()
    _layout: tuple = field(init=False, repr=False, compare=False)
    #: constituent -> ({element: amount}, charge), for elements, VA and ``species``
    _stoich: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if len(self.site_ratios) != len(self.constituents):
            raise ValueError(f"{self.name}: one site ratio per sublattice is required")
        stoich = {e: ({e: 1.0}, 0.0) for e in self.elements}
        stoich[VACANCY] = ({}, 0.0)
        for name, comp, charge in self.species:
            comp = dict(comp)
            if not set(comp) <= set(self.elements):
                raise ValueError(f"{self.name}: species {name} has elements outside {self.elements}")
            stoich[name] = (comp, float(charge))
        object.__setattr__(self, "_stoich", stoich)
        allowed = set(stoich)
        for s, cons in enumerate(self.constituents):
            if not cons or len(set(cons)) != len(cons) or not set(cons) <= allowed:
                raise ValueError(f"{self.name}: bad constituents {cons} on sublattice {s}")
            if tuple(sorted(cons)) != tuple(cons):
                raise ValueError(f"{self.name}: constituents on sublattice {s} must be sorted")
        for t in self.terms + (self.magnetic.tc + self.magnetic.beta if self.magnetic else ()):
            self._check_term(t)
        layout, i = [], 0
        for cons in self.constituents:
            layout.append(tuple(range(i, i + len(cons) - 1)))
            i += len(cons) - 1
        object.__setattr__(self, "_layout", tuple(layout))
        if not 0 < self.min_occupancy <= 1:
            raise ValueError(f"{self.name}: min_occupancy must be in (0, 1], got {self.min_occupancy}")
        if self.atoms_max <= 0:
            raise ValueError(f"{self.name}: no sublattice can hold an atom")

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

    def _atoms_of(self, constituent) -> float:
        return float(sum(self._stoich[constituent][0].values()))

    def _endmember_atoms(self) -> list[float]:
        """Atoms per formula unit of every endmember. Atoms are linear in y, so these bound them."""
        per = [[self._atoms_of(c) for c in cons] for cons in self.constituents]
        out = [0.0]
        for a, opts in zip(self.site_ratios, per):
            out = [t + a * n for t in out for n in sorted(set(opts))]
        return out

    @property
    def atoms_max(self) -> float:
        """Most atoms per formula unit, over the endmembers."""
        return max(self._endmember_atoms())

    @property
    def vacancy_limited(self) -> bool:
        """True if every sublattice can hold a vacancy, so that a formula unit could be empty."""
        return all(VACANCY in cons for cons in self.constituents)

    @property
    def charged(self) -> bool:
        """True if some constituent carries a charge: the phase then needs charge neutrality."""
        return any(self._stoich[c][1] != 0 for cons in self.constituents for c in cons)

    def charge(self, y):
        """Σ_s a_s Σ_c q_c y_s,c: the charge per formula unit (0 on the phase's domain if :attr:`charged`)."""
        Y = self._Y(y)
        total = 0.0
        for a, d in zip(self.site_ratios, Y):
            for c, v in d.items():
                q = self._stoich[c][1]
                if q:
                    total = total + (a * q) * v
        return total

    def domain_equalities(self, y) -> list:
        """Expressions in ``y`` that are 0 on the phase's domain: charge neutrality, scaled to order one."""
        if not self.charged:
            return []
        scale = max(abs(self._stoich[c][1]) * a for a, cons in zip(self.site_ratios, self.constituents) for c in cons)
        return [self.charge(y) * (1.0 / scale)]

    @property
    def atoms_min(self) -> float:
        """Fewest atoms per formula unit.

        Each sublattice that can hold VA may be empty. If every one can, the
        phase is restricted to :attr:`min_occupancy` · :attr:`atoms_max` (see
        :meth:`domain_constraints`).
        """
        if self.vacancy_limited:
            return self.min_occupancy * self.atoms_max
        return min(self._endmember_atoms())

    def domain_constraints(self, y) -> list:
        """Expressions in ``y`` that are ≥ 0 on the phase's domain, beyond the box.

        Only a vacancy-limited phase has one: atoms(y) − atoms_min ≥ 0.
        """
        if not self.vacancy_limited:
            return []
        total = 0.0
        for a in self.amounts(y):
            total = total + a
        return [(total - self.atoms_min) * (1.0 / self.atoms_max)]

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
        F = self._factors(y)
        Y = [{c: _prod(f) for c, f in d.items()} for d in F]
        G = self.energy_at(Y, T) + self._ideal(F, ops)
        if self.magnetic is not None:
            G = G + self.magnetic_at(Y, T, ops)
        return G

    def energy_at(self, Y, T: float):
        """Σ_terms P(Y) V(T) / RT: G/RT without the ideal mixing and magnetic terms.

        ``Y`` holds the site fractions per sublattice as dicts by constituent.
        They may be any expressions: :class:`PartitionedPhase` evaluates this
        at site-averaged fractions.
        """
        RT = R_GAS * T
        G = 0.0
        for t in self.terms:
            v = coef(t.value, T) / RT
            if v != 0.0:
                G = G + v * self._term_factor(t, Y)
        return G

    def ideal_at(self, Y, ops):
        """RT Σ_s a_s Σ_c y ln y, divided by RT, for site fractions ``Y`` given as expressions."""
        G = 0.0
        for a, d in zip(self.site_ratios, Y):
            if len(d) > 1:
                for v in d.values():
                    G = G + a * ops.xlogx(v)
        return G

    def magnetic_at(self, Y, T: float, ops):
        """G_mag/RT at site fractions ``Y`` (0 without magnetism)."""
        if self.magnetic is None:
            return 0.0
        return self.magnetic.term(self._magnetic_q(Y, T), T, ops)

    def _ideal(self, F, ops):
        G = 0.0
        for a, d in zip(self.site_ratios, F):
            for f in d.values():
                if f:
                    G = G + a * _xlogx_prod(f, ops)
        return G

    # ------------------------------------------------------------------ magnetism

    def _sum(self, terms, Y, T):
        total = 0.0
        for t in terms:
            v = coef(t.value, T)
            if v != 0.0:
                total = total + v * self._term_factor(t, Y)
        return total

    def _magnetic_sums(self, Y, T):
        return self._sum(self.magnetic.tc, Y, T), self._sum(self.magnetic.beta, Y, T)

    def _magnetic_q(self, Y, T):
        """The magnetic model's quantities (``Magnetic.quantities``) at site fractions ``Y``."""
        mag = self.magnetic
        return tuple(self._sum(getattr(mag, name), Y, T) for name in mag.quantities)

    def _Y(self, y):
        return [{c: _prod(f) for c, f in d.items()} for d in self._factors(y)]

    def curie_temperature(self, y, T: float) -> float:
        """T_c(y) in K, before the AFM division (0 for a phase without magnetism)."""
        if self.magnetic is None:
            return 0.0
        return float(self._sum(self.magnetic.tc, self._Y(np.asarray(y, float)), T))

    def neel_temperature(self, y, T: float) -> float:
        """T_N(y) in K (Xiong model; 0 otherwise)."""
        if self.magnetic is None:
            return 0.0
        return float(self._sum(self.magnetic.nt, self._Y(np.asarray(y, float)), T))

    def magnetic_moment(self, y, T: float) -> float:
        """β(y) in Bohr magnetons, before the AFM division (0 for a phase without magnetism)."""
        if self.magnetic is None:
            return 0.0
        return float(self._sum(self.magnetic.beta, self._Y(np.asarray(y, float)), T))

    def magnetic_branch(self, y, T: float):
        """The smooth piece of the magnetic term at numeric ``y``, or None without magnetism."""
        if self.magnetic is None:
            return None
        q = [float(v) for v in self._magnetic_q(self._Y(np.asarray(y, float)), T)]
        return self.magnetic.branch_at(q, T)

    def branch_constraints(self, y, T: float, branch) -> list:
        """discopt expressions in ``y`` that are ≥ 0 exactly on ``branch`` (``Magnetic.constraints``)."""
        return self.magnetic.constraints(self._magnetic_q(self._Y(y), T), T, branch)

    def magnetic_pieces(self, box, T: float) -> list:
        """The magnetic branches the site-fraction ``box`` may reach, each with its auxiliary-variable ranges.

        An interval evaluation of the magnetic quantities over the box rules
        out the others (``Magnetic.ranges``).
        """
        from discopt.calphad.ops import Interval

        q = self._magnetic_q(self._Y([Interval(lo, hi) for lo, hi in box]), T)
        return self.magnetic.ranges(q, T)

    def pricing_constraints(self, y, T: float, branch, s=None) -> tuple[list, list]:
        """``(equalities, inequalities)`` in discopt variables ``y`` that hold ``y`` on ``branch``.

        ``s`` holds the auxiliary variables of the branch's ranges
        (``Magnetic.pricing_rows``). Without them, the result is
        :meth:`branch_constraints` as inequalities.
        """
        return self.magnetic.pricing_rows(self._magnetic_q(self._Y(y), T), T, branch, s)

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
                for c, v in d.items():
                    n = self._stoich[c][0].get(e, 0.0)
                    if n:
                        total = total + (a * n) * v
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
            return np.concatenate([a / a.sum() - xb, [float(g) for g in self.domain_equalities(u)]])

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
        if self.magnetic is not None:
            G += R * T_ * self._magnetic_expr(Y)
        return G

    def _magnetic_expr(self, Y):
        """G_mag / RT as a SymPy ``Piecewise`` (see :mod:`~discopt.calphad.models.magnetic`)."""
        import sympy as sp

        from discopt.calphad.models.magnetic import _ordered, _paramagnetic
        from discopt.calphad.symbolic import T as T_

        def value(t):
            return t.value.expr() if hasattr(t.value, "expr") else sp.sympify(t.value)

        mag = self.magnetic
        tc = sum((self._term_factor(t, Y) * value(t) for t in mag.tc), sp.Integer(0))
        beta = sum((self._term_factor(t, Y) * value(t) for t in mag.beta), sp.Integer(0))
        if mag.xiong:
            from discopt.calphad.models.magnetic import _x_ord, _x_para

            nt = sum((self._term_factor(t, Y) * value(t) for t in mag.nt), sp.Integer(0))
            s = sp.Symbol("s", real=True)
            g = sp.Piecewise((0, s <= 0), (_x_para(s, mag.p), s <= 1), (_x_ord(s, mag.p), True))
            return sp.log(beta + 1) * (g.subs(s, tc / T_) + g.subs(s, nt / T_))
        afm = sp.nsimplify(mag.afm)
        tc_eff = sp.Piecewise((tc / afm, tc <= 0), (tc, True))
        beta_eff = sp.Piecewise((beta / afm, beta <= 0), (beta, True))
        s = sp.Symbol("s", positive=True)
        f = sp.Piecewise((_ordered(s, mag.p, mag.A), s > 1), (_paramagnetic(s, mag.A), True))
        return sp.log(beta_eff + 1) * f.subs(s, tc_eff / T_)
