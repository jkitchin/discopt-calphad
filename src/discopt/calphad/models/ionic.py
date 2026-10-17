"""The two-sublattice partially ionic liquid (TDB phase hint ``Y``).

Sundman's model (Calphad 15 (1991) 109–119), as pycalphad 0.11 implements it:

    (C⁺)_P (A⁻, Va, B⁰)_Q

with cations C on the first sublattice and anions A, a vacancy Va and neutral
species B on the second. The site ratios vary with the site fractions so that
the liquid is neutral for every y:

    Q = Σ_C q_C y_C,      P = Σ_A |q_A| y_A + Q y_Va

So no charge-balance constraint is needed. Per formula unit,

    G = Σ_params P(y) V(T) + RT [P Σ_C y_C ln y_C + Q Σ_j y_j ln y_j]

and a parameter's factor P(y) follows the CEF product with three changes:

- **Ordering.** On the second sublattice the constituents of a parameter are
  ordered charged species first, then Va, then neutrals, before the
  Redlich–Kister difference (y_first − y_second)^k is formed.
- **Anion-free parameters** (Va and neutrals only) are multiplied by Q. Each
  cation fraction in them is also multiplied by y_Va (the pair rule), unless the
  parameter is a single endmember with Va, such as ``G(LIQ,CU+1:VA)``.
- **Neutral-only parameters** may be written on one sublattice
  (``G(LIQ,S)``). They mean the second sublattice, with no cation factor.

Amounts per formula unit are P Σ_C y_C n_C + Q Σ_j y_j n_j: polynomial in y,
not linear. The fewest atoms per formula unit still sit at a vertex of the
domain. For fixed cation fractions, atoms are linear in the second
sublattice's fractions. For a fixed vertex there, they are α(q·y)(n·y) +
β(n·y) + γ(q·y) in the cation fractions, with α, β, γ ≥ 0. Along any segment
this is either concave or monotone, so its minimum is at an end.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models.cef import VACANCY, CEFPhase, CEFTerm, _prod, _xlogx_prod
from discopt.calphad.models.tfunc import coef
from discopt.calphad.ops import NUM

__all__ = ["IonicLiquidPhase"]


@dataclass(frozen=True)
class IonicLiquidPhase(CEFPhase):
    """A two-sublattice ionic liquid (see the module docstring).

    ``site_ratios`` is ignored: P and Q follow from the site fractions.
    A one-sublattice parameter must be given with a ``("*",)`` wildcard
    prepended for the cation sublattice.
    """

    def __post_init__(self):
        super().__post_init__()
        if len(self.constituents) != 2:
            raise ValueError(f"{self.name}: an ionic liquid has exactly two sublattices")
        if self.magnetic is not None:
            raise NotImplementedError(f"{self.name}: a magnetic ionic liquid")
        for c in self.constituents[0]:
            if self._stoich[c][1] <= 0:
                raise ValueError(f"{self.name}: {c} on the cation sublattice has no positive charge")
        for c in self.constituents[1]:
            if self._stoich[c][1] > 0:
                raise ValueError(f"{self.name}: cation {c} on the anion sublattice")

    # ------------------------------------------------------------------ site ratios

    def ratios(self, Y):
        """(P, Q) for site fractions ``Y`` (dicts by constituent)."""
        Q = 0.0
        for c, v in Y[0].items():
            Q = Q + self._stoich[c][1] * v
        P = 0.0
        for c, v in Y[1].items():
            q = self._stoich[c][1]
            if q < 0:
                P = P + (-q) * v
        if VACANCY in Y[1]:
            P = P + Q * Y[1][VACANCY]
        return P, Q

    @property
    def charged(self) -> bool:
        return False  # neutral for every y, through P and Q

    def domain_equalities(self, y) -> list:
        return []

    # ------------------------------------------------------------------ energy

    def _order(self, comps):
        """Second-sublattice order: charged species, then Va, then neutrals (pycalphad, Thermo-Calc)."""

        def key(c):
            if c == VACANCY:
                return (1, c)
            return (0 if self._stoich[c][1] != 0 else 2, c)

        return tuple(sorted(comps, key=key))

    def _factor(self, t: CEFTerm, Y, Q):
        cats, anis = t.constituents
        anions_present = any(c != "*" and self._stoich[c][1] < 0 for comps in t.constituents for c in comps)
        term = CEFTerm((cats, anis if anis == ("*",) else self._order(anis)), t.order, t.value)
        Yuse = Y
        if not anions_present:
            single_va = VACANCY in anis and max(len(cats), len(anis)) == 1
            if VACANCY in self.constituents[1] and not single_va:
                y_va = Y[1][VACANCY]
                Yuse = [{c: v * y_va for c, v in Y[0].items()}, Y[1]]
        f = self._term_factor(term, Yuse)
        return f * Q if not anions_present else f

    def ratio_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Bounds on (P, Q) over the whole domain: Q is a cation charge average, P at most max(|q_A|, Q)."""
        qc = [self._stoich[c][1] for c in self.constituents[0]]
        qa = [-self._stoich[c][1] for c in self.constituents[1] if self._stoich[c][1] < 0]
        p_hi = max(qa + ([max(qc)] if VACANCY in self.constituents[1] else []) + [0.0])
        return (0.0, p_hi), (min(qc), max(qc))

    def ratio_expressions(self, y):
        """(P, Q) as expressions in the variables ``y``."""
        return self.ratios(self._Y(y))

    def gibbs(self, y, ops=NUM, T=None, ratios=None):
        """G/RT per formula unit. ``ratios`` may supply (P, Q), e.g. as auxiliary variables kept equal to them."""
        if T is None:
            raise ValueError(f"{self.name}: temperature T is required")
        RT = R_GAS * T
        F = self._factors(y)
        Y = [{c: _prod(f) for c, f in d.items()} for d in F]
        P, Q = self.ratios(Y) if ratios is None else ratios
        G = 0.0
        for t in self.terms:
            v = coef(t.value, T) / RT
            if v != 0.0:
                G = G + v * self._factor(t, Y, Q)
        for ratio, d in zip((P, Q), F):
            s = 0.0
            for f in d.values():
                if f:
                    s = s + _xlogx_prod(f, ops)
            G = G + ratio * s
        return G

    def energy_at(self, Y, T: float):
        raise NotImplementedError(f"{self.name}: an ionic liquid as part of a partitioned phase")

    # ------------------------------------------------------------------ amounts

    def amounts(self, y):
        Y = self._Y(y)
        P, Q = self.ratios(Y)
        out = []
        for e in self.elements:
            total = 0.0
            for ratio, d in zip((P, Q), Y):
                part = 0.0
                for c, v in d.items():
                    n = self._stoich[c][0].get(e, 0.0)
                    if n:
                        part = part + n * v
                total = total + ratio * part
            out.append(total)
        return tuple(out)

    def _endmember_atoms(self) -> list[float]:
        """Atoms per formula unit at every vertex (one constituent per sublattice); see the module docstring."""
        out = []
        for c, a in product(*self.constituents):
            Y = [{x: float(x == c) for x in self.constituents[0]}, {x: float(x == a) for x in self.constituents[1]}]
            P, Q = self.ratios(Y)
            out.append(P * self._atoms_of(c) + Q * self._atoms_of(a))
        return out

    @property
    def vacancy_limited(self) -> bool:
        return False  # Q ≥ the smallest cation charge, so a formula unit always holds atoms

    @property
    def atoms_min(self) -> float:
        return min(self._endmember_atoms())

    def gibbs_expr(self):
        raise NotImplementedError(f"{self.name}: no symbolic view of an ionic liquid yet")
