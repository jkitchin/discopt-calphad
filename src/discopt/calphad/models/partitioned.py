"""Order–disorder (partitioned) phases: ``AMEND_PHASE_DESCRIPTION ... DISORDERED_PART``.

A partitioned phase, such as L1₂ (FCC_L12 over FCC_A1) or B2 (BCC_B2 over
BCC_A2), has an ordered description on several substitutional sublattices and a
disordered parent with one. With y the ordered site fractions and x their site
average over the substitutional sublattices (vacancies count as a species, and
interstitial sublattices are left as they are), pycalphad and Thermo-Calc use

    G = G_dis(x) + G_ord(y) − G_ord(x)

per mole of atoms (Connetable et al., Calphad 32 (2008) 361; Lukas, Fries and
Sundman, *Computational Thermodynamics* (2007) §5.8). Per ordered formula unit,
G_dis is scaled by λ = Σ_s a_s / a_dis, the ratio of substitutional sites. The
ordering contribution G_ord(y) − G_ord(x) vanishes wherever all substitutional
sublattices hold the same fractions, so the disordered state of the phase is
the parent phase exactly.

**Entropy.** The ideal mixing terms of G_dis(x) and G_ord(x) cancel. Both are
(Σ_s a_s) Σ_c x_c ln x_c on the substitutional sites, and the interstitial terms
agree because the two descriptions have the same interstitial-to-substitutional
site ratio. What remains is the ordered phase's configurational entropy,
written on single stick-breaking variables as for any :class:`CEFPhase`. So
every ``xlogx`` stays where discopt's ``entropy`` bounds are tightest. Only the
energy terms are evaluated at x, and they are polynomials in y.

**NEVER_DISORDER** (Lukas–Fries–Sundman eq. 5.162): G = G_dis(x) + G_ord(y).
The disordered part contributes no entropy, and nothing is subtracted.

**Magnetism.** As in pycalphad 0.11, the magnetic term is the disordered
part's, with its TC and BMAGN evaluated at x. A magnetic phase's branches
(:mod:`discopt.calphad.models.magnetic`) are then branches of T_c(x(y)), and
global pricing splits on them as for a plain magnetic phase. Thermo-Calc instead
partitions TC and BMAGN themselves, T_c = T_c,dis(x) + T_c,ord(y) − T_c,ord(x).
The two agree when the ordered part has no magnetic parameters. A phase whose
ordered part has some is rejected for now.

**Symmetry.** Substitutional sublattices with the same site ratio and the same
constituents may be equivalent: permuting them maps each ordered variant onto
another, and leaves G unchanged. That makes several equal global minima, one per
ordered variant. :attr:`PartitionedPhase.symmetry` holds the permutations that
leave G invariant, found numerically at construction. When a class of
equivalent sublattices is invariant under all its permutations,
:meth:`~PartitionedPhase.symmetry_constraints` orders the first constituent's
fraction along the class. Every point has a sorted image with the same G, so the
constraint removes duplicates and never the minimum.
:meth:`~PartitionedPhase.variants` lists the images of a point, which lets
composition sets be compared up to that symmetry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np

from discopt.calphad.models.cef import CEFPhase, _prod
from discopt.calphad.ops import NUM

__all__ = ["PartitionedPhase"]


@dataclass(frozen=True)
class PartitionedPhase:
    """An ordered phase with a disordered part (see the module docstring).

    ``ordered`` holds the ordered part's own parameters on its sublattices, and
    supplies the variables, the amounts and the configurational entropy.
    ``disordered`` is the parent phase. Its first sublattice is the
    substitutional one, and its others match the ordered interstitial
    sublattices in order.
    """

    name: str
    ordered: CEFPhase
    disordered: CEFPhase
    #: indices of the ordered sublattices that take part in the ordering
    substitutional: tuple[int, ...]
    never_disorder: bool = False
    #: permutations of the ordered sublattices that leave G unchanged (found at construction)
    symmetry: tuple[tuple[int, ...], ...] = field(init=False, compare=False)
    #: classes of equivalent sublattices whose every permutation leaves G unchanged
    _sorted_classes: tuple[tuple[int, ...], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        o, d = self.ordered, self.disordered
        if o.magnetic is not None:
            raise NotImplementedError(
                f"{self.name}: magnetic parameters on the ordered part (pycalphad and Thermo-Calc combine them "
                "with the disordered part's differently)"
            )
        if set(o.elements) != set(d.elements):
            raise ValueError(f"{self.name}: ordered and disordered parts have different elements")
        subst = self.substitutional
        if not subst:
            raise ValueError(f"{self.name}: no substitutional sublattice")
        for s in subst:
            if o.constituents[s] != d.constituents[0]:
                raise ValueError(
                    f"{self.name}: substitutional sublattice {s} has constituents {o.constituents[s]}, "
                    f"but the disordered part has {d.constituents[0]}"
                )
        inter = self.interstitial
        if len(inter) != len(d.constituents) - 1:
            raise ValueError(f"{self.name}: the ordered and disordered parts have different interstitial sublattices")
        for k, s in enumerate(inter, start=1):
            if o.constituents[s] != d.constituents[k]:
                raise ValueError(f"{self.name}: interstitial sublattice {s} does not match the disordered part")
            if not math.isclose(o.site_ratios[s] / self.a_ord, d.site_ratios[k] / d.site_ratios[0], rel_tol=1e-12):
                raise ValueError(f"{self.name}: interstitial site ratios differ from the disordered part's")
        sym, classes = self._find_symmetry()
        object.__setattr__(self, "symmetry", sym)
        object.__setattr__(self, "_sorted_classes", classes)

    # ------------------------------------------------------------------ structure

    @property
    def interstitial(self) -> tuple[int, ...]:
        return tuple(s for s in range(len(self.ordered.constituents)) if s not in self.substitutional)

    @property
    def a_ord(self) -> float:
        return float(sum(self.ordered.site_ratios[s] for s in self.substitutional))

    @property
    def scale(self) -> float:
        """λ: disordered formula units per ordered formula unit."""
        return self.a_ord / self.disordered.site_ratios[0]

    def _averaged(self, Y):
        """x: the site average of ``Y`` over the substitutional sublattices, by constituent."""
        r = self.ordered.site_ratios
        return {c: sum_(r[s] * Y[s][c] for s in self.substitutional) * (1.0 / self.a_ord) for c in Y[self.substitutional[0]]}

    def _at_x(self, Y, x=None):
        """(the ordered site fractions with x on every substitutional sublattice, the disordered part's)."""
        x = self._averaged(Y) if x is None else x
        y_ord = [dict(x) if s in self.substitutional else Y[s] for s in range(len(Y))]
        y_dis = [x] + [Y[s] for s in self.interstitial]
        return y_ord, y_dis

    # ------------------------------------------------------------------ protocol

    @property
    def elements(self):
        return self.ordered.elements

    @property
    def constituents(self):
        return self.ordered.constituents

    @property
    def site_ratios(self):
        return self.ordered.site_ratios

    @property
    def n_y(self) -> int:
        return self.ordered.n_y

    @property
    def y_bounds(self):
        return self.ordered.y_bounds

    @property
    def atoms_min(self) -> float:
        return self.ordered.atoms_min

    @property
    def atoms_max(self) -> float:
        return self.ordered.atoms_max

    @property
    def vacancy_limited(self) -> bool:
        return self.ordered.vacancy_limited

    @property
    def min_occupancy(self) -> float:
        return self.ordered.min_occupancy

    def domain_constraints(self, y) -> list:
        return self.ordered.domain_constraints(y)

    @property
    def magnetic(self):
        """The disordered part's magnetic description, or None (see the module docstring)."""
        return self.disordered.magnetic

    def amounts(self, y):
        return self.ordered.amounts(y)

    def initial_y(self, b):
        return self.ordered.initial_y(b)

    def site_fractions(self, y):
        return self.ordered.site_fractions(y)

    def variables_from_site_fractions(self, fractions):
        return self.ordered.variables_from_site_fractions(fractions)

    def endmembers(self):
        return self.ordered.endmembers()

    def average_expressions(self, y) -> dict:
        """x_c, the site averages over the substitutional sublattices, as expressions in ``y``."""
        return self._averaged(self.ordered._Y(y))

    def gibbs(self, y, ops=NUM, T=None, x=None):
        """G/RT per formula unit. ``x`` may supply the site averages (e.g. as auxiliary variables).

        Given ``x``, the caller must keep it equal to :meth:`average_expressions`.
        """
        if T is None:
            raise ValueError(f"{self.name}: temperature T is required")
        o, d = self.ordered, self.disordered
        F = o._factors(y)
        Y = [{c: _prod(f) for c, f in dd.items()} for dd in F]
        y_ord, y_dis = self._at_x(Y, x)
        G = o.energy_at(Y, T) + o._ideal(F, ops) + self.scale * d.energy_at(y_dis, T)
        if not self.never_disorder:
            G = G - o.energy_at(y_ord, T)
        if d.magnetic is not None:
            G = G + self.scale * d.magnetic.term(d._magnetic_q(y_dis, T), T, ops)
        return G

    def ordering_energy(self, y, T: float) -> float:
        """G_ord(y) − G_ord(x) in J per formula unit, without entropy (0 at any disordered y)."""
        Y = self.ordered._Y(np.asarray(y, float))
        y_ord, _ = self._at_x(Y)
        o = self.ordered
        return float(o.energy_at(Y, T) - o.energy_at(y_ord, T)) * 8.314462618 * T

    def degree_of_order(self, y) -> float:
        """max over constituents and substitutional sublattices of |y_s,c − x_c| (0 when disordered)."""
        sf = self.site_fractions(y)
        x = self._averaged(sf)
        return max(abs(sf[s][c] - x[c]) for s in self.substitutional for c in x)

    # ------------------------------------------------------------------ magnetism (the disordered part's, at x)

    def _magnetic_q(self, y, T, x=None):
        _, y_dis = self._at_x(self.ordered._Y(y), x)
        return self.disordered._magnetic_q(y_dis, T)

    def _magnetic_value(self, name, y, T) -> float:
        if self.magnetic is None or name not in self.magnetic.quantities:
            return 0.0
        q = self._magnetic_q(np.asarray(y, float), T)
        return float(q[self.magnetic.quantities.index(name)])

    def curie_temperature(self, y, T: float) -> float:
        return self._magnetic_value("tc", y, T)

    def neel_temperature(self, y, T: float) -> float:
        return self._magnetic_value("nt", y, T)

    def magnetic_moment(self, y, T: float) -> float:
        return self._magnetic_value("beta", y, T)

    def magnetic_branch(self, y, T: float):
        if self.magnetic is None:
            return None
        q = [float(v) for v in self._magnetic_q(np.asarray(y, float), T)]
        return self.magnetic.branch_at(q, T)

    def branch_constraints(self, y, T: float, branch) -> list:
        return self.magnetic.constraints(self._magnetic_q(y, T), T, branch)

    def magnetic_pieces(self, box, T: float):
        from discopt.calphad.ops import Interval

        return self.magnetic.ranges(self._magnetic_q([Interval(lo, hi) for lo, hi in box], T), T)

    def pricing_constraints(self, y, T: float, branch, s=None, x=None):
        return self.magnetic.pricing_rows(self._magnetic_q(y, T, x), T, branch, s)

    # ------------------------------------------------------------------ symmetry

    def _permute(self, y, perm):
        """The variables of the configuration whose sublattice s holds what sublattice perm[s] held at ``y``."""
        sf = self.site_fractions(y)
        return self.variables_from_site_fractions([sf[perm[s]] for s in range(len(sf))])

    def _find_symmetry(self):
        o = self.ordered
        classes: dict[tuple, list[int]] = {}
        for s in self.substitutional:
            classes.setdefault((o.site_ratios[s], o.constituents[s]), []).append(s)
        classes = [tuple(c) for c in classes.values() if len(c) > 1]
        if not classes or o.n_y == 0:
            return ((tuple(range(len(o.constituents))),), ())
        rng = np.random.default_rng(12345)
        pts = rng.uniform(0.05, 0.95, size=(4, o.n_y))
        temps = [600.0, 1100.0, 1700.0]
        ref = {}
        for T in temps:
            try:
                ref[T] = [float(self.gibbs(p, NUM, T)) for p in pts]
            except ValueError:  # T outside a parameter's range
                continue

        def invariant(perm):
            for T, g0 in ref.items():
                for p, g in zip(pts, g0):
                    if not math.isclose(float(self.gibbs(self._permute(p, perm), NUM, T)), g, rel_tol=1e-9, abs_tol=1e-9):
                        return False
            return True

        n = len(o.constituents)
        group = [tuple(range(n))]
        full = []
        for cls in classes:
            perms = []
            for q in permutations(cls):
                if q == cls:
                    continue
                perm = list(range(n))
                for a, b in zip(cls, q):
                    perm[a] = b
                if invariant(tuple(perm)):
                    perms.append(tuple(perm))
            group += perms
            if len(perms) == math.factorial(len(cls)) - 1:
                full.append(cls)
        return tuple(group), tuple(full)

    def variants(self, y) -> list[np.ndarray]:
        """``y`` and its images under :attr:`symmetry`: the same state, as other ordered variants."""
        y = np.asarray(y, float)
        return [y] + [self._permute(y, p) for p in self.symmetry[1:]]

    def symmetry_constraints(self, y) -> list:
        """Expressions ≥ 0 that pick one variant: the first constituent's fraction descends along each class."""
        out = []
        layout = self.ordered._layout
        for cls in self._sorted_classes:
            for a, b in zip(cls, cls[1:]):
                out.append(y[layout[a][0]] - y[layout[b][0]])
        return out


def sum_(it):
    total = 0.0
    for v in it:
        total = total + v
    return total
