"""Restricted equilibrium R(S): Gibbs minimization over fixed composition sets.

The problem, in intensive form:

    min  Σ_k N_k G_k(y_k)
    s.t. Σ_k N_k A_k(y_k) = b      (named ``balance_<E>``)
         N_k ≥ 0,  y_k inside its site-fraction box

It is solved locally with POUNCE. The chemical potentials are μ = −λ, where λ are
the multipliers of the balance constraints.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import discopt.modeling as dm
import numpy as np
from scipy.optimize import nnls

from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, amount_matrix, inner_bounds, tangent_distance
from discopt.calphad.ops import EXPR, NUM
from discopt.calphad.solvers.local import local_nlp_solve

__all__ = ["CompositionSet", "RestrictedResult", "solve_restricted"]


@dataclass
class CompositionSet:
    """One instance of a phase: site fractions ``y`` and amount ``amount`` (formula units)."""

    phase: Any
    y: np.ndarray
    amount: float = 0.0

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=float).reshape(-1)
        if self.y.size != self.phase.n_y:
            raise ValueError(f"{self.phase.name} expects {self.phase.n_y} site fractions, got {self.y.size}")

    @property
    def x(self) -> np.ndarray:
        """Overall mole fractions of the elements in this set."""
        a = np.array([float(v) for v in self.phase.amounts(self.y)], dtype=float)
        return a / a.sum()

    @property
    def atoms(self) -> float:
        """Atoms per formula unit."""
        return float(sum(float(v) for v in self.phase.amounts(self.y)))

    def __repr__(self) -> str:
        y = np.array2string(self.y, precision=8)
        return f"CompositionSet({self.phase.name}, y={y}, amount={self.amount:.8g})"


@dataclass
class RestrictedResult:
    status: str
    sets: list[CompositionSet]
    mu: np.ndarray
    #: Σ N_k G_k(y_k), in units of RT
    G: float
    #: |G − μ·b|, the term the certificate theorem adds to the pricing bound
    residual: float
    #: max |Σ N_k A_k(y_k) − b|
    primal_infeasibility: float
    #: max over sets with N_k > 0 of |Ψ(y_k)| (zero when every set touches the tangent plane)
    tangency_residual: float
    iterations: int
    wall_time: float

    @property
    def converged(self) -> bool:
        return self.status == "optimal"


def _elements(sets: Sequence[CompositionSet]) -> tuple[str, ...]:
    els = tuple(sets[0].phase.elements)
    for s in sets:
        if tuple(s.phase.elements) != els:
            raise ValueError("all phases must use the same element list")
    return els


def solve_restricted(
    sets: Sequence[CompositionSet],
    b,
    *,
    T: float | None = None,
    domain_floor: float = DOMAIN_FLOOR,
    amount_tol: float = 1e-10,
    options: dict | None = None,
    fixed: Sequence[bool] | None = None,
) -> RestrictedResult:
    """Solve R(S) locally, starting from the given composition sets.

    ``fixed[k]`` holds set k's site fractions at their starting values, so only
    its amount can change. The loop uses this for sets that were just added.
    """
    if not sets:
        raise ValueError("at least one composition set is required")
    b = np.asarray(b, dtype=float)
    els = _elements(sets)
    if b.shape != (len(els),):
        raise ValueError(f"b must have one entry per element {els}")

    atoms_min = min(s.phase.atoms_min for s in sets)
    n_max = 2.0 * b.sum() / atoms_min
    y0, boxes = [], []
    for k, s in enumerate(sets):
        box = inner_bounds(s.phase, domain_floor)
        yk = np.clip(s.y, [lo for lo, _ in box], [hi for _, hi in box]) if box else s.y
        if fixed is not None and fixed[k]:
            box = [(float(v), float(v)) for v in yk]
        y0.append(yk)
        boxes.append(box)

    # start amounts: nonnegative least squares at the starting compositions, kept off the bound
    n0, _ = nnls(amount_matrix([CompositionSet(s.phase, yk) for s, yk in zip(sets, y0)]), b)
    n0 = np.maximum(n0, 1e-3 * b.sum() / atoms_min / len(sets))
    x0 = np.concatenate([n0, *y0])

    # A magnetic phase is smooth only within one branch (see models/magnetic.py).
    # Each free set is solved inside the branch it starts in, held there by
    # constraints. If one of those constraints is active at the solution, the
    # set has reached the edge of its branch: switch to the neighbouring branch
    # and solve again from there. The IHJ function is C¹ at τ = 1, so a solution
    # that sits exactly on that edge is valid on both sides.
    branches = {k: s.phase.magnetic_branch(y0[k], T) for k, s in enumerate(sets) if getattr(s.phase, "magnetic", None)}
    # a fixed set cannot leave its branch, so it needs no constraints
    held = [k for k in branches if sets[k].phase.n_y and not (fixed is not None and fixed[k])]
    t0 = time.perf_counter()
    seen, best = set(), None
    for _ in range(1 + 3 * len(held)):
        seen.add(tuple(sorted(branches.items())))
        r, active = _solve_once(sets, b, T, boxes, branches, held, n_max, x0, els, options)
        ok = str(getattr(r.status, "value", r.status)) == "optimal"
        if best is None or (ok and (not best[1] or r.objective < best[0].objective)):
            best = (r, ok)
        if not ok or not active:
            break
        nxt = dict(branches)
        for k, i in active:
            nxt[k] = nxt[k]._replace(**{nxt[k]._fields[i]: not nxt[k][i]})
        if tuple(sorted(nxt.items())) in seen:
            break  # the solution sits on an edge between branches already tried
        branches = nxt
        x0 = np.asarray(r.x, dtype=float)
    r = best[0]
    wall = time.perf_counter() - t0

    K = len(sets)
    x = np.asarray(r.x, dtype=float)
    ys = np.split(x[K:], np.cumsum([s.phase.n_y for s in sets])[:-1])
    new = [CompositionSet(s.phase, yk, float(nk)) for s, yk, nk in zip(sets, ys, x[:K])]
    if r.multipliers is None:
        mu = np.full(len(els), np.nan)
    else:
        mu = -np.asarray(r.multipliers, dtype=float)[: len(els)]  # the balance rows come first

    G = float(sum(cs.amount * float(cs.phase.gibbs(cs.y, NUM, T)) for cs in new))
    amounts = np.array([cs.amount for cs in new])
    primal = float(np.max(np.abs(amount_matrix(new) @ amounts - b)))
    tangency = max(
        (abs(float(tangent_distance(cs.phase, cs.y, mu, NUM, T))) for cs in new if cs.amount > amount_tol),
        default=0.0,
    )
    return RestrictedResult(
        status=str(getattr(r.status, "value", r.status)),
        sets=new,
        mu=mu,
        G=G,
        residual=abs(G - float(mu @ b)),
        primal_infeasibility=primal,
        tangency_residual=tangency,
        iterations=int(getattr(r, "iterations", -1) or -1),
        wall_time=wall,
    )


#: a branch constraint within this of zero, at the solution, is active
BRANCH_TOL = 1e-7


def _solve_once(sets, b, T, boxes, branches, held, n_max, x0, els, options):
    """One local solve of R(S), with each magnetic set in ``held`` kept on its branch.

    Returns the NLP result, and the ``(set, component)`` pairs whose branch
    constraint is active at the solution.
    """
    m = dm.Model("restricted")
    # declaration order is the evaluator's variable order: all N, then each set's y
    N = [m.continuous(f"N{k}", lb=0.0, ub=n_max) for k in range(len(sets))]
    Y = [[m.continuous(f"y{k}_{i}", lb=lo, ub=hi) for i, (lo, hi) in enumerate(box)] for k, box in enumerate(boxes)]
    ops = [EXPR.with_branch(branches[k]) if k in branches else EXPR for k in range(len(sets))]
    m.minimize(sum(N[k] * s.phase.gibbs(Y[k], ops[k], T) for k, s in enumerate(sets)))
    # the balance rows first: their multipliers are −μ
    for j, e in enumerate(els):
        lhs = sum(N[k] * s.phase.amounts(Y[k])[j] for k, s in enumerate(sets))
        m.subject_to(lhs == float(b[j]), name=f"balance_{e}")
    for k in held:
        for i, g in enumerate(sets[k].phase.branch_constraints(Y[k], T, branches[k])):
            if g is not None:
                m.subject_to(g >= 0, name=f"branch{k}_{i}")

    r = local_nlp_solve(m, x0, options)
    active = []
    if held and r.x is not None:
        x = np.asarray(r.x, dtype=float)
        ys = np.split(x[len(sets) :], np.cumsum([s.phase.n_y for s in sets])[:-1])
        for k in held:
            for i, g in enumerate(sets[k].phase.branch_constraints(ys[k], T, branches[k])):
                if g is not None and float(g) <= BRANCH_TOL:
                    active.append((k, i))
    return r, active
