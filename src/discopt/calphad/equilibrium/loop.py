"""Composition-set generation: alternate restricted solves and global pricing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discopt.calphad.equilibrium.certificate import Certificate, certify
from discopt.calphad.equilibrium.common import DOMAIN_FLOOR
from discopt.calphad.equilibrium.pricing import PricingResult, price_phases
from discopt.calphad.equilibrium.restricted import CompositionSet, RestrictedResult, solve_restricted

__all__ = ["IterationRecord", "EquilibriumResult", "equilibrium", "is_duplicate"]


@dataclass
class IterationRecord:
    iteration: int
    restricted: RestrictedResult
    pricing: list[PricingResult]
    added: list[CompositionSet] = field(default_factory=list)
    #: destabilizing pricing points rejected as duplicates of existing sets
    rejected: list[PricingResult] = field(default_factory=list)
    #: sets dropped after R because their amount fell to (near) zero
    removed: list[CompositionSet] = field(default_factory=list)


@dataclass
class EquilibriumResult:
    phases: list
    b: np.ndarray
    composition_sets: list[CompositionSet]
    mu: np.ndarray
    #: total G of the restricted solution, in units of RT
    G: float
    certificate: Certificate
    history: list[IterationRecord]

    @property
    def n_iterations(self) -> int:
        return len(self.history)

    def summary(self) -> str:
        c = self.certificate
        lines = [
            f"{c.level.name} after {self.n_iterations} iteration(s); eps = {c.eps:g} RT",
            f"  mu/RT = {np.array2string(self.mu, precision=8)}   G/RT = {self.G:.10f}",
        ]
        for cs in self.composition_sets:
            lines.append(f"  {cs.phase.name}: x = {np.array2string(cs.x, precision=8)}  amount = {cs.amount:.8f}")
        for name, lb in c.lower_bounds.items():
            lb_s = "none" if lb is None else f"{lb:.3e}"
            lines.append(f"  pricing {name}: LB = {lb_s}  UB = {c.upper_bounds[name]:.3e}")
        if c.gap_bound is not None:
            lines.append(f"  G_R - G_global <= {c.gap_bound:.3e} RT (residual {c.residual:.1e})")
        lines += [f"  note: {msg}" for msg in c.messages]
        return "\n".join(lines)


def is_duplicate(sets, phase, y, tol: float) -> bool:
    """True if some set of the same phase has ‖y − y_α‖∞ < tol."""
    y = np.asarray(y, dtype=float)
    return any(s.phase is phase and float(np.max(np.abs(s.y - y))) < tol for s in sets)


def equilibrium(
    phases,
    *,
    x=None,
    b=None,
    eps: float = 1e-5,
    max_iter: int = 20,
    dedup_tol: float = 1e-4,
    amount_tol: float = 1e-10,
    T: float | None = None,
    gap_tolerance: float = 1e-6,
    time_limit: float = 600.0,
    domain_floor: float = DOMAIN_FLOOR,
) -> EquilibriumResult:
    """Compute an equilibrium by composition-set generation and certify it.

    ``phases`` is a phase model or a list of them. Pass the overall composition
    either as ``x`` (the fraction of the first element in a binary) or as the
    element amounts ``b``. ``eps`` is the stability tolerance, in RT per formula unit.
    """
    phases = list(phases) if isinstance(phases, (list, tuple)) else [phases]
    if (x is None) == (b is None):
        raise ValueError("give exactly one of x= or b=")
    if b is None:
        if len(phases[0].elements) != 2:
            raise ValueError("x= is only for binary systems; pass b= instead")
        b = (float(x), 1.0 - float(x))
    b = np.asarray(b, dtype=float)

    sets = [CompositionSet(phases[0], phases[0].initial_y(b), 1.0)]
    history: list[IterationRecord] = []
    for it in range(max_iter):
        R = solve_restricted(sets, b, domain_floor=domain_floor, amount_tol=amount_tol)
        active = [s for s in R.sets if s.amount > amount_tol] or [max(R.sets, key=lambda s: s.amount)]
        removed = [s for s in R.sets if s not in active]
        record = IterationRecord(it, R, [], removed=removed)
        history.append(record)
        if not R.converged:
            break
        record.pricing = price_phases(
            phases, R.mu, gap_tolerance=gap_tolerance, time_limit=time_limit, domain_floor=domain_floor
        )
        for p in record.pricing:
            if not p.destabilizing(eps):
                continue
            if is_duplicate(active + record.added, p.phase, p.y, dedup_tol):
                record.rejected.append(p)
            else:
                record.added.append(CompositionSet(p.phase, p.y, 0.0))
        if not record.added:
            break
        sets = active + record.added

    last = history[-1]
    cert = certify(last.restricted, last.pricing, b, eps, T=T)
    if last.added:
        cert.messages.append(f"max_iter={max_iter} reached with sets still being added")
    if last.rejected:
        cert.messages.append("destabilizing point rejected as a duplicate: R did not reach that tangency")
    return EquilibriumResult(
        phases=phases,
        b=b,
        composition_sets=[s for s in last.restricted.sets if s.amount > amount_tol],
        mu=last.restricted.mu,
        G=last.restricted.G,
        certificate=cert,
        history=history,
    )
