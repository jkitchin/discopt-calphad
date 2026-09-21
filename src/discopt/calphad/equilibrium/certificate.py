"""Certificate levels and the certificate theorem.

For any μ and every feasible state of the full problem,

    G ≥ μ·b + N_tot · min_φ LB_φ,

so the restricted solution is at most  r − N_tot,max · min(0, min_φ LB_φ)
above the global minimum, where r = |G_R − μ·b|. Every value is in units of RT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from discopt.calphad.equilibrium.common import R_GAS

__all__ = ["CertificateLevel", "Certificate", "certify", "R_GAS"]


class CertificateLevel(IntEnum):
    """Ordered from weakest to strongest (see docs/architecture.md §6)."""

    #: the restricted solve did not converge
    FAILED = 0
    #: R solved; no pricing, or a phase is still destabilizing (UB < −ε)
    LOCAL_EQUILIBRIUM = 1
    #: pricing ran, but no phase has a valid lower bound, or only local pricing ran
    STABILITY_SEARCHED = 2
    #: some phase has no valid bound, or LB < −ε while UB ≥ −ε
    GLOBAL_SEARCH_INCOMPLETE = 3
    #: every phase has LB ≥ −ε
    CERTIFIED_TO_TOLERANCE = 4
    #: every LB ≥ −certify_tol and r ≤ kkt_tol
    GLOBALLY_CERTIFIED = 5


@dataclass
class Certificate:
    level: CertificateLevel
    eps: float
    #: per phase name: valid lower bound on Ψ, or None
    lower_bounds: dict[str, float | None]
    #: per phase name: best Ψ value found
    upper_bounds: dict[str, float]
    worst_lb: float | None
    residual: float
    primal_infeasibility: float
    n_total_max: float
    #: bound on G_R − G_global in units of RT for the whole system, or None
    gap_bound: float | None
    T: float | None = None
    messages: list[str] = field(default_factory=list)

    @property
    def eps_J_per_mol(self) -> float | None:
        return None if self.T is None else self.eps * R_GAS * self.T

    @property
    def gap_bound_J(self) -> float | None:
        if self.T is None or self.gap_bound is None:
            return None
        return self.gap_bound * R_GAS * self.T


def certify(
    restricted,
    pricing,
    b,
    eps: float,
    *,
    kkt_tol: float = 1e-8,
    certify_tol: float = 1e-9,
    T: float | None = None,
) -> Certificate:
    """Assign a certificate level from a restricted result and its pricing results."""
    b = np.asarray(b, dtype=float)
    pricing = list(pricing or [])
    atoms_min = min((p.phase.atoms_min for p in pricing), default=1.0)
    n_total_max = float(b.sum() / atoms_min)
    lbs = {p.phase.name: p.lb for p in pricing}
    ubs = {p.phase.name: p.ub for p in pricing}
    valid = [lb for lb in lbs.values() if lb is not None]
    worst_lb = min(valid) if valid else None
    msgs: list[str] = []

    if not restricted.converged:
        level = CertificateLevel.FAILED
        msgs.append(f"restricted solve status: {restricted.status}")
    elif not pricing:
        level = CertificateLevel.LOCAL_EQUILIBRIUM
        msgs.append("no pricing performed")
    elif any(p.destabilizing(eps) for p in pricing):
        level = CertificateLevel.LOCAL_EQUILIBRIUM
        msgs += [f"{p.phase.name} destabilizing: UB = {p.ub:.3e} < -eps" for p in pricing if p.destabilizing(eps)]
    elif any(getattr(p, "bound_source", None) == "local" for p in pricing):
        level = CertificateLevel.STABILITY_SEARCHED
        msgs.append("local pricing only, so no global lower bounds: call res.certify() to certify")
    elif not valid:
        level = CertificateLevel.STABILITY_SEARCHED
        msgs.append("no phase has a valid global lower bound")
    elif len(valid) < len(pricing) or worst_lb < -eps:
        level = CertificateLevel.GLOBAL_SEARCH_INCOMPLETE
        msgs += [
            f"{p.phase.name}: no valid lower bound" + (f" ({p.status})" if getattr(p, "status", "") else "")
            for p in pricing
            if p.lb is None
        ]
        msgs += [f"{n}: LB = {lb:.3e} < -eps" for n, lb in lbs.items() if lb is not None and lb < -eps]
    elif -worst_lb <= certify_tol and restricted.residual <= kkt_tol:
        level = CertificateLevel.GLOBALLY_CERTIFIED
    else:
        level = CertificateLevel.CERTIFIED_TO_TOLERANCE

    gap_bound = None
    if valid and len(valid) == len(pricing):
        gap_bound = restricted.residual + n_total_max * max(0.0, -worst_lb)
    return Certificate(
        level=level,
        eps=eps,
        lower_bounds=lbs,
        upper_bounds=ubs,
        worst_lb=worst_lb,
        residual=restricted.residual,
        primal_infeasibility=restricted.primal_infeasibility,
        n_total_max=n_total_max,
        gap_bound=gap_bound,
        T=T,
        messages=msgs,
    )
