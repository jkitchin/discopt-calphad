"""Phase pricing: global minimization of the tangent-plane distance.

For a phase φ and chemical potentials μ:

    P_φ(μ) = min_y Ψ_φ(y) = G_φ(y) − μ·A_φ(y)

discopt's global solver runs on the inner box, where every site fraction is kept
at least ``domain_floor`` from its bounds. The boundary slabs outside that box
are bounded by interval evaluation of the same expression. The reported lower
bound is the smaller of the two, so it covers the full site-fraction domain.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import discopt.modeling as dm
import numpy as np

from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, inner_bounds, tangent_distance
from discopt.calphad.ops import EXPR, INTERVAL, NUM, Interval

__all__ = ["PricingResult", "price_phase", "price_phases", "boundary_lower_bound"]


@dataclass
class PricingResult:
    phase: Any
    mu: np.ndarray
    #: best site fractions found (the UB point), or None if the solve found none
    y: np.ndarray | None
    #: Ψ at ``y``, re-evaluated in floating point (``inf`` if there is no point)
    ub: float
    #: valid lower bound on Ψ over the full domain, or None when discopt gave no valid bound
    lb: float | None
    #: discopt's bound on the inner box (used only when ``gap_certified``)
    solver_lb: float | None
    #: interval lower bound on the boundary slabs
    boundary_lb: float
    status: str
    gap_certified: bool
    node_count: int
    wall_time: float

    def destabilizing(self, eps: float) -> bool:
        return self.ub < -eps

    def certified(self, eps: float) -> bool:
        return self.lb is not None and self.lb >= -eps


def boundary_lower_bound(phase, mu, domain_floor: float = DOMAIN_FLOOR) -> float:
    """Interval lower bound on Ψ over the slabs within ``domain_floor`` of a bound."""
    outer = phase.y_bounds
    lbs = []
    for i, (lo, hi) in enumerate(outer):
        for slab in ((lo, lo + domain_floor), (hi - domain_floor, hi)):
            box = [Interval(a, c) for a, c in outer]
            box[i] = Interval(*slab)
            lbs.append(tangent_distance(phase, box, mu, INTERVAL).lo)
    return min(lbs)


def price_phase(
    phase,
    mu,
    *,
    gap_tolerance: float = 1e-6,
    time_limit: float = 600.0,
    domain_floor: float = DOMAIN_FLOOR,
    **solve_kwargs,
) -> PricingResult:
    """Globally minimize Ψ_φ(y) = G_φ(y) − μ·A_φ(y) with discopt."""
    mu = np.asarray(mu, dtype=float)
    m = dm.Model(f"price_{phase.name}")
    box = inner_bounds(phase, domain_floor)
    y = [m.continuous(f"y{i}", lb=lo, ub=hi) for i, (lo, hi) in enumerate(box)]
    m.minimize(tangent_distance(phase, y, mu, EXPR))

    t0 = time.perf_counter()
    r = m.solve(gap_tolerance=gap_tolerance, time_limit=time_limit, **solve_kwargs)
    wall = time.perf_counter() - t0

    ystar, ub = None, math.inf
    if r.x:
        ystar = np.array([float(np.asarray(r.x[f"y{i}"])) for i in range(len(box))])
        ub = float(tangent_distance(phase, ystar, mu, NUM))
    # discopt 0.8.0 has no public bound-validity flag (discopt#1244), so gap_certified stands in for one.
    solver_lb = float(r.bound) if (r.gap_certified and r.bound is not None) else None
    boundary = boundary_lower_bound(phase, mu, domain_floor)
    return PricingResult(
        phase=phase,
        mu=mu,
        y=ystar,
        ub=ub,
        lb=None if solver_lb is None else min(solver_lb, boundary),
        solver_lb=solver_lb,
        boundary_lb=boundary,
        status=str(r.status),
        gap_certified=bool(r.gap_certified),
        node_count=int(r.node_count or 0),
        wall_time=wall,
    )


def price_phases(phases, mu, **kwargs) -> list[PricingResult]:
    """Price every phase. Runs serially for now; a parallel version waits on discopt#1246."""
    return [price_phase(ph, mu, **kwargs) for ph in phases]
