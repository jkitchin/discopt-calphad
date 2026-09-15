"""Phase pricing: global minimization of the tangent-plane distance.

For a phase φ and chemical potentials μ:

    P_φ(μ) = min_y Ψ_φ(y) = G_φ(y) − μ·A_φ(y)

discopt's global solver runs on the inner box, where every site fraction is kept
at least ``domain_floor`` from its bounds. The boundary slabs outside that box
are bounded by interval evaluation of the same expression. The reported lower
bound is the smaller of the two, so it covers the full site-fraction domain.
A phase with no site fractions (a stoichiometric compound) is priced by
evaluating Ψ exactly.
"""

from __future__ import annotations

import heapq
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
    #: where ``solver_lb`` came from: "exact", "tree" (a closed gap), "callback" (a certified
    #: best bound read during the search), or "none"
    bound_source: str = "none"
    #: number of discopt solves (stages) run
    stages: int = 1

    def destabilizing(self, eps: float) -> bool:
        return self.ub < -eps

    def certified(self, eps: float) -> bool:
        return self.lb is not None and self.lb >= -eps


def boundary_lower_bound(
    phase,
    mu,
    domain_floor: float = DOMAIN_FLOOR,
    T: float | None = None,
    *,
    target: float | None = None,
    max_boxes: int = 2000,
) -> float:
    """Interval lower bound on Ψ over the slabs within ``domain_floor`` of a bound.

    With several site fractions, a slab is thin in one direction but spans the
    full range of the others, so one interval evaluation is loose. Each slab is
    therefore refined best-first: the box with the lowest bound is bisected along
    its widest side. Refinement stops when that bound reaches ``target``, or when
    ``max_boxes`` evaluations have been spent on the slab. The result is the
    smallest bound among the remaining boxes, which is valid at any stopping
    point.
    """
    outer = phase.y_bounds

    def bound(box):
        return tangent_distance(phase, [Interval(a, c) for a, c in box], mu, INTERVAL, T).lo

    lbs = []
    for i, (lo, hi) in enumerate(outer):
        for slab in ((lo, lo + domain_floor), (hi - domain_floor, hi)):
            box = list(outer)
            box[i] = slab
            heap = [(bound(box), 0, box)]
            count = 1
            while heap[0][0] < (math.inf if target is None else target) and count < max_boxes:
                _, _, b = heapq.heappop(heap)
                j = max(range(len(b)), key=lambda k: b[k][1] - b[k][0])
                a, c = b[j]
                if c - a <= 1e-12:
                    heapq.heappush(heap, (bound(b), count, b))
                    break
                for half in ((a, 0.5 * (a + c)), (0.5 * (a + c), c)):
                    nb = list(b)
                    nb[j] = half
                    count += 1
                    heapq.heappush(heap, (bound(nb), count, nb))
            lbs.append(heap[0][0])
    return min(lbs) if lbs else math.inf


def _price_fixed(phase, mu, T) -> PricingResult:
    t0 = time.perf_counter()
    y = np.empty(0)
    psi = float(tangent_distance(phase, y, mu, NUM, T))
    return PricingResult(
        phase=phase, mu=mu, y=y, ub=psi, lb=psi, solver_lb=psi, boundary_lb=math.inf,
        status="exact", gap_certified=True, node_count=0, wall_time=time.perf_counter() - t0, bound_source="exact",
    )


def price_phase(
    phase,
    mu,
    *,
    T: float | None = None,
    eps: float | None = None,
    gap_tolerance: float = 1e-6,
    time_limit: float = 600.0,
    first_stage: float = 2.0,
    domain_floor: float = DOMAIN_FLOOR,
    **solve_kwargs,
) -> PricingResult:
    """Globally minimize Ψ_φ(y) = G_φ(y) − μ·A_φ(y) with discopt.

    With ``eps=None`` there is one solve, run until the gap closes or
    ``time_limit`` is reached.

    With ``eps`` set, pricing only has to decide whether LB ≥ −eps or UB < −eps.
    discopt 0.8 cannot be stopped from a callback: exceptions raised there are
    swallowed. So the solve runs in stages, starting at ``first_stage`` seconds
    and growing ×4 until the question is decided, the gap closes, or the total
    ``time_limit`` is used up. At every node batch, a ``node_callback`` records
    discopt's certified ``best_bound``. discopt documents this as a rigorous
    global bound that never over-reports. It is therefore a valid lower bound
    even from a stage that hits its time limit, and the largest one seen
    across stages is used.
    """
    mu = np.asarray(mu, dtype=float)
    if phase.n_y == 0:
        return _price_fixed(phase, mu, T)
    m = dm.Model(f"price_{phase.name}")
    box = inner_bounds(phase, domain_floor)
    y = [m.continuous(f"y{i}", lb=lo, ub=hi) for i, (lo, hi) in enumerate(box)]
    m.minimize(tangent_distance(phase, y, mu, EXPR, T))
    boundary = boundary_lower_bound(phase, mu, domain_floor, T, target=None if eps is None else -eps)

    seen: dict[str, float | None] = {"bound": None}

    def record_bound(ctx, model):
        b = ctx.best_bound
        if b is not None and math.isfinite(b) and (seen["bound"] is None or b > seen["bound"]):
            seen["bound"] = float(b)

    t0 = time.perf_counter()
    ystar, ub, nodes, stages = None, math.inf, 0, 0
    limit = time_limit if eps is None else min(first_stage, time_limit)
    while True:
        stages += 1
        remaining = time_limit - (time.perf_counter() - t0)
        r = m.solve(gap_tolerance=gap_tolerance, time_limit=max(min(limit, remaining), 0.1),
                    node_callback=record_bound, **solve_kwargs)
        nodes += int(r.node_count or 0)
        if r.x:
            yr = np.array([float(np.asarray(r.x[f"y{i}"])) for i in range(len(box))])
            ur = float(tangent_distance(phase, yr, mu, NUM, T))
            if ur < ub:
                ystar, ub = yr, ur
        if r.gap_certified and r.bound is not None:
            seen["bound"] = float(r.bound) if seen["bound"] is None else max(seen["bound"], float(r.bound))
        if r.gap_certified or eps is None or time.perf_counter() - t0 >= time_limit - 0.1:
            break
        lb_now = None if seen["bound"] is None else min(seen["bound"], boundary)
        if ub < -eps or (lb_now is not None and lb_now >= -eps):
            break
        limit *= 4.0
    wall = time.perf_counter() - t0

    solver_lb = seen["bound"]
    source = "none" if solver_lb is None else ("tree" if r.gap_certified else "callback")
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
        node_count=nodes,
        wall_time=wall,
        bound_source=source,
        stages=stages,
    )


def price_phases(phases, mu, **kwargs) -> list[PricingResult]:
    """Price every phase. Runs serially for now; a parallel version waits on discopt#1246."""
    return [price_phase(ph, mu, **kwargs) for ph in phases]
