"""Phase pricing: global minimization of the tangent-plane distance.

For a phase φ and chemical potentials μ:

    P_φ(μ) = min_y Ψ_φ(y) = G_φ(y) − μ·A_φ(y)

discopt's global solver runs on the inner box, where every site fraction is kept
at least ``domain_floor`` from its bounds. The boundary slabs outside that box
are bounded by interval evaluation of the same expression. The reported lower
bound is the smaller of the two, so it covers the full site-fraction domain.
A phase with no site fractions (a stoichiometric compound) is priced by
evaluating Ψ exactly.

:func:`price_phase_local` is the cheap alternative. It samples the box and
refines the best samples with a bounded local minimizer. The result is an upper
bound only: enough to generate composition sets, but not to certify.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any

import discopt.modeling as dm
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, inner_bounds, tangent_distance
from discopt.calphad.ops import EXPR, INTERVAL, NUM, Interval

__all__ = ["PricingResult", "price_phase", "price_phase_local", "price_phases", "boundary_lower_bound"]


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
    #: best bound read during the search), "none" (a global search that gave no bound), or
    #: "local" (local pricing, which never gives a bound)
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
        phase=phase,
        mu=mu,
        y=y,
        ub=psi,
        lb=psi,
        solver_lb=psi,
        boundary_lb=math.inf,
        status="exact",
        gap_certified=True,
        node_count=0,
        wall_time=time.perf_counter() - t0,
        bound_source="exact",
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
        return _price_fixed(phase, mu, T)  # exact, magnetic or not
    if getattr(phase, "magnetic", None) is not None:
        return _price_without_bound(phase, mu, T, domain_floor)
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
        r = m.solve(
            gap_tolerance=gap_tolerance,
            time_limit=max(min(limit, remaining), 0.1),
            node_callback=record_bound,
            **solve_kwargs,
        )
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


def _price_without_bound(phase, mu, T, domain_floor) -> PricingResult:
    """Search a phase that cannot be priced globally yet. The result has an upper bound and no lower bound.

    A magnetic phase has no interval extension yet (issue #6), so neither the
    global solve nor the boundary bound can run. It is searched locally. The
    certificate then reports ``GLOBAL_SEARCH_INCOMPLETE`` for it, or
    ``LOCAL_EQUILIBRIUM`` if the search found a destabilizing point.
    """
    res = price_phase_local(phase, mu, T=T, domain_floor=domain_floor)
    res.lb = res.solver_lb = None
    res.boundary_lb = -math.inf
    res.gap_certified = False
    res.bound_source = "none"
    res.status = "no global bound: magnetic phases have no interval extension yet"
    return res


def _psi_many(phase, pts, mu, T) -> np.ndarray:
    """Ψ at each row of ``pts``: vectorized when the model allows it, otherwise point by point."""
    try:
        with np.errstate(all="ignore"):
            v = np.asarray(tangent_distance(phase, list(pts.T), mu, NUM, T), dtype=float)
        if v.shape != (len(pts),):
            raise ValueError("model did not broadcast")
    except Exception:
        with np.errstate(all="ignore"):
            v = np.array([float(tangent_distance(phase, p, mu, NUM, T)) for p in pts])
    return np.where(np.isfinite(v), v, np.inf)


def price_phase_local(
    phase,
    mu,
    *,
    T: float | None = None,
    n_points: int = 2000,
    n_starts: int = 5,
    domain_floor: float = DOMAIN_FLOOR,
    seed: int = 0,
) -> PricingResult:
    """Minimize Ψ_φ(y) locally: sample the site-fraction box, then refine the best samples.

    The samples form a tensor grid when about ``n_points`` points still give at
    least five per axis, and a scrambled Halton sequence otherwise. The
    ``n_starts`` best samples that are spread apart are refined with L-BFGS-B
    inside the inner box. The result carries only an upper bound (``lb=None``,
    ``bound_source="local"``). A narrow well between samples can be missed, which
    is why ``equilibrium(certify="final")`` ends with global pricing.
    """
    mu = np.asarray(mu, dtype=float)
    if phase.n_y == 0:
        return _price_fixed(phase, mu, T)
    t0 = time.perf_counter()
    box = np.array(inner_bounds(phase, domain_floor), dtype=float)
    lo, hi = box[:, 0], box[:, 1]
    n = phase.n_y
    k = int(math.floor(n_points ** (1.0 / n) + 1e-9))
    if k >= 5:
        axes = np.meshgrid(*[np.linspace(a, c, k) for a, c in box], indexing="ij")
        pts = np.stack([g.ravel() for g in axes], axis=1)
    else:
        pts = lo + qmc.Halton(d=n, seed=seed).random(n_points) * (hi - lo)
    psi = _psi_many(phase, pts, mu, T)

    starts, spread = [], 0.1 * (hi - lo)
    for i in np.argsort(psi):
        if not np.isfinite(psi[i]) or len(starts) == n_starts:
            break
        if all(np.any(np.abs(pts[i] - s) > spread) for s in starts):
            starts.append(pts[i])

    def f(z):
        with np.errstate(all="ignore"):
            v = float(tangent_distance(phase, z, mu, NUM, T))
        return v if math.isfinite(v) else 1e300

    best_y, best = None, math.inf
    for z0 in starts:
        r = minimize(f, z0, method="L-BFGS-B", bounds=box)
        for z in (np.clip(r.x, lo, hi), z0):
            v = f(z)
            if v < best:
                best_y, best = np.array(z, dtype=float), v
    return PricingResult(
        phase=phase,
        mu=mu,
        y=best_y,
        ub=best,
        lb=None,
        solver_lb=None,
        boundary_lb=-math.inf,
        status="local",
        gap_certified=False,
        node_count=0,
        wall_time=time.perf_counter() - t0,
        bound_source="local",
        stages=0,
    )


def price_phases(phases, mu, *, method: str = "global", **kwargs) -> list[PricingResult]:
    """Price every phase with ``method="global"`` (:func:`price_phase`) or ``"local"`` (:func:`price_phase_local`).

    Runs serially for now; a parallel version waits on discopt#1246.
    """
    funcs = {"global": price_phase, "local": price_phase_local}
    if method not in funcs:
        raise ValueError(f"method must be one of {sorted(funcs)}, got {method!r}")
    return [funcs[method](ph, mu, **kwargs) for ph in phases]
