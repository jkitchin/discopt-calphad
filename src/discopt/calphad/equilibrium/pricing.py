"""Phase pricing: global minimization of the tangent-plane distance.

For a phase φ and chemical potentials μ:

    P_φ(μ) = min_y Ψ_φ(y) = G_φ(y) − μ·A_φ(y)

discopt's global solver runs on the inner box, where every site fraction is kept
at least ``domain_floor`` from its bounds. The boundary slabs outside that box
are bounded by interval evaluation of the same expression. The reported lower
bound is the smaller of the two, so it covers the full site-fraction domain.
A phase with no site fractions (a stoichiometric compound) is priced by
evaluating Ψ exactly. A magnetic phase is piecewise in y: it is split into the
branches it can reach, each solved as its own model, and the smallest bound
over them is the phase's (see ``_pieces``).

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
from discopt.calphad.ops import DUAL, EXPR, NUM, Dual, Interval

__all__ = [
    "PricingResult",
    "price_phase",
    "price_phase_local",
    "price_phases",
    "boundary_lower_bound",
    "interval_lower_bound",
]

#: phases with at most this many site-fraction variables are first priced by interval branch and bound
INTERVAL_MAX_VARS = 4


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
    #: "local" (local pricing, which never gives a bound), or "interval" (interval branch and bound
    #: over the whole domain decided the question before any discopt solve)
    bound_source: str = "none"
    #: number of discopt solves (stages) run
    stages: int = 1

    def destabilizing(self, eps: float) -> bool:
        return self.ub < -eps

    def certified(self, eps: float) -> bool:
        return self.lb is not None and self.lb >= -eps


#: a box side at most this wide is kept as an interval in the mean-value form, not expanded about its center
THIN_SIDE = 1e-6


def _taylor_lower(phase, mu, T, box) -> float:
    """Interval lower bound on Ψ over ``box``: the better of the natural and the mean-value form.

    One pass over a first-order Taylor model (:class:`~discopt.calphad.ops.Dual`)
    gives both: the natural enclosure of Ψ(box), and

        Ψ(y) ∈ Ψ(c) + Σ_j ∂_jΨ(box) · (y_j − c_j)

    for an expansion point c, by the mean value theorem. The natural form
    overestimates by O(width), the mean-value form by O(width²), so the latter
    closes far faster near a minimum. Sides are expanded about their midpoint,
    except next to the domain's edge: ln y in the entropy's derivative is
    unbounded there on one side only, and expanding about the far corner keeps
    the term finite (y_j − c_j has one sign). A minimizer of an ordered phase
    sits exactly there. Thin sides (a boundary slab's coordinate among them)
    and sides spanning the whole range stay intervals. A term the backend
    cannot expand, such as magnetism, enters as its interval over the box.
    """
    outer = phase.y_bounds
    expand = [
        j for j, ((a, c), (lo, hi)) in enumerate(zip(box, outer)) if c - a > THIN_SIDE and not (a <= lo and c >= hi)
    ]
    n = len(expand)
    point = {}
    for j in expand:
        (a, c), (lo, hi) = box[j], outer[j]
        point[j] = c if a <= lo else a if c >= hi else 0.5 * (a + c)
    ys = []
    for j, (a, c) in enumerate(box):
        if j in point:
            k = expand.index(j)
            ys.append(
                Dual(Interval(a, c), Interval(point[j]), tuple(Interval(1.0) if i == k else None for i in range(n)))
            )
        else:
            ys.append(Dual.const(Interval(a, c), n))
    psi = tangent_distance(phase, ys, mu, DUAL, T)
    if not isinstance(psi, Dual):
        return Interval._coerce(psi).lo
    acc = psi.c
    for k, j in enumerate(expand):
        if psi.d[k] is not None:
            a, c = box[j]
            x = point[j]
            # y − x has an exact sign at a corner; rounding 0 outward would turn 0 · ∞ into −∞
            if x == a:
                off = Interval(0.0, math.nextafter(c - a, math.inf))
            elif x == c:
                off = Interval(math.nextafter(a - c, -math.inf), 0.0)
            else:
                off = Interval(a, c) - x
            acc = acc + psi.d[k] * off
    lbs = [v for v in (psi.v.lo, acc.lo) if not math.isnan(v)]
    return max(lbs) if lbs else -math.inf


def _interval_bound_fn(phase, mu, T):
    """Box -> an interval lower bound on Ψ over the box (+inf if no point of it is in the phase's domain).

    The bound is the better of the natural interval form and the mean-value
    form (:func:`_taylor_lower`).
    """
    domain = getattr(phase, "domain_constraints", lambda y: [])
    equalities = getattr(phase, "domain_equalities", lambda y: [])

    def bound(box):
        iv = [Interval(a, c) for a, c in box]
        if any(Interval._coerce(g).hi < 0 for g in domain(iv)):
            return math.inf  # the box lies outside the phase's domain (see CEFPhase.domain_constraints)
        if any(not Interval._coerce(g).lo <= 0 <= Interval._coerce(g).hi for g in equalities(iv)):
            return math.inf  # no point of the box is charge-neutral
        return _taylor_lower(phase, mu, T, box)

    return bound


def _refine(bound, box, target: float | None, max_boxes: int) -> float:
    """Best-first interval branch and bound on ``box``: bisect the lowest box along its widest side.

    Stops when the lowest bound reaches ``target`` or after ``max_boxes``
    evaluations. The smallest bound among the remaining boxes is valid at any
    stopping point.
    """
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
    return heap[0][0]


def boundary_lower_bound(
    phase,
    mu,
    domain_floor: float = DOMAIN_FLOOR,
    T: float | None = None,
    *,
    target: float | None = None,
    max_boxes: int = 20000,
) -> float:
    """Interval lower bound on Ψ over the slabs within ``domain_floor`` of a bound.

    With several site fractions, a slab is thin in one direction but spans the
    full range of the others, so one interval evaluation is loose. Each slab is
    therefore refined best-first (:func:`_refine`) until its bound reaches
    ``target`` or ``max_boxes`` evaluations have been spent on it.
    """
    if domain_floor <= 0:
        return math.inf  # no slabs: the discopt box is the whole domain
    outer = phase.y_bounds
    bound = _interval_bound_fn(phase, mu, T)
    lbs = []
    for i, (lo, hi) in enumerate(outer):
        for slab in ((lo, lo + domain_floor), (hi - domain_floor, hi)):
            box = list(outer)
            box[i] = slab
            lbs.append(_refine(bound, box, target, max_boxes))
    return min(lbs) if lbs else math.inf


def interval_lower_bound(phase, mu, T: float | None = None, *, target: float | None = None, max_boxes: int = 20000):
    """Interval branch-and-bound lower bound on Ψ over the phase's whole domain.

    The natural interval extension overestimates by O(box width), so this
    closes quickly for a phase well above the tangent plane, and slowly for one
    that touches it. :func:`price_phase` runs it first and skips the discopt
    solve when it already decides the question.
    """
    return _refine(_interval_bound_fn(phase, mu, T), list(phase.y_bounds), target, max_boxes)


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

    A magnetic phase gives one model per branch it can reach. Each stage
    solves the branches still open, and the bound is the smallest over them.
    A branch that discopt proves infeasible inside the box contributes +∞.
    """
    mu = np.asarray(mu, dtype=float)
    if phase.n_y == 0:
        return _price_fixed(phase, mu, T)  # exact, magnetic or not
    if eps is not None and phase.n_y <= INTERVAL_MAX_VARS:
        # cheap first: interval branch and bound over the whole domain decides a phase well above the tangent
        t0 = time.perf_counter()
        lb_int = interval_lower_bound(phase, mu, T, target=-eps)
        if lb_int >= -eps:
            res = price_phase_local(phase, mu, T=T, domain_floor=domain_floor)
            res.lb, res.solver_lb, res.boundary_lb = lb_int, None, lb_int
            res.status, res.gap_certified, res.bound_source = "interval", True, "interval"
            res.wall_time = time.perf_counter() - t0
            return res
    box = inner_bounds(phase, domain_floor)
    pieces = _pieces(phase, mu, T, box)
    boundary = boundary_lower_bound(phase, mu, domain_floor, T, target=None if eps is None else -eps)

    t0 = time.perf_counter()
    ystar, ub, nodes, stages = None, math.inf, 0, 0
    limit = time_limit if eps is None else min(first_stage, time_limit)
    while True:
        stages += 1
        for pc in pieces:
            if eps is not None and ub < -eps:
                break  # decided: the phase destabilizes, whatever the other branches hold
            if pc.closed or (eps is not None and pc.bound is not None and pc.bound >= -eps):
                continue  # this piece is settled
            remaining = time_limit - (time.perf_counter() - t0)
            r = pc.model.solve(
                gap_tolerance=gap_tolerance,
                time_limit=max(min(limit, remaining), 0.1),
                node_callback=pc.record,
                **solve_kwargs,
            )
            pc.status = str(r.status)
            nodes += int(r.node_count or 0)
            if r.gap_certified and str(r.status) == "infeasible":
                pc.closed, pc.bound = True, math.inf  # this branch is empty inside the box
                continue
            if r.x:
                yr = np.array([float(np.asarray(r.x[f"y{i}"])) for i in range(len(box))])
                ur = float(tangent_distance(phase, yr, mu, NUM, T))
                if ur < ub:
                    ystar, ub = yr, ur
            if r.gap_certified and r.bound is not None:
                pc.closed = True
                pc.raise_bound(float(r.bound))
        closed = all(pc.closed for pc in pieces)
        if closed or eps is None or time.perf_counter() - t0 >= time_limit - 0.1:
            break
        lb_now = _min_bound(pieces)
        if ub < -eps or (lb_now is not None and min(lb_now, boundary) >= -eps):
            break
        limit *= 4.0
    wall = time.perf_counter() - t0

    solver_lb = _min_bound(pieces)
    source = "none" if solver_lb is None else ("tree" if closed else "callback")
    open_ = [pc.status for pc in pieces if not pc.closed]
    return PricingResult(
        phase=phase,
        mu=mu,
        y=ystar,
        ub=ub,
        lb=None if solver_lb is None else min(solver_lb, boundary),
        solver_lb=solver_lb,
        boundary_lb=boundary,
        status=open_[0] if open_ else ("optimal" if math.isfinite(ub) else "infeasible"),
        gap_certified=closed,
        node_count=nodes,
        wall_time=wall,
        bound_source=source,
        stages=stages,
    )


class _Piece:
    """One discopt pricing model, and the best certified bound seen on it."""

    def __init__(self, model):
        self.model = model
        self.bound: float | None = None
        self.closed = False
        self.status = "not solved"

    def raise_bound(self, b: float) -> None:
        if math.isfinite(b) and (self.bound is None or b > self.bound):
            self.bound = b

    def record(self, ctx, model) -> None:
        if ctx.best_bound is not None:
            self.raise_bound(float(ctx.best_bound))


def _min_bound(pieces) -> float | None:
    """The smallest bound over the pieces, or None while one of them has none."""
    if any(pc.bound is None for pc in pieces):
        return None
    return min(pc.bound for pc in pieces)


def _pieces(phase, mu, T, box) -> list[_Piece]:
    """The discopt models whose minima together give P_φ(μ) on ``box``.

    A smooth phase is one model. A magnetic phase is one model per branch that
    an interval evaluation over the box cannot rule out (see
    :mod:`discopt.calphad.models.magnetic`). Each holds y on its branch, with an
    auxiliary variable s = T_c'/T bounded to that branch's side of 1. The
    branches are closed sets that cover the box, and Ψ is continuous across
    their edges, so the smallest of their minima is the minimum over the box.
    """

    def model(name):
        m = dm.Model(name)
        y = [m.continuous(f"y{i}", lb=lo, ub=hi) for i, (lo, hi) in enumerate(box)]
        for i, g in enumerate(getattr(phase, "symmetry_constraints", lambda y: [])(y)):
            m.subject_to(g >= 0, name=f"symmetry{i}")
        for i, g in enumerate(getattr(phase, "domain_constraints", lambda y: [])(y)):
            m.subject_to(g >= 0, name=f"domain{i}")
        for i, g in enumerate(getattr(phase, "domain_equalities", lambda y: [])(y)):
            m.subject_to(g == 0, name=f"neutral{i}")
        kw = {}
        if hasattr(phase, "average_expressions"):
            # a partitioned phase: the site averages as variables, so G_dis(x) and G_ord(x) are
            # written in a few variables rather than as polynomials expanded in y
            x = {}
            for c, e in phase.average_expressions(y).items():
                x[c] = m.continuous(f"x_{c}", lb=0.0, ub=1.0)
                m.subject_to(x[c] - e == 0, name=f"average_{c}")
            kw["x"] = x
        if hasattr(phase, "ratio_expressions"):
            # an ionic liquid: the site ratios P and Q as bounded variables, so the entropy terms are
            # products of a variable and an entropy node rather than of a polynomial and one
            (p_lo, p_hi), (q_lo, q_hi) = phase.ratio_bounds()
            p_e, q_e = phase.ratio_expressions(y)
            P = m.continuous("P", lb=p_lo, ub=p_hi)
            Q = m.continuous("Q", lb=q_lo, ub=q_hi)
            m.subject_to(P - p_e == 0, name="ratio_P")
            m.subject_to(Q - q_e == 0, name="ratio_Q")
            kw["ratios"] = (P, Q)
        return m, y, kw

    def objective(y, kw, ops):
        if not kw:
            return tangent_distance(phase, y, mu, ops, T)
        return phase.gibbs(y, ops, T, **kw) - sum(float(m_) * a for m_, a in zip(mu, phase.amounts(y)))

    if getattr(phase, "magnetic", None) is None:
        m, y, kw = model(f"price_{phase.name}")
        m.minimize(objective(y, kw, EXPR))
        return [_Piece(m)]
    out = []
    for k, (branch, ranges) in enumerate(phase.magnetic_pieces(box, T)):
        m, y, kw = model(f"price_{phase.name}_{k}")
        x = kw.get("x")
        s = [m.continuous(f"s{j}", lb=lo, ub=hi) if hi > lo else None for j, (lo, hi) in enumerate(ranges)]
        if x is None:
            eqs, ineqs = phase.pricing_constraints(y, T, branch, s)
        else:
            eqs, ineqs = phase.pricing_constraints(y, T, branch, s, x=x)
        for i, g in enumerate(eqs):
            m.subject_to(g == 0, name=f"s_def{i}")
        for i, g in enumerate(ineqs):
            m.subject_to(g >= 0, name=f"branch{i}")
        m.minimize(objective(y, kw, EXPR.with_branch(branch, s)))
        out.append(_Piece(m))
    return out


def _outside(phase, z) -> bool:
    """True if ``z`` violates one of the phase's domain constraints."""
    return any(float(g) < 0 for g in getattr(phase, "domain_constraints", lambda y: [])(z))


def _psi_many(phase, pts, mu, T) -> np.ndarray:
    """Ψ at each row of ``pts`` (+inf outside the phase's domain): vectorized when the model allows it."""
    v = _psi_many_box(phase, pts, mu, T)
    if getattr(phase, "vacancy_limited", False):
        v = np.where([_outside(phase, p) for p in pts], np.inf, v)
    return v


def _psi_many_box(phase, pts, mu, T) -> np.ndarray:
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
    eqs = getattr(phase, "domain_equalities", lambda y: [])
    constrained = bool(eqs(pts[0]))
    if constrained:
        # Box samples are almost never charge-neutral: rank them by Ψ plus a penalty on the charge,
        # then refine under the equality (SLSQP), and keep only neutral results.
        psi = psi + 1e3 * np.array([sum(float(g) ** 2 for g in eqs(p)) for p in pts])

    starts, spread = [], 0.1 * (hi - lo)
    for i in np.argsort(psi):
        if not np.isfinite(psi[i]) or len(starts) == n_starts:
            break
        if all(np.any(np.abs(pts[i] - s) > spread) for s in starts):
            starts.append(pts[i])

    def f(z):
        if _outside(phase, z):
            return 1e300
        with np.errstate(all="ignore"):
            v = float(tangent_distance(phase, z, mu, NUM, T))
        return v if math.isfinite(v) else 1e300

    def neutral(z):
        return all(abs(float(g)) < 1e-9 for g in eqs(z))

    best_y, best = None, math.inf
    for z0 in starts:
        if constrained:
            cons = [{"type": "eq", "fun": lambda z, i=i: float(eqs(z)[i])} for i in range(len(eqs(z0)))]
            cons += [
                {"type": "ineq", "fun": lambda z, i=i: float(phase.domain_constraints(z)[i])}
                for i in range(len(getattr(phase, "domain_constraints", lambda y: [])(z0)))
            ]
            r = minimize(f, z0, method="SLSQP", bounds=box, constraints=cons, options={"maxiter": 300, "ftol": 1e-12})
            cands = [np.clip(r.x, lo, hi)]
        else:
            r = minimize(f, z0, method="L-BFGS-B", bounds=box)
            cands = [np.clip(r.x, lo, hi), z0]
        for z in cands:
            if constrained and not neutral(z):
                continue
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
