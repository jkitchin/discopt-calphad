"""Composition-set generation: alternate restricted solves and pricing, then certify."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from discopt.calphad.equilibrium.certificate import Certificate, CertificateLevel
from discopt.calphad.equilibrium.certificate import certify as certify_state
from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, tangent_distance
from discopt.calphad.equilibrium.pricing import PricingResult, price_phases
from discopt.calphad.equilibrium.restricted import CompositionSet, RestrictedResult, solve_restricted
from discopt.calphad.ops import NUM

__all__ = ["CERTIFY_MODES", "IterationRecord", "EquilibriumResult", "equilibrium", "initial_set", "is_duplicate"]

#: ``equilibrium(certify=...)``: "none" prices locally only; "final" prices locally and
#: globally once no local point destabilizes; "every" prices globally in every iteration
CERTIFY_MODES = ("none", "final", "every")


@dataclass
class IterationRecord:
    iteration: int
    restricted: RestrictedResult
    #: the pricing that decided this iteration (global once it ran, otherwise local)
    pricing: list[PricingResult]
    added: list[CompositionSet] = field(default_factory=list)
    #: destabilizing pricing points rejected as duplicates of existing sets
    rejected: list[PricingResult] = field(default_factory=list)
    #: sets dropped after R because their amount fell to (near) zero
    removed: list[CompositionSet] = field(default_factory=list)
    #: local pricing that ran before the global pricing in ``pricing``, if any
    local_pricing: list[PricingResult] | None = None


@dataclass
class EquilibriumResult:
    phases: list
    b: np.ndarray
    T: float | None
    composition_sets: list[CompositionSet]
    mu: np.ndarray
    #: total G of the restricted solution, in units of RT
    G: float
    certificate: Certificate
    history: list[IterationRecord]

    @property
    def n_iterations(self) -> int:
        return len(self.history)

    @property
    def phase_names(self) -> list[str]:
        return sorted({cs.phase.name for cs in self.composition_sets})

    def certify(
        self,
        *,
        eps: float | None = None,
        gap_tolerance: float = 1e-6,
        time_limit: float = 600.0,
        domain_floor: float = DOMAIN_FLOOR,
    ) -> Certificate:
        """Price every phase globally at the stored μ and replace :attr:`certificate`.

        Use this on a result from ``equilibrium(certify="none")``, for example on
        selected points of a phase diagram. If a phase destabilizes, the stored
        state is not the global equilibrium. The certificate then says so, and
        ``equilibrium(res.phases, b=res.b, T=res.T, initial_sets=res.composition_sets)``
        continues from it.
        """
        eps = self.certificate.eps if eps is None else eps
        last = self.history[-1]
        if not last.restricted.converged:
            return self.certificate
        pricing = price_phases(
            self.phases,
            last.restricted.mu,
            method="global",
            T=self.T,
            eps=eps,
            gap_tolerance=gap_tolerance,
            time_limit=time_limit,
            domain_floor=domain_floor,
        )
        if last.local_pricing is None and any(p.bound_source == "local" for p in last.pricing):
            last.local_pricing = last.pricing
        last.pricing = pricing
        cert = certify_state(last.restricted, pricing, self.b, eps, T=self.T)
        if cert.level == CertificateLevel.LOCAL_EQUILIBRIUM and any(p.destabilizing(eps) for p in pricing):
            cert.messages.append(
                "not the global equilibrium; continue with "
                "equilibrium(res.phases, b=res.b, T=res.T, initial_sets=res.composition_sets)"
            )
        self.certificate = cert
        return cert

    def summary(self) -> str:
        c = self.certificate
        t = "" if self.T is None else f"T = {self.T:g} K; "
        lines = [
            f"{c.level.name} after {self.n_iterations} iteration(s); {t}eps = {c.eps:g} RT",
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
    """True if some set of the same phase has ‖y − y_α‖∞ < tol (every set of a compound counts)."""
    y = np.asarray(y, dtype=float)
    return any(s.phase is phase and (y.size == 0 or float(np.max(np.abs(s.y - y))) < tol) for s in sets)


def _merge_coincident(sets, tol: float) -> list[CompositionSet]:
    """Merge sets of the same phase whose site fractions agree within ``tol``, adding the amounts.

    Sets that need no merge are returned as the same objects, so a caller can
    tell, by identity, which sets were removed.
    """
    out: list[CompositionSet] = []
    for s in sets:
        twin = next((o for o in out if is_duplicate([o], s.phase, s.y, tol)), None)
        if twin is None:
            out.append(s)
        else:
            i = next(i for i, o in enumerate(out) if o is twin)
            out[i] = CompositionSet(twin.phase, twin.y, twin.amount + s.amount)
    return out


def initial_set(phases, b, T=None) -> CompositionSet:
    """Starting set: among phases that can meet ``b`` alone, the one with the lowest G per atom."""
    best = None
    for ph in phases:
        y = ph.initial_y(b)
        if y is None:
            continue
        cs = CompositionSet(ph, y, float(np.sum(b)))
        g_atom = float(ph.gibbs(cs.y, NUM, T)) / cs.atoms
        if best is None or g_atom < best[0]:
            best = (g_atom, cs)
    if best is None:
        raise ValueError("no phase can meet b with a single composition set; pass initial_sets=")
    best[1].amount = float(np.sum(b)) / best[1].atoms
    return best[1]


def _is_active_compound(phase, active) -> bool:
    return phase.n_y == 0 and any(s.phase is phase for s in active)


def _center_mu(R, others, T, local_kw) -> np.ndarray:
    """μ = μ₀ + D t that maximizes the smallest per-atom tangent distance of ``others``.

    That smallest distance is concave in μ (a minimum of minima of linear
    functions), so the search has one maximum. It uses local pricing, which is
    enough to place μ; the certificate is then decided at that μ as usual.
    """
    from scipy.optimize import minimize, minimize_scalar

    mu0, D = np.asarray(R.mu, float), np.asarray(R.mu_directions, float)
    if not others:
        return mu0
    kw = {**local_kw, "n_points": min(400, local_kw.get("n_points", 400))}
    kw.pop("method", None)

    def worst(t):
        mu = mu0 + D @ np.atleast_1d(t)
        vals = []
        for p in price_phases(others, mu, method="local", **kw):
            atoms = float(sum(float(a) for a in p.phase.amounts(p.y))) if p.y is not None else 1.0
            vals.append(p.ub / max(atoms, 1e-12))
        return -min(vals)

    if D.shape[1] == 1:
        # bracket the maximum: the tangent distances fall without bound far from the compound
        scale = max(1.0, float(np.max(np.abs(mu0))))
        t = minimize_scalar(worst, bracket=(-scale, scale), tol=1e-6).x
    else:
        t = minimize(worst, np.zeros(D.shape[1]), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9}).x
    return mu0 + D @ np.atleast_1d(t)


def _generate(record: IterationRecord, active, eps: float, dedup_tol: float) -> None:
    """Add a set for each destabilizing pricing point that is not a duplicate."""
    for p in record.pricing:
        if not p.destabilizing(eps):
            continue
        if is_duplicate(active + record.added, p.phase, p.y, dedup_tol):
            record.rejected.append(p)
        else:
            record.added.append(CompositionSet(p.phase, p.y, 0.0))


def equilibrium(
    phases,
    *,
    x=None,
    b=None,
    T: float | None = None,
    eps: float = 1e-5,
    certify: str = "final",
    initial_sets=None,
    max_iter: int = 20,
    dedup_tol: float = 1e-4,
    amount_tol: float = 1e-10,
    trace_amount: float = 1e-5,
    gap_tolerance: float = 1e-6,
    time_limit: float = 600.0,
    domain_floor: float = DOMAIN_FLOOR,
    local_points: int = 2000,
    local_starts: int = 5,
) -> EquilibriumResult:
    """Compute an equilibrium by composition-set generation, and certify it.

    ``phases`` is a phase model or a list of them. Pass the overall composition
    either as ``x`` (the fraction of the first element in a binary) or as the
    element amounts ``b``. ``eps`` is the stability tolerance, in RT per formula
    unit. ``initial_sets`` overrides the automatic starting set. A set whose
    amount is below ``trace_amount``·Σb and whose tangent distance exceeds ``eps``
    is dropped, and R is solved again without it.

    ``certify`` chooses how phases are priced:

    - ``"final"`` (default): local pricing (:func:`price_phase_local`, with
      ``local_points`` samples and ``local_starts`` refinements) generates
      composition sets. Once it finds nothing, every phase is priced globally
      at that μ. A destabilizing point found there is added and the loop goes
      on, so the certificate is the same as with ``"every"``.
    - ``"every"``: global pricing in every iteration.
    - ``"none"``: local pricing only. The result is at best
      ``STABILITY_SEARCHED``; call :meth:`EquilibriumResult.certify` later.

    ``gap_tolerance`` and ``time_limit`` (seconds per phase) apply to global pricing.
    """
    if certify not in CERTIFY_MODES:
        raise ValueError(f"certify must be one of {CERTIFY_MODES}, got {certify!r}")
    phases = list(phases) if isinstance(phases, (list, tuple)) else [phases]
    if (x is None) == (b is None):
        raise ValueError("give exactly one of x= or b=")
    if b is None:
        if len(phases[0].elements) != 2:
            raise ValueError("x= is only for binary systems; pass b= instead")
        b = (float(x), 1.0 - float(x))
    b = np.asarray(b, dtype=float)
    global_kw = dict(
        method="global", T=T, eps=eps, gap_tolerance=gap_tolerance, time_limit=time_limit, domain_floor=domain_floor
    )
    local_kw = dict(method="local", T=T, n_points=local_points, n_starts=local_starts, domain_floor=domain_floor)

    sets = list(initial_sets) if initial_sets else [initial_set(phases, b, T)]
    history: list[IterationRecord] = []
    small = trace_amount * float(np.sum(b))
    new: list[CompositionSet] = []
    exhausted = False
    for it in range(max_iter):
        R = solve_restricted(sets, b, T=T, domain_floor=domain_floor, amount_tol=amount_tol)
        if new:
            # A just-added set starts with a near-zero amount, and from there the solve can move its
            # site fractions to where Ψ > 0 and settle at N = 0 (a valid but useless KKT point). So
            # first hold the new sets at their pricing points, and let amounts and the other sets
            # adjust; then free everything from that point. The better of the two solutions is kept.
            fixed = [any(s is n for n in new) for s in sets]
            R1 = solve_restricted(sets, b, T=T, domain_floor=domain_floor, amount_tol=amount_tol, fixed=fixed)
            if R1.converged:
                R2 = solve_restricted(R1.sets, b, T=T, domain_floor=domain_floor, amount_tol=amount_tol)
                if R2.converged and (not R.converged or R2.G < R.G - 1e-12):
                    R = R2
        record = IterationRecord(it, R, [])
        history.append(record)
        if not R.converged:
            break

        # An interior-point solve leaves a vanishing phase with a small positive amount, not an exact
        # zero. A set counts as inactive if its amount is negligible, or if it is small and sits
        # clearly above the tangent plane. If any set is dropped, R is solved again before pricing,
        # so the reported μ and G never include a ghost phase.
        def inactive(s, mu=R.mu):
            if s.amount <= amount_tol:
                return True
            return s.amount < small and float(tangent_distance(s.phase, s.y, mu, NUM, T)) > eps

        active = [s for s in R.sets if not inactive(s)] or [max(R.sets, key=lambda s: s.amount)]
        # Two sets of one phase can converge onto the same composition. Then they are one
        # set whose amount is split between them, and the output would list it twice.
        active = _merge_coincident(active, dedup_tol)
        record.removed = [s for s in R.sets if all(s is not a for a in active)]
        if record.removed:
            sets, new = active, []
            continue
        if R.mu_directions is not None:
            # Lone compounds that do not span every element leave μ free along mu_directions. Take
            # the μ there that keeps every other phase farthest above its tangent: the middle of the
            # compound's stability range. Any such μ gives a valid certificate; this one avoids
            # destabilizing a phase that a better μ would show to be stable.
            R.mu = _center_mu(R, [p for p in phases if not _is_active_compound(p, active)], T, local_kw)
            R.residual = abs(R.G - float(R.mu @ b))
        record.pricing = price_phases(phases, R.mu, **(global_kw if certify == "every" else local_kw))
        _generate(record, active, eps, dedup_tol)
        if not record.added and certify == "final":
            record.local_pricing, record.rejected = record.pricing, []
            record.pricing = price_phases(phases, R.mu, **global_kw)
            _generate(record, active, eps, dedup_tol)
        if not record.added:
            break
        sets, new = active + record.added, record.added
    else:
        exhausted = True

    last = history[-1]
    cert = certify_state(last.restricted, last.pricing, b, eps, T=T)
    if exhausted:
        cert.messages.append(f"max_iter={max_iter} reached before pricing found no new set")
    if last.rejected:
        cert.messages.append("destabilizing point rejected as a duplicate: R did not reach that tangency")
    return EquilibriumResult(
        phases=phases,
        b=b,
        T=T,
        composition_sets=[s for s in last.restricted.sets if s.amount > amount_tol],
        mu=last.restricted.mu,
        G=last.restricted.G,
        certificate=cert,
        history=history,
    )
