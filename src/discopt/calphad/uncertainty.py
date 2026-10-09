"""Uncertainty of a fitted model, carried through to its phase diagram.

A fit gives coefficients θ with a covariance, but what matters is how far the
phase boundaries can move. Here an ensemble of databases is mapped point by
point, and the spread of each boundary is reported. The ensemble can come
from two places, and they can be combined:

- **the fit's posterior** (:func:`posterior_databases`): θ drawn from
  N(θ̂, C), the fit's covariance. This carries the σ of the data, so it is as
  good as those σ;
- **a committee** (:func:`discopt.calphad.mlip.committee`): one fit per member
  of a committee or ensemble of calculators, each to its own data. Their
  disagreement is the calculator's own (epistemic) uncertainty, which no σ
  from a single model sees. With ``samples_per_member``, each member adds
  posterior draws too, so the spread is within-member plus between-member.

Each database's boundaries are tie-line ends at fixed T: for every
two-phase region, the median over the grid points inside it of the two
coexisting compositions. A region that a database does not have is ``nan``
for it, and :attr:`BoundarySpread.present` counts the databases that have
it. The equilibria are computed without the global certificate by default
(``certify="none"``), because an ensemble needs many of them; pass
``certify="final"`` to certify each.

Every boundary is a statement about the fitted models. How well they
represent the real alloy is what the spread of the committee estimates.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from discopt.calphad.database import ThermoDatabase
from discopt.calphad.fit import FitResult, _LinearModel, _replace

__all__ = ["database_at", "posterior_databases", "phase_boundaries", "boundary_spread", "BoundarySpread"]


def database_at(result: FitResult, theta) -> ThermoDatabase:
    """The fitted database with its free coefficients set to ``theta`` (ordered as :attr:`FitResult.names`)."""
    lm = _LinearModel(result.database, result.elements, result.free)
    theta = np.asarray(theta, float)
    if theta.shape != (len(result.names),):
        raise ValueError(f"theta needs {len(result.names)} values ({result.names}), got shape {theta.shape}")
    return _replace(result.database, lm.functions_at(theta))


def posterior_databases(result: FitResult, n: int = 20, seed: int = 0) -> list[ThermoDatabase]:
    """``n`` databases with θ drawn from N(θ̂, C), C the fit's covariance.

    A direction the data do not determine (infinite or undefined variance)
    raises: the posterior is then not a distribution.
    """
    C = np.asarray(result.covariance, float)
    if not np.all(np.isfinite(C)):
        raise ValueError("the fit's covariance is not finite: some coefficient is not determined by the data")
    rng = np.random.default_rng(seed)
    thetas = rng.multivariate_normal(np.asarray(result.values, float), C, size=n, method="eigh")
    return [database_at(result, t) for t in thetas]


def phase_boundaries(
    db: ThermoDatabase, elements, T: float, xs, *, phases=None, certify: str = "none", **equilibrium_kwargs
) -> dict[tuple[str, ...], np.ndarray]:
    """Tie-line ends of each two-phase region at ``T``, from equilibria at each x (of ``elements[0]``) in ``xs``.

    Returns {(phase, phase): [x in the first, x in the second]}, the phases
    sorted by name, each end the median over the grid points in the region.
    ``phases`` restricts the phases considered (names); default all.
    """
    from discopt.calphad import equilibrium
    from discopt.calphad.equilibrium.certificate import CertificateLevel

    models = [p for name, p in db.phase_models(tuple(elements)).items() if phases is None or name in phases]
    ends = defaultdict(list)
    for x in np.atleast_1d(np.asarray(xs, float)):
        r = equilibrium(models, x=float(x), T=float(T), certify=certify, **equilibrium_kwargs)
        if r.certificate.level < CertificateLevel.STABILITY_SEARCHED:
            continue
        sets = sorted((cs.phase.name, float(cs.x[0])) for cs in r.composition_sets)
        if len(sets) == 2 and sets[0][0] != sets[1][0]:
            ends[(sets[0][0], sets[1][0])].append([sets[0][1], sets[1][1]])
        elif len(sets) == 2:  # a miscibility gap: one phase twice, ends by composition
            ends[(sets[0][0], sets[1][0])].append(sorted([sets[0][1], sets[1][1]]))
    return {k: np.median(np.array(v), axis=0) for k, v in ends.items()}


@dataclass
class BoundarySpread:
    """Phase boundaries of an ensemble of databases."""

    #: (T, (phase, phase)) -> array (databases × 2) of tie-line ends, nan where a database lacks the region
    ends: dict = field(default_factory=dict)
    n: int = 0

    def present(self, key) -> int:
        return int(np.sum(np.all(np.isfinite(self.ends[key]), axis=1)))

    def mean(self, key) -> np.ndarray:
        return np.nanmean(self.ends[key], axis=0)

    def std(self, key) -> np.ndarray:
        """Standard deviation of each end over the databases that have the region (ddof=1)."""
        e = self.ends[key]
        e = e[np.all(np.isfinite(e), axis=1)]
        return np.std(e, axis=0, ddof=1) if len(e) > 1 else np.full(2, np.nan)

    def table(self) -> str:
        lines = [f"{'T (K)':>7s}  {'region':24s} {'present':>8s}  {'x1 mean ± std':>18s}  {'x2 mean ± std':>18s}"]
        for key in sorted(self.ends):
            T, pair = key
            m, s = self.mean(key), self.std(key)
            lines.append(
                f"{T:7.1f}  {' + '.join(pair):24s} {self.present(key):>3d}/{self.n:<4d}"
                f"  {m[0]:8.4f} ± {s[0]:<7.4f}  {m[1]:8.4f} ± {s[1]:<7.4f}"
            )
        return "\n".join(lines)


def boundary_spread(databases, elements, temperatures, xs, **kwargs) -> BoundarySpread:
    """:func:`phase_boundaries` of every database at every temperature, gathered by region."""
    dbs = list(databases)
    per = []
    for db in dbs:
        per.append(
            {
                (float(T), k): v
                for T in np.atleast_1d(temperatures)
                for k, v in phase_boundaries(db, elements, T, xs, **kwargs).items()
            }
        )
    keys = sorted(set().union(*per))
    ends = {k: np.array([p.get(k, [np.nan, np.nan]) for p in per], float) for k in keys}
    return BoundarySpread(ends, len(dbs))
