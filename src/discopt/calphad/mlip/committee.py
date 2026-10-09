"""A committee of calculators: one dataset and one fit per member.

Machine-learned potentials are often trained as committees or ensembles
(several networks from different seeds or data splits), and their
disagreement estimates the model's own error. Here each member produces its
own data through :func:`~discopt.calphad.mlip.compute`, and each dataset is
fitted with the same free parameters. The spread of the fitted coefficients
across members, and of the phase boundaries they give
(:func:`discopt.calphad.uncertainty.boundary_spread` on
:meth:`Committee.databases`), is the committee uncertainty carried through
to the diagram.

This differs from ``from_ase(..., uncertainty=)``, which turns a committee
spread into the σ of each data point for a single fit. That treats the
members' errors as independent from point to point; a committee fit keeps
each member's errors correlated across its own data, as they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Committee", "committee"]


@dataclass
class Committee:
    """The members' data and fits."""

    members: list[str]
    #: one MLIPResult and one FitResult per member
    results: list = field(default_factory=list)
    fits: list = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return list(self.fits[0].names)

    @property
    def values(self) -> np.ndarray:
        """Fitted coefficients, members × coefficients."""
        return np.array([f.values for f in self.fits], float)

    @property
    def mean(self) -> np.ndarray:
        return self.values.mean(axis=0)

    @property
    def std(self) -> np.ndarray:
        """Spread of each coefficient across members (ddof=1)."""
        return self.values.std(axis=0, ddof=1)

    @property
    def covariance(self) -> np.ndarray:
        """Covariance of the coefficients across members."""
        return np.atleast_2d(np.cov(self.values, rowvar=False, ddof=1))

    def databases(self, samples_per_member: int = 0, seed: int = 0) -> list:
        """One database per member, or ``samples_per_member`` posterior draws from each fit."""
        from discopt.calphad.uncertainty import posterior_databases

        if samples_per_member <= 0:
            return [f.database for f in self.fits]
        out = []
        for k, f in enumerate(self.fits):
            out += posterior_databases(f, samples_per_member, seed=seed + k)
        return out

    def summary(self) -> str:
        lines = [f"committee of {len(self.members)}: {', '.join(self.members)}"]
        width = max(len(n) for n in self.names)
        within = np.sqrt(np.mean([np.asarray(f.std, float) ** 2 for f in self.fits], axis=0))
        lines.append(f"{'coefficient':{width}s} {'mean':>14s} {'between':>12s} {'within':>12s}")
        for n, m, s, w in zip(self.names, self.mean, self.std, within):
            lines.append(f"{n:{width}s} {m:14.6g} {s:12.4g} {w:12.4g}")
        return "\n".join(lines)


def committee(models, db, elements, free, *, fit_options=None, **compute_kwargs) -> Committee:
    """Data from each model in ``models`` (:func:`~discopt.calphad.mlip.compute` with ``compute_kwargs``), each fitted.

    ``free`` and ``fit_options`` (a dict of keyword arguments) go to
    :func:`discopt.calphad.fit.fit`, the same for every member.
    """
    from discopt.calphad.fit import fit
    from discopt.calphad.mlip.data import compute

    models = list(models)
    if len(models) < 2:
        raise ValueError("a committee needs at least two models")
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"committee members need distinct names (they label the data and the cache): {names}")
    out = Committee(names)
    for m in models:
        res = compute(m, db, elements, **compute_kwargs)
        out.results.append(res)
        out.fits.append(fit(db, res.dataset, elements, free, **(fit_options or {})))
    return out
