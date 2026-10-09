"""Which measurement next? Design of experiments for a CALPHAD fit (issue #10).

Every data kind the fit uses gives rows that are linear in the coefficients θ,
or exactly linearized at the solution (driving forces and multi-sublattice μ).
A candidate measurement c, with σ-weighted rows J_c, therefore adds J_cᵀJ_c to
the Fisher information F = JᵀJ of the fit. The covariance after a set of
candidates is known before measuring anything:

    C_new = (F + Σ_c J_cᵀ J_c)⁻¹

This holds exactly for linear rows, and to first order for linearized ones.
The inverse carries a tiny ridge (1e-12 of the largest diagonal entry of the
scaled information), so a coefficient the data do not determine yet shows a
huge but finite variance, and a candidate that determines it ranks first.

- :func:`rank_candidates` scores each candidate on its own: the reduction in
  the variance of chosen coefficients (all by default), absolute and
  relative.
- :func:`design_measurements` chooses a batch of ``n``. It greedily adds the
  candidate that most improves the A-criterion (summed variance of the chosen
  coefficients) or the D-criterion (log det of the information), with exact
  multi-row updates. With ``method="exchange"`` it uses ``discopt.doe``'s
  modified Fedorov exchange instead. That needs discopt-doe, installed
  separately (it requires discopt ≥ 0.9 and Python ≥ 3.12), and candidates with
  a single row each.

A candidate is an :class:`~discopt.calphad.fit.Observation` (or a
:class:`~discopt.calphad.fit.Dataset` of them). Its ``value`` is not used,
but its ``sigma`` is: it is the measurement's expected standard uncertainty.
"""

from __future__ import annotations

import numpy as np

from discopt.calphad.fit import Dataset, FitResult, Observation, _LinearModel, _rows

__all__ = ["candidate_rows", "rank_candidates", "design_measurements"]


def _model(result: FitResult) -> _LinearModel:
    lm = _LinearModel(result.database, result.elements, result.free)
    lm.theta = np.asarray(result.values, float)
    return lm


def _list(candidates) -> list[Observation]:
    return list(candidates.observations if isinstance(candidates, Dataset) else candidates)


def candidate_rows(result: FitResult, candidates) -> list[np.ndarray]:
    """σ-weighted rows J_c (rows × coefficients) of each candidate, at the fitted θ."""
    lm = _model(result)
    out = []
    for c in _list(candidates):
        V, _, sig = _rows(lm, c)
        out.append(np.array(V, float)[:, 1:] / np.array(sig, float)[:, None])
    return out


def _indices(result: FitResult, parameters) -> list[int]:
    if parameters is None:
        return list(range(len(result.names)))
    missing = [p for p in parameters if p not in result.names]
    if missing:
        raise KeyError(f"unknown coefficients {missing}; coefficients are {result.names}")
    return [result.names.index(p) for p in parameters]


def _information(result: FitResult) -> np.ndarray:
    """F = JᵀJ of the fit, recovered from its covariance (the covariance is pinv(F), and pinv is an involution)."""
    return np.linalg.pinv(np.asarray(result.covariance, float))


def _covariance(F: np.ndarray) -> np.ndarray:
    """F⁻¹, scaled to unit diagonal and with a 1e-12 ridge, so an undetermined combination is huge but finite.

    A coefficient with no information at all (a zero row of F) gets variance ∞.
    """
    n = len(F)
    diag = np.diag(F)
    seen = diag > 1e-300
    C = np.full((n, n), np.nan)
    C[np.ix_(~seen, ~seen)] = 0.0
    C[~seen, ~seen] = np.inf
    if seen.any():
        Fs = F[np.ix_(seen, seen)]
        d = np.sqrt(np.diag(Fs))
        S = Fs / np.outer(d, d)
        C[np.ix_(seen, seen)] = np.linalg.inv(S + 1e-12 * np.eye(len(d))) / np.outer(d, d)
    return C


def _relative(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Fractional variance reduction; from ∞ to a finite variance counts as 1."""
    out = np.zeros_like(before)
    finite = np.isfinite(before) & (before > 0)
    out[finite] = (before[finite] - after[finite]) / before[finite]
    out[~np.isfinite(before) & np.isfinite(after)] = 1.0
    return out


def _score(F: np.ndarray, idx, criterion: str) -> tuple[int, float]:
    """(undetermined coefficients, criterion value); smaller is better in both."""
    C = _covariance(F)
    var = np.diag(C)[idx]
    missing = int(np.sum(~np.isfinite(var)))
    if criterion == "A":
        return missing, float(np.sum(var[np.isfinite(var)]))
    seen = np.isfinite(np.diag(C))
    return missing, -float(np.linalg.slogdet(F[np.ix_(seen, seen)] + 1e-12 * np.eye(int(seen.sum())))[1])


def rank_candidates(result: FitResult, candidates, parameters=None) -> list[dict]:
    """Each candidate's expected effect on the fitted coefficients, best first.

    For each candidate: ``variance_before`` and ``variance_after`` of the chosen
    coefficients (``parameters``: names as in :attr:`FitResult.names`, default
    all), the summed ``reduction`` over finite variances, and ``relative`` (the
    mean fractional reduction over the chosen coefficients, where going from an
    undetermined coefficient, variance ∞, to a determined one counts as 1).
    Sorted by ``relative``.
    """
    idx = _indices(result, parameters)
    F = _information(result)
    before = np.diag(_covariance(F))[idx]
    out = []
    for k, (c, J) in enumerate(zip(_list(candidates), candidate_rows(result, candidates))):
        after = np.diag(_covariance(F + J.T @ J))[idx]
        rel = _relative(before, after)
        finite = np.isfinite(before) & np.isfinite(after)
        out.append(
            {
                "index": k,
                "observation": c,
                "variance_before": before,
                "variance_after": after,
                "reduction": float(np.sum((before - after)[finite])),
                "relative": float(np.mean(rel)),
            }
        )
    return sorted(out, key=lambda r: -r["relative"])


def design_measurements(
    result: FitResult,
    candidates,
    n: int,
    *,
    criterion: str = "A",
    parameters=None,
    method: str = "greedy",
    allow_repeats: bool = False,
    seed: int = 0,
) -> dict:
    """Choose ``n`` measurements from ``candidates`` that together reduce the uncertainty most.

    ``criterion`` is ``"A"`` (minimize the summed variance of ``parameters``)
    or ``"D"`` (maximize log det of the information). Either way, determining a
    coefficient the data leave undetermined comes first. ``method="greedy"`` adds
    one candidate at a time with exact multi-row updates. ``method="exchange"``
    hands the rows to ``discopt.doe.candidate_exchange_design``, with the fit's
    information as ``prior_fim``. Returns the chosen indices and observations,
    and the variances of the chosen coefficients before and after.
    """
    cands = _list(candidates)
    rows = candidate_rows(result, cands)
    idx = _indices(result, parameters)
    F0 = _information(result)
    if criterion not in ("A", "D"):
        raise ValueError(f"criterion must be 'A' or 'D', got {criterion!r}")
    if method == "exchange":
        if any(J.shape[0] != 1 for J in rows):
            raise ValueError("method='exchange' needs candidates with one row each; use method='greedy'")
        try:
            from discopt.doe import candidate_exchange_design
        except ImportError as err:
            raise ImportError(
                "method='exchange' needs discopt-doe (pip install discopt-doe; it requires discopt >= 0.9 "
                "and Python >= 3.12)"
            ) from err
        prior = F0
        res = candidate_exchange_design(
            list(range(len(cands))),
            n,
            model_rows=np.vstack(rows),
            criterion=criterion,
            prior_fim=prior,
            allow_repeats=allow_repeats,
            seed=seed,
        )
        chosen = [int(k) for k in res.indices]
    elif method == "greedy":
        chosen, F = [], F0.copy()
        for _ in range(n):
            best, best_score = None, None
            for k, J in enumerate(rows):
                if not allow_repeats and k in chosen:
                    continue
                score = _score(F + J.T @ J, idx, criterion)
                if best_score is None or score < best_score:
                    best, best_score = k, score
            if best is None:
                break
            chosen.append(best)
            F = F + rows[best].T @ rows[best]
    else:
        raise ValueError(f"method must be 'greedy' or 'exchange', got {method!r}")
    F = F0.copy()
    for k in chosen:
        F = F + rows[k].T @ rows[k]
    return {
        "chosen": chosen,
        "observations": [cands[k] for k in chosen],
        "variance_before": np.diag(_covariance(F0))[idx],
        "variance_after": np.diag(_covariance(F))[idx],
    }
