"""Which structures to compute: design and active learning for calculator data.

Relaxations, SQS and especially phonons and MD cost real time, and not every
structure tells the fit as much as the next. A candidate value is known
before it is computed in everything except its value: its phase, its site
fractions, and the σ the calculator will give it (the model's error). That
is all the information matrix of a fit needs (:mod:`discopt.calphad.design`),
so the most informative structures can be chosen first.

- :func:`candidates` builds the candidate observations for given compounds
  and mixing grids, by running :func:`~discopt.calphad.mlip.compute` with a
  stand-in calculator whose energy is zero everywhere. The observations are
  exactly those a real run would give, with ``value`` 0 and σ ``sigma``.
- :func:`compute_arguments` turns chosen candidates back into the
  ``compounds=`` and ``mixing=`` arguments of :func:`compute`.
- :func:`suggest` ranks candidates against an existing fit and returns the
  next batch, ready for :func:`compute`.
- :func:`active_learning` loops: compute a batch, fit, choose the next batch,
  until a budget or a target precision is reached. ``on_batch`` is called
  after every batch with the model, the structures just computed and the
  result. It may return a new model, which is the hook for fine-tuning a
  potential on reference (DFT) values of the structures it chose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["candidates", "compute_arguments", "suggest", "active_learning", "ActiveLearningResult"]


def _zero_model():
    from ase.calculators.calculator import Calculator, all_changes

    from discopt.calphad.mlip.calculator import from_ase

    class Zero(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            n = len(self.atoms)
            self.results = {"energy": 0.0, "free_energy": 0.0, "forces": np.zeros((n, 3)), "stress": np.zeros(6)}

    return from_ase(Zero(), name="candidate")


def candidates(db, elements, *, compounds=(), mixing=None, structures=None, sigma: float = 2.0, T: float = 298.15):
    """Candidate observations (value 0, σ ``sigma`` kJ/mol) for the compounds and mixing grids given.

    ``compounds``, ``mixing`` and ``structures`` are as for
    :func:`~discopt.calphad.mlip.compute`. ``sigma`` is the expected σ of a
    computed value, a number or a dict by phase (typically the calculator's
    error, as ``model_sigma``). Mixing candidates are random-decoration
    observations; their rows are the same for every method.
    """
    from discopt.calphad.mlip.calculator import RelaxPolicy
    from discopt.calphad.mlip.data import compute
    from discopt.calphad.mlip.random_alloy import RandomOptions

    res = compute(
        _zero_model(),
        db,
        elements,
        compounds=compounds,
        mixing=mixing,
        structures=structures,
        method="random",
        random=RandomOptions(count=1, min_atoms=1, max_atoms=256),
        relax=RelaxPolicy(mode="fixed"),
        sigma=sigma,
        T=T,
    )
    for o in res.dataset:
        o.source = "candidate"
    return res.dataset


def _spec(o, elements):
    """("compound", phase, endmember) or ("mixing", phase, y) for a candidate or computed observation."""
    if o.kind == "HM_FORM":
        if o.site_fractions is None:
            raise ValueError(f"{o.phase}: a formation candidate needs site fractions")
        em = ":".join(max(d, key=d.get) for d in o.site_fractions)
        return ("compound", o.phase, em)
    if o.kind == "HM_MIX":
        mixed = [d for d in o.site_fractions if sum(v > 0 for v in d.values()) > 1]
        y = mixed[0][elements[0]] if mixed else o.site_fractions[0].get(elements[0], 0.0)
        return ("mixing", o.phase, round(float(y), 10))
    raise ValueError(f"no compute() argument makes a {o.kind} observation; candidates are HM_FORM and HM_MIX")


def compute_arguments(observations, elements) -> dict:
    """``{"compounds": {phase: [endmember, ...]}, "mixing": {phase: [y, ...]}}`` for chosen candidates."""
    elements = tuple(e.upper() for e in elements)
    compounds, mixing = {}, {}
    for o in observations:
        what, phase, v = _spec(o, elements)
        target = compounds if what == "compound" else mixing
        if v not in target.setdefault(phase, []):
            target[phase].append(v)
    out = {}
    if compounds:
        out["compounds"] = compounds
    if mixing:
        out["mixing"] = {p: sorted(v) for p, v in mixing.items()}
    return out


def _done(dataset, elements) -> set:
    out = set()
    for o in dataset:
        try:
            out.add(_spec(o, elements))
        except ValueError:
            continue
    return out


def suggest(result, cands, n: int, *, criterion: str = "A", parameters=None, exclude=None) -> dict:
    """The ``n`` candidates that most reduce the uncertainty of ``result``, and the ``compute()`` arguments for them.

    ``cands`` is a :class:`~discopt.calphad.fit.Dataset` from :func:`candidates`.
    Candidates already in ``exclude`` (a dataset of computed observations,
    default ``result.data``) are skipped.
    """
    from discopt.calphad.design import design_measurements
    from discopt.calphad.fit import Dataset

    elements = tuple(result.elements)
    done = _done(result.data if exclude is None else exclude, elements)
    pool = [o for o in cands if _spec(o, elements) not in done]
    if not pool:
        return {"observations": [], "arguments": {}, "variance_before": None, "variance_after": None}
    d = design_measurements(result, Dataset(pool), min(n, len(pool)), criterion=criterion, parameters=parameters)
    d["arguments"] = compute_arguments(d["observations"], elements)
    return d


@dataclass
class ActiveLearningResult:
    """The rounds of an active-learning run, and the final fit."""

    #: per round: the compute() arguments used, the fit after it, the summed variance of the coefficients
    rounds: list = field(default_factory=list)
    fit: object = None
    dataset: object = None
    model: object = None
    stopped: str = ""

    def table(self) -> str:
        lines = [f"{'round':>5s} {'computed':>9s} {'max rel. std':>13s}  batch"]
        for k, r in enumerate(self.rounds):
            lines.append(f"{k:5d} {r['computed']:9d} {r['max_relative_std']:13.3g}  {r['arguments']}")
        lines.append(f"stopped: {self.stopped}")
        return "\n".join(lines)


def _max_relative_std(fr) -> float:
    if fr.rank < len(fr.names):  # some combination is undetermined: its σ is not finite
        return float("inf")
    v, s = np.abs(np.asarray(fr.values, float)), np.asarray(fr.std, float)
    scale = np.where(v > 0, v, 1.0)
    return float(np.max(s / scale))


def active_learning(
    model,
    db,
    elements,
    free,
    *,
    compounds=(),
    mixing=None,
    structures=None,
    expected_sigma: float = 2.0,
    batch: int = 2,
    budget: int = 20,
    target: float | None = None,
    initial=None,
    criterion: str = "A",
    on_batch=None,
    fit_options=None,
    **compute_kwargs,
) -> ActiveLearningResult:
    """Compute, fit and choose the next structures, until ``budget`` values are computed or ``target`` is met.

    The candidates are the compounds and mixing grids given (as for
    :func:`~discopt.calphad.mlip.compute`), with expected σ
    ``expected_sigma`` (kJ/mol) for the design. The first batch is
    ``initial`` (``compute()`` arguments), or chosen by
    :func:`~discopt.calphad.design.initial_design` to determine every free
    coefficient. Each later batch is the ``batch`` candidates that most reduce
    the summed variance (``criterion="A"``) or the log-determinant
    (``"D"``). It stops when the largest relative standard deviation of the
    coefficients is below ``target``, when ``budget`` values have been
    computed, or when no candidates are left.

    ``on_batch(model, structures, result)`` is called after each batch with
    the relaxed structures of that batch (``ase.Atoms``) and the
    :class:`~discopt.calphad.mlip.MLIPResult`. If it returns a model, that
    model computes the next batches: the hook for fine-tuning a potential, or
    for swapping in a reference calculator. ``fit_options`` go to
    :func:`~discopt.calphad.fit.fit`, ``compute_kwargs`` to ``compute()``.
    """
    from discopt.calphad.design import initial_design
    from discopt.calphad.fit import Dataset, fit
    from discopt.calphad.mlip.data import compute

    elements = tuple(e.upper() for e in elements)
    cands = candidates(db, elements, compounds=compounds, mixing=mixing, structures=structures, sigma=expected_sigma)
    if initial is None:
        d = initial_design(db, elements, free, cands, criterion="D")
        if d["undetermined"]:
            raise ValueError(
                f"the candidates leave {d['undetermined']} free coefficients undetermined; add compounds or "
                "mixing compositions, or fix those coefficients"
            )
        initial = compute_arguments(d["observations"], elements)
    out = ActiveLearningResult(model=model)
    data = Dataset()
    args = initial
    while True:
        res = compute(out.model, db, elements, structures=structures, **args, **compute_kwargs)
        for o in res.dataset:
            data.add(o)
        fr = fit(db, data, elements, free, **(fit_options or {}))
        rel = _max_relative_std(fr)
        out.rounds.append({"arguments": args, "fit": fr, "computed": len(data), "max_relative_std": rel, "result": res})
        out.fit, out.dataset = fr, data
        if on_batch is not None:
            new = on_batch(out.model, [r.atoms() for r in res.relaxations], res)
            if new is not None:
                out.model = new
        if target is not None and rel <= target:
            out.stopped = f"target: largest relative std {rel:.3g} <= {target:g}"
            break
        if len(data) >= budget:
            out.stopped = f"budget: {len(data)} values computed"
            break
        nxt = suggest(fr, cands, min(batch, budget - len(data)), criterion=criterion)
        if not nxt["observations"]:
            out.stopped = "no candidates left"
            break
        args = nxt["arguments"]
    return out
