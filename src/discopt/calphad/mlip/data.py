"""Turn relaxed energies into a fitting :class:`~discopt.calphad.fit.Dataset`.

Two kinds of data come out:

- **formation enthalpies** (``HM_FORM``) of compounds and CEF endmembers,
  relative to each element in its reference (SER) structure, computed with
  the same model so that systematic errors largely cancel;
- **mixing enthalpies** (``HM_MIX``) of a solution phase on its own lattice,
  relative to the phase's pure endmembers, from special quasirandom
  structures (``method="sqs"``), averaged over random decorations
  (``method="random"``), or predicted for the random alloy by a cluster
  expansion that icet fits to enumerated structures
  (``method="cluster_expansion"``).

Every value is a 0 K energy standing in for an enthalpy at 298.15 K, and each
observation records that approximation in its ``note``. The pure elements are
never fitted: their SGTE descriptions stay as they are.

**σ.** With ``sigma="spread"`` (the default), σ² = σ_config² + σ_model².

- **σ_config** is the spread of the SQS behind a mixing enthalpy. Two SQS at
  one composition give a standard deviation with a single degree of freedom,
  which can be off by an order of magnitude either way. By default
  (``pool=True``) the variances of every composition of a phase are pooled,
  Σ(n_i − 1)s_i² / Σ(n_i − 1), so nine compositions with two SQS each give
  one σ_config with nine degrees of freedom. ``pool=False`` keeps one σ per
  composition. With random decorations it is the standard error of their
  mean, and with a cluster expansion the cross-validation RMSE of the
  expansion. A formation enthalpy comes from one structure and has no
  σ_config.
- **σ_model** is the calculator's own error, which no spread can see. It is
  ``model_sigma=`` (kJ/mol, a number or a dict by phase) when given, and
  otherwise the model's ensemble or committee spread (``from_ase(...,
  uncertainty=)``) if it has one. No first-principles value exists otherwise:
  take it from a benchmark, or from a comparison with DFT or experiment.

A value with neither part raises and asks for ``model_sigma=`` or ``sigma=``,
rather than inventing a number. A mixing enthalpy with σ_config but no
σ_model is used with a warning that its σ covers only configurational spread.
``sigma=`` (a number in kJ/mol, or a dict by phase) sets σ directly and
replaces both parts. Every row of :meth:`MLIPResult.table` reports the degrees
of freedom of its σ_config. After a fit, χ²/dof well above 1 means the σ are
too small for the scatter in the data.

**What the certificate covers.** A fit to these data, and the certified
equilibria computed from it, are exact statements about the *fitted model*.
They say nothing about how good the calculator is: its error reaches the
result only through σ.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from discopt.calphad.fit import Dataset
from discopt.calphad.mlip.calculator import EV_TO_J_PER_MOL, Relaxation, RelaxPolicy
from discopt.calphad.mlip.prototypes import PhaseStructure, atomic_volume, phase_structure

__all__ = ["APPROXIMATION", "MLIPResult", "compute", "dataset", "reference_energies"]

#: recorded on every observation made here
APPROXIMATION = "0 K energy used as the enthalpy at 298.15 K"
#: recorded instead when phonons are computed
#: recorded on cluster-expansion data
CE_NOTE = APPROXIMATION + "; random alloy from a cluster expansion"
MD_NOTE = "classical MD at temperature (enthalpy), alchemical thermodynamic integration (free energy)"
MC_NOTE = "canonical Monte Carlo on a cluster expansion of 0 K energies, short-range order included"
PHONON_NOTE = "static energy plus {kind} vibrational enthalpy (zero point and thermal) at 298.15 K"


@dataclass
class MLIPResult:
    """The dataset, and every relaxation and computed value behind it."""

    dataset: Dataset
    #: one row per value: kind, phase, configuration, x, value and sigma in kJ/mol, structures used,
    #: and the degrees of freedom of the configurational part of sigma (None without one)
    rows: list[dict] = field(default_factory=list)
    #: element -> relaxation of its reference structure
    references: dict[str, Relaxation] = field(default_factory=dict)
    relaxations: list[Relaxation] = field(default_factory=list)
    #: (label, relaxation) for every structure that left its prototype
    flagged: list[tuple[str, Relaxation]] = field(default_factory=list)
    #: phase -> :class:`~discopt.calphad.mlip.cluster_expansion.ClusterExpansionFit` (method="cluster_expansion")
    cluster_expansions: dict = field(default_factory=dict)
    #: phase -> list of :class:`~discopt.calphad.mlip.monte_carlo.SROResult` (``monte_carlo=``)
    short_range_order: dict = field(default_factory=dict)
    #: phase -> list of :class:`~discopt.calphad.mlip.liquid.LiquidResult` (``liquid=``)
    liquids: dict = field(default_factory=dict)

    def table(self) -> str:
        lines = [
            f"{'kind':8s} {'phase':14s} {'config':12s} {'x':>6s} {'value':>10s} {'sigma':>8s} {'dof':>4s}  (kJ/mol)"
        ]
        for r in self.rows:
            x = "" if r["x"] is None else f"{r['x']:.3f}"
            dof = "" if r.get("sigma_dof") is None else str(r["sigma_dof"])
            lines.append(
                f"{r['kind']:8s} {r['phase']:14s} {r['config']:12s} {x:>6s} {r['value']:10.3f} {r['sigma']:8.3f}"
                f" {dof:>4s}"
            )
        for label, rel in self.flagged:
            lines.append(f"flagged {label}: {rel.reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------- helpers


def _structure_for(name: str, structures) -> PhaseStructure:
    spec = (structures or {}).get(name)
    if spec is None:
        return phase_structure(name)
    if isinstance(spec, PhaseStructure):
        return spec
    if hasattr(spec[0], "get_positions"):  # (ase.Atoms, atoms of each sublattice)
        return PhaseStructure.from_atoms(spec[0], spec[1], name=name)
    return phase_structure(*spec)


def _constituents(db, name: str, elements) -> list[tuple[str, ...]]:
    allowed = set(elements) | {"VA"}
    return [tuple(sorted(c for c in s if c in allowed)) for s in db.phases[name].constituents]


def _relax(model, structures, policy: RelaxPolicy) -> list[Relaxation]:
    if hasattr(model, "relax"):
        return model.relax(structures, policy)
    warnings.warn(f"{model.name} has no relax(): the prototype check is skipped", stacklevel=3)
    out = []
    for a, e in zip(structures, model.energies(structures, policy)):
        out.append(
            Relaxation(model.name, a.get_chemical_formula(), len(a), float(e) * len(a), True, 0, 0.0, 0.0, 0.0, "", "")
        )
    return out


def _vegard(atoms) -> float:
    return float(np.mean([atomic_volume(s) for s in atoms.get_chemical_symbols()]))


def _model_sigma(model, structures) -> float | None:
    if not getattr(model, "has_uncertainty", hasattr(model, "uncertainty")):
        return None
    try:
        return float(np.mean(model.uncertainty(structures)))
    except NotImplementedError:
        return None


def _kj(value, phase: str, what: str) -> float | None:
    """A number or a dict-by-phase entry in kJ/mol, as J/mol; None when not given (or ``"spread"``)."""
    if isinstance(value, dict):
        value = value.get(phase)
    if value is None or isinstance(value, str):
        if isinstance(value, str) and value != "spread":
            raise ValueError(f"{what} must be 'spread', a number in kJ/mol, or a dict of those by phase; got {value!r}")
        return None
    if value <= 0:
        raise ValueError(f"{what} must be positive")
    return float(value) * 1000.0


def _sigma(phase: str, label: str, config: float | None, model: float | None, single: bool) -> float:
    """σ in J/mol from its configurational and model parts, both in J/mol (see the module docstring)."""
    parts = [p for p in (config, model) if p]
    if not parts:
        why = "comes from a single structure" if single else "has zero spread over its structures"
        raise ValueError(
            f"{label}: the value {why} and the model has no uncertainty estimate; pass model_sigma= or sigma= "
            f"(kJ/mol), for example model_sigma={{'{phase}': 2.0}}"
        )
    return math.sqrt(sum(p * p for p in parts))


def _check(label: str, rel: Relaxation, result: MLIPResult, keep_flagged: bool) -> bool:
    result.relaxations.append(rel)
    if rel.left_prototype:
        result.flagged.append((label, rel))
        warnings.warn(
            f"{label} left its prototype ({rel.reason}); {'kept' if keep_flagged else 'not used'}", stacklevel=3
        )
        return keep_flagged
    return True


# ---------------------------------------------------------------------------- references


def reference_energies(model, db, elements, *, relax=None, structures=None, result=None) -> dict[str, Relaxation]:
    """Each element relaxed in the structure of its reference phase (the database's ELEMENT line)."""
    policy = relax or RelaxPolicy()
    out = {}
    for e in elements:
        ref = db.elements[e].reference_phase if e in db.elements else ""
        if not ref:
            raise ValueError(f"{e} has no reference phase in the database")
        ps = _structure_for(ref, structures)
        cons = db.phases[ref].constituents if ref in db.phases else None
        endmember = []
        for s, sub in enumerate(ps.sublattices):
            if sub is None:
                endmember.append("VA")
            elif cons is None or e in cons[s]:
                endmember.append(e)
            else:
                raise ValueError(f"{ref}: sublattice {s} cannot hold {e}")
        atoms = ps.build(endmember, atomic_volume(e))
        rel = _relax(model, [atoms], policy)[0]
        if rel.left_prototype:
            raise ValueError(f"the reference structure of {e} ({ref}) left its prototype: {rel.reason}")
        out[e] = rel
        if result is not None:
            result.relaxations.append(rel)
    return out


# ---------------------------------------------------------------------------- formation


def _endmembers(db, name, elements, given):
    cons = _constituents(db, name, elements)
    if given is not None:
        out = []
        for em in given:
            parts = tuple(p.strip().upper() for p in (em.split(":") if isinstance(em, str) else em))
            if len(parts) != len(cons) or any(c not in s for c, s in zip(parts, cons)):
                raise ValueError(f"{name}: endmember {':'.join(parts)} does not fit {cons}")
            out.append(parts)
        return out
    out = [em for em in product(*cons) if len({c for c in em if c != "VA"}) >= 2]
    if not out:
        raise ValueError(f"{name} has no endmember with two or more elements in {'-'.join(elements)}")
    return out


def _vibrations(model, atoms, phonons, policy):
    from discopt.calphad.mlip import phonons as ph

    if phonons.quasi_harmonic:
        return ph.quasi_harmonic(model, atoms, phonons, policy)
    return ph.harmonic(model, atoms, phonons)


def _formation(
    model,
    db,
    elements,
    compounds,
    refs,
    policy,
    structures,
    sigma,
    model_sigma,
    T,
    result,
    phonons=None,
    entropy_sigma=None,
    heat_capacity_sigma=None,
):
    items = compounds.items() if isinstance(compounds, dict) else ((c, None) for c in compounds)
    ref_vib = {}
    if phonons is not None:
        if entropy_sigma is None:
            raise ValueError(
                "phonon entropies of formation need entropy_sigma= (J/(mol K)); one structure has no spread"
            )
        if phonons.heat_capacity and heat_capacity_sigma is None:
            raise ValueError("phonon heat capacities need heat_capacity_sigma= (J/(mol K))")
        for e in elements:
            vib = _vibrations(model, refs[e].atoms(), phonons, policy)
            if vib.imaginary_fraction > phonons.imaginary_tolerance:
                raise ValueError(
                    f"the reference structure of {e} has imaginary phonon modes "
                    f"({vib.imaginary_fraction:.1%}): it is mechanically unstable"
                )
            ref_vib[e] = vib
        note = PHONON_NOTE.format(kind="quasi-harmonic" if phonons.quasi_harmonic else "harmonic")
    for name, given in items:
        name = name.upper()
        ps = _structure_for(name, structures)
        ps.check_ratios(db.phases[name].site_ratios, _constituents(db, name, elements), name)
        for em in _endmembers(db, name, elements, given):
            label = f"{name} {':'.join(em)}"
            atoms = ps.build(em)
            atoms = ps.build(em, _vegard(atoms))
            rel = _relax(model, [atoms], policy)[0]
            if not _check(label, rel, result, False):
                continue
            symbols = atoms.get_chemical_symbols()
            x = {e: symbols.count(e.capitalize()) / len(symbols) for e in elements}
            dh = rel.energy_per_atom - sum(x[e] * refs[e].energy_per_atom for e in elements)
            vib = None
            if phonons is not None:
                vib = _vibrations(model, rel.atoms(), phonons, policy)
                if vib.imaginary_fraction > phonons.imaginary_tolerance:
                    rel.left_prototype = True
                    rel.reason = f"imaginary phonon modes ({vib.imaginary_fraction:.1%}): mechanically unstable"
                    result.flagged.append((label, rel))
                    warnings.warn(f"{label}: {rel.reason}; not used", stacklevel=3)
                    continue
                dh += vib.H(298.15) - sum(x[e] * ref_vib[e].H(298.15) for e in elements)
            sig = _kj(sigma, name, "sigma")
            if sig is None:
                unc = _kj(model_sigma, name, "model_sigma")
                if unc is None:
                    u = _model_sigma(model, [atoms])
                    if u is not None:
                        ref_unc = [_model_sigma(model, [refs[e].atoms()]) or 0.0 for e in elements]
                        unc = math.sqrt(u**2 + sum((x[e] * r) ** 2 for e, r in zip(elements, ref_unc)))
                        unc *= EV_TO_J_PER_MOL
                sig = _sigma(name, label, None, unc, single=True)
            result.dataset.add_formation(
                name,
                dh * EV_TO_J_PER_MOL,
                sig,
                endmember=":".join(em),
                T=T,
                source=model.name,
                note=APPROXIMATION if vib is None else note,
            )
            result.rows.append(
                {
                    "kind": "HM_FORM",
                    "phase": name,
                    "config": ":".join(em),
                    "x": x[elements[0]],
                    "value": dh * EV_TO_J_PER_MOL / 1000,
                    "sigma": sig / 1000,
                    "structures": 1,
                    "sigma_dof": None,
                }
            )
            if vib is not None:
                kinds = [("SM_FORM", "S", entropy_sigma)]
                if phonons.heat_capacity:
                    kinds.append(("CPM_FORM", "Cp", heat_capacity_sigma))
                for kind, prop, sig_k in kinds:
                    for t in phonons.temperatures:
                        val = getattr(vib, prop)(t) - sum(x[e] * getattr(ref_vib[e], prop)(t) for e in elements)
                        result.dataset.add_formation(
                            name,
                            val * EV_TO_J_PER_MOL,
                            float(sig_k),
                            endmember=":".join(em),
                            T=float(t),
                            source=model.name,
                            note=note,
                            kind=kind,
                        )
                        result.rows.append(
                            {
                                "kind": kind,
                                "phase": name,
                                "config": f"{':'.join(em)} {t:g} K",
                                "x": x[elements[0]],
                                "value": val * EV_TO_J_PER_MOL,
                                "sigma": float(sig_k),
                                "structures": 1,
                                "sigma_dof": None,
                            }
                        )


# ---------------------------------------------------------------------------- mixing


def _mixing(
    model,
    db,
    elements,
    mixing,
    method,
    policy,
    structures,
    sigma,
    model_sigma,
    pool,
    T,
    sqs,
    result,
    random=None,
    cluster_expansion=None,
    monte_carlo=None,
    entropy_sigma=None,
):
    from discopt.calphad.fit import Observation
    from discopt.calphad.mlip import random_alloy
    from discopt.calphad.mlip.sqs import SQSOptions, generate

    if method not in ("sqs", "random", "cluster_expansion"):
        raise ValueError(f"method={method!r}: use 'sqs', 'random' or 'cluster_expansion'")
    if monte_carlo is not None and method != "cluster_expansion":
        raise ValueError("monte_carlo= samples a cluster expansion: use it with method='cluster_expansion'")
    options = sqs or SQSOptions()
    roptions = random or random_alloy.RandomOptions()
    tag = "SQS" if method == "sqs" else "random"
    if len(elements) != 2:
        raise NotImplementedError("mixing enthalpies need a binary system")
    for name, grid in mixing.items():
        name = name.upper()
        ps = _structure_for(name, structures)
        cons = _constituents(db, name, elements)
        ps.check_ratios(db.phases[name].site_ratios, cons, name)
        mix = [s for s, c in enumerate(cons) if set(elements) <= set(c)]
        if len(mix) != 1 or any(len(c) > 1 for s, c in enumerate(cons) if s != mix[0]):
            raise NotImplementedError(
                f"{name}: SQS mixing needs exactly one sublattice mixing {elements}, "
                f"and every other sublattice fixed; the phase has {cons}"
            )
        # VA may also be allowed on the mixing sublattice: the SQS mix the two elements only, at a
        # vacancy fraction of zero, and the observations say so through their site fractions.
        m = mix[0]
        fixed = {s: c[0] for s, c in enumerate(cons) if s != m}
        ends = {}
        for e in elements:
            em = tuple(e if s == m else fixed[s] for s in range(len(cons)))
            atoms = ps.build(em)
            atoms = ps.build(em, _vegard(atoms))
            rel = _relax(model, [atoms], policy)[0]
            label = f"{name} {':'.join(em)}"
            if not _check(label, rel, result, False):
                raise ValueError(
                    f"{label}: the pure endmember of the mixing lattice left its prototype; "
                    "ΔH_mix needs it. Try RelaxPolicy(mode='symmetry')"
                )
            ends[e] = rel
        if method == "cluster_expansion":
            _cluster_expansion(
                model,
                name,
                ps,
                m,
                elements,
                fixed,
                ends,
                grid,
                policy,
                cluster_expansion,
                sigma,
                model_sigma,
                T,
                result,
                monte_carlo,
                entropy_sigma,
            )
            continue
        points = []
        for y in np.atleast_1d(np.asarray(grid, float)):
            y = float(y)
            # Vegard volume at this composition, from the two relaxed endmembers
            v = y * _volume(ends[elements[0]]) + (1 - y) * _volume(ends[elements[1]])
            if method == "sqs":
                cells = generate(ps, m, elements, fixed, y, v, options, getattr(model, "cache", None))
            else:
                cells = random_alloy.generate(ps, m, elements, fixed, y, v, roptions, offset=int(round(y * 1e6)))
            dh, xs, used = [], [], []
            for k, atoms in enumerate(cells):
                rel = _relax(model, [atoms], policy)[0]
                if not _check(f"{name} {tag} y={y:g} #{k}", rel, result, False):
                    continue
                dh.append(
                    rel.energy_per_atom
                    - y * ends[elements[0]].energy_per_atom
                    - (1 - y) * ends[elements[1]].energy_per_atom
                )
                xs.append(atoms.get_chemical_symbols().count(elements[0].capitalize()) / len(atoms))
                used.append(atoms)
            if dh:
                points.append((y, np.array(dh) * EV_TO_J_PER_MOL, float(np.mean(xs)), used))
        # configurational σ: pooled over the phase's compositions, or per composition. SQS are a
        # few optimized structures, so σ is their spread. Random decorations are independent
        # draws of the random-alloy average, so σ is the standard error of their mean, s/√n.
        dofs = [len(dh) - 1 for _, dh, _, _ in points]
        if pool and sum(dofs) > 0:
            pooled = math.sqrt(
                sum(float(np.var(dh, ddof=1)) * k for (_, dh, _, _), k in zip(points, dofs) if k > 0) / sum(dofs)
            )
            spreads = [(pooled, sum(dofs))] * len(points)
        else:
            spreads = [
                (float(np.std(dh, ddof=1)), k) if k > 0 else (None, None) for (_, dh, _, _), k in zip(points, dofs)
            ]
        config = [
            (s_ / math.sqrt(len(dh)) if (method == "random" and s_ is not None) else s_, k)
            for (s_, k), (_, dh, _, _) in zip(spreads, points)
        ]
        given = _kj(sigma, name, "sigma")
        user_model = _kj(model_sigma, name, "model_sigma")
        warned = False
        for (y, dh, x, used), (cfg, dof) in zip(points, config):
            label = f"{name} y={y:g}"
            if given is not None:
                sig = given
            else:
                unc = user_model
                if unc is None:
                    u = _model_sigma(model, used)
                    unc = None if u is None else u * EV_TO_J_PER_MOL
                sig = _sigma(name, label, cfg, unc, single=len(dh) < 2 and dof is None)
                if unc is None and not warned:
                    warnings.warn(
                        f"{name}: σ covers only the configurational ({tag}) spread, not the error of {model.name}; "
                        "pass model_sigma= (kJ/mol) to include it",
                        stacklevel=3,
                    )
                    warned = True
            sf = tuple({elements[0]: y, elements[1]: 1 - y} if s == m else {fixed[s]: 1.0} for s in range(len(cons)))
            value = float(np.mean(dh))
            result.dataset.add(
                Observation(
                    "HM_MIX",
                    T,
                    value,
                    sig,
                    phase=name,
                    x=x,
                    site_fractions=sf,
                    source=model.name,
                    note=APPROXIMATION,
                )
            )
            result.rows.append(
                {
                    "kind": "HM_MIX",
                    "phase": name,
                    "config": f"y={y:g}",
                    "x": x,
                    "value": value / 1000,
                    "sigma": sig / 1000,
                    "structures": len(dh),
                    "sigma_dof": None if given is not None else dof,
                }
            )


def _cluster_expansion(
    model,
    name,
    ps,
    m,
    elements,
    fixed,
    ends,
    grid,
    policy,
    options,
    sigma,
    model_sigma,
    T,
    result,
    monte_carlo=None,
    entropy_sigma=None,
):
    """HM_MIX of the random alloy on ``grid``, from a cluster expansion fitted to enumerated structures."""
    from discopt.calphad.fit import Observation
    from discopt.calphad.mlip import cluster_expansion as cem
    from discopt.calphad.mlip import monte_carlo as mcm

    options = options or cem.ClusterExpansionOptions()
    va, vb = _volume(ends[elements[0]]), _volume(ends[elements[1]])
    cs, symbol, candidates = cem.training_structures(ps, m, elements, fixed, vb, options)
    n_mix = sum(len(c) > 1 for c in cs.chemical_symbols)
    n_prim = len(cs.primitive_structure)
    train, energies, relaxed = [], [], []
    for k, ideal in enumerate(candidates):
        y = cem._site_fraction(ideal, elements, n_mix * len(ideal) // n_prim)
        atoms = ideal.copy()
        v = y * va + (1 - y) * vb
        atoms.set_cell(ideal.cell * (v / (ideal.get_volume() / len(ideal))) ** (1 / 3), scale_atoms=True)
        rel = _relax(model, [atoms], policy)[0]
        if not _check(f"{name} CE training #{k} {ideal.get_chemical_formula()}", rel, result, False):
            continue
        train.append(ideal)
        relaxed.append(atoms)
        energies.append(
            rel.energy_per_atom - y * ends[elements[0]].energy_per_atom - (1 - y) * ends[elements[1]].energy_per_atom
        )
    ce = cem.fit(cs, symbol, elements, train, energies, options)
    result.cluster_expansions[name] = ce
    # an exact fit leaves only round-off (~1e-17 eV/atom); below 1 µeV/atom the expansion has no error to report
    cfg = ce.rmse_validation * EV_TO_J_PER_MOL if ce.rmse_validation > 1e-6 else 0.0
    given = _kj(sigma, name, "sigma")
    unc = _kj(model_sigma, name, "model_sigma")
    if given is None and unc is None:
        u = _model_sigma(model, relaxed)
        unc = None if u is None else u * EV_TO_J_PER_MOL
        if unc is None:
            warnings.warn(
                f"{name}: σ covers only the cluster expansion's cross-validation error, not the error of "
                f"{model.name}; pass model_sigma= (kJ/mol) to include it",
                stacklevel=4,
            )
    dof = None if given is not None else len(train) - ce.n_parameters

    def add(kind, t, y, value, sig, config, note, unit=1000.0):
        sf = tuple({elements[0]: y, elements[1]: 1 - y} if s == m else {fixed[s]: 1.0} for s in range(len(fixed) + 1))
        x = y * n_mix / n_prim  # mole fraction: the other sites hold one element each
        result.dataset.add(
            Observation(kind, t, value, sig, phase=name, x=x, site_fractions=sf, source=model.name, note=note)
        )
        result.rows.append(
            {
                "kind": kind,
                "phase": name,
                "config": config,
                "x": x,
                "value": value / unit,
                "sigma": sig / unit,
                "structures": len(train),
                "sigma_dof": dof,
            }
        )

    for y in np.atleast_1d(np.asarray(grid, float)):
        y = float(y)
        if monte_carlo is None:
            sig = given if given is not None else _sigma(name, f"{name} y={y:g}", cfg, unc, single=False)
            add("HM_MIX", T, y, ce.random_alloy(y) * EV_TO_J_PER_MOL, sig, f"y={y:g} CE", CE_NOTE)
            continue
        runs = mcm.short_range_order(ce, y, monte_carlo)
        result.short_range_order.setdefault(name, []).extend(runs)
        for r in runs:
            strong = {k: a for k, a in r.alpha.items() if abs(a) > 0.5}
            if strong:
                warnings.warn(
                    f"{name} y={r.y:g} at {r.T:g} K: Warren-Cowley {strong}; so strongly ordered or clustered that "
                    "the sample may be below the expansion's ordering or miscibility temperature, and not one "
                    "solution phase",
                    stacklevel=4,
                )
            stat = r.H_error * EV_TO_J_PER_MOL
            sig = given if given is not None else _sigma(name, f"{name} y={r.y:g}", math.hypot(cfg, stat), unc, False)
            add("HM_MIX", r.T, r.y, r.H * EV_TO_J_PER_MOL, sig, f"y={r.y:g} MC {r.T:g}K", MC_NOTE)
            if r.S is not None:
                # the energy errors (expansion and model) taken over at T: an estimate, not a bound
                energy = math.hypot(cfg, unc or 0.0)
                s_sig = math.hypot(r.S_error * EV_TO_J_PER_MOL, energy / r.T)
                if entropy_sigma is not None:
                    s_sig = float(entropy_sigma)
                add("SM_MIX", r.T, r.y, r.S * EV_TO_J_PER_MOL, s_sig, f"y={r.y:g} MC {r.T:g}K", MC_NOTE, 1.0)


def _liquid(model, db, elements, liquid, options, sigma, model_sigma, result):
    """HM_MIX and GM_MIX of liquids from MD and thermodynamic integration."""
    from discopt.calphad.fit import Observation
    from discopt.calphad.mlip.liquid import liquid_mixing

    if len(elements) != 2:
        raise NotImplementedError("liquid mixing data need a binary system")
    for name, grid in liquid.items():
        name = name.upper()
        cons = _constituents(db, name, elements)
        if len(cons) != 1 or not set(elements) <= set(cons[0]):
            raise NotImplementedError(f"{name}: liquid data need one sublattice that mixes {elements}; it has {cons}")
        runs = liquid_mixing(model, elements, grid, options)
        result.liquids[name] = runs
        given = _kj(sigma, name, "sigma")
        unc = _kj(model_sigma, name, "model_sigma")
        if given is None and unc is None:
            warnings.warn(
                f"{name}: σ covers only the MD statistics, not the error of {model.name}; "
                "pass model_sigma= (kJ/mol) to include it",
                stacklevel=3,
            )
        for r in runs:
            sf = ({elements[0]: r.x, elements[1]: 1 - r.x},)
            values = [("HM_MIX", r.H, r.H_error)]
            if r.G is not None:
                values.append(("GM_MIX", r.G, r.G_error))
            for kind, v, e in values:
                label = f"{name} {kind} x={r.x:g} {r.T:g} K"
                sig = given if given is not None else _sigma(name, label, e * EV_TO_J_PER_MOL, unc, single=False)
                value = v * EV_TO_J_PER_MOL
                result.dataset.add(
                    Observation(
                        kind, r.T, value, sig, phase=name, x=r.x, site_fractions=sf, source=model.name, note=MD_NOTE
                    )
                )
                result.rows.append(
                    {
                        "kind": kind,
                        "phase": name,
                        "config": f"x={r.x:g} MD {r.T:g}K",
                        "x": r.x,
                        "value": value / 1000,
                        "sigma": sig / 1000,
                        "structures": 1,
                        "sigma_dof": None,
                    }
                )


def _volume(rel: Relaxation) -> float:
    return abs(float(np.linalg.det(np.asarray(rel.cell)))) / rel.natoms


# ---------------------------------------------------------------------------- entry points


def compute(
    model,
    db,
    elements,
    *,
    compounds=(),
    mixing=None,
    method="sqs",
    relax=None,
    sigma="spread",
    model_sigma=None,
    pool: bool = True,
    structures=None,
    sqs=None,
    random=None,
    cluster_expansion=None,
    monte_carlo=None,
    liquid=None,
    md=None,
    phonons=None,
    entropy_sigma: float | None = None,
    heat_capacity_sigma: float | None = None,
    T: float = 298.15,
) -> MLIPResult:
    """Formation and mixing enthalpies from ``model``, as a dataset with everything behind it.

    ``compounds`` lists phases whose endmembers with two or more elements get a
    formation enthalpy, or maps a phase to its endmembers (``{"AL2Y":
    ["AL:CU"]}``). ``mixing`` maps a solution phase to a grid of site
    fractions of the first element on its mixing sublattice. ``structures``
    maps a phase name to a :class:`PhaseStructure`, or to ``(prototype,
    sublattice sites...)``, or to ``(ase.Atoms, atoms of each sublattice)`` for
    a structure no prototype covers (:meth:`PhaseStructure.from_atoms`), for
    phases :data:`PHASES` does not know. ``sqs``
    is an :class:`~discopt.calphad.mlip.sqs.SQSOptions`. ``method="random"``
    averages over random decorations instead
    (:class:`~discopt.calphad.mlip.random_alloy.RandomOptions` via ``random=``),
    and then σ_config is the standard error of the mean, s/√n.
    ``method="cluster_expansion"`` relaxes the derivative superstructures of
    the lattice, has icet fit a cluster expansion to them
    (:class:`~discopt.calphad.mlip.cluster_expansion.ClusterExpansionOptions`
    via ``cluster_expansion=``), and predicts the random alloy at each grid
    composition; σ_config is the cross-validation RMSE, and the fit is kept in
    :attr:`MLIPResult.cluster_expansions`. Any grid composition works.
    With ``monte_carlo=`` (a
    :class:`~discopt.calphad.mlip.monte_carlo.MonteCarloOptions`), mchammer
    samples the expansion at each of its temperatures instead, so the data
    include short-range order: ``HM_MIX`` at each temperature and, by
    thermodynamic integration, ``SM_MIX`` (σ ``entropy_sigma`` when given).
    The Warren–Cowley parameters are in :attr:`MLIPResult.short_range_order`.

    ``liquid`` maps a liquid phase to a grid of mole fractions of the first
    element. Molecular dynamics with ``md=`` (a
    :class:`~discopt.calphad.mlip.liquid.LiquidOptions`) gives ``HM_MIX`` at
    each of its temperatures, and alchemical thermodynamic integration gives
    ``GM_MIX``; the runs are in :attr:`MLIPResult.liquids`.

    ``phonons`` (a :class:`~discopt.calphad.mlip.phonons.PhononOptions`)
    computes harmonic or quasi-harmonic phonons for every compound or
    endmember and for the elements' reference structures. The formation
    enthalpy then includes the vibrational enthalpy at 298.15 K, and entropies
    of formation (``SM_FORM``) are added at ``phonons.temperatures``, with σ
    ``entropy_sigma`` in J/(mol K); with ``phonons.heat_capacity``, heat
    capacities of formation (``CPM_FORM``) too, with ``heat_capacity_sigma``.
    A structure with imaginary modes is flagged and left out. In the table,
    entropy and heat-capacity rows are in J/(mol K), not kJ/mol.

    ``sigma``, ``model_sigma`` and ``pool`` set the uncertainties; see the
    module docstring.

    A structure that leaves its prototype is reported in
    :attr:`MLIPResult.flagged` and left out of the dataset.
    """
    elements = tuple(e.upper() for e in elements)
    policy = relax or RelaxPolicy()
    result = MLIPResult(Dataset())
    result.references = reference_energies(model, db, elements, relax=policy, structures=structures, result=result)
    if compounds:
        _formation(
            model,
            db,
            elements,
            compounds,
            result.references,
            policy,
            structures,
            sigma,
            model_sigma,
            T,
            result,
            phonons,
            entropy_sigma,
            heat_capacity_sigma,
        )
    if mixing:
        _mixing(
            model,
            db,
            elements,
            mixing,
            method,
            policy,
            structures,
            sigma,
            model_sigma,
            pool,
            T,
            sqs,
            result,
            random,
            cluster_expansion,
            monte_carlo,
            entropy_sigma,
        )
    if liquid:
        _liquid(model, db, elements, liquid, md, sigma, model_sigma, result)
    return result


def dataset(model, db, elements, **kwargs) -> Dataset:
    """:func:`compute`, returning only the :class:`~discopt.calphad.fit.Dataset`."""
    return compute(model, db, elements, **kwargs).dataset
