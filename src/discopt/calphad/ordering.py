"""Plain CEF stand-ins for order–disorder (partitioned) phases.

A partitioned phase such as L1₂ (FCC_L12) or B2 (BCC_B2) is described in two
parts, joined by ``AMEND_PHASE_DESCRIPTION ... DISORDERED_PART``:

    G = G_dis(x) + G_ord(y) − G_ord(x)

``y`` holds the ordered phase's site fractions. ``x`` is their site average
over the substitutional sublattices, which is the disordered phase's
composition. The configurational entropy is that of the ordered sublattices.
The solver models this form natively
(:class:`~discopt.calphad.models.partitioned.PartitionedPhase`), except with
magnetic parameters on the ordered part. A stand-in covers that case, and gives
a plain phase to refit.

:func:`stand_in` writes it as a plain CEF phase on the ordered sublattices. The
G and L parameters enter linearly, each with a known factor in y:

- φ_p(y) − φ_p(x(y)) for a parameter of the ordered part;
- the disordered factor at x(y), scaled from the disordered formula unit to the
  ordered one, for a parameter of the disordered part.

Each factor is projected by least squares onto a CEF basis on the ordered
sublattices: endmembers, Redlich–Kister terms up to ``max_order`` on each
sublattice, and reciprocal terms of orders 0 to 2. A stand-in parameter is therefore a fixed
linear combination of the source parameters' own functions. It is exact in T,
symbolic, and written as TDB, with one FUNCTION per source parameter.

What a stand-in is not:

- **Not always exact in y.** For B2 the projection is exact. For L1₂ the
  partitioned form contains terms of higher degree than a CEF basis
  represents, and :attr:`StandIn.max_error` reports the residual in J/mol.
- **Not complete.** Magnetic contributions (TC, BMAGN) and vacancies on
  substitutional sublattices are dropped and listed in :attr:`StandIn.dropped`.
- **A one-phase description.** Its disordered state is just a configuration of
  it, so it reproduces an order–disorder transition only as far as the
  projection is exact.

To refit, refit the parameters of the partitioned description (the disordered
parent, or the ordered part) and build the stand-in again. Its disordered limit
then follows the parent automatically. Refitting the stand-in's own
parameters breaks that tie.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
import sympy as sp

from discopt.calphad.database import MAGNETIC_KINDS, Parameter, Phase, ThermoDatabase
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.models.cef import CEFPhase, CEFTerm
from discopt.calphad.ops import NUM
from discopt.calphad.symbolic import PiecewiseT, function_ref

__all__ = ["StandIn", "stand_in"]

#: at T = 1/R a parameter of 1 J contributes exactly 1 to G/RT
_T_UNIT = 1.0 / R_GAS


@dataclass
class StandIn:
    """The result of :func:`stand_in`."""

    #: a copy of the database with the stand-in phase, its parameters and FUNCTIONs added; nothing else changes
    database: ThermoDatabase
    #: the stand-in phase's name
    name: str
    #: the partitioned phase and its disordered parent
    ordered: str
    disordered: str
    #: T (K) -> largest |G_stand-in − G_partitioned| over the sample grid, in J/mol of atoms
    max_error: dict[float, float] = field(default_factory=dict)
    #: what the stand-in leaves out
    dropped: list[str] = field(default_factory=list)
    #: number of stand-in parameters
    terms: int = 0

    def summary(self) -> str:
        err = ", ".join(f"{t:g} K: {e:.3g}" for t, e in self.max_error.items())
        lines = [
            f"{self.name}: stand-in for {self.ordered} (disordered part {self.disordered}), {self.terms} parameters",
            f"  largest |G error| in J/mol of atoms: {err}",
        ]
        lines += [f"  dropped: {d}" for d in self.dropped]
        return "\n".join(lines)


def _disordered_name(db: ThermoDatabase, ordered: str) -> str:
    for a in db.amendments(ordered):
        toks = a.replace(",", " ").split()
        for i, tok in enumerate(toks[:-1]):
            if tok in ("DIS_PART", "DISORDERED_PART") or "DISORDERED_PART".startswith(tok) and len(tok) >= 4:
                return toks[i + 1].upper()
    raise ValueError(f"{ordered} has no DISORDERED_PART amendment")


def _substitutional(constituents) -> list[int]:
    return [s for s, c in enumerate(constituents) if tuple(c) != ("VA",)]


def _clean(db: ThermoDatabase, names, elements) -> tuple[ThermoDatabase, list[str]]:
    """A copy whose two phases are plain: no DIS_PART link, no magnetism, no VA on substitutional sublattices."""
    dropped = []
    phases = dict(db.phases)
    params = []
    keep_va: dict[str, list[bool]] = {}
    for n in names:
        ph = phases[n]
        subst = _substitutional(ph.constituents)
        cons = []
        for s, c in enumerate(ph.constituents):
            c = tuple(x for x in c if x in elements or x == "VA")
            if s in subst and "VA" in c:
                dropped.append(f"vacancies on sublattice {s} of {n}")
                c = tuple(x for x in c if x != "VA")
            cons.append(c)
        phases[n] = dataclasses.replace(ph, constituents=tuple(cons))
        keep_va[n] = [s not in subst for s in range(len(cons))]
    for p in db.parameters:
        if p.phase in names:
            if p.kind in MAGNETIC_KINDS:
                dropped.append(f"magnetic parameter {p.label}")
                continue
            if any("VA" in c and not keep_va[p.phase][s] for s, c in enumerate(p.constituents)):
                continue  # a vacancy on a substitutional sublattice: dropped with it
        params.append(p)
    types = [t for t in db.type_definitions if not any(n in t.upper() for n in names)]
    out = dataclasses.replace(db, phases=phases, parameters=params, type_definitions=types)
    out.clear_cache()
    return out, sorted(set(dropped))


def _replace(db: ThermoDatabase, values: dict[int, PiecewiseT]) -> ThermoDatabase:
    params = [dataclasses.replace(p, function=values[i]) if i in values else p for i, p in enumerate(db.parameters)]
    return dataclasses.replace(db, parameters=params)


_ZERO = PiecewiseT((1e-3, 1e6), (sp.Integer(0),))
_ONE = PiecewiseT((1e-3, 1e6), (sp.Integer(1),))


def _factors(db: ThermoDatabase, phase: str, elements, points) -> dict[int, np.ndarray]:
    """{parameter index: its factor in G/RT per formula unit at each of ``points`` (model variables)}."""
    mine = [i for i, p in enumerate(db.parameters) if p.phase == phase and p.kind == "G"]
    zero = {i: _ZERO for i in mine}
    base = _replace(db, zero).phase_model(phase, elements)
    g0 = np.array([float(base.gibbs(u, NUM, _T_UNIT)) for u in points])
    out = {}
    for i in mine:
        model = _replace(db, {**zero, i: _ONE}).phase_model(phase, elements)
        out[i] = np.array([float(model.gibbs(u, NUM, _T_UNIT)) for u in points]) - g0
    return out


def _basis(constituents, max_order: int) -> list[tuple[tuple[tuple[str, ...], ...], int]]:
    """(constituent array, order) of every basis term on these sublattices."""
    terms = []
    for end in product(*constituents):
        terms.append((tuple((c,) for c in end), 0))
    mixing = [s for s, c in enumerate(constituents) if len(c) > 1]
    for s in mixing:
        for pair in combinations(constituents[s], 2):
            others = [constituents[t] if t != s else (pair,) for t in range(len(constituents))]
            for rest in product(*[[(c,) for c in o] if t != s else [pair] for t, o in enumerate(others)]):
                for k in range(max_order + 1):
                    terms.append((tuple(rest), k))
    for s1, s2 in combinations(mixing, 2):
        for p1 in combinations(constituents[s1], 2):
            for p2 in combinations(constituents[s2], 2):
                fixed = [
                    [(c,) for c in constituents[t]] if t not in (s1, s2) else [p1 if t == s1 else p2]
                    for t in range(len(constituents))
                ]
                for rest in product(*fixed):
                    for k in range(3):  # the CEF model supports reciprocal orders 0, 1 and 2
                        terms.append((tuple(rest), k))
    return terms


def stand_in(
    db: ThermoDatabase,
    ordered: str,
    elements,
    *,
    name: str | None = None,
    max_order: int = 3,
    grid: int = 21,
    temperatures=(300.0, 1000.0, 1600.0),
) -> StandIn:
    """A plain CEF stand-in for the partitioned phase ``ordered``, in the system ``elements``.

    The stand-in is added to a copy of ``db`` as phase ``name`` (default
    ``"<ordered>_SI"``), with the ordered phase's sublattices. ``max_order`` is
    the highest Redlich–Kister order in the basis. ``grid`` points per site
    fraction sample the projection. ``temperatures`` are where the accuracy is
    reported. See the module docstring for what a stand-in leaves out.
    """
    elements = tuple(e.upper() for e in elements)
    ordered = ordered.upper()
    dis = _disordered_name(db, ordered)
    name = (name or f"{ordered}_SI").upper()
    work, dropped = _clean(db, (ordered, dis), elements)

    ord_model = work.phase_model(ordered, elements)
    dis_model = work.phase_model(dis, elements)
    if not isinstance(ord_model, CEFPhase):
        raise NotImplementedError(f"{ordered}: expected a multi-sublattice phase")
    cons, ratios = ord_model.constituents, ord_model.site_ratios
    subst = _substitutional(cons)
    dsub = _substitutional(work.phases[dis].constituents)
    if len(dsub) != 1:
        raise NotImplementedError(f"{dis}: the disordered part needs exactly one substitutional sublattice")
    a_ord = sum(ratios[s] for s in subst)
    scale = a_ord / work.phases[dis].site_ratios[dsub[0]]

    # sample the ordered site fractions on a grid of the model variables
    axes = [np.linspace(0.0, 1.0, grid)] * ord_model.n_y
    points = [np.array(p) for p in product(*axes)]
    fractions = [ord_model.site_fractions(u) for u in points]

    def averaged(sf):
        """The ordered phase's site fractions with every substitutional sublattice replaced by the average x."""
        x = {c: sum(ratios[s] * sf[s].get(c, 0.0) for s in subst) / a_ord for c in elements}
        return [dict(x) if s in subst else sf[s] for s in range(len(cons))]

    avg_points = [ord_model.variables_from_site_fractions(averaged(sf)) for sf in fractions]

    def dis_variables(sf):
        x = averaged(sf)[subst[0]]
        dcons = dis_model.constituents if isinstance(dis_model, CEFPhase) else None
        if dcons is None:
            return np.array([x[dis_model.elements[0]]])
        full = [x if s == dsub[0] else {c: 1.0 for c in dcons[s]} for s in range(len(dcons))]
        return dis_model.variables_from_site_fractions(full)

    dis_points = [dis_variables(sf) for sf in fractions]

    # factors of every source parameter in G/RT per ordered formula unit
    factors: dict[int, np.ndarray] = {}
    f_ord_y = _factors(work, ordered, elements, points)
    f_ord_x = _factors(work, ordered, elements, avg_points)
    for i in f_ord_y:
        factors[i] = f_ord_y[i] - f_ord_x[i]
    for i, f in _factors(work, dis, elements, dis_points).items():
        factors[i] = scale * f

    # basis on the ordered sublattices
    terms = _basis(cons, max_order)
    plain = CEFPhase(name, elements, ratios, cons, ())
    g_ideal = np.array([float(plain.gibbs(u, NUM, _T_UNIT)) for u in points])
    A = np.empty((len(points), len(terms)))
    for j, (arr, k) in enumerate(terms):
        model = CEFPhase(name, elements, ratios, cons, (CEFTerm(arr, k, 1.0),))
        A[:, j] = np.array([float(model.gibbs(u, NUM, _T_UNIT)) for u in points]) - g_ideal
    coefs = {i: np.linalg.lstsq(A, f, rcond=None)[0] for i, f in factors.items()}

    # the stand-in: one FUNCTION per source parameter, and each basis term a combination of them
    # added to the caller's database, which is otherwise unchanged
    out = dataclasses.replace(db, phases=dict(db.phases), parameters=list(db.parameters), functions=dict(db.functions))
    out.phases[name] = Phase(name, "", tuple(ratios), tuple(tuple(c) for c in cons))
    fname = {}
    for n, i in enumerate(sorted(coefs)):
        fname[i] = f"{name[:6]}P{n}"
        out.functions[fname[i]] = work.parameters[i].function
    lo = min(work.parameters[i].function.t_min for i in coefs)
    hi = max(work.parameters[i].function.t_max for i in coefs)
    added = 0
    for j, (arr, k) in enumerate(terms):
        expr = sp.Integer(0)
        for i, c in coefs.items():
            if abs(c[j]) > 1e-9:
                expr = expr + float(np.round(c[j], 10)) * function_ref(fname[i])
        if expr != 0:
            out.parameters.append(Parameter("G", name, arr, k, PiecewiseT((lo, hi), (expr,))))
            added += 1
    out.clear_cache()

    # accuracy: the stand-in against the partitioned G, both evaluated here
    atoms = np.array([float(sum(ord_model.amounts(u))) for u in points])
    model = out.phase_model(name, elements)
    errors = {}
    for t in temperatures:
        values = {i: work.evaluate(work.parameters[i].function, t) for i in coefs}
        target = R_GAS * t * g_ideal + sum(v * factors[i] for i, v in values.items())
        mine = np.array([float(model.gibbs(u, NUM, t)) for u in points]) * R_GAS * t
        errors[float(t)] = float(np.max(np.abs(mine - target) / atoms))
    return StandIn(out, name, ordered, dis, errors, dropped, added)
