"""Thermodynamic database: elements, functions, phases and parameters, independent of TDB syntax.

A :class:`ThermoDatabase` keeps the CALPHAD structure symbolically: each FUNCTION
and PARAMETER is a :class:`~discopt.calphad.symbolic.PiecewiseT` of SymPy
expressions, with named function references left unevaluated. It evaluates
numerically at a fixed T, with a cache, and it builds phase models for the
equilibrium solver.

Supported phase structures, for any system of two or more elements:

- all sublattices singleton: becomes a :class:`StoichiometricPhase`;
- in a binary, one sublattice that mixes the two elements plus singleton
  sublattices (an element or VA): becomes a :class:`BinarySolution` with
  ``sites`` and ``fixed`` amounts (the named ``g``/``L`` view is kept);
- anything else with mixing (several mixing sublattices, VA in a mixing
  sublattice, three or more constituents, reciprocal and ternary parameters):
  becomes a :class:`~discopt.calphad.models.cef.CEFPhase`, one term per parameter.

A phase whose type code has a ``MAGNETIC`` TYPE_DEFINITION, and that has
TC/BMAGN parameters, always becomes a :class:`CEFPhase` with the
Inden–Hillert–Jarl term (see :mod:`discopt.calphad.models.magnetic`).

Everything else raises :class:`UnsupportedModelError` with the reason: the
Xiong magnetic model, order–disorder descriptions, ionic phases, other parameter
kinds, or CEF structures pycalphad also rejects.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import sympy as sp

from discopt.calphad.models.cef import CEFPhase, CEFTerm
from discopt.calphad.models.magnetic import Magnetic
from discopt.calphad.models.partitioned import PartitionedPhase
from discopt.calphad.models.solution import BinarySolution
from discopt.calphad.models.stoichiometric import StoichiometricPhase
from discopt.calphad.symbolic import P, PiecewiseT, T, function_ref, function_refs

__all__ = [
    "Element",
    "Species",
    "Phase",
    "Parameter",
    "DatabaseValue",
    "ThermoDatabase",
    "UnsupportedModelError",
    "PhaseNotApplicable",
]


#: parameter kinds of the magnetic contribution; ``BM`` is the TDB abbreviation of ``BMAGN``
MAGNETIC_KINDS = ("TC", "BMAGN", "BM")


#: sublattice symmetry implied by the phase options F and B of a 4-sublattice ordered phase, as in
#: pycalphad: F permutes all four sublattices; B permutes within {0, 1} and within {2, 3}, and swaps the pairs
SUBLATTICE_SYMMETRY = {"F": ((0, 1, 2, 3),), "B": ((0, 1), (2, 3))}


def _symmetric_group(array, symmetry) -> set:
    """Every constituent array equivalent to ``array`` under ``symmetry`` (pycalphad's ``generate_symmetric_group``)."""
    from itertools import chain, permutations

    n = len(array)
    seen = sorted(i for cls in symmetry for i in cls)
    out = set()
    for inner in product(*(permutations(cls) for cls in symmetry)):
        for outer in permutations(inner):
            order = list(chain.from_iterable(outer))
            new = list(array)
            for dst, src in zip(seen, order):
                new[dst] = array[src]
            out.add(tuple(new[i] for i in range(n)))
    return out


def _symmetric_expansion(params, hints) -> list:
    """``params`` plus, for each, its other symmetry images (the F and B phase options)."""
    sym = [SUBLATTICE_SYMMETRY[h] for h in hints if h in SUBLATTICE_SYMMETRY]
    if not sym:
        return list(params)
    out = list(params)
    for p in params:
        if len(p.constituents) <= max(max(c) for c in sym[0]):
            continue
        for arr in _symmetric_group(p.constituents, sym[0]) - {p.constituents}:
            out.append(dataclasses.replace(p, constituents=arr))
    return out


class UnsupportedModelError(NotImplementedError):
    """The phase uses a model feature that discopt-calphad does not support yet."""


class PhaseNotApplicable(ValueError):
    """The phase cannot exist in the requested element system."""


@dataclass(frozen=True)
class Element:
    name: str
    reference_phase: str = ""
    mass: float = 0.0
    h298: float = 0.0
    s298: float = 0.0


@dataclass(frozen=True)
class Species:
    name: str
    formula: str


@dataclass
class Phase:
    name: str
    typecodes: str
    site_ratios: tuple[float, ...]
    constituents: tuple[tuple[str, ...], ...] | None = None
    #: model hints from the name suffix (``LIQUID:L`` gives ``("L",)``)
    hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class Parameter:
    """One PARAMETER. ``kind`` is ``G`` for both G and L; ``constituents`` is sorted within each sublattice."""

    kind: str
    phase: str
    constituents: tuple[tuple[str, ...], ...]
    order: int
    function: PiecewiseT

    @property
    def label(self) -> str:
        array = ":".join(",".join(s) for s in self.constituents)
        return f"{self.kind}({self.phase},{array};{self.order})"

    def matches(self, array) -> bool:
        return len(array) == len(self.constituents) and all(
            p == ("*",) or p == a for p, a in zip(self.constituents, array)
        )


@dataclass(frozen=True, eq=False)
class DatabaseValue:
    """A sum of signed database parameters: callable at T (J per formula unit), symbolic via :meth:`expr`."""

    db: ThermoDatabase
    terms: tuple[tuple[float, Parameter], ...]

    def __call__(self, T_: float) -> float:
        return float(sum(s * self.db.evaluate(p.function, T_) for s, p in self.terms))

    def expr(self, expand: bool = False) -> sp.Expr:
        """The value as a SymPy expression in T; ``expand=True`` inlines named functions."""
        parts = []
        for s, p in self.terms:
            e = self.db.expand(p.function) if expand else _compact(p.function)
            parts.append(e if s == 1 else -e if s == -1 else s * e)
        return sp.Add(*parts)

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        return " + ".join(("-" if s < 0 else "") + p.label for s, p in self.terms)


def _compact(pw: PiecewiseT) -> sp.Expr:
    return pw.exprs[0] if len(pw.exprs) == 1 else pw.as_sympy()


@dataclass
class ThermoDatabase:
    elements: dict[str, Element] = field(default_factory=dict)
    species: dict[str, Species] = field(default_factory=dict)
    functions: dict[str, PiecewiseT] = field(default_factory=dict)
    phases: dict[str, Phase] = field(default_factory=dict)
    parameters: list[Parameter] = field(default_factory=list)
    #: raw TYPE_DEFINITION bodies, in file order
    type_definitions: list[str] = field(default_factory=list)
    #: if True, a T outside a function's ranges uses the nearest range instead of raising
    extrapolate: bool = False
    #: pressure (Pa) at which pressure-dependent expressions are evaluated
    pressure: float = 101325.0

    def __post_init__(self):
        self._cache: dict = {}
        self._compiled: dict = {}

    # ------------------------------------------------------------------ TDB I/O

    @classmethod
    def from_tdb(cls, path, *, skipped: list | None = None) -> ThermoDatabase:
        """Read a TDB file. See :func:`~discopt.calphad.io.tdb.parse_tdb` for ``skipped``."""
        return cls.from_string(Path(path).read_text(encoding="latin-1"), skipped=skipped)

    @classmethod
    def from_string(cls, text: str, *, skipped: list | None = None) -> ThermoDatabase:
        """Read TDB commands from a string. See :func:`~discopt.calphad.io.tdb.parse_tdb` for ``skipped``."""
        db = cls()
        db.add_tdb(text, skipped=skipped)
        return db

    def add_tdb(self, text: str, *, skipped: list | None = None) -> ThermoDatabase:
        """Add TDB commands, for example a single pasted PARAMETER line."""
        from discopt.calphad.io.tdb import parse_tdb

        parse_tdb(text, self, skipped=skipped)
        self.clear_cache()
        return self

    def to_tdb(self) -> str:
        """Write the database as TDB text."""
        from discopt.calphad.io.tdb import write_tdb

        return write_tdb(self)

    def clear_cache(self) -> None:
        """Forget cached numeric values; call after editing ``functions`` or ``parameters`` directly."""
        self._cache.clear()

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, what, T_: float, *, extrapolate: bool | None = None) -> float:
        """Value at temperature ``T_`` of a FUNCTION name or a :class:`PiecewiseT`, in J."""
        ext = self.extrapolate if extrapolate is None else extrapolate
        return self._eval(what, float(T_), ext, ())

    def _eval(self, what, t: float, ext: bool, stack: tuple[str, ...]) -> float:
        if isinstance(what, str):
            name = what.upper()
            if name in stack:
                raise ValueError(f"recursive FUNCTION definition: {' -> '.join((*stack, name))}")
            if name not in self.functions:
                raise KeyError(f"FUNCTION {name} is not defined" + (f" (used by {stack[-1]})" if stack else ""))
            pw, stack, key = self.functions[name], (*stack, name), (name, t, ext, self.pressure)
        else:
            pw, key = what, (what, t, ext, self.pressure)
        if key in self._cache:
            return self._cache[key]
        try:
            expr = pw.select(t, extrapolate=ext)
        except ValueError as err:
            where = f"FUNCTION {stack[-1]}" if isinstance(what, str) else "expression"
            raise ValueError(f"{where}: {err}") from None
        f, names = self._compile(expr)
        value = float(f(t, self.pressure, *(self._eval(n, t, ext, stack) for n in names)))
        self._cache[key] = value
        return value

    def _compile(self, expr):
        hit = self._compiled.get(expr)
        if hit is None:
            names = sorted(function_refs(expr))
            syms = [sp.Symbol(f"_ref{i}") for i in range(len(names))]
            body = expr.xreplace({function_ref(n): s for n, s in zip(names, syms)})
            hit = (sp.lambdify([T, P, *syms], body, "math"), names)
            self._compiled[expr] = hit
        return hit

    def expand(self, what) -> sp.Expr:
        """A FUNCTION name or :class:`PiecewiseT` as one SymPy expression with every named function inlined."""
        pw = self.functions[what.upper()] if isinstance(what, str) else what
        base = _compact(pw)
        return base.xreplace({function_ref(n): self.expand(n) for n in function_refs(base)})

    # ------------------------------------------------------------------ queries

    def phase_parameters(self, phase: str, kind: str | None = None) -> list[Parameter]:
        phase = phase.upper()
        return [p for p in self.parameters if p.phase == phase and (kind is None or p.kind == kind)]

    def amendments(self, phase: str) -> list[str]:
        """TYPE_DEFINITION amendments (``AMEND_PHASE_DESCRIPTION``/``A_P_D``) that name the phase."""
        out = []
        for body in self.type_definitions:
            toks = body.upper().split()
            for i, tok in enumerate(toks[:-1]):
                if tok in ("A_P_D", "AMEND_PHASE_DESCRIPTION") and toks[i + 1] == phase.upper():
                    out.append(" ".join(toks[i + 2 :]))
        return out

    # ------------------------------------------------------------------ phase models

    def phase_model(self, name: str, elements):
        """Build the solver's phase model for ``name`` in the binary system ``elements``.

        Raises :class:`PhaseNotApplicable` if the phase cannot exist with these
        elements, and :class:`UnsupportedModelError` if the phase needs a model
        feature that is not implemented yet.
        """
        elements = tuple(e.upper() for e in elements)
        if len(elements) < 2 or len(set(elements)) != len(elements):
            raise ValueError(f"phase models need at least two distinct elements, got {elements}")
        key = name.upper()
        if key not in self.phases:
            raise KeyError(f"PHASE {key} is not defined")
        ph = self.phases[key]
        if ph.constituents is None:
            raise ValueError(f"PHASE {key} has no CONSTITUENT command")
        allowed = set(elements) | {"VA"}
        subl = [tuple(sorted(c for c in s if c in allowed)) for s in ph.constituents]
        if any(not s for s in subl) or all(s == ("VA",) for s in subl):
            raise PhaseNotApplicable(f"{key}: no constituent set within {'-'.join(elements)}")

        params = [
            p for p in self.phase_parameters(key) if all(c == "*" or c in allowed for s in p.constituents for c in s)
        ]
        for p in params:
            if len(p.constituents) != len(subl):
                raise ValueError(f"{p.label}: {len(p.constituents)} sublattices, but {key} has {len(subl)}")
        kinds = {p.kind for p in params} - {"G"} - set(MAGNETIC_KINDS)
        if kinds:
            raise UnsupportedModelError(f"{key}: parameter kinds {sorted(kinds)} are not supported yet")
        part = self.disordered_part(key)
        if part is not None:
            return self._partitioned_phase(key, ph, elements, *part)
        if "I" in ph.hints:
            raise UnsupportedModelError(f"{key}: ionic liquid models are not supported yet")
        magnetic = self._magnetic(key, ph, [p for p in params if p.kind in MAGNETIC_KINDS])
        params = [p for p in params if p.kind == "G"]

        mixing = [i for i, s in enumerate(subl) if len(s) > 1]
        for arr in product(*subl):
            if not any(p.order == 0 and p.matches(tuple((c,) for c in arr)) for p in params):
                # Not applicable, rather than a malformed database. A multicomponent
                # file declares phases parameterized only for the subsystems they
                # were assessed in: COST507's ALLI passes the constituent test for
                # Al-Mg and then has no G for AL:MG. As a bare ValueError this
                # escaped `phase_models`, which skips only PhaseNotApplicable, and
                # lost the whole solve -- while `applicability`, which does catch
                # ValueError, had already reported the phase unusable. An explicit
                # `phases=` list still raises, so asking for one by name still fails.
                raise PhaseNotApplicable(f"{key}: no G parameter for endmember {':'.join(arr)}")
        if magnetic is not None:
            return self._cef_phase(key, ph, subl, params, elements, magnetic)
        if len(elements) == 2 and len(mixing) == 1 and "VA" not in subl[mixing[0]]:
            model = self._binary_solution(key, ph, subl, params, elements, mixing[0])
            if model is not None:
                return model
        if mixing:
            return self._cef_phase(key, ph, subl, params, elements)

        fixed = [0.0] * len(elements)
        for i, s in enumerate(subl):
            if s[0] != "VA":
                fixed[elements.index(s[0])] += ph.site_ratios[i]
        unused = [p.label for p in params if not all(len(c) == 1 for c in p.constituents)]
        if unused:
            raise UnsupportedModelError(f"{key}: interaction parameters on a phase without mixing: {unused}")
        return StoichiometricPhase(key, elements, tuple(fixed), g=DatabaseValue(self, tuple((1.0, p) for p in params)))

    def disordered_part(self, phase: str) -> tuple[str, bool] | None:
        """``(disordered phase, never_disorder)`` from a DISORDERED_PART or NEVER_DISORDER amendment, or None.

        As in pycalphad, the amendment applies to the phase named on the line,
        and keywords may be abbreviated per underscore-separated part
        (``DIS_PART``, ``NEVER_DIS``).
        """
        for a in self.amendments(phase):
            toks = a.replace(",", " ").split()
            if len(toks) < 2:
                continue
            for word, never in (("DISORDERED_PART", False), ("NEVER_DISORDER", True)):
                parts, full = toks[0].split("_"), word.split("_")
                if len(parts) == len(full) and all(f.startswith(q) and q for q, f in zip(parts, full)):
                    return toks[1].upper(), never
        return None

    def _partitioned_phase(self, key, ph, elements, dis_key, never) -> PartitionedPhase:
        """The ordered phase ``key`` with disordered part ``dis_key`` (see :mod:`discopt.calphad.models.partitioned`).

        Follows pycalphad: the ordered sublattices other than the interstitial
        one are extended to the disordered part's substitutional constituents,
        those with exactly these constituents order, and parameters missing from
        the ordered part are zero (its endmember G are ordering energies).
        """
        if dis_key not in self.phases:
            raise ValueError(f"{key}: disordered part {dis_key} is not defined")
        dph = self.phases[dis_key]
        allowed = set(elements) | {"VA"}
        dsub = [tuple(sorted(c for c in s if c in allowed)) for s in dph.constituents]
        if any(not s for s in dsub):
            raise PhaseNotApplicable(f"{key}: its disordered part {dis_key} has an empty sublattice")
        inter = set(dsub[1]) if len(dsub) == 2 else None
        if len(dsub) > 2:
            raise UnsupportedModelError(f"{key}: a disordered part with more than one interstitial sublattice")
        subl = []
        for s in ph.constituents:
            c = {x for x in s if x in allowed}
            if inter is None or c != inter:
                c |= set(dsub[0])
            subl.append(tuple(sorted(c)))
        subst = tuple(i for i, c in enumerate(subl) if c == dsub[0])

        def params_of(name, cons):
            return [
                p
                for p in self.phase_parameters(name)
                if len(p.constituents) == len(cons)
                and all(c == "*" or c in allowed for s in p.constituents for c in s)
            ]

        dparams = params_of(dis_key, dsub)
        for arr in product(*dsub):
            if not any(p.order == 0 and p.kind == "G" and p.matches(tuple((c,) for c in arr)) for p in dparams):
                raise PhaseNotApplicable(f"{key}: no G parameter for endmember {':'.join(arr)} of {dis_key}")
        oparams = _symmetric_expansion(params_of(key, subl), ph.hints)
        kinds = {p.kind for p in oparams + dparams} - {"G"} - set(MAGNETIC_KINDS)
        if kinds:
            raise UnsupportedModelError(f"{key}: parameter kinds {sorted(kinds)} are not supported yet")
        omag = [p for p in oparams if p.kind in MAGNETIC_KINDS]
        if omag and self.magnetic_type(key) is not None:
            raise UnsupportedModelError(
                f"{key}: magnetic parameters on the ordered part ({omag[0].label}); pycalphad and Thermo-Calc "
                "combine them with the disordered part's differently, so they are not supported yet"
            )
        dmag = self._magnetic(dis_key, dph, [p for p in dparams if p.kind in MAGNETIC_KINDS])
        try:
            ordered = self._cef_phase(key, ph, subl, [p for p in oparams if p.kind == "G"], elements)
            disordered = self._cef_phase(dis_key, dph, dsub, [p for p in dparams if p.kind == "G"], elements, dmag)
            return PartitionedPhase(key, ordered, disordered, subst, never)
        except UnsupportedModelError:
            raise
        except NotImplementedError as err:
            raise UnsupportedModelError(str(err)) from err
        except ValueError as err:
            raise UnsupportedModelError(f"{key}: {err}") from err

    def magnetic_type(self, phase: str) -> tuple[float, float] | None:
        """``(afm, p)`` from the MAGNETIC TYPE_DEFINITION that applies to the phase, or None.

        As in pycalphad, a TYPE_DEFINITION applies through the phase's type-code
        characters, not through the phase name written on the line. For example,
        Al-Fe_sundman2009.tdb writes ``A_P_D @ MAGNETIC``. When several apply, the
        last one wins.
        """
        codes = self.phases[phase.upper()].typecodes
        out = None
        for body in self.type_definitions:
            toks = body.replace(",", " ").split()
            # <char> GES A_P_D <phase> MAGNETIC <afm> <p>
            if len(toks) >= 7 and toks[0] in codes and len(toks[4]) >= 3 and "MAGNETIC".startswith(toks[4].upper()):
                out = (float(toks[5]), float(toks[6]))
        return out

    def _magnetic(self, key, ph, params) -> Magnetic | None:
        """The phase's magnetic description. None if it has no MAGNETIC type or no TC/BMAGN parameters.

        Parameters without a MAGNETIC type definition are ignored, as pycalphad
        ignores them.
        """
        hint = self.magnetic_type(key)
        if hint is None or not params:
            return None
        afm, p = hint
        if afm == 0:
            raise UnsupportedModelError(f"{key}: the Xiong magnetic model (AFM factor 0) is not supported yet")

        def terms(kinds):
            return tuple(
                CEFTerm(q.constituents, q.order, DatabaseValue(self, ((1.0, q),))) for q in params if q.kind in kinds
            )

        return Magnetic(afm=afm, p=p, tc=terms(("TC",)), beta=terms(("BMAGN", "BM")))

    def _cef_phase(self, key, ph, subl, params, elements, magnetic: Magnetic | None = None) -> CEFPhase:
        """Every parameter becomes one CEF term (see :mod:`discopt.calphad.models.cef`)."""
        terms = [CEFTerm(p.constituents, p.order, DatabaseValue(self, ((1.0, p),))) for p in params]
        # pycalphad/Thermo-Calc convention: a lone ternary L0 implies L1 = L2 = L0
        for p in params:
            if p.order == 0 and max(len(c) for c in p.constituents) == 3:
                if not any(q is not p and q.constituents == p.constituents for q in params):
                    terms += [CEFTerm(p.constituents, k, DatabaseValue(self, ((1.0, p),))) for k in (1, 2)]
        try:
            return CEFPhase(key, elements, tuple(ph.site_ratios), tuple(subl), tuple(terms), magnetic)
        except NotImplementedError as err:
            raise UnsupportedModelError(str(err)) from err

    def _binary_solution(self, key, ph, subl, params, elements, i) -> BinarySolution | None:
        """The phase as a :class:`BinarySolution`, or None if some parameter does not fit that form."""
        fixed = [0.0, 0.0]
        for j, s in enumerate(subl):
            if j != i and s[0] != "VA":
                fixed[elements.index(s[0])] += ph.site_ratios[j]

        used: set[int] = set()

        def value(array, scale=1.0, required=True) -> DatabaseValue:
            terms = [p for p in params if p.order == 0 and p.matches(array)]
            if required and not terms:
                label = ":".join(",".join(s) for s in array)
                # Same reclassification as the endmember sweep above.
                raise PhaseNotApplicable(f"{key}: no G parameter for endmember {label}")
            used.update(id(p) for p in terms)
            return DatabaseValue(self, tuple((scale, p) for p in terms))

        ends = [tuple((e,) if j == i else s for j, s in enumerate(subl)) for e in elements]
        g = tuple(value(arr) for arr in ends)
        pair = subl[i]  # sorted, as stored in parameters
        flip = pair != elements
        interactions = [p for p in params if p.matches(tuple(pair if j == i else s for j, s in enumerate(subl)))]
        max_k = max((p.order for p in interactions), default=-1)
        L = []
        for k in range(max_k + 1):
            terms = [p for p in interactions if p.order == k]
            used.update(id(p) for p in terms)
            # RK uses (y_first - y_second) in the sorted order; odd orders change sign when elements are reversed
            sign = -1.0 if (flip and k % 2) else 1.0
            L.append(DatabaseValue(self, tuple((sign, p) for p in terms)))
        if any(id(p) not in used for p in params):
            return None  # e.g. a wildcard on the mixing sublattice: the CEF form handles it
        return BinarySolution(key, elements, g=g, L=tuple(L), sites=ph.site_ratios[i], fixed=tuple(fixed))

    def phase_models(self, elements, phases=None) -> dict:
        """Phase models for an element system.

        With ``phases=None``, every phase that is applicable and supported is
        included, and the rest are skipped (see :meth:`applicability`). With an
        explicit list, a phase that cannot be built raises.
        """
        if phases is not None:
            return {p.upper(): self.phase_model(p, elements) for p in phases}
        out = {}
        for name in self.phases:
            try:
                out[name] = self.phase_model(name, elements)
            except (PhaseNotApplicable, UnsupportedModelError):
                pass
        return out

    def applicability(self, elements) -> dict[str, str]:
        """For each phase: ``"ok"`` or the reason it cannot be used in the binary system."""
        out = {}
        for name in self.phases:
            try:
                self.phase_model(name, elements)
                out[name] = "ok"
            except (PhaseNotApplicable, UnsupportedModelError, ValueError, KeyError) as err:
                out[name] = f"{type(err).__name__}: {err}"
        return out
