"""Fit CALPHAD parameters to experimental data.

Standard CALPHAD parameters enter the Gibbs energy *linearly*. For fixed site
fractions and temperature,

    G(y, T; θ) = RT·g_conf(y) + Σ_p v_p(T; θ)·φ_p(y),

where φ_p(y) is the site-fraction factor of parameter p and v_p(T) is its value.
A free parameter has a symbolic form that is linear in its coefficients, for
example ``"a + b*T"``. Every derived quantity used as data is therefore linear
in the coefficients θ:

- H = G − T ∂G/∂T, S = −∂G/∂T and Cp = −T ∂²G/∂T², with the T-derivatives
  taken symbolically (SymPy) for both fixed and free parameters;
- mixing quantities, relative to the phase's own endmembers;
- formation enthalpies (``HM_FORM``), relative to the pure elements in their
  reference phases (the ELEMENT lines of the database, normally SER);
- chemical potentials μ from the tangent at a given composition, which gives
  activities, and the zero-phase-fraction conditions of a measured tie-line or
  invariant (μ equal in every listed phase, and every listed compound on the
  common tangent).

The fit is a weighted *linear* least-squares problem, optionally with bounds.
It is solved exactly, and its covariance follows directly. The factors φ_p are
read from the phase models by probing: every parameter of a phase is set to 0,
then one at a time to 1 J. So this module works for any phase model the
database can build, without re-deriving the model.

A thermochemical observation gives its phase's configuration either as the
mole fraction ``x`` (phases with one site fraction or none) or as explicit
``site_fractions``, one dict per sublattice. The second form reaches every
endmember of a multi-sublattice (CEF) phase, which is what calculated data
from :mod:`discopt.calphad.mlip` supplies. Every observation carries a
``source`` label, and :meth:`FitResult.to_tdb` lists the sources it used.

Scope: μ-based data (activities, tie-lines) needs a
binary system and phases with one site fraction or none. Phase-boundary points
without a measured partner composition, and errors in the measured
compositions, need equilibrium solves inside the fit and are not handled yet.
:meth:`FitResult.check_equilibria` runs certified equilibria at the tie-line
data after the fit.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import lsq_linear

from discopt.calphad.database import Parameter, ThermoDatabase
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.ops import NUM
from discopt.calphad.symbolic import P, PiecewiseT, T, function_ref, function_refs

__all__ = ["Observation", "Dataset", "FreeParameter", "FitResult", "fit", "THERMO_KINDS"]

#: thermochemical data kinds, in J (or J/K) per mole of atoms
THERMO_KINDS = ("GM", "HM", "SM", "CPM", "GM_MIX", "HM_MIX", "SM_MIX", "CPM_MIX", "HM_FORM")
_PROP = {"GM": "G", "HM": "H", "SM": "S", "CPM": "CP"}
#: at T = 1/R a coefficient of 1 J contributes exactly 1 to G/RT
_T_UNIT = 1.0 / R_GAS
_CONST0 = PiecewiseT((1e-3, 1e6), (sp.Integer(0),))
_CONST1 = PiecewiseT((1e-3, 1e6), (sp.Integer(1),))
_ENDPOINT = 1e-14


# ---------------------------------------------------------------------------- data


def _broadcast(*args):
    return [a.ravel() for a in np.broadcast_arrays(*args)]


@dataclass
class Observation:
    """One data point.

    ``kind`` is one of :data:`THERMO_KINDS`, ``"ACR"`` (activity) or ``"ZPF"``
    (phases in equilibrium). ``x`` is the mole fraction of the first element of
    the fitted system. A thermochemical observation may give ``site_fractions``
    instead: one dict ``{constituent: fraction}`` per sublattice, where missing
    constituents are 0. ``sigma`` is the standard uncertainty, in the units of
    ``value``. For ZPF data it is in J/mol, applied to each μ-equality residual.

    ``HM_FORM`` is the enthalpy of formation per mole of atoms, relative to the
    pure elements in their reference phases: the database's ELEMENT lines, or
    ``reference`` (one phase for every element) when it is given.
    ``source`` names where the value came from (``"experiment"``, ``"dft"``,
    ``"EMT"``, ...). ``note`` records an approximation made in producing it.
    """

    kind: str
    T: float
    value: float = 0.0
    sigma: float = 1.0
    phase: str | None = None
    x: float | None = None
    element: str | None = None
    #: reference phase for an activity (default: the same phase)
    reference: str | None = None
    #: ZPF only: ((phase, x or None), ...); a phase may appear twice (miscibility gap)
    phases: tuple = ()
    citation: str = ""
    #: per sublattice, {constituent: site fraction}; an alternative to ``x``
    site_fractions: tuple | None = None
    #: where the value came from, e.g. "experiment", "dft", "EMT", "uma-s-1p1/omat"
    source: str = ""
    #: an approximation made in producing the value
    note: str = ""

    def __post_init__(self):
        self.kind = self.kind.upper()
        if self.kind not in (*THERMO_KINDS, "ACR", "ZPF"):
            raise ValueError(f"unknown data kind {self.kind!r}; use one of {THERMO_KINDS}, ACR, ZPF")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.kind == "ACR" and self.value <= 0:
            raise ValueError("an activity must be positive")
        self.phases = tuple((str(n).upper(), None if xx is None else float(xx)) for n, xx in self.phases)
        if self.phase is not None:
            self.phase = self.phase.upper()
        if self.site_fractions is not None:
            sf = []
            for sub in self.site_fractions:
                d = {str(c).upper(): float(v) for c, v in dict(sub).items()}
                if any(v < 0 for v in d.values()) or not math.isclose(sum(d.values()), 1.0, abs_tol=1e-9):
                    raise ValueError(f"site fractions {d} must be nonnegative and sum to 1")
                sf.append(d)
            self.site_fractions = tuple(sf)
        if self.kind in THERMO_KINDS:
            if self.phase is None:
                raise ValueError(f"{self.kind}: a phase is required")
            if self.x is None and self.site_fractions is None:
                raise ValueError(f"{self.kind}: give x or site_fractions")


class Dataset:
    """A list of :class:`Observation`, with builders and JSON/CSV readers."""

    def __init__(self, observations=()):
        self.observations: list[Observation] = list(observations)

    def __len__(self):
        return len(self.observations)

    def __iter__(self):
        return iter(self.observations)

    def add(self, obs: Observation) -> Dataset:
        self.observations.append(obs)
        return self

    def add_thermo(self, kind, phase, T, x, value, sigma, citation: str = "") -> Dataset:
        """Thermochemical data; ``T``, ``x``, ``value`` and ``sigma`` broadcast against each other."""
        for t, xx, v, s in zip(*_broadcast(T, x, value, sigma)):
            self.add(Observation(kind, float(t), float(v), float(s), phase=phase, x=float(xx), citation=citation))
        return self

    def add_activity(self, element, phase, T, x, value, sigma, reference=None, citation: str = "") -> Dataset:
        """Activities of ``element`` relative to the pure element in ``reference`` (default: ``phase``)."""
        for t, xx, v, s in zip(*_broadcast(T, x, value, sigma)):
            self.add(
                Observation(
                    "ACR",
                    float(t),
                    float(v),
                    float(s),
                    phase=phase,
                    x=float(xx),
                    element=element.upper(),
                    reference=reference,
                    citation=citation,
                )
            )
        return self

    def add_formation(
        self,
        phase,
        value,
        sigma,
        *,
        endmember=None,
        site_fractions=None,
        x=None,
        T: float = 298.15,
        reference=None,
        source: str = "",
        note: str = "",
        citation: str = "",
    ) -> Dataset:
        """An enthalpy of formation (``HM_FORM``, J/mol of atoms) of one configuration of ``phase``.

        The configuration is an ``endmember`` (``"AL:CU"``, one constituent per
        sublattice), explicit ``site_fractions``, or ``x`` for a phase with one
        site fraction or none.
        """
        if endmember is not None:
            parts = endmember.split(":") if isinstance(endmember, str) else list(endmember)
            site_fractions = tuple({c.strip(): 1.0} for c in parts)
        return self.add(
            Observation(
                "HM_FORM",
                float(T),
                float(value),
                float(sigma),
                phase=phase,
                x=None if x is None else float(x),
                reference=reference,
                site_fractions=site_fractions,
                source=source,
                note=note,
                citation=citation,
            )
        )

    def sources(self) -> dict[str, int]:
        """{source label: number of observations}; an empty label is reported as ``"unspecified"``."""
        out: dict[str, int] = {}
        for o in self.observations:
            out[o.source or "unspecified"] = out.get(o.source or "unspecified", 0) + 1
        return out

    def add_equilibrium(self, T, phases, sigma: float = 500.0, citation: str = "") -> Dataset:
        """Coexisting phases at ``T``: a tie-line, or an invariant with three phases.

        ``phases`` is a dict ``{name: x}`` or a sequence of ``(name, x)`` pairs.
        Use pairs when a phase appears twice. A compound's ``x`` may be None.
        """
        pairs = list(phases.items()) if isinstance(phases, dict) else list(phases)
        if len(pairs) < 2:
            raise ValueError("an equilibrium needs at least two phases")
        return self.add(Observation("ZPF", float(T), 0.0, float(sigma), phases=tuple(pairs), citation=citation))

    # ------------------------------------------------------------------ I/O

    def to_records(self) -> list[dict]:
        out = []
        for o in self.observations:
            d = {k: v for k, v in dataclasses.asdict(o).items() if v not in (None, "", ())}
            if o.phases:
                d["phases"] = [list(p) for p in o.phases]
            if o.site_fractions is not None:
                d["site_fractions"] = [dict(s) for s in o.site_fractions]
            out.append(d)
        return out

    @classmethod
    def from_records(cls, records) -> Dataset:
        return cls(Observation(**{**r, "phases": tuple(tuple(p) for p in r.get("phases", ()))}) for r in records)

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_records(), indent=1))

    @classmethod
    def from_json(cls, path) -> Dataset:
        return cls.from_records(json.loads(Path(path).read_text()))

    @classmethod
    def from_csv(cls, path) -> Dataset:
        """Thermochemical or activity rows.

        Columns: kind, phase, T, x, value, sigma, and optionally element, reference, citation.
        """
        import csv

        ds = cls()
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                row = {k.strip(): v.strip() for k, v in row.items() if v is not None and v.strip() != ""}
                for k in ("T", "x", "value", "sigma"):
                    if k in row:
                        row[k] = float(row[k])
                ds.add(Observation(**row))
        return ds


# ---------------------------------------------------------------------------- free parameters


_LABEL = re.compile(r"^\s*([A-Za-z]+)\s*\(\s*([^,;]+?)\s*,\s*(.+?)\s*;\s*(\d+)\s*\)\s*$")


@dataclass(frozen=True)
class FreeParameter:
    """A database parameter whose value is ``form``: linear in ``coefficients``, and any function of T."""

    label: str
    form: sp.Expr
    coefficients: tuple[sp.Symbol, ...]
    offset: sp.Expr
    basis: tuple[sp.Expr, ...]

    @classmethod
    def parse(cls, label: str, form="a + b*T", functions=()) -> FreeParameter:
        """``functions`` are database FUNCTION names the form may use, e.g. ``"a + b*T + 2*GHSERAL + GHSERCU"``.

        A trailing ``#``, as in TDB syntax, is allowed.
        """
        if isinstance(form, str):
            names = {f.upper(): function_ref(f) for f in functions}
            text = re.sub(r"([A-Za-z_][A-Za-z0-9_]*)#", r"\1", form)
            local = {"T": T, "ln": sp.log, "LN": sp.log, **names}
            e = sp.sympify(text, locals=local)
        else:
            e = sp.sympify(form)
        coefs = tuple(sorted((s for s in e.free_symbols if s != T), key=lambda s: s.name))
        if not coefs:
            raise ValueError(f"{label}: the form {form!r} has no coefficients")
        basis = tuple(sp.diff(e, c) for c in coefs)
        if any(b.free_symbols - {T} for b in basis):
            raise ValueError(f"{label}: the form {form!r} must be linear in its coefficients")
        return cls(label, e, coefs, e.subs({c: 0 for c in coefs}), basis)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f"{self.label}.{c.name}" for c in self.coefficients)


def _parse_label(label: str):
    m = _LABEL.match(label)
    if not m:
        raise ValueError(f"cannot read parameter label {label!r}; expected e.g. 'L(LIQUID,AL,ZN;0)'")
    kind, phase, array, order = m.groups()
    kind = "G" if kind.upper() == "L" else kind.upper()
    subl = tuple(tuple(sorted(c.strip().upper() for c in s.split(","))) for s in array.split(":"))
    return kind, phase.upper(), subl, int(order)


def _replace(db: ThermoDatabase, values: dict[int, PiecewiseT]) -> ThermoDatabase:
    """A copy of ``db`` with the functions of the parameters at the given indices replaced."""
    params = [dataclasses.replace(p, function=values[i]) if i in values else p for i, p in enumerate(db.parameters)]
    return dataclasses.replace(db, parameters=params)


# ---------------------------------------------------------------------------- linear model


class _LinearModel:
    """Every data quantity as a vector [constant, coefficient row] that is linear in θ."""

    def __init__(self, db: ThermoDatabase, elements, free: list[FreeParameter]):
        params = list(db.parameters)
        self.free_index: dict[int, FreeParameter] = {}
        for fp in free:
            kind, phase, subl, order = _parse_label(fp.label)
            hit = [
                i
                for i, p in enumerate(params)
                if (p.kind, p.phase, p.constituents, p.order) == (kind, phase, subl, order)
            ]
            if not hit:
                if phase not in db.phases:
                    raise KeyError(f"{fp.label}: no phase {phase} in the database")
                params.append(Parameter(kind, phase, subl, order, PiecewiseT.constant(0)))  # a new parameter
                hit = [len(params) - 1]
            if hit[0] in self.free_index:
                raise ValueError(f"{fp.label} is listed twice")
            self.free_index[hit[0]] = fp
        self.db = dataclasses.replace(db, parameters=params)
        self.elements = tuple(e.upper() for e in elements)
        self.free = list(free)
        self.names = [n for fp in self.free for n in fp.names]
        self.n = len(self.names)
        self._col = {}
        for fp in self.free:
            for c in fp.coefficients:
                self._col[(fp.label, c)] = 1 + len(self._col)
        self._phases: dict[str, _PhaseBasis] = {}
        self._fixed_fn: dict = {}
        self._free_fn: dict = {}

    def phase(self, name: str) -> _PhaseBasis:
        name = name.upper()
        if name not in self._phases:
            self._phases[name] = _PhaseBasis(self, name)
        return self._phases[name]

    def _derivs(self, i: int, t: float) -> np.ndarray:
        """(value, ∂/∂T, ∂²/∂T²) of parameter ``i`` at ``t``, as a 3 × (n+1) array."""
        out = np.zeros((3, self.n + 1))
        fp = self.free_index.get(i)
        if fp is None:
            p = self.db.parameters[i]
            expr = p.function.select(t, extrapolate=self.db.extrapolate)
            fn = self._fixed_fn.get((i, expr))
            if fn is None:
                e = self._inline(expr)
                d1 = sp.diff(e, T)
                fn = sp.lambdify(T, [d1, sp.diff(d1, T)], "math")
                self._fixed_fn[(i, expr)] = fn
            d1, d2 = fn(t)
            out[:, 0] = (self.db.evaluate(p.function, t), d1, d2)
            return out
        fn = self._free_fn.get(i)
        if fn is None:
            exprs = [self._inline(fp.offset), *fp.basis]
            fn = sp.lambdify(T, [[e, sp.diff(e, T), sp.diff(e, T, 2)] for e in exprs], "math")
            self._free_fn[i] = fn
        vals = fn(t)
        out[:, 0] = vals[0]
        for c, v in zip(fp.coefficients, vals[1:]):
            out[:, self._col[(fp.label, c)]] = v
        return out

    def _inline(self, expr):
        refs = function_refs(expr)
        if refs:
            expr = expr.xreplace({function_ref(n): self.db.expand(n) for n in refs})
        return expr.subs(P, self.db.pressure)

    def prop(self, i: int, prop: str, t: float) -> np.ndarray:
        v, d1, d2 = self._derivs(i, t)
        return {"G": v, "H": v - t * d1, "S": -d1, "CP": -t * d2}[prop]


class _PhaseBasis:
    def __init__(self, lm: _LinearModel, name: str):
        db, el = lm.db, lm.elements
        self.lm, self.name = lm, name
        self.model = db.phase_model(name, el)
        if getattr(self.model, "magnetic", None) is not None:
            # The probe below reads each parameter's coefficient in G. That works
            # only because G is linear in every parameter, and TC and BMAGN enter
            # through ln(β + 1) f(T/T_c).
            raise NotImplementedError(f"{name}: fitting data on a magnetic phase is not supported yet")
        allowed = set(el) | {"VA"}
        #: constituents per sublattice within the fitted elements, as the phase model sees them
        self.sublattices = [tuple(sorted(c for c in s if c in allowed)) for s in db.phases[name].constituents]
        self.params = [i for i, p in enumerate(db.parameters) if p.phase == name]
        zero = {i: _CONST0 for i in self.params}
        self.model0 = _replace(db, zero).phase_model(name, el)
        self.probes = [_replace(db, {**zero, i: _CONST1}).phase_model(name, el) for i in self.params]

    def F(self, prop: str, y, t: float) -> np.ndarray:
        """``prop`` (G, H, S or CP) per formula unit at site fractions ``y``, in J (J/K for S and CP)."""
        g0 = float(self.model0.gibbs(y, NUM, _T_UNIT))
        out = np.zeros(self.lm.n + 1)
        for i, probe in zip(self.params, self.probes):
            phi = float(probe.gibbs(y, NUM, _T_UNIT)) - g0
            if phi != 0.0:
                out += phi * self.lm.prop(i, prop, t)
        out[0] += {"G": R_GAS * t * g0, "S": -R_GAS * g0}.get(prop, 0.0)
        return out

    def y_of(self, x: float) -> np.ndarray:
        if self.model.n_y == 0:
            return np.empty(0)
        if self.model.n_y != 1:
            raise NotImplementedError(f"{self.name}: fitting data needs a phase with at most one site fraction")
        y = self.model.initial_y((x, 1.0 - x))
        if y is None:
            raise ValueError(f"{self.name}: x = {x:g} is outside the phase's composition range")
        return np.asarray(y, float)

    def y_from_site_fractions(self, fractions) -> np.ndarray:
        """Model variables for site fractions given per sublattice as ``{constituent: fraction}``."""
        if len(fractions) != len(self.sublattices):
            raise ValueError(
                f"{self.name}: {len(fractions)} sublattices of site fractions given, the phase has "
                f"{len(self.sublattices)} {self.sublattices}"
            )
        full = []
        for cons, d in zip(self.sublattices, fractions):
            stray = {c for c, v in d.items() if v > 0 and c not in cons}
            if stray:
                raise ValueError(f"{self.name}: {sorted(stray)} cannot occupy a sublattice of {cons}")
            full.append({c: float(d.get(c, 0.0)) for c in cons})
        m = self.model
        if m.n_y == 0:
            return np.empty(0)
        if hasattr(m, "variables_from_site_fractions"):
            return np.asarray(m.variables_from_site_fractions(full), float)
        mixing = [i for i, c in enumerate(self.sublattices) if len(c) > 1]
        if len(mixing) != 1:
            raise ValueError(f"{self.name}: expected one mixing sublattice, found {len(mixing)}")
        return np.array([full[mixing[0]][self.lm.elements[0]]])

    def atoms(self, y) -> float:
        return float(sum(self.model.amounts(y)))

    def thermo(self, kind: str, x: float | None, t: float, y=None) -> np.ndarray:
        if kind == "HM_FORM":
            raise ValueError("HM_FORM needs reference phases: use formation()")
        prop = _PROP[kind.removesuffix("_MIX")]
        y = self.y_of(x) if y is None else np.asarray(y, float)
        F = self.F(prop, y, t)
        if kind.endswith("_MIX"):
            if self.model.n_y != 1:
                raise ValueError(f"{kind}: mixing quantities need a phase with one site fraction")
            y1 = float(y[0])
            F = F - y1 * self.F(prop, [1.0 - _ENDPOINT], t) - (1 - y1) * self.F(prop, [_ENDPOINT], t)
        return F / self.atoms(y)

    def mu(self, x: float, t: float) -> np.ndarray:
        """Chemical potentials (J/mol) of the two elements on the tangent at ``x``: a 2 × (n+1) array."""
        if len(self.lm.elements) != 2 or self.model.n_y != 1:
            raise NotImplementedError(f"{self.name}: μ data needs a binary phase with one site fraction")
        y = float(self.y_of(x)[0])
        h = min(1e-4, y / 50.0, (1.0 - y) / 50.0)
        Fs = [self.F("G", [y + k * h], t) for k in (-2, -1, 1, 2)]
        dF = (Fs[0] - 8 * Fs[1] + 8 * Fs[2] - Fs[3]) / (12 * h)
        N = np.array(self.model.amounts([y]), float)
        dN = (np.array(self.model.amounts([y + h]), float) - np.array(self.model.amounts([y - h]), float)) / (2 * h)
        return np.linalg.solve(np.array([N, dN]), np.array([self.F("G", [y], t), dF]))

    def pure_mu(self, element: str, t: float, prop: str = "G") -> np.ndarray:
        """μ of the pure element in this phase, per mole of atoms (``prop="H"``: its enthalpy)."""
        k = self.lm.elements.index(element)
        if self.model.n_y == 0:
            candidates = [np.empty(0)]
        elif self.model.n_y == 1:
            candidates = [np.array([1.0 - _ENDPOINT]), np.array([_ENDPOINT])]
        elif all(element in c or "VA" in c for c in self.sublattices):
            candidates = [
                self.y_from_site_fractions([{element if element in c else "VA": 1.0} for c in self.sublattices])
            ]
        else:
            candidates = []
        for y in candidates:
            N = np.array(self.model.amounts(y), float)
            if N[k] > 0 and np.all(np.delete(N, k) < 1e-9):
                return self.F(prop, y, t) / N[k]
        raise ValueError(f"{self.name} has no pure-{element} endmember to use as a reference")

    def formation(self, y, t: float, reference: str | None = None) -> np.ndarray:
        """Enthalpy of formation per mole of atoms at ``y``, relative to the elements' reference phases."""
        N = np.array(self.model.amounts(y), float)
        out = self.F("H", y, t) / N.sum()
        for e, n in zip(self.lm.elements, N):
            if n > 1e-12:
                ref = reference or self.lm.db.elements[e].reference_phase
                if not ref:
                    raise ValueError(f"{e} has no reference phase in the database; pass reference=")
                out = out - (n / N.sum()) * self.lm.phase(ref).pure_mu(e, t, prop="H")
        return out


# ---------------------------------------------------------------------------- rows


def _rows(lm: _LinearModel, obs: Observation):
    """(vectors, observed values, sigmas) for one observation."""
    if obs.kind in THERMO_KINDS:
        ph = lm.phase(obs.phase)
        y = None if obs.site_fractions is None else ph.y_from_site_fractions(obs.site_fractions)
        if obs.kind == "HM_FORM":
            vec = ph.formation(ph.y_of(obs.x) if y is None else y, obs.T, obs.reference)
        else:
            vec = ph.thermo(obs.kind, obs.x, obs.T, y)
        return [vec], [obs.value], [obs.sigma]
    if obs.kind == "ACR":
        k = lm.elements.index(obs.element)
        mu = lm.phase(obs.phase).mu(obs.x, obs.T)[k]
        ref = lm.phase(obs.reference or obs.phase).pure_mu(obs.element, obs.T)
        return [(mu - ref) / (R_GAS * obs.T)], [math.log(obs.value)], [obs.sigma / obs.value]
    # ZPF: μ equal across solution phases; each compound on the tangent
    sols = [(lm.phase(n), x) for n, x in obs.phases if lm.phase(n).model.n_y > 0]
    comps = [lm.phase(n) for n, _ in obs.phases if lm.phase(n).model.n_y == 0]
    if not sols:
        raise ValueError("a ZPF observation needs at least one solution phase to fix μ")
    for ph, x in sols:
        if x is None:
            raise ValueError(f"ZPF: composition of {ph.name} is missing")
    ref = sols[0][0].mu(sols[0][1], obs.T)
    vecs = []
    for ph, x in sols[1:]:
        vecs.extend(ref - ph.mu(x, obs.T))
    for ph in comps:
        N = np.array(ph.model.amounts(), float)
        vecs.append((N @ ref - ph.F("G", np.empty(0), obs.T)) / N.sum())
    return vecs, [0.0] * len(vecs), [obs.sigma] * len(vecs)


# ---------------------------------------------------------------------------- fit


@dataclass
class FitResult:
    names: list[str]
    values: np.ndarray
    std: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    #: weighted residuals (model − data)/σ, one per row
    residuals: np.ndarray
    #: index of the observation behind each residual row
    row_observation: np.ndarray
    chi2: float
    dof: int
    rank: int
    database: ThermoDatabase
    data: Dataset
    elements: tuple[str, ...]
    free: list[FreeParameter] = field(default_factory=list)
    active_bounds: list[str] = field(default_factory=list)

    @property
    def parameters(self) -> dict[str, float]:
        return dict(zip(self.names, map(float, self.values)))

    def residuals_by_kind(self) -> dict[str, tuple[int, float]]:
        """{kind: (number of rows, RMS weighted residual)}."""
        out = {}
        kinds = np.array([self.data.observations[i].kind for i in self.row_observation])
        for k in dict.fromkeys(kinds):
            r = self.residuals[kinds == k]
            out[str(k)] = (int(r.size), float(np.sqrt(np.mean(r**2))))
        return out

    def summary(self) -> str:
        lines = [
            f"fit of {len(self.names)} coefficients to {len(self.data)} observations "
            f"({self.residuals.size} residuals): chi2 = {self.chi2:.4g}, dof = {self.dof}, rank = {self.rank}"
        ]
        for n, v, s in zip(self.names, self.values, self.std):
            flag = "  (at bound)" if n in self.active_bounds else ""
            lines.append(f"  {n:32s} = {v: .6g}  ± {s:.3g}{flag}")
        for k, (m, rms) in self.residuals_by_kind().items():
            lines.append(f"  {k:8s} {m:4d} rows, RMS weighted residual {rms:.3g}")
        return "\n".join(lines)

    def to_tdb(self) -> str:
        """The fitted database as TDB text, with a provenance header."""
        head = [
            f"$ parameters fitted by discopt.calphad.fit to {len(self.data)} observations "
            f"(chi2 = {self.chi2:.6g}, dof = {self.dof})"
        ]
        head += [f"$   {n} = {v:.10g} +/- {s:.3g}" for n, v, s in zip(self.names, self.values, self.std)]
        head.append("$ data sources: " + ", ".join(f"{k} ({n})" for k, n in self.data.sources().items()))
        notes = sorted({o.note for o in self.data if o.note})
        head += [f"$ approximation: {n}" for n in notes]
        return "\n".join(head) + "\n" + self.database.to_tdb()

    def plot_residuals(self, ax=None):
        import matplotlib.pyplot as plt

        ax = ax or plt.gca()
        kinds = [self.data.observations[i].kind for i in self.row_observation]
        for j, k in enumerate(dict.fromkeys(kinds)):
            idx = [i for i, kk in enumerate(kinds) if kk == k]
            ax.plot(idx, self.residuals[idx], "o", color=f"C{j}", label=k)
        ax.axhline(0, color="k", lw=0.8)
        for s in (-2, 2):
            ax.axhline(s, color="0.6", ls=":", lw=0.8)
        ax.set_xlabel("residual row")
        ax.set_ylabel("(model − data) / σ")
        ax.legend(fontsize=8)
        return ax

    def check_equilibria(self, eps: float = 1e-5, **kwargs) -> list[dict]:
        """A certified equilibrium at each ZPF observation's mean composition, compared with the observed phases."""
        from discopt.calphad.equilibrium import equilibrium

        phases = list(self.database.phase_models(self.elements).values())
        out = []
        for obs in self.data:
            if obs.kind != "ZPF":
                continue
            xs = []
            for name, x in obs.phases:
                if x is None:
                    a = np.array(self.database.phase_model(name, self.elements).amounts(), float)
                    x = a[0] / a.sum()
                xs.append(x)
            xbar = float(np.mean(xs))
            res = equilibrium(phases, x=xbar, T=obs.T, eps=eps, **kwargs)
            observed = sorted(n for n, _ in obs.phases)
            computed = sorted(cs.phase.name for cs in res.composition_sets)
            out.append(
                {
                    "T": obs.T,
                    "x": xbar,
                    "observed": observed,
                    "computed": computed,
                    "match": observed == computed,
                    "certificate": res.certificate.level.name,
                    "result": res,
                }
            )
        return out


def fit(db: ThermoDatabase, data: Dataset, elements, free, *, bounds=None) -> FitResult:
    """Fit free parameter coefficients of ``db`` to ``data`` by weighted linear least squares.

    ``free`` is a list of parameter labels (each gets the form ``a + b*T``), or a
    dict ``{label: form}``, for example ``{"L(FCC_A1,AL,ZN;0)": "a + b*T"}``. A
    label that does not exist yet in the database is added as a new parameter.
    A form may be any expression in T that is linear in its coefficients, and
    it may use the database's FUNCTIONs, as a compound's G usually does:
    ``{"G(ALCU_THETA,AL:CU;0)": "a + b*T + 2*GHSERAL + GHSERCU"}``. A free
    parameter replaces every temperature range of the original.

    ``bounds`` maps coefficient names (``"<label>.<coef>"``, see
    :attr:`FreeParameter.names`) to ``(lo, hi)``.

    The covariance is (JᵀJ)⁻¹, with J the σ-weighted design matrix. It is
    absolute: it assumes the given σ are the true uncertainties.
    """
    free = {label: "a + b*T" for label in free} if not isinstance(free, dict) else free
    fps = [FreeParameter.parse(label, form, functions=db.functions) for label, form in free.items()]
    lm = _LinearModel(db, elements, fps)

    vecs, obs_v, sig, owner = [], [], [], []
    for j, o in enumerate(data):
        v, ov, s = _rows(lm, o)
        vecs += v
        obs_v += ov
        sig += s
        owner += [j] * len(v)
    if not vecs:
        raise ValueError("no data")
    V = np.array(vecs)
    sig = np.array(sig)
    J = V[:, 1:] / sig[:, None]
    r0 = (np.array(obs_v) - V[:, 0]) / sig
    scale = np.linalg.norm(J, axis=0)
    scale[scale == 0] = 1.0
    Js = J / scale
    rank = int(np.linalg.matrix_rank(Js))
    if rank < lm.n:
        warnings.warn(
            f"the data determine only {rank} of {lm.n} coefficients; the solution is not unique", stacklevel=2
        )

    lo = np.full(lm.n, -np.inf)
    hi = np.full(lm.n, np.inf)
    for name, (a, b) in (bounds or {}).items():
        if name not in lm.names:
            raise KeyError(f"bound on unknown coefficient {name!r}; coefficients are {lm.names}")
        k = lm.names.index(name)
        lo[k], hi[k] = (-np.inf if a is None else a), (np.inf if b is None else b)
    if np.isfinite(lo).any() or np.isfinite(hi).any():
        sol = lsq_linear(Js, r0, bounds=(lo * scale, hi * scale), method="bvls", tol=1e-14)
        theta = sol.x / scale
    else:
        theta = np.linalg.lstsq(Js, r0, rcond=None)[0] / scale
    active = [n for n, t, a, b in zip(lm.names, theta, lo, hi) if np.isclose(t, a) or np.isclose(t, b)]

    resid = J @ theta - r0
    cov = np.linalg.pinv(Js.T @ Js) / np.outer(scale, scale)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)

    fitted = {}
    for i, fp in lm.free_index.items():
        subs = {c: sp.Float(float(theta[lm.names.index(n)]), 15) for c, n in zip(fp.coefficients, fp.names)}
        old = lm.db.parameters[i].function
        fitted[i] = PiecewiseT((old.t_min, old.t_max), (sp.nsimplify(fp.form.subs(subs), rational=False),))
    return FitResult(
        names=list(lm.names),
        values=theta,
        std=std,
        covariance=cov,
        correlation=corr,
        residuals=resid,
        row_observation=np.array(owner),
        chi2=float(resid @ resid),
        dof=int(resid.size - rank),
        rank=rank,
        database=_replace(lm.db, fitted),
        data=data,
        elements=lm.elements,
        free=fps,
        active_bounds=active,
    )
