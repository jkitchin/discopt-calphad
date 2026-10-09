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

Two kinds of row depend on θ through a minimization over site fractions:

- μ of a phase with several site fractions (a CEF phase such as a Laves
  phase): its site fractions at a given composition are the internal
  equilibrium, the minimum of G at fixed x;
- a phase-boundary point whose partner composition was not measured (a
  liquidus or solvus point): the unmeasured phase must touch the tangent of the
  measured one, so its driving force, the minimum over composition of
  (G − μ·N)/ΣN, is zero. This is ESPEI's driving-force residual.

At the minimizer the row's θ-derivative is exact (envelope theorem). The fit
then iterates linear solves (Gauss–Newton with step halving) until θ stops
changing. With only linear rows, one solve is exact.

The fit is weighted least squares, optionally with bounds, and its covariance
follows from the final linearization. The factors φ_p are read from the phase
models by probing: every parameter of a phase is set to 0, then one at a time
to 1 J. So this module works for any phase model the database can build,
without re-deriving the model.

A thermochemical observation gives its phase's configuration either as the
mole fraction ``x`` (phases with one site fraction or none) or as explicit
``site_fractions``, one dict per sublattice. The second form reaches every
endmember of a multi-sublattice (CEF) phase, which is what calculated data
from :mod:`discopt.calphad.mlip` supplies. Every observation carries a
``source`` label, and :meth:`FitResult.to_tdb` lists the sources it used.

ESPEI JSON datasets are read by :meth:`Dataset.from_espei`.

Compositions are a float in a binary (the first element's fraction), or, in
any system, a dict by element or a sequence (the last element may be left
out). μ of a phase comes from the gradient of G per mole of atoms; a phase
whose compositions do not span the system (a line phase in a ternary) is put
on the reference tangent instead, and stationary along its own variables. E
compounds with independent compositions fix μ when no composition was
measured. Errors in variables still need a binary. The
driving-force minimum is a local search (sampling, then refinement), so a very
narrow well of an unmeasured phase can be missed. By default each measured
composition is taken as exact, and its uncertainty goes into the J/mol σ of
the row; ``fit(..., errors_in_variables=True)`` instead fits the compositions
too, each with its own ``sigma_x``, which gives calibrated uncertainties.
:meth:`FitResult.check_equilibria` runs certified equilibria at the
phase-equilibrium data after the fit.
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

from discopt.calphad.database import MAGNETIC_KINDS, Parameter, ThermoDatabase
from discopt.calphad.equilibrium.common import R_GAS
from discopt.calphad.ops import NUM
from discopt.calphad.symbolic import P, PiecewiseT, T, function_ref, function_refs, tdb_str

__all__ = ["Observation", "Dataset", "FreeParameter", "FitResult", "fit", "predict", "enthalpy_form", "THERMO_KINDS"]

#: thermochemical data kinds, in J (or J/K) per mole of atoms
THERMO_KINDS = ("GM", "HM", "SM", "CPM", "GM_MIX", "HM_MIX", "SM_MIX", "CPM_MIX", "HM_FORM", "SM_FORM", "CPM_FORM")
_FORM = {"HM_FORM": "H", "SM_FORM": "S", "CPM_FORM": "CP"}
_PROP = {"GM": "G", "HM": "H", "SM": "S", "CPM": "CP"}
#: at T = 1/R a coefficient of 1 J contributes exactly 1 to G/RT
_T_UNIT = 1.0 / R_GAS
_CONST0 = PiecewiseT((1e-3, 1e6), (sp.Integer(0),))
_CONST1 = PiecewiseT((1e-3, 1e6), (sp.Integer(1),))
_ENDPOINT = 1e-14


# ---------------------------------------------------------------------------- data


def _normalize_x(x):
    """A composition as given: None, a float (binary: the first element), a dict by element, or a tuple."""
    if x is None:
        return None
    if isinstance(x, dict):
        return {str(k).upper(): float(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(float(v) for v in x)
    return float(x)


def _composition(elements, x) -> np.ndarray:
    """Mole fractions of every element, in ``elements`` order, from a composition as given.

    A float is the first element's fraction in a binary. A dict or a sequence
    may leave out the last element, which is then 1 minus the rest.
    """
    E = len(elements)
    if x is None:
        raise ValueError("a composition is required")
    if isinstance(x, dict):
        given = {k.upper(): float(v) for k, v in x.items()}
        stray = set(given) - set(elements)
        if stray:
            raise ValueError(f"composition names {sorted(stray)}, not in {elements}")
        missing = [e for e in elements if e not in given]
        if len(missing) > 1:
            raise ValueError(f"composition {given} leaves out {missing}; at most one may be implied")
        v = np.array([given.get(e, np.nan) for e in elements])
        if missing:
            v[elements.index(missing[0])] = 1.0 - np.nansum(v)
    elif isinstance(x, (tuple, list, np.ndarray)):
        v = np.array(x, float)
        if v.size == E - 1:
            v = np.append(v, 1.0 - v.sum())
        if v.size != E:
            raise ValueError(f"composition {tuple(x)} does not match {elements}")
    else:
        if E != 2:
            raise ValueError(f"a single mole fraction is only for binaries; give one per element of {elements}")
        v = np.array([float(x), 1.0 - float(x)])
    if np.any(v < -1e-12) or abs(v.sum() - 1.0) > 1e-9:
        raise ValueError(f"composition {v} must be nonnegative and sum to 1")
    return np.clip(v, 0.0, 1.0)


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
    ``reference`` (one phase for every element) when it is given. ``SM_FORM``
    and ``CPM_FORM`` are the entropy and heat capacity of formation, in
    J/(mol K), on the same terms.
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
    #: ZPF only: standard uncertainty of each measured composition (mole fraction), for
    #: ``fit(..., errors_in_variables=True)``
    sigma_x: float | None = None

    def __post_init__(self):
        self.kind = self.kind.upper()
        if self.kind not in (*THERMO_KINDS, "ACR", "ZPF"):
            raise ValueError(f"unknown data kind {self.kind!r}; use one of {THERMO_KINDS}, ACR, ZPF")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.kind == "ACR" and self.value <= 0:
            raise ValueError("an activity must be positive")
        self.phases = tuple((str(n).upper(), _normalize_x(xx)) for n, xx in self.phases)
        self.x = _normalize_x(self.x)
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
        #: (file, reason) for data a reader did not use (see :meth:`from_espei`)
        self.skipped: list[tuple[str, str]] = []

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
        kind: str = "HM_FORM",
    ) -> Dataset:
        """An enthalpy of formation (``HM_FORM``, J/mol of atoms) of one configuration of ``phase``.

        ``kind="SM_FORM"`` or ``"CPM_FORM"`` gives an entropy or heat capacity of
        formation instead, in J/(mol K).

        The configuration is an ``endmember`` (``"AL:CU"``, one constituent per
        sublattice), explicit ``site_fractions``, or ``x`` for a phase with one
        site fraction or none.
        """
        if endmember is not None:
            parts = endmember.split(":") if isinstance(endmember, str) else list(endmember)
            site_fractions = tuple({c.strip(): 1.0} for c in parts)
        return self.add(
            Observation(
                kind,
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

    def add_equilibrium(
        self, T, phases, sigma: float = 500.0, citation: str = "", sigma_x: float | None = None
    ) -> Dataset:
        """Coexisting phases at ``T``: a tie-line, or an invariant with three phases.

        ``phases`` is a dict ``{name: x}`` or a sequence of ``(name, x)`` pairs.
        Use pairs when a phase appears twice. A compound's ``x`` may be None.
        ``sigma_x`` is the standard uncertainty of each measured composition, used
        by ``fit(..., errors_in_variables=True)``.
        """
        pairs = list(phases.items()) if isinstance(phases, dict) else list(phases)
        if len(pairs) < 2:
            raise ValueError("an equilibrium needs at least two phases")
        if sigma_x is not None and sigma_x <= 0:
            raise ValueError("sigma_x must be positive")
        return self.add(
            Observation("ZPF", float(T), 0.0, float(sigma), phases=tuple(pairs), citation=citation, sigma_x=sigma_x)
        )

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
    def from_espei(cls, paths, db: ThermoDatabase, elements, *, phase_map=None, sigma=None) -> Dataset:
        """ESPEI JSON datasets (files or directories); see :mod:`discopt.calphad.io.espei`.

        Data that cannot be used are listed in :attr:`skipped` with the reason.
        """
        from discopt.calphad.io.espei import read_espei

        ds, skipped = read_espei(paths, db, elements, phase_map=phase_map, sigma=sigma)
        ds.skipped = skipped
        return ds

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
        #: the coefficients at which configuration-dependent rows are evaluated (see fit())
        self.theta = np.zeros(self.n)
        #: set when a row depended on ``theta`` through a minimization over site fractions
        self.nonlinear = False
        self._db_at: dict = {}

    def functions_at(self, theta) -> dict[int, PiecewiseT]:
        """The free parameters' functions with coefficients ``theta``, by parameter index."""
        out = {}
        for i, fp in self.free_index.items():
            subs = {c: sp.Float(float(theta[self.names.index(n)]), 15) for c, n in zip(fp.coefficients, fp.names)}
            old = self.db.parameters[i].function
            out[i] = PiecewiseT((old.t_min, old.t_max), (sp.nsimplify(fp.form.subs(subs), rational=False),))
        return out

    def model_at(self, name: str):
        """The phase model of ``name`` with the free parameters at :attr:`theta`."""
        key = tuple(np.round(self.theta, 12))
        if key not in self._db_at:
            self._db_at = {key: (_replace(self.db, self.functions_at(self.theta)), {})}
        db, models = self._db_at[key]
        if name not in models:
            models[name] = db.phase_model(name, self.elements)
        return models[name]

    def initial_theta(self) -> np.ndarray:
        """Coefficients that best reproduce each parameter's original function over 298.15–3000 K (0 if new)."""
        theta = np.zeros(self.n)
        for i, fp in self.free_index.items():
            fn = self.db.parameters[i].function
            ts = np.linspace(max(fn.t_min, 298.15), min(fn.t_max, 3000.0), 25)
            rows, rhs = [], []
            for t in ts:
                try:
                    v = self.db.evaluate(fn, float(t))
                    d = self._derivs(i, float(t))[0]
                except (ValueError, TypeError):  # outside a range of the parameter or of a FUNCTION it uses
                    continue
                cols = [self._col[(fp.label, c)] for c in fp.coefficients]
                rows.append(d[cols])
                rhs.append(v - d[0])
            if rows:
                sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
                for c, v in zip(fp.coefficients, sol):
                    theta[self._col[(fp.label, c)] - 1] = v
        return theta

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
        mine = [i for i, p in enumerate(db.parameters) if p.phase == name]
        magnetic = [i for i in mine if db.parameters[i].kind in MAGNETIC_KINDS]
        if any(i in lm.free_index for i in magnetic):
            # The probe below reads each parameter's coefficient in G, which works because G
            # is linear in G and L parameters. TC and BMAGN enter through ln(β + 1) f(T/T_c).
            raise NotImplementedError(f"{name}: TC and BMAGN cannot be fitted (G is not linear in them)")
        allowed = set(el) | {"VA"}
        #: constituents per sublattice within the fitted elements, as the phase model sees them
        self.sublattices = [tuple(sorted(c for c in s if c in allowed)) for s in db.phases[name].constituents]
        self.params = [i for i in mine if i not in magnetic]
        # With TC and BMAGN at 0 the magnetic term vanishes (ln(0 + 1) = 0), so the probes see
        # only the linear part. The magnetic term itself is fixed and is added in F().
        zero = {i: _CONST0 for i in mine}
        self.model0 = _replace(db, zero).phase_model(name, el)
        self.probes = [_replace(db, {**zero, i: _CONST1}).phase_model(name, el) for i in self.params]
        self._magnetic = None
        if getattr(self.model, "magnetic", None) is not None:
            self._magnetic = (self.model, _replace(db, {i: _CONST0 for i in magnetic}).phase_model(name, el))

    def F(self, prop: str, y, t: float) -> np.ndarray:
        """``prop`` (G, H, S or CP) per formula unit at site fractions ``y``, in J (J/K for S and CP)."""
        g0 = float(self.model0.gibbs(y, NUM, _T_UNIT))
        out = np.zeros(self.lm.n + 1)
        for i, probe in zip(self.params, self.probes):
            phi = float(probe.gibbs(y, NUM, _T_UNIT)) - g0
            if phi != 0.0:
                out += phi * self.lm.prop(i, prop, t)
        out[0] += {"G": R_GAS * t * g0, "S": -R_GAS * g0}.get(prop, 0.0)
        if self._magnetic is not None:
            out[0] += self._magnetic_part(prop, y, t)
        return out

    def _magnetic_part(self, prop: str, y, t: float) -> float:
        """The fixed magnetic contribution to ``prop`` per formula unit, with T-derivatives by differences."""
        full, plain = self._magnetic

        def gm(tt):
            return (float(full.gibbs(y, NUM, tt)) - float(plain.gibbs(y, NUM, tt))) * R_GAS * tt

        g = gm(t)
        if prop == "G":
            return g
        h = 0.05  # K: the IHJ term is C¹ at T_c, so Cp is only piecewise smooth there
        try:
            lo, hi = gm(t - h), gm(t + h)
            d1, d2 = (hi - lo) / (2 * h), (hi - 2 * g + lo) / (h * h)
        except ValueError:  # at the lower end of a temperature range (298.15 K): one-sided, second order
            g1, g2, g3 = gm(t + h), gm(t + 2 * h), gm(t + 3 * h)
            d1 = (-3 * g + 4 * g1 - g2) / (2 * h)
            d2 = (2 * g - 5 * g1 + 4 * g2 - g3) / (h * h)
        return {"S": -d1, "H": g - t * d1, "CP": -t * d2}[prop]

    def _determined(self) -> bool:
        """Whether a composition fixes the site fractions (no internal degree of freedom)."""
        return self.model.n_y <= len(self.lm.elements) - 1

    def spans(self) -> bool:
        """Whether the phase's compositions span the whole system (not a line or a point in it).

        Only such a phase has a full tangent of its own, and can fix every μ.
        """
        E = len(self.lm.elements)
        if self.model.n_y < E - 1:
            return False
        rng = np.random.default_rng(1)
        u = rng.uniform(0.2, 0.8, self.model.n_y)

        def comp(v):
            a = np.array([float(w) for w in self.model.amounts(v)])
            return a / a.sum()

        J = np.array([(comp(u + 1e-6 * e) - comp(u - 1e-6 * e)) / 2e-6 for e in np.eye(self.model.n_y)])
        return int(np.linalg.matrix_rank(J, tol=1e-8)) == E - 1

    def touch_rows(self, x, mu: np.ndarray, t: float) -> list[np.ndarray]:
        """Rows that put a measured phase whose compositions do not span the system on the tangent.

        At its measured composition it touches the tangent of ``mu``, and the
        tangent distance is stationary along each of its own site fractions.
        """
        y = self.y_of(x) if self._determined() else self.y_at(x, t)
        rows = [self.tangent_row(y, mu, t)]
        for k in range(self.model.n_y):
            h = min(1e-6, y[k] / 2, (1 - y[k]) / 2) or 1e-9
            yp, yn = y.copy(), y.copy()
            yp[k] += h
            yn[k] -= h
            Gp = self.F("G", yp, t) - np.array(self.model.amounts(yp), float) @ mu
            Gn = self.F("G", yn, t) - np.array(self.model.amounts(yn), float) @ mu
            rows.append((Gp - Gn) / (2 * h) / self.atoms(y))
        return rows

    def y_of(self, x) -> np.ndarray:
        """Site fractions at composition ``x`` (see :func:`_composition`), when the composition fixes them."""
        if self.model.n_y == 0:
            return np.empty(0)
        if not self._determined():
            raise NotImplementedError(
                f"{self.name}: the composition does not fix its {self.model.n_y} site fractions; use y_at"
            )
        b = _composition(self.lm.elements, x)
        y = self.model.initial_y(tuple(b))
        if y is None:
            raise ValueError(f"{self.name}: composition {np.round(b, 6)} is outside the phase's composition range")
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

    def thermo(self, kind: str, x: float | None, t: float, y=None, site_fractions=None) -> np.ndarray:
        """A thermochemical quantity per mole of atoms.

        ``_MIX`` is relative to the phase's own endmembers. With one site
        fraction that is the two pure ends. With several, it needs
        ``site_fractions`` in which exactly one sublattice is mixed: the
        reference is then Σ_c y_c · (the endmember with c alone on that
        sublattice, every other sublattice as given).
        """
        if kind in _FORM:
            raise ValueError(f"{kind} needs reference phases: use formation()")
        prop = _PROP[kind.removesuffix("_MIX")]
        y = self.y_of(x) if y is None else np.asarray(y, float)
        F = self.F(prop, y, t)
        if kind.endswith("_MIX"):
            if self.model.n_y == 1:
                y1 = float(y[0])
                F = F - y1 * self.F(prop, [1.0 - _ENDPOINT], t) - (1 - y1) * self.F(prop, [_ENDPOINT], t)
            else:
                if site_fractions is None:
                    raise ValueError(f"{kind} on {self.name}: give site_fractions (it has several site fractions)")
                mixed = [s for s, d in enumerate(site_fractions) if sum(v > 0 for v in d.values()) > 1]
                if len(mixed) != 1:
                    raise ValueError(f"{kind} on {self.name}: needs exactly one mixed sublattice, got {len(mixed)}")
                m = mixed[0]
                for c, v in site_fractions[m].items():
                    if v > 0:
                        end = [dict(d) if s != m else {c: 1.0} for s, d in enumerate(site_fractions)]
                        F = F - v * self.F(prop, self.y_from_site_fractions(end), t)
        return F / self.atoms(y)

    def y_at(self, x, t: float) -> np.ndarray:
        """Site fractions at overall composition ``x``.

        When the composition fixes them (:meth:`_determined`), they follow
        directly. Otherwise they are the internal equilibrium: the minimum of G
        at fixed composition, for the free parameters at
        :attr:`_LinearModel.theta`. That makes rows built from them depend on θ,
        and :func:`fit` iterates.
        """
        if self._determined():
            return self.y_of(x)
        from scipy.optimize import minimize

        self.lm.nonlinear = True
        model = self.lm.model_at(self.name)
        b = _composition(self.lm.elements, x)

        def comp(u):
            a = np.array([float(v) for v in model.amounts(u)])
            return a[:-1] / a.sum() - b[:-1]

        def g(u):
            with np.errstate(all="ignore"):
                v = float(model.gibbs(u, NUM, t))
            return v if math.isfinite(v) else 1e300

        from scipy.optimize import least_squares

        rng = np.random.default_rng(0)
        starts = []
        for u0 in [np.full(model.n_y, 0.5), *rng.uniform(0.02, 0.98, size=(11, model.n_y))]:
            r = least_squares(comp, u0, bounds=(0.0, 1.0), xtol=1e-14, ftol=1e-14, gtol=1e-14)
            if np.max(np.abs(r.fun)) < 1e-9 and all(np.max(np.abs(r.x - s)) > 1e-3 for s in starts):
                starts.append(r.x)
        if not starts:
            raise ValueError(f"{self.name}: composition {np.round(b, 6)} is outside the phase's composition range")
        best, best_g = None, math.inf
        for u0 in starts:
            r = minimize(
                g,
                u0,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * model.n_y,
                constraints=[{"type": "eq", "fun": comp}],
                options={"ftol": 1e-14, "maxiter": 300},
            )
            for u in (np.clip(r.x, 0.0, 1.0), u0):
                if np.max(np.abs(comp(u))) < 1e-8 and g(u) < best_g:
                    best, best_g = np.array(u), g(u)
        return best

    def mu(self, x, t: float) -> np.ndarray:
        """Chemical potentials (J/mol) of every element on the tangent at composition ``x``: an E × (n+1) array.

        In a binary, a phase with one site fraction uses the exact tangent of G
        per formula unit. Otherwise G per mole of atoms, g(x), is evaluated at
        the site fractions of each composition (:meth:`y_at`), and differentiated
        along directions that exchange element i with the last one:
        μ_last = g − Σ_i x_i ∂g/∂x_i and μ_i = μ_last + ∂g/∂x_i. Every g is a
        row linear in θ (for an internal equilibrium, by the envelope theorem),
        so μ is too.
        """
        if self.model.n_y == 0:
            raise NotImplementedError(f"{self.name}: a compound has no tangent of its own")
        E = len(self.lm.elements)
        if not (E == 2 and self.model.n_y == 1):
            b = _composition(self.lm.elements, x)

            def gm(v):
                y = self.y_at(v, t)
                return self.F("G", y, t) / self.atoms(y)

            g0 = gm(b)
            dg = []
            for i in range(E - 1):
                h = min(1e-4, b[i] / 50.0, b[-1] / 50.0)
                if h <= 0:
                    raise ValueError(f"{self.name}: μ at composition {np.round(b, 6)} needs every element present")
                d = np.zeros(E)
                d[i], d[-1] = 1.0, -1.0
                f = [gm(b + k * h * d) for k in (-2, -1, 1, 2)]
                dg.append((f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h))
            mu_last = g0 - sum(b[i] * dg[i] for i in range(E - 1))
            return np.array([mu_last + dg[i] for i in range(E - 1)] + [mu_last])
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

    def tangent_row(self, y, mu: np.ndarray, t: float) -> np.ndarray:
        """(G − μ·N)/ΣN at fixed site fractions ``y``, per mole of atoms, as a row linear in θ."""
        N = np.array(self.model.amounts(y), float)
        return (self.F("G", y, t) - N @ mu) / N.sum()

    def driving_force(self, mu: np.ndarray, t: float) -> np.ndarray:
        """min over composition of (G − μ·N)/ΣN, per mole of atoms, as a row linear in θ.

        ``mu`` is a 2 × (n+1) array of chemical-potential rows. The minimum is
        found at the current :attr:`_LinearModel.theta` (local search, as in
        :func:`~discopt.calphad.equilibrium.pricing.price_phase_local`); the row is
        exact there, and its θ-derivative is the derivative at the minimizer.
        """
        from discopt.calphad.equilibrium.pricing import price_phase_local

        self.lm.nonlinear = True
        theta1 = np.concatenate([[1.0], self.lm.theta])
        mu_now = mu @ theta1 / (R_GAS * t)
        y = price_phase_local(self.lm.model_at(self.name), mu_now, T=t).y
        N = np.array(self.model.amounts(y), float)
        return (self.F("G", y, t) - N @ mu) / N.sum()

    def formation(self, y, t: float, reference: str | None = None, prop: str = "H") -> np.ndarray:
        """Enthalpy (``prop="S"``: entropy, ``"CP"``: heat capacity) of formation per mole of atoms at ``y``.

        Relative to the elements in their reference phases.
        """
        N = np.array(self.model.amounts(y), float)
        out = self.F(prop, y, t) / N.sum()
        for e, n in zip(self.lm.elements, N):
            if n > 1e-12:
                ref = reference or self.lm.db.elements[e].reference_phase
                if not ref:
                    raise ValueError(f"{e} has no reference phase in the database; pass reference=")
                out = out - (n / N.sum()) * self.lm.phase(ref).pure_mu(e, t, prop=prop)
        return out


# ---------------------------------------------------------------------------- rows


def _rows(lm: _LinearModel, obs: Observation):
    """(vectors, observed values, sigmas) for one observation."""
    if obs.kind in THERMO_KINDS:
        ph = lm.phase(obs.phase)
        y = None if obs.site_fractions is None else ph.y_from_site_fractions(obs.site_fractions)
        if obs.kind in _FORM:
            vec = ph.formation(ph.y_of(obs.x) if y is None else y, obs.T, obs.reference, prop=_FORM[obs.kind])
        else:
            vec = ph.thermo(obs.kind, obs.x, obs.T, y, obs.site_fractions)
        return [vec], [obs.value], [obs.sigma]
    if obs.kind == "ACR":
        k = lm.elements.index(obs.element)
        mu = lm.phase(obs.phase).mu(obs.x, obs.T)[k]
        ref = lm.phase(obs.reference or obs.phase).pure_mu(obs.element, obs.T)
        return [(mu - ref) / (R_GAS * obs.T)], [math.log(obs.value)], [obs.sigma / obs.value]
    return _zpf_rows(lm, obs)


def _zpf_reference(lm: _LinearModel, obs: Observation):
    """μ rows of a phase-equilibrium observation, and the compounds it did not use to fix them.

    From the tangent of the first phase with a measured composition, or, without
    one, from two compounds (binary only).
    """
    known = [(lm.phase(n), x) for n, x in obs.phases if lm.phase(n).model.n_y > 0 and x is not None]
    comps = [lm.phase(n) for n, _ in obs.phases if lm.phase(n).model.n_y == 0]
    full = [(ph, x) for ph, x in known if ph.spans()]
    if full:
        return full[0][0].mu(full[0][1], obs.T), comps
    E = len(lm.elements)
    used = []
    for c in comps:  # E compounds with independent compositions fix μ
        trial = used + [c]
        if np.linalg.matrix_rank(np.array([p.model.amounts() for p in trial], float)) == len(trial):
            used = trial
        if len(used) == E:
            break
    if len(used) < E:
        raise ValueError(f"ZPF: μ needs a measured composition, or {E} compounds with independent compositions")
    Nmat = np.array([p.model.amounts() for p in used], float)
    ref = np.linalg.solve(Nmat, np.array([p.F("G", np.empty(0), obs.T) for p in used]))
    return ref, [c for c in comps if all(c is not u for u in used)]


def _zpf_rows(lm: _LinearModel, obs: Observation):
    """Rows of a phase-equilibrium observation, all of them zero at equilibrium (J/mol of atoms).

    μ comes from the first phase with a measured composition (its tangent), or,
    without one, from two compounds. Then every other measured phase has the
    same μ, every compound lies on the tangent, and every phase whose
    composition was not measured touches it: its driving force, the minimum
    over composition of (G − μ·N)/ΣN, is zero.
    """
    if len(obs.phases) < 2:
        raise ValueError("a ZPF observation needs at least two phases")
    known = [(lm.phase(n), x) for n, x in obs.phases if lm.phase(n).model.n_y > 0 and x is not None]
    unknown = [lm.phase(n) for n, x in obs.phases if lm.phase(n).model.n_y > 0 and x is None]
    ref, comps = _zpf_reference(lm, obs)
    full = [(ph, x) for ph, x in known if ph.spans()]
    vecs = []
    for ph, x in full[1:]:
        vecs.extend(ref - ph.mu(x, obs.T))
    for ph, x in known:
        if not ph.spans():  # a line or point phase: on the tangent, and stationary along its own variables
            vecs.extend(ph.touch_rows(x, ref, obs.T))
    for ph in comps:
        N = np.array(ph.model.amounts(), float)
        vecs.append((N @ ref - ph.F("G", np.empty(0), obs.T)) / N.sum())
    for ph in unknown:
        vecs.append(ph.driving_force(ref, obs.T))
    return vecs, [0.0] * len(vecs), [obs.sigma] * len(vecs)


# ---------------------------------------------------------------------------- fit


def _leverage(result) -> np.ndarray:
    """Diagonal of the hat matrix J C Jᵀ, one per residual row (0 for rows without a θ-dependence)."""
    from discopt.calphad.design import candidate_rows

    J = np.vstack(candidate_rows(result, result.data.observations))
    C = np.asarray(result.covariance, float)
    h = np.einsum("ij,jk,ik->i", J, C, J)
    if len(h) != len(result.residuals):  # extra rows (errors in variables) carry no θ-leverage here
        h = np.concatenate([h, np.zeros(len(result.residuals) - len(h))])
    return h


def _scaled(data: Dataset, scales: dict[str, float]) -> Dataset:
    out = Dataset(
        dataclasses.replace(o, sigma=o.sigma * scales.get(o.source or "unspecified", 1.0)) for o in data.observations
    )
    out.skipped = list(data.skipped)
    return out


def _source_scaled_fit(db, data, elements, free, source_scale, shrink, max_rounds, kwargs) -> FitResult:
    """A fit with each source's σ scaled: fixed factors, or estimated (``"auto"``) from that source's residuals."""
    sources = list(data.sources())
    spec = {k: "auto" for k in sources} if source_scale == "auto" else dict(source_scale)
    unknown = sorted(set(spec) - set(sources))
    if unknown:
        raise KeyError(f"source_scale names sources not in the data: {unknown}; the data have {sources}")
    for k, v in spec.items():
        if v != "auto" and not (isinstance(v, (int, float)) and v > 0):
            raise ValueError(f"source_scale[{k!r}] must be a positive number or 'auto', got {v!r}")
    scales = {k: (1.0 if v == "auto" else float(v)) for k, v in spec.items()}
    auto = [k for k, v in spec.items() if v == "auto"]
    result = fit(db, _scaled(data, scales), elements, free, **kwargs)
    for _ in range(max_rounds if auto else 0):
        by = result.residuals_by_source()
        change = 0.0
        for k in auto:
            n, chi2, dof = by.get(k, (0, 0.0, 0.0))
            if dof < 1.0:  # the data cannot estimate this source's scale
                continue
            ratio = math.sqrt(chi2 / dof)
            new = scales[k] * ratio
            if not shrink:
                new = max(new, 1.0)
            change = max(change, abs(new / scales[k] - 1.0))
            scales[k] = new
        if change < 1e-3:
            break
        result = fit(db, _scaled(data, scales), elements, free, **kwargs)
    result.source_scales = scales
    return result


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
    #: linearizations used (1 when every row is linear in θ), and whether θ converged
    iterations: int = 1
    converged: bool = True
    #: errors in variables: (observation index, phase, measured x, fitted x, sigma_x) per composition
    compositions: list = field(default_factory=list)
    #: global consistency (``consistency="global"``): status, rounds, cuts, worst lower bound, and
    #: per condition and phase the certified bounds in J/mol of atoms; None when not requested
    consistency: dict | None = None
    #: per-source σ scale factors used (``source_scale=``), or None
    source_scales: dict | None = None

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

    def residuals_by_source(self) -> dict[str, tuple[int, float, float]]:
        """{source: (rows, χ² of its rows, effective degrees of freedom)}.

        The effective degrees of freedom are the rows less their leverage
        (the diagonal of the hat matrix), so they sum to rows − rank over all
        sources. χ²/dof well above 1 for a source says its σ are too small
        relative to the rest of the data; see ``fit(source_scale=)``.
        """
        lev = _leverage(self)
        src = np.array([self.data.observations[i].source or "unspecified" for i in self.row_observation])
        out = {}
        for k in dict.fromkeys(src):
            m = src == k
            out[str(k)] = (int(m.sum()), float(np.sum(self.residuals[m] ** 2)), float(m.sum() - lev[m].sum()))
        return out

    def summary(self) -> str:
        lines = [
            f"fit of {len(self.names)} coefficients to {len(self.data)} observations "
            f"({self.residuals.size} residuals): chi2 = {self.chi2:.4g}, dof = {self.dof}, rank = {self.rank}"
        ]
        if self.iterations > 1 or not self.converged:
            lines.append(f"  {self.iterations} linearizations, {'converged' if self.converged else 'NOT converged'}")
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
                model = self.database.phase_model(name, self.elements)
                if x is None and model.n_y == 0:
                    a = np.array(model.amounts(), float)
                    xs.append(a / a.sum())
                elif x is not None:
                    xs.append(_composition(self.elements, x))
            bbar = np.mean(xs, axis=0)
            xbar = float(bbar[0]) if len(self.elements) == 2 else tuple(map(float, bbar))
            res = equilibrium(phases, b=tuple(bbar), T=obs.T, eps=eps, **kwargs)
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


def predict(db: ThermoDatabase, data, elements) -> np.ndarray:
    """What ``db`` predicts for each observation, in the observation's own units.

    Thermochemical kinds give the property (J/mol of atoms, or J/(mol K)), and
    ``ACR`` gives the activity. A ``ZPF`` observation gives the largest
    absolute residual of its rows in J/mol (0 when the database reproduces the
    equilibrium). Use it to compare calculated or measured data with an
    assessment before fitting anything.
    """
    observations = list(data)
    lm = _LinearModel(db, elements, [])
    lm.theta = np.zeros(0)
    out = []
    for o in observations:
        vecs, _, _ = _rows(lm, o)
        values = [float(v[0]) for v in vecs]
        if o.kind == "ACR":
            out.append(math.exp(values[0]))
        elif o.kind == "ZPF":
            out.append(max(abs(v) for v in values))
        else:
            out.append(values[0])
    return np.array(out)


def enthalpy_form(db: ThermoDatabase, label: str) -> str:
    """A free form for ``label`` whose constant ``a`` is free and whose T-dependence is the database's.

    For the database's ``G(ALCU_THETA,AL:CU;0) = -47406 + 6.75*T + 2*GHSERAL + GHSERCU``
    this is ``"a + 6.75*T + 2*GHSERAL# + GHSERCU#"``. That is the form to fit to
    enthalpies alone (0 K energies, formation enthalpies), which carry no
    information about entropy. The parameter must have one temperature range.
    """
    kind, phase, subl, order = _parse_label(label)
    hits = [p for p in db.parameters if (p.kind, p.phase, p.constituents, p.order) == (kind, phase, subl, order)]
    if not hits:
        raise KeyError(f"{label} is not in the database")
    fn = hits[0].function
    if len(fn.exprs) != 1:
        raise ValueError(f"{label} has {len(fn.exprs)} temperature ranges; write its form by hand")
    expr = sp.expand(fn.exprs[0])
    const, rest = expr.as_coeff_Add()
    if rest == 0:
        return "a"
    text = tdb_str(rest)
    return f"a - {text[1:]}" if text.startswith("-") else f"a + {text.removeprefix('+')}"


def _design(lm: _LinearModel, data: Dataset):
    """Rows at the current θ: (V, observed values, sigmas, owning observation)."""
    vecs, obs_v, sig, owner = [], [], [], []
    for j, o in enumerate(data):
        v, ov, s = _rows(lm, o)
        vecs += v
        obs_v += ov
        sig += s
        owner += [j] * len(v)
    if not vecs:
        raise ValueError("no data")
    return np.array(vecs), np.array(obs_v, float), np.array(sig, float), owner


def fit(
    db: ThermoDatabase,
    data: Dataset,
    elements,
    free,
    *,
    bounds=None,
    initial=None,
    max_iter: int = 30,
    tol: float = 1e-3,
    verbose: bool = False,
    errors_in_variables: bool = False,
    consistency: str | None = None,
    consistency_eps: float = 10.0,
    consistency_time_limit: float = 120.0,
    max_cut_rounds: int = 10,
    source_scale=None,
    source_scale_shrink: bool = False,
    source_scale_rounds: int = 20,
) -> FitResult:
    """Fit free parameter coefficients of ``db`` to ``data`` by weighted least squares.

    ``free`` is a list of parameter labels (each gets the form ``a + b*T``), or a
    dict ``{label: form}``, for example ``{"L(FCC_A1,AL,ZN;0)": "a + b*T"}``. A
    label that does not exist yet in the database is added as a new parameter.
    A form may be any expression in T that is linear in its coefficients, and
    it may use the database's FUNCTIONs, as a compound's G usually does:
    ``{"G(ALCU_THETA,AL:CU;0)": "a + b*T + 2*GHSERAL + GHSERCU"}``. A free
    parameter replaces every temperature range of the original.

    ``bounds`` maps coefficient names (``"<label>.<coef>"``, see
    :attr:`FreeParameter.names`) to ``(lo, hi)``.

    Most data give rows that are linear in the coefficients, and one linear
    solve is exact. Rows that need a minimization over site fractions (the
    driving force of a phase whose composition was not measured, and μ of a
    phase with several site fractions) depend on θ through the minimizer. The
    fit then iterates: rows at θ_k, a linear solve, and a step halved until χ²
    decreases. It stops when every coefficient moves by less than ``tol``
    times its standard deviation, or when no step decreases χ² (the minimizers
    are found numerically, so χ² has a small noise floor), and after
    ``max_iter`` linearizations at most. At a minimizer the row's θ-derivative
    is exact (envelope theorem), so this is Gauss–Newton with the exact
    Jacobian. It starts from ``initial`` (a dict by coefficient name), or from
    the coefficients that best reproduce each parameter's original function (0
    for a new parameter). ``verbose=True`` prints χ² and the step at each
    linearization.

    The covariance is (JᵀJ)⁻¹, with J the σ-weighted design matrix at the
    solution. It is absolute: it assumes the given σ are the true uncertainties.

    **Errors in variables.** By default every measured composition is taken as
    exact, and its uncertainty can enter only through the J/mol σ of its rows.
    With ``errors_in_variables=True``, each measured composition of a ZPF
    observation that has ``sigma_x`` becomes an unknown too. It gets its own
    residual, (x − x_measured)/σ_x, and the equilibrium rows are evaluated at the
    fitted compositions. The ZPF row σ then stands for model error only, and
    can be small. The problem is nonlinear in the compositions. It is solved by
    trust-region least squares (``scipy.optimize.least_squares``) from the
    ordinary fit. The parameter columns of the Jacobian are exact, and each
    composition column is a central difference of its own observation's rows.
    The covariance of θ is the θ block of (JᵀJ)⁻¹ over all unknowns, and the
    degrees of freedom count the compositions as parameters.
    :attr:`FitResult.compositions` lists the measured and fitted compositions.

    **Global consistency.** A fit can match every datum and still make a phase
    stable where the data rule it out. With ``consistency="global"``, after
    the fit every phase that a phase-equilibrium observation does not list is
    priced globally (discopt) at that observation's μ. If its tangent distance
    falls below −``consistency_eps`` (J/mol of atoms) somewhere, that point
    becomes a cut, Ψ(y; θ) ≥ −ε/2, which is linear in θ at fixed y, and the fit
    is solved again as a constrained least-squares problem. (The cut is tighter
    than the check, so a refit lands inside the certified region.) This repeats until no
    phase falls below −ε, for at most ``max_cut_rounds`` rounds. The result's
    :attr:`FitResult.consistency` then says ``CERTIFIED``: every certified lower
    bound is ≥ −ε. It says ``INCOMPLETE`` when a bound could not be raised that
    far (magnetic phases have none yet, and ``consistency_time_limit`` bounds
    each pricing), and ``NOT_CERTIFIED`` when rounds ran out. Phases listed in
    an observation are not checked there: their deviation from the tangent is
    that observation's own residual.

    **Per-source weighting.** Data from different sources (``Observation.source``:
    experiment, DFT, an MLIP) come with σ set by different people and methods,
    which may not be on the same footing. :meth:`FitResult.residuals_by_source`
    compares them: each source's χ² against its effective degrees of freedom.
    ``source_scale`` multiplies the σ of each source: a dict ``{source:
    factor}``, where a factor may be ``"auto"``, or ``"auto"`` for every source.
    An ``"auto"`` factor is estimated from the data (a Birge ratio per source,
    √(χ²_s/dof_s), iterated with refits until no factor changes by more than
    0.1%, ``source_scale_rounds`` at most). It only grows σ unless
    ``source_scale_shrink=True``: a source that fits better than its σ say is
    kept at its stated σ. A source with less than one effective degree of
    freedom cannot be estimated and keeps factor 1. The factors used are in
    :attr:`FitResult.source_scales`, and :attr:`FitResult.data` holds the
    scaled σ.
    """
    if source_scale is not None:
        kwargs = dict(
            bounds=bounds,
            initial=initial,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            errors_in_variables=errors_in_variables,
            consistency=consistency,
            consistency_eps=consistency_eps,
            consistency_time_limit=consistency_time_limit,
            max_cut_rounds=max_cut_rounds,
        )
        return _source_scaled_fit(
            db, data, elements, free, source_scale, source_scale_shrink, source_scale_rounds, kwargs
        )
    free = {label: "a + b*T" for label in free} if not isinstance(free, dict) else free
    fps = [FreeParameter.parse(label, form, functions=db.functions) for label, form in free.items()]
    lm = _LinearModel(db, elements, fps)

    lo = np.full(lm.n, -np.inf)
    hi = np.full(lm.n, np.inf)
    for name, (a, b) in (bounds or {}).items():
        if name not in lm.names:
            raise KeyError(f"bound on unknown coefficient {name!r}; coefficients are {lm.names}")
        k = lm.names.index(name)
        lo[k], hi[k] = (-np.inf if a is None else a), (np.inf if b is None else b)
    bounded = bool(np.isfinite(lo).any() or np.isfinite(hi).any())

    theta = lm.initial_theta()
    for name, v in (initial or {}).items():
        if name not in lm.names:
            raise KeyError(f"initial value for unknown coefficient {name!r}; coefficients are {lm.names}")
        theta[lm.names.index(name)] = float(v)
    theta = np.clip(theta, lo, hi)

    # Global consistency: rows c with c·[1, θ] ≥ −consistency_eps/2. The cut is tighter than the
    # check (−consistency_eps), so a refit lands strictly inside the certified region, not on its edge.
    cuts: list[np.ndarray] = []

    def solve(V, obs_v, sig):
        J = V[:, 1:] / sig[:, None]
        r0 = (obs_v - V[:, 0]) / sig
        scale = np.linalg.norm(J, axis=0)
        scale[scale == 0] = 1.0
        Js = J / scale
        if bounded:
            t = lsq_linear(Js, r0, bounds=(lo * scale, hi * scale), method="bvls", tol=1e-14).x
        else:
            t = np.linalg.lstsq(Js, r0, rcond=None)[0]
        if cuts:
            from scipy.optimize import minimize

            A = np.array([c[1:] / scale for c in cuts])
            b = np.array([-0.5 * consistency_eps - c[0] for c in cuts])
            box = [
                (None if not np.isfinite(a) else a * sc, None if not np.isfinite(c) else c * sc)
                for a, c, sc in zip(lo, hi, scale)
            ]
            res = minimize(
                lambda t: 0.5 * float(np.sum((Js @ t - r0) ** 2)),
                t,
                jac=lambda t: Js.T @ (Js @ t - r0),
                method="SLSQP",
                bounds=box,
                constraints=[{"type": "ineq", "fun": lambda t: A @ t - b, "jac": lambda t: A}],
                options={"ftol": 1e-15, "maxiter": 1000},
            )
            t = res.x
        return t / scale

    def chi2(V, obs_v, sig, th):
        r = (V[:, 0] + V[:, 1:] @ th - obs_v) / sig
        return float(r @ r)

    def std_of(V, sig):
        J = V[:, 1:] / sig[:, None]
        sc = np.linalg.norm(J, axis=0)
        sc[sc == 0] = 1.0
        return np.sqrt(np.clip(np.diag(np.linalg.pinv((J / sc).T @ (J / sc))), 0, None)) / sc

    def run(theta, restart=False):
        """The fit from ``theta``; with ``restart``, the first step goes to the constrained solution in full."""
        lm.theta = theta
        V, obs_v, sig, owner = _design(lm, data)
        iterations, converged = 1, True
        if lm.nonlinear:
            converged = False
            current = chi2(V, obs_v, sig, theta)
            if verbose:
                print(f"fit: start chi2 = {current:.6g}")
            iterations = 0
            while iterations < max_iter:
                iterations += 1
                target = solve(V, obs_v, sig)
                sd = std_of(V, sig)
                step, accepted = 1.0, False
                while step >= 1e-3:
                    trial = theta + step * (target - theta)
                    lm.theta = trial
                    Vt, ot, st, owner_t = _design(lm, data)
                    value = chi2(Vt, ot, st, trial)
                    # a new cut can make the previous θ infeasible: then the first step is taken in full
                    if value <= current or (restart and iterations == 1):
                        accepted = True
                        break
                    step /= 2
                moved = np.max(np.abs(trial - theta) / np.where(sd > 0, sd, 1.0))
                if verbose:
                    print(
                        f"fit: linearization {iterations}: chi2 = {value:.6g}, step {step:g}, "
                        f"largest move {moved:.3g} sigma{'' if accepted else ' (rejected)'}"
                    )
                if not accepted:
                    lm.theta = theta  # no step decreases chi2: stationary within the minimizers' noise
                    converged = True
                    break
                theta, V, obs_v, sig, owner, current = trial, Vt, ot, st, owner_t, value
                if moved < tol:
                    converged = True
                    break
            if not converged:
                warnings.warn(f"the fit did not converge in {max_iter} iterations", stacklevel=3)
        else:
            theta = solve(V, obs_v, sig)
            lm.theta = theta
        return theta, V, obs_v, sig, owner, iterations, converged

    theta, V, obs_v, sig, owner, iterations, converged = run(theta)
    report = None
    if consistency is not None:
        if consistency != "global":
            raise ValueError(f"consistency must be None or 'global', got {consistency!r}")
        if errors_in_variables:
            raise NotImplementedError("consistency='global' with errors_in_variables is not supported yet")
        rounds = 0
        while True:
            report = _consistency_check(lm, data, theta, consistency_eps, consistency_time_limit, verbose)
            rounds += 1
            report["rounds"] = rounds
            report["cuts"] = len(cuts)
            new = report.pop("violations")
            if not new or rounds > max_cut_rounds:
                if new:
                    report["status"] = "NOT_CERTIFIED"
                break
            cuts.extend(new)
            theta, V, obs_v, sig, owner, it2, converged = run(theta, restart=True)
            iterations += it2

    if errors_in_variables:
        return _errors_in_variables(lm, data, fps, theta, lo, hi, iterations, converged)

    J = V[:, 1:] / sig[:, None]
    r0 = (obs_v - V[:, 0]) / sig
    scale = np.linalg.norm(J, axis=0)
    scale[scale == 0] = 1.0
    Js = J / scale
    rank = int(np.linalg.matrix_rank(Js))
    if rank < lm.n:
        warnings.warn(
            f"the data determine only {rank} of {lm.n} coefficients; the solution is not unique", stacklevel=2
        )
    active = [n for n, t, a, b in zip(lm.names, theta, lo, hi) if np.isclose(t, a) or np.isclose(t, b)]

    resid = J @ theta - r0
    cov = np.linalg.pinv(Js.T @ Js) / np.outer(scale, scale)
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)

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
        database=_replace(lm.db, lm.functions_at(theta)),
        data=data,
        elements=lm.elements,
        free=fps,
        active_bounds=active,
        iterations=iterations,
        converged=converged,
        consistency=report,
    )


def _errors_in_variables(lm, data, fps, theta, lo, hi, iterations, converged) -> FitResult:
    """θ and the measured compositions together, by trust-region least squares (see :func:`fit`)."""
    from collections import defaultdict

    from scipy.optimize import least_squares

    obs = list(data)
    slots = []  # (observation, position in its phases, measured x, sigma_x)
    for j, o in enumerate(obs):
        if o.kind == "ZPF" and o.sigma_x:
            for k, (name, x) in enumerate(o.phases):
                if x is not None and lm.phase(name).model.n_y > 0:
                    if not isinstance(x, float):
                        raise NotImplementedError("errors_in_variables needs a binary (one mole fraction per phase)")
                    slots.append((j, k, float(x), float(o.sigma_x)))
    if not slots:
        raise ValueError("errors_in_variables needs ZPF observations with sigma_x and measured compositions")
    n, m = lm.n, len(slots)
    xm = np.array([s_[2] for s_ in slots])
    sx = np.array([s_[3] for s_ in slots])
    by_obs = defaultdict(list)
    for i, (j, k, _, _) in enumerate(slots):
        by_obs[j].append((i, k))

    def rows(j, z):
        o = obs[j]
        if j in by_obs:
            ph = list(o.phases)
            for i, k in by_obs[j]:
                ph[k] = (ph[k][0], float(z[i]))
            o = dataclasses.replace(o, phases=tuple(ph))
        V, ov, sg = _rows(lm, o)
        return np.array(V, float), np.array(ov, float), np.array(sg, float)

    def residuals(p):
        th, z = p[:n], p[n:]
        lm.theta = th
        out = []
        for j in range(len(obs)):
            V, ov, sg = rows(j, z)
            out.append((V[:, 0] + V[:, 1:] @ th - ov) / sg)
        out.append((z - xm) / sx)
        return np.concatenate(out)

    def jacobian(p):
        th, z = p[:n], p[n:]
        lm.theta = th
        blocks = []
        for j in range(len(obs)):
            V, ov, sg = rows(j, z)
            B = np.zeros((len(sg), n + m))
            B[:, :n] = V[:, 1:] / sg[:, None]
            for i, _ in by_obs.get(j, ()):
                h = min(1e-6, 0.5 * z[i], 0.5 * (1 - z[i]))
                zp, zn = z.copy(), z.copy()
                zp[i] += h
                zn[i] -= h
                Vp, _, _ = rows(j, zp)
                Vn, _, _ = rows(j, zn)
                B[:, n + i] = ((Vp[:, 0] + Vp[:, 1:] @ th) - (Vn[:, 0] + Vn[:, 1:] @ th)) / (2 * h) / sg
            blocks.append(B)
        C = np.zeros((m, n + m))
        C[:, n:] = np.diag(1.0 / sx)
        blocks.append(C)
        return np.vstack(blocks)

    p0 = np.concatenate([theta, np.clip(xm, 1e-9, 1 - 1e-9)])
    lower = np.concatenate([lo, np.full(m, 1e-9)])
    upper = np.concatenate([hi, np.full(m, 1 - 1e-9)])
    sol = least_squares(
        residuals,
        p0,
        jac=jacobian,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-10,
        max_nfev=200,
    )
    p = sol.x
    resid = residuals(p)
    J = jacobian(p)
    scale = np.linalg.norm(J, axis=0)
    scale[scale == 0] = 1.0
    Js = J / scale
    rank = int(np.linalg.matrix_rank(Js))
    if rank < n + m:
        warnings.warn(
            f"the data determine only {rank} of {n + m} unknowns (coefficients and compositions)", stacklevel=3
        )
    cov_all = np.linalg.pinv(Js.T @ Js) / np.outer(scale, scale)
    cov = cov_all[:n, :n]
    std = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)
    theta = p[:n]
    owner = []
    for j in range(len(obs)):
        owner += [j] * len(rows(j, p[n:])[2])
    owner += [s_[0] for s_ in slots]
    active = [nm for nm, t, a, b in zip(lm.names, theta, lo, hi) if np.isclose(t, a) or np.isclose(t, b)]
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
        database=_replace(lm.db, lm.functions_at(theta)),
        data=data,
        elements=lm.elements,
        free=fps,
        active_bounds=active,
        iterations=iterations + int(sol.nfev),
        converged=converged and bool(sol.success),
        compositions=[(j, obs[j].phases[k][0], x, float(p[n + i]), sx_) for i, (j, k, x, sx_) in enumerate(slots)],
    )


def _consistency_check(lm, data, theta, eps: float, time_limit: float, verbose: bool) -> dict:
    """Price every phase absent from each phase-equilibrium observation at that observation's μ.

    Bounds are certified (discopt's global pricing) and converted to J/mol of
    atoms. A phase whose best point lies below −``eps`` is a violation, and its
    tangent-distance row at that point becomes a cut. A phase whose lower bound
    could not be raised to −``eps`` makes the result ``INCOMPLETE``.
    """
    from discopt.calphad.equilibrium.pricing import price_phase

    lm.theta = theta
    theta1 = np.concatenate([[1.0], theta])
    names = list(lm.db.phase_models(lm.elements))
    out = {"status": "CERTIFIED", "eps": eps, "worst_lower_bound": math.inf, "conditions": [], "violations": []}
    undecided = False
    for j, o in enumerate(data):
        if o.kind != "ZPF":
            continue
        try:
            ref, _ = _zpf_reference(lm, o)
        except ValueError:
            continue
        T = o.T
        RT = R_GAS * T
        mu_rt = (ref @ theta1) / RT
        listed = {n.upper() for n, _ in o.phases}
        for name in names:
            if name in listed:
                continue
            model = lm.model_at(name)
            atoms_min = float(model.atoms_min)
            r = price_phase(model, mu_rt, T=T, eps=eps * atoms_min / RT, time_limit=time_limit)
            atoms = float(sum(float(a) for a in model.amounts(r.y))) if r.y is not None else atoms_min
            ub = r.ub * RT / max(atoms, 1e-12)
            lb = None if r.lb is None else r.lb * RT / atoms_min
            out["conditions"].append({"observation": j, "T": T, "phase": name, "lower": lb, "upper": ub})
            if lb is not None:
                out["worst_lower_bound"] = min(out["worst_lower_bound"], lb)
            if ub < -eps:
                out["violations"].append(lm.phase(name).tangent_row(r.y, ref, T))
                if verbose:
                    print(f"fit: {name} is {-ub:.4g} J/mol below the tangent at observation {j} ({T:g} K)")
            elif lb is None or lb < -eps:
                undecided = True
    if out["violations"]:
        out["status"] = "VIOLATED"
    elif undecided:
        out["status"] = "INCOMPLETE"
    return out
