"""The Python side of the browser app: databases in, JSON-shaped results out.

This module runs inside Pyodide. It holds no state the page cannot rebuild, and
it never imports matplotlib -- the page draws, so points can appear as they are
computed instead of arriving as one finished picture.

Every point is solved cold, from ``equilibrium()``'s own starting set. Warm
starting each point from its neighbour (``initial_sets=``) is twice as fast and
was measured to be wrong: across a 120-point Cu-Mg grid it disagreed with cold
at 10 points, and at 9 of them its G was *higher* -- it had settled into a
single-phase minimum and stayed there. Certified solves refereed three of those
points and matched the cold answer exactly. The failure is silent, so the speed
is not worth it. (Path continuation is the principled version of that idea and
does keep the right branch; see issue #4.)
"""

from __future__ import annotations

import json
import time

import numpy as np

from discopt.calphad import R_GAS, ThermoDatabase, equilibrium

#: name -> (ThermoDatabase, pressure it was built at, {elements: {phase: model}})
_DATABASES: dict[str, tuple] = {}


def load_database(name: str, text: str, pressure: float = 101325.0) -> str:
    """Parse a TDB and report what it covers.

    The page fetches the file and passes its text, so the database never has to
    exist as a file inside the Pyodide sandbox.
    """
    db = ThermoDatabase.from_string(text)
    db.pressure = float(pressure)
    _DATABASES[name] = (db, float(pressure), {})
    elements = sorted(e for e in db.elements if e not in ("/-", "VA"))
    return json.dumps({"name": name, "elements": elements, "phases": sorted(db.phases)})


def _models(name: str, elements: tuple[str, ...], pressure: float):
    db, built_at, cache = _DATABASES[name]
    if pressure != built_at:
        # Pressure feeds expression evaluation and the evaluated values are
        # cached per pressure, so the models are rebuilt rather than reused.
        db.pressure = float(pressure)
        _DATABASES[name] = (db, float(pressure), cache := {})
    if elements not in cache:
        cache[elements] = list(db.phase_models(elements).values())
    return cache[elements]


def _describe(result, elements, T: float) -> dict:
    """An equilibrium result as plain JSON-able data, in J/mol as well as RT."""
    n_atoms = float(np.sum(result.b))
    sets = [
        {
            "phase": cs.phase.name,
            "x": [float(v) for v in cs.x],
            "amount": float(cs.amount),
            "atoms": float(cs.atoms),
            # Fraction of the system's atoms in this composition set -- what a
            # lever-rule reading of the diagram gives.
            "fraction": float(cs.amount * cs.atoms / n_atoms) if n_atoms else 0.0,
        }
        for cs in result.composition_sets
    ]
    return {
        "elements": list(elements),
        "T": float(T),
        "x": float(result.b[0] / n_atoms) if n_atoms else 0.0,
        "b": [float(v) for v in result.b],
        "phases": list(result.phase_names),
        "sets": sets,
        # G is in RT for the whole system; per mole of atoms in J/mol is what
        # thermodynamic tables and pycalphad report.
        "G_RT": float(result.G),
        "GM": float(result.G * R_GAS * T / n_atoms) if n_atoms else 0.0,
        "mu": [float(v * R_GAS * T) for v in result.mu],
        "certificate": {
            "level": result.certificate.level.name,
            "eps": float(result.certificate.eps),
            "messages": list(result.certificate.messages),
        },
        "n_iterations": result.n_iterations,
    }


def point(name: str, elements, T: float, x=None, b=None, pressure: float = 101325.0, certify: bool = False) -> str:
    """One equilibrium, by mole fraction ``x`` or by element amounts ``b``."""
    elements = tuple(elements)
    models = _models(name, elements, pressure)
    started = time.perf_counter()
    kwargs = {"b": [float(v) for v in b]} if b is not None else {"x": float(x)}
    result = equilibrium(models, T=float(T), certify="final" if certify else "none", **kwargs)
    out = _describe(result, elements, T)
    out["ms"] = 1000.0 * (time.perf_counter() - started)
    out["certified"] = bool(certify)
    return json.dumps(out)


def diagram(name, elements, t_lo, t_hi, n_t, x_lo, x_hi, n_x, pressure=101325.0, emit=None) -> str:
    """Sweep a T-x grid, handing each point to ``emit`` as it is solved.

    ``emit`` is a JavaScript callback. Calling it per point is the whole reason
    the work happens here rather than behind a single ``run()``: a Cu-Mg grid is
    tens of seconds, and the page fills in as it goes.
    """
    elements = tuple(elements)
    models = _models(name, elements, pressure)
    temperatures = np.linspace(float(t_lo), float(t_hi), int(n_t))
    fractions = np.linspace(float(x_lo), float(x_hi), int(n_x))
    started, solved, failed = time.perf_counter(), 0, 0

    for T in temperatures:
        for x in fractions:
            try:
                result = equilibrium(models, x=float(x), T=float(T), certify="none")
            except Exception as exc:  # noqa: BLE001 - one bad point must not end the sweep
                failed += 1
                if emit is not None:
                    emit(json.dumps({"T": float(T), "x": float(x), "error": f"{type(exc).__name__}: {exc}"}))
                continue
            if not result.composition_sets:
                # A restricted solve that did not converge has nothing to draw.
                failed += 1
                if emit is not None:
                    emit(json.dumps({"T": float(T), "x": float(x), "error": "restricted solve did not converge"}))
                continue
            solved += 1
            if emit is not None:
                emit(json.dumps(_describe(result, elements, float(T))))

    return json.dumps(
        {
            "points": int(n_t) * int(n_x),
            "solved": solved,
            "failed": failed,
            "seconds": time.perf_counter() - started,
        }
    )
