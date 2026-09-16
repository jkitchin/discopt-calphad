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
is not worth it. Re-solving only the disagreeing points and keeping the lower G
was measured too: it repairs those rows but fragments the drawn boundaries and
costs a third again in wall clock, because the tracer keys on neighbouring
points agreeing rather than on each point being optimal. See ``web/README.md``.
(Path continuation is the principled version of the idea and does keep the
right branch.)
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from discopt.calphad import R_GAS, ThermoDatabase, equilibrium

#: name -> (ThermoDatabase, pressure it was built at, {elements: {phase: model}})
_DATABASES: dict[str, tuple] = {}


def load_database(name: str, text: str, pressure: float = 101325.0) -> str:
    """Parse a TDB and report what it covers.

    The page fetches the file and passes its text, so the database never has to
    exist as a file inside the Pyodide sandbox.

    A file the user picked is parsed leniently: one corrupt command should cost
    that command rather than the file, so the commands that would not parse come
    back in ``skipped`` for the page to report. It reports rather than swallows,
    because a dropped parameter shows up as wrong energies, not as an error.
    """
    skipped: list[dict] = []
    db = ThermoDatabase.from_string(text, skipped=skipped)
    db.pressure = float(pressure)
    _DATABASES[name] = (db, float(pressure), {})
    elements = sorted(e for e in db.elements if e not in ("/-", "VA"))
    return json.dumps({"name": name, "elements": elements, "phases": sorted(db.phases), "skipped": skipped})


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


#: The vacancy, which occupies a site without contributing an atom. A phase can
#: therefore be pure in one element while a whole sublattice holds vacancies.
_VACANCY = "VA"


def _pure_variables(model, element: str):
    """Site-fraction variables putting ``model`` at pure ``element``, or None.

    A phase can be pure only where some configuration of its sublattices holds
    that element and nothing else: FCC_A1 as ``(AL,ZN)1`` can, AL3TI cannot,
    and a phase whose second sublattice is vacancies can, because vacancies
    carry no atoms. Rather than reason about which model class this is, each
    candidate is put to ``amounts`` -- the model's own account of how many moles
    of each element one formula unit holds -- and accepted only if it answers
    with this element alone.
    """
    candidates = []
    constituents = getattr(model, "constituents", None)
    if constituents is not None:
        fractions = []
        for sublattice in constituents:
            occupant = element if element in sublattice else _VACANCY
            if occupant not in sublattice:
                fractions = None
                break
            fractions.append({c: (1.0 if c == occupant else 0.0) for c in sublattice})
        if fractions is not None:
            candidates.append(model.variables_from_site_fractions(fractions))
    elif model.n_y == 0:
        candidates.append(())
    elif model.n_y == 1:
        # One mixing site: pure is an end of it, and which end depends on how
        # the model orders its elements, so both are offered and checked.
        candidates += [np.array([1.0]), np.array([0.0])]
    for y in candidates:
        amounts = dict(zip(model.elements, model.amounts(y)))
        total = sum(amounts.values())
        if total > 0 and amounts.get(element, 0.0) >= total * (1 - 1e-9):
            return y
    return None


def _pure_holder(pure, T: float):
    """Which of ``pure`` has the lowest molar G at ``T`` -- the stable phase."""
    best, best_g = None, math.inf
    for phase, model, y in pure:
        try:
            # gibbs() is G/RT for one formula unit, so this is J per mole of
            # atoms: the phases have different formula units and only a molar
            # quantity compares them.
            g = float(model.gibbs(y, T=T)) * R_GAS * T / sum(model.amounts(y))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        if g < best_g:
            best, best_g = phase, g
    return best


def pure_transitions(name, elements, pressure=101325.0, t_lo=300.0, t_hi=2000.0, tol=0.02) -> str:
    """Where a *pure* element changes phase: a root find, not an equilibrium.

    The diagram's boundaries are traced from tie-lines, and a pure element has
    no tie-line -- with one component there is nothing for a solve to balance.
    That is why the samples are held just inside x = 0 and 1, and why the
    liquidus stopped short of the melting points instead of meeting them.

    With one component, though, the answer needs no solve at all: whichever
    phase has the lowest molar G holds, and a transition is exactly where two of
    them cross. So this is a one-dimensional root find on temperature, over the
    same models the diagram is drawn from, and it puts the endpoints of the
    boundaries where they belong rather than extrapolating toward them. Al melts
    at 933 K and Zn at 693 K, and those are the two points the Al-Zn liquidus
    has to reach.

    Returned as ``{element, x, T, below, above}`` per transition, where ``x`` is
    the mole fraction of ``elements[0]`` -- so the second element's transitions
    sit at ``x = 0``.
    """
    elements = tuple(elements)
    models = _models(name, elements, pressure)
    t_lo, t_hi = float(t_lo), float(t_hi)
    out = []
    for index, element in enumerate(elements):
        pure = [(model.name, model, y) for model in models if (y := _pure_variables(model, element)) is not None]
        # One phase that can be pure has nothing to cross with.
        if len(pure) < 2:
            continue
        # A scan cheap enough to be free and fine enough not to step over a
        # phase that is stable across only a few kelvin, then a bisection on
        # each change of holder. Every call here is an expression evaluation.
        steps = max(1, math.ceil((t_hi - t_lo) / 2.0))
        temps = [t_lo + (t_hi - t_lo) * k / steps for k in range(steps + 1)]
        was_T, was = temps[0], _pure_holder(pure, temps[0])
        for T in temps[1:]:
            now = _pure_holder(pure, T)
            if now is not None and was is not None and now != was:
                lo, hi = was_T, T
                while hi - lo > tol:
                    mid = (lo + hi) / 2
                    if _pure_holder(pure, mid) == was:
                        lo = mid
                    else:
                        hi = mid
                x = 1.0 if index == 0 else 0.0
                out.append({"element": element, "x": x, "T": (lo + hi) / 2, "below": was, "above": now})
            was_T, was = T, now
    return json.dumps(out)


def _emit_point(models, elements, T: float, x: float, ident, emit) -> bool:
    """Solve one point and hand it to ``emit``. True if it solved.

    ``ident`` is echoed back untouched. Where to sample is the caller's
    decision, and for the browser it is no longer a grid this module could
    reconstruct -- it subdivides adaptively -- so each request carries a label
    and the answers are matched up by label rather than by position.
    """
    T = float(T)
    x = float(x)
    try:
        result = equilibrium(models, x=x, T=T, certify="none")
    except Exception as exc:  # noqa: BLE001 - one bad point must not end the sweep
        failure = f"{type(exc).__name__}: {exc}"
    else:
        # A restricted solve that did not converge has nothing to draw.
        failure = None if result.composition_sets else "restricted solve did not converge"
    if failure is not None:
        if emit is not None:
            emit(json.dumps({"T": T, "x": x, "id": ident, "error": failure}))
        return False
    if emit is not None:
        out = _describe(result, elements, T)
        out["id"] = ident
        emit(json.dumps(out))
    return True


def _sweep_summary(total: int, solved: int, failed: int, started: float) -> str:
    return json.dumps(
        {
            "points": total,
            "solved": solved,
            "failed": failed,
            "seconds": time.perf_counter() - started,
        }
    )


def points(name, elements, coords_json: str, pressure=101325.0, emit=None) -> str:
    """Solve an explicit list of ``{"T", "x", "id"}`` points, in order.

    The browser drives this rather than ``diagram``: it starts from a coarse
    grid and then subdivides only the cells a phase boundary runs through, which
    is not a shape ``linspace`` can describe. Passing the coordinates keeps that
    decision in one place -- the page, which is what has the solved points to
    decide from.
    """
    elements = tuple(elements)
    models = _models(name, elements, pressure)
    coords = json.loads(coords_json)
    started, solved, failed = time.perf_counter(), 0, 0
    for item in coords:
        if _emit_point(models, elements, item["T"], item["x"], item.get("id"), emit):
            solved += 1
        else:
            failed += 1
    return _sweep_summary(len(coords), solved, failed, started)


def diagram(name, elements, t_lo, t_hi, n_t, x_lo, x_hi, n_x, pressure=101325.0, emit=None) -> str:
    """Sweep a uniform T-x grid, handing each point to ``emit`` as it is solved.

    ``emit`` is a JavaScript callback in the browser. Calling it per point is
    the whole reason the work happens here rather than behind a single
    ``run()``: a Cu-Mg grid is tens of seconds, and the page fills in as it
    goes. This is the entry point for a caller that just wants the grid; the
    page itself uses ``points``.
    """
    elements = tuple(elements)
    models = _models(name, elements, pressure)
    started, solved, failed = time.perf_counter(), 0, 0
    for T in np.linspace(float(t_lo), float(t_hi), int(n_t)):
        for x in np.linspace(float(x_lo), float(x_hi), int(n_x)):
            if _emit_point(models, elements, T, x, None, emit):
                solved += 1
            else:
                failed += 1
    return _sweep_summary(int(n_t) * int(n_x), solved, failed, started)
