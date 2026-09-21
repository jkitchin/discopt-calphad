"""Speed and agreement: discopt-calphad against pycalphad on the same TDB, phases, and conditions.

Timed per condition:

* pycalphad cold    the first ``pycalphad.equilibrium`` call for that system in this process (builds the models)
* pycalphad warm    the same call again
* discopt none      ``equilibrium(certify="none")``: local pricing, no certificate
* discopt final     ``equilibrium(certify="final")``: certified (global pricing at the end), ``time_limit`` per phase

Then a batch: one pycalphad call over an x grid against a loop of ``certify="none"`` solves.
Agreement is checked on the stable phases, GM (J/mol of atoms) and μ (J/mol).

Run: uv run python benchmarks/pycalphad_comparison.py [time_limit]
"""

import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pycalphad
from pycalphad import Database
from pycalphad import equilibrium as pyeq
from pycalphad import variables as v

import discopt
from discopt.calphad import R_GAS, ThermoDatabase, equilibrium

warnings.filterwarnings("ignore")
TL = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
PYDB = Path(pycalphad.__file__).parent / "tests" / "databases"

SYSTEMS = [
    ("Al-Zn", PYDB / "alzn_mey.tdb", ("AL", "ZN"), [(0.5, 500.0), (0.3, 600.0), (0.8, 700.0)]),
    ("Cu-Mg", PYDB / "cumg.tdb", ("CU", "MG"), [(0.75, 700.0), (0.62, 900.0), (0.2, 700.0)]),
]


def timed(f, *args, **kwargs):
    t0 = time.perf_counter()
    out = f(*args, **kwargs)
    return out, time.perf_counter() - t0


def py_eq(dbf, els, names, x, T):
    return pyeq(dbf, [*els, "VA"], names, {v.X(els[0]): x, v.T: T, v.P: 101325, v.N: 1})


def py_phases(eq, n):
    rows = np.asarray(eq.Phase).reshape(n, -1)
    return [sorted({str(p) for p in row if str(p)}) for row in rows]


def ours(res):
    """Phases, GM in J/mol of atoms, μ in J/mol."""
    return res.phase_names, res.G * R_GAS * res.T / float(res.b.sum()), res.mu * R_GAS * res.T


def main():
    print(f"pycalphad {pycalphad.__version__}, discopt {discopt.__version__}, Python {platform.python_version()}")
    print(f"time_limit per phase for certify='final': {TL:g} s\n")
    print(
        f"{'system':6s} {'x':>5s} {'T':>6s} | {'py cold':>8s} {'py warm':>8s} | {'none':>7s} {'final':>8s} | "
        "phases                  dGM J/mol  max dmu J/mol  certificate (final)"
    )
    for label, tdb, els, points in SYSTEMS:
        dbf = Database(str(tdb))
        models = ThermoDatabase.from_tdb(tdb).phase_models(els)
        names, phases = sorted(models), list(models.values())
        for i, (x, T) in enumerate(points):
            _, t_cold = timed(py_eq, dbf, els, names, x, T)
            eq, t_warm = timed(py_eq, dbf, els, names, x, T)
            fast, t_none = timed(equilibrium, phases, x=x, T=T, certify="none")
            full, t_final = timed(equilibrium, phases, x=x, T=T, certify="final", time_limit=TL)
            (pp,), pg, pm = py_phases(eq, 1), float(np.asarray(eq.GM).ravel()[0]), np.asarray(eq.MU).ravel()
            (op, og, om), (fp, fg, fm) = ours(full), ours(fast)
            same = f"{'+'.join(pp)}" if pp == op == fp else f"py {pp} | final {op} | none {fp}"
            dg = max(abs(og - pg), abs(fg - pg))
            dmu = max(np.max(np.abs(om - pm)), np.max(np.abs(fm - pm)))
            note = "  (includes pycalphad model build)" if i == 0 else ""
            print(
                f"{label:6s} {x:5.2f} {T:6.0f} | {t_cold:8.3f} {t_warm:8.3f} | {t_none:7.3f} {t_final:8.2f} | "
                f"{same:22s} {dg:9.3f}  {dmu:13.3f}  {full.certificate.level.name}{note}"
            )

        xs = np.linspace(0.05, 0.95, 19)
        T = points[0][1]
        batch, t_py = timed(py_eq, dbf, els, names, xs, T)
        t0 = time.perf_counter()
        results = [equilibrium(phases, x=float(xi), T=T, certify="none") for xi in xs]
        t_ours = time.perf_counter() - t0
        gm_py = np.asarray(batch.GM).reshape(-1)
        gm_ours = np.array([ours(r)[1] for r in results])
        agree = sum(p == r.phase_names for p, r in zip(py_phases(batch, len(xs)), results))
        print(
            f"{label:6s} batch of {len(xs)} x at T={T:g}: pycalphad {t_py:.2f} s, discopt none {t_ours:.2f} s; "
            f"phases agree at {agree}/{len(xs)}; max dGM {np.max(np.abs(gm_ours - gm_py)):.3f} J/mol\n"
        )


if __name__ == "__main__":
    main()
