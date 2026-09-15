"""Simplex-aware bound study: pricing two-sublattice CEF phases with discopt 0.8.

Compares, at equal time limits, three ways to globally minimize Ψ = G − μ·A:

* stick-breaking   the phase's own box variables (u ∈ [0,1]^n_y), relative gap 1e-6 or 1e-4
* explicit simplex one variable per constituent, Σy = 1 as an equality constraint
* staged           ``price_phase(eps=1e-5)``: stop as soon as LB ≥ −eps or UB < −eps

Cases: a synthetic reciprocal phase (A,B)1(A,B)2 whose minimum is well above zero,
and CU2MG from pycalphad's cumg.tdb at 700 K with μ taken from pure FCC Cu and HCP Mg.
Results are recorded in docs/architecture.md §9.

Run: uv run python benchmarks/simplex_bound_study.py [time_limit]
"""

import sys
import time
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pycalphad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_cef import _recip_phase  # noqa: E402

from discopt.calphad import ThermoDatabase  # noqa: E402
from discopt.calphad.equilibrium.common import DOMAIN_FLOOR, R_GAS, tangent_distance  # noqa: E402
from discopt.calphad.equilibrium.pricing import price_phase  # noqa: E402
from discopt.calphad.models.tfunc import coef  # noqa: E402
from discopt.calphad.ops import EXPR, NUM  # noqa: E402

TL = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
T = 700.0


def stick(p, mu, gap):
    m = dm.Model("stick")
    u = [m.continuous(f"u{i}", lb=DOMAIN_FLOOR, ub=1 - DOMAIN_FLOOR) for i in range(p.n_y)]
    m.minimize(tangent_distance(p, u, mu, EXPR, T))
    r = m.solve(gap_tolerance=gap, time_limit=TL)
    return r.status, r.gap_certified, r.node_count, r.bound


def explicit(p, mu, gap):
    m = dm.Model("explicit")
    Y = []
    for s, cons in enumerate(p.constituents):
        if len(cons) == 1:
            Y.append({cons[0]: 1.0})
            continue
        v = {c: m.continuous(f"y{s}_{c}", lb=DOMAIN_FLOOR, ub=1 - DOMAIN_FLOOR) for c in cons}
        m.subject_to(sum(v.values()) == 1.0)
        Y.append(v)
    G = 0.0
    for term in p.terms:
        G = G + (coef(term.value, T) / (R_GAS * T)) * p._term_factor(term, Y)
    for a, d in zip(p.site_ratios, Y):
        for yc in d.values():
            if not isinstance(yc, float):
                G = G + a * EXPR.xlogx(yc)
    amounts = [sum(a * d[e] for a, d in zip(p.site_ratios, Y) if e in d) for e in p.elements]
    m.minimize(G - sum(float(mm) * am for mm, am in zip(mu, amounts)))
    r = m.solve(gap_tolerance=gap, time_limit=TL)
    return r.status, r.gap_certified, r.node_count, r.bound


def staged(p, mu, _gap):
    r = price_phase(p, mu, T=T, eps=1e-5, time_limit=TL)
    return f"{r.status}/{r.stages} stage(s)", r.certified(1e-5) or r.destabilizing(1e-5), r.node_count, r.lb


def main():
    db = ThermoDatabase.from_tdb(Path(pycalphad.__file__).parent / "tests/databases/cumg.tdb")
    ph = db.phase_models(("CU", "MG"))
    mu_cumg = np.array([ph["FCC_A1"].gibbs([1.0], T=T), ph["HCP_A3"].gibbs([0.0], T=T)])
    cases = [("reciprocal (A,B)1(A,B)2", _recip_phase(), np.array([-2.0, -1.5])), ("CU2MG", ph["CU2MG"], mu_cumg)]
    variants = [
        ("stick-breaking, gap 1e-6", stick, 1e-6),
        ("explicit simplex, gap 1e-6", explicit, 1e-6),
        ("stick-breaking, gap 1e-4", stick, 1e-4),
        ("explicit simplex, gap 1e-4", explicit, 1e-4),
        ("staged, eps 1e-5", staged, None),
    ]
    print(f"time limit {TL:g} s per solve; 'decided' = gap closed (solvers) or sign decided (staged)")
    for cname, p, mu in cases:
        u = [v.ravel() for v in np.meshgrid(np.linspace(0, 1, 801), np.linspace(0, 1, 801))]
        print(f"\n== {cname}: grid min Ψ ≈ {float(np.min(tangent_distance(p, u, mu, NUM, T))):.6f}")
        for vname, fn, gap in variants:
            t0 = time.perf_counter()
            status, decided, nodes, bound = fn(p, mu, gap)
            print(
                f"  {vname:28s} {time.perf_counter() - t0:6.1f} s  {status!s:24s} decided={decided!s:5s} "
                f"nodes={nodes:6d}  bound={bound}"
            )


if __name__ == "__main__":
    main()
