<div align="center">

![discopt-calphad logo](images/logo.png)

</div>

# discopt-calphad

Certified CALPHAD phase equilibrium as a discopt plugin (`import discopt.calphad`).

**Method.** A restricted multiphase Gibbs-energy minimization over the active
composition sets is solved locally. Its elemental-balance multipliers are the
chemical potentials μ. For every phase, discopt's global solver then minimizes
the tangent-plane distance Ψ(y) = G(y) − μ·A(y):

- a negative upper bound adds a new composition set (this is how miscibility
  gaps are found);
- a lower bound ≥ −ε for every phase certifies the result to within ε·RT
  per mole of formula units.

**Status.** Pre-alpha.

- **MVP 1:** a binary regular solution with a miscibility gap.
- **MVP 2:** binary systems with several phases: Redlich–Kister solutions,
  line compounds, T-dependent energies, and phase appearance and
  disappearance.
- **MVP 3:** TDB import and export into a symbolic `ThermoDatabase` (SymPy
  expressions with named functions kept), and binary phase models built from
  it. Unsupported models (magnetic, order–disorder, ionic) raise with the
  reason.
- **MVP 4:** compound-energy-formalism phases (`CEFPhase`: several sublattices,
  endmembers, binary/reciprocal/ternary interactions, vacancies) in any number
  of elements, checked against pycalphad. Pricing stops as soon as its answer
  is decided: LB ≥ −ε or UB < −ε. Open: when a two-sublattice phase is
  *active*, its bound stalls near −1e-4, so the result is
  `GLOBAL_SEARCH_INCOMPLETE`.
- **Fitting (issue #2):** weighted linear least squares of TDB parameters
  (symbolic T-forms) to thermochemical, activity and tie-line data. Results
  include the covariance, a fitted TDB, and certified equilibrium checks.

See `docs/architecture.md` for the design and
`docs/symbolic-diagrams-fitting.md` for the planned symbolic model, phase
diagram, and fitting layers.

```python
from discopt.calphad import RegularSolution, equilibrium

phase = RegularSolution("LIQ", omega=3.0)   # G/RT, omega = Omega/RT
res = equilibrium(phase, x=0.5, eps=1e-5)
print(res.certificate, [cs.x for cs in res.composition_sets], res.mu)
```

```python
from discopt.calphad import BinarySolution, StoichiometricPhase, TPoly, equilibrium

# energies in J/mol; TPoly(a, b, c, d, e) = a + bT + cT ln T + dT^2 + e/T
liq = BinarySolution("LIQUID", g=(TPoly(10000, -10), TPoly(15000, -10)), L=(-5000,))
fcc = BinarySolution("FCC_A1", L=(20000, 3000))
ab = StoichiometricPhase("AB", ("A", "B"), (0.5, 0.5), g=-5000)
res = equilibrium([liq, fcc, ab], x=0.3, T=1000, eps=1e-5)
print(res.summary())   # FCC_A1 + AB, CERTIFIED_TO_TOLERANCE
```

```python
from discopt.calphad import ThermoDatabase, equilibrium

db = ThermoDatabase.from_tdb("examples/data/alzn_mey.tdb")
print(db.applicability(("AL", "ZN")))        # which phases can be modelled, and why not
phases = db.phase_models(("AL", "ZN"))
print(phases["LIQUID"].L[0].expr())          # 10465.5 - 3.39259*T  (symbolic, SymPy)
db.add_tdb("PARAMETER G(LIQUID,AL,ZN;1) 298.15 -500; 6000 N !")   # edit in TDB syntax
res = equilibrium(list(db.phase_models(("AL", "ZN")).values()), x=0.5, T=500)
print(db.to_tdb()[:200])
```

```python
from discopt.calphad import ThermoDatabase, equilibrium

db = ThermoDatabase.from_tdb("examples/data/cumg_nims.tdb")
phases = db.phase_models(("CU", "MG"))
cu2mg = phases["CU2MG"]                      # CEFPhase (CU,MG)2(CU,MG)1
print(cu2mg.gibbs_expr())                    # SymPy in y0_CU, y1_MG, ..., T, R
res = equilibrium(list(phases.values()), x=0.75, T=700)
print(res.summary())                         # FCC_A1 + CU2MG
```

```python
from discopt.calphad import ThermoDatabase
from discopt.calphad.fit import Dataset, fit

db = ThermoDatabase.from_tdb("examples/data/alzn_mey.tdb")
data = Dataset.from_json("examples/data/alzn_synthetic.json")   # HM_MIX, SM_MIX, ACR, ZPF
res = fit(db, data, ("AL", "ZN"), {"L(FCC_A1,AL,ZN;0)": "a + b*T", "L(LIQUID,AL,ZN;0)": "a + b*T"})
print(res.summary())                         # values ± σ, residuals by data kind
open("alzn_fit.tdb", "w").write(res.to_tdb())
```

## Development

```bash
uv sync
uv run pytest
uv run python examples/regular_solution_gap.py   # writes examples/regular_solution_gap.png
uv run python examples/alzn_tdb.py               # TDB -> certified Al–Zn equilibria
uv run python examples/cumg_cef.py               # CEF Laves phase in Cu–Mg
uv run python examples/alzn_fit.py               # fit Al–Zn parameters to data
uv run python benchmarks/simplex_bound_study.py  # pricing formulations for 2-sublattice phases
```

pycalphad is a dev dependency used only as a test oracle; the package never imports it.
