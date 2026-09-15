# discopt-calphad

![discopt-calphad logo](assets/logo.png)

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
- **MVP 2:** binary systems with several phases: Redlich–Kister solutions, line
  compounds, T-dependent energies, and phase appearance and disappearance.
- **MVP 3:** TDB import and export into a symbolic `ThermoDatabase`, and binary
  phase models built from it.
- **MVP 4:** compound-energy-formalism phases in any number of elements, checked
  against pycalphad, with staged pricing. Open: when a two-sublattice phase is
  *active*, its bound stalls near −1e-4, so the result is `GLOBAL_SEARCH_INCOMPLETE`.
- **Fitting:** weighted linear least squares of TDB parameters to
  thermochemical, activity and tie-line data.

## Quickstart

```bash
uv sync
uv run pytest -m "not slow"
```

```python
from discopt.calphad import RegularSolution, equilibrium

phase = RegularSolution("LIQ", omega=3.0)  # G/RT, omega = Omega/RT
res = equilibrium(phase, x=0.5, eps=1e-5)
print(res.certificate, [cs.x for cs in res.composition_sets], res.mu)
```

```python
from discopt.calphad import ThermoDatabase, equilibrium

db = ThermoDatabase.from_tdb("examples/data/alzn_mey.tdb")
print(db.applicability(("AL", "ZN")))  # which phases can be modelled, and why not
phases = db.phase_models(("AL", "ZN"))
print(phases["LIQUID"].L[0].expr())  # 10465.5 - 3.39259*T  (symbolic, SymPy)
res = equilibrium(list(phases.values()), x=0.5, T=500)
print(res.summary())
```

```python
from discopt.calphad import ThermoDatabase
from discopt.calphad.fit import Dataset, fit

db = ThermoDatabase.from_tdb("examples/data/alzn_mey.tdb")
data = Dataset.from_json("examples/data/alzn_synthetic.json")  # HM_MIX, SM_MIX, ACR, ZPF
res = fit(db, data, ("AL", "ZN"), {"L(FCC_A1,AL,ZN;0)": "a + b*T", "L(LIQUID,AL,ZN;0)": "a + b*T"})
print(res.summary())  # values ± σ, residuals by data kind
```

## Where to go next

- [Examples](examples.md): the scripts in `examples/` with their figures.
- [Architecture](architecture.md): the formulation, certificates, and roadmap.
- [Symbolic layer, diagrams, fitting](symbolic-diagrams-fitting.md): the design
  of the symbolic model, phase diagrams and fitting.
- [API reference](api.md).

pycalphad is a dev dependency used only as a test oracle; the package never imports it.
