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
- **Pricing modes:** `certify="final"` (default) prices locally while
  generating sets and globally once at the end. `certify="none"` skips the
  global step, and `res.certify()` adds it later.

## Related projects

- **[pycalphad](https://pycalphad.org)** is an open-source Python library for
  CALPHAD models and equilibria. It reads TDB databases, supports a broad range
  of models (magnetic, order–disorder, ionic liquids and more), and is mature
  and widely used. We use it as a test oracle for our Gibbs energies and
  equilibria.
- **[ESPEI](https://espei.org)** builds on pycalphad. It generates CALPHAD
  parameters from data and quantifies their uncertainty with Bayesian (MCMC)
  sampling.
- **[OpenCalphad](https://www.opencalphad.com)** is an open-source Fortran
  equilibrium software package.
- **Thermo-Calc, FactSage and Pandat** are the commercial standards. They come
  with assessed multicomponent databases.

For real alloy work today, use those tools. discopt-calphad is a pre-alpha
research code. It covers far fewer models, and it is much slower. On the same
TDB files (`benchmarks/pycalphad_comparison.py`, pycalphad 0.11.2, Apple M4 Pro),
a warm pycalphad point takes about 5 ms. An uncertified solve here
(`certify="none"`) takes 20–80 ms, and a certified one takes 6–65 s. For a
19-point composition sweep, pycalphad is about 50–90× faster than
`certify="none"`. Where both codes find the same phases, GM agrees to within
0.02 J/mol and μ to within 0.1 J/mol.

## Why another CALPHAD code

A CALPHAD equilibrium is a nonconvex minimization. The established codes find
the global minimum well in practice, typically by sampling each phase's
composition space on a grid and then refining locally. But the answer carries
no proof: nothing shows that a composition between the grid points, or a narrow
miscibility gap, lowers G further. discopt-calphad asks whether the equilibrium
can come with a **certificate**. The certificate is a rigorous lower bound, from
global optimization, showing that no composition of any candidate phase lowers
G by more than ε.

## What is different here

- **Certified or explicitly not.** Every result carries a certificate level and
  a bound on G_found − G_global. When the global search cannot close the gap,
  the result says `GLOBAL_SEARCH_INCOMPLETE`, with the bound it did reach.
- **Grid-free correctness.** Grids may supply starting points, but correctness
  rests on the global pricing bound, never on them.
- **Symbolic models.** SymPy expressions that keep TDB function names. You can
  inspect, differentiate, and edit them, and write them back to TDB.
- **Part of an optimization language.** The equilibrium subproblems are discopt
  models. They can later go inside constrained fitting, alloy design, and
  checks that a fit predicts no spurious phases.
- **Fitting tied to certification.** Fitted databases are checked with certified
  equilibria at the data conditions.

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
from discopt.calphad import databases, equilibrium

db = databases.load("alzn_mey")        # fetched once from its pinned URL, hash-checked
print(db.applicability(("AL", "ZN")))  # which phases can be modelled, and why not
phases = db.phase_models(("AL", "ZN"))
print(phases["LIQUID"].L[0].expr())  # 10465.5 - 3.39259*T  (symbolic, SymPy)
res = equilibrium(list(phases.values()), x=0.5, T=500)
print(res.summary())
```

```python
from discopt.calphad import databases
from discopt.calphad.fit import Dataset, fit

db = databases.load("alzn_mey")
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
