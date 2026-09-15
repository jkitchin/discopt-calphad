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
  it. Unsupported models (CEF, magnetic, order–disorder) raise with the
  reason.

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

## Development

```bash
uv sync
uv run pytest
uv run python examples/regular_solution_gap.py   # writes examples/regular_solution_gap.png
uv run python examples/alzn_tdb.py               # TDB -> certified Al–Zn equilibria
```

pycalphad is a dev dependency used only as a test oracle; the package never imports it.
