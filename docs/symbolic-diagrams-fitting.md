# discopt-calphad: symbolic layer, syntaxes, phase diagrams, fitting (draft 1)

Companion to `architecture.md`. Status: design proposal, no code yet.

The package has two headline features of equal weight:

1. **A symbolic, structure-preserving thermodynamic model** that users can inspect,
   differentiate, print, export and fit.
2. **Certified equilibrium** (POUNCE restricted solve + discopt global pricing).

Phase diagrams and parameter fitting are built on both.

## 1. Why a symbolic layer, and what is new relative to pycalphad

pycalphad already builds symbolic Gibbs energies. Checked on pycalphad 0.11.1:
`Model(db, comps, phase).GM` is a symengine expression, split into named
contributions `ref, idmix, xsmix, mag, 2st, ein, vol, ord`, and it can be
differentiated (`GM.diff(v.T)`). Printing it for a binary RK phase shows how it
*flattens* the model:

```text
... + 1.0*(FCC0A*FCC0B*(20000.0 - 3.0*T) + 1000.0*(FCC0A - FCC0B)*FCC0A*FCC0B)/(FCC0A + FCC0B)
    + 8.3145*T*(1.0*Piecewise((FCC0A*log(FCC0A), 1e-15 < FCC0A), (0, True)) + ...)/(FCC0A + FCC0B)
```

- Redlich–Kister is expanded into arithmetic.
- x ln x becomes a `Piecewise` with a `1e-15` cutoff.
- The whole model is normalized by site sums.
- The expression exists to generate fast numerical code.

What discopt-calphad adds:

| Property | pycalphad | discopt-calphad |
|---|---|---|
| Symbolic G | yes (symengine) | yes (SymPy) |
| CALPHAD structure kept (XLogX, RK, CEF endmember, piecewise-T) | lowered immediately | kept as unevaluated function nodes |
| x ln x at 0 | Piecewise, 1e-15 cutoff | exact XLogX node, derivative `log(x)+1`, lowered to discopt `entropy` |
| Same expression drives certified global bounds | no | yes |
| Symbolic → TDB round trip | n/a | yes (the structure is still there) |
| Parameters as first-class symbols (for fitting) | via `parameters=` substitution | yes; G is linear in most parameters, and that is used |

## 2. Symbolic representation

**Engine: SymPy.** It matches discopt-mkm, which uses SymPy. discopt itself uses
SymPy internally to derive envelopes (`_relax/symbolic/envelope_deriver.derive_envelope`
takes a univariate SymPy expression). SymPy is not a discopt dependency, so we
add it. Speed for large multicomponent databases is a known risk: pycalphad
moved to symengine for this. Mitigations: per-phase caching, lowering once per
phase and not per point, symengine as an optional accelerator.

**CALPHAD nodes as unevaluated `sympy.Function` subclasses**, each with `fdiff`,
LaTeX printing and a TDB printer:

| Node | Meaning | Lowers to (discopt) |
|---|---|---|
| `XLogX(y)` | y ln y, = 0 at y = 0 | `FunctionCall("entropy", y)` |
| `RedlichKister(yi, yj, (L0, L1, …))` | yi yj Σ L_k (yi−yj)^k | polynomial (later: dedicated node) |
| `Endmember(phase, "AL:NI:VA")` | Π_s y_{s,c_s} · G_endmember | product (later: simplex-aware node) |
| `Interaction(phase, "AL,NI:VA", k)` | CEF interaction term | product × RK factor |
| `PiecewiseT(((T0,T1,expr),…))` | TDB temperature ranges | `if_else` chain, or fixed branch when T is fixed |
| `GHSER("AL")`, `Function("GHSERAL")` | named TDB functions, kept by name | inlined expression |
| `Magnetic(...)` (later) | IHJ magnetic model | — |

Symbols: `T`, `P`, site fractions `Y(phase, sublattice, species)`, mole fractions
`X(phase, element)`, and **parameter symbols** such as `L("FCC_A1", "AL,NI:VA", 0)`.
Parameter symbols carry values from the database, or they are free, for fitting.

**Things users can do with a phase model:**

```python
fcc = db.phases["FCC_A1"]
fcc.GM                     # SymPy expression with CALPHAD nodes intact
fcc.GM.expand_calphad()    # the same expression, fully lowered (like pycalphad)
fcc.contributions          # {'ref': ..., 'idmix': ..., 'xsmix': ...}
fcc.HM, fcc.SM, fcc.CPM    # H = G − T ∂G/∂T, S = −∂G/∂T, Cp = −T ∂²G/∂T²  (symbolic)
fcc.MU("AL")               # partial molar Gibbs energy (CEF formula, symbolic)
fcc.hessian_x()            # stability matrix; spinodal = det = 0
fcc.subs(T=1000)           # partial evaluation
fcc.linear_in(params)      # G = g0(y,T) + Σ θ_k φ_k(y,T): the basis for fitting
fcc.latex(), fcc.to_tdb()  # print or export
fcc.lower(model, y_vars)   # discopt Expression (one lowering, used everywhere)
```

Numbers (plots, grids, test references) are always computed by evaluating the
*lowered discopt expression* or its SymPy form through a single lambdify path, so
no formula is hand-written twice.

## 3. Syntaxes: several front ends, one representation

All four build the same `PhaseModel` with the same SymPy nodes.

```python
# (a) TDB file
db = Database("alzn.tdb")

# (b) TDB lines, pasted
db.add_tdb("PARAMETER L(FCC_A1,AL,ZN:VA;0) 298.15 +7297+.47512*T; 6000 N !")

# (c) CALPHAD parameter notation in Python
fcc = db.phase("FCC_A1", "(AL,ZN)1 (VA)1")
fcc.G["AL:VA"] = GHSER("AL")
fcc.L["AL,ZN:VA", 0] = 7297 + 0.47512*T

# (d) Free-form expression (non-standard models; still certifiable)
with db.phase("MYPHASE", "(A,B)1") as p:
    xA, xB = p.X("A"), p.X("B")
    p.GM = p.ref() + R*T*(XLogX(xA) + XLogX(xB)) + W*xA*xB*(1 + c*xB)
```

Calculations accept keyword conditions (`T=, P=, N=, X={...}, W={...}`),
Thermo-Calc/OpenCalphad condition strings (`"T=1000 P=1e5 N=1 X(ZN)=0.4"`), and
pycalphad-style `{v.T: ..., v.X("ZN"): ...}` through our own `variables` module.
Results use CALPHAD state-variable names: `res["NP(FCC_A1#2)"]`, `res.X`,
`res.MU`, `res.GM`, plus `res.certificate`.

## 4. Phase diagrams

Goal: one line for common diagrams, with certification status visible.

```python
from discopt.calphad import binary, ternary, isopleth, gx_plot

pd = binary(db, ["AL", "ZN"], T=(300, 1000), X="ZN", certify=True)
pd.plot()                         # boundaries, labels, invariants; uncertified points marked
pd.boundaries, pd.invariants      # data: tie-line endpoints, eutectic/peritectic T and compositions
pd.to_dataframe()                 # for comparison with experiments

ternary(db, ["AL", "CU", "MG"], T=700).plot()     # isothermal section
isopleth(db, ["FE", "C", "CR"], X={"CR": 0.05}, vary="C", T=(800, 1800)).plot()

gx_plot(db, ["AL", "ZN"], T=600)  # G–x curves, common tangents, convex hull (teaching/debugging)
```

Algorithm (grid-free where possible):

1. **Seed:** certified equilibria on a coarse (T, x) grid. The grid only finds
   starting points; correctness comes from pricing at every point.
2. **Trace** each two-phase boundary by continuation. Coexistence equations
   `MU_i^α(y^α,T) = MU_i^β(y^β,T)` come *symbolically* from the model and are
   solved by POUNCE, warm-started from the previous step. Each traced point is
   priced, and the diagram is optionally certified.
3. **Detect invariants** when pricing finds a third phase with zero driving force
   along a boundary. Refine T with a bracketing solve.
4. **Report** per-segment certificate levels. `plot()` draws uncertified pieces
   dashed.

Plotting uses matplotlib as an optional extra (`[plot]`, as in discopt-thermo).
The diagram object is plain data, so other backends are possible.

## 5. Fitting parameters to data

> **Status (issue #2, closed).** Steps 1 and 2 without errors in variables
> are implemented in `src/discopt/calphad/fit.py`, with an ESPEI JSON reader in
> `src/discopt/calphad/io/espei.py`; see `examples/alzn_fit.py`,
> `examples/cumg_espei_fit.py` (real Cu–Mg data) and `tests/test_fit.py`. The
> implemented API differs from the sketch below:
>
> ```python
> data = Dataset().add_thermo("HM_MIX", "LIQUID", T=1000, x=[0.1, 0.5], value=[...], sigma=200)
> data.add_activity("ZN", "LIQUID", T=1000, x=0.4, value=0.55, sigma=0.02)
> data.add_equilibrium(700, {"LIQUID": 0.21, "FCC_A1": 0.50}, sigma=200)   # x = x(first element)
> res = fit(db, data, ("AL", "ZN"), {"L(FCC_A1,AL,ZN;0)": "a + b*T", "L(LIQUID,AL,ZN;0)": "a"})
> res.summary(); res.database.to_tdb(); res.check_equilibria()
> ```
>
> Free parameters are TDB labels with a symbolic T-form that is linear in its
> coefficients. The site-fraction factors φ_p are found by probing the phase
> models, so no separate `linear_in` is needed. H, S and Cp come from symbolic
> T-derivatives. μ-based data (activities, ZPF) need a binary system. Boundary
> points without a measured partner composition use the driving-force
> residual (the unmeasured phase touches the measured phase's tangent), and
> μ of a multi-sublattice phase uses its internal equilibrium. Both are
> nonlinear in θ through a minimizer, and the fit iterates linear solves
> (Gauss–Newton with the exact, envelope-theorem Jacobian). Not yet done,
> each now its own issue:
> - ~~errors in variables (step 2's nonlinear part): #8~~ done:
>   `fit(..., errors_in_variables=True)`, solved with scipy's trust-region
>   least squares on the existing rows, since `discopt.estimate` needs an
>   algebraic model and the tangent rows are numeric. Calibration is checked in
>   `benchmarks/eiv_coverage.py`;
> - ~~global consistency (step 3): #9~~ done: `fit(..., consistency="global")`
>   prices every unlisted phase at every phase-equilibrium observation, and
>   adds cuts until all certified lower bounds are ≥ −ε. Phases listed in an
>   observation are not checked there, because their deviation is that
>   observation's own residual;
> - FIM-based design (step 4): #10;
> - μ-based data in ternaries: #11.

```python
from discopt.calphad.fit import Dataset, fit

data = Dataset.from_espei("datasets/")            # or build by hand:
data.add_tieline(T=650, phases={"FCC_A1": {"ZN": 0.17}, "HCP_A3": {"ZN": 0.97}}, sigma=0.01)
data.add_thermo("HM_MIX", phase="LIQUID", T=1000, X={"ZN": [0.1, 0.3, 0.5]}, values=[...], sigma=200)
data.add_invariant("eutectic", T=654, X={"ZN": 0.89}, phases=["LIQUID", "FCC_A1", "HCP_A3"])

free = [fcc.L["AL,ZN:VA", 0], fcc.L["AL,ZN:VA", 1], liq.L["AL,ZN", 0]]
res = fit(db, data, parameters=free, consistency="global")
res.summary()               # values, covariance, correlation, residuals by data type
res.database.to_tdb("alzn_fit.tdb")
pd_fit = binary(res.database, ["AL", "ZN"], T=(300, 1000), X="ZN"); pd_fit.plot(data=data)
```

Formulation, taking each data type in order of difficulty:

1. **Thermochemical data** (HM_MIX, SM_MIX, CPM, activities, at stated composition
   and phase). The model output is a symbolic derivative of G. Standard CALPHAD
   parameters appear *linearly* (`fcc.linear_in(free)`), so this is weighted
   linear least squares: exact, unique if identifiable, no initial guess needed.
2. **Tie-line / phase-boundary (ZPF) data** with measured compositions. The
   residuals are the equilibrium conditions MU_i^α − MU_i^β (or tangent-plane
   distances) at the measured compositions. These are linear in the parameters,
   giving a linear least-squares warm start. Errors-in-variables (compositions
   also uncertain, as free variables) make it a nonlinear least-squares problem.
   It is posed with `discopt.estimate` (`Experiment`/`ExperimentModel`,
   `estimate_parameters`), which also gives a FIM/covariance.
3. **Global consistency (new).** A fit can match the data at data points while
   predicting a spurious miscibility gap or an unobserved stable phase elsewhere.
   This is a well-known CALPHAD fitting problem. `consistency="global"` adds the
   semi-infinite constraints

   Ψ_φ(y; θ, μ_data) ≥ 0  for all y ∈ Y_φ and all phases φ not observed at each data point,

   and handles them by cutting planes:
   - fit;
   - globally price every phase at every data condition (discopt);
   - add the most violating y as a new constraint;
   - refit;
   - stop when every lower bound ≥ −ε.

   The result is a parameter set *certified* not to predict phases that
   contradict the data at the data conditions. discopt's `bilevel` module cannot
   do this: it replaces the follower with its KKT conditions and requires a
   convex follower, and the equilibrium problem is not convex.
4. **Uncertainty and design.** The covariance comes from the estimation FIM.
   discopt-doe (`discopt.doe`) chooses which (T, x) measurements best reduce
   uncertainty in the parameters or in a predicted boundary. The Experiment
   interface is the same one.

ESPEI dataset import is a convenience for reusing existing curated data. We do
not depend on ESPEI; its MCMC approach is a comparison baseline, not a component.

## 6. Additions to the roadmap

- MVP 1 includes the symbolic node set for the regular solution (`XLogX`, `RedlichKister`),
  `GM/HM/SM/MU/hessian_x`, LaTeX, and `lower()`, with tests comparing symbolic derivatives
  against JAX derivatives of the lowered discopt expression.
- MVP 2 adds `gx_plot` and a binary diagram for simple phases (grid seeding + tracing).
- A fitting track begins at MVP 2: thermochemical linear fit, then ZPF, then global consistency
  on a synthetic binary whose true parameters are known.
- MVP 3 (TDB) adds TDB export from the symbolic form and round-trip tests.

## 7. Open issues added by this document

1. SymPy speed and memory for multicomponent CEF phases, and whether symengine is needed.
2. Whether RK and endmember nodes should become dedicated discopt nodes or stay
   plugin-side and lower to primitives. This depends on the discopt relaxation hook.
3. How robust boundary tracing is near critical points and congruent points, where
   the coexistence Jacobian is singular. It needs a bordered or pseudo-arclength
   continuation.
4. Global consistency cost: phases × data points global solves per cutting-plane round.
   Pruning phases that already have a certified bound from the last round must stay sound.
5. Weighting data of different types (energies, compositions, temperatures) in one objective.
