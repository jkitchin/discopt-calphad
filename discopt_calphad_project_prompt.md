# Project: A discopt-based globally certified CALPHAD equilibrium solver

I want to design and implement an open-source CALPHAD thermodynamic equilibrium package whose distinguishing feature is that thermodynamic models are represented directly in the **discopt modeling language**, allowing us to use both fast local nonlinear optimization and deterministic global optimization with specialized thermodynamic relaxations.

The package should be compatible with existing CALPHAD thermodynamic databases, especially standard `.tdb` files, rather than introducing a new database ecosystem.

The long-term goal is a solver architecture that combines:

1. existing CALPHAD database compatibility,
2. a discopt-native equation representation,
3. POUNCE as the fast sparse nonlinear equilibrium solver,
4. discopt as the deterministic global optimization/certification engine,
5. CALPHAD-specific convex/concave envelopes and relaxations,
6. phase generation/stability testing instead of brute-force monolithic MINLP,
7. rigorous detection of miscibility gaps and multiple composition sets,
8. optional certificates that a computed equilibrium is globally stable.

The design should borrow the useful modeling ideas of `discopt-mkm`: users should be able to construct thermodynamic models programmatically from equations, species, phases, reactions, constraints, and parameters, while imported CALPHAD databases should compile into exactly the same internal representation.

Do not simply wrap pycalphad. pycalphad may be used as a reference implementation and validation oracle, and its TDB parser may initially be studied or optionally reused if licensing permits, but the core thermodynamic representation and optimization layer should be independent and discopt-native.

---

# 1. Mathematical problem

At fixed temperature \(T\), pressure \(P\), and total elemental composition \(b\), equilibrium is the solution of

\[
\min_{\{n_\alpha,y_\alpha\}}
G =
\sum_{\alpha}
n_\alpha G_\alpha(T,P,y_\alpha)
\]

subject to

\[
\sum_\alpha n_\alpha A_\alpha(y_\alpha)=b,
\]

\[
n_\alpha\ge0,
\]

and the phase-specific site-fraction constraints

\[
y_{\alpha,s,i}\ge0,
\qquad
\sum_i y_{\alpha,s,i}=1.
\]

Here:

- \(\alpha\) denotes a phase/composition-set instance,
- \(n_\alpha\) is its amount,
- \(y_\alpha\) contains its internal site fractions,
- \(A_\alpha(y_\alpha)\) maps site fractions to elemental composition,
- \(G_\alpha(T,P,y_\alpha)\) is the molar Gibbs energy.

For a fixed set of phase instances this is a sparse, generally nonconvex NLP.

POUNCE should solve this **restricted equilibrium problem**.

The Lagrange multipliers on elemental balances give chemical potentials \(\mu\).

For every possible phase \(\phi\), define its stability/pricing problem

\[
\Delta G_\phi^*
=
\min_{y\in Y_\phi}
\left[
G_\phi(T,P,y)
-
\mu^T A_\phi(y)
\right].
\]

This should be solved globally by discopt.

If

\[
\Delta G_\phi^* < -\epsilon,
\]

then the current assemblage is not globally stable. Add the corresponding composition set and resolve the restricted equilibrium problem.

If a rigorous global lower bound satisfies

\[
LB_\phi \ge -\epsilon
\]

for every phase, then the equilibrium should be reported as **globally certified to tolerance \(\epsilon\)**.

This decomposition is the central algorithm.

Conceptually:

```text
load database
      |
compile phase Gibbs models to discopt expressions
      |
construct initial composition sets
      |
      v
+----------------------------+
| POUNCE restricted          |
| Gibbs minimization         |
+----------------------------+
      |
      | chemical potentials
      v
+----------------------------+
| GLOBAL PHASE PRICING       |
| one problem per phase      |
| solved by discopt          |
+----------------------------+
      |
      +---- negative driving force ----> add composition set
      |                                  and return to POUNCE
      |
      +---- all globally >= -tol ------> certified equilibrium
```

Treat this primarily as a **decomposition / nonlinear column-generation problem**, not as a monolithic MINLP.

---

# 2. Compatibility requirements

The first version must support ordinary open CALPHAD `.tdb` databases.

Design an intermediate representation so that additional formats can later be supported without modifying the optimization layer.

The parser/database layer should understand at least:

- `ELEMENT`
- `SPECIES`
- `FUNCTION`
- `PHASE`
- `CONSTITUENT`
- `PARAMETER`
- phase sublattice definitions
- site ratios
- Gibbs-energy parameters
- interaction parameters
- piecewise temperature expressions
- references/comments where feasible
- magnetic parameters eventually
- ordering/type definitions eventually

Do not require every obscure TDB extension for the MVP.

Use a capability matrix such as:

```text
Feature                         MVP     Later
------------------------------------------------
ELEMENT                         yes
SPECIES                         yes
FUNCTION                        yes
PHASE                           yes
CONSTITUENT                     yes
G parameters                    yes
L parameters                    yes
multiple sublattices            yes
piecewise T functions           yes
magnetic contribution           later
ionic liquids                   later
ordered/disordered coupling     later
aqueous models                  later
pressure-dependent models       later
kinetics/mobility               later
```

Unsupported constructs must produce explicit, informative errors rather than silently changing the thermodynamics.

Implement TDB round-trip tests where practical:

```text
TDB -> internal database -> TDB
```

Semantic equivalence is more important than byte-for-byte equality.

---

# 3. Internal thermodynamic representation

Create a clean intermediate thermodynamic representation that is independent of TDB syntax.

Suggested objects include:

```python
Element
Species
Sublattice
Phase
Constituent
ThermoFunction
ThermoParameter
ReferenceState
ThermoDatabase
CompositionSet
EquilibriumProblem
EquilibriumResult
```

A `Phase` should know:

- name,
- sublattices,
- site ratios,
- allowed species on each sublattice,
- thermodynamic model type,
- parameters,
- Gibbs-energy expression.

Example conceptual API:

```python
phase = Phase(
    "FCC_A1",
    sublattices=[
        Sublattice(["FE", "NI", "CR"], sites=1.0)
    ],
)
```

For a two-sublattice model:

```python
phase = Phase(
    "B2",
    sublattices=[
        Sublattice(["FE", "AL"], sites=0.5),
        Sublattice(["FE", "AL"], sites=0.5),
    ],
)
```

The crucial rule is:

> Every thermodynamic model must eventually compile into a discopt expression graph.

There should not be separate implementations of Gibbs energy for local and global optimization.

Use one expression representation for:

- function evaluation,
- automatic differentiation,
- sparse Jacobian/Hessian construction,
- interval evaluation,
- convex relaxations,
- concave relaxations,
- deterministic global optimization.

---

# 4. discopt-native modeling API

Provide a programmatic API inspired by `discopt-mkm`.

The user should be able to define models without a TDB file.

For example, something conceptually like:

```python
from discopt_thermo import System, Phase

sys = System(elements=["A", "B"])

liq = sys.phase("LIQUID")

xA = liq.fraction("A")
xB = liq.fraction("B")

liq.gibbs(
    xA * GA(T)
    + xB * GB(T)
    + R*T*(xA*log(xA) + xB*log(xB))
    + L(T)*xA*xB
)
```

Then:

```python
result = sys.equilibrium(
    T=1000,
    P=1e5,
    composition={"A": 0.4, "B": 0.6},
)
```

The exact syntax should conform naturally to existing discopt conventions.

Imported TDB models and handwritten models must ultimately generate the same objects and expression graphs.

---

# 5. Separate thermodynamic models from phase instances

Do not equate a phase model with one equilibrium phase occurrence.

For example, one FCC thermodynamic model may have two equilibrium composition sets:

```text
FCC_A1#1
FCC_A1#2
```

because of a miscibility gap.

Represent this explicitly:

```python
PhaseModel("FCC_A1")
```

versus

```python
CompositionSet(
    phase="FCC_A1",
    amount=...,
    site_fractions=...
)
```

There may be zero, one, two, or more composition sets corresponding to one phase model.

This distinction is essential for globally correct phase splitting.

---

# 6. Restricted equilibrium solver using POUNCE

Build the restricted equilibrium NLP from an active set of composition sets.

Decision variables:

- amount of each composition set,
- independent site fractions for each active composition set.

Constraints:

- elemental balances,
- sublattice normalization,
- charge balance when necessary,
- other model-specific constraints.

Bounds:

\[
n_\alpha\ge0,
\qquad
0\le y\le1.
\]

Objective:

\[
\sum_\alpha n_\alpha G_\alpha.
\]

Exploit sparsity aggressively.

Each phase contributes mostly local blocks to the Jacobian/Hessian, with elemental balance constraints coupling the phases.

POUNCE should expose:

- equilibrium phase amounts,
- phase compositions,
- objective Gibbs energy,
- elemental-balance multipliers / chemical potentials,
- KKT residuals,
- active constraints,
- convergence diagnostics.

Implement excellent warm starts because nearby equilibrium calculations in phase-diagram tracing should differ only slightly.

---

# 7. Global phase-stability / pricing problem

For each phase model \(\phi\), solve

\[
\min_y
\Psi_\phi(y)
=
G_\phi(T,P,y)-\mu^TA_\phi(y)
\]

subject to its sublattice/simplex constraints.

Run these phase problems independently, potentially in parallel.

Return:

```python
StabilityResult(
    phase="FCC_A1",
    composition=...,
    upper_bound=...,
    lower_bound=...,
    gap=...,
    globally_solved=True,
)
```

The interpretation should be explicit.

If

```text
upper_bound < -tol
```

a destabilizing composition has been found.

If

```text
lower_bound >= -tol
```

the phase is proven unable to destabilize the current solution.

If neither holds, continue global optimization until the requested tolerance/resource limit is met.

---

# 8. Composition-set generation

If a pricing problem finds

\[
\Psi_\phi(y^*)<-\epsilon,
\]

create a new composition set centered at \(y^*\).

This must work even if the phase already exists in the active set.

Example:

```text
Current equilibrium:
FCC_A1#1

Global pricing finds:
FCC_A1 at a distant composition with ΔG < 0

Add:
FCC_A1#2
```

Then rerun POUNCE.

This should naturally discover miscibility gaps.

Include safeguards to avoid creating duplicate composition sets near an existing one.

---

# 9. CALPHAD-specific relaxation library

Do not treat Gibbs energy as an arbitrary black-box nonlinear expression.

Create special discopt operators/relaxations for common thermodynamic terms.

## 9.1 Ideal mixing

Recognize

\[
x\log x.
\]

Properties:

\[
f(x)=x\log x
\]

is convex for \(x>0\), with the continuous extension

\[
f(0)=0.
\]

Use its exact convex representation/lower bound wherever possible.

Handle \(x=0\) robustly without introducing numerical `log(0)` failures.

Implement a dedicated operator if useful:

```python
xlogx(x)
```

rather than relying on naïve multiplication of `x * log(x)`.

---

# 10. Bilinear and multilinear CEF terms

CEF endmember energies produce products such as

\[
y_i^{(1)}y_j^{(2)}G_{i:j}
\]

and possibly higher-order products across sublattices.

Do not simply recursively apply generic McCormick relaxations if tighter simplex-aware hulls are available.

Research and implement strong convex/concave envelopes for multilinear terms whose variables belong to simplices:

\[
y_i^{(s)}\ge0,
\qquad
\sum_i y_i^{(s)}=1.
\]

Compare:

1. naïve interval bounds,
2. recursive McCormick,
3. multivariate convex-hull formulations,
4. simplex-aware formulations.

Benchmark bound quality.

---

# 11. Redlich-Kister-specific relaxations

Recognize terms of the form

\[
x_Ax_B
\sum_k
L_k(T)
(x_A-x_B)^k.
\]

For a binary substitutional phase with

\[
x_B=1-x_A,
\]

this becomes a univariate polynomial-like function plus entropy.

Create a specialized representation where appropriate.

Investigate:

- exact derivative bounds,
- monotonicity,
- curvature intervals,
- polynomial convex envelopes,
- Bernstein polynomial bounds,
- interval polynomial methods,
- direct convex/concave envelope construction.

Do not assume generic McCormick is optimal.

A key research question is:

> How much tighter can CALPHAD-specific relaxations make deterministic global optimization compared with generic factorable relaxations?

---

# 12. Gibbs-energy convexification

Investigate direct use of thermodynamic convexity.

For a binary phase,

\[
G(x)
\]

may be nonconvex.

The stable multiphase Gibbs energy is related to the lower convex envelope

\[
G^{**}(x).
\]

The ordinary common-tangent construction is the one-dimensional manifestation of this convexification.

Investigate whether this structure can be incorporated directly into the global solver rather than rediscovered solely through branch-and-bound.

Consider:

- lower convex hulls,
- supporting hyperplanes,
- tangent-plane distance,
- Legendre-Fenchel / biconjugate interpretations,
- adaptive generation of supporting planes,
- convexification over composition simplices.

Determine whether phase stability can be certified partly by generating supporting hyperplanes instead of spatially branching everywhere.

This may become one of the central algorithmic contributions.

---

# 13. Reaction-based modeling interface

In addition to conventional CALPHAD phase models, provide a reaction/species interface inspired by `discopt-mkm`.

For example:

```python
A = Species("A")
B = Species("B")
AB = Species("AB")

r1 = Reaction(
    reactants={A: 1, B: 1},
    products={AB: 1},
)
```

However, distinguish clearly between:

- stoichiometric reaction notation,
- equilibrium constraints,
- CALPHAD phase models,
- species chemical potentials.

A reaction should compile to stoichiometry and chemical-potential relations rather than become a completely separate solver subsystem.

Eventually this should make it possible to combine:

- CALPHAD condensed phases,
- gas-phase reactions,
- aqueous or solution chemistry,
- surface chemistry,
- microkinetic models,

inside one discopt equation ecosystem.

Do not attempt all these domains in the MVP.

---

# 14. Avoid a monolithic MINLP initially

Although phase existence could be represented using binaries,

\[
z_\phi\in\{0,1\},
\]

do not make this the primary formulation.

Prefer the complementarity/phase-generation structure:

\[
n_\phi\ge0,
\]

\[
\Delta G_\phi\ge0,
\]

\[
n_\phi\Delta G_\phi=0.
\]

Implement the outer algorithm as active-set/column generation:

```python
while True:
    eq = solve_restricted_equilibrium(active_sets)

    mu = eq.chemical_potentials

    stability = solve_all_phase_pricing_problems(mu)

    if all(p.lower_bound >= -tol for p in stability):
        return CertifiedEquilibrium(...)

    candidate = most_destabilizing_phase(stability)

    active_sets.add(
        CompositionSet(
            candidate.phase,
            candidate.composition
        )
    )
```

Allow multiple destabilizing phases to be added at once as an optional strategy.

---

# 15. Initialization

Global reliability must not depend critically on a good initial phase assemblage.

Implement several initialization strategies:

### Strategy A
Start with obvious pure-element/endmember phases.

### Strategy B
Use coarse thermodynamic sampling analogous to existing CALPHAD algorithms.

### Strategy C
Run global phase pricing immediately from an initial chemical-potential estimate.

### Strategy D
Warm-start from a neighboring equilibrium state.

For phase-diagram calculations, continuation/warm-starting should be the default.

The deterministic global phase pricing step should eventually make coarse grids optional rather than foundational.

---

# 16. Certification levels

Do not report simply:

```text
success = True
```

Define levels of equilibrium confidence.

For example:

```python
LOCAL_EQUILIBRIUM
STABILITY_SEARCHED
GLOBALLY_CERTIFIED
CERTIFIED_TO_TOLERANCE
GLOBAL_SEARCH_INCOMPLETE
```

A result might report:

```text
Equilibrium Gibbs energy:   -8.492837e4 J/mol
POUNCE KKT residual:         2.1e-10
Global stability tolerance:  1.0e-7 J/mol
Worst phase lower bound:     3.7e-8 J/mol
Certificate:                 GLOBAL
```

If the global solver terminates early, report the remaining uncertainty honestly.

---

# 17. Phase-rule checks

Use thermodynamics to validate solver results.

Implement checks based on the Gibbs phase rule.

At fixed \(T\) and \(P\),

\[
F=C-P
\]

for ordinary systems, accounting carefully for constraints and special cases.

Use such checks as diagnostics, not as a substitute for optimization.

Detect suspicious solutions such as:

- too many independent stable phases,
- duplicate composition sets,
- singular phase fractions,
- inconsistent chemical potentials.

---

# 18. Initial MVP

Do not begin with industrial multicomponent databases.

Build a minimal vertical slice.

## MVP 1: Binary regular solution

Implement one substitutional phase:

\[
G(x)=
xG_A^0+
(1-x)G_B^0+
RT[x\log x+(1-x)\log(1-x)]
+
\Omega x(1-x).
\]

Tests should cover:

1. convex single-phase region,
2. nonconvex miscibility gap,
3. exact common-tangent coexistence compositions,
4. recovery of two composition sets,
5. deterministic global stability certificate.

Compare against a dense brute-force reference.

This should validate the entire architecture before TDB complexity is introduced.

---

# 19. MVP 2: Multiple simple phases

Implement:

```text
LIQUID
FCC_A1
BCC_A2
```

for a simple binary system.

Verify that the algorithm can:

- select the stable phase,
- find two-phase equilibria,
- discover absent stable phases,
- remove unstable phases,
- handle phase disappearance continuously,
- find miscibility gaps.

---

# 20. MVP 3: TDB import

Choose a small open binary `.tdb` database.

Parse it.

Compile its phases to discopt.

Compare:

```text
discopt-calphad
vs
pycalphad
vs
OpenCalphad
```

at a grid of:

- temperatures,
- compositions.

Compare:

- stable phases,
- phase fractions,
- phase compositions,
- Gibbs energies,
- chemical potentials.

All discrepancies should generate reproducible regression cases.

---

# 21. Reference implementation comparisons

Use pycalphad and/or OpenCalphad as reference implementations.

Create automated cross-validation where feasible.

For each equilibrium case record:

```python
{
    "T": ...,
    "P": ...,
    "overall_composition": ...,
    "stable_phases": ...,
    "phase_amounts": ...,
    "phase_compositions": ...,
    "chemical_potentials": ...,
    "G": ...
}
```

Use tolerances appropriate to thermodynamic calculations rather than exact floating-point equality.

---

# 22. Adversarial tests

Create cases specifically designed to expose weaknesses of sampling-based equilibrium algorithms.

Examples:

### Narrow Gibbs minimum

Construct a phase having a narrow stable composition range that a coarse composition grid misses.

### Deep but narrow metastability violation

Construct

\[
\Delta G(x)<0
\]

only on a small interval.

### Miscibility gap

Construct two minima separated by a broad unstable region.

### Nearly tangent phases

Two Gibbs curves should differ by very small energies.

### Critical point

Approach disappearance of a miscibility gap.

### Trace phase

Create equilibrium with one phase amount near zero.

### Multicomponent narrow basin

Construct a ternary phase having a small stable region inside the composition simplex.

The global solver should locate and certify these cases independently of arbitrary sampling density.

---

# 23. Performance experiments

Measure separately:

### POUNCE restricted solve

- iterations,
- factorization time,
- derivative time,
- warm-start performance.

### discopt phase pricing

- branch-and-bound nodes,
- initial lower bound,
- final gap,
- effect of specialized envelopes.

Compare relaxation strategies:

```text
generic intervals
generic McCormick
xlogx-aware
simplex-aware
Redlich-Kister-aware
CALPHAD-specific combined
```

Measure how each changes the number of global optimization nodes.

---

# 24. Parallelism

Pricing problems are naturally parallel by phase.

Design the API so that

```python
stability = price_phases(
    phases,
    mu,
    workers=...
)
```

can execute concurrently.

Do not introduce distributed infrastructure in the MVP.

A simple local parallel implementation is sufficient initially.

---

# 25. Derivatives

Use automatic differentiation from the discopt expression graph wherever possible.

Support:

- objective value,
- gradient,
- sparse Jacobian,
- Hessian or Hessian-vector products.

Do not rely on finite differences for ordinary thermodynamic models.

The same expression tree should generate information needed by both:

- POUNCE,
- discopt global relaxations.

---

# 26. Numerical treatment of boundaries

Thermodynamic models commonly contain

\[
x\log x
\]

with \(x=0\).

Implement mathematically correct limiting behavior:

\[
\lim_{x\to0^+} x\log x = 0.
\]

Do not solve this merely by replacing zero with an arbitrary `1e-12` everywhere.

Derivatives near boundaries require deliberate treatment.

Document the mathematical and numerical strategy.

---

# 27. Architecture

Start with a modular layout approximately like:

```text
discopt-calphad/
    database/
        tdb_parser.py
        objects.py

    thermo/
        system.py
        phase.py
        species.py
        composition_set.py
        functions.py

    models/
        substitutional.py
        cef.py
        ideal_mixing.py
        redlich_kister.py

    equilibrium/
        problem.py
        restricted.py
        stability.py
        phase_generation.py
        result.py

    relaxations/
        xlogx.py
        simplex.py
        multilinear.py
        redlich_kister.py
        gibbs_hull.py

    solvers/
        pounce.py
        global_discopt.py

    io/
        tdb.py

    tests/
        ...
```

Adapt this to existing discopt conventions rather than enforcing these exact filenames.

---

# 28. API goal

Ultimately I want something approximately this simple:

```python
from discopt_calphad import Database, equilibrium

db = Database("Fe-C.tdb")

result = equilibrium(
    db,
    components=["FE", "C"],
    phases="all",
    T=1000.0,
    P=101325.0,
    X={"C": 0.02},
    local_solver="pounce",
    global_solver="discopt",
    certify=True,
)

print(result.phases)
print(result.phase_fractions)
print(result.compositions)
print(result.chemical_potentials)
print(result.certificate)
```

Programmatic construction should look similar:

```python
from discopt_calphad import ThermoSystem

system = ThermoSystem(["A", "B"])

with system.phase("LIQUID") as p:
    xA = p.X("A")
    xB = p.X("B")

    p.G = (
        xA * GA(T)
        + xB * GB(T)
        + R*T*(xlogx(xA) + xlogx(xB))
        + L(T)*xA*xB
    )

result = system.equilibrium(
    T=1000,
    X={"A": 0.35},
    certify=True,
)
```

---

# 29. Important design principle

Do not design the global solver as an afterthought.

The entire point of using discopt is that the thermodynamic expression graph should preserve enough semantic information to exploit structure.

Avoid lowering an expression such as

```text
RedlichKister(...)
```

immediately into hundreds of primitive arithmetic nodes if doing so destroys information useful for bounding.

Allow expression nodes such as:

```text
IdealMixing
XLogX
RedlichKister
CEFEndmember
CEFInteraction
SimplexProduct
PiecewiseThermoFunction
```

to survive long enough for the global optimizer to select specialized relaxations.

Lower them only when necessary.

---

# 30. Research questions to answer during development

Maintain a research notebook answering these questions experimentally.

1. Can deterministic phase pricing replace composition grids for equilibrium initialization/stability testing?

2. How expensive is globally solving individual phase stability problems?

3. How much do CALPHAD-specific envelopes improve bounds compared with generic McCormick relaxations?

4. Can simplex structure produce dramatically tighter CEF relaxations?

5. Can Gibbs-energy convexification/common-tangent structure be integrated directly into discopt?

6. Is global phase pricing substantially cheaper than globally optimizing the complete multiphase equilibrium?

7. How effectively can POUNCE warm-start successive restricted equilibria?

8. Can we provide rigorous global equilibrium certificates for realistic binary and ternary systems?

9. At what dimensionality does deterministic global pricing become difficult?

10. Can chemical-potential bounds from the restricted problem improve global phase bounds?

11. Can inactive phases be eliminated cheaply before full global pricing?

12. Can convex phases be recognized and solved without branch-and-bound?

13. Can the global solver automatically prove a phase Gibbs model convex over its feasible composition domain?

14. Can branch-and-bound reuse information between nearby temperatures/compositions during phase-diagram tracing?

---

# 31. Do not overbuild the first iteration

The first implementation milestone should NOT be:

> Read every CALPHAD database and reproduce Thermo-Calc.

It should be:

> Demonstrate that POUNCE + globally solved phase pricing correctly and certifiably solves a binary nonconvex Gibbs equilibrium problem, including a miscibility gap.

Only after that works should we add:

1. a clean internal thermodynamic representation,
2. TDB parsing,
3. CEF models,
4. real databases,
5. multicomponent systems.

---

# 32. First concrete implementation task

Start by examining the existing `discopt`, `pounce`, and `discopt-mkm` repositories.

Before writing substantial code:

1. identify how discopt currently represents variables, expressions, bounds, constraints, convex relaxations, and global optimization nodes;
2. identify the cleanest way to expose a new thermodynamic modeling package;
3. identify how POUNCE accepts NLP models and derivatives;
4. identify patterns in `discopt-mkm` worth reusing;
5. propose the smallest architecture that does not duplicate expression systems.

Then implement a binary regular-solution prototype.

The prototype must contain:

```python
class RegularSolutionPhase:
    ...

class RestrictedEquilibrium:
    ...

class PhaseStabilityProblem:
    ...
```

or equivalent abstractions consistent with discopt.

Use the model

\[
G(x,T)
=
xG_A(T)
+
(1-x)G_B(T)
+
RT[x\log x+(1-x)\log(1-x)]
+
\Omega x(1-x).
\]

Choose parameters that produce a miscibility gap.

Demonstrate numerically:

1. the nonconvex Gibbs curve;
2. a local single-composition solution that is not globally stable;
3. global phase pricing detecting the missing composition;
4. generation of a second composition set;
5. re-solution by POUNCE;
6. recovery of the common-tangent equilibrium;
7. global lower bounds proving no remaining destabilizing composition exists.

Write tests for every step.

---

# 33. Deliverables for the first development pass

Produce:

### A. Architecture note

Approximately 2–4 pages explaining:

- mathematical formulation,
- data model,
- solver decomposition,
- relationship between POUNCE and discopt,
- composition-set generation,
- certification logic.

### B. Minimal working prototype

Binary regular solution with a miscibility gap.

### C. Tests

Unit tests plus an end-to-end equilibrium/certification test.

### D. Plot/example

Plot:

- original Gibbs curve,
- common tangent,
- coexistence compositions,
- global lower envelope if available.

### E. Roadmap

Specific next steps to reach:

```text
regular solution
    ->
multiple phases
    ->
TDB import
    ->
CEF
    ->
binary real database
    ->
ternary database
    ->
phase-diagram tracing
```

### F. Open technical issues

List unresolved questions rather than hiding approximations.

---

# 34. Quality requirements

Prefer:

- small composable abstractions,
- mathematical transparency,
- exact derivatives,
- deterministic tests,
- reproducible examples,
- explicit tolerances,
- strong typing where practical,
- minimal hidden state.

Avoid:

- opaque solver heuristics,
- duplicated thermodynamic formulas,
- arbitrary clipping without documentation,
- dependence on composition grids for correctness,
- giant monolithic equilibrium functions,
- premature GUI work.

Correctness and solver architecture are the priorities.

---

# 35. Long-term vision

The package should eventually make the following distinction possible:

```text
pycalphad/OpenCalphad:
    "Here is the equilibrium we found."

discopt-calphad:
    "Here is the equilibrium, and here is a bound proving that no
     admissible composition of any candidate phase can lower the
     Gibbs energy by more than ε."
```

The scientific opportunity is not merely to implement another CALPHAD equilibrium solver.

The objective is to determine whether **equation-aware deterministic global optimization plus a high-performance local NLP solver can provide more reliable and certifiable CALPHAD equilibrium calculations than conventional grid-assisted phase-stability algorithms**, while remaining compatible with existing CALPHAD databases and workflows.

Start with the smallest experiment capable of proving or disproving that hypothesis.
