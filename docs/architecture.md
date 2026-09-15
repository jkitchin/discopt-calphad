# discopt-calphad: architecture note (draft 1)

Status: proposal for review, no code yet. Written 2026-09-15, after surveying
discopt 0.8.0 (PyPI), POUNCE 0.11.0, discopt-mkm and discopt-thermo.

## 1. Goal and hypothesis

discopt-calphad is a CALPHAD equilibrium plugin for the discopt modeling
language. It returns an equilibrium **together with a bound** showing that no
admissible composition of any candidate phase can lower the Gibbs energy by more
than ε.

The hypothesis to test first: a local sparse NLP solve (POUNCE) on a *restricted*
set of composition sets, plus a *globally solved* per-phase pricing problem
(discopt spatial branch-and-bound), certifiably solves a binary nonconvex
equilibrium with a miscibility gap. It must do so without relying on a
composition grid for correctness. Everything else (TDB, CEF, real databases,
tracing) waits until this is shown.

## 2. Mathematical formulation

Fixed T, P and elemental amounts b ∈ ℝ^E. Phase models φ have site fractions
y ∈ Y_φ, a product of simplices with one simplex per sublattice. The phase model
gives a molar Gibbs energy G_φ(T,P,y) and a composition map A_φ(y) ∈ ℝ^E: moles of
each element per formula unit.

**Restricted problem R(𝒮).** 𝒮 is a finite set of composition sets α, each an
instance of some phase φ(α). Solve

  min Σ_α N_α G_φ(α)(y_α)  s.t.  Σ_α N_α A_φ(α)(y_α) = b,  N_α ≥ 0,  y_α ∈ Y_φ(α).

Multipliers μ ∈ ℝ^E on the elemental balances are the chemical potentials.
Stationarity gives, for every α with N_α > 0:

- (tangent plane) G_φ(y_α) = μᵀA_φ(y_α)
- (tangency) ∇_y [G_φ − μᵀA_φ](y_α) ⟂ Y_φ

**Pricing problem P_φ(μ).** For each phase model φ (active or not):

  Ψ_φ* = min_{y∈Y_φ} Ψ_φ(y),  Ψ_φ(y) = G_φ(y) − μᵀA_φ(y).

Ψ_φ is the tangent-plane distance.

**Certificate theorem.** Suppose every phase has a valid global lower bound
LB_φ ≤ Ψ_φ*, and let LB = min_φ LB_φ. Then any feasible multiphase configuration
with total formula units N_tot = Σ N_α satisfies

  G = Σ_α N_α G_α ≥ Σ_α N_α (μᵀA_α + LB) = μᵀb + N_tot·LB.

At an exact KKT point of R(𝒮), G_R = μᵀb, which gives

  G_R − G_global ≤ −N_tot,max · LB,

where N_tot,max bounds the total formula units of any feasible configuration.
This is finite for condensed phases because each formula unit contains at least
one atom. So LB ≥ −ε certifies global optimality to within ε·N_tot,max.

When R(𝒮) is solved only approximately:

- Replace G_R = μᵀb by G_R = μᵀb + r, where r is computed from the returned
  primal–dual pair, not assumed zero.
- Pricing against an inexact μ is still valid. The theorem holds for *any* μ. An
  inaccurate μ only weakens LB; it never makes the certificate unsound.

This is the key property. The certificate needs no global solve of the coupled
multiphase problem: only a valid lower bound for each phase, independently.

Formulation choice (to confirm by experiment):

- **Intensive form (proposed default).** Variables are (N_α, y_α) as above. The
  balances are bilinear, but the *same* lowered expression G_φ(y) is used in both
  R and P_φ. This is the "one formula" rule.
- **Homogeneous form (alternative).** Variables are species amounts
  n_{α,i} = N_α y_{α,i}. The balances are linear and the Gibbs energy is
  homogeneous of degree 1. For the binary regular solution:
  G = Σ n_i G_i° + RT[Σ entropy(n_i) − entropy(N)] + Ω n_A n_B / N.
  It is better conditioned for the balances but singular as N_α → 0. It also
  needs a second (homogenized) lowering of each phase.

## 3. Data model

A thermodynamic representation with no TDB syntax. TDB import and hand-written
models both produce these objects:

| Object | Holds |
|---|---|
| `Element`, `Species` | name, mass, stoichiometry (element → coefficient), charge |
| `Sublattice` | allowed constituents, site ratio |
| `PhaseModel` | name, sublattices, model type, list of `GibbsTerm`s, A_φ(y) |
| `ThermoFunction` | `PiecewiseT` expression in T (and P later) |
| `ThermoParameter` | kind (G, L_k, TC, BMAGN, …), constituent array, order k, `ThermoFunction` |
| `ThermoDatabase` | elements, species, functions, phase models, reference states |
| `CompositionSet` | phase model reference, label (`FCC_A1#2`), N, y, warm-start data |
| `EquilibriumProblem` | database/system, components, T, P, b, candidate phases, tolerances |
| `EquilibriumResult` | composition sets, μ, G, KKT diagnostics, `StabilityResult`s, certificate |

`PhaseModel` and `CompositionSet` stay separate (prompt §5). Nothing assumes one
set per phase, and composition-set labels are assigned by the outer loop.

## 4. One expression system

The Gibbs model is a list of **semantic term nodes** that live in the plugin:
`Endmember` (CEF endmember product), `IdealMixing`/`XLogX`, `RedlichKister`,
`CEFInteraction`, `PiecewiseT`, and later `Magnetic`. Each term has exactly one
lowering, `term.lower(vars, T) -> discopt Expression`. After lowering, discopt
does everything: evaluation, JAX derivatives, sparsity, interval bounds,
McCormick relaxations, branch-and-bound. Plots and brute-force references in
tests also evaluate the *lowered discopt expression*, so no Gibbs formula is
written twice.

Facts checked in discopt 0.8.0 that shape this:

- discopt already has an internal node `FunctionCall("entropy", x)` = x ln x. It
  has a JAX evaluator (`_relax/dag_compiler.py:239`), a McCormick relaxation
  (`_relax/mccormick.py:748`, convex part exact, secant on the concave side), an
  interval rule (`_relax/convexity/interval.py:498`) and NL lowering for POUNCE
  (`_nl_expr_compiler.py:869`). `XLogX` lowers to this node. That covers §9.1 and
  §26 for the MVP.
- The node is not public API, so we build it with `dm.FunctionCall("entropy", …)`.
  The evaluator clamps the log argument at 1e-300 (x·log(max(x,1e-300))); that is
  a documented value exactly at 0, not a global 1e-12 floor. Its derivative
  (ln x + 1) is unbounded at 0. The strategy is to keep y strictly in the interior
  for POUNCE (barrier) and use the exact convex envelope in branch-and-bound. This
  needs the upstream issue under §8.
- discopt has **no plugin hook for relaxations**. Dispatch is hard-coded per
  function name. Specialized CALPHAD envelopes (simplex-aware multilinear,
  Redlich–Kister polynomial envelopes, Gibbs-hull cuts) therefore land in
  discopt itself as PRs, each benchmarked against generic McCormick. The term
  nodes survive in the plugin so a future dispatch can pick a relaxation per term
  (§29). Until then they lower to primitives discopt already bounds.

Modeling API, following discopt-mkm/thermo style:

```python
from discopt.calphad import ThermoSystem, xlogx, R
sys = ThermoSystem(["A", "B"])
with sys.phase("LIQUID") as p:
    xA, xB = p.X("A"), p.X("B")
    p.G = xA*GA(p.T) + xB*GB(p.T) + R*p.T*(xlogx(xA) + xlogx(xB)) + L0*xA*xB
res = sys.equilibrium(T=1000.0, X={"A": 0.35}, certify=True)
```

`p.X(...)` returns a plugin variable handle, not a discopt `Variable`, because a
phase is lowered into several models: each composition set in R and one pricing
model per φ. Arithmetic on handles builds term nodes, which are lowered per model.

## 5. Solver decomposition

```
ThermoDatabase / ThermoSystem
        │ lower terms → discopt expressions
        ▼
 initial 𝒮 (endmembers / neighbour warm start)
        ▼
 ┌─ R(𝒮): dm.Model, nlp_solver="pounce" ──► N, y, μ = constraint_duals["balance_<E>"]
 │      ▼
 │  P_φ(μ) for every φ: dm.Model, global m.solve() ──► UB_φ, LB_φ = result.bound, y*_φ
 │      ▼
 │  any UB_φ < −ε ?  yes → add CompositionSet(φ, y*_φ) (dedup) ─┘
 │                   no  → all LB_φ ≥ −ε ?  yes → certified
 │                                         no  → GLOBAL_SEARCH_INCOMPLETE
```

- **POUNCE (local).** *MVP 1 finding:* `m.solve()` on R does not give a local solve.
  R is bilinear, so discopt 0.8.0 runs global branch-and-bound on it: the solve
  hit a 120 s time limit and returned no duals. R therefore goes straight to
  discopt's evaluator and local NLP backend (`make_evaluator` +
  `get_nlp_solver("auto")`, as discopt-speciation does). This path takes 7–11
  POUNCE iterations and about a millisecond, and returns multipliers in
  constraint declaration order. μ = −λ, checked against the analytic common
  tangent. Both imports are private discopt API and sit behind one adapter,
  `solvers/local.py`. A direct `pounce.Problem` path with dual warm starts can
  replace it later (discopt#1247).
- **discopt (global).** P_φ is small: dimension = number of independent site
  fractions. It is solved per phase and independently, so `price_phases(phases,
  mu, workers=)` can use a process pool. It reads `objective`, `bound`, `gap`,
  `gap_certified`, `node_count`, `status`.
- **Composition-set generation.** A composition set is added for y*_φ when
  UB_φ < −ε, even if φ is already active. Before adding, it is compared with the
  existing sets of the same phase in the site-fraction metric
  ‖y − y_α‖_∞ < δ_dup. A match means R did not converge to that set's tangency,
  so the loop re-solves R from y* instead of duplicating the set. Optionally all
  destabilizing phases are added per iteration. Sets with N_α below a documented
  threshold after R are removed. The removal is logged and re-checked by the next
  pricing round, so dropping a set can never hide instability.

## 6. Certification logic

Ψ is scaled to **G/RT (dimensionless)**, so tolerances mean the same thing at any
T. ε_J/mol = ε·RT is reported alongside. Levels, in order of increasing strength:

| Level | Condition |
|---|---|
| `LOCAL_EQUILIBRIUM` | R solved to KKT tolerance; no pricing done |
| `STABILITY_SEARCHED` | pricing ran with local/heuristic solves only (no valid bound) |
| `GLOBAL_SEARCH_INCOMPLETE` | some φ has no bound, or LB_φ < −ε, UB_φ ≥ −ε at a limit; the worst LB is reported |
| `CERTIFIED_TO_TOLERANCE` | every φ has LB_φ ≥ −ε, with the certificate-theorem gap bound reported |
| `GLOBALLY_CERTIFIED` | as above, and ε is at or below the requested certification tolerance and r is below the KKT tolerance |

Discrepancies with discopt that the certificate must handle:

- `gap_tolerance` is relative, but Ψ* → 0 at equilibrium. discopt 0.8.0 also has
  an absolute gap test (`_DEFAULT_ABS_GAP_TOL = 1e-6`, `solver.py:4449`), which
  appears to be a module constant rather than a solve option (to verify). The
  certificate therefore uses LB_φ ≥ −ε directly, never `gap_certified` alone.
- discopt's bound comes from floating-point relaxations, not outward-rounded
  interval arithmetic. The reported certificate says so, and we do not claim
  rounding-safe rigor.
- Phase-rule diagnostics (§17) are reported next to the certificate but never
  used to decide it: number of sets with N_α > 0 versus E, duplicate sets,
  near-zero N_α, disagreement between μ from duals and ∂G/∂n evaluated
  numerically.

## 7. Package layout (adapted to discopt conventions)

```
discopt-calphad/                       # setuptools, src layout, uv, py3.12, MIT
  pyproject.toml                       # name discopt-calphad, include discopt.calphad*, discopt>=0.8
  src/discopt/calphad/                 # PEP 420 namespace subpackage (no src/discopt/__init__.py)
    __init__.py                        # public API + __all__
    objects.py                         # Element, Species, Sublattice, PhaseModel, CompositionSet, ...
    terms.py                           # semantic Gibbs term nodes + lower()
    system.py                          # ThermoSystem programmatic API
    models/regular.py, redlich_kister.py, cef.py
    equilibrium/restricted.py, pricing.py, generation.py, certificate.py, result.py, loop.py
    solvers/pounce.py, global_discopt.py
    io/tdb.py                          # MVP 3: TDB reader/writer
    symbolic.py, database.py           # MVP 3: SymPy T-expressions, PiecewiseT, ThermoDatabase
  tests/                               # pytest, `slow` marker, module-scoped fixtures
  examples/                            # notebooks built by build_notebooks.py (thermo style)
  docs/architecture.md, roadmap.md, research-notebook.md
```

pycalphad (0.11.1 is in the global env) is a **test-only** cross-validation
oracle: an optional dev dependency, never imported by the package. Check the
license before reusing any part of its TDB parser.

## 8. Prototype plan (MVP 1, binary regular solution)

G(x)/RT = x g_A + (1−x) g_B + entropy(x) + entropy(1−x) + ω x(1−x), with ω = Ω/RT.
Take g_A = g_B = 0 and ω = 3, which is above the critical value ω_c = 2. The
binodal then solves ln(x/(1−x)) = ω(2x−1) analytically: x ≈ 0.0707 and 0.9293.
Case ω = 1.5 is the convex control. Tests, one per step of §32:

1. G is nonconvex: G'' < 0 on the spinodal interval, computed through the lowered expression.
2. A single composition set at x = 0.5 is a KKT point of R, but the pricing
   UB < −ε.
3. Global pricing finds y* in the destabilizing basin, and its LB/UB bracket
   agrees with a dense-grid minimum of Ψ.
4. A second set is added; dedup rejects a y* near the existing one.
5. POUNCE re-solve converges, with N_α > 0 for both sets.
6. The compositions match the analytic binodal to within 1e-6, μ matches the
   common tangent, and phase amounts follow the lever rule.
7. Every LB_φ ≥ −ε, the level is `CERTIFIED_TO_TOLERANCE`, and the ω = 1.5 case
   certifies as single-phase.

An example plot shows: G(x), the common tangent, the binodal points, and the
lower convex envelope G** computed by a dense-grid hull. The hull is labelled as
the reference, not as solver output.

## 9. Open technical issues (not hidden)

Items marked *MVP 1* were measured with discopt 0.8.0 / POUNCE 0.11.0 while
building the prototype. The upstream requests are tracked in discopt#1249.

1. **Sound bound near the boundary.** *MVP 1, confirmed problem.* With
   `entropy` arguments whose lower bound is exactly 0, the 1-D pricing solve
   returns `status="feasible"`, `bound=None` and `gap_certified=False`. With
   bounds [1e-9, 1−1e-9] it certifies. Workaround: every discopt model keeps
   site fractions `DOMAIN_FLOOR = 1e-9` from their bounds. Pricing bounds the
   excluded slabs with outward-rounded interval arithmetic
   (`pricing.boundary_lower_bound`), so LB still covers the full domain. Remove
   the workaround once discopt#1242 lands.
2. **Absolute gap control.** *MVP 1, confirmed problem.* `_DEFAULT_ABS_GAP_TOL
   = 1e-6` cannot be changed per solve (`abs_tol` is accepted and ignored on
   this route). At a converged equilibrium LB ≈ −9.9e-7 RT, so ε must exceed
   about 1e-6. `GLOBALLY_CERTIFIED` (LB ≥ −1e-9, r ≤ 1e-8) is unreachable, and
   MVP 1 stops at `CERTIFIED_TO_TOLERANCE`. Tracked in discopt#1243.
3. **Validity of `result.bound` at limits.** *MVP 1:* `gap_certified and bound
   is not None` stands in for bound validity; no LB is used otherwise. For the
   convex control (ω = 1.5) pricing still returned a certified B&B bound.
   Limit-termination semantics are unverified; tracked in discopt#1244.
4. **Intensive or homogeneous R.** *MVP 1:* intensive R converges in 7–11
   POUNCE iterations for every case tested (ω = 3 at x = 0.5, asymmetric g at
   x = 0.3). When a set is added, its first amount comes from NNLS at the
   starting compositions, floored away from 0. Vanishing sets (N_α → 0) are
   filtered by `amount_tol` but have not been stress-tested. The homogeneous
   form is deferred.
   *MVP 2, confirmed problem:* the interior-point solve does not drive a
   vanishing set's amount to zero. The amount stalls at 1e-10 to 1e-7, set by
   the barrier, while the set sits clearly above the tangent plane (Ψ = 0.03 to
   0.25 RT). The loop now drops a set if N_α ≤ `amount_tol`, or if
   N_α < `trace_amount`·Σb and Ψ_α > ε, and then re-solves R before pricing.
   This keeps ghost phases out of the reported μ and G. It is only a
   bookkeeping rule: the certificate always comes from the final R plus full
   pricing.
5. **Formula units with vacancies/CEF.** A_φ(y) and "atoms per formula unit"
   vary with vacancy fraction. This changes the N_tot,max term in the
   certificate and normalization of pricing.
6. **Custom relaxations upstream.** discopt has no registration hook. We need a
   design agreement with discopt for per-node relaxation dispatch before §9–12
   research code can influence branch-and-bound. Otherwise it can only be
   benchmarked offline.
7. **Private discopt API.** The direct POUNCE warm start needs
   `discopt._tape_nlp_evaluator`. Pin, isolate in one adapter, and request a
   public hook.
8. **Duplicate and trace sets.** δ_dup and the removal threshold for N_α are
   heuristics that affect efficiency. The design keeps them out of correctness,
   because pricing re-checks everything, and that claim needs a test.
9. **Parallel pricing and JAX.** A process pool with JAX-compiled models needs
   spawn-safe workers and compile caching. Measure before committing to it.
   *MVP 1:* pricing is the dominant cost. The 1-D regular-solution pricing
   problem takes about 14.7k B&B nodes and 4–7 s. The whole ω = 3 loop takes
   about 16 s, of which the restricted solves are milliseconds. Pricing runs
   serially for now (discopt#1245, #1246). The node count is also the baseline
   for the research question on xlogx-specific relaxations (discopt#1248).
10. **Floating-point rigor.** The boundary-slab interval bound pads by one ulp
    per operation (4e-16 relative on `xlogx`). discopt's B&B bound is not
    rounding-safe. The certificate is "to tolerance in floating point", not
    verified.
11. **Dual ordering.** μ is read by position from `NLPResult.multipliers`, which
    relies on the constraint declaration order that discopt 0.8.0 keeps. The
    tests check μ against the analytic tangent for asymmetric g, so a change
    would be caught.
12. **Degenerate μ with compounds.** *MVP 2.* If the only active set is a line
    compound and b matches its stoichiometry exactly, μ is not unique: any μ
    with μ·A = G works, and R returns whichever multiplier POUNCE lands on.
    Pricing is still valid for that μ, but a poor choice can certify a looser
    bound than needed, or add sets that later vanish. No test exercises this
    case yet. The fix is to choose μ (for example, the μ that maximizes
    min pricing LB) and belongs with ternaries.
13. **Units.** `RegularSolution` is in RT units and ignores T. `BinarySolution`,
    `StoichiometricPhase` and `TPoly` take J/mol and divide by RT inside
    `gibbs`. Every solver quantity (G, μ, Ψ, ε, LB) stays in RT units.
    `Certificate.eps_J_per_mol` converts when T is known.
14. **TDB conventions.** *MVP 3, measured against pycalphad 0.11.2.* Three
    conventions change numbers:
    - **Range boundaries.** Temperature ranges are `[lo, hi)`, with the last
      range closed. alfe.tdb's GHSERAL jumps by 2.3e-4 J at 700 K, so the
      other convention disagrees at the breakpoint. We follow pycalphad.
    - **Gas constant.** pycalphad uses R = 8.3145, and we use CODATA
      8.314462618. Ideal mixing terms therefore differ by about
      4e-5·T·|Σ y ln y| J/mol. The oracle tests correct for this and then agree
      to 1e-10 relative.
    - **Default limits and pressure.** A `,` temperature limit is taken as
      unbounded. P-dependent expressions (for example GAS phases) are
      evaluated at `ThermoDatabase.pressure` (101325 Pa).

    Tolerated on the last range only, because published files do it: a
    missing `N` (`; 6000 01DUP`), a missing lower or upper T
    (`G(LIQUID,AL) +GALLIQ; N`), and stray `"` after `!`. Unknown commands and
    other malformed ranges raise `TDBSyntaxError`.
15. **Binary database models only.** `ThermoDatabase.phase_model` builds
    `BinarySolution` (one mixing sublattice, with site ratio and fixed
    singleton sublattices) or `StoichiometricPhase`. Several features raise
    `UnsupportedModelError` with the reason:
    - mixing on more than one sublattice (CEF, MVP 4);
    - VA in a mixing sublattice;
    - TC/BMAGN (magnetic);
    - DISORDERED_PART;
    - ionic liquids;
    - parameter kinds or orders the binary model cannot represent.

    `phase_models(elements)` skips such phases silently.
    `applicability(elements)` lists the reason for each one, and should be
    checked before trusting a diagram. In Al–Fe, for example, BCC_A2 and
    FCC_A1 are skipped because they are magnetic.

## 10. Roadmap after MVP 1

**MVP 1 status: done.**

- **Package:** `src/discopt/calphad` with ops backends (NUM/EXPR/INTERVAL),
  `RegularSolution`, the restricted solve, pricing, generation, certificate and
  plot.
- **Tests:** `tests/test_regular_solution.py` has one test per §8 step plus a
  slow asymmetric case. `tests/test_units.py` has interval soundness, the
  boundary bound, and every certificate level. 27 tests pass in about 30 s.
- **Example:** `examples/regular_solution_gap.py` produces the G(x) /
  common-tangent / hull plot.

**MVP 2 status: done** (several simple phases, T dependence, phase appearance
and disappearance).

- **Phase protocol:** `gibbs(y, ops, T)`, with T passed through `equilibrium`,
  R, and pricing. `initial_y(b)` may return None.
- **Models:**
  - `BinarySolution`: Redlich–Kister with any number of L_k. Every coefficient
    is a number or a callable of T.
  - `StoichiometricPhase` (`n_y = 0`): priced by exact evaluation, with no
    global solve.
  - `TPoly`: a + bT + cT ln T + dT² + e/T.
- **Loop:**
  - Several phases.
  - The start is the lowest-G single-set phase, or `initial_sets=`.
  - Trace sets are removed with a re-solve (§9.4).
  - Deduplication handles compounds.
  - `EquilibriumResult.T` and `.phase_names`.
- **Plot:** `plot_gx` takes a list of phases, including compounds. `gx_curves`
  gives the sampled curves.
- **Tests:** `tests/test_multiphase.py` covers:
  - the analytic ideal-lens tie-line, lever rule and μ at 1250 K;
  - appearance and disappearance at 900 K and 1600 K from the wrong start
    phase;
  - a three-phase LIQUID + FCC (gap) + AB system against a 200k-point convex
    hull, with G within 1e-7 RT and the same phases.

  `tests/test_models.py` covers the RK value, the evaluator Hessian, interval
  enclosure, compound pricing, dedup and initialization. There are 46 tests in
  total.
- **Example:** `examples/multiphase_lens.py` gives the lens with certified
  tie-lines and a three-phase G–x plot. The three-phase case needs 4
  iterations: it adds FCC, adds AB, and then removes a trace LIQUID.

**MVP 3 status: done** (TDB import and export, symbolic database, binary
database phase models).

- **`symbolic.py`:**
  - A TDB expression parser to SymPy in `T` and `P`. A named function
    reference `GHSERAL#` stays unevaluated as `GHSERAL(T)`.
  - `PiecewiseT`: ranges, `select`, `as_sympy`, `to_tdb`.
  - `tdb_str`, which prints SymPy back to TDB syntax.
- **`io/tdb.py`:**
  - Reads ELEMENT, SPECIES, FUNCTION, PHASE (with `:L` style hints),
    CONSTITUENT, PARAMETER (G/L, any order, `*` wildcards), and
    TYPE_DEFINITION, kept raw for amendment checks.
  - Ignores bookkeeping commands and raises on anything else.
  - Keywords may be abbreviated per `_`/`-` part.
  - The writer's output re-reads to identical functions and parameters.
- **`database.py`:** `ThermoDatabase`.
  - Evaluates functions numerically with caching and recursion or undefined
    checks.
  - `expand()` inlines named functions.
  - `phase_model`/`phase_models`/`applicability` (§9.15).
  - Model coefficients are `DatabaseValue`s: callable at T, and symbolic via
    `.expr()` with the parameter labels kept.
  - `BinarySolution` gained `sites` and `fixed` for singleton sublattices such
    as `(AL,ZN)1(VA)0.5`, and odd RK orders flip sign when the requested
    element order is reversed.
- **Tests:** `tests/test_tdb.py` has 46 tests:
  - Parser and printer round-trip, and syntax errors.
  - Range and extrapolation behaviour, P-dependence, and recursive or
    undefined functions.
  - Model construction and every unsupported-model error.
  - G versus pycalphad (test-only oracle) for Al–Zn, Cu–Mg and Al–Fe phases at
    5 temperatures (including breakpoints) to 1e-10 relative.
  - Round-trip of 10 published databases.
  - Slow: a certified Al–Zn equilibrium versus `pycalphad.equilibrium` (GM,
    μ, phases).

  A sweep (not in CI) reads and round-trips 36 of pycalphad's 38 test TDB files.
- **Example:** `examples/alzn_tdb.py` reads the an Mey Al–Zn database, copied
  with attribution to `examples/data/`. It prints the symbolic view and
  certifies three two-phase states (FCC gap, FCC + HCP, LIQUID + HCP) with
  G–x panels.

Next:

CEF (multi-sublattice endmembers and interactions, simplex-aware bound
study) → small open binary database vs pycalphad on a T–x grid → ternary →
phase-diagram tracing with warm starts in both POUNCE and pricing.
