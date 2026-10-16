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
- **Local and global pricing (`equilibrium(certify=...)`).** Generating sets only
  needs a point with UB_φ < −ε, and μ changes after every R, so a lower bound
  at an intermediate μ is thrown away. `price_phase_local` samples the box and
  refines the best samples with L-BFGS-B (an upper bound only,
  `bound_source="local"`).
  - `"final"` (the default) prices locally until no local point destabilizes,
    then prices globally once at that μ. A destabilizing global point is added
    and the loop continues, so the certificate is the same as with `"every"`.
  - `"every"` prices globally in every iteration.
  - `"none"` prices locally only. The level is at best `STABILITY_SEARCHED`, and
    `EquilibriumResult.certify()` runs the global pricing later at the stored μ
    (for example on selected phase-diagram points). If a phase destabilizes
    there, the level drops to `LOCAL_EQUILIBRIUM` and the loop can resume from
    the stored sets.
  The global pricing at the converged μ is the expensive part in every mode that
  certifies: Ψ* ≈ 0 there, so LB ≥ −ε is the hardest decision.

## 6. Certification logic

Ψ is scaled to **G/RT (dimensionless)**, so tolerances mean the same thing at any
T. ε_J/mol = ε·RT is reported alongside. Levels, in order of increasing strength:

| Level | Condition |
|---|---|
| `LOCAL_EQUILIBRIUM` | R solved to KKT tolerance; no pricing done |
| `STABILITY_SEARCHED` | pricing ran with local/heuristic solves only (no valid bound), e.g. `certify="none"` |
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
  pyproject.toml                       # name discopt-calphad, include discopt.calphad*, discopt>=0.9
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
15. **Database models.** `ThermoDatabase.phase_model` builds a
    `BinarySolution` (binary, one mixing sublattice, no VA),
    a `StoichiometricPhase` (no mixing), or a `CEFPhase` (everything else,
    MVP 4; see §9.16). Several features raise `UnsupportedModelError` with the
    reason:
    - a missing endmember G parameter;
    - interaction structures pycalphad also rejects (§9.16);
    - the Xiong magnetic model (AFM factor 0; the IHJ model is supported, §9.18);
    - DISORDERED_PART with vacancies on a substitutional sublattice, with
      magnetic parameters on the ordered part, or with mismatched interstitial
      sublattices (§9.21);
    - ionic liquids;
    - parameter kinds or orders the binary model cannot represent.

    `phase_models(elements)` skips such phases silently.
16. **CEF conventions.** *MVP 4, measured against pycalphad 0.11.2.*
    - **Variables.** Each sublattice with m > 1 constituents gets m − 1
      stick-breaking variables u ∈ [0, 1]: y_j = u_j Π_{i<j}(1 − u_i), and the
      last y is Π(1 − u_i). The domain stays a box, so the phase protocol,
      `DOMAIN_FLOOR` and the boundary-slab bound are unchanged. xlogx of a
      product is expanded as Σ_i xlogx(f_i) Π_{j≠i} f_j, keeping the entropy
      node discopt can bound.
    - **Terms.** One term per parameter: the product of the listed site
      fractions (a `*` wildcard contributes 1), times an interaction factor.
      The factor is (y_first − y_second)^k for a binary RK on one mixing
      sublattice, in sorted constituent order. For a reciprocal parameter
      (two mixing sublattices), order 1 puts the difference on the second
      mixing sublattice and order 2 on the first. For a ternary, order k
      multiplies by v_k = y_k + (1 − Σy)/3. A lone ternary L0 implies
      L1 = L2 = L0.
    - **Rejected, as in pycalphad.** Reciprocal orders > 2, reciprocal terms
      with 3 constituents on a sublattice, order > 0 with more than two mixing
      sublattices, and more than 3 mixing constituents.
    - **Vacancies.** A_φ(y) counts atoms on non-VA constituents.
      `atoms_min` is the sum of site ratios over sublattices without VA, and
      N_tot,max = Σb/atoms_min. A phase with a vacancy on every sublattice
      would have `atoms_min = 0`. It is restricted to a minimum occupancy
      instead (§9.22).
    - **Oracle.** pycalphad's GM is per mole of atoms, with R = 8.3145.
      Eleven phases (Laves C14/C15, CU2MG, AL13FE4, ALMG_GAMMA, AL3NI2, CHI_RENB,
      AL8FE5_D82, ternary liquids) agree to 1e-9 relative at 3 temperatures.
17. **Pricing with more than one site-fraction variable.** *MVP 4, measured.*
    With `gap_tolerance` 1e-6, which is *relative* in `Model.solve`, a
    2-variable CEF pricing problem does not close in 30–60 s. Pricing does not
    need a closed gap, though: it only needs to decide LB ≥ −ε or UB < −ε.
    discopt 0.8 cannot stop from a callback, because exceptions raised in
    `node_callback` are logged and swallowed. `price_phase(eps=...)` therefore
    runs in stages (2 s, then ×4, up to `time_limit`) and stops once the
    answer is decided. A `node_callback` records the largest `ctx.best_bound`,
    which discopt documents as a certified global dual bound that never
    over-reports, so a stage that hits its time limit still gives a valid LB
    (`PricingResult.bound_source = "callback"`). The boundary slabs of a
    multi-variable phase are thin in one direction only, so a single interval
    evaluation is loose (−0.90 against a true minimum of 2.74). They are
    refined best-first by bisection until the bound reaches −ε. With both
    changes, the reciprocal test phase certifies and CU2MG destabilizes in one
    2 s stage, where the fixed 1e-6 gap previously took 124 s.
    A bound from the loop's pricing is loose by design. The certificate's gap
    bound stays valid but is N_tot,max·ε rather than a tight number, and
    `price_phase(eps=None)` still closes the gap.
    Formulation results are in `benchmarks/simplex_bound_study.py`:

    | 30 s limit | reciprocal (Ψ* ≈ 2.743) | CU2MG (Ψ* ≈ −5.965) |
    |---|---|---|
    | stick-breaking, gap 1e-6 | not closed; 12k nodes; bound 2.7397 | not closed; 28k nodes |
    | explicit y with Σy = 1, gap 1e-6 | not closed; 4.7k nodes; bound 2.710 | not closed |
    | stick-breaking, gap 1e-4 | not closed | closed in 5.2 s (4.6k nodes) |
    | explicit y with Σy = 1, gap 1e-4 | not closed | closed in 20.8 s |
    | staged, ε = 1e-5 | certified in 2 s | destabilizing in 2 s |

    At equal time, the explicit simplex gives weaker bounds than
    stick-breaking, because discopt does not exploit the equality. Raising
    `DOMAIN_FLOOR` to 1e-6 did not help. This is the measured case for the
    simplex structure request (discopt-feature-requests P2-2).
    `applicability(elements)` lists the reason for each one, and should be
    checked before trusting a diagram. In Dupin's Al–Ni, for example, BCC_B2 is
    skipped because it has vacancies on its substitutional sublattices
    (§9.21).
18. **Magnetic ordering.** *Issue #6; measured against pycalphad 0.11.2.* `models/magnetic.py` implements the Inden–Hillert–Jarl term
    RT ln(β' + 1) f(T/T_c') per formula unit, and a magnetic phase is always a
    `CEFPhase`.
    - **Conventions, as pycalphad applies them.**
      - A MAGNETIC TYPE_DEFINITION applies through the phase's type-code
        character, not through the phase name on the line
        (Al-Fe_sundman2009 writes `A_P_D @ MAGNETIC`).
      - TC or BMAGN parameters with no MAGNETIC type are ignored.
      - `BM` abbreviates `BMAGN`.
      - T_c ≤ 0 and β ≤ 0 are divided by the AFM factor.
      - AFM factor 0 selects the Xiong model, which raises for now.
    - **Checked.** G matches `pycalphad.calculate` to 1e-9 relative for 8
      magnetic phases in 7 databases, at 300–1800 K. Four equilibria match
      `pycalphad.equilibrium` in phases and compositions: the Cr–Fe BCC gap,
      the γ loop, Fe–Nb BCC + Laves, and the Cr–V antiferromagnetic gap.
      The one difference is at T = T_c exactly. There pycalphad's Piecewise has
      no branch for τ = 1 and returns 0; the IHJ function is continuous there
      and we return f(1).
    - **Branches.** Written in s = T_c'/T, f is a polynomial on s ≤ 1 and has
      bounded negative powers on s > 1. The two branches agree in value and
      slope at s = 1 (C¹; the heat capacity jumps). The model is also piecewise
      in the signs of T_c and β. A `Branch` names one smooth piece. The numeric
      backend picks it per point. A discopt expression is built on one branch
      (`EXPR.with_branch`).
    - **Restricted solve.** Each free magnetic set is held on the branch it
      starts in, by three polynomial constraints that are ≥ 0 exactly on that
      branch. The constraint rows come after the balance rows, so μ is still
      read from the first multipliers. If a branch constraint is active at the
      solution, that component is flipped and the problem is solved again from
      there. A branch combination already tried ends the loop, which is the
      case where the solution sits on the edge itself. In the Cr–Fe gap at
      700 K, the Cr-rich set starts with T_c > 0, stops on T_c = 0, and
      converges on the antiferromagnetic branch in the second solve.
    - **Interval extension.** f is continuous and strictly decreasing in s on
      all of s ≥ 0. On the paramagnetic side every term falls. The slopes of
      the two branches agree at s = 1. On the ordered side, the slope
      −(79/(140p) − c(s⁻⁴/2 + s⁻¹⁰/15 + s⁻¹⁶/40))/A only grows more negative
      with s, because the subtracted series shrinks. So f over [s_lo, s_hi]
      is [f(s_hi), f(s_lo)], whether or not the interval straddles 1, and
      f ≤ f(0) = 0. T_c' and β' are nonnegative piecewise-linear images of
      T_c and β: x for x > 0 and x/afm for x ≤ 0. Their range is the image of
      the endpoints, with 0 added when the interval contains 0. ln(1 + β') is
      increasing. libm's `pow` and `log1p` are not correctly rounded, so the
      endpoint values are padded by 1e-14 relative. With these rules, the
      boundary-slab bound and every `INTERVAL` evaluation cover magnetic
      phases. Random boxes in five oracle phases at 300–1500 K give no
      enclosure violations.
    - **Global pricing by branch.** An interval evaluation of T_c and β over
      the site-fraction box tells which branches the box can reach
      (`magnetic_pieces`). Usually there are three to six of the eight. Each one
      becomes its own discopt model, with:
      - the sign constraints of its branch;
      - an auxiliary variable s = T_c'/T, tied to y by the equality
        s − T_c'(y)/T = 0, and bounded to [s_lo, 1] or [1, s_hi].

      Bounding s keeps the negative powers on the ordered side away from
      s = 0 in the relaxation. Without it they would have no finite bound.
      The branches are closed sets that cover the box. Ψ is continuous across
      their edges: f is C¹ at s = 1, and T_c' and β' are 0 from both sides
      at a sign change. So the smallest of the branch minima is the minimum
      over the box, and the smallest of their certified bounds bounds it.
      A branch that discopt proves infeasible contributes +∞. The staged ε
      loop runs over all open branches in each stage, stops as soon as one
      gives UB < −ε, and otherwise stops once every branch bound is ≥ −ε.
    - **Measured.**
      - Cr–Fe BCC at 700 K (six branches): every branch closes in 0.5–4 s. The
        bound matches a 20 001-point grid minimum to 3e-7.
      - Ni–Fe FCC (COST 507, 600 K), Al–Fe BCC (800 K) and Cr–V BCC (300 K)
        also close and match their grids.
      - Certified with every phase, σ and Laves included, all at 13–19 s per
        point:
        - the Cr–Fe BCC gap at 700 K (ferromagnetic Fe-rich + antiferromagnetic
          Cr-rich);
        - σ + BCC at 800 and 1000 K;
        - σ at 900 K;
        - FCC inside the γ loop at 1200 K.
      - See `examples/crfe_magnetic.py` and `examples/fe_magnetic.py`.
    - **Still open.** The Xiong model (AFM factor 0), and TC/BMAGN as free
      parameters in `fit`.
    - **Fitting.** G is not linear in TC and BMAGN, so `fit` cannot free
      them. It does fit the G and L parameters of a magnetic phase, and data
      referenced to one (a formation enthalpy against magnetic FCC Ni): the
      probes run with TC and BMAGN at zero, which removes the magnetic term
      exactly, and the fixed magnetic term is added back, with its
      T-derivatives for H, S and Cp taken by finite differences.

19. **Site fractions on the domain floor** (issue #13). A terminal phase can
    have an equilibrium solubility far below `DOMAIN_FLOOR` (about 1e-14 Al in
    FCC Ni next to Al₃Ni, in an EMT-fitted Ni–Al database). The optimum then
    lies on the bound. The interior-point solve reaches it but does not
    declare convergence, and a warm start exactly on a bound is not interior
    either. `solve_restricted` therefore treats a free site fraction within
    10 floors of a bound as an active constraint. It solves with that fraction
    pinned first, then frees everything from there, keeping the free solve
    only if it succeeds and is no worse. After a failed solve, it pins the
    fractions that ended on a bound and tries once more. Separately, when
    several new sets were added at once and R fails, the loop backs off to
    the single most destabilizing one. Pricing still checks every bound
    afterwards, so the certificate is unaffected. When only stoichiometric
    sets are active, R is a linear program, and if μ is not unique the loop
    takes the μ in the middle of the compounds' stability range (§9.12).
20. **Order–disorder stand-ins** (issue #14). `ordering.stand_in` rewrites a
    partitioned phase, G_dis(x) + G_ord(y) − G_ord(x), as a plain CEF phase.
    Every source parameter's factor in y is projected onto endmembers,
    Redlich–Kister terms up to order 3, and reciprocal terms of order 0–2.
    Each stand-in parameter is then a fixed linear combination of the source
    parameters' functions. For Dupin's Ni–Al, L1₂ and B2 are exact, matching
    pycalphad to 1e-10 J/mol, apart from what is dropped: magnetism, and
    vacancies on substitutional sublattices. Since issue #6, partitioned phases
    are modeled natively (§9.21). A stand-in remains useful for refitting, or
    for a phase the native model rejects.
21. **Order–disorder (partitioned) phases.** *Issue #6; measured against
    pycalphad 0.11.2.* `models/partitioned.py`. A DISORDERED_PART amendment
    builds a `PartitionedPhase`:

    G = λ G_dis(x) + G_ord(y) − G_ord(x)

    per ordered formula unit. x is the site average over the substitutional
    sublattices, and λ = Σ_s a_s / a_dis. NEVER_DISORDER gives
    G = λ G_dis(x) + G_ord(y).
    - **Conventions, as pycalphad applies them.**
      - The amendment applies to the phase named on its line, not through the
        type code.
      - Ordered sublattices other than the interstitial one are extended to
        the disordered part's constituents. Those with exactly its
        substitutional constituents take part in the ordering.
      - Parameters missing from the ordered part are 0.
      - Vacancies count as a species in x.
      - The F and B phase options imply every symmetry image of each ordered
        parameter: F permutes all four sublattices, B permutes within {0, 1}
        and {2, 3} and swaps the pairs. The images are added even when the
        file lists them too, as pycalphad does. They are added when the model
        is built, so TDB export writes the file's own parameters.
      - The magnetic term is the disordered part's, with T_c(x) and β(x). A
        phase whose ordered part has its own TC or BMAGN is rejected:
        pycalphad gives those a separate magnetic term, Thermo-Calc
        partitions T_c instead, and the two disagree.
    - **Entropy.** The ideal terms of λG_dis(x) and G_ord(x) cancel exactly.
      Both are (Σ_s a_s) Σ x ln x on the substitutional sites, and the
      interstitial terms agree because the interstitial-to-substitutional site
      ratios must be equal. Only the ordered sublattices' entropy is left, on
      single stick-breaking variables. Every other term is a polynomial in y.
    - **Checked.** G matches `pycalphad.calculate` to 1e-9 relative at
      500–1500 K, at random and disordered points, for:
      - FCC_L12 (Dupin's Al–Ni and alcrni);
      - B2 (alfe.tdb, COST 507 Al–Ni and Fe–Ni);
      - CoV's FCC_4SL, four sublattices with the F option;
      - CoV's σ, NEVER_DISORDER.

      In Al–Ni, γ, γ′ and γ + γ′ match `pycalphad.equilibrium` in phases and
      compositions. Both sides use the same phase list, without BCC_A2 and
      BCC_B2 (below). pycalphad names a disordered L1₂ set FCC_L12, and it
      has exactly FCC_A1's G.
    - **Pricing.** The site averages are auxiliary variables, tied to y by
      linear equalities, so G_dis(x) and G_ord(x) are written in a few
      variables. Without them, a non-magnetic L1₂ pricing problem took 55 s
      to close; with them, 15 s. Magnetic branches are branches of T_c(x(y)).
      Each is solved with its constraints in x, as in §9.18.
    - **Symmetry.** Substitutional sublattices with the same site ratio and
      constituents form a class. The permutations that leave G unchanged are
      found numerically when the model is built, at three temperatures and
      four random points:
      - B2: 2;
      - CoV's FCC_4SL: 24;
      - L1₂: none, since its two sublattices have different ratios.

      When a class is invariant under all its permutations, pricing requires
      the first constituent's fraction to descend along the class. Each point
      has a sorted image with the same G, so this removes duplicate minima and
      never the minimum. `is_duplicate` compares composition sets up to the
      same symmetry.
    - **Restricted solve.** A new ordered set has one sublattice nearly pure
      and a small amount. Freed from its pricing point with the default
      interior-point barrier, it was pushed back from the bound, slid into the
      disordered basin, and the loop cycled. The free solve after the
      held-set solve therefore starts with `mu_init = 1e-6`.
    - **Not supported yet.**
      - **Magnetic parameters on the ordered part** (Sundman's Al–Fe,
        alcfe_b2).
      - **Mismatched interstitial sublattices.** COST 507's BCC_B2 in seven
        binaries, such as Al–N, which pycalphad rejects too.
    - **Registry.** COST 507 goes from 7 to 338 of 345 complete binaries, and
      the COST 507 Al–Fe file is complete.
22. **Vacancies on every sublattice.** *Issue #6.* Dupin's BCC_A2 and BCC_B2,
    and alcrni's B2, can hold a vacancy on every sublattice, so a formula unit
    can be empty.
    - **The model has no global minimum there.** These databases set
      G(VA:…:VA) = 0. Take a fraction δ of atoms in an otherwise empty
      lattice. Its G per atom is about G_atom + L_atom,VA + RT ln δ, which
      goes to −∞ as δ → 0. Read literally, the model lets a huge amount of
      nearly empty lattice dissolve any atoms at unbounded entropy per atom.
      pycalphad never meets this, because its point sampling does not reach
      δ ≈ e⁻²⁰, and a local minimizer never starts there. No certificate can
      cover the full simplex, and no amount bound N_tot,max exists.
    - **The domain.** A phase with a vacancy on every sublattice is
      restricted to formula units holding at least `min_occupancy` (default
      0.5, `ThermoDatabase.min_occupancy`) of `atoms_max` in atoms:
      `CEFPhase.domain_constraints` gives atoms(y) − atoms_min ≥ 0. Then
      atoms_min = min_occupancy · atoms_max > 0, and the certificate theorem
      (§2) holds unchanged on that domain. Constitutional vacancies in real
      phases are a few percent: Al-rich NiAl at 1200 K reaches about 4% of
      sites. The constraint is never active at an equilibrium, and the
      certificate's messages state the assumption.
    - **Where it applies.**
      - The constraint is a row in the restricted solve, after the balance
        rows.
      - It is a constraint in every global pricing model.
      - The boundary-slab bound prunes a box whose interval of the
        constraint is entirely negative.
      - Local pricing gives such points +∞.
    - **Checked.**
      - G matches pycalphad to 1e-9 for Dupin's BCC_B2 and alcrni's B2.
      - Al–Ni equilibria with every phase match `pycalphad.equilibrium` in
        phases and compositions, including Al-rich β + Al₃Ni₂ (0.556 /
        0.594 at 1200 K). There the vacancy-free stand-in was off by up to
        0.04.
      - On a toy phase with G(VA:VA) = 0, global pricing matches a dense
        grid restricted to the domain, while a point with δ = 1e-8 lies more
        than 10 RT per atom below the tangent.
    - **Open: certification with β active.** At 1200 K, β + Al₃Ni₂
      (x = 0.58) and β + γ′ (x = 0.3) end `GLOBAL_SEARCH_INCOMPLETE`. The
      bound of BCC_B2 (four variables, plus the occupancy and symmetry
      constraints) stalls near −3 after 600 s. The bounds of the other
      active phases, Al₃Ni₂ and FCC_L12, stall at −4.8e-3 and −4.4e-5. This
      is the active-phase weakness of §9.17, with a larger problem.

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
- **Example:** `examples/alzn_tdb.py` reads the an Mey Al–Zn database, fetched
  from the registry (`databases.load("alzn_mey")`). It prints the symbolic view and
  certifies three two-phase states (FCC gap, FCC + HCP, LIQUID + HCP) with
  G–x panels.

**MVP 4 status: done** (CEF and the simplex-aware bound study).

- **`models/cef.py`:** `CEFPhase` and `CEFTerm`, with stick-breaking
  variables, endmember, binary RK, reciprocal and ternary Muggianu terms, VA,
  and `gibbs_expr()` (SymPy in `y<s>_<constituent>`, `T`, `R`, with database
  labels kept). It also has `site_fractions`, `variables_from_site_fractions`,
  `endmembers` and a least-squares `initial_y`.
- **Database:** phases in any number of elements are routed to `CEFPhase`
  when the binary model does not fit (§9.15, §9.16).
- **Pricing:** staged, callback-bounded early stopping and a refined
  boundary-slab bound (§9.17).
- **Plot:** `gx_curves` samples multi-variable phases on a grid and keeps the
  lowest G per x bin.
- **Tests:** `tests/test_cef.py` covers:
  - stick-breaking round trip and G checked by hand;
  - one-sublattice CEF equal to `BinarySolution`, and ternary orders;
  - interval enclosure;
  - G against pycalphad for 11 phases;
  - pricing bound validity and staged certification in about 2 s each;
  - slow: Cu–Mg equilibria with CU2MG against `pycalphad.equilibrium`.
- **Example:** `examples/cumg_cef.py` gives the symbolic CU2MG G and three
  Cu–Mg states with G–x panels.
- **Benchmark:** `benchmarks/simplex_bound_study.py`.

**Open (found when verifying MVP 4).** The phase assemblages in Cu–Mg are
right. Certification to ε = 1e-5 still fails whenever the CEF phase CU2MG is
*active*:
- its own pricing minimum is Ψ = 0, at the composition set;
- the tree lower bound stalls at −7e-5 … −4e-4 after the full 600 s;
- the result is `GLOBAL_SEARCH_INCOMPLETE`, with G_R − G_global ≤ 4e-4 RT;
- both slow tests in `tests/test_cef.py` fail on the certificate assertion
  (x = 0.62, T = 900 K: LB −3.7e-4).

Inactive CEF phases, and active one-variable phases, certify quickly. Likely
remedies:
- exclude a small box around each known zero of Ψ, and bound that box
  analytically using the local convexity of the composition set;
- give discopt the known minimizer as an incumbent together with a
  bound-target stop (P0-4).

**Fitting (issue #2): first version.** `src/discopt/calphad/fit.py`. Free
parameters are TDB labels with symbolic T-forms that are linear in their
coefficients.
- **Linear model:** G is linear in the coefficients. The site-fraction
  factors are found by probing the phase models (all of a phase's parameters
  set to 0, then each set to 1 J at T = 1/R).
- **Derived quantities:** H, S and Cp come from symbolic T-derivatives. μ comes
  from the tangent (5-point stencil in y).
- **Data kinds:**
  - GM, HM, SM, CPM and their _MIX forms;
  - activities (ACR);
  - ZPF with measured compositions (μ equality, and compounds on the tangent).
- **Solve:** weighted linear least squares (bounded BVLS when bounds are
  given), absolute covariance (JᵀJ)⁻¹, rank warning.
- **Result:** `FitResult` gives a database with the fitted forms,
  `to_tdb` with a provenance header, `check_equilibria` (certified
  equilibria at the ZPF data), and `plot_residuals`.
- **Tests:** `tests/test_fit.py` (9 tests, about 9 s): exact recovery,
  noisy recovery within 4σ with χ²/dof in range, Gibbs–Helmholtz and tangent
  checks, a tie-line fixing L0, bounds, and JSON/CSV round trips.
- **Example:** `examples/alzn_fit.py` recovers six Al–Zn coefficients from
  33 synthetic residuals (χ² = 19.2, dof 27).
- **Deferred:**
  - ZPF points without a partner composition;
  - errors in variables (`discopt.estimate`);
  - μ data for multi-variable phases;
  - global consistency by cutting planes (docs/symbolic-diagrams-fitting.md §5.3);
  - comparison with a real dataset.

Next:

small open binary database vs pycalphad on a T–x grid → ternary →
phase-diagram tracing with warm starts in both POUNCE and pricing. Open
upstream: P0-4 (stop at a bound target) would replace staged pricing.
