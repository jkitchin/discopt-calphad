# discopt feature requests from discopt-calphad (draft 1, not filed)

Checked against discopt `main` @ `a837ffb8` (2026-09-15). Paths are relative to
`python/discopt/`. Each request gives: why CALPHAD needs it, current state, a
minimal API sketch, and an acceptance test. The API sketches are suggestions;
discopt maintainers own the design.

Priority key:
- **P0** blocks the MVP 1 certificate.
- **P1** needed for multiple phases, tracing and fitting at useful speed.
- **P2** research enablers (CALPHAD-specific relaxations, §9–12 of the project prompt).

---

## P0-1. Make `entropy` (x ln x) a complete, public, boundary-correct operator

**Why.** Every CALPHAD phase has Σ y ln y over site fractions in [0, 1]. The
global pricing bound is only as good as the bound on this term near y = 0.

**Current state.**
- **Not public.** There is no `dm.entropy`/`dm.xlogx`; the node can only be built
  as `FunctionCall("entropy", x)`. It is recovered from `x*log(x)` at
  `solver.py:9142`.
- **Interval rule.** `_relax/convexity/interval.py:498-519` (interval arithmetic
  in the convexity-certificate module) returns `[-inf, inf]` when `lo <= 0` "so
  the certificate soundly abstains". A site fraction's box always starts at
  `lo = 0`, so convexity proofs over the whole simplex abstain on every entropy
  term. That blocks the convex fast path for phases that really are convex
  (prompt research questions 12–13). Whether branch-and-bound node bounding also
  goes through this rule is not verified; `relax_entropy` handles `lb = 0`
  through its clamp. The fix is to treat `lo == 0` as in-domain with f(0) = 0;
  `lo < 0` should still abstain.
- **Rust.** There is no Entropy in `crates/discopt-core/src/expr.rs:50`
  `MathFunc`, so Rust FBBT/presolve cannot handle it.
- **Symbolic diff.** `bilevel/symbolic_diff.py:34` refuses `entropy`.
- **McCormick.** `_relax/mccormick.py:748` works: exact convex side, secant
  concave side, log argument clamped at 1e-300.

**Request.**
- Export `dm.xlogx(x)`, keeping the existing `entropy` node.
- Exact interval rule on [l, u] ⊆ [0, ∞) using f(0) = 0: the minimum is −1/e if
  1/e ∈ [l, u], else min(f(l), f(u)); the maximum is max(f(l), f(u)).
- Rust `MathFunc::Entropy` with FBBT forward and backward rules.
- `diff(entropy(u)) = (log(u) + 1)·u'`.
- Documented derivative behavior at 0.

**Acceptance.**
- `min y ln y + (1−y) ln(1−y) + 3y(1−y)` over y ∈ [0, 1] solves globally to
  optimal with a certified absolute gap ≤ 1e-9. The root interval bound is finite.
- Interval tests at [0, u] for u < 1/e, u = 1/e and u > 1/e.

## P0-2. Absolute gap tolerance as a solve option

**Why.** At a converged equilibrium the pricing optimum Ψ* → 0. A relative gap
is meaningless there, and certification is "LB ≥ −ε" in absolute units (G/RT).

**Current state.** Only the module constant `_DEFAULT_ABS_GAP_TOL = 1e-6`
(`solver.py:4842`); `Model.solve` exposes the relative `gap_tolerance` only
(`core.py:5636`). The AMP path already has `abs_tol`.

**Request.** `Model.solve(..., abs_gap_tolerance=1e-6)`, applied on every
branch-and-bound path. Stop when either the relative or the absolute criterion
holds, and report which one stopped the solve.

**Acceptance.**
- With `abs_gap_tolerance=1e-10`, a problem with optimum ≈ 1e-8 is not declared
  converged at a gap of 1e-7.
- `result.solver_stats` records which criterion stopped the solve.

**Measured (MVP 4).** For a 2-variable CEF pricing problem with optimum −5.96,
a relative gap of 1e-6 does not close in 60 s, while 1e-4 closes in 5 s.
Neither is the right question. Pricing needs only a sign decision against
−ε, which leads to P0-4.

## P0-4. Stop a global solve from a callback, or at a bound target

**Why.** Pricing is decided as soon as the global dual bound reaches −ε
(certified) or an incumbent goes below −ε (destabilizing). Waiting for the gap
to close wastes most of the time. For the reciprocal test phase, the bound
passes −ε within 2 s, but a relative gap of 1e-6 is not closed after 60 s.

**Current state.**
- `node_callback(ctx, model)` returns None. An exception raised inside it is
  logged ("Node callback raised an exception") and swallowed, and the solve
  continues.
- `incumbent_callback` can reject a solution but cannot stop the search.
- The workaround in `pricing.price_phase` is staged solves with growing time
  limits. It records `ctx.best_bound` from a node callback and throws away the
  tree between stages.

**Request.** Either option would do:
- `node_callback` may return `dm.STOP` (or raise `dm.StopSolve`) to terminate
  with the current incumbent and a valid bound; or
- `Model.solve(..., bound_target=L, objective_target=U)`: stop when the
  certified bound is ≥ L or an incumbent is ≤ U, and report which one stopped
  the solve.

**Acceptance.** For `min Ψ` with Ψ* = 2.74, `bound_target=-1e-5` returns within
one node batch after the bound crosses the target. `result.bound` is valid
(P0-3), and the status names the stop reason.

## P0-3. Documented, machine-readable validity of `SolveResult.bound`

**Why.** The certificate uses `bound` as a *valid global lower bound* even when
the solve stops at a time or node limit. It must never read a bound that is not
valid.

**Current state.**
- `gap_certified` exists, and `__post_init__` drops or clears the bound in some
  cases (`core.py:3333-3393`).
- `tree_bound_valid` is internal (`solver.py:3979-4091`).
- The convex fast path sets `bound = obj_val` (`solver.py:19746`), which is only
  valid if the convexity proof is rigorous.
- What `bound` holds at `time_limit`/`node_limit` is undocumented.

**Request.**
- A public `result.bound_valid: bool`, with docs covering every termination
  status.
- `result.bound_source` (`"bnb_tree"`, `"convex_proof"`, `"root_relaxation"`, `None`).

**Acceptance.**
- A table-driven test over status × route asserts that `bound_valid` implies
  `bound ≤` the true optimum, checked on instances with known optima, including
  a stop at `node_limit=3`.

## P1-1. Change Parameter values between global solves without rebuilding relaxations

**Why.** Pricing problems are solved many times: once per phase per outer
iteration, and per (T, x) point in tracing and fitting. Only μ and T change,
and both are natural `Parameter`s.

**Current state.** The evaluator cache ignores `Parameter.value`
(`_evaluator_cache.py:22-40`, good), but the relaxation compiler bakes parameter
values in as constants (`_relax/relaxation_compiler.py:426-431`), so relaxations
are recompiled for every solve.

**Request.** Relaxation closures take parameter values as runtime inputs, or a
parameter-value fingerprint lets compiled relaxations be reused.

**Acceptance.** 100 re-solves of one pricing model with different μ show
relaxation compile time only on the first solve. The bounds match fresh models
to 1e-12.

## P1-2. Batch or parallel global solves of many small independent models

**Why.** Pricing is "one small global problem per phase". Diagrams and fitting
multiply that by the number of conditions.

**Current state.** No public API (`__init__.py:312` mentions batch harnesses
only; `_relax/batch_evaluator.py` is internal).

**Request.** `dm.solve_batch(models, workers=..., **solve_kwargs) -> list[SolveResult]`.
It should be process-safe with JAX (spawn, compile cache) and return results in
order.

**Acceptance.** 32 independent pricing models with `workers=8` give the same
results as serial solves and a measured speedup.

## P1-3. Dual warm starts and KKT diagnostics through `Model.solve`

**Why.** Restricted-equilibrium solves along a phase-diagram trace differ only
slightly between points. POUNCE supports primal-dual warm starts
(`pounce.WarmStart`), but discopt passes only `x0` (`solvers/nlp_pounce.py:184`).
The certificate also needs the KKT residual, to bound the |G_R − μᵀb| term.

**Current state.** `initial_solution` is primal only (`core.py:5706`). There is
no `WarmStart` in discopt, and `SolveResult` has no KKT field; `kkt_error`
exists only on internal results.

**Request.**
- `Model.solve(initial_solution=..., warm_start=previous_result)`, which forwards
  x, constraint multipliers, bound multipliers and μ to POUNCE.
- `SolveResult.kkt` = {`primal_infeasibility`, `dual_infeasibility`,
  `complementarity`, `kkt_error`} for NLP routes.

**Acceptance.**
- A re-solve after a 1% parameter change takes fewer POUNCE iterations with
  `warm_start` than with a primal-only start.
- `result.kkt["kkt_error"]` matches POUNCE's `final_kkt_error`.

## P2-1. Public operator registration hook

**Why.** CALPHAD-specific univariate operators need their own evaluator, interval
rule and tight envelope instead of a factorable decomposition. Examples: binary
Redlich–Kister plus entropy as a single function of x, and piecewise-T functions.
This is research question 3 of the prompt. Other plugins (mkm: Arrhenius/Langmuir;
thermo: activity models) need the same.

**Current state.** Absent. Operator names are hard-coded in
`_relax/relaxation_compiler.py:983-1002`, `_relax/convexity/interval_eval.py:471`,
`_nl_expr_compiler.py:1018` and Rust `expr.rs:50`. `_relax/symbolic/registry.py`
exists but the solver never reads it. `dm.custom` with MCBox-traceable bodies is
the only external route, and it has no FBBT and no `.nl` export.

**Request.**

```python
dm.register_function(
    name="rk_binary",
    arity=1,
    evaluate=jax_fn,               # (x, *consts) -> value
    derivative=jax_fn_or_None,     # else jax.grad
    interval=fn,                   # ([lo, hi], *consts) -> [lo, hi], must be sound
    relax=fn,                      # (x, lb, ub, *consts) -> (cv, cc), must be sound
    lower=callable_or_None,        # expansion to primitives for .nl/POUNCE and Rust FBBT fallback
    convexity=None | "convex" | "concave" | callable,
)
```

Registered functions become `FunctionCall` nodes usable anywhere. Where Rust
cannot evaluate the node, the solver falls back to `lower`, or treats the node
as opaque with the registered interval. A registration-time soundness check
could reuse `_relax/symbolic/verification.verify_envelope` by sampling.

**Acceptance.**
- Register an operator whose envelope is tighter than McCormick for
  `x(1−x)(L0 + L1(2x−1))`. The global solve uses it, as shown in
  `solver_stats`, and needs fewer nodes than the primitive form.
- A deliberately unsound `relax` is rejected by the verification check.

## P2-2. Multivariate relaxation hook and structure declarations (simplex)

**Why.** CEF endmember terms are products of site fractions from different
sublattices, and each sublattice's fractions lie on a simplex (y ≥ 0, Σy = 1).
Simplex-aware hulls of such products can be much tighter than recursive
McCormick. This is research question 4 and the core of §10.

**Current state.** Partial. Multilinear hull separation exists
(`_relax/multilinear_separation.py`, `edge_concave.py`) but does not know about
simplex domains. There is no way to declare a simplex, and `recognize_and_inject`
(`solver.py:8062`) is a fixed pattern.

**Request.**
- `m.simplex(vars, total=1.0)`: a declared structure. It adds the constraint and
  exposes the structure to FBBT, OBBT and relaxations.
- `dm.register_relaxation(pattern, relax_fn)` for multivariate node groups. The
  pattern matches, say, a product of variables each belonging to a declared
  simplex. `relax_fn` returns valid linear or convex inequalities in (x, w),
  given the node's box.

**Acceptance.** For the pricing problem of a 2-sublattice B2 phase, the root
bound with the simplex-aware relaxation is at least as tight as the default and
strictly tighter on a stated instance.

**Measured (MVP 4, `benchmarks/simplex_bound_study.py`).** Two formulations
were compared, each with a 30 s limit: explicit site fractions with Σy = 1 as a
plain equality, and stick-breaking box variables.
- On the reciprocal (A,B)1(A,B)2 case, the explicit form reached bound 2.710 in
  4.7k nodes. Stick-breaking reached 2.7397 in 12k nodes (optimum 2.7433).
- On CU2MG with a relative gap of 1e-4, the explicit form closed in 20.8 s and
  stick-breaking in 5.2 s.

An undeclared simplex costs bound quality. A declared one should do better
than both.

## P2-3. Cut callbacks at fractional and spatial nodes

**Why.** Gibbs-convexification ideas (§12: supporting hyperplanes of G, tangent
planes found by earlier pricing rounds, cuts carried from neighbouring (T, x)
points) are valid inequalities that should be added at any branch-and-bound node.

**Current state.** `lazy_constraints` runs only at integer-feasible nodes
(`callbacks.py:78,104`, `core.py:5722`). `NodeCallback` cannot add cuts.

**Request.** A `CutCallback(node_context) -> list[CutResult]` called at every
node, spatial branching included. The context gives the node box, the relaxation
solution and the incumbent. Cuts can be declared globally valid or local to the
node.

**Acceptance.** A pricing problem where tangent-plane cuts from a previous
solution are supplied through the callback solves with fewer nodes. Adding an
invalid cut is flagged by `validate=True`.

## P2-4. Public SymPy ↔ discopt translation

**Why.** The discopt-calphad symbolic layer is SymPy-based, and so is discopt-mkm.
A shared, tested translator avoids each plugin writing its own.

**Current state.** Only the private one-way `model_to_sympy`
(`_relax/symbolic/cut_recognizer.py:171`) exists. LaTeX output is public
(`modeling/latex.py:109`).

**Request.** `discopt.symbolic.from_sympy(expr, symbol_map) -> Expression` and
`to_sympy(expr) -> (sympy.Expr, symbol_map)`, with an extension point mapping
custom `sympy.Function` subclasses to registered discopt functions (P2-1).
SymPy stays an optional import.

**Acceptance.** Round trip on the elementary function set including `entropy`,
checked by value and gradient at random points.

## P2-5. Wire the symbolic envelope deriver to registered univariate functions

**Why.** For binary phases the pricing function is univariate in x at fixed
(T, μ), so an exact convex envelope can be derived automatically.

**Current state.** Offline only: `derive_envelope` is "design-time only" and
nothing outside `_relax/symbolic/` imports it. The SymPy-free
`_relax/symbolic/runtime.py` (`convex_envelope`, `single_inflection_envelope`)
could run at solve time.

**Request.** Allow `register_function(..., relax="derive")`: at registration,
derive the envelope from a SymPy form, codegen it, and verify it.

**Acceptance.** A registered binary RK + entropy operator gets a derived envelope
that passes `verify_envelope`.

## P2-6. Verified (outward-rounded) bound mode

**Why.** "Certified" claims in thermodynamic databases should survive rounding
scrutiny.

**Current state.** Partial internals: `_relax/outward_rounding.py` (#956) and the
Neumaier–Shcherbina `safe_bound` (`solver_tuning.py:127`). There is no
user-facing mode.

**Request.** `Model.solve(verified_bound=True)`, which uses the safe bound
throughout and reports `bound_source`.

**Acceptance.** On instances where the relaxation LP optimum lies within 1e-15 of
the true optimum, the verified bound is ≤ the exact rational optimum.

## P2-7. User branching priorities (lower priority)

**Why.** In CEF pricing, branching on a sublattice's simplex coordinates, or on
the dominant constituent first, may beat generic selection. This would be
measured before being relied on.

**Current state.** Internal only (`_branch_priority_integer_vars`,
`solver.py:91`).

**Request.** `m.branch_priority({var: int})` for continuous spatial branching.

---

## Usable today without upstream changes

- `FunctionCall("entropy", x)` with McCormick bounds (interval weakness noted in P0-1).
- `dm.custom` with MCBox-traceable bodies, for early experiments with custom operators.
- `discopt.parametric`: documented as a stable contract for plugins.
- `discopt.estimate` (Experiment, ExperimentModel, estimate_parameters) for fitting.
- `Variable.fix` / `Model.fixed` (#1241) for holding composition sets or parameters.
- `validate=True`, `compute_iis`, `infeasibility_certificate` for diagnostics.

## Suggested filing

Proposal only in this draft. P0-1 to P2-7 were later filed as discopt#1242–#1249.
P0-4 (stop from a callback, or at a bound target) was added in MVP 4 and has
not been filed yet.

- One tracking issue: "Plugin needs from discopt-calphad".
- One issue per P0 item and P1-1…P1-3, filed now, because they block or slow the MVP.
- P2 items as a single design issue, "Relaxation extension API (operators,
  multivariate, simplex structure, node cuts)", since they share one design.
