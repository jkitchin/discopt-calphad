---
name: discopt-calphad
description: >
  Use when computing CALPHAD phase equilibria, phase diagrams or thermodynamic
  properties with discopt-calphad (the discopt.calphad package): loading a TDB
  file or fetching a published database by name, solving and certifying an
  equilibrium, reading a certificate level, viewing or editing parameters
  symbolically and exporting TDB, plotting G–x curves, fitting CALPHAD
  parameters to experimental data, or computing CALPHAD data (formation and
  mixing enthalpies) from an ASE calculator such as a machine-learned
  potential (UMA, MACE), EAM or EMT, including making a phase diagram of an
  alloy from such a calculator. Also use it when writing a custom phase model
  for its solver.
---

# discopt-calphad

discopt-calphad computes CALPHAD phase equilibria with a **certificate**. A
local solver (POUNCE) finds a candidate equilibrium. discopt's global solver
then prices every phase, which proves that no phase or composition set is
missing, to a stated tolerance. pycalphad serves as a test oracle only; the
package never imports it, but if it is installed it is a useful reference for
phases this package cannot model.

Know the limits before you promise a result:

- **Systems:** binary systems are best supported. Ternaries work for
  equilibria and thermochemical fits, but not for μ-based fit data.
- **Unsupported phases:** order–disorder phases (B2, L1₂ in Ni–Al and Fe–Al)
  cannot be modelled directly, but `ordering.stand_in` turns each into a plain
  phase (see section 1). Ionic liquids cannot be modelled.
- **Magnetism:** any equilibrium containing a magnetic phase (FCC Ni, BCC Fe,
  Co) certifies at most to `GLOBAL_SEARCH_INCOMPLETE`.
- **Calculators:** a calculator (MLIP, EAM, EMT) gives data for a fit, never a
  phase model. For a phase diagram from one, follow
  [reference/mlip-phase-diagram.md](reference/mlip-phase-diagram.md).

Import everything from `discopt.calphad`. It is a namespace subpackage of
`discopt`.

## 1. Get a database

Published TDB files are fetched by name from a registry, downloaded once and
checked against a pinned SHA-256. A local file works too:
`ThermoDatabase.from_tdb(path)`.

```python
from discopt.calphad import databases

db = databases.load("alzn_mey")                        # a ThermoDatabase
[e.name for e in databases.find(["AL", "ZN"])]          # every registered file that forms Al–Zn
entry = databases.get("alzn_mey")
print(entry.reference)                                  # the assessment the parameters come from
print(databases.find(["CR", "FE"], complete=True)[0].name)  # a file that models every Cr–Fe phase
```

Always check what the database can model in your system before trusting a
result. A phase the package cannot model is **skipped**, not approximated:

```python
status = db.applicability(("AL", "ZN"))     # {phase: "ok" | "<Error>: reason"}
assert all(v == "ok" for v in status.values()), status
phases = db.phase_models(("AL", "ZN"))      # {name: phase model}, supported phases only
print(sorted(phases))                        # ['FCC_A1', 'HCP_A3', 'LIQUID']
```

`databases.get(name).unsupported(elements)` lists the same unsupported phases
without loading the file, and `databases.fetch(name)` gives the local TDB path
(for pycalphad, say). A large multicomponent file such as `cost507` can form a
binary while lacking that binary's intermetallics: its other phases still show
up as "ok", through their pure-element endmembers. Prefer a file assessed for
the system, and list which phases it actually has before using it.

If a phase the user cares about is unsupported, say so before computing
anything. An order–disorder (`DISORDERED_PART`) phase can be rewritten as a
plain stand-in with `discopt.calphad.ordering.stand_in(db, "FCC_L12",
elements)`. For Ni–Al the stand-in is exact, but it drops magnetism and
vacancies on substitutional sublattices, so report its `dropped` list. The
details are in [reference/mlip-phase-diagram.md](reference/mlip-phase-diagram.md),
"When a key phase is unsupported".

The order of the element tuple matters: `x` below is the mole fraction of the
**first** element.

## 2. Solve an equilibrium

```python
from discopt.calphad import equilibrium

res = equilibrium(list(phases.values()), x=0.5, T=500.0, certify="none")  # x = x_AL, T in K
print(res.summary())
for cs in res.composition_sets:          # one per coexisting phase instance
    print(cs.phase.name, cs.x, cs.amount)   # overall mole fractions, amount in formula units
print(res.phase_names, res.mu, res.G)    # mu and G are in units of RT (see Conventions)
```

- Give the composition either as `x=` (binary only) or as element amounts `b=`.
  In a ternary: `b=(0.2, 0.3, 0.5)`, in the order of the phase model's
  `elements`.
- The same phase can appear twice, for example two FCC_A1 sets in a
  miscibility gap.
- `cs.y` holds the model's variables, not site fractions. A one-sublattice
  solution has `[y_first]`. A CEF phase has stick-breaking variables, and
  `cs.phase.site_fractions(cs.y)` turns them into one dict per sublattice
  (which also tells an ordered set from a disordered one).
- `certify` modes:
  - `"final"` (default): local pricing while composition sets are generated,
    then global pricing once. From 10 s to 10 min per equilibrium.
  - `"none"`: local pricing only. About 0.05 s for Al–Zn. Use it for sweeps and
    diagrams, then call `res.certify()` on the points that matter. With
    several multi-sublattice phases competing, it takes 0.5–30 s per point.
- A result below `STABILITY_SEARCHED` did not converge; see the next section.
  In a sweep, drop such points or retry at x ± 1e-3, and mark them on the plot.
  Raising `max_iter` rarely helps.
  - `"every"`: global pricing in every iteration. Slowest; for debugging.

```python
# slow
cert = res.certify()                     # global pricing at the stored μ, about 20 s here
print(cert.level.name, cert.gap_bound)   # CERTIFIED_TO_TOLERANCE, bound on G_R − G_global in RT
```

## 3. Read the certificate

`res.certificate.level` is a `CertificateLevel`, ordered from weakest to
strongest:

| level | what it guarantees |
|---|---|
| `FAILED` | the restricted (local) solve did not converge. Nothing. |
| `LOCAL_EQUILIBRIUM` | a local minimum. Either no pricing ran, or a phase still destabilizes (the result is not the equilibrium). |
| `STABILITY_SEARCHED` | a local search over every phase found nothing destabilizing. This is what `certify="none"` and the browser app reach. A search is not a proof. |
| `GLOBAL_SEARCH_INCOMPLETE` | global pricing ran, but some phase has no valid lower bound, or its bound is below −ε. Read `certificate.messages` for which phase and why. Magnetic phases are always here for now. |
| `CERTIFIED_TO_TOLERANCE` | every phase is proven to have Ψ ≥ −ε. No missing phase or set can lower G by more than `gap_bound` (in RT). |
| `GLOBALLY_CERTIFIED` | as above, with ε at 1e-9 and a KKT residual ≤ 1e-8. |

`certificate.lower_bounds` and `upper_bounds` give the per-phase pricing
bounds. `certificate.messages` explains anything short of certified.

## 4. Symbolic parameters, editing, export

The database keeps every FUNCTION and PARAMETER as SymPy expressions:

```python
liq = phases["LIQUID"]
print(liq.L[0].expr())                    # 10465.5 - 3.39259*T, the L0 interaction of the liquid
print(db.evaluate("GHSERAL", 700.0))      # a named FUNCTION, in J, at 700 K

db2 = databases.load("alzn_mey")
db2.add_tdb("PARAMETER G(LIQUID,AL,ZN;1) 298.15 -500; 6000 N !")  # edit in TDB syntax
db2.add_tdb("PHASE AZ % 2 1 1 ! CONSTITUENT AZ :AL:ZN: !")  # any TDB command, a new phase too
db2.add_tdb("PARAMETER G(AZ,AL:ZN;0) 298.15 -2000+GHSERAL#+GHSERZN#; 6000 N !")
text = db2.to_tdb()                       # write TDB
from discopt.calphad import ThermoDatabase

assert len(ThermoDatabase.from_string(text).parameters) == len(db2.parameters)
```

Parameters with the same label **add**. Adding a line that already exists
doubles it rather than replacing it.

A CEF phase (several sublattices) gives its full G as SymPy with
`phase.gibbs_expr()`.

## 5. Plot G–x curves

Needs matplotlib, the `plot` extra.

```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from discopt.calphad.plot import plot_gx

fig, ax = plt.subplots()
plot_gx(list(phases.values()), result=res, T=500.0, ax=ax)   # curves, the μ tangent, the sets
fig.savefig("alzn_500K.png")
```

## 6. Phase diagrams

There is no diagram function in the package yet. A T–x diagram is a sweep of
`certify="none"` equilibria; certify selected points afterward. For each point,
record the phases and their compositions. The coexisting compositions are the
phase boundaries. A tie-line collector, which keeps each tie-line once and
skips failed points, is in
[reference/mlip-phase-diagram.md](reference/mlip-phase-diagram.md), step 4.

```python
import numpy as np

rows = []
for T in (500.0, 600.0):
    for x in np.linspace(0.05, 0.95, 7):
        r = equilibrium(list(phases.values()), x=float(x), T=T, certify="none")
        rows.append((T, float(x), [(cs.phase.name, round(float(cs.x[0]), 4)) for cs in r.composition_sets]))
print(rows[0])
```

The browser app (`web/`, published at
https://kitchingroup.cheme.cmu.edu/discopt-calphad/app/) does this with adaptive
refinement, boundary tracing and invariants.

## 7. Fit parameters to data

Parameters enter G linearly, so for most data the fit is weighted linear
least squares and needs no initial guess. Data kinds:
- `GM`, `HM`, `SM`, `CPM` and their `_MIX` forms, via `add_thermo`;
- activities, via `add_activity`;
- phase equilibria via `add_equilibrium`: tie-lines, invariants, and one-sided
  boundary points (a liquidus point is `[("LIQUID", 0.3), ("FCC_A1", None)]`).

A phase whose composition is `None` must touch the tangent of the measured
one (its driving force is zero), and μ of a multi-sublattice phase comes from
its internal equilibrium.

**No spurious phases.** `fit(..., consistency="global")` checks every phase
that a phase-equilibrium observation does not list, with certified global
pricing at that observation's μ. A phase more than `consistency_eps` (J/mol,
default 10) below the tangent becomes a cut, and the fit is solved again.
`res.consistency["status"]` is then `CERTIFIED`, `INCOMPLETE` (magnetic
phases have no bound yet), or `NOT_CERTIFIED`. Each round costs one global
pricing per phase per observation, from seconds upward.

**Uncertain compositions.** Give `add_equilibrium(..., sigma_x=0.005)` and
`fit(..., errors_in_variables=True)`. The measured compositions are then
fitted too, and the parameter σ are calibrated: χ²/dof ≈ 1, and 97% coverage
in a 40-seed test. Set the row `sigma` small (model error only). Without
this, σ_x must be converted to J/mol by hand, and the resulting σ are not
reliable. With the dilute-slope conversion they came out about 3.5× too
large. Both depend on the parameters, so the fit then
iterates linear solves (`res.iterations`, `res.converged`), starting from the
database's own values or from `initial={coefficient: value}`.

```python
import dataclasses

from discopt.calphad.fit import Dataset, fit

x = np.linspace(0.1, 0.9, 9)
L0 = (10465.5, -3.39259)                  # the true liquid L0 = a + b*T
data = Dataset()
data.add_thermo("HM_MIX", "LIQUID", 1000.0, x, x * (1 - x) * L0[0], 1.0)   # H_mix = x(1-x) a
data.add_thermo("SM_MIX", "LIQUID", 1000.0, x, -x * (1 - x) * L0[1] - 8.314462618 * (x * np.log(x) + (1 - x) * np.log(1 - x)), 1e-3)

start = dataclasses.replace(db, parameters=[p for p in db.parameters if p.label != "G(LIQUID,AL,ZN;0)"])
fitted = fit(start, data, ("AL", "ZN"), {"L(LIQUID,AL,ZN;0)": "a + b*T"})
print(fitted.summary())                   # values ± σ, residuals by kind
new_tdb = fitted.to_tdb()                 # with a provenance header
```

`FitResult` also has `database`, `correlation`, `residuals_by_kind()`,
`plot_residuals(ax)` and `check_equilibria()`. The last solves a certified
equilibrium at each tie-line datum.

- **Magnetic phases.** G and L parameters of a magnetic phase fit normally,
  and so does data referenced to one, such as a formation enthalpy against
  magnetic FCC Ni. TC and BMAGN cannot be freed, because G is not linear in
  them.
- **Partly fitted phases.** Fit every parameter of a phase that the data
  cover. An endmember or interaction you leave out keeps its old value, and
  can make the phase stable or unstable for reasons the data never saw.
- **Names.** Fitted coefficients are named `"<label>.<coefficient>"`, for
  example `fitted.parameters["L(LIQUID,AL,ZN;0).a"]`. These are also the keys
  of `bounds=` and `initial=`.
- **Comparing data with a database.** `predict(db, data, elements)` gives
  what a database says about each observation, with no fit.
  `enthalpy_form(db, label)` gives a form whose constant is free and whose
  T-dependence is the database's: the form to fit to enthalpies alone.

ESPEI JSON datasets are read directly. Phases are renamed to the database's
names, and anything that cannot be used is listed with the reason:

```python
from pathlib import Path

import discopt.calphad

cumg = databases.load("cumg_nims")
espei_dir = Path(discopt.calphad.__file__).parents[3] / "examples" / "data" / "espei-cu-mg"  # in a checkout
espei = Dataset.from_espei(espei_dir, cumg, ("CU", "MG"), phase_map={"LAVES_C15": "CU2MG"})
print(len(espei), espei.skipped[0])          # 317 observations; (file, reason) for what was not used
liquid = Dataset([o for o in espei if o.phase == "LIQUID" and o.kind in ("HM_MIX", "ACR")])
res_liq = fit(cumg, liquid, ("CU", "MG"), {"L(LIQUID,CU,MG;0)": "a + b*T", "L(LIQUID,CU,MG;1)": "a"})
print(res_liq.summary())
```

ESPEI's default σ apply (500 J/mol for H, 1000 J/mol for a ZPF driving
force, ...); `sigma={"HM": 300.0}` overrides them. The full Cu–Mg fit, with
all 238 phase-boundary points, is `examples/cumg_espei_fit.py`.

More on data and forms:
- `add_formation(phase, value, sigma, endmember="AL:CU")` adds an enthalpy of
  formation (`HM_FORM`, J/mol of atoms) relative to the elements' reference
  phases. Thermochemical observations may give `site_fractions` (one dict per
  sublattice) instead of `x`, which reaches every endmember of a CEF phase.
- A form may use database FUNCTIONs, as compound parameters do:
  `{"G(ALCU_THETA,AL:CU;0)": "a + b*T + 2*GHSERAL + GHSERCU"}`.
- Every `Observation` has a `source` label (and a `note` for approximations);
  `to_tdb()` lists the sources in its header.

## 8. Data from an ASE calculator (MLIPs, EAM, EMT)

`discopt.calphad.mlip` turns any ASE calculator into fitting data: formation
enthalpies of compounds and CEF endmembers (against the elements in their
reference structures, same model), and mixing enthalpies from SQS (icet).
Needs the `mlip` and `sqs` extras. The calculator produces **data**, never a
phase model, so the certificate covers the fitted model only; the
calculator's error enters only through σ.

```python
from ase.calculators.emt import EMT   # or FAIRChemCalculator (UMA), mace_mp(), EAM, ...

from discopt.calphad import mlip

alcu = databases.load("alcuy_zhang")
model = mlip.from_ase(EMT(), name="EMT")            # name = the source label on every observation
result = mlip.compute(
    model, alcu, ("AL", "CU"),
    compounds={"ALCU_THETA": ["AL:CU"]},            # endmembers; a list of phases takes all of them
    structures={"ALCU_THETA": ("C16", ["8h"], ["4a"])},  # phases mlip.PHASES does not know
    mixing={"FCC_A1": [0.5]},                       # site fractions of AL on the mixing sublattice
    model_sigma=2.0,                                # kJ/mol, the calculator's own error
    sqs=mlip.SQSOptions(n_steps=200),               # 2 SQS per composition; fewer annealing steps
)
print(result.table())                               # kJ/mol, and any structure that left its prototype
fitted_alcu = fit(alcu, result.dataset, ("AL", "CU"), {"G(ALCU_THETA,AL:CU;0)": "a + 6.75*T + 2*GHSERAL + GHSERCU"})
```

- **σ.** σ² = σ_config² + σ_model². σ_config is the SQS spread, pooled over
  a phase's compositions by default (`pool=True`), because two SQS alone give
  a σ with one degree of freedom. σ_model is `model_sigma=` (kJ/mol, a number
  or a dict by phase), or the model's ensemble spread from `from_ase(...,
  uncertainty=)`. It is usually the larger part, and nothing computes it for
  you. A formation enthalpy (one structure) without either **raises**; SQS
  data without σ_model **warn**. `sigma=` sets σ directly.
- **Relaxation.** `RelaxPolicy(mode="symmetry" | "full" | "fixed", fmax=,
  steps=)`. A structure that leaves its prototype (symmetry-breaking cell
  strain, large displacements, a space-group change, or no convergence) is
  listed in `result.flagged` and left out of the data.
- **Check the calculator first.** Compare its values with the database's
  before fitting: `predict(db, result.dataset, elements)`. ASE's EMT is a toy
  whose cross-interactions are not fitted. It gets Al–Cu and Ni–Al formation
  enthalpies wrong in sign, so use it only to test the pipeline.
- **Structures.**
  - `mlip.PHASES` maps generic names (FCC_A1, BCC_A2, HCP_A3, L12_FCC, B2_BCC,
    D019_HCP, LAVES_C14/C15/C36, SIGMA) to prototypes. B2_BCC, L12_FCC and
    D019_HCP expect the layout `(A,B)(A,B)(VA)`.
  - Other names take `(prototype, sites per sublattice...)`;
    `mlip.PROTOTYPES` lists the prototypes.
  - A structure no prototype covers takes `(ase.Atoms, atom indices per
    sublattice)`.
  - Endmembers with a single element are computed only when listed
    explicitly.
- **Cache.** `from_ase(..., cache=True)` keeps relaxations on disk, keyed by
  the source label: change the label when the model changes.
- A 0 K energy stands in for H at 298.15 K, and every observation's `note`
  says so. Fit only enthalpy coefficients from such data, and keep or fit the
  T coefficients from other data. Only `method="sqs"` exists yet.

## Conventions and pitfalls

- **Units.**
  - `phase.gibbs(y, T=T)` is **G/RT per formula unit**, not J/mol.
  - `res.mu` is μ/RT per mole of each element.
  - `res.G` is G/RT for the whole amount `b`.
  - To get J per mole of atoms: `G * R_GAS * T / atoms`, with
    `atoms = sum(phase.amounts(y))`.
- **R.** `R_GAS = 8.314462618` (CODATA). pycalphad uses 8.3145, so comparisons
  differ by about 4e-5·T·S_mix J/mol. The tests correct for this.
- **x is the first element.** `equilibrium(..., x=0.3)` with elements
  `("AL", "ZN")` means x_Al = 0.3. `cs.x` lists every element in the same order.
- **Temperature ranges.**
  - TDB ranges are `[lo, hi)`, and the last one includes its upper end, as in
    pycalphad. This matters at a discontinuous breakpoint.
  - A T outside the ranges raises `ValueError`. Set
    `ThermoDatabase.extrapolate = True` to use the nearest range instead.
- **Unsupported models raise with the reason.** `UnsupportedModelError` covers:
  - order–disorder (`DISORDERED_PART`), unless replaced by `ordering.stand_in`;
  - ionic liquids;
  - phases whose every sublattice can be empty (an all-vacancy configuration);
  - the Xiong magnetic model;
  - some high-order CEF interactions.

  `phase_models()` skips such phases silently, so check `applicability()`.
  Inden–Hillert–Jarl magnetic phases are supported, but certify at most to
  `GLOBAL_SEARCH_INCOMPLETE` (no interval bounds for that term yet).
- **Speed.**
  - A local (`certify="none"`) solve takes tens of milliseconds with simple
    phases, and 0.5–30 s when several multi-sublattice (CEF) phases compete.
  - Certification takes 10 s to 10 min, and longer for active CEF phases.
  - Global pricing uses `time_limit` seconds per phase at most (default 600).
  - A UMA relaxation takes tens of seconds on a CPU.
- **Stoichiometric phases** have `n_y == 0` and are priced exactly.
- **Browser build.** Nothing certifies in the browser; discopt's Rust extension
  has no WebAssembly build.

## More

- `docs/architecture.md`: the formulation, the certificate theorem, the solver
  decomposition, and §9 (every convention, measured against pycalphad).
- `docs/databases.md`: the registry, and where to find more TDB files.
- `docs/mlip.md`: data from ASE calculators, and what the certificate covers.
- [reference/mlip-phase-diagram.md](reference/mlip-phase-diagram.md): a phase
  diagram from a calculator, step by step, and what to do about unsupported
  phases, failed points and time.
- [reference/phase-protocol.md](reference/phase-protocol.md): the interface a
  custom phase model implements.
