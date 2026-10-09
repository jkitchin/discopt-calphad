# Recipe: a phase diagram from a calculator (MLIP, EAM, EMT)

A calculator gives 0 K energies of solid configurations. A phase diagram needs
much more:

- G(y, T) of every competing phase;
- the liquid;
- the pure elements;
- every entropy and heat-capacity term.

So the calculator **refits the enthalpy parts of an existing description**. It
does not replace the description. The steps:

1. Start from a database that has the system's phases.
2. Check that the calculator describes the system at all.
3. Refit only the enthalpy constants.
4. Map the diagram of the fitted and of the original database, and compare them.

The code below runs as written. It uses EMT on Al–Cu so it needs nothing
beyond ASE and icet; the same steps apply to UMA or MACE with their calculators.

## 1. A starting database, and what it can model

```python
import numpy as np

from discopt.calphad import databases, equilibrium, mlip
from discopt.calphad.equilibrium.certificate import CertificateLevel
from discopt.calphad.fit import enthalpy_form, fit, predict

EL = ("AL", "CU")
print([e.name for e in databases.find(EL)])            # registered files that form Al–Cu
db = databases.load("alcuy_zhang")
problems = {k: v for k, v in db.applicability(EL).items() if v != "ok" and "NotApplicable" not in v}
print(problems)                                         # {} here; see "When a key phase is unsupported"
print(databases.get("alcuy_zhang").unsupported(EL))    # (): phases of this binary the package cannot model
```

Without a database for the system there is nothing to refit. You would have to
write the phases, the SGTE unaries and a liquid by hand (`add_tdb` accepts
PHASE, CONSTITUENT, FUNCTION and PARAMETER lines). Say so to the user rather
than inventing them.

## 2. Compute the data, then check the calculator

```python
from ase.calculators.emt import EMT   # UMA: FAIRChemCalculator(pretrained_mlip.get_predict_unit("uma-s-1p1"), task_name="omat")

model = mlip.from_ase(EMT(), name="EMT", cache=True)
result = mlip.compute(
    model, db, EL,
    compounds={"ALCU_THETA": ["AL:CU"]},
    structures={"ALCU_THETA": ("C16", ["8h"], ["4a"])},
    mixing={"FCC_A1": [0.25, 0.5, 0.75]},
    model_sigma=2.0,                                   # kJ/mol; see σ below
    sqs=mlip.SQSOptions(n_steps=200),
)
print(result.table())

assessed = predict(db, result.dataset, EL)              # what the database says about the same quantities
for o, a in zip(result.dataset, assessed):
    print(f"{o.kind:8s} {o.phase:12s} {o.value / 1000:8.1f} {a / 1000:8.1f}  kJ/mol (calculator, assessment)")
```

**Always make this comparison, and stop if it fails.**

- **The check.** Compare signs and magnitudes against the assessment, or
  against any known formation enthalpy.
- **When to stop.** Errors of tens of kJ/mol, or the wrong sign, mean the
  calculator cannot describe the system. The fit and the diagram would then be
  meaningless, and certifying them proves nothing about the alloy.
- **This example fails the check.** ASE's EMT is a toy: its cross-interactions
  are not fitted. Here it gives θ-Al2Cu ΔH_f ≈ +15 kJ/mol against −15.8 in the
  assessment, and it gets Ni–Al wrong by 50–100 kJ/mol in sign. The rest of
  this recipe continues with it only to show the mechanics.
- **What a good model gives.** UMA (`uma-s-1p1/omat`) is within about
  1 kJ/mol for Al–Cu.
- **If the check fails and the user still wants the result,** produce it, but
  label every derived figure and number as a demonstration of the pipeline,
  not a prediction for the alloy. Put the comparison table next to it, and say
  which calculator would be worth trying instead.

## 3. Refit the enthalpy parts

```python
free = {"G(ALCU_THETA,AL:CU;0)": enthalpy_form(db, "G(ALCU_THETA,AL:CU;0)")}
for k in range(2):                                      # 3 compositions: fit L0 and L1, keep the assessed L2
    label = f"L(FCC_A1,AL,CU:VA;{k})"
    free[label] = enthalpy_form(db, label)              # "a + <the assessed T-dependence>"
print(free)
fitted = fit(db, result.dataset, EL, free)
print(fitted.summary())
```

- **Fit only enthalpy constants.** 0 K data say nothing about entropy.
  `enthalpy_form` frees the constant and keeps the database's T-dependence,
  GHSER functions included.
- **Fit every endmember of a phase that the data cover.** An endmember you
  leave out keeps the assessment's value, and it can then dominate. In one
  eval, an unfitted Al₃Ni₂ endmember at −427 kJ kept the phase stable although
  the calculator made it unstable. `mlip.compute(compounds=[phase])` without a
  list computes every endmember with two or more elements.
- **Equivalent endmembers cannot be tied.** For example AL:NI and NI:AL of a
  B2 phase. Compute and fit both, since equal structures give equal data.
  Otherwise copy the fitted value to the other label afterwards.
- **Keep what the calculator does not supply:** the liquid, the pure elements,
  and any phase you have no structure for.
- **But a partial refit can be unbalanced.** If the calculator moves the solids
  a long way, a liquid or compound kept at its assessed values will dominate.
  In the Ni–Al eval, the assessed liquid (L0 ≈ −207 kJ) stayed stable down to
  about 1000 K once EMT had made every solid positive. Compare the refit
  solids with the kept phases (`predict` on a few points of each). Report a
  large mismatch as a limit of the result, rather than tuning the kept phases
  to hide it.
- **σ_model** is your estimate of the calculator's error. The comparison in
  step 2 is the best evidence for it. The fit's χ² cannot see a systematic
  offset, because the fit absorbs it.

## 4. Map and compare the diagrams

There is no diagram function yet. Collect two-phase tie-lines on a T–x grid
with local solves, keep each once, and skip points the solver did not converge:

```python
def tie_lines(database, temps, xs):
    phases = list(database.phase_models(EL).values())
    out, failed = [], 0
    for T in temps:
        seen = set()
        for x in xs:
            r = equilibrium(phases, x=float(x), T=float(T), certify="none")
            if r.certificate.level < CertificateLevel.STABILITY_SEARCHED:
                failed += 1                             # see "Points that fail" below
                continue
            sets = sorted((cs.phase.name, float(cs.x[0])) for cs in r.composition_sets)
            key = tuple((n, round(v, 3)) for n, v in sets)
            if len(sets) == 2 and abs(sets[0][1] - sets[1][1]) > 1e-3 and key not in seen:
                seen.add(key)
                out.append((float(T), sets))
    return out, failed


for name, database in [("assessed", db), ("fitted", fitted.database)]:
    lines, failed = tie_lines(database, [600.0, 800.0], np.linspace(0.6, 0.95, 5))
    print(name, len(lines), "tie-lines,", failed, "failed points")
```

Plot each tie-line's two ends at its T, for both databases, on one T–x axis.
The fitted database also has `fitted.to_tdb()`, with a provenance header.

## When a key phase is unsupported

- **Order–disorder phases.** `applicability()` may report
  `order-disorder (DISORDERED_PART) models are not supported yet`, often for
  the phases that matter most (B2 and L1₂ in Ni–Al and Fe–Al). Options:
  - Tell the user, and stop there.
  - Compare against pycalphad, if it is installed. It models these phases, and
    the package itself never imports it.
  - Add a plain ordered stand-in with `add_tdb` (PHASE, CONSTITUENT and
    PARAMETER lines). Its endmembers must be derived from the partitioned
    model by hand. Say plainly that the stand-in has no ordering transition,
    and that its homogeneity range is too narrow.
- **Magnetic phases** (FCC Ni, BCC Fe, Co) fit normally; only TC and BMAGN
  cannot be freed. Any equilibrium that contains them certifies at most to
  `GLOBAL_SEARCH_INCOMPLETE`. So Ni-, Fe- and Co-based systems cannot be fully
  certified yet.

## Structures

- `mlip.PHASES` gives the generic names. Note that `B2_BCC`, `L12_FCC` and
  `D019_HCP` expect the three-sublattice layout `(A,B)(A,B)(VA)`. A phase whose
  layout differs needs its own map.
- **Registered prototypes** (`mlip.PROTOTYPES`: A1–A4, B2, L1₂, D0₁₉, C14, C15,
  C36, C16, D8_b) take `structures={"PHASE": ("C16", ["8h"], ["4a"])}`, with
  one entry of Wyckoff sites per sublattice, or None for a vacancy-only one.
- **Anything else** takes `structures={"PHASE": (atoms, sublattices)}`. `atoms`
  is an `ase.Atoms` of the cell, and `sublattices` lists the atom indices of
  each sublattice in the phase's order (None for a vacancy-only one). The
  symbols don't matter; each endmember decorates the sites anew.
- Endmembers with a single element (for example `AL:AL`) are lattice
  stabilities, and are computed only when you list them explicitly.
- `mixing=` (SQS) needs exactly one sublattice that mixes the two elements,
  without VA. Every other sublattice must be fixed. A phase such as Dupin's
  BCC_A2, (AL,NI,VA)(VA), is refused.

## Points that fail, and time

- **What the levels mean.** `FAILED` means the local solve did not converge.
  `LOCAL_EQUILIBRIUM` after `max_iter` means a phase kept destabilizing. Drop
  such points, or retry at x ± 1e-3. Raising `max_iter` rarely helps. Mark them
  on the plot rather than hiding them.
- **Local solves.** These take tens of ms with simple phases, and 0.5–30 s once
  several multi-sublattice phases compete. Size the grid accordingly, and run
  large sweeps in the background.
- **Certification.** It takes 10 s to 10 min per point. Certify a few points
  that matter, not the whole grid.
- **Calculator time.** A UMA relaxation of a 10–30 atom cell takes tens of
  seconds on a CPU. EMT is instant. `cache=True` makes reruns free.
