# Examples

Each example is a script in `examples/`. Run it from the repository root with
`uv run python examples/<name>.py`; the figures shown here are its stored output.

## Regular solution with a miscibility gap

The MVP 1 problem: a binary regular solution whose pricing step finds the second
composition set and certifies the gap.

![regular solution gap](assets/examples/regular_solution_gap.png)

??? example "examples/regular_solution_gap.py"
    ```python
    --8<-- "examples/regular_solution_gap.py"
    ```

## Several phases: a two-phase lens

Redlich–Kister solutions and T-dependent energies, with phases appearing and
disappearing (MVP 2).

![multiphase lens](assets/examples/multiphase_lens.png)

??? example "examples/multiphase_lens.py"
    ```python
    --8<-- "examples/multiphase_lens.py"
    ```

## Al–Zn from a TDB database

A published assessment read from TDB into a symbolic `ThermoDatabase` (MVP 3).

![Al-Zn from TDB](assets/examples/alzn_tdb.png)

??? example "examples/alzn_tdb.py"
    ```python
    --8<-- "examples/alzn_tdb.py"
    ```

## Cu–Mg with a compound-energy-formalism Laves phase

CU2MG as a two-sublattice `CEFPhase` (MVP 4). With CU2MG active, the certificate
is currently `GLOBAL_SEARCH_INCOMPLETE`; see the architecture notes.

![Cu-Mg CEF](assets/examples/cumg_cef.png)

??? example "examples/cumg_cef.py"
    ```python
    --8<-- "examples/cumg_cef.py"
    ```

## Fitting Al–Zn parameters to data

Synthetic mixing enthalpies, entropies, activities and a tie-line refitted to
recover six interaction coefficients, then checked with a certified equilibrium.

![Al-Zn fit](assets/examples/alzn_fit.png)

??? example "examples/alzn_fit.py"
    ```python
    --8<-- "examples/alzn_fit.py"
    ```

## Al–Cu from a machine-learned potential

Formation and SQS mixing enthalpies from an ASE calculator (UMA here), compared
with the Zhang assessment in kJ/mol. The enthalpy parts of the FCC
interactions and of θ-Al2Cu are then refitted to the calculated data alone, and
the Al-rich diagram is drawn for both databases (issue #5).

![Al-Cu from UMA](assets/examples/alcu_mlip_uma.png)

??? example "examples/alcu_mlip.py"
    ```python
    --8<-- "examples/alcu_mlip.py"
    ```

## Cu–Mg fitted to real data, against the assessment

The Cu–Mg datasets curated for ESPEI, read with `Dataset.from_espei`:

- liquid mixing enthalpies and Mg activities;
- formation enthalpies of Cu2Mg and CuMg2;
- DFT mixing enthalpies of FCC;
- 238 phase-boundary points from 1907–1978, each with only one phase's
  composition.

Ten coefficients of the liquid, FCC, Cu2Mg and CuMg2 parameters are refitted.
One-sided points enter as driving forces, and the fit iterates linear solves.
The fitted diagram (red) is drawn over the NIMS assessment (gray) and the data.

![Cu-Mg fit to ESPEI data](assets/examples/cumg_espei_fit.png)

??? example "examples/cumg_espei_fit.py"
    ```python
    --8<-- "examples/cumg_espei_fit.py"
    ```
