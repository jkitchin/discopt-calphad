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
