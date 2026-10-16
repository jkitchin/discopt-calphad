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

## Ni-rich Al–Ni: γ and γ′ with the native order–disorder model

γ′ (Ni₃Al, L1₂) is a partitioned phase over γ (FCC_A1) in Dupin, Ansara and
Sundman's assessment (Calphad 25 (2001) 279–298). Here it is modeled natively
as G_dis(x) + G_ord(y) − G_ord(x) (issue #6). The ordered state is a second
well in site-fraction space, and pricing finds it as a new composition set of
FCC_L12.

On the Ni-rich side (x(Al) ≤ 0.30, 900–1700 K, 261 points, none failed), the
γ + γ′, γ′ + Al₃Ni₅ and L + γ boundaries agree with pycalphad on the same
phases to five decimals. Dupin's β (B2) allows vacancies on its substitutional
sublattices, which the native model does not support yet, so both maps leave
it out. In the full assessment, β borders γ′ above x(Al) ≈ 0.27. Two points
are also certified globally (`CERTIFIED_TO_TOLERANCE`): γ + γ′ at 1200 K and
γ′ at 1000 K. Bounding the active FCC_L12 phase takes about 500 s each; every
other phase takes seconds.

![Ni-rich Al-Ni](assets/examples/alni_order_disorder.png)

??? example "examples/alni_order_disorder.py"
    ```python
    --8<-- "examples/alni_order_disorder.py"
    ```

## Ni–Al with stand-ins for its order–disorder phases

Dupin's Al–Ni assessment describes γ′ (L1₂) and β (B2) as partitioned phases,
which the solver cannot model yet. `ordering.stand_in` rewrites each as a plain
phase; both are exact (pycalphad agrees to 1e-10 J/mol), with magnetism and B2
vacancies dropped. The diagram matches pycalphad's full model to within 0.001
in x except on the Al-rich side of β (up to 0.046), where the dropped vacancies
matter.

![Ni-Al with stand-ins](assets/examples/alni_stand_in.png)

??? example "examples/alni_stand_in.py"
    ```python
    --8<-- "examples/alni_stand_in.py"
    ```


## Pure iron: the magnetic term against the SGTE data

The Inden–Hillert–Jarl magnetic term (issue #6) on pure Fe, from the SGTE unary
data (Dinsdale, Calphad 15 (1991) 317–425). The model is from Inden (1976) and
Hillert and Jarl (Calphad 2 (1978) 227–238).

- The crossings of G reproduce the tabulated transitions: α → γ at 1184.81 K
  and γ → δ at 1667.47 K to 0.01 K, and melting at 1811 K to 0.05 K.
- With the magnetic term removed, BCC is not stable anywhere from 300 to
  1400 K.
- C_p has its λ peak at 1042 K, against T_C = 1043 K.
- G agrees with pycalphad to 6e-13 relative.

![pure Fe](assets/examples/fe_magnetic.png)

??? example "examples/fe_magnetic.py"
    ```python
    --8<-- "examples/fe_magnetic.py"
    ```

## Cr–Fe with magnetic BCC: a certified diagram

The Cr–Fe diagram of the `crfenb_jacob` database. Its BCC magnetic parameters
are from Andersson and Sundman, Calphad 11 (1987) 83–92. Fe is ferromagnetic
and Cr antiferromagnetic. The magnetic term shapes:

- the γ loop;
- the low-temperature BCC miscibility gap, between a ferromagnetic and an
  antiferromagnetic set;
- the σ boundaries.

The diagram is mapped with local pricing (325 points, none failed). Then one
point per temperature from 600 to 1800 K is priced globally, splitting each
magnetic phase into its branches, and all 13 are certified to ε. pycalphad,
on the same database, finds the same two-phase regions, with every phase
boundary equal to within 5e-5 in x(Cr).

![Cr-Fe magnetic](assets/examples/crfe_magnetic.png)

??? example "examples/crfe_magnetic.py"
    ```python
    --8<-- "examples/crfe_magnetic.py"
    ```
