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

## Al–Ni with its order–disorder phases: γ′ and β

Dupin, Ansara and Sundman's assessment (Calphad 25 (2001) 279–298) describes
γ′ (Ni₃Al, L1₂) and β (NiAl, B2) as partitioned phases, modeled natively as
G_dis(x) + G_ord(y) − G_ord(x) (issue #6). Ordered states are second wells in
site-fraction space, and pricing finds them as new composition sets. β also
holds constitutional vacancies on both sublattices. It is modeled over formula
units at least half occupied (`min_occupancy`), because the model has no
global minimum below that.

The diagram covers 900–1700 K and 49 compositions (441 points, 4 failed). Every
two-phase region is the same as pycalphad's on the same database, with
boundaries equal to five decimals, including Al-rich β + Al₃Ni₂. At 1200 K, β's
vacancy fraction rises from 2e-4 to 9e-2 across its range, x(Al) = 0.38 to 0.55.

Global certification:
- γ + γ′ at 1200 K and γ′ at 1000 K are certified (`CERTIFIED_TO_TOLERANCE`).
  Bounding the active FCC_L12 takes about 500 s.
- With β active, certification does not close in the time limit
  (`GLOBAL_SEARCH_INCOMPLETE`). The bound of the four-variable B2 stalls near
  −3 after 600 s.

![Al-Ni order-disorder](assets/examples/alni_order_disorder.png)

??? example "examples/alni_order_disorder.py"
    ```python
    --8<-- "examples/alni_order_disorder.py"
    ```

## Ni–Al with stand-ins for its order–disorder phases

Dupin's Al–Ni assessment describes γ′ (L1₂) and β (B2) as partitioned phases.
Both are modeled natively now. This example replaces β with
`ordering.stand_in`'s plain phase, which is exact (pycalphad agrees to 1e-10
J/mol) except that it drops magnetism and B2 vacancies, to show what that
costs. The diagram matches pycalphad's full model to within 0.001
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

## Fe–Mn liquid: the Einstein and two-state contributions

Third-generation unary descriptions model a liquid down to 0 K. They add an
Einstein solid for the vibrations (THETA holds ln θ_E) and the two-state model
for the liquid–amorphous transition (GD), from Ågren (Phys. Chem. Liq. 18
(1988) 123–139) and the Ringberg workshop (Ågren et al., Calphad 19 (1995)
449–480). Chen and Sundman (J. Phase Equilib. 22 (2001) 631–644) describe Fe in
this form. The script uses pycalphad's test database femn.tdb: one liquid with
Xiong magnetism, Einstein and two-state terms, and no interactions.

- G agrees with pycalphad to 1e-15 relative over 28 compositions and
  temperatures.
- In pure liquid Fe, 0.035 of the atoms are liquid-like at 1000 K and 0.28 at
  the melting point. C_p rises to about 3R as the Einstein vibrations switch
  on, then climbs through a broad two-state hump. The peak at 200 K is the
  magnetic transition of this test file.
- All 15 Fe–Mn liquid equilibria (600–1800 K) are certified, with the same
  assemblage as pycalphad.

![Fe-Mn liquid](assets/examples/femn_two_state.png)

??? example "examples/femn_two_state.py"
    ```python
    --8<-- "examples/femn_two_state.py"
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

## Cu–O with the two-sublattice ionic liquid

Schramm, Behr, Löser and Wetzig's assessment (J. Phase Equilib. Diffus. 26
(2005) 605–612) describes liquid Cu–O as one ionic liquid,
(Cu⁺, Cu²⁺, Cu³⁺)_P(O²⁻, Va)_Q (Sundman, Calphad 15 (1991) 109). Its site
ratios follow from charge neutrality, so one phase spans metallic copper
(Cu⁺:Va) and the oxide melt, with a miscibility gap between them.

The map covers 1100–1700 K and x(O) ≤ 0.6:
- **Agreement:** at 352 points we find the same state as pycalphad on the same
  database.
- **Lower G:** at 18 points our state is lower in G. Both codes compute the same
  G for every phase, so ours is the better minimum. At 14 of them, pycalphad
  misses the liquid miscibility gap (✕ in the figure).
- **Missing state:** at 19 points pycalphad returns no state at all.
- **One local miss:** at one point we differ without being lower. It's a
  local-pricing miss of the second liquid near the invariant; global pricing
  would find it.

The eutectic L → FCC + Cu₂O comes out at 1339.4 K (1066.3 °C). The monotectic
L₂ → L₁ + Cu₂O is at 1497.1 K, where Cu₂O touches the common tangent of the
two liquids (x(O) = 0.093 and 0.309). Certification works for the solid
assemblages: Cu₂O + CuO at 1000 K is `GLOBALLY_CERTIFIED`. With the liquid
active it ends `GLOBAL_SEARCH_INCOMPLETE`.

![Cu-O](assets/examples/cuo_ionic.png)

??? example "examples/cuo_ionic.py"
    ```python
    --8<-- "examples/cuo_ionic.py"
    ```

## ZrO₂–Nd₂O₃: charged sublattice phases

Fabrichnaya and Seifert's assessment (Calphad 32 (2008) 142–151) describes the
oxide solid solutions with charged species, held neutral by an equality on
their site fractions:
- fluorite, tetragonal and monoclinic ZrO₂ (Nd³⁺, Zr⁴⁺)₂(O²⁻, Va)₄;
- pyrochlore Nd₂Zr₂O₇;
- the A, H and X forms of Nd₂O₃;
- the ionic liquid.

With fixed valences, one neutrality or balance row is redundant, and the
restricted solve leaves it out.

The section from ZrO₂ to NdO₁.₅ covers 1200–2800 K (425 points). Every point
solves, and 423 have the same phases and compositions as pycalphad. At the
other two, a trace of pyrochlore sits next to fluorite. The two states differ by
0.03–0.04 J/mol, within the gas-constant difference between the codes.

![ZrO2-Nd2O3](assets/examples/zro2_nd2o3.png)

??? example "examples/zro2_nd2o3.py"
    ```python
    --8<-- "examples/zro2_nd2o3.py"
    ```
