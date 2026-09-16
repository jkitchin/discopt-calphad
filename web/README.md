# The browser app

A phase diagram computed in the browser: pick two elements, get a T–x diagram,
with the points drawn as they are solved. No server, no upload, no install.

```bash
./web/build-web.sh                      # stage the wheels and databases
python3 -m http.server -d web 8000      # then open http://localhost:8000
```

## What runs where

| | |
|---|---|
| `index.html` | markup, styling, the periodic table's shell |
| `app.js` | the page: element selection, adaptive sampling, the SVG plot, phase boundaries, the hover tooltip, streaming, the detail panel |
| `elements.js` | all 118 elements with their positions, generated from ASE's data |
| `worker.js` | boots Pyodide, installs the three wheels, and runs one solve at a time |
| `calphad_app.py` | the Python driver — databases in, JSON out. Runs unchanged off-browser |
| `pure_wheel.py` | repacks discopt as a pure-Python wheel (see below) |
| `build-web.sh` | stages `wheels/` and `data/`; `--check` verifies what is staged |

Solving happens in a worker because a solve is synchronous: on the main thread
the page would freeze and Cancel could never be read. Cancel is a
`worker.terminate()`, for the same reason — nothing else can interrupt one.

## Reading the diagram

Boundaries are traced from the tie-lines, not contoured. In a two-phase field
the coexisting compositions *are* the boundary, so every multi-phase point
already carries one exact boundary composition per phase, and joining those by
temperature draws the solvus, solidus and liquidus at the precision they were
solved to. A contour could never be finer than the sampling.

### Where the points go

Not a uniform grid, and the page decides rather than `linspace` — a shape
`linspace` cannot describe. One fact drives the whole design: **in a binary at
fixed T and P the boundary compositions are a function of T alone.** Every
overall composition inside a two-phase field sits on the same tie-line and
reports the same pair of compositions. So refining in x adds no boundary detail
whatsoever — only refining in T adds vertices — and a uniform grid wastes
solves twice over: a row of eight cells at one temperature reports the same
tie-line eight times, and most of a fine grid lands deep inside single-phase
fields where there is nothing to find.

So a run is several passes: a coarse grid of cells, then one pass per
refinement round. A cell is refined only when a neighbour disagrees about which
phases hold it — in a binary, moving along a row from FCC to HCP *must* pass
through FCC+HCP, so a disagreement is proof that a boundary lies between them —
and it is split only **across** the axis that disagreement is on. Above or
below splits it in T, left or right in x, and only a diagonal splits it into
four. A boundary running left-to-right therefore keeps the same number of cells
along its length while doubling its temperature resolution every round, so a
round costs about what the boundaries are *long* rather than what the diagram
is *large*. Both sides of a disagreement are marked, so a large cell beside a
small one is refined too instead of being resolved from one side only.
Neighbours are found by probing just outside each edge and asking which cell
contains that point, which needs no adjacency bookkeeping and stays correct
however unevenly the cells have been split.

Each cell's single solve sits at a low-discrepancy offset inside it — a Halton
sequence, base 2 in x and base 3 in T — rather than at its centre. That is
resolution for nothing: 64 cells report 64 distinct temperatures instead of 8,
and it removes the axis-aligned aliasing a uniform grid produces. It is
deterministic, so the same run twice draws the same diagram rather than a
differently jittered one.

Refinement is also what makes the wash mean something. It is one rectangle per
cell that still owns a patch of the diagram, drawn only where a single phase
holds that cell, so the fine cells cluster along the boundaries and show where
the run spent its effort.

This is what the uniform grid could not do. A boundary exists only where two
sampled temperatures both solved that field, so at the old 10×12 default the
Al–Zn eutectic, the FCC miscibility dome and both monotectoid corners all fell
between sampled rows and drew no line at all rather than a guessed one. They
now resolve — see *Measured* below. What neither a grid nor a scatter can
promise is finding a field thinner than the spacing it started at; landing
inside one even once, though, yields its exact boundary compositions.

### Cutting a branch

A branch has to be cut where the composition jumps: across the Al–Zn
monotectoid near 550 K the FCC coexisting with HCP goes from x(Al) 0.90 to
0.37, Al-rich below and Zn-rich above, and joining those drew a chord clean
across the diagram. `cutAtJumps` compares each step with the median of the
branch's other steps, so the test scales with the temperature spacing instead
of needing a tolerance per system — which matters more now that the spacing
varies across the diagram by design.

Hovering reads out the point under the cursor — temperature, overall
composition, and every coexisting phase with its composition and fraction. It
resolves from the cell the cursor is in, which is exact and needs no catchment
radius; failing that, from the nearest drawn tie-line end within a few pixels,
which is how a point solved outside any live cell still answers.

### Not drawn yet

The invariant horizontals are missing. The 654 K eutectic and the 550 K
monotectoid are lines on a published diagram, but `boundaries()` only ever
emits an x(T) branch per assemblage, so a three-phase reaction appears as the
place where several branches happen to stop at the same temperature rather than
as the line joining them. Finding one properly means bisecting in T for the
temperature where the assemblage changes and drawing across the compositions
there, which is a few solves, not a sweep.

The liquidus and solidus also stop just short of the pure elements. A cell's
sample is held inside x = 0.002…0.998, because a pure element has nothing for a
solve to balance, so the two branches approach the melting points — 933 K for
Al, 693 K for Zn — without meeting at them.

## What it cannot do: certify

This is the honest limit of the browser build, and the app says so on every
point rather than offering a button that throws.

discopt's restricted solve goes through POUNCE, which has an emscripten build.
*Global pricing* — the proof that no phase is missing — goes through
`discopt.Model.solve()` into `discopt._rust`, a Rust extension with no
WebAssembly build. So in the browser:

| | |
|---|---|
| works | the restricted solve, and local pricing (scipy L-BFGS-B) |
| reaches | `STABILITY_SEARCHED` — every phase searched from many starts, none destabilizing |
| cannot reach | `CERTIFIED_TO_TOLERANCE`, `GLOBALLY_CERTIFIED` |

A search is not a proof. For the certificate, install the package and use
`equilibrium(..., certify="final")` or `result.certify()`.

## Three wheels

`pounce-solver` is built for emscripten and is **valid for exactly one Pyodide
build** — `worker.js` pins the version and refuses a manifest that disagrees,
because micropip would otherwise report the mismatch as "no matching
distribution", which reads like a missing package. It is not on PyPI in that
form; `build-web.sh` copies it from a POUNCE checkout (`POUNCE_REPO=`, or
`POUNCE_WHEEL=` for the file itself).

`discopt` is repacked by `pure_wheel.py`: the Rust extension is dropped, jax and
jaxlib are filtered out of `Requires-Dist` (no wasm build, never imported here),
and the tag becomes `py3-none-any`. `discopt-calphad` is pure Python already.

numpy, scipy and sympy come from Pyodide's own builds, loaded before micropip
runs so it never tries to compile a source distribution.

`build-web.sh` rebuilds two of the three wheels from this checkout, so a
committed copy of those would go stale the moment the source moved; `wheels/`
and `data/` are gitignored for that reason.

The emscripten wheel is the exception and **is** committed, in `vendor/`. It
cannot be rebuilt here or in CI — it is not on PyPI in that form — so a fresh
clone would otherwise need a POUNCE checkout to build the page at all. It is
also why `vendor/` is not `wheels/`: staging starts with `rm -rf wheels/`.

That wheel also has to be free of **relaxed SIMD**, which is easy to ship by
accident. pulp 0.22.2, reached through faer, compiled a `RelaxedSimd` backend
unconditionally; WebAssembly validates an entire module before executing any of
it, so 46 `f64x2.relaxed_madd` sitting in functions nothing could ever reach
were enough to stop the page starting in any browser without that proposal —
and it is invisible on a current one. pulp 0.22.3 puts the backend behind a
cargo feature this dependency graph leaves off, and POUNCE's `build-wheel.sh`
now inspects the module it built and refuses to stage one carrying a relaxed
opcode. If this page ever dies in micropip with a `CompileError`, check there
first.

## Published

The docs workflow stages the app and copies it next to the built site, at
[`/discopt-calphad/app/`](https://kitchingroup.cheme.cmu.edu/discopt-calphad/app/).
It is not part of the MkDocs build, so a local `mkdocs serve` will not show it —
serve `web/` directly, as above.

## Measured

Per point, on an M4 Pro — the browser costs about twice native:

| | native | Pyodide 0.28.3 |
|---|---|---|
| Al–Zn | 32 ms | 69 ms |
| Cu–Mg | 74 ms | 144 ms |

The default is 8×8 starting cells and three refinement rounds, which on Al–Zn
comes to 954 points and 66 s. That is measured rather than estimated: how far
refinement spreads depends on how long the boundaries are, which is a property
of the system, so the page states the starting cost before the run and what it
actually cost after. Pyodide itself is a ~10 MB download, cached after the
first visit.

Eight times the old 10×12 grid's 8 s, and that is the trade — the grid was
cheap and drew the wrong diagram. Against Massalski's Al–Zn the adaptive run
puts the eutectic at 650 K, x(Al) 0.118 (reference 654 K, 0.113), the FCC
composition there at 0.325 (0.330), the HCP composition at 0.031 (0.028), the
Zn-rich monotectoid corner at 548 K, 0.410 (550 K, 0.410), and closes the
miscibility dome near 622 K (624.6 K). A fourth round doubles the cost to 1944
points for the same ten boundaries and no new feature, which is why the default
stops at three; a coarser 6×8 start saves only 7% and fragments the branches,
so the cost is in the rounds, not the starting grid.

Every point is solved cold. Warm starting from the neighbouring point
(`initial_sets=`) is twice as fast and was measured to be *wrong*: on a
120-point Cu–Mg grid it disagreed with cold at 10 points and had the higher G at
9 of them, having settled into a single-phase minimum and stayed there.
Certified referees matched the cold answer. The failure is silent — a plausible
single-phase result, no error — so the speed is not worth it. Path continuation
is the principled version of that idea and does hold the right branch; see
issue #4.

Pressure is wired through to the database, but neither bundled TDB has a
pressure-dependent parameter: G at 1 GPa is identical to G at 1 atm to the last
digit.
