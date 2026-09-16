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
| `app.js` | the page: element selection, the SVG plot, phase boundaries, the hover tooltip, streaming, the detail panel |
| `elements.js` | all 118 elements with their positions, generated from ASE's data |
| `worker.js` | boots Pyodide, installs the three wheels, and runs one solve at a time |
| `calphad_app.py` | the Python driver — databases in, JSON out. Runs unchanged off-browser |
| `pure_wheel.py` | repacks discopt as a pure-Python wheel (see below) |
| `build-web.sh` | stages `wheels/` and `data/`; `--check` verifies what is staged |

Solving happens in a worker because a solve is synchronous: on the main thread
the page would freeze and Cancel could never be read. Cancel is a
`worker.terminate()`, for the same reason — nothing else can interrupt one.

## Reading the diagram

Boundaries are traced from the tie-lines, not contoured off the grid. In a
two-phase field the coexisting compositions *are* the boundary, so every
multi-phase point already carries one exact boundary composition per phase, and
joining those by temperature draws the solvus, solidus and liquidus at the
precision they were solved to. A contour could never be finer than the grid.
The faint wash behind the lines is one cell per solved point and *is* only as
fine as the grid, which is why it is faint.

Two things follow from tracing rather than contouring. A boundary exists only
where two adjacent temperatures both solved that field, so a feature the grid
straddles — the FCC miscibility gap, at the default 10×12 spacing — gets no
line at all rather than a guessed one. And a branch has to be cut where the
composition jumps: across the Al–Zn monotectoid near 550 K the FCC coexisting
with HCP goes from x(Al) 0.90 to 0.37, Al-rich below and Zn-rich above, and
joining those drew a chord clean across the diagram. `cutAtJumps` compares each
step with the median of the branch's other steps, so the test scales with the
temperature spacing instead of needing a tolerance per system.

Hovering reads out the point under the cursor — temperature, overall
composition, and every coexisting phase with its composition and fraction. It
resolves from the grid cell the cursor is in, or from a tie-line end, which for
a nearly pure phase sits outside the swept range and so owns no cell.

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

The default 10×12 grid is therefore about 8 s for Al–Zn and 17 s for Cu–Mg; the
page states the estimate before the run rather than after. Pyodide itself is a
~10 MB download, cached after the first visit.

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
