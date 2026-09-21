# The browser app

[**Open the app →**](/discopt-calphad/app/)

Pick two elements from a periodic table, get a T–x phase diagram computed in
your own tab. No server, no upload, no install: CPython is compiled to
WebAssembly by [Pyodide](https://pyodide.org), POUNCE solves each restricted
Gibbs-energy minimization, and this package generates the composition sets.
Points are drawn as they are solved.

The systems come from the package's [database registry](databases.md). The app
offers every binary in which all phases can be modeled: Al–Zn, Cu–Mg, Al–Mg,
Pb–Sn, Au–Sn, Nb–Re, Cu–O, Al–Cu, Al–Y, Cu–Y, Ti–V, Cr–Ti, Cr–V, Cr–Fe, Cr–Nb,
Fe–Nb, and a few more from COST 507. Five of them (Cr–Ti, Cr–V, Cr–Fe, Cr–Nb,
Fe–Nb) have magnetic phases.
No TDB file is served with the app. Each one is downloaded from its pinned URL
the first time a pair needs it, and checked against the registry's SHA-256. You
can also open a TDB file of your own, and solve a single point with the
composition given as a mole fraction or as amounts in moles.

## What it cannot do: certify

This is the honest limit of the browser build, and the app says so on every
point rather than offering a button that throws.

The restricted solve goes through POUNCE, which has an emscripten build.
*Global pricing* — the proof that no phase is missing — goes through
`discopt.Model.solve()` into `discopt._rust`, a Rust extension with no
WebAssembly build.

| | |
|---|---|
| works | the restricted solve, and local pricing (scipy L-BFGS-B) |
| reaches | `STABILITY_SEARCHED` — every phase searched from many starts, none destabilizing |
| cannot reach | `CERTIFIED_TO_TOLERANCE`, `GLOBALLY_CERTIFIED` |

A search is not a proof. For the certificate, install the package and use
`equilibrium(..., certify="final")` or `result.certify()`.

## Speed

Per point, measured on an M4 Pro — the browser costs about twice native:

| | native | Pyodide 0.28.3 |
|---|---|---|
| Al–Zn | 32 ms | 69 ms |
| Cu–Mg | 74 ms | 144 ms |

So the default 10×12 grid is roughly 8 s for Al–Zn and 17 s for Cu–Mg. The page
states that estimate *before* the run rather than after, and Cancel stops a run
in progress. Pyodide itself is a ~10 MB download, cached after the first visit.

## Running it locally

```bash
./web/build-web.sh                      # stage the wheels and the database registry
python3 -m http.server -d web 8000      # then open http://localhost:8000
```

See `web/README.md` for how the three wheels are assembled and why one of them
is committed.
