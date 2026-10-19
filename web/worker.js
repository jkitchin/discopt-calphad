// The worker: CPython in WebAssembly, the CALPHAD package installed into it,
// and one equilibrium solve at a time.
//
// Why a worker at all -- a solve is synchronous and a grid is tens of seconds
// of them. On the main thread the page would freeze solid and Cancel would
// never be read. Here the page stays live, each point is posted as it is
// found, and Cancel is a `terminate()`.
//
// Three wheels, all local:
//
//   pounce-solver   the interior-point solver, built for emscripten. Not on
//                   PyPI in that form -- it is staged from the POUNCE repo by
//                   build-web.sh, and is valid for exactly one Pyodide build.
//   discopt         repacked without its Rust extension, which has no wasm build.
//                   Solving never reaches it; certifying does, which is the one
//                   thing this page cannot do (see pure_wheel.py).
//   discopt-calphad this package, pure Python already.

// Pinned so a Pyodide release cannot change what this page runs. Keep in step
// with build-web.sh, which checks it. `?pyodide=<base-url>` serves it yourself.
const PYODIDE_VERSION = '0.28.3';
const params = new URLSearchParams(self.location.search);
const PYODIDE_URL = params.get('pyodide') || `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

const say = (text, detail) => self.postMessage({ type: 'status', text, detail });

let pyodide = null;
let app = null;
const databases = new Set();

const ready = (async () => {
  say(`loading Python (Pyodide ${PYODIDE_VERSION}, ~10 MB, cached after the first visit)…`);
  // `pyodide.mjs`, not `pyodide.js`: this is a module worker, where
  // `importScripts` does not exist.
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  const res = await fetch('./wheels/manifest.json');
  if (!res.ok) {
    throw new Error('the wheels are not staged here — run web/build-web.sh, then reload.');
  }
  const manifest = await res.json();
  // An emscripten wheel is valid only for the Pyodide build it was compiled
  // against. Say that plainly instead of letting micropip report it as "no
  // matching distribution", which reads like a missing package.
  if (manifest.pyodide_version !== PYODIDE_VERSION) {
    throw new Error(
      `the staged pounce-solver wheel was built for Pyodide ${manifest.pyodide_version}, but this ` +
        `page runs ${PYODIDE_VERSION} — re-stage with web/build-web.sh`,
    );
  }

  // numpy, scipy and sympy come from Pyodide's own builds. Loading them first
  // means micropip has them already and never tries to resolve a source
  // distribution it cannot compile.
  say('loading numpy, scipy, sympy…');
  await pyodide.loadPackage(['micropip', 'numpy', 'scipy', 'sympy']);

  say('installing the solver and the CALPHAD package…');
  const micropip = pyodide.pyimport('micropip');
  const { pounce, discopt, discopt_calphad: calphad } = manifest.wheels;
  await micropip.install([`./wheels/${pounce}`, `./wheels/${discopt}`, `./wheels/${calphad}`]);

  // The driver is fetched rather than bundled into this file so it stays
  // readable Python that runs unchanged off-browser.
  pyodide.FS.writeFile('/home/pyodide/calphad_app.py', await (await fetch('./calphad_app.py')).text());
  // discopt 0.9 lowers models through an "arena tape" built by discopt._rust,
  // which this build does not have, and it imports it unguarded: left on, every
  // solve raises ModuleNotFoundError. Off, it uses the pure-Python tape, which
  // is the path this build was made for. Read at call time, so setting it here,
  // before the first solve, is enough.
  await pyodide.runPythonAsync(
    'import os, sys; os.environ["DISCOPT_ARENA_TAPE"] = "0"; sys.path.insert(0, "/home/pyodide")',
  );
  app = pyodide.pyimport('calphad_app');

  say('ready');
  self.postMessage({ type: 'ready' });
})().catch((err) => {
  self.postMessage({ type: 'fatal', message: explain(err) });
});

/**
 * Turn a boot failure into something a visitor can act on.
 *
 * The one that actually reaches people is the solver's wasm refusing to
 * compile. The build pulls in pulp (through faer), and pulp 0.22.2 compiles
 * its wasm `RelaxedSimd` backend unconditionally: 46 `f64x2.relaxed_madd` in
 * three functions that can never run, because nothing ever sets the flag that
 * would select them. Dead code is still fatal here -- WebAssembly validates
 * the whole module up front, so a browser without the relaxed-SIMD proposal
 * rejects all of it. pulp 0.22.3 puts that backend behind a cargo feature
 * which this dependency graph leaves off, so a rebuilt wheel is clean; until
 * a visitor gets one, say this plainly rather than letting Pyodide's micropip
 * CompileError traceback be the answer.
 */
function explain(err) {
  const text = String((err && (err.message || err.toString())) || err);
  if (/relaxed[ _-]?simd/i.test(text) || (/CompileError/.test(text) && /doesn't parse/.test(text))) {
    return (
      'this browser cannot run the solver: its WebAssembly build uses relaxed SIMD, ' +
      'which this browser does not support. That needs Chrome or Edge 114+, Firefox 120+, ' +
      'or Safari 18.4+ (where it may still be behind a flag).'
    );
  }
  if (/Could not load dynamic lib|LinkError/i.test(text)) {
    return `the solver failed to load in this browser — ${text}`;
  }
  return text;
}

/**
 * Parse a database into Python once, and remember that we did.
 *
 * A registry database is downloaded from its pinned URL: raw.githubusercontent
 * sends `Access-Control-Allow-Origin: *`, so this site serves no TDB files. The
 * download is checked against the registry's SHA-256 before Python sees it.
 * A file the visitor opened arrives as text on the request instead, because it
 * exists nowhere else. Either way the text reaches Python as a string, so no
 * database ever has to exist as a file inside the Pyodide sandbox.
 */
async function ensureDatabase(db) {
  if (databases.has(db.name)) return;
  let text = db.text;
  if (text == null) {
    say(`downloading ${db.file}…`);
    const res = await fetch(db.url);
    if (!res.ok) throw new Error(`could not download ${db.file}: HTTP ${res.status} from ${db.url}`);
    const bytes = await res.arrayBuffer();
    await checkHash(db, bytes);
    // TDB files are Latin-1, as the Python reader assumes.
    text = new TextDecoder('latin1').decode(bytes);
  }
  say(`reading ${db.file}…`);
  const info = JSON.parse(app.load_database(db.name, text, db.pressure ?? 101325));
  databases.add(db.name);
  self.postMessage({ type: 'database', info });
}

/**
 * Refuse a download whose SHA-256 is not the registry's.
 *
 * `crypto.subtle` exists only in a secure context: https, or localhost. On
 * plain http elsewhere the check cannot run. The file is then used without
 * the check, and the page says so rather than hiding it.
 */
async function checkHash(db, bytes) {
  if (!db.sha256) return;
  if (!self.crypto?.subtle) {
    say(`reading ${db.file} — not hash-checked (this page is not served over https)`);
    return;
  }
  const digest = await self.crypto.subtle.digest('SHA-256', bytes);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
  if (hex !== db.sha256) {
    throw new Error(`${db.file} from ${db.url} does not match the registry's SHA-256 (got ${hex}); not used`);
  }
}

// Neither handler reports a status when it finishes. The page writes its own --
// the run summary, and the invitation to click a point -- and a trailing
// say('ready') from here would land after it and wipe it off the screen.
const handlers = {
  // Parse a file the visitor opened, and nothing else. What it covers is the
  // answer, and only Python can say: the elements and phases come back on the
  // `database` message, which is what re-enables the periodic table.
  async database({ db }) {
    await ensureDatabase(db);
  },

  // Where each pure element changes phase. This is not a sweep: with one
  // component the stable phase is whichever has the lowest molar G, so it is a
  // root find on temperature over the models this system already built, and it
  // costs no equilibria. The page asks for it before the first pass and uses
  // the answers as the endpoints of the boundaries it draws.
  async transitions({ db, elements, tLo, tHi, pressure }) {
    await ensureDatabase(db);
    const ends = JSON.parse(
      app.pure_transitions(db.name, elements, pressure ?? 101325, tLo, tHi),
    );
    self.postMessage({ type: 'transitions', ends });
  },

  async point({ db, elements, T, x, b, pressure, certify }) {
    await ensureDatabase(db);
    // calphad_app.point() can certify -- it is an ordinary module that also runs
    // natively -- but global pricing imports discopt._rust, which is not in this
    // build. Refuse here, where the reason can be given, rather than letting a
    // ModuleNotFoundError traceback arrive as the answer.
    if (certify) {
      throw new Error(
        "certification prices every phase globally through discopt's Rust solver, which has " +
          'no WebAssembly build — run it natively',
      );
    }
    const result = JSON.parse(app.point(db.name, elements, T, x, b, pressure ?? 101325, false));
    self.postMessage({ type: 'point', point: result });
  },

  // One pass of a run: solve exactly these points, in this order. The page
  // decides where they are -- it starts from a coarse grid and then subdivides
  // only the cells a phase boundary runs through, which is a shape this side
  // could not reconstruct from a few numbers. So a pass is a list of
  // coordinates, each labelled, and the answers carry the label back.
  async points({ db, elements, coords, pressure }) {
    await ensureDatabase(db);
    self.postMessage({ type: 'pass-start', total: coords.length });
    // Points leave Python through this callback, one JSON string each, as they
    // are solved. `emit` is passed in rather than imported so the Python side
    // has no JavaScript in it.
    const emit = (json) => self.postMessage({ type: 'pass-point', point: JSON.parse(json) });
    const summary = JSON.parse(
      app.points(db.name, elements, JSON.stringify(coords), pressure ?? 101325, emit),
    );
    self.postMessage({ type: 'pass-done', summary });
  },
};

self.onmessage = async (event) => {
  const { type, ...payload } = event.data;
  const handler = handlers[type];
  if (!handler) return;
  try {
    await ready;
    if (!pyodide) return;
    await handler(payload);
  } catch (err) {
    // A Python exception arrives with its traceback in `message`. Report it as
    // a failed request, not as a dead page: the worker is still usable.
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};
