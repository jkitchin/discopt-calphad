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
  await pyodide.runPythonAsync('import sys; sys.path.insert(0, "/home/pyodide")');
  app = pyodide.pyimport('calphad_app');

  say('ready');
  self.postMessage({ type: 'ready' });
})().catch((err) => {
  self.postMessage({ type: 'fatal', message: String(err && err.message ? err.message : err) });
});

/** Parse a database into Python once, and remember that we did. */
async function ensureDatabase(db) {
  if (databases.has(db.name)) return;
  say(`reading ${db.file}…`);
  const text = await (await fetch(`./data/${db.file}`)).text();
  const info = JSON.parse(app.load_database(db.name, text, db.pressure ?? 101325));
  databases.add(db.name);
  self.postMessage({ type: 'database', info });
}

// Neither handler reports a status when it finishes. The page writes its own --
// the run summary, and the invitation to click a point -- and a trailing
// say('ready') from here would land after it and wipe it off the screen.
const handlers = {
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

  async diagram({ db, elements, tLo, tHi, nT, xLo, xHi, nX, pressure }) {
    await ensureDatabase(db);
    self.postMessage({ type: 'diagram-start', total: nT * nX });
    // Points leave Python through this callback, one JSON string each, as they
    // are solved. `emit` is passed in rather than imported so the Python side
    // has no JavaScript in it.
    const emit = (json) => self.postMessage({ type: 'diagram-point', point: JSON.parse(json) });
    const summary = JSON.parse(
      app.diagram(db.name, elements, tLo, tHi, nT, xLo, xHi, nX, pressure ?? 101325, emit),
    );
    self.postMessage({ type: 'diagram-done', summary });
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
