// Page logic: choose two elements, ask the worker for equilibria, draw them as
// they arrive. The worker owns Python and the solver; this side is a periodic
// table, a plot, and a table of numbers.

import { ELEMENTS } from './elements.js';

/**
 * The databases this page can use. The first two are what build-web.sh stages,
 * one binary each; opening a file appends to this list, which is why it is not
 * frozen.
 *
 * `ms` is the measured cost of one point under Pyodide (M4 Pro, Pyodide 0.28.3)
 * — about twice the native rate. It is only used to warn before a long run.
 * `tLo`/`tHi` are a useful temperature window for that system. A file the
 * visitor opened can supply none of the three: it has never been timed here,
 * and it covers many systems, each with its own window. So it carries them as
 * null and the page says what it does not know rather than inventing it.
 */
const DATABASES = [
  { name: 'Al-Zn', file: 'alzn_mey.tdb', elements: ['AL', 'ZN'], tLo: 400, tHi: 950, ms: 69 },
  { name: 'Cu-Mg', file: 'cumg_nims.tdb', elements: ['CU', 'MG'], tLo: 400, tHi: 1200, ms: 144 },
];

// `elements` is the ordered pair to solve -- the axis labels and Python both
// want exactly two -- while `covers` is everything the file can do. For a
// bundled binary they are the same; for a picked file they are not, which is
// why the two are separate fields rather than one.
for (const db of DATABASES) db.covers = new Set(db.elements);

/** Element symbols some loaded database covers. Grows when a file is opened. */
const WITH_DATA = new Set(DATABASES.flatMap((db) => db.elements));
const $ = (id) => document.getElementById(id);

// Names and parse errors out of a TDB are file contents, and they are written
// into innerHTML, so they are escaped on the way.
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

const state = {
  chosen: [],          // element symbols, in the order picked
  mode: 'diagram',     // 'diagram' | 'point'
  composition: 'fraction', // 'fraction' | 'amounts'
  points: [],          // every point received for the current run
  phaseColors: new Map(),
  running: false,
  selected: null,      // the point whose detail is shown
  hover: null,         // the point under the cursor, described by the tooltip
  invariants: [],      // three-phase horizontals: { T, xs, phases }
  criticals: [],       // where a miscibility gap closes: { phase, T, x, ... }
  pureEnds: [],        // pure-element transitions: { element, x, T, below, above }
  epoch: 0,            // bumped per run, so work cached per run is dropped
};

/**
 * The run in progress. A run is several passes: the starting grid, one pass per
 * refinement round, then a few narrow passes bisecting for the invariant
 * temperatures. Kept apart from `state` because none of it outlives the run --
 * `state.points` and `state.invariants` are the result, this is the bookkeeping.
 */
const run = {
  cells: new Map(),    // id -> cell, for every cell this run has asked for
  round: 0,            // pass number; 1 is the starting grid
  rounds: 0,           // how many refinement passes were asked for
  expected: 0,         // points in the current pass
  arrived: 0,          // points received in the current pass
  started: 0,
  seconds: 0,
  failed: 0,
  stopped: false,
  // 'grid' while the cells are being refined, then 'invariants' while the
  // three-phase temperatures are bisected. The two phases send the same kind of
  // pass and are told apart by this, because what to do when a pass lands
  // differs entirely.
  phase: 'grid',
  brackets: [],        // the invariants being bisected, see invariants section
  steps: 0,            // bisection passes so far, as a stopping guard
};

// Distinguishable at small sizes and in both themes; assigned in order of first
// appearance so a phase keeps its colour for a whole run.
const PALETTE = ['#1b6b8a', '#b4530a', '#1f7a4d', '#8a3d8a', '#a32020', '#6a6a1f', '#2a4b9b', '#8a5a2a'];

function colorFor(phase) {
  if (!state.phaseColors.has(phase)) {
    state.phaseColors.set(phase, PALETTE[state.phaseColors.size % PALETTE.length]);
  }
  return state.phaseColors.get(phase);
}

/**
 * The database to use for the chosen pair, if any covers it.
 *
 * A bundled file covers exactly one binary; a file the visitor opened covers
 * every binary among its elements, so "covers" is containment, not equality. A
 * bundled exact match wins where there is one, because that assessment is what
 * this page has been checked against, and an opened file answers for the rest.
 */
function currentDatabase() {
  if (state.chosen.length !== 2) return null;
  const want = [...state.chosen].sort().join('-');
  const exact = DATABASES.find((db) => !db.text && [...db.elements].sort().join('-') === want);
  if (exact) return exact;
  const covering = DATABASES.find((db) => state.chosen.every((el) => db.covers.has(el)));
  // The pair in play comes from the selection rather than the file, because a
  // multicomponent file has no pair of its own. Everything downstream -- the
  // axis labels, the worker message, Python's `elements` -- wants two, in order.
  return covering ? { ...covering, elements: [...state.chosen] } : null;
}

/** The file being parsed, until Python reports what is in it. */
let picked = null;

/**
 * Add a database the visitor opened, and turn on the elements it covers.
 *
 * The text stays here rather than being handed over once. Cancel is a
 * `worker.terminate()`, so every database parsed inside the worker dies with
 * it and the page has to be able to say it again — which is why the text rides
 * on the database object the run messages already carry.
 */
function registerDatabase({ name, file, text }, info) {
  const covers = new Set(info.elements);
  const db = {
    name,
    file,
    text,
    covers,
    elements: [...covers],
    label: `${file} · ${info.elements.length} elements`,
    tLo: null,
    tHi: null,
    ms: null,
  };
  const at = DATABASES.findIndex((d) => d.name === name);
  if (at >= 0) DATABASES[at] = db;
  else DATABASES.push(db);
  for (const el of covers) WITH_DATA.add(el);
  refreshTable();
}

// --- periodic table ---------------------------------------------------------

function buildTable() {
  const table = $('ptable');
  let lastRow = 0;
  for (const el of ELEMENTS.slice().sort((a, b) => a.row - b.row || a.col - b.col)) {
    // The f-block is drawn detached, as it is on a wall chart.
    if (el.row === 9 && lastRow !== 9) {
      const gap = document.createElement('div');
      gap.className = 'fblock';
      table.append(gap);
    }
    lastRow = el.row;
    const cell = document.createElement('button');
    cell.className = 'el';
    cell.style.gridRow = String(el.row === 9 ? 9 : el.row === 10 ? 10 : el.row);
    cell.style.gridColumn = String(el.col);
    cell.innerHTML = `<span class="z">${el.z}</span>${el.sym}`;
    cell.dataset.sym = el.sym.toUpperCase();
    cell.dataset.name = el.name;
    // The listener goes on every cell rather than only the ones with data,
    // because which elements have data changes when a file is opened -- and a
    // disabled button fires no clicks anyway.
    cell.addEventListener('click', () => toggleElement(el.sym.toUpperCase()));
    table.append(cell);
  }
  refreshTable();
}

/**
 * Enable the elements some loaded database covers.
 *
 * Called again whenever a database is added, because opening a file is what
 * turns most of this table on: the two bundled files reach four elements
 * between them, and one multicomponent file reaches tens.
 */
function refreshTable() {
  for (const cell of document.querySelectorAll('.el')) {
    const has = WITH_DATA.has(cell.dataset.sym);
    cell.classList.toggle('has-data', has);
    cell.disabled = !has;
    cell.title = has
      ? `${cell.dataset.name} — click to select`
      : `${cell.dataset.name} — no loaded database covers this element`;
  }
}

function toggleElement(sym) {
  const at = state.chosen.indexOf(sym);
  if (at >= 0) state.chosen.splice(at, 1);
  // Two elements make a binary; a third replaces the oldest.
  else if (state.chosen.push(sym) > 2) state.chosen.shift();
  renderSelection();
}

function renderSelection() {
  for (const cell of document.querySelectorAll('.el')) {
    cell.classList.toggle('chosen', state.chosen.includes(cell.dataset.sym));
  }
  const db = currentDatabase();
  const pairs = DATABASES.map((d) => `<span class="pill">${esc(d.label || d.elements.join('–'))}</span>`).join('');
  if (db) {
    const info = state.databaseInfo?.[db.name];
    // A binary file's phases fit on the line. A multicomponent file has
    // hundreds, and the ones that matter are the ones this pair can form, which
    // is what the run itself reports.
    const phases = info
      ? ` · ${info.phases.length > 12 ? `${info.phases.length} phases` : `phases: ${info.phases.join(', ')}`}`
      : '';
    $('systems').innerHTML =
      `<strong>${db.elements.join('–')}</strong> from <code>${esc(db.file)}</code>${phases}${skippedNote(info)}`;
    // A picked file spans many systems, each with its own useful temperature
    // window, so it offers no default and the fields are left as they are.
    if (db.tLo != null) $('t-lo').value = db.tLo;
    if (db.tHi != null) $('t-hi').value = db.tHi;
    setXLabels(db);
  } else if (state.chosen.length === 2) {
    $('systems').innerHTML =
      `No loaded database covers ${state.chosen.join('–')}. Loaded: ${pairs} — open a TDB file below for more.`;
  } else {
    $('systems').innerHTML = `Pick two elements. Loaded: ${pairs}`;
  }
  updateEstimate();
  $('run').disabled = !db || state.running || !state.ready;
}

/**
 * What a file cost to parse, when it cost anything.
 *
 * Reported rather than swallowed. A command that will not parse is usually a
 * PARAMETER, and a dropped parameter shows up as wrong energies rather than as
 * an error, so the page has to say it lost one. COST507 loses exactly two this
 * way, to a `!` typed where a `1` belongs.
 */
function skippedNote(info) {
  if (!info?.skipped?.length) return '';
  const n = info.skipped.length;
  const first = info.skipped[0];
  return `<br><span class="warn-text">${n} command${n === 1 ? '' : 's'} in this file would not parse and `
    + `${n === 1 ? 'was' : 'were'} skipped; everything else loaded. First: <code>${esc(first.command)}</code> — `
    + `${esc(first.reason)}. A skipped parameter shows up as wrong energies, not as an error.</span>`;
}

/** Say what a run will cost before it is started, not after. */
function updateEstimate() {
  const db = currentDatabase();
  if (!db || state.mode !== 'diagram') {
    $('estimate').textContent = '';
    return;
  }
  const n = Number($('n-t').value) * Number($('n-x').value);
  const rounds = Number($('n-refine').value);
  const seconds = (n * db.ms) / 1000;
  // An opened file has never been timed here, and a made-up number would read
  // exactly like a measured one. Say what is known: a multicomponent database
  // carries far more phases than a bundled binary, and the run reports its real
  // cost when it finishes.
  const cost = db.ms
    ? `${n} points to start, about ${seconds < 1 ? '1' : seconds.toFixed(0)} s — `
      + `${db.elements.join('–')} measures ${db.ms} ms per point in the browser. `
    : `${n} points to start, at a cost per point this page has not measured — `
      + 'a multicomponent file offers many more phases per solve than the bundled binaries do at '
      + '69–144 ms, so expect several times that. ';
  $('estimate').textContent =
    cost
    + (rounds
      ? `Then ${rounds} refinement round${rounds === 1 ? '' : 's'}, each subdividing only the `
        + 'cells a phase boundary runs through, and only across the boundary — so the added cost '
        + 'follows how long the boundaries are, not how large the diagram is. How many points '
        + 'that comes to depends on the system, so the run reports what it actually cost. '
      : 'Refinement is off, so the boundaries are spaced in temperature no more finely than the '
        + 'starting grid. ')
    + 'Points are drawn as they are solved, and Cancel stops the run.';
}

function setXLabels(db) {
  const [a, b] = db.elements;
  $('pt-x-label').childNodes[0].nodeValue = `x(${a})`;
  $('pt-a-label').childNodes[0].nodeValue = `moles ${a}`;
  $('pt-b-label').childNodes[0].nodeValue = `moles ${b}`;
}

// --- plot -------------------------------------------------------------------

const PAD = { left: 62, right: 18, top: 16, bottom: 46 };
// The frame the plot was last drawn against, so the cursor can be mapped back
// to a temperature and a composition. See drawPlot.
let frame = null;
const W = 760;
const H = 460;
const SVG_NS = 'http://www.w3.org/2000/svg';

function node(name, attrs, text) {
  const n = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (text !== undefined) n.textContent = text;
  return n;
}

/**
 * One entry per phase actually present, with duplicates merged.
 *
 * The solver can return the same phase twice at the same composition -- two
 * composition sets that converged onto each other. They are one phase on the
 * diagram, so drawing both puts two dots in one place and traces the boundary
 * through it twice. The amounts still add, though: each set carries its own
 * share of the material, so merging has to sum them or the fractions stop
 * summing to 1 (Al-Zn at 400 K splits HCP 0.491/0.491 -- drop one and the
 * table reads 49%, not 98%).
 */
function uniqueSets(p) {
  const out = [];
  for (const s of p.sets) {
    const same = out.find((o) => o.phase === s.phase && Math.abs(o.x[0] - s.x[0]) < 1e-6);
    if (same) {
      same.amount += s.amount;
      same.fraction += s.fraction;
    } else {
      out.push({ ...s, x: [...s.x] });
    }
  }
  return out;
}

/**
 * Cut a branch wherever one step in composition is an outlier against the
 * median of its other steps -- a discontinuity rather than a boundary.
 *
 * Judging a step against the branch's own steps, instead of against a fixed
 * tolerance, is what makes this hold on any grid: the legitimate steps grow
 * with the temperature spacing and the test grows with them, while a
 * discontinuity does not. Each step is compared with the median of the others
 * so that the jump cannot raise the threshold that is meant to catch it. Two
 * points carry no evidence either way and are left alone.
 */
function cutAtJumps(pts) {
  if (pts.length < 3) return [pts];
  const steps = pts.slice(1).map((q, i) => Math.abs(q.x - pts[i].x));
  const pieces = [];
  let start = 0;
  steps.forEach((step, i) => {
    const others = steps.filter((_, k) => k !== i).sort((a, b) => a - b);
    const mid = others.length / 2;
    const median = others.length % 2
      ? others[Math.floor(mid)]
      : (others[mid - 1] + others[mid]) / 2;
    // The floor keeps a branch that barely moves at all -- a solvus down the
    // edge of the diagram -- from being cut at a step that is large only in
    // proportion to its neighbours.
    if (step > Math.max(0.12, 4 * median)) {
      pieces.push(pts.slice(start, i + 1));
      start = i + 1;
    }
  });
  pieces.push(pts.slice(start));
  return pieces;
}

/**
 * Phase boundaries, traced from the tie-lines rather than contoured off the grid.
 *
 * In a two-phase field the coexisting compositions *are* the boundary: each
 * tie-line ends on the edge of a single-phase region. So every multi-phase
 * point already carries one exact boundary composition per phase, and joining
 * them up temperature by temperature draws the solvus, solidus and liquidus
 * directly. Contouring the region map instead could never be finer than the
 * grid, which is what makes the filled cells look blocky.
 *
 * Every point in a field reports the same tie-line, so the compositions are
 * gathered per temperature and deduplicated first: the eleven points across
 * the Al-Zn two-phase field at 400 K carry one boundary composition per phase
 * between them, not eleven. Branches then grow a temperature at a time, each
 * extended by its nearest composition in the next row, which keeps the two
 * sides of a miscibility gap apart without having to label them.
 *
 * Keying by the assemblage as well as the phase stops LIQUID's edge against
 * FCC being joined to its edge against HCP across a eutectic, but that alone
 * is not enough: one assemblage can hold two boundaries that never meet.
 * Across the Al-Zn monotectoid near 550 K the FCC coexisting with HCP jumps
 * from x(Al) 0.90 to 0.37 -- Al-rich FCC below it, Zn-rich above -- and
 * joining those drew a chord clean across the diagram. Hence cutAtJumps.
 */
function boundaries() {
  const fields = new Map();
  for (const p of state.points) {
    if (p.error) continue;
    const sets = uniqueSets(p);
    if (sets.length < 2) continue;
    const assemblage = sets.map((s) => s.phase).sort().join('+');
    for (const s of sets) {
      const key = `${assemblage}|${s.phase}`;
      if (!fields.has(key)) fields.set(key, new Map());
      const byT = fields.get(key);
      // Rounded, because refinement bisects: two cells that reach the same
      // temperature by different arithmetic must still group together.
      const T = Math.round(p.T * 1e4) / 1e4;
      const xs = byT.get(T) || [];
      if (!xs.some((x) => Math.abs(x - s.x[0]) < 1e-6)) xs.push(s.x[0]);
      byT.set(T, xs);
    }
  }

  // The pure-element ends, put in as exact vertices rather than approached by
  // sampling. A pure element's transition is a crossing of two molar energies --
  // no tie-line, and no solve at all -- so it is computed (see
  // `pure_transitions`), and it is the one point on a boundary that is known
  // exactly rather than to the resolution of a grid. Attached only to a field
  // that was actually sampled: the Al-Zn liquidus reaches Al's melting point
  // because the FCC+LIQUID field runs into it, while an end belonging to a
  // field this run never saw is not drawn from one vertex alone.
  for (const end of state.pureEnds) {
    const assemblage = [end.below, end.above].sort().join('+');
    for (const phase of [end.below, end.above]) {
      const byT = fields.get(`${assemblage}|${phase}`);
      if (!byT) continue;
      const T = Math.round(end.T * 1e4) / 1e4;
      const xs = byT.get(T) || [];
      if (!xs.some((x) => Math.abs(x - end.x) < 1e-6)) xs.push(end.x);
      byT.set(T, xs);
    }
  }

  const out = [];
  for (const [key, byT] of fields) {
    const phase = key.slice(key.indexOf('|') + 1);
    const assemblage = key.slice(0, key.indexOf('|'));
    const temps = [...byT.keys()].sort((a, b) => a - b);
    const done = [];
    let open = [];
    for (const T of temps) {
      const xs = [...byT.get(T)].sort((a, b) => a - b);
      // Nearest first, so a branch follows its own side of a gap.
      const pairs = [];
      open.forEach((c, ci) => xs.forEach((x, xi) => {
        pairs.push({ ci, xi, d: Math.abs(x - c.pts[c.pts.length - 1].x) });
      }));
      pairs.sort((a, b) => a.d - b.d);
      const tookBranch = new Set();
      const tookX = new Set();
      for (const pr of pairs) {
        if (tookBranch.has(pr.ci) || tookX.has(pr.xi)) continue;
        tookBranch.add(pr.ci);
        tookX.add(pr.xi);
        open[pr.ci].pts.push({ x: xs[pr.xi], T });
      }
      // A branch with nothing to join in this row has ended. A composition
      // with no branch to join starts one, which is a field opening up.
      const next = [];
      open.forEach((c, ci) => (tookBranch.has(ci) ? next : done).push(c));
      xs.forEach((x, xi) => { if (!tookX.has(xi)) next.push({ pts: [{ x, T }] }); });
      open = next;
    }
    for (const branch of [...done, ...open]) {
      for (const piece of cutAtJumps(branch.pts)) {
        // One point is not a line.
        if (piece.length > 1) out.push({ phase, assemblage, pts: piece });
      }
    }
  }
  closeAtCriticals(out);
  return out;
}

/**
 * Join the two branches of a closing miscibility gap at its critical point.
 *
 * The two sides of a dome are ends of one tie-line, so they are traced as two
 * branches and each stops at the last temperature where the two compositions
 * were still apart. That leaves the arch open at the top however hard the
 * temperature is refined, because the gap closes to nothing: the lens thins
 * faster than bisection narrows. The apex is the one vertex the two branches
 * share, so once it has been found it is appended to both.
 */
function closeAtCriticals(branches) {
  for (const cp of state.criticals) {
    for (const b of branches) {
      // Only the field of two sets of one phase closes at a critical point. The
      // same phase's other branches -- its solidus, its solvus against another
      // phase -- belong to different fields and must not be pulled to the apex.
      if (b.assemblage !== cp.assemblage || b.pts.length < 2) continue;
      const first = b.pts[0];
      const last = b.pts[b.pts.length - 1];
      const atEnd = Math.abs(last.T - cp.T) <= Math.abs(first.T - cp.T);
      const stop = atEnd ? last : first;
      // A guard rather than a search. The branch ends at the highest row that
      // still resolved two compositions, and the apex was fitted from that row
      // and the ones below it, so the two are a known distance apart -- `gap`
      // carries it. A branch further off than that belongs to some other field.
      if (Math.abs(stop.T - cp.T) > cp.gap || Math.abs(stop.x - cp.x) > 0.3) continue;
      if (atEnd) b.pts.push({ x: cp.x, T: cp.T });
      else b.pts.unshift({ x: cp.x, T: cp.T });
    }
  }
}

// --- adaptive grid ----------------------------------------------------------
//
// A uniform T-x sweep is a poor way to find a phase boundary. The boundary
// composition at a given temperature is a property of the temperature alone:
// every overall composition inside a two-phase field sits on the same tie-line
// and so reports the same pair of coexisting compositions. Refining in x
// therefore adds no boundary detail at all -- all eleven points across the
// Al-Zn field at 400 K carry one answer between them. What adds detail is
// temperature, where each new value is one more vertex on every branch. And a
// grid fine enough in T to place the Al-Zn eutectic spends almost all of its
// solves deep inside single-phase fields, where nothing is happening.
//
// So sample adaptively. The diagram is covered by cells, each holding one solve
// at its centre, and a cell is subdivided when its assemblage differs from a
// neighbour's -- which is exactly the statement that a boundary runs between
// them. The part that makes it affordable is that a cell is split only *across*
// the boundary: a disagreement with the neighbour above or below splits it in
// temperature, one to the left or right splits it in composition, and only a
// diagonal splits it into four. A boundary running left to right therefore
// keeps the same number of cells along its length while doubling its
// temperature resolution every round, so a round costs about what the
// boundaries are long, not what the diagram is large.

// Cells span the full width, so the wash covers the plot and no inset is needed
// to keep off a pure element -- a cell centre is never one. Deep refinement
// against x = 0 or 1 does get very close, though, so the sample is held just
// inside while the cell keeps its true extent.
const X_LO = 0;
const X_HI = 1;
// A ceiling on a run, so a large starting grid and seven rounds cannot quietly
// turn into a quarter of an hour. Reaching it stops refinement and says so.
const POINT_BUDGET = 2500;

// A cell's solve sits at a low-discrepancy offset inside it rather than at its
// centre, which is the one change here that buys resolution for nothing. On a
// uniform grid every cell in a row shares a temperature, so a row of eight
// cells reports one temperature eight times over -- and because a boundary
// composition depends only on temperature, seven of those solves tell the
// boundaries nothing the first has not already said. Offset them and the same
// eight solves report eight temperatures, each contributing its own vertex.
//
// The offsets come from the Halton sequence rather than a random generator, so
// they are spread instead of clustered and the same run twice draws the same
// diagram. This is quasi-random sampling kept inside a cell decomposition: the
// cells are what make "my neighbour disagrees" a cheap, exact test for a
// boundary running between two solves, which a scattered set could only answer
// by triangulating.
let seq = 0;

/** Radical inverse of n in `base`: van der Corput, the 1-D Halton sequence. */
function radicalInverse(n, base) {
  let out = 0;
  let f = 1 / base;
  for (let k = n; k > 0; k = Math.floor(k / base)) {
    out += (k % base) * f;
    f /= base;
  }
  return out;
}

function cellAt(x0, x1, t0, t1, level) {
  seq += 1;
  // Held well away from the edges, so a sample never lands on the line between
  // two cells, where which one it belongs to would be a rounding accident.
  const u = 0.15 + 0.7 * radicalInverse(seq, 2);
  const v = 0.15 + 0.7 * radicalInverse(seq, 3);
  return {
    x0, x1, t0, t1, level,
    id: `${level}:${x0.toFixed(7)}:${t0.toFixed(4)}`,
    xc: (x0 + x1) / 2,                                  // geometric centre
    // A pure element has nothing for a solve to balance, and refinement pressed
    // against x = 0 or 1 gets very close, so the sample is held just inside.
    x: Math.min(0.998, Math.max(0.002, x0 + u * (x1 - x0))),
    T: t0 + v * (t1 - t0),
  };
}

/** The starting grid: uniform cells, one solve at the centre of each. */
function initialCells() {
  const tLo = Number($('t-lo').value);
  const tHi = Number($('t-hi').value);
  const nT = Math.max(1, Number($('n-t').value));
  const nX = Math.max(1, Number($('n-x').value));
  const out = [];
  for (let j = 0; j < nT; j += 1) {
    for (let i = 0; i < nX; i += 1) {
      out.push(cellAt(
        X_LO + ((X_HI - X_LO) * i) / nX, X_LO + ((X_HI - X_LO) * (i + 1)) / nX,
        tLo + ((tHi - tLo) * j) / nT, tLo + ((tHi - tLo) * (j + 1)) / nT,
        0,
      ));
    }
  }
  return out;
}

/** What phases hold a point, as one comparable label. */
function assemblageOf(p) {
  return uniqueSets(p).map((s) => s.phase).sort().join('+');
}

/** The cells that still own a patch of the diagram, with their solved point. */
function liveCells() {
  return state.points.filter((p) => p.cell && !p.superseded && !p.error);
}

// How many buckets across and up `cellIndex` lays over the diagram.
const INDEX_N = 48;

/**
 * Point location over the live cells: which cell holds (x, T)?
 *
 * Three things here ask that question -- the refinement's neighbour probes, the
 * bracket hunts, and the wash -- and each asks it several thousand times for one
 * diagram. Scanning the cells for every probe makes all of that quadratic, which
 * is slow enough to feel when a hover redraws the plot. The cells tile the
 * diagram at deliberately uneven sizes, so this buckets them on a fixed lattice
 * instead: a cell joins every bucket its extent overlaps, and a probe then
 * searches one bucket rather than the whole run.
 *
 * Built per caller from the cells it is asking about, because the callers
 * disagree about which cells count -- the wash excludes cells it has already
 * rejected -- and an index is much cheaper to build than to be wrong about.
 */
function cellIndex(live) {
  if (!live.length) return { at: () => null };
  let tLo = Infinity;
  let tHi = -Infinity;
  for (const p of live) {
    if (p.cell.t0 < tLo) tLo = p.cell.t0;
    if (p.cell.t1 > tHi) tHi = p.cell.t1;
  }
  const span = tHi - tLo || 1;
  const col = (x) => Math.min(INDEX_N - 1, Math.max(0, Math.floor(x * INDEX_N)));
  const row = (T) => Math.min(INDEX_N - 1, Math.max(0, Math.floor(((T - tLo) / span) * INDEX_N)));
  const buckets = new Map();
  for (const p of live) {
    const c = p.cell;
    for (let j = row(c.t0); j <= row(c.t1); j += 1) {
      for (let i = col(c.x0); i <= col(c.x1); i += 1) {
        const k = j * INDEX_N + i;
        const held = buckets.get(k);
        if (held) held.push(p);
        else buckets.set(k, [p]);
      }
    }
  }
  return {
    at(x, T) {
      const held = buckets.get(row(T) * INDEX_N + col(x));
      if (!held) return null;
      // A point outside every cell is off the diagram, and the callers all read
      // that as "no neighbour" rather than as a boundary.
      return held.find((p) => x >= p.cell.x0 && x < p.cell.x1
        && T >= p.cell.t0 && T < p.cell.t1) || null;
    },
  };
}

/**
 * Which cells a boundary runs through, and across which axis, as
 * point -> { x, t }.
 *
 * Neighbours are found by probing just outside the middle of each edge and
 * asking which cell contains that point. That needs no adjacency bookkeeping
 * and stays correct however unevenly the cells have been subdivided. Both sides
 * of a disagreement are marked, so a large cell beside a small one is refined
 * too, rather than leaving the boundary resolved from one side only.
 *
 * A cell that failed to solve is left out entirely: refining around it would
 * spend the budget chasing a solver failure, which more points cannot fix.
 */
function crossedCells() {
  const live = liveCells();
  const index = cellIndex(live);
  const holding = (x, T) => index.at(x, T);
  const marks = new Map();
  const mark = (p, axis) => {
    const m = marks.get(p) || { x: false, t: false };
    m[axis] = true;
    marks.set(p, m);
  };
  for (const p of live) {
    const c = p.cell;
    const mine = assemblageOf(p);
    const e = Math.min(c.x1 - c.x0, c.t1 - c.t0) * 1e-4;
    for (const [probe, axis] of [
      [holding(c.x0 - e, c.T), 'x'], [holding(c.x1 + e, c.T), 'x'],
      [holding(c.xc, c.t0 - e), 't'], [holding(c.xc, c.t1 + e), 't'],
    ]) {
      // No neighbour means the edge of the diagram, which is not a boundary.
      if (!probe || probe === p || assemblageOf(probe) === mine) continue;
      mark(p, axis);
      mark(probe, axis);
    }
  }
  return marks;
}

/** How many points a refinement of these marks would cost. */
function refinementCost(marks) {
  let n = 0;
  for (const [, m] of marks) n += (m.x ? 2 : 1) * (m.t ? 2 : 1);
  return n;
}

/**
 * Split each marked cell across the axis whose boundary crosses it.
 *
 * The parent's solve is real data at a real temperature, so it keeps its vertex
 * on the boundary and is never thrown away; it only stops owning a patch of the
 * diagram, which its children now cover.
 */
function refinementCells(marks) {
  const out = [];
  for (const [p, m] of marks) {
    const c = p.cell;
    const xm = (c.x0 + c.x1) / 2;
    const tm = (c.t0 + c.t1) / 2;
    const xs = m.x ? [[c.x0, xm], [xm, c.x1]] : [[c.x0, c.x1]];
    const ts = m.t ? [[c.t0, tm], [tm, c.t1]] : [[c.t0, c.t1]];
    for (const [x0, x1] of xs) {
      for (const [t0, t1] of ts) out.push(cellAt(x0, x1, t0, t1, c.level + 1));
    }
    p.superseded = true;
  }
  return out;
}

// --- invariants -------------------------------------------------------------
//
// The horizontals a published diagram draws: the eutectic, the monotectoid --
// a three-phase reaction, which in a binary at fixed pressure happens at one
// temperature only, so it is a line rather than a field.
//
// `boundaries()` cannot produce one. It emits an x(T) branch per assemblage, so
// a reaction shows up only as several branches that happen to stop near the same
// temperature, and "near" is as coarse as the cells are tall.
//
// They are found instead, and the signature is simple. Crossing an ordinary
// phase boundary, one side is single-phase and the other is not: liquid above,
// liquid+FCC below. Crossing an invariant, *both* sides are multi-phase and the
// assemblages disagree -- L+HCP above the Al-Zn eutectic and FCC+HCP below,
// FCC+FCC above the monotectoid and FCC+HCP below. In a binary two different
// two-phase fields cannot touch, so something invariant lies between them.
//
// Note the monotectoid: both of its upper composition sets are FCC_A1, one
// Al-rich and one Zn-rich. Counting distinct phase *names* would see two phases
// and miss the reaction entirely, so this counts composition sets.
//
// Once bracketed, the temperature is bisected -- about ten solves each, and all
// the brackets step together in one pass, so the whole thing costs a second.

/** Within this many kelvin, a bracket is as narrow as it needs to be. */
const INVARIANT_TOL = 0.05;
/** A guard on the loop, not a target: 12 halvings take 550 K under the tol. */
const INVARIANT_STEPS = 12;

/**
 * Temperature brackets that contain a three-phase reaction, one per reaction.
 *
 * Every x between the outer two compositions brackets the same invariant, so
 * the pairs are grouped by which assemblages meet and only the narrowest
 * bracket of each group is kept -- the refinement has usually already made one
 * of them much tighter than the rest.
 */
function invariantBrackets() {
  const live = liveCells();
  const index = cellIndex(live);
  const holding = (x, T) => index.at(x, T);
  const best = new Map();
  for (const p of live) {
    const c = p.cell;
    const below = uniqueSets(p);
    if (below.length < 2) continue;
    const above = holding(c.xc, c.t1 + (c.t1 - c.t0) * 1e-4);
    if (!above || above.cell.t0 < c.t1 - 1e-9) continue;
    const upper = uniqueSets(above);
    if (upper.length < 2) continue;
    const loKey = assemblageOf(p);
    const hiKey = assemblageOf(above);
    if (loKey === hiKey) continue;
    // The two assemblages have to share a phase. At a real reaction one phase
    // is on both sides of it; two assemblages with nothing in common are two
    // unrelated fields with something skipped between them, and bisecting
    // would invent a line across compositions that never coexist.
    const names = new Set(below.map((s) => s.phase));
    if (!upper.some((s) => names.has(s.phase))) continue;
    // Solved temperatures, not cell edges: those are where the assemblages are
    // actually known, so they are what genuinely brackets the reaction.
    const key = [loKey, hiKey].sort().join(' / ');
    // The bracketing solves are carried along, not looked up again later: they
    // are what knows the compositions on each side, and matching a point back by
    // its floating-point temperature is a needless way to lose it.
    const found = { x: c.xc, tLo: p.T, tHi: above.T, loKey, hiKey, loPoint: p, hiPoint: above };
    const prev = best.get(key);
    if (!prev || found.tHi - found.tLo < prev.tHi - prev.tLo) best.set(key, found);
  }
  return [...best.values()];
}

/**
 * The critical points: where a miscibility gap closes, found by extrapolating
 * the gap rather than by searching for its top.
 *
 * This is the other feature `boundaries()` cannot draw. A gap closes at a
 * critical point -- the top of the Al-Zn FCC dome -- and approaching it the two
 * coexisting compositions merge, so the tie-line shortens to nothing and the
 * two branches stop short with the arch open between them.
 *
 * The first version of this bracketed the apex and bisected it, like an
 * invariant, and that was the wrong instrument rather than a fixable bug.
 * Bisection asks "one phase here, or two?", and near the apex that question
 * stops having a reliable answer, because the two minima it distinguishes are
 * converging on each other. It put the Al-Zn apex at 623.3 K, 0.01 K above a
 * row whose own gap was still 0.085 wide -- a shape no closing dome has --
 * because one solve 2.4 K below the top answered "one phase" where there were
 * two.
 *
 * So the apex is fitted. A gap closes parabolically: its half-width w obeys
 * T = Tc - c w^2, which is both why refinement never reaches the top and
 * exactly how to extrapolate to it. Least squares over the rows that are
 * comfortably resolved gives Tc where w vanishes, costs no solves at all --
 * those rows are already on the screen -- and is far steadier than the search
 * it replaces: on Al-Zn the top eight rows give 625.75 K with a 4 mK residual,
 * and windows from four to twenty rows agree within 0.06 K.
 */
const CRITICAL_ROWS = 8;

function criticalPoints() {
  // Rows of a closing gap: one phase, two compositions. Every solve is eligible,
  // refinement points and bisection probes alike -- a probe is an ordinary
  // equilibrium that happens to have been asked for by a bracket.
  const byPhase = new Map();
  for (const p of state.points) {
    if (p.error) continue;
    const sets = uniqueSets(p);
    if (sets.length !== 2 || sets[0].phase !== sets[1].phase) continue;
    const xs = [sets[0].x[0], sets[1].x[0]].sort((a, b) => a - b);
    if (!byPhase.has(sets[0].phase)) byPhase.set(sets[0].phase, []);
    byPhase.get(sets[0].phase).push({
      T: p.T,
      w: (xs[1] - xs[0]) / 2,
      xc: (xs[0] + xs[1]) / 2,
    });
  }
  const top = Number($('t-hi').value);
  const out = [];
  for (const [phase, all] of byPhase) {
    // The highest rows, and only those. The parabola describes the gap near its
    // top; rows far down the flanks bend away from it, and they bend one way, so
    // a window that reaches too far reads the apex too high -- Al-Zn drifts from
    // 625.8 K to 628.7 K as the window slides 20 K down. Where the window sits
    // matters much more than how wide it is.
    all.sort((a, b) => b.T - a.T);
    const rows = all.slice(0, CRITICAL_ROWS);
    if (rows.length < 4) continue;
    const fit = fitToZeroWidth(rows);
    if (!fit) continue;
    // Extrapolating upwards is the point of this; extrapolating downwards, or
    // past the ceiling, means these rows are not the top of a closing dome --
    // a gap cut off by the window, or a field that never closes at all.
    if (!(fit.T > rows[0].T) || fit.T > top) continue;
    out.push({
      phase,
      assemblage: `${phase}+${phase}`,
      T: fit.T,
      x: fit.x,
      // What the apex rests on: how far apart the last two coexisting
      // compositions still were, how far past them the fit reached, and how well
      // the rows it used actually lay on a parabola. Together they say how much
      // of this point is measured and how much is extrapolation.
      width: 2 * rows[0].w,
      reach: fit.T - rows[0].T,
      rms: fit.rms,
      // How far a branch may stop short of the apex and still be joined to it.
      // The fit reaches past the highest solved row by construction, so the
      // branches stop at least that far short and the allowance has to cover it.
      gap: Math.max(5, 3 * (fit.T - rows[0].T)),
    });
  }
  return out;
}

/**
 * Least squares of T and of the gap's centre against w^2, extrapolated to w = 0.
 *
 * The centre is fitted the same way rather than averaged: a dome is not quite
 * symmetric, so the midpoint of the two compositions drifts as the gap narrows
 * -- 0.6469 to 0.6492 down the Al-Zn rows -- and what is wanted is the value it
 * is drifting towards, not the average of the places it has been.
 */
function fitToZeroWidth(rows) {
  const n = rows.length;
  const X = rows.map((r) => r.w * r.w);
  const mx = X.reduce((u, v) => u + v, 0) / n;
  let den = 0;
  for (const v of X) den += (v - mx) ** 2;
  // Every row the same width apart: nothing to extrapolate along.
  if (!(den > 0)) return null;
  const against = (values) => {
    const my = values.reduce((u, v) => u + v, 0) / n;
    let num = 0;
    for (let i = 0; i < n; i++) num += (X[i] - mx) * (values[i] - my);
    const m = num / den;
    return { m, at0: my - m * mx };
  };
  const T = against(rows.map((r) => r.T));
  // The gap has to widen as it cools. A fit that says otherwise is not a dome.
  if (!(T.m < 0)) return null;
  let rss = 0;
  for (let i = 0; i < n; i++) rss += (rows[i].T - (T.at0 + T.m * X[i])) ** 2;
  return {
    T: T.at0,
    x: against(rows.map((r) => r.xc)).at0,
    rms: Math.sqrt(rss / n),
  };
}

/** The midpoint solve each unconverged bracket wants next. */
function bisectionCells() {
  const out = [];
  for (const b of run.brackets) {
    // A bracket that has landed on its reaction is done, however wide it still
    // is: three phases in one solve is the answer bisection was looking for.
    if (b.hit || b.tHi - b.tLo <= INVARIANT_TOL) continue;
    const T = (b.tLo + b.tHi) / 2;
    // A bisection point owns no patch of the diagram -- it is a probe at one
    // temperature, not a cell -- so it carries no extent and gets no wash. The
    // label ties the answer back to its bracket.
    b.probe = { id: `inv${out.length}:${T.toFixed(5)}`, T, x: b.x };
    out.push(b.probe);
  }
  return out;
}

/**
 * Narrow each bracket by the midpoint that just came back.
 *
 * Three phases in one solve is the answer, not a surprise: the reaction is
 * invariant, so a solve that lands on it reports every phase taking part, at
 * exactly the temperature and the compositions where they do. Keep it and stop.
 * Bisection only exists to get close enough for that to happen.
 *
 * Otherwise the midpoint agrees with one side or the other and that half is
 * kept. An assemblage matching neither, with fewer than three phases, means
 * something else lies in between -- a field thinner than the bracket -- and the
 * bracket is abandoned rather than narrowed onto a reaction it cannot name.
 */
function applyBisection(points) {
  for (const b of run.brackets) {
    if (!b.probe) continue;
    const got = points.get(b.probe.id);
    b.probe = null;
    if (!got || got.error) {
      b.dead = true;
      continue;
    }
    if (uniqueSets(got).length >= 3) {
      b.hit = got;
      continue;
    }
    const key = assemblageOf(got);
    if (key === b.loKey) {
      b.tLo = got.T;
      b.loPoint = got;
    } else if (key === b.hiKey) {
      b.tHi = got.T;
      b.hiPoint = got;
    } else {
      b.dead = true;
    }
  }
  run.brackets = run.brackets.filter((b) => !b.dead);
}

/**
 * The finished horizontals: temperature, the compositions it spans, and the
 * phases that meet there.
 *
 * A bracket that landed on the reaction has all three phases in one solve, and
 * that is used as it stands. Otherwise the span is taken from both sides of the
 * converged bracket together, because each side is a two-phase solve that knows
 * only its own two compositions while the reaction needs all three.
 */
function invariantLines() {
  const out = [];
  // A landed probe describes a reaction better than a merely narrow bracket, so
  // those are considered first and the duplicate check below keeps them.
  const brackets = [...run.brackets].sort((a, b) => (b.hit ? 1 : 0) - (a.hit ? 1 : 0));
  for (const b of brackets) {
    const lo = b.hit || b.loPoint;
    const hi = b.hit || b.hiPoint;
    if (!lo || !hi) continue;
    const sets = b.hit ? uniqueSets(b.hit) : [...uniqueSets(lo), ...uniqueSets(hi)];
    const xs = [];
    for (const s of sets) {
      // Merge the phase that sits on both sides of the reaction: it is one
      // composition, reported twice, a hair either side of one temperature.
      if (!xs.some((v) => Math.abs(v - s.x[0]) < 2e-3)) xs.push(s.x[0]);
    }
    xs.sort((a, b2) => a - b2);
    // Two compositions is not a reaction -- it is one two-phase field whose
    // assemblage label changed for some other reason. Three is the binary case;
    // more means the bracket still holds something unresolved.
    if (xs.length < 3) continue;
    // A landed probe knows the temperature exactly -- the three phases it
    // returned coexist at that one temperature and nowhere else -- so the
    // bracket midpoint is only the fallback for a bracket that never landed.
    const T = b.hit ? b.hit.T : (b.tLo + b.tHi) / 2;
    // Two brackets can converge on one reaction, and the Al-Zn eutectic does: it
    // is bracketed both as L+FCC over FCC+HCP and as L+HCP over FCC+HCP, two
    // different assemblage pairs that `invariantBrackets` is right to keep, which
    // then bisect onto the same temperature and the same three compositions. That
    // is one reaction, and drawing it twice doubles the ink and the label.
    const twice = out.some((o) => Math.abs(o.T - T) < 1 && o.xs.length === xs.length
      && o.xs.every((v, i) => Math.abs(v - xs[i]) < 2e-3));
    if (twice) continue;
    out.push({
      T,
      xs,
      phases: [...new Set(sets.map((s) => s.phase))].sort(),
    });
  }
  return out;
}

/** Data ranges for the current run, so axes do not jump as points stream in. */
function ranges() {
  if (state.mode === 'point') {
    const ts = state.points.map((p) => p.T);
    // Nothing solved yet: frame the temperature that is about to be solved, so
    // the axes are drawn rather than collapsing on an empty extent.
    if (!ts.length) {
      const t = Number($('pt-t').value) || 500;
      return { xLo: 0, xHi: 1, tLo: t - 100, tHi: t + 100 };
    }
    const pad = Math.max(50, (Math.max(...ts) - Math.min(...ts)) * 0.1);
    return { xLo: 0, xHi: 1, tLo: Math.min(...ts) - pad, tHi: Math.max(...ts) + pad };
  }
  return {
    xLo: 0,
    xHi: 1,
    tLo: Number($('t-lo').value),
    tHi: Number($('t-hi').value),
  };
}

/**
 * The wash: which single phase holds each region, drawn as regions.
 *
 * This was a rectangle per cell for a long time, and the shape of a cell is the
 * wrong shape for the answer. A cell is washed by the phase found at its
 * sample, and a cell with a boundary running through it is single-phase only
 * *at* that sample -- the rest of it belongs to the other side. Painted anyway,
 * its edges were stair-steps crossing the very lines the tie-lines were traced
 * to draw. Washing only cells whose neighbours all agree fixed the crossing but
 * not the shape: it left every field edged by a ragged white seam a cell wide,
 * and the cells against the Al-Zn liquidus are still an eighth of the axis
 * across, so the seam was the most visible thing on the diagram.
 *
 * The boundaries themselves are the right edge to fill to. At a fixed
 * temperature the branch curves cut the composition axis into intervals, and
 * each interval is one field -- 2 crossings below the Al-Zn monotectoid, 4
 * between it and the top of the dome, 2 again above where the dome has closed.
 * So this scans temperature, interpolates every branch to get that row of
 * crossings, and pairs consecutive rows into bands: one quad per band per
 * interval, its sides slanting with the two boundaries it lies between. Nothing
 * steps, because the edge of the wash *is* the boundary.
 *
 * Stitching a field's bands into one field-shaped polygon is the obvious next
 * move, and it was tried and taken out again. It means deciding which interval
 * of a row continues which interval of the row below, and that reading comes
 * from the cells, so it wobbles: a track that closes and reopens on a single bad
 * row leaves a seam across the middle of a field, and two tracks open at once
 * paint it twice. On Al-Zn that showed as a blue block beside the HCP curve
 * ignoring the curve, and as pale stripes ending in mid-air under the liquidus.
 * The quads are kept apart and the *drawing* joins them instead -- one path per
 * phase, composited once however many quads it holds.
 *
 * Which phase fills an interval comes from the crossings that bound it, not
 * from the cells. A crossing is a tie-line end -- the composition of a named
 * phase, whose conjugate is the other end of the same tie-line -- and the
 * single-phase field lies on the side away from that conjugate, and is that
 * phase. So the curves that say where a field is also say what is in it, at any
 * width. Asking a cell cannot do that for a field narrower than a cell: nearly
 * every cell overlapping the Al-Zn HCP strip has its sample outside it, in the
 * two-phase region, so the strip went unwashed below 611 K while its own
 * boundary ran on to 400 K, and the liquid wedge at the Zn corner and the FCC
 * field under the monotectoid were cut off flat the same way.
 *
 * The lenses are still unpainted, now for a reason rather than for want of an
 * answer: both crossings that bound one face away from it -- the liquidus says
 * single-phase liquid to its left, the solidus single-phase FCC to its right --
 * so nothing names the interval between them. Two crossings that disagree paint
 * nothing either.
 *
 * One strip has no curve to name it. Above the highest melting point every row
 * is liquid and no branch crosses one to say so, which left a white band across
 * the top of Al-Zn above 934 K. There, and only there, a cell is asked: it is
 * as far from a boundary as a cell ever gets, and above the last curve a binary
 * holds nothing but liquid in any case.
 *
 * The foot of the diagram is not the mirror of that and must not be treated as
 * one. Curves stop at the lowest row that was solved, not at the edge of a
 * field, so what lies below them is usually still two-phase. Asking a cell
 * there painted a band of FCC clear across the bottom of Al-Zn near 415 K,
 * where the answer is FCC+HCP. Below the curves nothing is claimed. An interval holding two phases is left unpainted, as before --
 * those are the lenses, and a wash across one would claim a single phase where
 * the whole point is that two coexist.
 *
 * Memoised per run, per point count and per frame, because hovering redraws the
 * plot and this is far too much work to repeat for a redraw that changed
 * nothing but which dot is circled.
 */
const WASH_BANDS = 120;

let washMemo = { key: '', fills: [] };

function washFills(brs, tLo, tHi) {
  const key = `${state.epoch}:${state.points.length}:${tLo}:${tHi}`;
  if (washMemo.key === key) return washMemo.fills;
  const index = cellIndex(liveCells());
  const eps = (tHi - tLo) * 1e-4;
  // An even sweep, plus a hair either side of every temperature where the
  // structure changes: an invariant, a closing dome, the end of any branch. A
  // band that straddles such a temperature has a different number of intervals
  // at its two ends and cannot be filled, so those bands are made thin enough
  // to vanish rather than left to show as a stripe across the diagram.
  const ts = new Set();
  for (let i = 0; i <= WASH_BANDS; i += 1) ts.add(tLo + ((tHi - tLo) * i) / WASH_BANDS);
  const breaks = [
    ...state.invariants.map((inv) => inv.T),
    ...state.criticals.map((cp) => cp.T),
  ];
  for (const b of brs) breaks.push(b.pts[0].T, b.pts[b.pts.length - 1].T);
  for (const T of breaks) {
    if (T > tLo + eps && T < tHi - eps) { ts.add(T - eps); ts.add(T + eps); }
  }
  const scan = [...ts].filter((T) => T >= tLo && T <= tHi).sort((a, b) => a - b);

  /**
   * Where every branch crosses one temperature -- the composition, the phase it
   * is the composition of, and the assemblage whose tie-line it ends.
   */
  const crossAt = (T) => {
    const xs = [];
    for (const b of brs) {
      for (let k = 1; k < b.pts.length; k += 1) {
        const p = b.pts[k - 1];
        const q = b.pts[k];
        if (q.T === p.T) continue;
        if (T < Math.min(p.T, q.T) || T > Math.max(p.T, q.T)) continue;
        const x = p.x + ((T - p.T) / (q.T - p.T)) * (q.x - p.x);
        if (!xs.some((v) => Math.abs(v.x - x) < 1e-9)) {
          xs.push({ x, phase: b.phase, assemblage: b.assemblage });
        }
      }
    }
    xs.sort((a, b2) => a.x - b2.x);
    return xs;
  };

  /** The composition axis cut at those crossings, the two ends included. */
  const edgesOf = (cr) => [0, ...cr.map((c) => c.x), 1];

  /**
   * Which way the single-phase field lies from each crossing: away from the
   * other end of its own tie-line. -1 is to the left, 1 to the right, 0 for a
   * crossing whose conjugate is not in this row and which therefore says
   * nothing. The nearest same-assemblage crossing is the conjugate, which is
   * what keeps the two sides of a dome -- one assemblage, one phase, two
   * crossings -- pointing outwards rather than at each other.
   */
  const sidesOf = (cr) => cr.map((c, i) => {
    let conj = null;
    for (let j = 0; j < cr.length; j += 1) {
      if (j === i || cr[j].assemblage !== c.assemblage) continue;
      if (!conj || Math.abs(cr[j].x - c.x) < Math.abs(conj.x - c.x)) conj = cr[j];
    }
    if (!conj) return 0;
    return conj.x > c.x ? -1 : 1;
  });

  // A reaction temperature is the one place this cannot be asked. Every branch
  // stops there, so there are no crossings to fill between, and the index
  // answers with whatever single phase the reaction produced -- which would
  // paint that phase clear across the diagram. The bands holding one are
  // eps-thin by construction, so they are left unpainted instead.
  const reactions = state.invariants.map((inv) => inv.T);

  // The highest temperature any boundary reaches. Above it there is no curve to
  // name a field, and none is needed -- see the note on the cell fallback.
  let topOfCurves = -Infinity;
  for (const b of brs) {
    for (const p of b.pts) if (p.T > topOfCurves) topOfCurves = p.T;
  }

  // One quad per band per interval, accumulated by phase. They are not stitched
  // into field-shaped polygons: doing that means tracking which interval
  // continues which, and a track that closes and reopens on a wobble leaves a
  // seam across the middle of a field. Quads cannot wobble, and the drawing
  // makes the joins between them invisible -- see `drawPlot`.
  const byPhase = new Map();
  let lower = crossAt(scan[0]);
  for (let i = 1; i < scan.length; i += 1) {
    const t0 = scan[i - 1];
    const t1 = scan[i];
    const upper = crossAt(t1);
    if (!reactions.some((T) => T > t0 && T < t1)) {
      const tm = (t0 + t1) / 2;
      const mid = crossAt(tm);
      // Slant a quad's sides to follow the boundaries only where the band has
      // the same intervals at both ends. Where a field opens or closes inside
      // it, that end is squared off at the band's middle instead -- such a band
      // is eps-thin, so the square end is under a pixel tall.
      const a = edgesOf(lower.length === mid.length ? lower : mid);
      const b = edgesOf(upper.length === mid.length ? upper : mid);
      const m = edgesOf(mid);
      const side = sidesOf(mid);
      for (let k = 1; k < m.length; k += 1) {
        if (m[k] - m[k - 1] <= 0) continue;
        // An interval is named by whichever of its two bounding crossings faces
        // it. One is the usual case; two that agree is a field bounded by the
        // curves of two different assemblages, as the Al-Zn FCC field is by its
        // solidus and the dome. Two that disagree, or none at all, paints
        // nothing -- the lenses land here, both of their crossings facing away.
        const left = k >= 2 ? mid[k - 2] : null;
        const right = k - 1 < mid.length ? mid[k - 1] : null;
        const votes = [];
        if (left && side[k - 2] === 1) votes.push(left.phase);
        if (right && side[k - 1] === -1) votes.push(right.phase);
        if (votes.some((v) => v !== votes[0])) continue;
        let phase = votes[0];
        // Whatever names an interval has to close it too: the far edge must be
        // the same phase's curve, or the axis. A stoichiometric phase is why.
        // Its field has no width, so both of its crossings sit at one
        // composition, and each then points away from its own two-phase region
        // -- straight into the neighbouring one. On Cu-Mg that washed the whole
        // CuMg2 + Cu2Mg field as a single phase. Two single-phase fields never
        // touch, so an edge carrying a different phase is proof the interval
        // between them is not single-phase.
        if (phase && ((left && left.phase !== phase) || (right && right.phase !== phase))) continue;
        if (!phase && tm > topOfCurves) {
          // Above every curve, and only there. No branch reaches this row to
          // name what it holds, and above the last boundary a binary holds
          // nothing but liquid, so the cell is both reliable here -- as far
          // from a boundary as a cell ever gets -- and merely being asked for
          // the name of something already known.
          const hold = index.at((m[k - 1] + m[k]) / 2, tm);
          // Merged, not `p.phases`: a lone phase returned as two composition
          // sets is still one phase, and would otherwise get no wash.
          const sets = hold ? uniqueSets(hold) : [];
          if (sets.length !== 1) continue;
          phase = sets[0].phase;
        }
        // Nothing named this interval and nothing was entitled to guess. That is
        // the two-phase background, and the honest wash for it is none: without
        // this the unnamed intervals collect under a single `undefined` key and
        // paint the whole field between the curves in one colour.
        if (!phase) continue;
        if (!byPhase.has(phase)) byPhase.set(phase, []);
        byPhase.get(phase).push([
          { x: a[k - 1], T: t0 }, { x: a[k], T: t0 },
          { x: b[k], T: t1 }, { x: b[k - 1], T: t1 },
        ]);
      }
    }
    lower = upper;
  }
  const fills = [...byPhase].map(([phase, quads]) => ({ phase, quads }));
  washMemo = { key, fills };
  return fills;
}

function drawPlot() {
  const svg = $('plot');
  svg.replaceChildren();
  const db = currentDatabase();
  if (!db) {
    // An empty frame reads as broken. Say what the space is waiting for.
    svg.append(node('text', {
      x: W / 2, y: H / 2, 'text-anchor': 'middle', class: 'axis-title',
    }, 'Pick two elements above, then Compute.'));
    return;
  }
  const { xLo, xHi, tLo, tHi } = ranges();
  const sx = (x) => PAD.left + ((x - xLo) / (xHi - xLo)) * (W - PAD.left - PAD.right);
  const sy = (t) => H - PAD.bottom - ((t - tLo) / (tHi - tLo || 1)) * (H - PAD.top - PAD.bottom);
  // Remember the frame the cursor will be read against. Recomputing it from the
  // inputs on each pointermove would describe a diagram that is no longer on
  // screen, as soon as someone edits a field after a run.
  frame = { xLo, xHi, tLo, tHi };

  for (let i = 0; i <= 10; i += 1) {
    const x = xLo + ((xHi - xLo) * i) / 10;
    svg.append(node('line', {
      x1: sx(x), x2: sx(x), y1: PAD.top, y2: H - PAD.bottom,
      stroke: 'currentColor', 'stroke-opacity': 0.07,
    }));
    svg.append(node('text', { x: sx(x), y: H - PAD.bottom + 16, 'text-anchor': 'middle' }, x.toFixed(1)));
  }
  for (let i = 0; i <= 5; i += 1) {
    const t = tLo + ((tHi - tLo) * i) / 5;
    svg.append(node('line', {
      x1: PAD.left, x2: W - PAD.right, y1: sy(t), y2: sy(t),
      stroke: 'currentColor', 'stroke-opacity': 0.07,
    }));
    svg.append(node('text', { x: PAD.left - 8, y: sy(t) + 4, 'text-anchor': 'end' }, t.toFixed(0)));
  }
  svg.append(node('rect', {
    x: PAD.left, y: PAD.top, width: W - PAD.left - PAD.right, height: H - PAD.top - PAD.bottom,
    class: 'frame',
  }));
  svg.append(node('text', {
    x: (PAD.left + W - PAD.right) / 2, y: H - 10, 'text-anchor': 'middle', class: 'axis-title',
  }, `x(${db.elements[0]})  —  mole fraction`));
  const yTitle = node('text', {
    x: 14, y: (PAD.top + H - PAD.bottom) / 2, 'text-anchor': 'middle', class: 'axis-title',
    transform: `rotate(-90 14 ${(PAD.top + H - PAD.bottom) / 2})`,
  }, 'T (K)');
  svg.append(yTitle);

  // Three layers, back to front: a wash marking which single phase holds each
  // region, the boundary lines traced from the tie-lines, then the tie-line ends
  // themselves. The branches are computed once and shared, because the wash is
  // filled to them: it is one piece of geometry drawn twice, and asking for it
  // twice would let the fill and the line it is filled to drift apart.
  const brs = boundaries();
  // One path per phase rather than one shape per piece of wash. Two translucent
  // shapes that abut show an antialiasing hairline along the edge they share,
  // and two that overlap paint the colour twice; inside a single path both go
  // away, because the rasterizer takes one coverage for the whole path -- its
  // internal edges are interior to it, and its overlaps fill once.
  for (const f of washFills(brs, tLo, tHi)) {
    svg.append(node('path', {
      d: f.quads.map((q) => `M${q.map((p) => `${sx(p.x)},${sy(p.T)}`).join('L')}Z`).join(''),
      fill: colorFor(f.phase), 'fill-opacity': 0.11, stroke: 'none',
    }));
  }

  // Straight segments between solved temperatures, not a spline: a smooth curve
  // through these points would invent structure between them that was never
  // computed.
  for (const b of brs) {
    svg.append(node('polyline', {
      points: b.pts.map((q) => `${sx(q.x)},${sy(q.T)}`).join(' '),
      fill: 'none', stroke: colorFor(b.phase), 'stroke-width': 1.7,
      'stroke-opacity': 0.9, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }));
  }

  // The critical points: an open circle, not a dot, because nothing was solved
  // there and nothing can be. The apex is where the gap closes, so it is the one
  // place on the dome no equilibrium reports two compositions -- it is fitted
  // from the rows below it, and drawn as the join it is: the one vertex both
  // sides of the dome share.
  for (const cp of state.criticals) {
    const mark = node('circle', {
      cx: sx(cp.x), cy: sy(cp.T), r: 3, fill: 'var(--bg)',
      stroke: colorFor(cp.phase), 'stroke-width': 1.4,
    });
    mark.append(node('title', {},
      `${cp.phase} critical point near ${cp.T.toFixed(1)} K, x = ${cp.x.toFixed(3)}`
      + ` — fitted from the gap, last resolved ${cp.width.toFixed(3)} wide`
      + ` ${cp.reach.toFixed(1)} K below this, to within ${cp.rms.toFixed(3)} K`));
    svg.append(mark);
  }

  // The invariant horizontals, over the boundaries rather than under them: the
  // branches run into this line and stop, so it is what should read as
  // continuous where they meet it. Drawn in the ink colour, not a phase colour,
  // because three phases meet on it and none of them owns it.
  for (const inv of state.invariants) {
    const y = sy(inv.T);
    const rule = node('line', {
      x1: sx(inv.xs[0]), x2: sx(inv.xs[inv.xs.length - 1]), y1: y, y2: y,
      stroke: 'var(--ink)', 'stroke-width': 1.5, 'stroke-opacity': 0.75,
      'stroke-linecap': 'round',
    });
    // The rule is the one feature on the plot that is not a dot the tooltip can
    // read, and nothing about it says which phases meet on it. A title names the
    // reaction on hover.
    rule.append(node('title', {}, `${inv.phases.join(' + ')} at ${inv.T.toFixed(1)} K`));
    svg.append(rule);
    // A tick at each composition the reaction actually involves. The line spans
    // the outer two, but the middle one -- the phase being consumed -- is the
    // reaction, and without a mark there is nothing to say where it sits.
    for (const x of inv.xs) {
      svg.append(node('line', {
        x1: sx(x), x2: sx(x), y1: y - 3.5, y2: y + 3.5,
        stroke: 'var(--ink)', 'stroke-width': 1.5, 'stroke-opacity': 0.75,
      }));
    }
    // The right end of an invariant is exactly where a branch runs into it, so
    // the label lands on that branch -- 654 K read as "65_ K" over the Al-Zn
    // liquidus. A background-coloured stroke painted under the glyphs keeps it
    // legible without moving it off the end it belongs to.
    svg.append(node('text', {
      x: sx(inv.xs[inv.xs.length - 1]) + 5, y: y - 4, 'font-size': 10,
      'fill-opacity': 0.75, stroke: 'var(--bg)', 'stroke-width': 3,
      'paint-order': 'stroke', 'stroke-linejoin': 'round',
    }, `${inv.T.toFixed(0)} K`));
  }

  for (const p of state.points) {
    if (p.error) {
      svg.append(node('path', {
        d: `M${sx(p.x) - 3},${sy(p.T) - 3} l6,6 M${sx(p.x) + 3},${sy(p.T) - 3} l-6,6`,
        stroke: 'var(--bad)', 'stroke-width': 1.2,
      }));
      continue;
    }
    const chosen = state.selected === p;
    const sets = uniqueSets(p);
    for (const s of sets) {
      // No <title>, and no click listener either. The hover tooltip covers the
      // whole plot and answers immediately, where a native tooltip fires only
      // over the dot itself and only after a delay; the click is resolved the
      // same way, which is what lets these be small.
      svg.append(node('circle', {
        // Every dot is drawn at the same size. A tie-line end used to be more
        // than twice this radius, and at refinement densities that read as a
        // chain of beads thicker than the line it was tracing -- when it is the
        // line that carries the shape. So a boundary dot is set apart by being
        // darker rather than larger, and a single-phase sample stays faint: it
        // is one point inside a field whose wash already says what holds it,
        // and its job is to show where the solves actually went.
        cx: sx(s.x[0]), cy: sy(p.T), r: 1.4,
        fill: colorFor(s.phase), 'fill-opacity': sets.length > 1 ? 1 : 0.26,
        stroke: chosen ? 'var(--ink)' : 'none', 'stroke-width': chosen ? 1.5 : 0,
      }));
    }
  }

  // The hovered point, marked on top of everything. For a multi-phase point the
  // dashed segment joining its coexisting compositions is the tie-line itself --
  // the lever rule, drawn.
  if (state.hover && !state.hover.error && state.points.includes(state.hover)) {
    const p = state.hover;
    const sets = uniqueSets(p);
    // These are drawn last and so sit above the dots, where SVG hit-testing
    // would hand them the click meant for the dot underneath. They are
    // decoration; nothing should be able to hit them.
    const mark = node('g', { 'pointer-events': 'none' });
    if (sets.length > 1) {
      const xs = sets.map((s) => s.x[0]);
      mark.append(node('line', {
        x1: sx(Math.min(...xs)), x2: sx(Math.max(...xs)), y1: sy(p.T), y2: sy(p.T),
        stroke: 'var(--ink)', 'stroke-width': 1, 'stroke-opacity': 0.5, 'stroke-dasharray': '4 3',
      }));
    }
    mark.append(node('circle', {
      cx: sx(p.x), cy: sy(p.T), r: 3.2, fill: 'none',
      stroke: 'var(--ink)', 'stroke-width': 1.1, 'stroke-opacity': 0.75,
    }));
    for (const s of sets) {
      mark.append(node('circle', {
        cx: sx(s.x[0]), cy: sy(p.T), r: 5.4, fill: 'none',
        stroke: colorFor(s.phase), 'stroke-width': 1.6,
      }));
    }
    svg.append(mark);
  }
  renderLegend();
}

function renderLegend() {
  const seen = [...state.phaseColors.keys()].sort();
  $('legend').innerHTML = seen
    .map((p) => `<span><i class="swatch" style="background:${colorFor(p)}"></i>${p}</span>`)
    .join('') + (seen.length
      ? ' <span>· lines = phase boundaries, dots = tie-line ends, wash = single phase'
        + ' · hover anywhere for the composition there</span>'
      : '');
}

// --- detail panel -----------------------------------------------------------

function showDetail(p) {
  const db = currentDatabase();
  const rows = uniqueSets(p)
    .map((s) => `<tr>
        <td>${s.phase}</td>
        <td class="num">${s.x.map((v) => v.toFixed(4)).join(' / ')}</td>
        <td class="num">${(100 * s.fraction).toFixed(2)}%</td>
        <td class="num">${s.amount.toFixed(6)}</td>
      </tr>`)
    .join('');
  const level = p.certificate.level;
  const cls = level === 'GLOBALLY_CERTIFIED' || level === 'CERTIFIED_TO_TOLERANCE' ? 'good'
    : level === 'FAILED' ? '' : 'partial';
  const explain = {
    STABILITY_SEARCHED: 'Local pricing searched every phase from many starting points and found none that would lower G. That is a search, not a proof — and it is as far as the browser can go.',
    CERTIFIED_TO_TOLERANCE: `Every phase was priced globally: none destabilizes by more than ε = ${p.certificate.eps}.`,
    GLOBALLY_CERTIFIED: 'Every phase was priced globally to optimality. No phase is missing.',
    GLOBAL_SEARCH_INCOMPLETE: 'Global pricing hit its time or gap limit before proving stability for every phase.',
    FAILED: 'The restricted solve did not converge.',
  }[level] || '';

  $('detail').innerHTML = `
    <table>
      <thead><tr><th>Phase</th><th>x(${db.elements.join(') / x(')})</th><th>Atom fraction</th><th>Amount</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="note">
      T = ${p.T.toFixed(1)} K, overall x(${db.elements[0]}) = ${p.x.toFixed(4)} ·
      G = ${p.GM.toFixed(3)} J/mol-atoms (${p.G_RT.toFixed(6)} RT) ·
      μ = ${p.mu.map((m) => m.toFixed(1)).join(', ')} J/mol ·
      ${p.n_iterations} iteration${p.n_iterations === 1 ? '' : 's'}${p.ms ? ` · ${p.ms.toFixed(0)} ms` : ''}
    </p>
    <div class="cert ${cls}"><strong>${level}</strong> — ${explain}
      ${p.certificate.messages.length ? `<br>${p.certificate.messages.join('<br>')}` : ''}
    </div>
    <p class="note">
      Going further than this — proving no phase is missing — means pricing every phase
      globally, and that runs through discopt's Rust solver, which has no WebAssembly
      build. It is the one step this page cannot take. Natively it is
      <code>result.certify()</code>, or <code>equilibrium(..., certify="final")</code>.
    </p>
  `;
}

// --- worker -----------------------------------------------------------------

let worker = null;

function spawnWorker() {
  if (worker) worker.terminate();
  state.ready = false;
  // The worker inherits this page's query string, so ?pyodide= reaches it.
  worker = new Worker(`./worker.js${self.location.search}`, { type: 'module' });
  worker.onmessage = onMessage;
  renderSelection();
}

function onMessage({ data }) {
  switch (data.type) {
    case 'status':
      setStatus(data.text);
      break;
    case 'ready':
      state.ready = true;
      setStatus('ready');
      renderSelection();
      break;
    case 'database':
      state.databaseInfo = { ...state.databaseInfo, [data.info.name]: data.info };
      // The elements a picked file covers are only knowable once Python has
      // parsed it, so registration finishes here rather than at the file input.
      // A re-parse after Cancel arrives here too, with nothing pending.
      if (picked && picked.name === data.info.name) {
        registerDatabase(picked, data.info);
        setStatus(`${picked.file}: ${data.info.elements.length} elements, ${data.info.phases.length} phases`);
        picked = null;
      }
      renderSelection();
      break;
    case 'pass-start':
      run.expected = data.total;
      run.arrived = 0;
      drawPlot();
      break;
    case 'pass-point':
      receivePoint(data.point);
      break;
    case 'pass-done':
      run.seconds += data.summary.seconds;
      run.failed += data.summary.failed;
      nextPass();
      break;
    case 'point':
      state.points = state.mode === 'point' ? [...state.points, data.point] : state.points;
      state.selected = data.point;
      finishRun();
      drawPlot();
      showDetail(data.point);
      setStatus(`solved in ${data.point.ms} ms · ${data.point.phases.join(' + ')}`);
      break;
    case 'transitions':
      // These arrive during the first pass and change where the boundaries
      // end, so the plot is redrawn rather than waiting for the next point.
      state.pureEnds = data.ends;
      drawPlot();
      break;
    case 'error':
      finishRun();
      setStatus(data.message, true);
      break;
    case 'fatal':
      state.ready = false;
      finishRun();
      setStatus(data.message, true);
      break;
  }
}

function setStatus(text, isError = false) {
  $('status').textContent = text;
  $('status').classList.toggle('err', isError);
}

function startRun() {
  state.running = true;
  $('run').disabled = true;
  $('cancel').disabled = false;
}

function finishRun() {
  state.running = false;
  $('cancel').disabled = true;
  $('progress-bar').style.width = '0';
  renderSelection();
}

function runPoint({ T, x, b }) {
  const db = currentDatabase();
  if (!db) return;
  startRun();
  setStatus('solving…');
  worker.postMessage({
    type: 'point', db, elements: db.elements, T, x, b,
    pressure: Number($('pressure').value),
  });
}

// --- passes -----------------------------------------------------------------

// Redrawing on every point is the whole value of streaming, but a refined run
// carries far more of them than a grid did and every redraw retraces every
// boundary. Cap the rate: the eye cannot read faster than this, and each pass
// draws once more when it ends, so nothing is left off the screen.
let lastDraw = 0;

function drawSoon() {
  const now = performance.now();
  if (now - lastDraw < 60) return;
  lastDraw = now;
  drawPlot();
}

/** The points of the pass in flight, by label, for a pass that needs them. */
let arriving = new Map();

function receivePoint(point) {
  // The cell is matched back by the label it was asked for, because a pass is a
  // list of coordinates and nothing about the order says where they came from.
  // A bisection probe has no cell, by design -- it owns no patch of the diagram.
  const cell = run.cells.get(point.id);
  if (cell) point.cell = cell;
  arriving.set(point.id, point);
  state.points.push(point);
  run.arrived += 1;
  $('progress-bar').style.width = `${(100 * run.arrived) / (run.expected || 1)}%`;
  drawSoon();
  const what = run.phase === 'invariants'
    ? `locating invariants (${run.steps})`
    : (run.round === 1 ? 'grid' : `refining (round ${run.round - 1})`);
  setStatus(
    `${what}: ${run.arrived} of ${run.expected} · ${state.points.length} points so far · `
    + `${((performance.now() - run.started) / 1000).toFixed(1)} s`,
  );
}

/**
 * Ask the worker for one pass: exactly these points, in this order.
 *
 * `countsAsRound` separates the two kinds of pass. A refinement pass is a round
 * and is reported as one; a bisection pass is a handful of probes and should not
 * make the run look like it refined another time.
 */
function sendPass(cells, countsAsRound = true) {
  const db = currentDatabase();
  if (!db || !cells.length) {
    endRun();
    return;
  }
  if (countsAsRound) run.round += 1;
  arriving = new Map();
  for (const c of cells) if (c.x0 !== undefined) run.cells.set(c.id, c);
  worker.postMessage({
    type: 'points', db, elements: db.elements,
    coords: cells.map((c) => ({ id: c.id, T: c.T, x: c.x })),
    pressure: Number($('pressure').value),
  });
}

/**
 * After a pass: refine, bisect, or stop.
 *
 * The two phases run one after the other. Refinement has to finish first
 * because the brackets are read off the cells it leaves behind -- a coarse cell
 * brackets an invariant across 70 K, and bisecting from there would work but
 * would spend its solves recovering resolution the refinement was going to
 * produce anyway.
 */
function nextPass() {
  if (run.stopped) return;
  if (run.phase === 'invariants') {
    applyBisection(arriving);
    run.steps += 1;
    const probes = run.steps < INVARIANT_STEPS ? bisectionCells() : [];
    if (probes.length) {
      sendPass(probes, false);
      return;
    }
    state.invariants = invariantLines();
    state.criticals = criticalPoints();
    endRun();
    return;
  }
  if (run.round > run.rounds) {
    startInvariants();
    return;
  }
  const marks = crossedCells();
  if (!marks.size) {
    startInvariants('every cell agrees with its neighbours');
    return;
  }
  // Counted before the cells are made, because making them supersedes their
  // parents -- and a run that stops at the budget should keep the wash it has.
  if (state.points.length + refinementCost(marks) > POINT_BUDGET) {
    startInvariants(`stopped at the ${POINT_BUDGET}-point budget`);
    return;
  }
  sendPass(refinementCells(marks));
}

/** Hand over from refining to locating the three-phase horizontals. */
function startInvariants(why) {
  run.why = why;
  run.phase = 'invariants';
  run.brackets = invariantBrackets();
  run.steps = 0;
  const probes = bisectionCells();
  if (!probes.length) {
    endRun();
    return;
  }
  sendPass(probes, false);
}

function endRun(why) {
  finishRun();
  drawPlot();
  const rounds = Math.max(0, run.round - 1);
  const n = state.invariants.length;
  setStatus(
    `${state.points.length} points in ${run.seconds.toFixed(1)} s · `
    + `${rounds} refinement round${rounds === 1 ? '' : 's'}`
    + ` · ${n} invariant${n === 1 ? '' : 's'}`
    + (state.criticals.length
      ? ` · ${state.criticals.length} critical point${state.criticals.length === 1 ? '' : 's'}`
      : '')
    + (run.failed ? ` · ${run.failed} failed` : '')
    + (why || run.why ? ` · ${why || run.why}` : '')
    + ' · hover for the composition, click a point for its certificate',
  );
}

// --- wiring -----------------------------------------------------------------

function setMode(mode) {
  state.mode = mode;
  $('mode-diagram').setAttribute('aria-pressed', String(mode === 'diagram'));
  $('mode-point').setAttribute('aria-pressed', String(mode === 'point'));
  $('diagram-controls').hidden = mode !== 'diagram';
  $('point-controls').hidden = mode === 'diagram';
  state.points = [];
  state.selected = null;
  clearHover();
  $('detail').innerHTML = '';
  drawPlot();
  updateEstimate();
}

function setComposition(kind) {
  state.composition = kind;
  $('comp-fraction').setAttribute('aria-pressed', String(kind === 'fraction'));
  $('comp-amounts').setAttribute('aria-pressed', String(kind === 'amounts'));
  $('pt-x-label').hidden = kind !== 'fraction';
  $('pt-a-label').hidden = kind !== 'amounts';
  $('pt-b-label').hidden = kind !== 'amounts';
}

buildTable();
renderSelection();
// Draw once before anything is picked, so the plot area starts with its "pick
// two elements" placeholder instead of an empty frame that reads as broken.
drawPlot();
spawnWorker();

// --- hover ------------------------------------------------------------------

function clearHover() {
  state.hover = null;
  $('tip').hidden = true;
}

/** Client coordinates to SVG user units, which is what the scales are in. */
function toSvg(svg, ev) {
  const ctm = svg.getScreenCTM();
  if (!ctm) return null;
  return new DOMPoint(ev.clientX, ev.clientY).matrixTransform(ctm.inverse());
}

/**
 * The solved point the cursor is pointing at, or null in the gaps.
 *
 * Two ways to point at one. The first is its cell. Cells tile the diagram
 * exactly, so containment is unambiguous and, unlike a nearest-node test, needs
 * no assumption about the spacing -- which matters now that refinement makes
 * the spacing deliberately uneven. A cell whose children have replaced it no
 * longer owns its patch, so it is passed over here.
 *
 * The second is its dots. A tie-line end is drawn at the coexisting
 * composition, which can be a long way from the overall composition the cell
 * was solved at -- for a nearly pure phase it sits against the edge of the
 * diagram. Those dots are the most inviting thing on the plot to point at, so
 * they have to answer too, and they are matched by distance on screen, because
 * that is what pointing at something a few pixels wide means.
 *
 * The cell wins when both match: it is the region the cursor is actually in,
 * and every point across a two-phase field shares the same pair of tie-line
 * ends, so the dots alone cannot say which point was meant.
 */
function pointAt(svgPt) {
  if (!frame || !state.points.length) return null;
  const { xLo, xHi, tLo, tHi } = frame;
  const x = xLo + ((svgPt.x - PAD.left) / (W - PAD.left - PAD.right)) * (xHi - xLo);
  const T = tLo + ((H - PAD.bottom - svgPt.y) / (H - PAD.top - PAD.bottom)) * (tHi - tLo);
  const px = (v) => PAD.left + ((v - xLo) / (xHi - xLo)) * (W - PAD.left - PAD.right);
  const py = (v) => H - PAD.bottom - ((v - tLo) / (tHi - tLo || 1)) * (H - PAD.top - PAD.bottom);
  let dot = null;
  let dotD = 7;   // px, a little wider than the largest dot drawn
  for (const p of state.points) {
    const c = p.cell;
    if (c && !p.superseded && x >= c.x0 && x < c.x1 && T >= c.t0 && T < c.t1) return p;
    if (p.error) continue;
    for (const s of uniqueSets(p)) {
      const d = Math.hypot(px(s.x[0]) - svgPt.x, py(p.T) - svgPt.y);
      if (d < dotD) {
        dotD = d;
        dot = p;
      }
    }
  }
  return dot;
}

function tipHtml(p) {
  const db = currentDatabase();
  const a = db ? db.elements[0] : 'A';
  const head = `T = ${p.T.toFixed(0)} K · overall x(${a}) = ${p.x.toFixed(3)}`;
  if (p.error) return `<div class="tip-head">${head}</div><b>did not solve</b>`;
  const sets = uniqueSets(p);
  const rows = sets
    .map((s) => `<tr>
        <td><i class="sw" style="background:${colorFor(s.phase)}"></i>${s.phase}</td>
        <td class="num">${s.x[0].toFixed(4)}</td>
        <td class="num">${(100 * s.fraction).toFixed(1)}%</td>
      </tr>`)
    .join('');
  return `<div class="tip-head">${head}</div>
    <b>${sets.map((s) => s.phase).join(' + ')}</b>
    <table>
      <thead><tr><th></th><th class="num">x(${a})</th><th class="num">of atoms</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/** Keep the tooltip beside the cursor, flipped before it runs off the panel. */
function placeTip(ev) {
  const tip = $('tip');
  const wrap = tip.parentElement.getBoundingClientRect();
  const x = ev.clientX - wrap.left;
  const y = ev.clientY - wrap.top;
  const left = x + 14 + tip.offsetWidth > wrap.width ? x - 14 - tip.offsetWidth : x + 14;
  const top = y + 14 + tip.offsetHeight > wrap.height ? y - 14 - tip.offsetHeight : y + 14;
  tip.style.left = `${Math.max(0, left)}px`;
  tip.style.top = `${Math.max(0, top)}px`;
}

{
  const plot = $('plot');
  plot.addEventListener('pointermove', (ev) => {
    const svgPt = toSvg(plot, ev);
    const p = svgPt ? pointAt(svgPt) : null;
    if (!p) {
      if (state.hover) { clearHover(); drawPlot(); }
      return;
    }
    const tip = $('tip');
    // Only redraw when the point changes -- once per cell crossed, not once per
    // pixel. Repositioning the tooltip is just two style writes.
    if (p !== state.hover) {
      state.hover = p;
      tip.innerHTML = tipHtml(p);
      drawPlot();
    }
    tip.hidden = false;
    placeTip(ev);
  });
  // The click that opens the detail panel is resolved the way the tooltip is,
  // rather than by a listener on each dot. The dots are 1.4 units across now --
  // too small to ask anyone to hit -- and `pointAt` already answers "what is
  // being pointed at", by cell first and then by nearby dot, which is the
  // question a click asks too. Clicking anywhere in a cell now works, and there
  // is one listener instead of several thousand.
  plot.addEventListener('click', (ev) => {
    const svgPt = toSvg(plot, ev);
    const p = svgPt ? pointAt(svgPt) : null;
    if (!p) return;
    state.selected = p;
    showDetail(p);
    drawPlot();
  });
  plot.addEventListener('pointerleave', () => {
    if (state.hover) { clearHover(); drawPlot(); } else clearHover();
  });
}

$('mode-diagram').addEventListener('click', () => setMode('diagram'));
$('mode-point').addEventListener('click', () => setMode('point'));
$('comp-fraction').addEventListener('click', () => setComposition('fraction'));
$('comp-amounts').addEventListener('click', () => setComposition('amounts'));
for (const id of ['n-t', 'n-x', 'n-refine']) $(id).addEventListener('input', updateEstimate);

$('run').addEventListener('click', () => {
  const db = currentDatabase();
  if (!db) return;
  state.phaseColors = new Map();
  state.selected = null;
  $('detail').innerHTML = '';
  if (state.mode === 'point') {
    state.points = [];
    if (state.composition === 'amounts') {
      runPoint({ T: Number($('pt-t').value), b: [Number($('pt-a').value), Number($('pt-b').value)] });
    } else {
      runPoint({ T: Number($('pt-t').value), x: Number($('pt-x').value) });
    }
    return;
  }
  state.points = [];
  clearHover();
  // Reset the sampling sequence too, so the same run twice draws the same
  // diagram rather than a differently jittered one.
  seq = 0;
  state.invariants = [];
  state.criticals = [];
  state.pureEnds = [];
  // Work cached for the length of a run is keyed on this, so it is dropped in
  // one place here rather than invalidated from several.
  state.epoch += 1;
  run.cells = new Map();
  run.round = 0;
  run.rounds = Number($('n-refine').value);
  run.started = performance.now();
  run.seconds = 0;
  run.failed = 0;
  run.stopped = false;
  // Back to refining. Without this a second run picks up where the first one
  // finished -- in the bisection phase -- and never lays a grid at all.
  run.phase = 'grid';
  run.brackets = [];
  run.steps = 0;
  run.why = undefined;
  startRun();
  setStatus('solving…');
  // Asked for before the first pass, not after the last: this is a root find
  // on temperature over the models this run will solve with, so it costs no
  // equilibria, and having it early means the boundaries reach the pure
  // elements from the moment they are first drawn.
  worker.postMessage({
    type: 'transitions',
    db,
    elements: db.elements,
    tLo: Number($('t-lo').value),
    tHi: Number($('t-hi').value),
    pressure: Number($('pressure').value),
  });
  sendPass(initialCells());
});

$('cancel').addEventListener('click', () => {
  // A solve runs synchronously inside the worker, so nothing short of killing
  // the thread can interrupt one. The points already drawn stay.
  // Stop the refinement loop as well as the thread: a pass already in flight
  // must not queue the next one up on its way out.
  run.stopped = true;
  setStatus('cancelled — restarting Python…');
  finishRun();
  spawnWorker();
});

/**
 * Open a TDB from the visitor's own disk.
 *
 * Nothing is uploaded and nothing is redistributed: the file is read in this
 * tab and parsed by the Python running in it. That is the whole reason this is
 * a file input rather than a list of databases to download — the multicomponent
 * files worth having are published under their own terms, so the page reads one
 * the visitor already has instead of serving a copy.
 */
$('tdb-file').addEventListener('change', async (ev) => {
  const file = ev.target.files?.[0];
  if (!file) return;
  setStatus(`reading ${file.name}…`);
  let text;
  try {
    text = await file.text();
  } catch (err) {
    setStatus(`could not read ${file.name}: ${err.message}`, true);
    return;
  }
  const name = file.name.replace(/\.tdb$/i, '');
  picked = { name, file: file.name, text };
  // Queues if Python is still booting; the worker awaits its own readiness.
  worker.postMessage({
    type: 'database',
    db: { name, file: file.name, text, pressure: Number($('pressure').value) },
  });
});

const themeSelect = $('theme');
try { themeSelect.value = localStorage.getItem('calphad-theme') || 'auto'; } catch (e) { /* private mode */ }
themeSelect.addEventListener('change', () => {
  const pref = themeSelect.value;
  try { localStorage.setItem('calphad-theme', pref); } catch (e) { /* private mode */ }
  const dark = pref === 'dark' || (pref === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  drawPlot();
});
