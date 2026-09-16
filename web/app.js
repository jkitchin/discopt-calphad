// Page logic: choose two elements, ask the worker for equilibria, draw them as
// they arrive. The worker owns Python and the solver; this side is a periodic
// table, a plot, and a table of numbers.

import { ELEMENTS } from './elements.js';

/** The databases build-web.sh stages. Each covers exactly one binary. */
// `ms` is the measured cost of one point under Pyodide (M4 Pro, Pyodide 0.28.3)
// — about twice the native rate. It is only used to warn before a long run.
const DATABASES = [
  { name: 'Al-Zn', file: 'alzn_mey.tdb', elements: ['AL', 'ZN'], tLo: 400, tHi: 950, ms: 69 },
  { name: 'Cu-Mg', file: 'cumg_nims.tdb', elements: ['CU', 'MG'], tLo: 400, tHi: 1200, ms: 144 },
];

const WITH_DATA = new Set(DATABASES.flatMap((db) => db.elements));
const $ = (id) => document.getElementById(id);

const state = {
  chosen: [],          // element symbols, in the order picked
  mode: 'diagram',     // 'diagram' | 'point'
  composition: 'fraction', // 'fraction' | 'amounts'
  points: [],          // every point received for the current run
  phaseColors: new Map(),
  running: false,
  selected: null,      // the point whose detail is shown
  hover: null,         // the point under the cursor, described by the tooltip
};

/**
 * The run in progress. A run is several passes: the starting grid, then one
 * pass per refinement round. Kept apart from `state` because none of it
 * outlives the run -- `state.points` is the result, this is the bookkeeping.
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

/** The database covering the chosen pair, if there is one. */
function currentDatabase() {
  if (state.chosen.length !== 2) return null;
  const want = [...state.chosen].sort().join('-');
  return DATABASES.find((db) => [...db.elements].sort().join('-') === want) || null;
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
    cell.className = 'el' + (WITH_DATA.has(el.sym.toUpperCase()) ? ' has-data' : '');
    cell.style.gridRow = String(el.row === 9 ? 9 : el.row === 10 ? 10 : el.row);
    cell.style.gridColumn = String(el.col);
    cell.innerHTML = `<span class="z">${el.z}</span>${el.sym}`;
    cell.dataset.sym = el.sym.toUpperCase();
    if (WITH_DATA.has(el.sym.toUpperCase())) {
      cell.title = `${el.name} — click to select`;
      cell.addEventListener('click', () => toggleElement(el.sym.toUpperCase()));
    } else {
      cell.title = `${el.name} — no database staged for this element`;
      cell.disabled = true;
    }
    table.append(cell);
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
  const pairs = DATABASES.map((d) => `<span class="pill">${d.elements.join('–')}</span>`).join('');
  if (db) {
    const info = state.databaseInfo?.[db.name];
    const phases = info ? ` · phases: ${info.phases.join(', ')}` : '';
    $('systems').innerHTML = `<strong>${db.elements.join('–')}</strong> from <code>${db.file}</code>${phases}`;
    $('t-lo').value = db.tLo;
    $('t-hi').value = db.tHi;
    setXLabels(db);
  } else if (state.chosen.length === 2) {
    $('systems').innerHTML =
      `No staged database covers ${state.chosen.join('–')}. Available: ${pairs}`;
  } else {
    $('systems').innerHTML = `Pick two elements. Available: ${pairs}`;
  }
  updateEstimate();
  $('run').disabled = !db || state.running || !state.ready;
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
  $('estimate').textContent =
    `${n} points to start, about ${seconds < 1 ? '1' : seconds.toFixed(0)} s — `
    + `${db.elements.join('–')} measures ${db.ms} ms per point in the browser. `
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

  const out = [];
  for (const [key, byT] of fields) {
    const phase = key.slice(key.indexOf('|') + 1);
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
        if (piece.length > 1) out.push({ phase, pts: piece });
      }
    }
  }
  return out;
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
  const holding = (x, T) => live.find((p) => x >= p.cell.x0 && x < p.cell.x1
    && T >= p.cell.t0 && T < p.cell.t1);
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
  // cell, the boundary lines traced from the tie-lines, then the tie-line ends
  // themselves. Each cell is drawn at its own extent, so refinement shows in
  // the wash directly -- fine where a boundary runs, coarse where nothing
  // happens. A superseded cell is skipped because its children cover the same
  // ground, and drawing both would double the wash there.
  for (const p of state.points) {
    if (p.error || p.superseded || !p.cell) continue;
    const sets = uniqueSets(p);
    // Merged, not `p.phases`: a lone phase returned as two composition sets is
    // still one phase holding the cell, and would otherwise get no wash.
    if (sets.length !== 1) continue;
    const c = p.cell;
    svg.append(node('rect', {
      x: sx(c.x0), y: sy(c.t1), width: sx(c.x1) - sx(c.x0), height: sy(c.t0) - sy(c.t1),
      fill: colorFor(sets[0].phase), 'fill-opacity': 0.11, stroke: 'none',
    }));
  }

  // Straight segments between solved temperatures, not a spline: a smooth curve
  // through these points would invent structure between them that was never
  // computed.
  for (const b of boundaries()) {
    svg.append(node('polyline', {
      points: b.pts.map((q) => `${sx(q.x)},${sy(q.T)}`).join(' '),
      fill: 'none', stroke: colorFor(b.phase), 'stroke-width': 1.7,
      'stroke-opacity': 0.9, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
    }));
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
      // No <title>: the hover tooltip covers the whole plot area and answers
      // immediately, where a native tooltip only fires over the dot itself and
      // only after a delay.
      const dot = node('circle', {
        // Refinement turned a few hundred points into a few thousand, and at that
        // density a single-phase dot stops being information: it is one sample
        // inside a field whose wash already says what holds it. Kept small and
        // faint, so it reads as a stipple of where the solves actually went and
        // still answers a click, while the multi-phase dots -- which sit exactly
        // on a boundary -- stay the ones the eye follows.
        cx: sx(s.x[0]), cy: sy(p.T), r: sets.length > 1 ? 3.1 : 1.4,
        fill: colorFor(s.phase), 'fill-opacity': sets.length > 1 ? 0.95 : 0.26,
        stroke: chosen ? 'var(--ink)' : 'none', 'stroke-width': chosen ? 1.5 : 0,
      });
      dot.style.cursor = 'pointer';
      dot.addEventListener('click', () => { state.selected = p; showDetail(p); drawPlot(); });
      svg.append(dot);
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

function receivePoint(point) {
  // The cell is matched back by the label it was asked for, because a pass is a
  // list of coordinates and nothing about the order says where they came from.
  const cell = run.cells.get(point.id);
  if (cell) point.cell = cell;
  state.points.push(point);
  run.arrived += 1;
  $('progress-bar').style.width = `${(100 * run.arrived) / (run.expected || 1)}%`;
  drawSoon();
  setStatus(
    `${run.round === 1 ? 'grid' : `refining (round ${run.round - 1})`}: `
    + `${run.arrived} of ${run.expected} · ${state.points.length} points so far · `
    + `${((performance.now() - run.started) / 1000).toFixed(1)} s`,
  );
}

/** Ask the worker for one pass: exactly these cells, in this order. */
function sendPass(cells) {
  const db = currentDatabase();
  if (!db || !cells.length) {
    endRun();
    return;
  }
  run.round += 1;
  for (const c of cells) run.cells.set(c.id, c);
  worker.postMessage({
    type: 'points', db, elements: db.elements,
    coords: cells.map((c) => ({ id: c.id, T: c.T, x: c.x })),
    pressure: Number($('pressure').value),
  });
}

/** After a pass: subdivide the cells a boundary runs through, or stop. */
function nextPass() {
  if (run.stopped) return;
  if (run.round > run.rounds) {
    endRun();
    return;
  }
  const marks = crossedCells();
  if (!marks.size) {
    endRun('every cell agrees with its neighbours');
    return;
  }
  // Counted before the cells are made, because making them supersedes their
  // parents -- and a run that stops at the budget should keep the wash it has.
  if (state.points.length + refinementCost(marks) > POINT_BUDGET) {
    endRun(`stopped at the ${POINT_BUDGET}-point budget`);
    return;
  }
  sendPass(refinementCells(marks));
}

function endRun(why) {
  finishRun();
  drawPlot();
  const rounds = Math.max(0, run.round - 1);
  setStatus(
    `${state.points.length} points in ${run.seconds.toFixed(1)} s · `
    + `${rounds} refinement round${rounds === 1 ? '' : 's'}`
    + (run.failed ? ` · ${run.failed} failed` : '')
    + (why ? ` · ${why}` : '')
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
  run.cells = new Map();
  run.round = 0;
  run.rounds = Number($('n-refine').value);
  run.started = performance.now();
  run.seconds = 0;
  run.failed = 0;
  run.stopped = false;
  startRun();
  setStatus('solving…');
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

const themeSelect = $('theme');
try { themeSelect.value = localStorage.getItem('calphad-theme') || 'auto'; } catch (e) { /* private mode */ }
themeSelect.addEventListener('change', () => {
  const pref = themeSelect.value;
  try { localStorage.setItem('calphad-theme', pref); } catch (e) { /* private mode */ }
  const dark = pref === 'dark' || (pref === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  drawPlot();
});
