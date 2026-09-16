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
  expected: 0,
  started: 0,
  selected: null,      // the point whose detail is shown
  hover: null,         // the point under the cursor, described by the tooltip
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
  const seconds = (n * db.ms) / 1000;
  $('estimate').textContent =
    `${n} points, about ${seconds < 1 ? '1' : seconds.toFixed(0)} s — ` +
    `${db.elements.join('–')} measures ${db.ms} ms per point in the browser. ` +
    'They are drawn as they are solved, and Cancel stops the run.';
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
      const xs = byT.get(p.T) || [];
      if (!xs.some((x) => Math.abs(x - s.x[0]) < 1e-6)) xs.push(s.x[0]);
      byT.set(p.T, xs);
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
  const nX = state.mode === 'diagram' ? Number($('n-x').value) : 20;
  const nT = state.mode === 'diagram' ? Number($('n-t').value) : 10;
  // Remember the frame the cursor will be read against. Recomputing it from the
  // inputs on each pointermove would describe a diagram that is no longer on
  // screen, as soon as someone edits a field after a run.
  frame = { xLo, xHi, tLo, tHi, nX, nT };

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
  // themselves. The wash is deliberately faint -- it is one cell per solved
  // point and so only ever as fine as the grid, which is exactly why the
  // boundaries are drawn as lines instead.
  const cellW = (W - PAD.left - PAD.right) / Math.max(nX - 1, 1);
  const cellH = (H - PAD.top - PAD.bottom) / Math.max(nT - 1, 1);

  for (const p of state.points) {
    if (p.error || p.phases.length !== 1) continue;
    svg.append(node('rect', {
      x: sx(p.x) - cellW / 2, y: sy(p.T) - cellH / 2, width: cellW, height: cellH,
      fill: colorFor(p.phases[0]), 'fill-opacity': 0.11, stroke: 'none',
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
        cx: sx(s.x[0]), cy: sy(p.T), r: sets.length > 1 ? 3.1 : 2.1,
        fill: colorFor(s.phase), 'fill-opacity': sets.length > 1 ? 0.95 : 0.5,
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
    case 'diagram-start':
      state.points = [];
      clearHover();
      state.expected = data.total;
      state.started = performance.now();
      drawPlot();
      break;
    case 'diagram-point':
      state.points.push(data.point);
      $('progress-bar').style.width = `${(100 * state.points.length) / state.expected}%`;
      // Redrawing on every point is cheap next to a solve, and is the point of
      // streaming: the diagram fills in row by row.
      drawPlot();
      setStatus(`${state.points.length} of ${state.expected} points · ${((performance.now() - state.started) / 1000).toFixed(1)} s`);
      break;
    case 'diagram-done':
      finishRun();
      setStatus(
        `${data.summary.solved} points in ${data.summary.seconds.toFixed(1)} s` +
        (data.summary.failed ? ` · ${data.summary.failed} failed` : '') +
        ' · hover for the composition, click a point for its certificate',
      );
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
 * Two ways to point at one. The first is its cell: a point owns a little over
 * half a grid step in each direction, which tiles the swept area so anywhere
 * inside the diagram answers, and nowhere outside it pretends to.
 *
 * The second is its dots. A tie-line end is drawn at the coexisting
 * composition, which can be a long way from the overall composition at the
 * centre of the cell -- for a nearly pure phase it lands outside the swept
 * range altogether, owning no cell at all. Those dots are the most inviting
 * thing on the plot to point at, so they have to answer too.
 *
 * The cell wins when both match: it is the region the cursor is actually in,
 * and every point across a two-phase field shares the same pair of tie-line
 * ends, so the dots alone cannot say which point was meant.
 */
function pointAt(svgPt) {
  if (!frame || !state.points.length) return null;
  const { xLo, xHi, tLo, tHi, nX, nT } = frame;
  const x = xLo + ((svgPt.x - PAD.left) / (W - PAD.left - PAD.right)) * (xHi - xLo);
  const T = tLo + ((H - PAD.bottom - svgPt.y) / (H - PAD.top - PAD.bottom)) * (tHi - tLo);
  // The sweep runs just inside the pure elements, matching the run request.
  const dx = (0.98 - 0.02) / Math.max(nX - 1, 1);
  const dT = (tHi - tLo) / Math.max(nT - 1, 1);
  let cell = null;
  let cellD = Infinity;
  let dot = null;
  let dotD = Infinity;
  for (const p of state.points) {
    const at = Math.abs(p.T - T) / (dT || 1);
    if (at > 0.6) continue;
    const ax = Math.abs(p.x - x) / dx;
    if (ax <= 0.6 && ax * ax + at * at < cellD) {
      cellD = ax * ax + at * at;
      cell = p;
    }
    if (p.error) continue;
    for (const s of uniqueSets(p)) {
      const adot = Math.abs(s.x[0] - x) / dx;
      if (adot <= 0.3 && adot * adot + at * at < dotD) {
        dotD = adot * adot + at * at;
        dot = p;
      }
    }
  }
  return cell || dot;
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
for (const id of ['n-t', 'n-x']) $(id).addEventListener('input', updateEstimate);

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
  startRun();
  setStatus('solving…');
  worker.postMessage({
    type: 'diagram', db, elements: db.elements,
    tLo: Number($('t-lo').value), tHi: Number($('t-hi').value), nT: Number($('n-t').value),
    // The endpoints are pure elements, where a solve has nothing to balance, so
    // the sweep stays just inside them.
    xLo: 0.02, xHi: 0.98, nX: Number($('n-x').value),
    pressure: Number($('pressure').value),
  });
});

$('cancel').addEventListener('click', () => {
  // A solve runs synchronously inside the worker, so nothing short of killing
  // the thread can interrupt one. The points already drawn stay.
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
