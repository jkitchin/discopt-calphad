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
const W = 760;
const H = 460;
const SVG_NS = 'http://www.w3.org/2000/svg';

function node(name, attrs, text) {
  const n = document.createElementNS(SVG_NS, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, String(v));
  if (text !== undefined) n.textContent = text;
  return n;
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

  // Grid cells first (the region map), then tie-line endpoints on top. A
  // multi-phase point puts one dot at each coexisting phase's own composition —
  // those dots are what trace the phase boundaries.
  const nX = state.mode === 'diagram' ? Number($('n-x').value) : 20;
  const nT = state.mode === 'diagram' ? Number($('n-t').value) : 10;
  const cellW = (W - PAD.left - PAD.right) / Math.max(nX - 1, 1);
  const cellH = (H - PAD.top - PAD.bottom) / Math.max(nT - 1, 1);

  for (const p of state.points) {
    if (p.error) continue;
    const fill = p.phases.length === 1 ? colorFor(p.phases[0]) : 'none';
    if (p.phases.length === 1) {
      svg.append(node('rect', {
        x: sx(p.x) - cellW / 2, y: sy(p.T) - cellH / 2, width: cellW, height: cellH,
        fill, 'fill-opacity': 0.16, stroke: 'none',
      }));
    }
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
    for (const s of p.sets) {
      const dot = node('circle', {
        cx: sx(s.x[0]), cy: sy(p.T), r: p.phases.length > 1 ? 3.1 : 2.1,
        fill: colorFor(s.phase), 'fill-opacity': p.phases.length > 1 ? 0.95 : 0.5,
        stroke: chosen ? 'var(--ink)' : 'none', 'stroke-width': chosen ? 1.5 : 0,
      });
      dot.style.cursor = 'pointer';
      dot.addEventListener('click', () => { state.selected = p; showDetail(p); drawPlot(); });
      const title = node('title', {});
      title.textContent =
        `${p.phases.join(' + ')}\nT = ${p.T.toFixed(0)} K, overall x = ${p.x.toFixed(3)}\n` +
        `${s.phase}: x = ${s.x[0].toFixed(4)}, ${(100 * s.fraction).toFixed(1)}% of atoms`;
      dot.append(title);
      svg.append(dot);
    }
  }
  renderLegend();
}

function renderLegend() {
  const seen = [...state.phaseColors.keys()].sort();
  $('legend').innerHTML = seen
    .map((p) => `<span><i class="swatch" style="background:${colorFor(p)}"></i>${p}</span>`)
    .join('') + (seen.length ? ' <span>· filled band = single phase, dots = coexisting compositions (tie-line ends)</span>' : '');
}

// --- detail panel -----------------------------------------------------------

function showDetail(p) {
  const db = currentDatabase();
  const rows = p.sets
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
        ' · click any point for detail',
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
