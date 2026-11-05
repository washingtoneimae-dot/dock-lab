/* docklab — UI, camera, adaptive quality, chart */
import { parsePDB } from './target.js';
import { DockEngine } from './engine.js';

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));

// Quench ramps to 0 K over roughly this long, rather than cutting the
// temperature in a single frame, so each pose has time to relax.
const QUENCH_RAMP_MS = 40000;
const REHEAT_COOL_RATE = 0.022; // K per ms

const canvas = $('#canvas');
const stage = $('#stage');
const panel = $('#panel');
const loading = $('#loading');
const loadingText = $('#loading-text');
const errorCard = $('#error');
const errorText = $('#error-text');
const toastEl = $('#toast');
const fileInput = $('#file');
const chart = $('#chart');
const qualityBadge = $('#quality-badge');

const els = {
  fps: $('#s-fps'), atoms: $('#s-atoms'), poses: $('#s-poses'),
  awake: $('#s-awake'), moving: $('#s-moving'),
  best: $('#s-best'), state: $('#s-state'),
  energy: $('#m-energy'), rmsd: $('#m-rmsd'), mean: $('#m-mean'),
  hudTarget: $('#hud-target'), hudPdb: $('#hud-pdb'), hudGpu: $('#hud-gpu'), hudQuality: $('#hud-quality'),
  tName: $('#t-name'), tSub: $('#t-sub'), tCounts: $('#t-counts'),
  run: $('#run'), runLabel: $('#run-label'),
  substeps: $('#s-substeps'), temp: $('#s-temp'), cutoff: $('#s-cutoff'),
  chartNote: $('#chart-note')
};

const coarse = matchMedia('(pointer: coarse)').matches;
const params = new URLSearchParams(location.search);
const paramPoses = Number(params.get('poses'));
// Fixed starting configuration: 2,000 poses and 2 substeps, matching the
// documented docking defaults. Users can still change either from the panel,
// and auto-quality may lower substeps later if the GPU cannot keep up.
const defaultPoses = Number.isFinite(paramPoses) && paramPoses >= 64
  ? Math.min(3000, Math.round(paramPoses))
  : 2000;
const dprCap = coarse ? 1.25 : 1.5;

const state = {
  running: true,
  fps: 60,
  slowTime: 0,
  fastTime: 0,
  quality: 2,
  qualityTimer: 0,
  cooling: false,
  coolTarget: 90,
  coolRate: REHEAT_COOL_RATE,
  quenching: false,
  quenchAnnounced: false,
  dragging: false,
  lastX: 0,
  lastY: 0,
  lastStat: null,
  toastTimer: 0,
  dpr: Math.min(devicePixelRatio || 1, dprCap)
};

let engine = null;
let target = null;
let lastFrame = performance.now();
let lastHud = 0;

// --------------------------------------------------------------- boot ----

async function boot() {
  try {
    loadingText.textContent = 'Fetching 1M17 · EGFR kinase / erlotinib';
    const url = new URL('../assets/1M17.pdb', import.meta.url);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Could not load 1M17.pdb (HTTP ${res.status}). Run this folder through the included server.`);
    const text = await res.text();

    loadingText.textContent = 'Parsing 2,511 protein atoms + AQ4 ligand';
    target = parsePDB(text, { ligandRes: 'AQ4' });

    loadingText.textContent = 'Building 3D voxel hash + GPU pose textures';
    engine = new DockEngine(canvas, {
      poseCount: defaultPoses,
      substeps: 2,
      temperature: 330,
      ljScale: 1.0,
      esScale: 1.0,
      burial: 1.0,
      solvent: 1.0,
      torsions: true,
      autoRotate: true,
      seedFraction: 0.08
    });
    window.docklab = {
      get engine() { return engine; },
      get target() { return target; },
      get state() { return state; }
    };

    engine.onstats = onStats;
    engine.resize(stage.clientWidth, stage.clientHeight, state.dpr);
    engine.setTarget(target);
    engine.fps = 60;

    // debug hooks for headless/CI isolation
    if (params.has('nostep')) engine.step = () => {};
    if (params.has('norender')) engine.render = () => {};
    if (params.has('noread')) engine._readStats = () => {};
    if (params.has('rec')) engine.opts.showReceptor = params.get('rec') !== '0';
    if (params.has('swarm')) engine.opts.showSwarm = params.get('swarm') !== '0';
    if (params.has('best')) engine.opts.showBest = params.get('best') !== '0';
    if (params.has('lines')) engine.opts.showLines = params.get('lines') !== '0';
    if (params.has('surface')) engine.opts.showSurface = params.get('surface') !== '0';

    updateTargetUI(target);
    updateEngineUI();
    bindControls();
    bindCamera();
    bindKeys();

    els.hudGpu.textContent = shortGPU(engine.renderer);
    els.hudQuality.textContent = `${formatCount(engine.recCount + engine.poseCount * engine.ligCount)} atom field`;

    requestAnimationFrame(loop);
    requestAnimationFrame(() => {
      loading.classList.add('done');
      toast('60k atom field online · drag to orbit, F to focus the best pose');
    });
  } catch (err) {
    console.error(err);
    loading.classList.add('done');
    errorCard.hidden = false;
    errorText.textContent = err.message || String(err);
  }
}

function shortGPU(name) {
  if (!name) return 'WebGL2';
  const s = String(name);
  if (/SwiftShader/i.test(s)) return 'SwiftShader (software)';
  if (/UHD Graphics/i.test(s)) {
    const m = s.match(/UHD Graphics[^()]*/i);
    return 'Intel ' + (m ? m[0].trim() : 'UHD');
  }
  if (/Iris/i.test(s)) return 'Intel Iris';
  if (/Apple/i.test(s)) return s.replace(/ANGLE \(|\)/g, '').slice(0, 28);
  if (/AMD|Radeon/i.test(s)) return 'AMD Radeon';
  if (/NVIDIA|GeForce|RTX|GTX/i.test(s)) return 'NVIDIA GPU';
  return s.slice(0, 30);
}

// --------------------------------------------------------------- loop ----

function loop(now) {
  const raw = now - lastFrame;
  const dt = Math.min(50, raw);
  lastFrame = now;
  if (raw > 0) {
    const inst = 1000 / raw;
    state.fps += (inst - state.fps) * 0.08;
  }

  if (!document.hidden && state.running) {
    if (state.cooling) {
      const t = engine.opts.temperature;
      const nt = Math.max(state.coolTarget, t - dt * state.coolRate);
      if (nt !== t) {
        engine.setOptions({ temperature: nt });
        const slider = $('#temperature');
        slider.value = String(Math.round(nt));
        updateRange(slider);
        $('#o-temp').textContent = `${Math.round(nt)} K`;
        els.temp.textContent = Math.round(nt);
      }
      if (nt <= state.coolTarget + 0.5) {
        // Land exactly on the target so 0 K really disables wake and migration.
        engine.setOptions({ temperature: state.coolTarget });
        const slider = $('#temperature');
        slider.value = String(state.coolTarget);
        updateRange(slider);
        $('#o-temp').textContent = `${Math.round(state.coolTarget)} K`;
        els.temp.textContent = Math.round(state.coolTarget);
        state.cooling = false;
        if (state.quenching && state.coolTarget <= 0.5) {
          toast('Quench reached 0 K · wake and migration off · watching mean |ΔE|');
        } else {
          toast(`Annealed to ${Math.round(state.coolTarget)} K`);
        }
      }
    }
    engine.step();
  }

  engine.fps = Math.round(state.fps);
  engine.render();

  if (now - lastHud > 220) {
    lastHud = now;
    els.fps.textContent = String(Math.round(state.fps));
    els.fps.classList.toggle('is-low', state.fps < 40);
    adaptQuality(dt);
  }

  requestAnimationFrame(loop);
}

function adaptQuality(dt) {
  if (!engine) return;
  if (state.fps < 42) {
    state.slowTime += dt;
    state.fastTime = 0;
  } else if (state.fps > 57) {
    state.fastTime += dt;
    state.slowTime = 0;
  } else {
    state.slowTime = Math.max(0, state.slowTime - dt * 0.4);
    state.fastTime = 0;
  }

  if (state.slowTime > 1800 && state.quality > 0) {
    state.slowTime = 0;
    state.quality--;
    applyQuality();
  } else if (state.fastTime > 6500 && state.quality < 2) {
    state.fastTime = 0;
    state.quality++;
    applyQuality();
  }
}

function applyQuality() {
  const q = state.quality;
  let substeps = 2;
  let scale = 1.0;
  if (q === 1) { substeps = 1; scale = 0.82; }
  if (q === 0) { substeps = 1; scale = 0.68; }
  engine.setOptions({ substeps });
  engine.renderScale = scale;
  engine.resize(stage.clientWidth, stage.clientHeight, state.dpr);
  els.substeps.textContent = substeps;
  const slider = $('#substeps');
  slider.value = String(substeps);
  updateRange(slider);
  $('#o-substeps').textContent = String(substeps);

  qualityBadge.hidden = q === 2;
  qualityBadge.textContent = q === 2 ? 'auto quality' : `auto quality · ${Math.round(scale * 100)}% res · ${substeps}×`;
  clearTimeout(state.qualityTimer);
  state.qualityTimer = setTimeout(() => { qualityBadge.hidden = true; }, 2600);
}

// --------------------------------------------------------------- stats ----

function onStats(s) {
  state.lastStat = s;
  els.atoms.textContent = formatCount(s.atoms);
  els.poses.textContent = formatCount(s.poses);
  els.awake.textContent = String(s.awake ?? s.active ?? '–');
  els.moving.textContent = String(s.moving ?? '–');
  els.best.textContent = s.best == null ? '–' : s.best.toFixed(1);
  els.energy.textContent = s.best == null ? '–' : s.best.toFixed(1);
  els.mean.textContent = s.mean == null ? '–' : s.mean.toFixed(1);
  els.rmsd.textContent = s.rmsd == null ? '–' : s.rmsd.toFixed(2);
  els.rmsd.classList.toggle('good', s.rmsd != null && s.rmsd < 2.0);

  const converged = state.quenching && s.converged && engine.opts.temperature <= 0.5;
  if (state.quenching) {
    const label = !state.running ? 'paused' : converged ? 'converged' : 'quench';
    els.state.textContent = label;
    els.state.classList.toggle('is-paused', !state.running);
    els.state.classList.toggle('is-converged', converged && state.running);
    els.chartNote.textContent = label;
    if (converged && !state.quenchAnnounced) {
      state.quenchAnnounced = true;
      const dE = s.energyDelta == null ? '—' : s.energyDelta.toFixed(3);
      const best = s.best == null ? '—' : s.best.toFixed(1);
      toast(`Quench converged · mean |ΔE| ${dE} · best ${best} kcal/mol`);
    }
  } else {
    els.state.textContent = state.running ? 'live' : 'paused';
    els.state.classList.toggle('is-paused', !state.running);
    els.state.classList.toggle('is-converged', false);
    els.chartNote.textContent = state.running ? 'live' : 'paused';
  }
  els.hudQuality.textContent = `${formatCount(s.atoms)} atoms · ${formatCount(s.poses)} poses`;
  drawChart(s.history);
}

function formatCount(n) {
  if (n == null) return '–';
  if (n < 1000) return String(n);
  return (n / 1000).toFixed(n < 10000 ? 1 : 1) + 'k';
}

// --------------------------------------------------------------- chart ----

function drawChart(history) {
  const ctx = chart.getContext('2d');
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const w = chart.clientWidth;
  const h = chart.clientHeight;
  if (!w || !h) return;
  if (chart.width !== Math.round(w * dpr) || chart.height !== Math.round(h * dpr)) {
    chart.width = Math.round(w * dpr);
    chart.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const padL = 8, padR = 8, padT = 10, padB = 8;
  const iw = w - padL - padR;
  const ih = h - padT - padB;

  ctx.strokeStyle = 'rgba(148,174,210,0.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {
    const y = padT + (ih * i) / 3;
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(w - padR, y);
    ctx.stroke();
  }

  if (!history || history.length < 2) {
    ctx.fillStyle = 'rgba(131,145,166,0.55)';
    ctx.font = '500 9px "SFMono-Regular", ui-monospace, monospace';
    ctx.fillText('waiting for first energy readback…', padL + 2, padT + 16);
    return;
  }

  const vals = [];
  for (const p of history) {
    if (Number.isFinite(p.best)) vals.push(p.best);
    if (Number.isFinite(p.mean)) vals.push(p.mean);
  }
  let lo = Math.min(...vals);
  let hi = Math.max(...vals);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return;
  if (hi - lo < 1) { hi += 0.5; lo -= 0.5; }
  const pad = (hi - lo) * 0.12;
  lo -= pad; hi += pad;

  const x = (i) => padL + (iw * i) / (history.length - 1);
  const y = (v) => padT + ih - ((v - lo) / (hi - lo)) * ih;

  // best fill
  ctx.beginPath();
  ctx.moveTo(x(0), padT + ih);
  for (let i = 0; i < history.length; i++) {
    const v = Number.isFinite(history[i].best) ? history[i].best : lo;
    ctx.lineTo(x(i), y(v));
  }
  ctx.lineTo(x(history.length - 1), padT + ih);
  ctx.closePath();
  const grad = ctx.createLinearGradient(0, padT, 0, padT + ih);
  grad.addColorStop(0, 'rgba(198,255,94,0.18)');
  grad.addColorStop(1, 'rgba(198,255,94,0.00)');
  ctx.fillStyle = grad;
  ctx.fill();

  // mean line
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const v = Number.isFinite(history[i].mean) ? history[i].mean : lo;
    const px = x(i), py = y(v);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = 'rgba(255,77,141,0.65)';
  ctx.lineWidth = 1.1;
  ctx.stroke();

  // best line
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const v = Number.isFinite(history[i].best) ? history[i].best : lo;
    const px = x(i), py = y(v);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = 'rgba(198,255,94,0.95)';
  ctx.lineWidth = 1.5;
  ctx.shadowColor = 'rgba(198,255,94,0.5)';
  ctx.shadowBlur = 6;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // last dot
  const last = history[history.length - 1];
  if (Number.isFinite(last.best)) {
    ctx.beginPath();
    ctx.arc(x(history.length - 1), y(last.best), 2.2, 0, Math.PI * 2);
    ctx.fillStyle = '#c6ff5e';
    ctx.fill();
  }

  ctx.fillStyle = 'rgba(131,145,166,0.6)';
  ctx.font = '500 8px "SFMono-Regular", ui-monospace, monospace';
  ctx.fillText(hi.toFixed(0), padL + 2, padT + 8);
  ctx.fillText(lo.toFixed(0), padL + 2, padT + ih - 2);
}

// ------------------------------------------------------------ controls ----

function bindControls() {
  $('#poses').value = String(engine.poseCount);
  $('#temperature').value = String(Math.round(engine.opts.temperature));
  $('#substeps').value = String(engine.opts.substeps);
  $('#lj').value = String(engine.opts.ljScale);
  $('#es').value = String(engine.opts.esScale);
  $('#burial').value = String(engine.opts.burial);
  $('#solvent').value = String(engine.opts.solvent);
  $('#search').value = String(engine.opts.searchScale);

  updateRange($('#poses'));
  updateRange($('#temperature'));
  updateRange($('#substeps'));
  updateRange($('#lj'));
  updateRange($('#es'));
  updateRange($('#burial'));
  updateRange($('#solvent'));
  updateRange($('#search'));

  $('#o-poses').textContent = Number(engine.poseCount).toLocaleString();
  $('#o-temp').textContent = `${Math.round(engine.opts.temperature)} K`;
  $('#o-substeps').textContent = String(engine.opts.substeps);
  $('#o-lj').textContent = `${engine.opts.ljScale.toFixed(2)}×`;
  $('#o-es').textContent = `${engine.opts.esScale.toFixed(2)}×`;
  $('#o-burial').textContent = `${engine.opts.burial.toFixed(2)}×`;
  $('#o-solvent').textContent = `${engine.opts.solvent.toFixed(2)}×`;
  $('#o-search').textContent = `${engine.opts.searchScale.toFixed(2)}×`;

  bindRange('poses', 'o-poses', (v) => `${Number(v).toLocaleString()}`, (v) => {
    engine.setOptions({ poseCount: Number(v) });
    const s = state.lastStat;
    if (s) els.hudQuality.textContent = `${formatCount(s.atoms)} atoms · ${formatCount(s.poses)} poses`;
    toast(`${Number(v).toLocaleString()} poses · ${formatCount(engine.recCount + engine.poseCount * engine.ligCount)} atoms`);
  }, true);

  bindRange('temperature', 'o-temp', (v) => `${Math.round(v)} K`, (v) => {
    state.cooling = false;
    state.quenching = false;
    state.quenchAnnounced = false;
    engine.setOptions({ temperature: Number(v) });
    els.temp.textContent = Math.round(v);
  });

  bindRange('substeps', 'o-substeps', (v) => String(v), (v) => {
    engine.setOptions({ substeps: Number(v) });
    els.substeps.textContent = v;
  });

  bindRange('lj', 'o-lj', (v) => `${Number(v).toFixed(2)}×`, (v) => engine.setOptions({ ljScale: Number(v) }));
  bindRange('es', 'o-es', (v) => `${Number(v).toFixed(2)}×`, (v) => engine.setOptions({ esScale: Number(v) }));
  bindRange('burial', 'o-burial', (v) => `${Number(v).toFixed(2)}×`, (v) => engine.setOptions({ burial: Number(v) }));
  bindRange('solvent', 'o-solvent', (v) => `${Number(v).toFixed(2)}×`, (v) => engine.setOptions({ solvent: Number(v) }));
  bindRange('search', 'o-search', (v) => `${Number(v).toFixed(2)}×`, (v) => engine.setOptions({ searchScale: Number(v) }));

  els.run.addEventListener('click', toggleRun);
  $('#reset').addEventListener('click', () => {
    engine.reset(true);
    state.cooling = false;
    state.quenching = false;
    state.quenchAnnounced = false;
    toast('Pose swarm re-seeded');
  });
  $('#reheat').addEventListener('click', () => {
    engine.reheat();
    engine.setOptions({ temperature: 620 });
    state.cooling = true;
    state.coolTarget = 90;
    state.coolRate = REHEAT_COOL_RATE;
    state.quenching = false;
    state.quenchAnnounced = false;
    const slider = $('#temperature');
    slider.value = '620';
    updateRange(slider);
    $('#o-temp').textContent = '620 K';
    els.temp.textContent = '620';
    toast('Reheat to 620 K · annealing');
  });
  $('#quench').addEventListener('click', () => {
    const startT = Math.max(0, engine.opts.temperature);
    state.quenching = true;
    state.quenchAnnounced = false;
    state.coolTarget = 0;
    state.cooling = startT > 0.5;
    state.coolRate = state.cooling ? startT / QUENCH_RAMP_MS : REHEAT_COOL_RATE;
    if (!state.cooling) {
      engine.setOptions({ temperature: 0 });
      const slider = $('#temperature');
      slider.value = '0';
      updateRange(slider);
      $('#o-temp').textContent = '0 K';
      els.temp.textContent = '0';
    }
    if (!state.running) toggleRun();
    els.state.textContent = 'quench';
    els.state.classList.toggle('is-paused', false);
    els.state.classList.toggle('is-converged', false);
    els.chartNote.textContent = 'quench';
    toast(state.cooling
      ? `Quench · cooling ${Math.round(startT)} → 0 K over ~${Math.round(QUENCH_RAMP_MS / 1000)} s`
      : 'Quench · 0 K · wake and migration off · watching mean |ΔE|');
  });
  $('#focus-best').addEventListener('click', () => {
    if (!engine.best) return toast('No best pose yet');
    engine.focusBest();
    engine.setOptions({ autoRotate: false });
    setSwitch('auto-rotate', false);
    toast(`Focused pose #${engine.best.index + 1} · ${engine.best.energy.toFixed(1)} kcal/mol`);
  });
  $('#reset-cam').addEventListener('click', () => {
    engine.resetCamera();
    toast('Camera reset');
  });

  const toggles = [
    ['show-receptor', 'showReceptor'],
    ['show-swarm', 'showSwarm'],
    ['show-best', 'showBest'],
    ['flex', 'torsions'],
    ['show-crystal', 'showCrystal'],
    ['auto-rotate', 'autoRotate'],
    ['sleep', 'sleep']
  ];
  for (const [id, key] of toggles) {
    const el = $('#' + id);
    el.addEventListener('click', () => {
      const next = !engine.opts[key];
      engine.setOptions({ [key]: next });
      setSwitch(id, next);
    });
  }

  $('#collapse').addEventListener('click', () => {
    document.body.classList.toggle('panel-collapsed');
    const collapsed = document.body.classList.contains('panel-collapsed');
    $('#collapse').textContent = collapsed ? '+' : '−';
    $('#collapse').setAttribute('aria-expanded', String(!collapsed));
    setTimeout(fitCanvas, 260);
  });

  $('#load-pdb').addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', async () => {
    const file = fileInput.files && fileInput.files[0];
    if (!file) return;
    try {
      loading.classList.remove('done');
      loadingText.textContent = `Parsing ${file.name} · picking largest HETATM ligand`;
      await new Promise((r) => requestAnimationFrame(r));
      const text = await file.text();
      const next = parsePDB(text);
      target = next;
      engine.setTarget(next);
      engine.resize(stage.clientWidth, stage.clientHeight, state.dpr);
      state.cooling = false;
      state.quenching = false;
      state.quenchAnnounced = false;
      updateTargetUI(next);
      updateEngineUI();
      els.hudGpu.textContent = shortGPU(engine.renderer);
      loading.classList.add('done');
      toast(`${next.pdbId} loaded · ${formatCount(next.receptor.count)} receptor atoms · ${next.ligandName}`);
    } catch (err) {
      console.error(err);
      loading.classList.add('done');
      toast(`Load failed: ${err.message}`);
    } finally {
      fileInput.value = '';
    }
  });

  // keyboard shortcut hint / hide UI
  window.addEventListener('resize', fitCanvas);
  const ro = new ResizeObserver(fitCanvas);
  ro.observe(stage);
}

function bindRange(id, out, fmt, apply, onRelease = false) {
  const input = $('#' + id);
  const output = $('#' + out);
  const run = () => {
    output.textContent = fmt(input.value);
    updateRange(input);
    if (!onRelease) apply(input.value);
  };
  input.addEventListener('input', run);
  if (onRelease) input.addEventListener('change', () => apply(input.value));
}

function updateRange(input) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  const v = Number(input.value);
  const pct = max > min ? ((v - min) / (max - min)) * 100 : 0;
  input.style.setProperty('--fill', pct + '%');
}

function setSwitch(id, on) {
  const el = $('#' + id);
  el.classList.toggle('on', on);
  el.setAttribute('aria-pressed', String(on));
}

function updateEngineUI() {
  els.substeps.textContent = engine.opts.substeps;
  els.temp.textContent = Math.round(engine.opts.temperature);
  els.cutoff.textContent = engine.opts.cutoff.toFixed(1);
  $('#o-substeps').textContent = engine.opts.substeps;
  $('#o-temp').textContent = `${Math.round(engine.opts.temperature)} K`;
  els.state.textContent = state.running ? (state.quenching ? 'quench' : 'live') : 'paused';
  els.state.classList.toggle('is-paused', !state.running);
  els.state.classList.toggle('is-converged', false);
}

const KNOWN_TARGETS = {
  '1M17': { name: 'EGFR kinase', ligand: 'erlotinib' }
};

function updateTargetUI(t) {
  const known = KNOWN_TARGETS[t.pdbId];
  const raw = known ? known.name : (t.title && t.title.length > 2 ? t.title : 'Protein–ligand complex');
  const title = raw.length > 46 ? raw.slice(0, 44) + '…' : raw;
  const lig = known ? known.ligand : t.ligandName;
  els.tName.textContent = title;
  els.tSub.textContent = `${t.pdbId} · ${lig} (${t.ligandName})`;
  els.tCounts.textContent = `${t.receptor.count.toLocaleString()} receptor atoms · ${t.ligand.count} ligand atoms · ${t.ligand.torsions ? t.ligand.torsions.count : 0} torsions`;
  els.hudTarget.textContent = known ? known.name : title;
  els.hudPdb.textContent = `${t.pdbId} · ${lig}`;
}

function toggleRun() {
  state.running = !state.running;
  els.run.classList.toggle('is-paused', !state.running);
  els.runLabel.textContent = state.running ? 'Pause' : 'Run';
  const label = state.running ? (state.quenching ? 'quench' : 'live') : 'paused';
  els.state.textContent = label;
  els.state.classList.toggle('is-paused', !state.running);
  els.state.classList.toggle('is-converged', false);
  els.chartNote.textContent = label;
  if (state.running) lastFrame = performance.now();
}

// -------------------------------------------------------------- camera ----

function bindCamera() {
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    state.dragging = true;
    state.lastX = e.clientX;
    state.lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    engine.setOptions({ autoRotate: false });
    setSwitch('auto-rotate', false);
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!state.dragging) return;
    const dx = e.clientX - state.lastX;
    const dy = e.clientY - state.lastY;
    state.lastX = e.clientX;
    state.lastY = e.clientY;
    engine.camera.yaw -= dx * 0.005;
    engine.camera.pitch = Math.max(-1.35, Math.min(1.35, engine.camera.pitch - dy * 0.005));
  });
  const end = (e) => {
    state.dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('dblclick', () => engine.resetCamera());
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const f = Math.exp(e.deltaY * 0.0011);
    engine.camera.distance = Math.max(26, Math.min(280, engine.camera.distance * f));
  }, { passive: false });
}

function bindKeys() {
  window.addEventListener('keydown', (e) => {
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); toggleRun(); }
    else if (e.key === 'r' || e.key === 'R') { engine.reset(true); toast('Pose swarm re-seeded'); }
    else if (e.key === 'f' || e.key === 'F') {
      if (engine.best) { engine.focusBest(); toast('Focused best pose'); }
    } else if (e.key === 'h' || e.key === 'H') {
      document.body.classList.toggle('hide-ui');
    } else if (e.key === 'c' || e.key === 'C') {
      const next = !engine.opts.showCrystal;
      engine.setOptions({ showCrystal: next });
      setSwitch('show-crystal', next);
    }
  });
}

// -------------------------------------------------------------- resize ----

function fitCanvas() {
  if (!engine) return;
  const r = stage.getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return;
  state.dpr = Math.min(devicePixelRatio || 1, dprCap);
  engine.resize(r.width, r.height, state.dpr);
}

// --------------------------------------------------------------- toast ----

function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toastEl.classList.remove('show'), 2800);
}

boot();
