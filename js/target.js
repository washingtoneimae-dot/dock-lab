/**
 * target.js — PDB ingest, atom typing, 3D spatial hash, ligand template.
 *
 * The receptor is static, so its grid is built once. The ligand is rigid and
 * replicated as many independent poses; each pose samples the same grid on the
 * GPU. Everything is typed arrays; no per-frame allocation happens here.
 */

export const MAX_POSES = 3072;

// Pose render targets use all eight colour attachments on the iGPU:
// pos, quat, vel, ang + 2 angle textures + 2 angular-velocity textures.
export const MAX_TORSIONS = 8;

export const ELEMENT = {
  H:  { sigma: 2.40, epsilon: 0.020, mass: 1.008,  radius: 0.24, color: [0.86, 0.90, 0.96] },
  C:  { sigma: 3.40, epsilon: 0.086, mass: 12.011, radius: 0.42, color: [0.62, 0.68, 0.78] },
  N:  { sigma: 3.25, epsilon: 0.170, mass: 14.007, radius: 0.40, color: [0.36, 0.66, 1.00] },
  O:  { sigma: 2.96, epsilon: 0.210, mass: 15.999, radius: 0.38, color: [1.00, 0.36, 0.42] },
  S:  { sigma: 3.55, epsilon: 0.250, mass: 32.060, radius: 0.48, color: [1.00, 0.78, 0.30] },
  P:  { sigma: 3.74, epsilon: 0.200, mass: 30.974, radius: 0.47, color: [1.00, 0.55, 0.20] },
  F:  { sigma: 3.00, epsilon: 0.061, mass: 18.998, radius: 0.35, color: [0.55, 1.00, 0.78] },
  CL: { sigma: 3.50, epsilon: 0.150, mass: 35.450, radius: 0.45, color: [0.50, 1.00, 0.55] },
  BR: { sigma: 3.60, epsilon: 0.200, mass: 79.904, radius: 0.47, color: [0.85, 0.45, 0.35] },
  I:  { sigma: 3.70, epsilon: 0.250, mass: 126.90, radius: 0.50, color: [0.70, 0.45, 0.85] },
  ZN: { sigma: 2.20, epsilon: 0.350, mass: 65.380, radius: 0.40, color: [0.50, 0.65, 0.85] },
  MG: { sigma: 2.10, epsilon: 0.250, mass: 24.305, radius: 0.38, color: [0.20, 0.85, 0.35] },
  CA: { sigma: 2.60, epsilon: 0.250, mass: 40.078, radius: 0.45, color: [0.50, 0.55, 0.60] },
  MN: { sigma: 2.20, epsilon: 0.300, mass: 54.938, radius: 0.40, color: [0.65, 0.50, 0.75] },
  FE: { sigma: 2.20, epsilon: 0.350, mass: 55.845, radius: 0.40, color: [0.85, 0.50, 0.20] },
  CU: { sigma: 2.20, epsilon: 0.300, mass: 63.546, radius: 0.40, color: [0.78, 0.55, 0.30] },
  NA: { sigma: 2.50, epsilon: 0.100, mass: 22.990, radius: 0.45, color: [0.65, 0.40, 0.85] },
  K:  { sigma: 3.00, epsilon: 0.100, mass: 39.098, radius: 0.50, color: [0.55, 0.35, 0.75] },
  DEFAULT: { sigma: 3.40, epsilon: 0.086, mass: 12.011, radius: 0.42, color: [0.62, 0.68, 0.78] }
};

export const ELEMENT_INDEX = {
  H: 0, C: 1, N: 2, O: 3, S: 4, P: 5, F: 6, CL: 7, BR: 8, I: 9,
  ZN: 10, MG: 11, CA: 12, MN: 13, FE: 14, CU: 15, NA: 16, K: 17, DEFAULT: 18
};

export function elementInfo(sym) {
  return ELEMENT[String(sym || '').toUpperCase()] || ELEMENT.DEFAULT;
}

// ------------------------------------------------------------------ PDB ----

function field(line, a, b) { return line.slice(a, b).trim(); }

function parseElement(line, atomName) {
  let el = field(line, 76, 78).toUpperCase().replace(/[^A-Z]/g, '');
  if (!el || !ELEMENT[el]) {
    const clean = String(atomName || '').trim().replace(/^\d+/, '');
    const two = clean.slice(0, 2).toUpperCase();
    if (ELEMENT[two]) el = two;
    else {
      const one = clean.slice(0, 1).toUpperCase();
      el = ELEMENT[one] ? one : 'C';
    }
  }
  return ELEMENT[el] ? el : 'C';
}

const WATER_RES = new Set(['HOH', 'WAT', 'TIP', 'SOL', 'DOD']);
const SOLVENT_RES = new Set([
  'NAG', 'BMA', 'MAN', 'GAL', 'FUC', // glycan sugars
  'EDO', 'PEG', 'PG4', 'GOL', 'SO4', 'PO4', 'ACT', 'DMS', 'TRS', 'FMT', 'CIT', 'MPD', 'IPA', 'ACE', 'NME', 'NH4'
]);
const METAL_RES = new Set(['ZN', 'MG', 'CA', 'MN', 'FE', 'CU', 'NA', 'K', 'NI', 'CO']);

/**
 * Parse a PDB. Receptor = ATOM records + retained cofactors/metals.
 * Ligand = chosen single drug candidate molecule instance.
 */
export function parsePDB(text, opts = {}) {
  const lines = String(text).split(/\r?\n/);
  const records = [];
  const instances = new Map(); // key -> { resName, chain, resSeq, records: [] }

  for (const line of lines) {
    if (line.length < 54) continue;
    const rec = line.slice(0, 6).trim();
    if (rec !== 'ATOM' && rec !== 'HETATM') continue;
    const alt = line[16] || ' ';
    if (alt !== ' ' && alt !== 'A') continue;
    const resName = field(line, 17, 20);
    if (!resName) continue;
    const atomName = field(line, 12, 16);
    const chain = line[21] || ' ';
    const resSeq = parseInt(field(line, 22, 26), 10) || 0;
    const x = parseFloat(field(line, 30, 38));
    const y = parseFloat(field(line, 38, 46));
    const z = parseFloat(field(line, 46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    const element = parseElement(line, atomName);
    const isHet = rec === 'HETATM';
    const atomObj = { isHet, resName, atomName, chain, resSeq, x, y, z, element };
    records.push(atomObj);

    if (isHet && !WATER_RES.has(resName)) {
      const key = `${chain}:${resSeq}:${resName}`;
      if (!instances.has(key)) {
        instances.set(key, { resName, chain, resSeq, atoms: [] });
      }
      instances.get(key).atoms.push(atomObj);
    }
  }

  // Identify the target ligand molecule instance
  let chosenKey = null;
  const specifiedRes = opts.ligandRes ? opts.ligandRes.toUpperCase() : null;

  if (specifiedRes) {
    let bestCount = 0;
    for (const [key, inst] of instances) {
      if (inst.resName.toUpperCase() === specifiedRes && inst.atoms.length > bestCount) {
        bestCount = inst.atoms.length;
        chosenKey = key;
      }
    }
  }

  if (!chosenKey) {
    // Rank candidate instances: prioritize non-solvent, non-metal organic drug candidates
    let bestScore = -1;
    for (const [key, inst] of instances) {
      const rn = inst.resName.toUpperCase();
      if (METAL_RES.has(rn)) continue;
      const isSolvent = SOLVENT_RES.has(rn);
      // Non-solvent molecules get a 1000-point boost so true drug ligands beat glycan clusters
      const score = inst.atoms.length + (isSolvent ? 0 : 1000);
      if (score > bestScore) {
        bestScore = score;
        chosenKey = key;
      }
    }
  }

  // Fallback: take largest instance if nothing was selected
  if (!chosenKey && instances.size > 0) {
    let maxAtoms = -1;
    for (const [key, inst] of instances) {
      if (inst.atoms.length > maxAtoms) {
        maxAtoms = inst.atoms.length;
        chosenKey = key;
      }
    }
  }

  const receptor = [];
  const ligand = [];
  for (const r of records) {
    const key = `${r.chain}:${r.resSeq}:${r.resName}`;
    if (r.isHet) {
      if (chosenKey && key === chosenKey) {
        ligand.push(r);
      } else if (!WATER_RES.has(r.resName.toUpperCase())) {
        // Keep cofactors (e.g. FPP), catalytic metal ions (ZN, MG, MN, FE),
        // and structural ligands as part of the rigid receptor pocket!
        receptor.push(r);
      }
    } else {
      receptor.push(r);
    }
  }

  if (!ligand.length) throw new Error('No ligand HETATM group found in this PDB.');
  if (!receptor.length) throw new Error('No protein ATOM records found in this PDB.');

  // center on the receptor centroid
  let cx = 0, cy = 0, cz = 0;
  for (const a of receptor) { cx += a.x; cy += a.y; cz += a.z; }
  cx /= receptor.length; cy /= receptor.length; cz /= receptor.length;

  let lx = 0, ly = 0, lz = 0;
  for (const a of ligand) { lx += a.x; ly += a.y; lz += a.z; }
  lx /= ligand.length; ly /= ligand.length; lz /= ligand.length;

  const nRec = receptor.length;
  const recPos = new Float32Array(nRec * 3);
  const recParam = new Float32Array(nRec * 4);
  const recElement = new Uint8Array(nRec);
  const recRes = [];
  const ca = [];

  for (let i = 0; i < nRec; i++) {
    const a = receptor[i];
    const info = elementInfo(a.element);
    const x = a.x - cx, y = a.y - cy, z = a.z - cz;
    recPos[i * 3] = x; recPos[i * 3 + 1] = y; recPos[i * 3 + 2] = z;
    recParam[i * 4] = info.sigma;
    recParam[i * 4 + 1] = info.epsilon;
    recParam[i * 4 + 2] = receptorCharge(a.resName, a.atomName, a.element);
    recParam[i * 4 + 3] = info.radius;
    recElement[i] = ELEMENT_INDEX[a.element] ?? 18;
    recRes.push(a.resName, a.atomName, a.resSeq, a.chain);
    if (a.atomName === 'CA') ca.push({ x, y, z, resSeq: a.resSeq, chain: a.chain });
  }

  // ligand local frame, centered at the crystal centroid
  const nLig = ligand.length;
  const ligOffset = new Float32Array(nLig * 3);
  const ligParam = new Float32Array(nLig * 4);
  const ligElement = new Uint8Array(nLig);
  const ligRadius = new Float32Array(nLig);
  let siteRadius = 0;
  const rawCharges = new Float32Array(nLig);
  let chargeSum = 0;

  for (let i = 0; i < nLig; i++) {
    const a = ligand[i];
    const info = elementInfo(a.element);
    const x = a.x - lx, y = a.y - ly, z = a.z - lz;
    ligOffset[i * 3] = x; ligOffset[i * 3 + 1] = y; ligOffset[i * 3 + 2] = z;
    ligParam[i * 4] = info.sigma;
    ligParam[i * 4 + 1] = info.epsilon;
    ligParam[i * 4 + 3] = info.mass;
    ligRadius[i] = info.radius;
    ligElement[i] = ELEMENT_INDEX[a.element] ?? 10;
    siteRadius = Math.max(siteRadius, Math.hypot(x, y, z));
    const q = ligandCharge(a.element);
    rawCharges[i] = q;
    chargeSum += q;
  }
  const qShift = chargeSum / nLig;
  for (let i = 0; i < nLig; i++) ligParam[i * 4 + 2] = rawCharges[i] - qShift;

  const siteCenter = [lx - cx, ly - cy, lz - cz];
  const searchRadius = siteRadius + 9.0;

  const bonds = inferBonds(ligOffset, nLig);
  const torsions = buildTorsions(ligOffset, bonds, nLig, ligParam);

  // CA trace, split at chain breaks
  const trace = [];
  for (let i = 1; i < ca.length; i++) {
    const a = ca[i - 1], b = ca[i];
    if (a.chain !== b.chain) continue;
    if (Math.abs(a.resSeq - b.resSeq) > 2) continue;
    const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
    if (dx * dx + dy * dy + dz * dz > 25) continue;
    trace.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }

  let pdbId = 'PDB';
  const hm = text.match(/^HEADER.*\s([0-9][A-Z0-9]{3})\s*$/m);
  if (hm) pdbId = hm[1].toUpperCase();
  const titleParts = [];
  const titleRe = /^TITLE\s+\d*\s*(.*)$/gm;
  let tm;
  while ((tm = titleRe.exec(text)) !== null) titleParts.push(tm[1].trim());
  const title = titleParts.join(' ').replace(/\s+/g, ' ').trim() || 'Protein–ligand complex';
  const chosenInstance = chosenKey ? instances.get(chosenKey) : null;
  const ligName = chosenInstance ? chosenInstance.resName : (specifiedRes || ligand[0].resName || 'LIG');

  return {
    pdbId,
    title,
    ligandName: ligName,
    receptor: {
      count: nRec,
      pos: recPos,
      param: recParam,
      element: recElement,
      res: recRes,
      trace: new Float32Array(trace),
      caPos: new Float32Array(ca.flatMap((a) => [a.x, a.y, a.z])),
      caCount: ca.length
    },
    ligand: {
      count: nLig,
      offset: ligOffset,
      param: ligParam,
      element: ligElement,
      radius: ligRadius,
      bonds: new Uint16Array(bonds),
      torsions,
      siteCenter,
      siteRadius,
      searchRadius
    },
    mass: ligandMass(ligParam, nLig),
    inertia: ligandInertia(ligOffset, ligParam, nLig)
  };
}

function receptorCharge(resName, atomName, element) {
  const el = String(element || '').toUpperCase();
  if (el === 'ZN' || el === 'MG' || el === 'CA' || el === 'MN' || el === 'FE' || el === 'CU' || el === 'NI') return 2.0;
  if (el === 'NA' || el === 'K') return 1.0;
  const rn = String(resName || '').toUpperCase();
  const an = String(atomName || '').toUpperCase().trim();
  if (rn === 'ASP' && (an === 'OD1' || an === 'OD2')) return -0.6;
  if (rn === 'GLU' && (an === 'OE1' || an === 'OE2')) return -0.6;
  if (rn === 'LYS' && an === 'NZ') return 1.0;
  if (rn === 'ARG' && (an === 'NE' || an === 'NH1' || an === 'NH2')) return 0.4;
  if (rn === 'HIS' && (an === 'ND1' || an === 'NE2')) return 0.25;
  if (an === 'O' || an === 'OXT' || an === 'OT1' || an === 'OT2') return -0.35;
  if (an === 'N') return -0.25;
  if (an.startsWith('O')) return -0.30;
  if (an.startsWith('N')) return -0.20;
  return 0.0;
}

function ligandCharge(element) {
  switch (element) {
    case 'O': return -0.32;
    case 'N': return -0.25;
    case 'S': return 0.05;
    case 'H': return 0.08;
    default: return 0.08;
  }
}

function ligandMass(param, n) {
  let m = 0;
  for (let i = 0; i < n; i++) m += param[i * 4 + 3];
  return m;
}

function ligandInertia(offset, param, n) {
  let I = 0;
  for (let i = 0; i < n; i++) {
    const x = offset[i * 3], y = offset[i * 3 + 1], z = offset[i * 3 + 2];
    I += param[i * 4 + 3] * (x * x + y * y + z * z);
  }
  return Math.max(I, 1);
}

export function inferBonds(offset, n) {
  const bonds = [];
  const cut2 = 1.9 * 1.9;
  for (let i = 0; i < n; i++) {
    const ax = offset[i * 3], ay = offset[i * 3 + 1], az = offset[i * 3 + 2];
    for (let j = i + 1; j < n; j++) {
      const bx = offset[j * 3], by = offset[j * 3 + 1], bz = offset[j * 3 + 2];
      const dx = ax - bx, dy = ay - by, dz = az - bz;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > 0.16 && d2 < cut2) bonds.push(i, j);
    }
  }
  return bonds;
}

// ------------------------------------------------------------ torsions ----

function bondKey(a, b) {
  return a < b ? a * 65536 + b : b * 65536 + a;
}

/**
 * Connected component reachable from `start` after removing the a–b edge.
 * `seen[otherEnd]` being true means the removed edge was in a ring.
 */
function componentMask(adj, n, start, exA, exB) {
  const seen = new Uint8Array(n);
  const list = [];
  const stack = [start];
  seen[start] = 1;
  while (stack.length) {
    const u = stack.pop();
    list.push(u);
    const nb = adj[u];
    for (let i = 0; i < nb.length; i++) {
      const v = nb[i];
      if ((u === exA && v === exB) || (u === exB && v === exA)) continue;
      if (!seen[v]) { seen[v] = 1; stack.push(v); }
    }
  }
  return { seen, list };
}

function minAngleAt(offset, adj, a, b) {
  const ax = offset[a * 3], ay = offset[a * 3 + 1], az = offset[a * 3 + 2];
  let bx = offset[b * 3] - ax, by = offset[b * 3 + 1] - ay, bz = offset[b * 3 + 2] - az;
  const bl = Math.hypot(bx, by, bz) || 1;
  bx /= bl; by /= bl; bz /= bl;
  let best = Math.PI;
  for (const c of adj[a]) {
    if (c === b) continue;
    let cx = offset[c * 3] - ax, cy = offset[c * 3 + 1] - ay, cz = offset[c * 3 + 2] - az;
    const cl = Math.hypot(cx, cy, cz) || 1;
    cx /= cl; cy /= cl; cz /= cl;
    const dot = Math.max(-1, Math.min(1, bx * cx + by * cy + bz * cz));
    const ang = Math.acos(dot);
    if (ang < best) best = ang;
  }
  return best;
}

/**
 * Build a rooted torsion tree.
 *
 * A bond is rotatable when it is a bridge (removing it disconnects the ligand),
 * both endpoints are heavy atoms with at least one other neighbour, the
 * smallest side has >= 2 atoms (drops terminal methyl spins), and neither
 * endpoint is collinear (drops alkyne-type bonds). Candidates are ranked by the
 * size of their smaller side, so the eight slots go to bonds that move the
 * largest pieces of the ligand.
 *
 * Each torsion rotates the side away from the ligand's largest rigid core. The
 * resulting subtrees are either disjoint or nested, which lets the GPU apply
 * ancestors before descendants with a per-atom bitmask.
 */
export function buildTorsions(offset, bonds, n, param) {
  const adj = Array.from({ length: n }, () => []);
  for (let b = 0; b < bonds.length; b += 2) {
    const a = bonds[b], c = bonds[b + 1];
    adj[a].push(c);
    adj[c].push(a);
  }

  const candidates = [];
  for (let b = 0; b < bonds.length; b += 2) {
    const a = bonds[b], c = bonds[b + 1];
    if (adj[a].length < 2 || adj[c].length < 2) continue;
    const sideC = componentMask(adj, n, c, a, c);
    if (sideC.seen[a]) continue; // ring bond
    const sideA = componentMask(adj, n, a, a, c);
    const minSide = Math.min(sideA.list.length, sideC.list.length);
    if (minSide < 2) continue;
    // Collinear bonds (e.g. alkyne C≡C) have no meaningful dihedral.
    if (minAngleAt(offset, adj, a, c) > 2.97 || minAngleAt(offset, adj, c, a) > 2.97) continue;
    candidates.push({ a, c, sideA: sideA.list, sideC: sideC.list, impact: minSide });
  }

  candidates.sort((x, y) => y.impact - x.impact);
  const chosen = candidates.slice(0, MAX_TORSIONS);
  if (!chosen.length) {
    return {
      count: 0,
      axis: new Float32Array(MAX_TORSIONS * 3),
      pivot: new Float32Array(MAX_TORSIONS * 3),
      invInertia: new Float32Array(MAX_TORSIONS),
      atomMask: new Uint16Array(n),
      subtrees: [],
      parent: [],
      depth: []
    };
  }

  // The rigid core is the largest component left after removing every chosen
  // rotatable bond. Every torsion then rotates the branch away from that core.
  const removed = new Set();
  for (const t of chosen) removed.add(bondKey(t.a, t.c));
  const visited = new Uint8Array(n);
  let core = null;
  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    const seen = new Uint8Array(n);
    const list = [];
    const stack = [i];
    seen[i] = 1;
    while (stack.length) {
      const u = stack.pop();
      list.push(u);
      for (const v of adj[u]) {
        if (removed.has(bondKey(u, v))) continue;
        if (!seen[v]) { seen[v] = 1; stack.push(v); }
      }
    }
    for (let j = 0; j < n; j++) if (seen[j]) visited[j] = 1;
    if (!core || list.length > core.length) core = list;
  }
  const coreSet = new Uint8Array(n);
  for (const i of core) coreSet[i] = 1;
  const coreRoot = core[0];

  const items = [];
  for (const t of chosen) {
    const sideC = componentMask(adj, n, t.c, t.a, t.c);
    const childIsC = !sideC.seen[coreRoot];
    const child = childIsC ? t.c : t.a;
    const parentAtom = childIsC ? t.a : t.c;
    const subtree = childIsC ? sideC.list : t.sideA;
    const px = offset[parentAtom * 3], py = offset[parentAtom * 3 + 1], pz = offset[parentAtom * 3 + 2];
    let ax = offset[child * 3] - px;
    let ay = offset[child * 3 + 1] - py;
    let az = offset[child * 3 + 2] - pz;
    const al = Math.hypot(ax, ay, az) || 1;
    ax /= al; ay /= al; az /= al;

    let inertia = 0;
    for (const i of subtree) {
      const rx = offset[i * 3] - px, ry = offset[i * 3 + 1] - py, rz = offset[i * 3 + 2] - pz;
      const proj = rx * ax + ry * ay + rz * az;
      inertia += param[i * 4 + 3] * Math.max(0, rx * rx + ry * ry + rz * rz - proj * proj);
    }

    // Immediate parent = smallest chosen subtree containing the pivot atom.
    let parent = -1;
    let parentSize = Infinity;
    for (let j = 0; j < chosen.length; j++) {
      if (chosen[j] === t) continue;
      const sj = componentMask(adj, n, chosen[j].c, chosen[j].a, chosen[j].c);
      const other = componentMask(adj, n, chosen[j].a, chosen[j].a, chosen[j].c);
      const sub = !sj.seen[coreRoot] ? sj : other;
      if (sub.seen[parentAtom] && sub.list.length < parentSize) {
        parent = j;
        parentSize = sub.list.length;
      }
    }

    items.push({ child, parentAtom, subtree, ax, ay, az, px, py, pz, inertia, parent, depth: 0 });
  }

  // Resolve depths, then sort parents before children. Equal-depth torsions are
  // disjoint, so their relative order is irrelevant.
  function depthOf(i, guard = 0) {
    if (items[i].depth || guard > MAX_TORSIONS) return items[i].depth;
    items[i].depth = items[i].parent < 0 ? 0 : depthOf(items[i].parent, guard + 1) + 1;
    return items[i].depth;
  }
  for (let i = 0; i < items.length; i++) depthOf(i);
  const order = items.map((_, i) => i).sort((a, b) => items[a].depth - items[b].depth);
  const rank = new Int32Array(items.length);
  for (let i = 0; i < order.length; i++) rank[order[i]] = i;

  const count = items.length;
  const axis = new Float32Array(MAX_TORSIONS * 3);
  const pivot = new Float32Array(MAX_TORSIONS * 3);
  const invInertia = new Float32Array(MAX_TORSIONS);
  const atomMask = new Uint16Array(n);
  const subtrees = [];
  const parent = new Int32Array(count);
  const depth = new Int32Array(count);

  for (let newIdx = 0; newIdx < count; newIdx++) {
    const it = items[order[newIdx]];
    axis[newIdx * 3] = it.ax;
    axis[newIdx * 3 + 1] = it.ay;
    axis[newIdx * 3 + 2] = it.az;
    pivot[newIdx * 3] = it.px;
    pivot[newIdx * 3 + 1] = it.py;
    pivot[newIdx * 3 + 2] = it.pz;
    invInertia[newIdx] = 1 / Math.max(it.inertia, 0.5);
    subtrees.push(new Uint16Array(it.subtree));
    parent[newIdx] = it.parent < 0 ? -1 : rank[it.parent];
    depth[newIdx] = it.depth;
    for (const i of it.subtree) atomMask[i] |= 1 << newIdx;
  }

  return { count, axis, pivot, invInertia, atomMask, subtrees, parent, depth, core: new Uint16Array(core) };
}

/**
 * Apply the torsion chain to every atom, matching the GLSL implementation.
 * Row-major 3x3 rotation + translation; used for clash checks, RMSD and tests.
 */
export function torsionTransformOffsets(target, angles, out) {
  const n = target.ligand.count;
  const off = target.ligand.offset;
  const t = target.ligand.torsions;
  if (!t || !t.count || !angles || angles.length < t.count) {
    if (out) out.set(off);
    return out || off;
  }
  if (!out) out = new Float32Array(n * 3);
  const count = t.count;
  for (let i = 0; i < n; i++) {
    let rx = off[i * 3], ry = off[i * 3 + 1], rz = off[i * 3 + 2];
    let r00 = 1, r01 = 0, r02 = 0, r10 = 0, r11 = 1, r12 = 0, r20 = 0, r21 = 0, r22 = 1;
    let tx = 0, ty = 0, tz = 0;
    const mask = t.atomMask[i];
    for (let k = 0; k < count; k++) {
      if (!(mask & (1 << k))) continue;
      const ax0 = t.axis[k * 3], ay0 = t.axis[k * 3 + 1], az0 = t.axis[k * 3 + 2];
      const ax = r00 * ax0 + r01 * ay0 + r02 * az0;
      const ay = r10 * ax0 + r11 * ay0 + r12 * az0;
      const az = r20 * ax0 + r21 * ay0 + r22 * az0;
      const px0 = t.pivot[k * 3], py0 = t.pivot[k * 3 + 1], pz0 = t.pivot[k * 3 + 2];
      const px = r00 * px0 + r01 * py0 + r02 * pz0 + tx;
      const py = r10 * px0 + r11 * py0 + r12 * pz0 + ty;
      const pz = r20 * px0 + r21 * py0 + r22 * pz0 + tz;
      const a = angles[k];
      const c = Math.cos(a), s = Math.sin(a), ic = 1 - c;
      let vx = rx - px, vy = ry - py, vz = rz - pz;
      const dot = ax * vx + ay * vy + az * vz;
      const cx = ay * vz - az * vy;
      const cy = az * vx - ax * vz;
      const cz = ax * vy - ay * vx;
      vx = vx * c + cx * s + ax * dot * ic;
      vy = vy * c + cy * s + ay * dot * ic;
      vz = vz * c + cz * s + az * dot * ic;
      rx = vx + px; ry = vy + py; rz = vz + pz;

      const xx = ax * ax * ic + c, xy = ax * ay * ic - az * s, xz = ax * az * ic + ay * s;
      const yx = ay * ax * ic + az * s, yy = ay * ay * ic + c, yz = ay * az * ic - ax * s;
      const zx = az * ax * ic - ay * s, zy = az * ay * ic + ax * s, zz = az * az * ic + c;
      const n00 = xx * r00 + xy * r10 + xz * r20;
      const n01 = xx * r01 + xy * r11 + xz * r21;
      const n02 = xx * r02 + xy * r12 + xz * r22;
      const n10 = yx * r00 + yy * r10 + yz * r20;
      const n11 = yx * r01 + yy * r11 + yz * r21;
      const n12 = yx * r02 + yy * r12 + yz * r22;
      const n20 = zx * r00 + zy * r10 + zz * r20;
      const n21 = zx * r01 + zy * r11 + zz * r21;
      const n22 = zx * r02 + zy * r12 + zz * r22;
      r00 = n00; r01 = n01; r02 = n02;
      r10 = n10; r11 = n11; r12 = n12;
      r20 = n20; r21 = n21; r22 = n22;
      const ux = tx - px, uy = ty - py, uz = tz - pz;
      tx = xx * ux + xy * uy + xz * uz + px;
      ty = yx * ux + yy * uy + yz * uz + py;
      tz = zx * ux + zy * uy + zz * uz + pz;
    }
    out[i * 3] = rx; out[i * 3 + 1] = ry; out[i * 3 + 2] = rz;
  }
  return out;
}

// -------------------------------------------------------------- 3D grid ----

/**
 * Uniform voxel hash with counting sort. Cell size must be >= cutoff/2 so a
 * 3x3x3 neighbourhood covers the interaction cutoff.
 */
export function buildGrid(pos, count, cellSize = 4.5, margin = 10.0) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < count; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  minX -= margin; minY -= margin; minZ -= margin;
  maxX += margin; maxY += margin; maxZ += margin;

  const nx = Math.max(1, Math.ceil((maxX - minX) / cellSize));
  const ny = Math.max(1, Math.ceil((maxY - minY) / cellSize));
  const nz = Math.max(1, Math.ceil((maxZ - minZ) / cellSize));
  const cellCount = nx * ny * nz;

  const counts = new Uint32Array(cellCount);
  const cellOf = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const ix = Math.min(nx - 1, Math.max(0, Math.floor((pos[i * 3] - minX) / cellSize)));
    const iy = Math.min(ny - 1, Math.max(0, Math.floor((pos[i * 3 + 1] - minY) / cellSize)));
    const iz = Math.min(nz - 1, Math.max(0, Math.floor((pos[i * 3 + 2] - minZ) / cellSize)));
    const c = ix + iy * nx + iz * nx * ny;
    cellOf[i] = c;
    counts[c]++;
  }

  const starts = new Uint32Array(cellCount);
  let sum = 0;
  for (let c = 0; c < cellCount; c++) { starts[c] = sum; sum += counts[c]; }
  const cursor = starts.slice();
  const indices = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const c = cellOf[i];
    indices[cursor[c]++] = i;
  }

  const meta = new Float32Array(cellCount * 4);
  for (let c = 0; c < cellCount; c++) {
    meta[c * 4] = starts[c];
    meta[c * 4 + 1] = counts[c];
  }

  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    dims: [nx, ny, nz],
    cellSize,
    cellCount,
    starts,
    counts,
    indices,
    meta
  };
}

export function cellIndex(grid, x, y, z) {
  const ix = Math.floor((x - grid.min[0]) / grid.cellSize);
  const iy = Math.floor((y - grid.min[1]) / grid.cellSize);
  const iz = Math.floor((z - grid.min[2]) / grid.cellSize);
  if (ix < 0 || iy < 0 || iz < 0 || ix >= grid.dims[0] || iy >= grid.dims[1] || iz >= grid.dims[2]) return -1;
  return ix + iy * grid.dims[0] + iz * grid.dims[0] * grid.dims[1];
}

export function clashes(grid, recPos, world, radius = 2.0) {
  const r2cut = radius * radius;
  const nx = grid.dims[0], ny = grid.dims[1];
  const ix = Math.floor((world[0] - grid.min[0]) / grid.cellSize);
  const iy = Math.floor((world[1] - grid.min[1]) / grid.cellSize);
  const iz = Math.floor((world[2] - grid.min[2]) / grid.cellSize);
  for (let dz = -1; dz <= 1; dz++) {
    const z = iz + dz;
    if (z < 0 || z >= grid.dims[2]) continue;
    for (let dy = -1; dy <= 1; dy++) {
      const y = iy + dy;
      if (y < 0 || y >= ny) continue;
      for (let dx = -1; dx <= 1; dx++) {
        const x = ix + dx;
        if (x < 0 || x >= nx) continue;
        const c = x + y * nx + z * nx * ny;
        const start = grid.starts[c], count = grid.counts[c];
        for (let k = 0; k < count; k++) {
          const idx = grid.indices[start + k];
          const rx = recPos[idx * 3] - world[0];
          const ry = recPos[idx * 3 + 1] - world[1];
          const rz = recPos[idx * 3 + 2] - world[2];
          if (rx * rx + ry * ry + rz * rz < r2cut) return true;
        }
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------- poses ----

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomQuat(rnd) {
  const u1 = rnd(), u2 = rnd(), u3 = rnd();
  const s1 = Math.sqrt(1 - u1), s2 = Math.sqrt(u1);
  return [
    s1 * Math.sin(2 * Math.PI * u2),
    s1 * Math.cos(2 * Math.PI * u2),
    s2 * Math.sin(2 * Math.PI * u3),
    s2 * Math.cos(2 * Math.PI * u3)
  ];
}

function quatMul(a, b) {
  return [
    a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
    a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
    a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]
  ];
}

function quatNormalize(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}

export function randomPerturbedQuat(rnd, maxAngle) {
  const axis = [rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1];
  const al = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const ang = (rnd() * 2 - 1) * maxAngle;
  const s = Math.sin(ang / 2);
  const q = [axis[0] / al * s, axis[1] / al * s, axis[2] / al * s, Math.cos(ang / 2)];
  return quatNormalize(q);
}

function rotate(q, v) {
  const x = q[0], y = q[1], z = q[2], w = q[3];
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx)
  ];
}

function sampleTorsions(rnd, count, nearCrystal, out) {
  const TWO_PI_3 = (2 * Math.PI) / 3;
  for (let k = 0; k < count; k++) {
    if (nearCrystal) {
      out[k] = (rnd() * 2 - 1) * 0.28;
    } else {
      const pick = rnd();
      const base = pick < 0.55 ? 0 : pick < 0.775 ? TWO_PI_3 : -TWO_PI_3;
      out[k] = base + (rnd() * 2 - 1) * 0.35;
    }
  }
  return out;
}

/**
 * Build initial pose buffers. A small fraction is seeded near the crystal pose
 * so redocking has a guaranteed reference; the rest is a random search.
 * Torsion angles are sampled near staggered rotamers; the crystal conformation
 * is the zero-angle reference for every torsion.
 */
export function makeInitialPoses(target, count, seed = 0x9e3779b9, seedFraction = 0.06) {
  const rnd = mulberry32(seed);
  const n = target.ligand.count;
  const grid = buildGrid(target.receptor.pos, target.receptor.count, 8.0, 10.0);
  const pos = new Float32Array(MAX_POSES * 4);
  const quat = new Float32Array(MAX_POSES * 4);
  const vel = new Float32Array(MAX_POSES * 4);
  const ang = new Float32Array(MAX_POSES * 4);
  const torsA = new Float32Array(MAX_POSES * 4);
  const torsB = new Float32Array(MAX_POSES * 4);
  const torsVelA = new Float32Array(MAX_POSES * 4);
  const torsVelB = new Float32Array(MAX_POSES * 4);

  const sc = target.ligand.siteCenter;
  const R = target.ligand.searchRadius;
  const off = target.ligand.offset;
  const recPos = target.receptor.pos;
  const seedCount = Math.floor(count * seedFraction);
  const tors = target.ligand.torsions;
  const nTors = tors ? tors.count : 0;
  const angles = new Float32Array(MAX_TORSIONS);
  const local = new Float32Array(n * 3);

  const world = [0, 0, 0];
  for (let i = 0; i < count; i++) {
    let p, q;
    const seeded = i < seedCount;
    if (seeded) {
      const rr = Math.pow(rnd(), 1 / 3) * 2.2;
      const th = rnd() * Math.PI * 2, ph = Math.acos(2 * rnd() - 1);
      p = [
        sc[0] + rr * Math.sin(ph) * Math.cos(th),
        sc[1] + rr * Math.sin(ph) * Math.sin(th),
        sc[2] + rr * Math.cos(ph)
      ];
      q = randomPerturbedQuat(rnd, 0.55);
    } else {
      let ok = false;
      for (let attempt = 0; attempt < 24 && !ok; attempt++) {
        const rr = R * Math.pow(rnd(), 1 / 3);
        const th = rnd() * Math.PI * 2, ph = Math.acos(2 * rnd() - 1);
        p = [
          sc[0] + rr * Math.sin(ph) * Math.cos(th),
          sc[1] + rr * Math.sin(ph) * Math.sin(th),
          sc[2] + rr * Math.cos(ph)
        ];
        q = randomQuat(rnd);
        sampleTorsions(rnd, nTors, seeded, angles);
        if (nTors) torsionTransformOffsets(target, angles, local);
        else local.set(off);
        ok = true;
        for (let a = 0; a < n; a++) {
          const r = rotate(q, [local[a * 3], local[a * 3 + 1], local[a * 3 + 2]]);
          world[0] = p[0] + r[0]; world[1] = p[1] + r[1]; world[2] = p[2] + r[2];
          if (clashes(grid, recPos, world, 2.05)) { ok = false; break; }
        }
      }
      if (!p) p = [sc[0], sc[1], sc[2]];
      if (!q) q = [0, 0, 0, 1];
    }
    if (seeded) sampleTorsions(rnd, nTors, true, angles);
    pos[i * 4] = p[0]; pos[i * 4 + 1] = p[1]; pos[i * 4 + 2] = p[2]; pos[i * 4 + 3] = 0;
    quat[i * 4] = q[0]; quat[i * 4 + 1] = q[1]; quat[i * 4 + 2] = q[2]; quat[i * 4 + 3] = q[3];
    vel[i * 4] = 0; vel[i * 4 + 1] = 0; vel[i * 4 + 2] = 0; vel[i * 4 + 3] = 0;
    ang[i * 4] = 0; ang[i * 4 + 1] = 0; ang[i * 4 + 2] = 0; ang[i * 4 + 3] = 1;
    for (let k = 0; k < 4; k++) {
      torsA[i * 4 + k] = k < nTors ? angles[k] : 0;
      torsB[i * 4 + k] = 4 + k < nTors ? angles[4 + k] : 0;
    }
  }

  return { pos, quat, vel, ang, torsA, torsB, torsVelA, torsVelB, grid };
}

export function crystalPose(target) {
  const nTors = target.ligand.torsions ? target.ligand.torsions.count : 0;
  return {
    pos: target.ligand.siteCenter.slice(),
    quat: [0, 0, 0, 1],
    tors: new Float32Array(nTors)
  };
}

export function rmsdToCrystal(target, posePos, poseQuat, poseTors) {
  const n = target.ligand.count;
  const off = target.ligand.offset;
  const sc = target.ligand.siteCenter;
  const tor = target.ligand.torsions;
  const local = poseTors && tor && poseTors.length >= tor.count
    ? torsionTransformOffsets(target, poseTors)
    : off;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const v = [local[i * 3], local[i * 3 + 1], local[i * 3 + 2]];
    const r = rotate(poseQuat, v);
    const dx = posePos[0] + r[0] - (sc[0] + off[i * 3]);
    const dy = posePos[1] + r[1] - (sc[1] + off[i * 3 + 1]);
    const dz = posePos[2] + r[2] - (sc[2] + off[i * 3 + 2]);
    sum += dx * dx + dy * dy + dz * dz;
  }
  return Math.sqrt(sum / n);
}
