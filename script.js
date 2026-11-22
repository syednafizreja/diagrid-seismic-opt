import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

// ----------------------------- Constants -----------------------------
const COLORS = {
  chalk: 0xf8f9fb, steel: 0x143055, cyan: 0x0ea5c6, crimson: 0xc2410c,
  glulam: 0xcaa472, steelDark: 0x33424f, trunk: 0x3b4a58, fuseNiti: 0x0ea5c6, fuseSteel: 0xc2410c,
};
const HEX = { steel: '#143055', cyan: '#0ea5c6', crimson: '#c2410c', muted: '#94a3b8', grid: '#d9e0e8' };
const DATA_FILES = {
  gp: 'data/gp_surrogate.json',
  pareto: 'data/pareto_canopies.json',
  diff: 'data/diff_id_uncertainty.json',
};
const TARGETS = ['peak_drift_pct', 'abs_residual_drift_pct', 'grav_defl_mm', 'c_total_per_m2'];
const KPI_DECIMALS = { peak_drift_pct: 2, abs_residual_drift_pct: 3, grav_defl_mm: 1, c_total_per_m2: 1 };
const PARAM_ORDER = ['E', 'sigma_Ms', 'sigma_Mf', 'sigma_As', 'sigma_Af', 'eps_L'];
const SLIDERS = [
  { key: 'span', label: 'Clear span (m)', min: 14, max: 28, step: 0.1, value: 20, decimals: 1 },
  { key: 'arch_rise_ratio', label: 'Arch rise ratio', min: 0.08, max: 0.22, step: 0.005, value: 0.15, decimals: 3 },
  { key: 'diagrid_divisions', label: 'Diagrid divisions', min: 6, max: 12, step: 2, value: 8, decimals: 0 },
  { key: 'glulam_depth', label: 'Glulam depth (m)', min: 0.22, max: 0.48, step: 0.01, value: 0.35, decimals: 2 },
  { key: 'niti_fuse_area', label: 'NiTi fuse area (mm²)', min: 60, max: 260, step: 5, value: 100, decimals: 0 },
];
const GROUP_STYLE = {
  GLULAM_DIAGRID: { radius: 0.055, color: COLORS.glulam, roughness: 0.85, metalness: 0.0 },
  STEEL_EDGE: { radius: 0.085, color: COLORS.steelDark, roughness: 0.4, metalness: 0.8 },
  TREE_TRUNK: { radius: 0.16, color: COLORS.trunk, roughness: 0.4, metalness: 0.85 },
  TREE_BRANCH: { radius: 0.09, color: COLORS.trunk, roughness: 0.4, metalness: 0.85 },
  SEISMIC_FUSE: { radius: 0.075, color: COLORS.cyan, roughness: 0.35, metalness: 0.6 },
};

// ----------------------------- State -----------------------------
const state = {
  mode: 'pareto',
  displayMode: 'architectural',
  chartTab: 'hysteresis',
  fuse: 'niti',
  data: { gp: null, pareto: null, diff: null },
  selectedDesign: null,
  model: null, // { nodes: [[x,y,z],...], elements: [{i,j,group,util}], nodeIds: [] }
  playing: false,
  playFrame: 0,
  playClock: 0,
};

async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status}). Serve the repo root over http:// (see README).`);
  return res.json();
}

// ----------------- Pure-JS Matérn-5/2 GP (matches sklearn) -----------------
// Replicates tests/test_surrogate_gp.py::_browser_matern_prediction exactly.
function gpPredict(caseData, target, features) {
  const model = caseData.models[target];
  const mean = caseData.scaler.mean;
  const scale = caseData.scaler.scale;
  const ls = model.length_scale;
  const X = caseData.X_train_standardized;
  const n = X.length;
  const dim = features.length;
  const xs = new Float64Array(dim);
  for (let j = 0; j < dim; j++) xs[j] = (features[j] - mean[j]) / scale[j];
  const c = model.constant_value;
  const sq5 = Math.sqrt(5);
  const k = new Float64Array(n);
  for (let a = 0; a < n; a++) {
    const Xa = X[a];
    let s = 0;
    for (let j = 0; j < dim; j++) { const d = (Xa[j] - xs[j]) / ls[j]; s += d * d; }
    const r = Math.sqrt(s);
    const t = sq5 * r;
    k[a] = c * (1 + t + (5 * r * r) / 3) * Math.exp(-t);
  }
  let mu = 0;
  for (let a = 0; a < n; a++) mu += k[a] * model.alpha[a];
  // Forward substitution: solve L y = k, where L = cholesky (lower-triangular).
  const L = model.cholesky;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = k[i];
    const Li = L[i];
    for (let j = 0; j < i; j++) sum -= Li[j] * y[j];
    y[i] = sum / Li[i];
  }
  let proj = 0;
  for (let i = 0; i < n; i++) proj += y[i] * y[i];
  const variance = c + model.noise_level - proj;
  return { mean: mu, std: Math.sqrt(Math.max(variance, 0)) };
}

function sliderFeatures() {
  return SLIDERS.map((s) => s.value);
}

// ----------------- Geometry (JS port of canopy_geometry) -----------------
function buildCanopy(params) {
  const span = params.span;
  const archRise = params.arch_rise_ratio * span;
  const nx = Math.round(params.diagrid_divisions);
  const nz = Math.max(4, Math.floor(nx / 2));
  const width = 0.5 * span;
  const nodes = [];
  const idOf = {};
  const roofIds = [];
  for (let j = 0; j <= nz; j++) {
    const z = -width / 2 + (j * width) / nz;
    for (let i = 0; i <= nx; i++) {
      const x = -span / 2 + (i * span) / nx;
      const y = 4.5 + archRise * (1 - (2 * x / span) ** 2) * (1 - 0.5 * (2 * z / width) ** 2);
      const id = nodes.length;
      nodes.push([x, y, z]);
      roofIds.push(id);
      idOf[`${i},${j}`] = id;
    }
  }
  const elements = [];
  const add = (group, a, b) => elements.push({ i: a, j: b, group });
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      add('GLULAM_DIAGRID', idOf[`${i},${j}`], idOf[`${i + 1},${j + 1}`]);
      add('GLULAM_DIAGRID', idOf[`${i + 1},${j}`], idOf[`${i},${j + 1}`]);
    }
  }
  for (const i of [0, nx]) for (let j = 0; j < nz; j++) add('STEEL_EDGE', idOf[`${i},${j}`], idOf[`${i},${j + 1}`]);
  for (const j of [0, nz]) for (let i = 0; i < nx; i++) add('STEEL_EDGE', idOf[`${i},${j}`], idOf[`${i + 1},${j}`]);
  const colPos = [
    [-0.32 * span, -0.25 * width], [-0.32 * span, 0.25 * width],
    [0.32 * span, -0.25 * width], [0.32 * span, 0.25 * width],
  ];
  const yb = 0.55 * 4.5;
  for (const [xc, zc] of colPos) {
    const bif = nodes.length; nodes.push([xc, yb, zc]);
    const ground = nodes.length; nodes.push([xc, 0, zc]);
    add('TREE_TRUNK', ground, bif);
    const nearest = roofIds
      .map((id) => ({ id, d: (nodes[id][0] - xc) ** 2 + (nodes[id][2] - zc) ** 2 }))
      .sort((a, b) => a.d - b.d).slice(0, 4);
    for (const { id } of nearest) add('TREE_BRANCH', bif, id);
    for (const xa of [xc - 1.5, xc + 1.5]) {
      const fg = nodes.length; nodes.push([xa, 0, zc]);
      add('SEISMIC_FUSE', bif, fg);
    }
  }
  return { nodes, elements };
}

function designToModel(design) {
  return {
    nodes: design.nodes.map((n) => n.xyz_m),
    elements: design.elements.map((e) => ({
      i: e.node_ids[0], j: e.node_ids[1], group: e.group_tag, util: e.group_peak_static_utilization,
    })),
  };
}

// ----------------------------- Three.js scene -----------------------------
let renderer, scene, camera, controls, structureGroup, lineMesh, nodeMesh;
const groupMeshes = {}; // group -> { mesh, els }
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _mid = new THREE.Vector3();
const _dir = new THREE.Vector3(), _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3(), _mat = new THREE.Matrix4(), _up = new THREE.Vector3(0, 1, 0);
const _col = new THREE.Color();

function setElementMatrix(mesh, index, a, b, radius) {
  _a.fromArray(a); _b.fromArray(b);
  _mid.addVectors(_a, _b).multiplyScalar(0.5);
  _dir.subVectors(_b, _a);
  const len = _dir.length() || 1e-6;
  _dir.normalize();
  _quat.setFromUnitVectors(_up, _dir);
  _scale.set(radius, len, radius);
  _mat.compose(_mid, _quat, _scale);
  mesh.setMatrixAt(index, _mat);
}

function initScene() {
  const container = document.getElementById('viewport');
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(COLORS.chalk);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 2000);
  camera.position.set(30, 22, 34);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 4, 0);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x8a97a5, 0.6));
  const dir = new THREE.DirectionalLight(0xffffff, 1.5);
  dir.position.set(24, 36, 18);
  dir.castShadow = true;
  dir.shadow.mapSize.set(2048, 2048);
  const sc = dir.shadow.camera;
  sc.left = -45; sc.right = 45; sc.top = 45; sc.bottom = -45; sc.near = 1; sc.far = 120;
  scene.add(dir);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(500, 500),
    new THREE.MeshStandardMaterial({ color: 0xeceff4, roughness: 1.0, metalness: 0.0 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = -0.02;
  ground.receiveShadow = true;
  scene.add(ground);

  structureGroup = new THREE.Group();
  scene.add(structureGroup);
  window.addEventListener('resize', onResize);
}

function onResize() {
  const container = document.getElementById('viewport');
  if (!container.clientWidth) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

// ----------------- Structure meshes (instanced) -----------------
const DISPLACEMENT_SCALE = 40; // visual amplification of real OpenSeesPy drift for legibility

function disposeStructure() {
  for (const g of Object.keys(groupMeshes)) {
    const { mesh } = groupMeshes[g];
    structureGroup.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose();
    delete groupMeshes[g];
  }
  for (const m of [lineMesh, nodeMesh]) {
    if (m) { structureGroup.remove(m); m.geometry.dispose(); m.material.dispose(); }
  }
  lineMesh = null; nodeMesh = null;
}

function rebuildStructure() {
  disposeStructure();
  if (!state.model) return;
  const nodes = state.model.nodes;
  const byGroup = {};
  for (const e of state.model.elements) (byGroup[e.group] = byGroup[e.group] || []).push(e);
  for (const [group, els] of Object.entries(byGroup)) {
    const style = GROUP_STYLE[group];
    const geom = new THREE.CylinderGeometry(1, 1, 1, 12, 1);
    const mat = new THREE.MeshStandardMaterial({ color: style.color, roughness: style.roughness, metalness: style.metalness });
    const mesh = new THREE.InstancedMesh(geom, mat, els.length);
    mesh.castShadow = true; mesh.receiveShadow = true;
    els.forEach((e, idx) => {
      setElementMatrix(mesh, idx, nodes[e.i], nodes[e.j], style.radius);
      mesh.setColorAt(idx, _col.set(0xffffff));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    structureGroup.add(mesh);
    groupMeshes[group] = { mesh, els };
  }
  const positions = [];
  for (const e of state.model.elements) positions.push(...nodes[e.i], ...nodes[e.j]);
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  lineMesh = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: COLORS.steel }));
  structureGroup.add(lineMesh);
  nodeMesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(0.12, 10, 10),
    new THREE.MeshStandardMaterial({ color: COLORS.steel }),
    nodes.length
  );
  nodes.forEach((p, idx) => { _mat.makeTranslation(p[0], p[1], p[2]); nodeMesh.setMatrixAt(idx, _mat); });
  nodeMesh.instanceMatrix.needsUpdate = true;
  structureGroup.add(nodeMesh);
  applyDisplayMode();
}

function utilizationColor(u) {
  const t = Math.min(Math.max((u || 0) / 1.2, 0), 1);
  const c0 = new THREE.Color(0x2b6cb0), c1 = new THREE.Color(0xd97706), c2 = new THREE.Color(0xc2410c);
  return t < 0.5 ? c0.lerp(c1, t / 0.5) : c1.lerp(c2, (t - 0.5) / 0.5);
}

function fuseBaseColor() { return state.fuse === 'niti' ? COLORS.fuseNiti : COLORS.fuseSteel; }

function applyDisplayMode() {
  const mode = state.displayMode;
  if (lineMesh) lineMesh.visible = mode === 'wireframe' || mode === 'technical';
  if (nodeMesh) nodeMesh.visible = mode === 'technical';
  for (const [group, entry] of Object.entries(groupMeshes)) {
    const { mesh, els } = entry;
    mesh.visible = mode !== 'wireframe';
    const base = group === 'SEISMIC_FUSE' ? fuseBaseColor() : GROUP_STYLE[group].color;
    const useUtil = mode === 'utilization' && state.mode === 'pareto';
    mesh.material.color.set(useUtil ? 0xffffff : mode === 'technical' ? COLORS.steel : base);
    els.forEach((e, idx) => {
      if (useUtil) {
        const u = e.util ? (state.fuse === 'niti' ? e.util.niti : e.util.steel) : 0;
        mesh.setColorAt(idx, utilizationColor(u));
      } else {
        mesh.setColorAt(idx, _col.set(0xffffff));
      }
    });
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
}

// Recompute element transforms from real recorded displacements at a frame.
function applyFrameDisplacements(frame) {
  const design = state.selectedDesign;
  if (!design) return;
  const q = design.quake_frames[state.fuse];
  if (!q) return;
  const base = state.model.nodes;
  const disp = q.nodal_displacements_m_rad[frame];
  const ids = q.node_ids;
  const displaced = base.map((p) => p.slice());
  for (let k = 0; k < ids.length; k++) {
    const id = ids[k], d = disp[k];
    displaced[id] = [base[id][0] + d[0] * DISPLACEMENT_SCALE, base[id][1] + d[1] * DISPLACEMENT_SCALE, base[id][2] + d[2] * DISPLACEMENT_SCALE];
  }
  for (const [group, { mesh, els }] of Object.entries(groupMeshes)) {
    els.forEach((e, idx) => setElementMatrix(mesh, idx, displaced[e.i], displaced[e.j], GROUP_STYLE[group].radius));
    mesh.instanceMatrix.needsUpdate = true;
  }
  if (lineMesh) {
    const pos = lineMesh.geometry.attributes.position;
    let k = 0;
    for (const e of state.model.elements) {
      pos.setXYZ(k++, ...displaced[e.i]);
      pos.setXYZ(k++, ...displaced[e.j]);
    }
    pos.needsUpdate = true;
  }
  if (nodeMesh) {
    displaced.forEach((p, idx) => { _mat.makeTranslation(p[0], p[1], p[2]); nodeMesh.setMatrixAt(idx, _mat); });
    nodeMesh.instanceMatrix.needsUpdate = true;
  }
}

// ----------------------------- 2D canvas charts -----------------------------
function setupCanvas(canvas) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 320, h = canvas.clientHeight || 280;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}
const linScale = (d0, d1, p0, p1) => (v) => p0 + (v - d0) * ((p1 - p0) / ((d1 - d0) || 1));

// Generic XY plot. opts.region optionally restricts to a sub-rectangle (CSS px).
function plotXY(canvas, opts) {
  const { ctx, w, h } = opts.prepped || setupCanvas(canvas);
  const R = opts.region || { x: 0, y: 0, w, h };
  const m = { l: 46, r: 12, t: 12, b: 30, ...(opts.margin || {}) };
  const left = R.x + m.l, right = R.x + R.w - m.r, top = R.y + m.t, bottom = R.y + R.h - m.b;
  const xs = [], ys = [];
  for (const s of opts.series) for (const p of s.points) { xs.push(p[0]); ys.push(p[1]); }
  let [x0, x1] = opts.xlim || [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = opts.ylim || [Math.min(...ys), Math.max(...ys)];
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  if (!opts.xlim) { const p = (x1 - x0) * 0.06; x0 -= p; x1 += p; }
  if (!opts.ylim) { const p = (y1 - y0) * 0.08; y0 -= p; y1 += p; }
  const sx = linScale(x0, x1, left, right), sy = linScale(y0, y1, bottom, top);
  ctx.font = '10px "Segoe UI", sans-serif';
  ctx.lineWidth = 1;
  const n = opts.ticks || 5;
  for (let i = 0; i <= n; i++) {
    const xv = x0 + ((x1 - x0) * i) / n, px = sx(xv);
    ctx.strokeStyle = '#eef2f6'; ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.textAlign = 'center'; ctx.fillText(xv.toFixed(opts.xdec ?? 1), px, bottom + 13);
    const yv = y0 + ((y1 - y0) * i) / n, py = sy(yv);
    ctx.strokeStyle = '#eef2f6'; ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(yv.toFixed(opts.ydec ?? 1), left - 5, py + 3);
  }
  ctx.strokeStyle = '#cbd5e1'; ctx.strokeRect(left, top, right - left, bottom - top);
  ctx.fillStyle = '#143055'; ctx.textAlign = 'center';
  if (opts.xlabel) ctx.fillText(opts.xlabel, (left + right) / 2, R.y + R.h - 3);
  if (opts.ylabel) {
    ctx.save(); ctx.translate(R.x + 11, (top + bottom) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(opts.ylabel, 0, 0); ctx.restore();
  }
  if (opts.title) { ctx.textAlign = 'left'; ctx.fillStyle = '#143055'; ctx.fillText(opts.title, left, top - 3); }
  for (const s of opts.series) {
    if (s.type === 'scatter') {
      for (const p of s.points) {
        ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), s.radius || 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = s.color; ctx.globalAlpha = s.alpha ?? 0.85; ctx.fill();
      }
      ctx.globalAlpha = 1;
    } else {
      ctx.beginPath();
      s.points.forEach((p, i) => { const X = sx(p[0]), Y = sy(p[1]); if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); });
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.5; ctx.setLineDash(s.dash || []); ctx.stroke(); ctx.setLineDash([]);
    }
  }
  return { ctx, sx, sy, left, right, top, bottom };
}

// ----------------------------- Pareto chart + selection -----------------------------
let paretoHits = [];

function drawPareto() {
  const designs = state.data.pareto.designs;
  const canvas = document.getElementById('chart-pareto');
  const series = [
    {
      type: 'scatter', color: HEX.crimson, radius: 6,
      points: designs.filter((d) => d.front_case === 'steel').map((d) => [d.params.span, d.kpis.steel.c_total_per_m2]),
    },
    {
      type: 'scatter', color: HEX.cyan, radius: 6,
      points: designs.filter((d) => d.front_case === 'niti').map((d) => [d.params.span, d.kpis.niti.c_total_per_m2]),
    },
  ];
  const { ctx, sx, sy } = plotXY(canvas, { series, xlabel: 'Clear span (m)', ylabel: 'Carbon', title: 'Click a certified design', ydec: 0 });
  paretoHits = designs.map((d) => ({ design: d, px: sx(d.params.span), py: sy(d.kpis[d.front_case].c_total_per_m2) }));
  for (const h of paretoHits) {
    if (h.design === state.selectedDesign) {
      ctx.beginPath(); ctx.arc(h.px, h.py, 10, 0, 2 * Math.PI);
      ctx.strokeStyle = HEX.steel; ctx.lineWidth = 2.5; ctx.stroke();
    }
  }
}

function setKPI(target, mainText, subText, unsafe) {
  const el = document.querySelector(`.kpi[data-kpi="${target}"]`);
  if (!el) return;
  el.querySelector('.kpi-value').innerHTML = subText ? `${mainText} <small>${subText}</small>` : mainText;
  el.classList.toggle('unsafe', !!unsafe);
}

function updateKPIsFromDesign(design) {
  const k = design.kpis[state.fuse];
  setKPI('peak_drift_pct', k.peak_drift_pct.toFixed(2), 'exact', !k.peak_drift_safe);
  setKPI('abs_residual_drift_pct', k.abs_residual_drift_pct.toFixed(3), 'exact', false);
  setKPI('grav_defl_mm', k.grav_defl_mm.toFixed(1), `≤${(design.params.span * 1000 / 250).toFixed(0)}`, !k.gravity_deflection_safe);
  setKPI('c_total_per_m2', k.c_total_per_m2.toFixed(1), 'exact', false);
}

function updateReadout() {
  const el = document.getElementById('design-readout');
  const d = state.selectedDesign;
  if (!d) { el.innerHTML = '<em>No design selected.</em>'; return; }
  const k = d.kpis[d.front_case];
  el.innerHTML = `<table>
    <tr><td>Design</td><td>${d.id} (${d.front_case})</td></tr>
    <tr><td>Span</td><td>${d.params.span.toFixed(2)} m</td></tr>
    <tr><td>Divisions</td><td>${d.params.diagrid_divisions}</td></tr>
    <tr><td>Glulam depth</td><td>${d.params.glulam_depth.toFixed(3)} m</td></tr>
    <tr><td>NiTi area</td><td>${d.params.niti_fuse_area.toFixed(0)} mm&sup2;</td></tr>
    <tr><td>Exact peak drift</td><td>${k.peak_drift_pct.toFixed(2)} %</td></tr>
    <tr><td>Exact residual</td><td>${k.abs_residual_drift_pct.toFixed(3)} %</td></tr>
    <tr><td>Lifecycle carbon</td><td>${k.c_total_per_m2.toFixed(1)} kgCO&#8322;e/m&sup2;</td></tr>
    <tr><td>First period</td><td>${k.T1_s.toFixed(3)} s</td></tr>
  </table>`;
}

function selectDesign(design) {
  state.selectedDesign = design;
  state.fuse = design.front_case;
  state.playFrame = 0;
  state.playing = false;
  state.model = designToModel(design);
  rebuildStructure();
  updateReadout();
  updateKPIsFromDesign(design);
  const btn = document.getElementById('play-quake');
  btn.disabled = false;
  btn.textContent = '\u25B6 Play 0.40g Earthquake (Steel vs NiTi)';
  btn.classList.remove('playing');
  drawActiveChart();
}

document.getElementById('chart-pareto').addEventListener('click', (ev) => {
  if (!state.data.pareto) return;
  const rect = ev.target.getBoundingClientRect();
  const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  let best = null, bestDist = 1e9;
  for (const h of paretoHits) {
    const dd = (h.px - x) ** 2 + (h.py - y) ** 2;
    if (dd < bestDist) { bestDist = dd; best = h; }
  }
  if (best && bestDist < 30 * 30) selectDesign(best.design);
});

function drawHysteresis() {
  const canvas = document.getElementById('chart-hysteresis');
  const design = state.selectedDesign;
  if (!design) {
    plotXY(canvas, { series: [{ type: 'line', color: '#e5e7eb', points: [[0, 0], [1, 1]] }], xlim: [0, 1], ylim: [0, 1], title: 'Select a certified design first' });
    return;
  }
  const frames = design.quake_frames.steel.lead_fuse_strain.length;
  const xvals = [], yvals = [];
  for (const fuse of ['steel', 'niti']) {
    const q = design.quake_frames[fuse];
    for (let i = 0; i < q.lead_fuse_strain.length; i++) { xvals.push(q.lead_fuse_strain[i]); yvals.push(q.lead_fuse_stress_MPa[i]); }
  }
  const xpad = (Math.max(...xvals) - Math.min(...xvals)) * 0.08 || 0.01;
  const ypad = (Math.max(...yvals) - Math.min(...yvals)) * 0.08 || 10;
  const upTo = state.playing ? state.playFrame + 1 : frames;
  const series = ['steel', 'niti'].map((fuse) => {
    const q = design.quake_frames[fuse];
    const pts = [];
    for (let i = 0; i < upTo; i++) pts.push([q.lead_fuse_strain[i], q.lead_fuse_stress_MPa[i]]);
    return { type: 'line', color: fuse === 'niti' ? HEX.cyan : HEX.crimson, width: 1.7, points: pts };
  });
  plotXY(canvas, {
    series,
    xlim: [Math.min(...xvals) - xpad, Math.max(...xvals) + xpad],
    ylim: [Math.min(...yvals) - ypad, Math.max(...yvals) + ypad],
    xlabel: 'Lead fuse strain', ylabel: 'Stress (MPa)',
    title: 'Fuse hysteresis (0.40g, drift x' + DISPLACEMENT_SCALE + ')', xdec: 3, ydec: 0,
  });
}

function drawDiffID() {
  const diff = state.data.diff;
  const canvas = document.getElementById('chart-diffid');
  const prepped = setupCanvas(canvas);
  const { ctx, w, h } = prepped;
  const loops = diff.loops;
  const H1 = h * 0.56;
  plotXY(canvas, {
    prepped,
    region: { x: 0, y: 0, w, h: H1 },
    series: [
      { type: 'scatter', color: HEX.muted, radius: 1.5, alpha: 0.5, points: loops.strain.map((s, i) => [s, loops.observed_stress_MPa[i]]) },
      { type: 'line', color: HEX.steel, width: 1.8, points: loops.strain.map((s, i) => [s, loops.true_stress_MPa[i]]) },
      { type: 'line', color: HEX.cyan, width: 1.4, dash: [4, 3], points: loops.strain.map((s, i) => [s, loops.recovered_stress_MPa[i]]) },
    ],
    xlabel: '', ylabel: 'Stress', title: 'Inverse ID: observed vs true vs recovered', xdec: 3, ydec: 0,
  });
  const tp = diff.true_parameters, rp = diff.recovered_parameters, ci = diff.confidence_intervals_95;
  const bx = 46, bw = w - 46 - 12, by0 = H1 + 20, bh = h - by0 - 16;
  const x0 = -0.5, x1 = 5.5, yv0 = -12, yv1 = 12;
  const px = (i) => bx + ((i - x0) / (x1 - x0)) * bw;
  const py = (v) => by0 + bh - ((v - yv0) / (yv1 - yv0)) * bh;
  ctx.fillStyle = '#eef2f6'; ctx.fillRect(bx, py(8), bw, py(-8) - py(8));
  ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(bx, py(0)); ctx.lineTo(bx + bw, py(0)); ctx.stroke();
  ctx.strokeStyle = '#cbd5e1'; ctx.strokeRect(bx, by0, bw, bh);
  ctx.strokeStyle = HEX.steel; ctx.lineWidth = 2;
  PARAM_ORDER.forEach((n, i) => {
    const lo = (ci[n][0] / tp[n] - 1) * 100, hi = (ci[n][1] / tp[n] - 1) * 100;
    ctx.beginPath(); ctx.moveTo(px(i), py(lo)); ctx.lineTo(px(i), py(hi)); ctx.stroke();
  });
  PARAM_ORDER.forEach((n, i) => {
    const v = (rp[n] / tp[n] - 1) * 100;
    ctx.beginPath(); ctx.arc(px(i), py(v), 4.5, 0, 2 * Math.PI); ctx.fillStyle = HEX.cyan; ctx.fill();
  });
  ctx.fillStyle = '#64748b'; ctx.font = '9px "Segoe UI", sans-serif'; ctx.textAlign = 'center';
  PARAM_ORDER.forEach((n, i) => ctx.fillText(n, px(i), by0 + bh + 11));
  ctx.textAlign = 'left'; ctx.fillStyle = '#143055'; ctx.fillText('Recovery error % + 95% CI', bx, by0 - 5);
}

function drawActiveChart() {
  if (state.chartTab === 'pareto') drawPareto();
  else if (state.chartTab === 'hysteresis') drawHysteresis();
  else drawDiffID();
}

// ----------------------------- Playback + explore -----------------------------
let lastTs = 0;
let exploreDirty = false;
const FRAME_MS = 90;

function startPlayback() {
  if (!state.selectedDesign) return;
  state.playing = true;
  state.playFrame = 0;
  state.playClock = 0;
  const btn = document.getElementById('play-quake');
  btn.textContent = '\u23F8 Pause Earthquake';
  btn.classList.add('playing');
  applyFrameDisplacements(0);
  if (state.chartTab === 'hysteresis') drawHysteresis();
}

function stopPlayback(ended) {
  state.playing = false;
  const btn = document.getElementById('play-quake');
  btn.classList.remove('playing');
  btn.textContent = ended ? '\u25B6 Play 0.40g Earthquake (Steel vs NiTi)' : '\u25B6 Resume Earthquake';
}

function animate(ts) {
  requestAnimationFrame(animate);
  const dt = lastTs ? ts - lastTs : 16;
  lastTs = ts;
  if (state.playing && state.selectedDesign) {
    state.playClock += dt;
    if (state.playClock >= FRAME_MS) {
      state.playClock = 0;
      const frames = state.selectedDesign.quake_frames[state.fuse].time_s.length;
      state.playFrame += 1;
      if (state.playFrame >= frames) { state.playFrame = frames - 1; stopPlayback(true); }
      applyFrameDisplacements(state.playFrame);
      if (state.chartTab === 'hysteresis') drawHysteresis();
    }
  }
  if (state.mode === 'explore' && exploreDirty) {
    exploreDirty = false;
    applyExploreModel();
  }
  controls.update();
  renderer.render(scene, camera);
}

function buildSliderUI() {
  const host = document.getElementById('sliders');
  host.innerHTML = '';
  for (const s of SLIDERS) {
    const row = document.createElement('div');
    row.className = 'slider-row';
    const label = document.createElement('label');
    label.innerHTML = `<span>${s.label}</span><span class="val">${s.value.toFixed(s.decimals)}</span>`;
    const input = document.createElement('input');
    input.type = 'range'; input.min = s.min; input.max = s.max; input.step = s.step; input.value = s.value;
    input.addEventListener('input', () => {
      s.value = parseFloat(input.value);
      label.querySelector('.val').textContent = s.value.toFixed(s.decimals);
      exploreDirty = true;
    });
    row.appendChild(label); row.appendChild(input);
    host.appendChild(row);
  }
}

function exploreParams() {
  const p = {};
  for (const s of SLIDERS) p[s.key] = s.value;
  return p;
}

function applyExploreModel() {
  const p = exploreParams();
  state.model = buildCanopy(p);
  rebuildStructure();
  // GP evaluation for the selected fuse case.
  const caseData = state.data.gp.cases[state.fuse];
  const features = [p.span, p.arch_rise_ratio, p.diagrid_divisions, p.glulam_depth, p.niti_fuse_area];
  const out = {};
  for (const t of TARGETS) out[t] = gpPredict(caseData, t, features);
  const pd = out.peak_drift_pct, ar = out.abs_residual_drift_pct, gd = out.grav_defl_mm, cc = out.c_total_per_m2;
  const gravLimit = (p.span * 1000) / 250;
  setKPI('peak_drift_pct', pd.mean.toFixed(2), `&plusmn;${(1.96 * pd.std).toFixed(2)}`, pd.mean + 1.96 * pd.std > 2.0);
  setKPI('abs_residual_drift_pct', ar.mean.toFixed(3), `&plusmn;${(1.96 * ar.std).toFixed(3)}`, false);
  setKPI('grav_defl_mm', gd.mean.toFixed(1), `&plusmn;${(1.96 * gd.std).toFixed(1)}`, gd.mean + 1.96 * gd.std > gravLimit);
  setKPI('c_total_per_m2', cc.mean.toFixed(1), `&plusmn;${(1.96 * cc.std).toFixed(1)}`, false);
}

// ----------------------------- UI wiring + init -----------------------------
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('#mode-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  document.getElementById('pareto-controls').classList.toggle('hidden', mode !== 'pareto');
  document.getElementById('explore-controls').classList.toggle('hidden', mode !== 'explore');
  if (mode === 'pareto') {
    if (state.playing) stopPlayback(false);
    const design = state.selectedDesign || (state.data.pareto && state.data.pareto.designs[0]);
    if (design) {
      state.selectedDesign = design;
      state.fuse = design.front_case;
      state.model = designToModel(design);
      rebuildStructure();
      updateReadout();
      updateKPIsFromDesign(design);
    }
  } else {
    applyExploreModel();
  }
  drawActiveChart();
}

function wireUI() {
  document.querySelectorAll('#mode-toggle button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  document.getElementById('display-mode').addEventListener('change', (e) => {
    state.displayMode = e.target.value;
    applyDisplayMode();
  });
  document.querySelectorAll('#chart-tabs button').forEach((btn) => btn.addEventListener('click', () => {
    state.chartTab = btn.dataset.tab;
    document.querySelectorAll('#chart-tabs button').forEach((b) => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.chart-canvas').forEach((c) => c.classList.add('hidden'));
    document.getElementById('chart-' + state.chartTab).classList.remove('hidden');
    drawActiveChart();
  }));
  document.querySelectorAll('#explore-fuse button').forEach((btn) => btn.addEventListener('click', () => {
    state.fuse = btn.dataset.fuse;
    document.querySelectorAll('#explore-fuse button').forEach((b) => b.classList.toggle('active', b === btn));
    if (state.mode === 'explore') applyExploreModel();
    else if (state.selectedDesign) { updateKPIsFromDesign(state.selectedDesign); applyDisplayMode(); }
  }));
  document.getElementById('play-quake').addEventListener('click', () => {
    if (!state.selectedDesign) return;
    if (state.playing) stopPlayback(false);
    else startPlayback();
  });
  document.getElementById('export-png').addEventListener('click', () => {
    renderer.render(scene, camera);
    const a = document.createElement('a');
    a.href = renderer.domElement.toDataURL('image/png');
    a.download = 'diagrid_canopy.png';
    a.click();
  });
  window.addEventListener('resize', () => setTimeout(drawActiveChart, 80));
}

async function init() {
  initScene();
  buildSliderUI();
  wireUI();
  const loading = document.getElementById('loading');
  try {
    const [gp, pareto, diff] = await Promise.all([
      loadJSON(DATA_FILES.gp), loadJSON(DATA_FILES.pareto), loadJSON(DATA_FILES.diff),
    ]);
    state.data = { gp, pareto, diff };
    loading.classList.add('hidden');
  } catch (err) {
    loading.classList.add('error');
    loading.textContent = 'Failed to load Day 4 artifacts:\n' + err.message;
    return;
  }
  if (state.data.pareto && state.data.pareto.designs.length) selectDesign(state.data.pareto.designs[0]);
  requestAnimationFrame(animate);
}

init();







