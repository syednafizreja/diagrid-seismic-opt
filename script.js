import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const DATA_FILES = {
  gp: 'data/gp_surrogate.json',
  pareto: 'data/pareto_canopies.json',
  diff: 'data/diff_id_uncertainty.json',
};
const TARGETS = ['peak_drift_pct', 'abs_residual_drift_pct', 'grav_defl_mm', 'c_total_per_m2'];
const PARAM_ORDER = ['E', 'sigma_Ms', 'sigma_Mf', 'sigma_As', 'sigma_Af', 'eps_L'];

const SLIDERS = [
  { key: 'span', label: 'Span', unit: 'm', min: 14, max: 28, step: 0.1, value: 20, def: 20, decimals: 1 },
  { key: 'arch_rise_ratio', label: 'Arch rise', unit: 'ratio', min: 0.08, max: 0.22, step: 0.005, value: 0.15, def: 0.15, decimals: 3 },
  { key: 'diagrid_divisions', label: 'Diagrid divisions', unit: '', min: 6, max: 12, step: 2, value: 8, def: 8, decimals: 0 },
  { key: 'glulam_depth', label: 'Glulam depth', unit: 'm', min: 0.22, max: 0.48, step: 0.01, value: 0.35, def: 0.35, decimals: 2 },
  { key: 'niti_fuse_area', label: 'Fuse area', unit: 'mm²', min: 60, max: 260, step: 5, value: 100, def: 100, decimals: 0 },
];

const GROUP_STYLE = {
  GLULAM_DIAGRID: { radius: 0.075, depth: 0.16, dens: 500 },
  STEEL_EDGE: { radius: 0.085, dens: 7850 },
  TREE_TRUNK: { radius: 0.16, dens: 7850 },
  TREE_BRANCH: { radius: 0.09, dens: 7850 },
  SEISMIC_FUSE: { radius: 0.075, dens: 7850 },
};
const STEEL_YIELD_STRAIN = 0.001775; // 0.1775%

const COLORS = {
  steelBlue: 0x1b4f9c,
  niti: 0x0097b2,
  vermilion: 0xc93b2b,
  steelDark: 0x252f3d,
  glulamTint: 0xffffff,
  fuseSilver: 0xbcc7d1,
};

const state = {
  mode: 'explore',            // 'explore' (GP) | 'pareto' (exact certified)
  displayMode: 'arctic',      // arctic | rendered | utilization | wireframe
  chartTab: 'hysteresis', fuse: 'niti',
  data: { gp: null, pareto: null, diff: null },
  selectedDesign: null, model: null,
  playing: false, playFrame: 0, playClock: 0, mag: 10,
  glazing: true, grid: true, axes: true, steelYielded: false,
};

async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status}). Serve the repo root over http://.`);
  return res.json();
}

// Pure-JS Matérn-5/2 GP matching exported scikit-learn parameters exactly.
function gpPredict(caseData, target, features) {
  const model = caseData.models[target];
  const mean = caseData.scaler.mean, scale = caseData.scaler.scale;
  const ls = model.length_scale, X = caseData.X_train_standardized;
  const n = X.length, dim = features.length;
  const xs = new Float64Array(dim);
  for (let j = 0; j < dim; j++) xs[j] = (features[j] - mean[j]) / scale[j];
  const c = model.constant_value, sq5 = Math.sqrt(5), k = new Float64Array(n);
  for (let a = 0; a < n; a++) {
    let s = 0;
    for (let j = 0; j < dim; j++) { const d = (X[a][j] - xs[j]) / ls[j]; s += d * d; }
    const r = Math.sqrt(s), t = sq5 * r;
    k[a] = c * (1 + t + (5 * r * r) / 3) * Math.exp(-t);
  }
  let mu = 0;
  for (let a = 0; a < n; a++) mu += k[a] * model.alpha[a];
  const L = model.cholesky, y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let sum = k[i];
    for (let j = 0; j < i; j++) sum -= L[i][j] * y[j];
    y[i] = sum / L[i][i];
  }
  let proj = 0;
  for (let i = 0; i < n; i++) proj += y[i] * y[i];
  return { mean: mu, std: Math.sqrt(Math.max(c + model.noise_level - proj, 0)) };
}

function gridDims(divisions) {
  const nx = Math.round(divisions), nz = Math.max(4, Math.floor(nx / 2));
  return { nx, nz };
}
function buildCanopy(params) {
  const span = params.span, archRise = params.arch_rise_ratio * span;
  const { nx, nz } = gridDims(params.diagrid_divisions), width = 0.5 * span;
  const nodes = [], idOf = {}, roofIds = [];
  for (let j = 0; j <= nz; j++) {
    const z = -width / 2 + (j * width) / nz;
    for (let i = 0; i <= nx; i++) {
      const x = -span / 2 + (i * span) / nx;
      const y = 4.5 + archRise * (1 - (2 * x / span) ** 2) * (1 - 0.5 * (2 * z / width) ** 2);
      const id = nodes.length;
      nodes.push([x, y, z]); roofIds.push(id); idOf[`${i},${j}`] = id;
    }
  }
  const elements = [], add = (group, a, b) => elements.push({ i: a, j: b, group });
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    add('GLULAM_DIAGRID', idOf[`${i},${j}`], idOf[`${i + 1},${j + 1}`]);
    add('GLULAM_DIAGRID', idOf[`${i + 1},${j}`], idOf[`${i},${j + 1}`]);
  }
  for (const i of [0, nx]) for (let j = 0; j < nz; j++) add('STEEL_EDGE', idOf[`${i},${j}`], idOf[`${i},${j + 1}`]);
  for (const j of [0, nz]) for (let i = 0; i < nx; i++) add('STEEL_EDGE', idOf[`${i},${j}`], idOf[`${i + 1},${j}`]);
  const colPos = [[-0.32 * span, -0.25 * width], [-0.32 * span, 0.25 * width], [0.32 * span, -0.25 * width], [0.32 * span, 0.25 * width]];
  const yb = 0.55 * 4.5;
  for (const [xc, zc] of colPos) {
    const bif = nodes.length; nodes.push([xc, yb, zc]);
    const ground = nodes.length; nodes.push([xc, 0, zc]);
    add('TREE_TRUNK', ground, bif);
    const nearest = roofIds.map((id) => ({ id, d: (nodes[id][0] - xc) ** 2 + (nodes[id][2] - zc) ** 2 })).sort((a, b) => a.d - b.d).slice(0, 4);
    for (const { id } of nearest) add('TREE_BRANCH', bif, id);
    for (const xa of [xc - 1.5, xc + 1.5]) {
      const fg = nodes.length; nodes.push([xa, 0, zc]); add('SEISMIC_FUSE', bif, fg);
    }
  }
  return { nodes, elements, nx, nz, span, width };
}
function designToModel(design) {
  const { nx, nz } = gridDims(design.params.diagrid_divisions);
  return {
    nodes: design.nodes.map((n) => n.xyz_m),
    elements: design.elements.map((e) => ({ i: e.node_ids[0], j: e.node_ids[1], group: e.group_tag, util: e.group_peak_static_utilization })),
    nx, nz, span: design.params.span, width: 0.5 * design.params.span, id: design.id,
  };
}

// -------------------- Procedural studio textures --------------------
function makeWoodTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 512;
  const x = c.getContext('2d');
  const g = x.createLinearGradient(0, 0, 512, 0);
  g.addColorStop(0, '#b98752'); g.addColorStop(0.5, '#d0a16a'); g.addColorStop(1, '#b77f48');
  x.fillStyle = g; x.fillRect(0, 0, 512, 512);
  let seed = 17;
  const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
  for (let i = 0; i < 190; i++) {
    const y = rnd() * 512, amp = 1 + rnd() * 5, tone = rnd() > 0.5 ? '86,51,20' : '244,213,166';
    x.strokeStyle = `rgba(${tone},${0.06 + rnd() * 0.15})`; x.lineWidth = 0.5 + rnd() * 1.4;
    x.beginPath(); x.moveTo(0, y);
    for (let px = 0; px <= 512; px += 16) x.lineTo(px, y + Math.sin(px / 55 + i) * amp + Math.sin(px / 19) * amp * 0.3);
    x.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(1, 4);
  tex.colorSpace = THREE.SRGBColorSpace; tex.anisotropy = 8;
  return tex;
}
function makeContactShadowTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const x = c.getContext('2d');
  const g = x.createRadialGradient(128, 128, 5, 128, 128, 126);
  g.addColorStop(0, 'rgba(22,30,45,0.20)'); g.addColorStop(0.55, 'rgba(22,30,45,0.08)'); g.addColorStop(1, 'rgba(22,30,45,0)');
  x.fillStyle = g; x.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
}

let renderer, scene, camera, controls, structureGroup, ground, gridHelper, contactShadow, axesHelper;
let woodTexture, keyLight, hemiLight, rimLight;
const groupMeshes = {}, detailMeshes = [];
let lineMesh = null, nodeMesh = null, glazingMesh = null, glazingBase = null;
const _a = new THREE.Vector3(), _b = new THREE.Vector3();
const _mid = new THREE.Vector3(), _dir = new THREE.Vector3();
const _quat = new THREE.Quaternion(), _scale = new THREE.Vector3();
const _mat = new THREE.Matrix4(), _up = new THREE.Vector3(0, 1, 0), _col = new THREE.Color();

function setElementMatrix(mesh, index, a, b, radius, depth) {
  _a.fromArray(a); _b.fromArray(b);
  _mid.addVectors(_a, _b).multiplyScalar(0.5);
  _dir.subVectors(_b, _a);
  const len = _dir.length() || 1e-6;
  _dir.normalize(); _quat.setFromUnitVectors(_up, _dir);
  _scale.set(radius, len, depth || radius);
  _mat.compose(_mid, _quat, _scale);
  mesh.setMatrixAt(index, _mat);
}
function makeModelAxes(len) {
  const g = new THREE.Group();
  const mk = (color) => new THREE.LineBasicMaterial({ color, depthTest: false, transparent: true, opacity: 0.9 });
  const add = (a, b, color) => {
    const gm = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]);
    g.add(new THREE.Line(gm, mk(color)));
  };
  add([0, 0, 0], [len, 0, 0], 0xd64545);
  add([0, 0, 0], [0, len, 0], 0x2e9e62);
  add([0, 0, 0], [0, 0, len], 0x1b4f9c);
  return g;
}

let resizeObserver = null;
function initResizeObserver() {
  const container = document.getElementById('view');
  if (typeof ResizeObserver === 'undefined') { window.addEventListener('resize', onResize); return; }
  resizeObserver = new ResizeObserver(() => onResize());
  resizeObserver.observe(container);
}
function onResize() {
  const container = document.getElementById('view');
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}

function initScene() {
  const container = document.getElementById('view');
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
  } catch (e) {
    throw new Error('WebGL context could not be created — this browser appears to lack WebGL support.');
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const w = container.clientWidth || 1, h = container.clientHeight || 1;
  renderer.setSize(w, h, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f5f7);
  scene.fog = new THREE.Fog(0xf4f5f7, 45, 115);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  camera = new THREE.PerspectiveCamera(42, w / h, 0.1, 600);
  camera.position.set(29, 20, 32);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.07;
  controls.target.set(0, 4, 0); controls.maxPolarAngle = Math.PI * 0.93;
  controls.minDistance = 2; controls.maxDistance = 180;

  hemiLight = new THREE.HemisphereLight(0xdcecff, 0x8c7b6a, 1.05); scene.add(hemiLight);
  keyLight = new THREE.DirectionalLight(0xffe6c7, 2.6);
  keyLight.position.set(24, 36, 18); keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(2048, 2048);
  Object.assign(keyLight.shadow.camera, { left: -42, right: 42, top: 42, bottom: -42, near: 1, far: 120 });
  keyLight.shadow.bias = -0.00035; keyLight.shadow.normalBias = 0.025;
  scene.add(keyLight);
  rimLight = new THREE.DirectionalLight(0xa9c9ff, 0.9); rimLight.position.set(-24, 15, -22); scene.add(rimLight);

  ground = new THREE.Mesh(new THREE.PlaneGeometry(600, 600), new THREE.MeshStandardMaterial({ color: 0xe9edf2, roughness: 1, metalness: 0 }));
  ground.rotation.x = -Math.PI / 2; ground.position.y = -0.035; ground.receiveShadow = true; scene.add(ground);

  gridHelper = new THREE.GridHelper(120, 60, 0x9aa6b4, 0xd5dbe2);
  gridHelper.position.y = -0.02;
  gridHelper.material.transparent = true; gridHelper.material.opacity = 0.7; gridHelper.material.depthWrite = false;
  scene.add(gridHelper);

  contactShadow = new THREE.Mesh(new THREE.PlaneGeometry(65, 65), new THREE.MeshBasicMaterial({ map: makeContactShadowTexture(), transparent: true, depthWrite: false }));
  contactShadow.rotation.x = -Math.PI / 2; contactShadow.position.y = -0.015; scene.add(contactShadow);

  axesHelper = makeModelAxes(4); scene.add(axesHelper);
  woodTexture = makeWoodTexture();
  structureGroup = new THREE.Group(); scene.add(structureGroup);

  initResizeObserver();
}

// -------------------- Structure, joints, glazing --------------------
function groupBaseColor(group) {
  if (group === 'GLULAM_DIAGRID') return COLORS.glulamTint;
  if (group === 'SEISMIC_FUSE') return COLORS.fuseSilver;
  return COLORS.steelDark;
}
function materialFor(group) {
  if (group === 'GLULAM_DIAGRID') return new THREE.MeshStandardMaterial({ map: woodTexture, bumpMap: woodTexture, bumpScale: 0.025, roughness: 0.62, metalness: 0.02 });
  if (group === 'SEISMIC_FUSE') return new THREE.MeshStandardMaterial({ color: COLORS.fuseSilver, metalness: 0.85, roughness: 0.20, emissive: new THREE.Color(COLORS.niti), emissiveIntensity: 0.0 });
  return new THREE.MeshStandardMaterial({ color: COLORS.steelDark, metalness: 0.65, roughness: 0.35 });
}
function disposeStructure() {
  for (const g of Object.keys(groupMeshes)) {
    const { mesh } = groupMeshes[g];
    structureGroup.remove(mesh); mesh.geometry.dispose(); mesh.material.dispose(); delete groupMeshes[g];
  }
  for (const m of detailMeshes) { structureGroup.remove(m); m.geometry.dispose(); m.material.dispose(); }
  detailMeshes.length = 0;
  for (const m of [lineMesh, nodeMesh, glazingMesh]) {
    if (m) { structureGroup.remove(m); m.geometry.dispose(); m.material.dispose(); }
  }
  lineMesh = null; nodeMesh = null; glazingMesh = null; glazingBase = null;
}
function rebuildStructure() {
  disposeStructure();
  if (!state.model) return;
  const nodes = state.model.nodes, byGroup = {};
  for (const e of state.model.elements) (byGroup[e.group] = byGroup[e.group] || []).push(e);
  for (const [group, els] of Object.entries(byGroup)) {
    const style = GROUP_STYLE[group];
    const geom = group === 'GLULAM_DIAGRID' ? new THREE.BoxGeometry(1, 1, 1) : new THREE.CylinderGeometry(1, 1, 1, 14, 1);
    const mesh = new THREE.InstancedMesh(geom, materialFor(group), els.length);
    mesh.castShadow = true; mesh.receiveShadow = true;
    els.forEach((e, idx) => {
      setElementMatrix(mesh, idx, nodes[e.i], nodes[e.j], style.radius, style.depth);
      mesh.setColorAt(idx, _col.set(0xffffff));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    structureGroup.add(mesh); groupMeshes[group] = { mesh, els };
  }
  const trunks = byGroup.TREE_TRUNK || [], fuses = byGroup.SEISMIC_FUSE || [];
  const collars = new THREE.InstancedMesh(new THREE.SphereGeometry(0.23, 16, 12), new THREE.MeshStandardMaterial({ color: 0x253445, metalness: 0.8, roughness: 0.24 }), trunks.length);
  trunks.forEach((e, i) => { _mat.makeTranslation(...nodes[e.j]); collars.setMatrixAt(i, _mat); });
  collars.castShadow = true; structureGroup.add(collars); detailMeshes.push(collars);
  const bases = new THREE.InstancedMesh(new THREE.BoxGeometry(0.75, 0.13, 0.75), new THREE.MeshStandardMaterial({ color: 0x64748b, metalness: 0.45, roughness: 0.5 }), trunks.length);
  trunks.forEach((e, i) => { _mat.makeTranslation(nodes[e.i][0], 0.065, nodes[e.i][2]); bases.setMatrixAt(i, _mat); });
  bases.castShadow = true; bases.receiveShadow = true; structureGroup.add(bases); detailMeshes.push(bases);
  const clevis = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.09, 0.09, 0.28, 10), new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.82, roughness: 0.24 }), fuses.length);
  fuses.forEach((e, i) => { _mat.makeTranslation(nodes[e.j][0], 0.14, nodes[e.j][2]); clevis.setMatrixAt(i, _mat); });
  structureGroup.add(clevis); detailMeshes.push(clevis);
  const positions = [];
  for (const e of state.model.elements) positions.push(...nodes[e.i], ...nodes[e.j]);
  const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  lineMesh = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x24303f, transparent: true, opacity: 0.55 }));
  structureGroup.add(lineMesh);
  nodeMesh = new THREE.InstancedMesh(new THREE.SphereGeometry(0.09, 10, 8), new THREE.MeshStandardMaterial({ color: 0x143055, metalness: 0.5, roughness: 0.3 }), nodes.length);
  nodes.forEach((p, idx) => { _mat.makeTranslation(...p); nodeMesh.setMatrixAt(idx, _mat); });
  nodeMesh.instanceMatrix.needsUpdate = true; structureGroup.add(nodeMesh);
  rebuildGlazing();
  applyDisplayMode();
}
function rebuildGlazing() {
  const { nx, nz } = state.model, nodes = state.model.nodes;
  const verts = [];
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
    for (const id of [a, b, c, b, d, c]) verts.push(...nodes[id]);
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geom.computeVertexNormals();
  const mat = new THREE.MeshPhysicalMaterial({ color: 0x9acbe8, roughness: 0.08, metalness: 0.0, transparent: true, opacity: 0.16, transmission: 0.12, thickness: 0.04, side: THREE.DoubleSide, depthWrite: false });
  glazingMesh = new THREE.Mesh(geom, mat); glazingMesh.renderOrder = 2;
  structureGroup.add(glazingMesh);
  glazingBase = { nx, nz };
}
function updateGlazing(nodes) {
  if (!glazingMesh || !glazingBase) return;
  const { nx, nz } = glazingBase, pos = glazingMesh.geometry.attributes.position;
  let k = 0;
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const a = j * (nx + 1) + i, b = a + 1, c = a + nx + 1, d = c + 1;
    for (const id of [a, b, c, b, d, c]) pos.setXYZ(k++, ...nodes[id]);
  }
  pos.needsUpdate = true;
}

function utilizationColor(u) {
  const t = Math.min(Math.max((u || 0) / 1.2, 0), 1);
  const c0 = new THREE.Color(0x2b6cb0), c1 = new THREE.Color(0xd97706), c2 = new THREE.Color(0xc93b2b);
  return t < 0.5 ? c0.clone().lerp(c1, t / 0.5) : c1.clone().lerp(c2, (t - 0.5) / 0.5);
}
function applyDisplayMode() {
  const mode = state.displayMode;
  const arctic = mode === 'arctic';
  if (lineMesh) lineMesh.visible = (mode === 'wireframe' || mode === 'arctic') ? true : false;
  if (nodeMesh) nodeMesh.visible = (mode === 'utilization' || mode === 'rendered');
  if (glazingMesh) glazingMesh.visible = state.glazing && mode !== 'wireframe';
  for (const m of detailMeshes) m.visible = mode !== 'wireframe';
  document.getElementById('scalebar').classList.toggle('hidden', mode !== 'utilization');
  for (const [group, { mesh, els }] of Object.entries(groupMeshes)) {
    mesh.visible = mode !== 'wireframe';
    const useUtil = mode === 'utilization' && state.mode === 'pareto';
    const mat = mesh.material;
    if (group === 'GLULAM_DIAGRID') { mat.map = arctic ? null : woodTexture; mat.needsUpdate = true; }
    if (useUtil) {
      mat.color.set(0xffffff);
      els.forEach((e, idx) => { const u = e.util ? (state.fuse === 'niti' ? e.util.niti : e.util.steel) : 0; mesh.setColorAt(idx, utilizationColor(u)); });
    } else {
      mat.color.set(arctic ? 0xf1f3f5 : groupBaseColor(group));
      els.forEach((e, idx) => mesh.setColorAt(idx, _col.set(0xffffff)));
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
  if (lineMesh) lineMesh.material.color.set(arctic && mode === 'arctic' ? 0x24303f : 0x52718f);
}
function resetFuseEmissive() {
  const fm = groupMeshes.SEISMIC_FUSE && groupMeshes.SEISMIC_FUSE.mesh.material;
  if (fm) { fm.emissive.set(COLORS.niti); fm.emissiveIntensity = 0; }
}
function applyFrameDisplacements(frame) {
  const design = state.selectedDesign;
  if (!design || !state.model) return;
  const q = design.quake_frames[state.fuse];
  if (!q) return;
  const base = state.model.nodes, disp = q.nodal_displacements_m_rad[frame], ids = q.node_ids;
  const displaced = base.map((p) => p.slice());
  for (let k = 0; k < ids.length; k++) {
    const id = ids[k], d = disp[k], s = state.mag;
    displaced[id] = [base[id][0] + d[0] * s, base[id][1] + d[1] * s, base[id][2] + d[2] * s];
  }
  for (const [group, { mesh, els }] of Object.entries(groupMeshes)) {
    els.forEach((e, idx) => setElementMatrix(mesh, idx, displaced[e.i], displaced[e.j], GROUP_STYLE[group].radius, GROUP_STYLE[group].depth));
    mesh.instanceMatrix.needsUpdate = true;
  }
  if (lineMesh) {
    const pos = lineMesh.geometry.attributes.position; let k = 0;
    for (const e of state.model.elements) { pos.setXYZ(k++, ...displaced[e.i]); pos.setXYZ(k++, ...displaced[e.j]); }
    pos.needsUpdate = true;
  }
  if (nodeMesh) { displaced.forEach((p, idx) => { _mat.makeTranslation(...p); nodeMesh.setMatrixAt(idx, _mat); }); nodeMesh.instanceMatrix.needsUpdate = true; }
  if (detailMeshes.length >= 3) {
    const trunks = state.model.elements.filter((e) => e.group === 'TREE_TRUNK');
    trunks.forEach((e, i) => { _mat.makeTranslation(...displaced[e.j]); detailMeshes[0].setMatrixAt(i, _mat); });
    detailMeshes[0].instanceMatrix.needsUpdate = true;
    const fuses = state.model.elements.filter((e) => e.group === 'SEISMIC_FUSE');
    fuses.forEach((e, i) => { _mat.makeTranslation(displaced[e.j][0], displaced[e.j][1] + 0.14, displaced[e.j][2]); detailMeshes[2].setMatrixAt(i, _mat); });
    detailMeshes[2].instanceMatrix.needsUpdate = true;
  }
  updateGlazing(displaced);
  const strain = Math.abs(q.lead_fuse_strain[frame] || 0);
  const fm = groupMeshes.SEISMIC_FUSE && groupMeshes.SEISMIC_FUSE.mesh.material;
  if (fm) {
    if (state.fuse === 'niti') {
      fm.emissive.set(COLORS.niti); fm.emissiveIntensity = 0.06 + Math.min(2.2, (strain / 0.06) * 2.2);
    } else {
      if (strain > STEEL_YIELD_STRAIN) state.steelYielded = true;
      fm.emissive.set(COLORS.vermilion);
      fm.emissiveIntensity = state.steelYielded ? 2.4 : 0.06 + Math.min(1.4, (strain / STEEL_YIELD_STRAIN) * 1.4);
    }
  }
}

// -------------------- Smooth architectural cameras --------------------
let camAnim = null;
function flyTo(position, target, duration = 950) {
  camAnim = { start: performance.now(), duration, fromPos: camera.position.clone(), fromTarget: controls.target.clone(), toPos: new THREE.Vector3(...position), toTarget: new THREE.Vector3(...target) };
}
function modelSpan() {
  if (!state.model) return 20;
  const xs = state.model.nodes.map((n) => n[0]);
  return Math.max(...xs) - Math.min(...xs);
}
function fitCamera(snap = false) {
  if (!state.model || !state.model.nodes || state.model.nodes.length === 0) return;
  const box = new THREE.Box3();
  for (const n of state.model.nodes) box.expandByPoint(new THREE.Vector3(...n));
  const center = new THREE.Vector3(); box.getCenter(center);
  const size = new THREE.Vector3(); box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 1);
  const fovRad = (camera.fov * Math.PI) / 180;
  const distance = (maxDim / (2 * Math.tan(fovRad / 2))) * 1.6;
  const pos = [center.x + distance * 0.55, center.y + distance * 0.45, center.z + distance * 0.65];
  const target = [center.x, center.y, center.z];
  controls.maxDistance = distance * 4;
  controls.minDistance = maxDim * 0.15;
  if (snap) {
    camera.position.set(...pos); controls.target.set(...target);
  } else flyTo(pos, target, 900);
}
function setCameraPreset(name) {
  const s = modelSpan();
  if (name === 'perspective') flyTo([s * 1.35, s * 0.85, s * 1.45], [0, 5, 0]);
  else if (name === 'elevation') flyTo([0, 6, s * 1.85], [0, 4, 0]);
  else if (name === 'plan') flyTo([0, s * 2.4, 0.01], [0, 0, 0]);
  else if (name === 'fuse') {
    const trunk = state.model && state.model.elements.find((e) => e.group === 'TREE_TRUNK');
    if (!trunk) return;
    const bif = state.model.nodes[trunk.j];
    const anchor = state.model.elements.find((e) => e.group === 'SEISMIC_FUSE' && e.i === trunk.j);
    const anchorPos = anchor ? state.model.nodes[anchor.j] : [bif[0] + 1.5, 0, bif[2]];
    const mid = [(bif[0] + anchorPos[0]) / 2, (bif[1] + anchorPos[1]) / 2, (bif[2] + anchorPos[2]) / 2];
    flyTo([mid[0] + 3.2, mid[1] + 2.4, mid[2] + 3.8], mid, 1100);
  }
  document.getElementById('camera-current').textContent = (name.charAt(0).toUpperCase() + name.slice(1)) + ' ▾';
  if (name === 'fuse') document.getElementById('camera-current').textContent = 'Fuse Detail ▾';
}

// -------------------- HiDPI scientific charts --------------------
function cssVar(name) { const v = getComputedStyle(document.body).getPropertyValue(name).trim(); return v || null; }
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 320, h = canvas.clientHeight || 230;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, dpr };
}
const linScale = (d0, d1, p0, p1) => (v) => p0 + (v - d0) * ((p1 - p0) / ((d1 - d0) || 1));
function chartColors() {
  return { text: '#5c6675', grid: '#dce1e8', ink: '#161b22', accent: '#1b4f9c', niti: '#0097b2', steel: '#c93b2b', ghost: '#94a3b8' };
}
function plotXY(canvas, opts) {
  const { ctx, w, h } = opts.prepped || setupCanvas(canvas);
  const R = opts.region || { x: 0, y: 0, w, h };
  const m = { l: 45, r: 10, t: 12, b: 29, ...(opts.margin || {}) };
  const left = R.x + m.l, right = R.x + R.w - m.r, top = R.y + m.t, bottom = R.y + R.h - m.b;
  const xs = [], ys = [];
  for (const s of opts.series) for (const p of s.points) { xs.push(p[0]); ys.push(p[1]); }
  let [x0, x1] = opts.xlim || [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = opts.ylim || [Math.min(...ys), Math.max(...ys)];
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  if (!opts.xlim) { const p = (x1 - x0) * 0.06; x0 -= p; x1 += p; }
  if (!opts.ylim) { const p = (y1 - y0) * 0.08; y0 -= p; y1 += p; }
  const sx = linScale(x0, x1, left, right), sy = linScale(y0, y1, bottom, top), C = chartColors();
  ctx.font = '9px "JetBrains Mono", monospace'; ctx.lineWidth = 1;
  const n = opts.ticks || 5;
  for (let i = 0; i <= n; i++) {
    const xv = x0 + ((x1 - x0) * i) / n, px = sx(xv);
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
    ctx.fillStyle = C.text; ctx.textAlign = 'center'; ctx.fillText(xv.toFixed(opts.xdec ?? 1), px, bottom + 13);
    const yv = y0 + ((y1 - y0) * i) / n, py = sy(yv);
    ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(yv.toFixed(opts.ydec ?? 1), left - 5, py + 3);
  }
  ctx.strokeStyle = C.grid; ctx.strokeRect(left, top, right - left, bottom - top);
  ctx.fillStyle = C.ink; ctx.textAlign = 'center';
  if (opts.xlabel) ctx.fillText(opts.xlabel, (left + right) / 2, R.y + R.h - 3);
  if (opts.ylabel) { ctx.save(); ctx.translate(R.x + 10, (top + bottom) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(opts.ylabel, 0, 0); ctx.restore(); }
  if (opts.title) { ctx.textAlign = 'left'; ctx.fillStyle = C.ink; ctx.fillText(opts.title, left, top - 3); }
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

let paretoHits = [];
function drawPareto() {
  const designs = state.data.pareto.designs, canvas = document.getElementById('chart-pareto');
  const C = chartColors();
  const series = ['steel', 'niti'].map((fuse) => ({
    type: 'scatter', color: fuse === 'niti' ? C.niti : C.steel, radius: 5.5,
    points: designs.filter((d) => d.front_case === fuse).map((d) => [d.params.span, d.kpis[fuse].c_total_per_m2]),
  }));
  const { ctx, sx, sy } = plotXY(canvas, { series, xlabel: 'Clear span (m)', ylabel: 'Carbon', title: 'Lifecycle carbon vs. clear span', ydec: 0 });
  paretoHits = designs.map((d) => ({ design: d, px: sx(d.params.span), py: sy(d.kpis[d.front_case].c_total_per_m2) }));
  for (const h of paretoHits) if (h.design === state.selectedDesign) {
    ctx.beginPath(); ctx.arc(h.px, h.py, 9.5, 0, 2 * Math.PI); ctx.strokeStyle = C.ink; ctx.lineWidth = 2; ctx.stroke();
  }
  document.getElementById('chart-caption').textContent = 'Hover for exact KPIs · click to load the certified design';
}
function nearestPareto(ev) {
  const rect = ev.target.getBoundingClientRect(), x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  let best = null, bd = 1e9;
  for (const h of paretoHits) { const d = (h.px - x) ** 2 + (h.py - y) ** 2; if (d < bd) { bd = d; best = h; } }
  return { best, dist: Math.sqrt(bd), x, y };
}
function drawHysteresis() {
  const canvas = document.getElementById('chart-hysteresis'), design = state.selectedDesign;
  if (!design) { plotXY(canvas, { series: [{ type: 'line', color: '#94a3b8', points: [[0, 0], [1, 1]] }], xlim: [0, 1], ylim: [0, 1], title: 'Load a certified design first' }); return; }
  const C = chartColors();
  const xvals = [], yvals = [];
  for (const fuse of ['steel', 'niti']) {
    const q = design.quake_frames[fuse];
    for (let i = 0; i < q.lead_fuse_strain.length; i++) { xvals.push(q.lead_fuse_strain[i]); yvals.push(q.lead_fuse_stress_MPa[i]); }
  }
  const xpad = (Math.max(...xvals) - Math.min(...xvals)) * 0.08 || 0.001;
  const ypad = (Math.max(...yvals) - Math.min(...yvals)) * 0.08 || 10;
  const frames = design.quake_frames[state.fuse].lead_fuse_strain.length;
  const upTo = state.playing || state.playFrame > 0 ? state.playFrame + 1 : frames;
  const other = state.fuse === 'niti' ? 'steel' : 'niti';
  const ghost = design.quake_frames[other], q = design.quake_frames[state.fuse];
  const activeColor = state.fuse === 'niti' ? C.niti : C.steel;
  const pts = [];
  for (let i = 0; i < upTo; i++) pts.push([q.lead_fuse_strain[i], q.lead_fuse_stress_MPa[i]]);
  const series = [
    { type: 'line', color: C.ghost, width: 1, alpha: 0.45, points: ghost.lead_fuse_strain.map((e, i) => [e, ghost.lead_fuse_stress_MPa[i]]) },
    { type: 'line', color: activeColor, width: 1.9, points: pts },
  ];
  const { ctx, sx, sy } = plotXY(canvas, { series, xlim: [Math.min(...xvals) - xpad, Math.max(...xvals) + xpad], ylim: [Math.min(...yvals) - ypad, Math.max(...yvals) + ypad], xlabel: 'Lead fuse strain ε', ylabel: 'Stress σ (MPa)', title: `${state.fuse.toUpperCase()} live · ${other} ghost`, xdec: 4, ydec: 0 });
  if (state.playing || state.playFrame > 0) {
    const f = Math.min(state.playFrame, frames - 1), ex = q.lead_fuse_strain[f], sig = q.lead_fuse_stress_MPa[f];
    ctx.beginPath(); ctx.arc(sx(ex), sy(sig), 5.5, 0, 2 * Math.PI); ctx.fillStyle = activeColor; ctx.fill();
  }
  const f = Math.min(state.playFrame, frames - 1);
  document.getElementById('chart-caption').textContent = `t = ${q.time_s[f].toFixed(2)} s   ·   ε = ${q.lead_fuse_strain[f].toFixed(5)}   ·   σ = ${q.lead_fuse_stress_MPa[f].toFixed(1)} MPa`;
}
function drawDiffID() {
  const diff = state.data.diff, canvas = document.getElementById('chart-diffid'), loops = diff.loops, C = chartColors();
  plotXY(canvas, {
    series: [
      { type: 'scatter', color: C.ghost, radius: 1.3, alpha: 0.35, points: loops.strain.map((s, i) => [s, loops.observed_stress_MPa[i]]) },
      { type: 'line', color: C.ink, width: 1.4, points: loops.strain.map((s, i) => [s, loops.true_stress_MPa[i]]) },
      { type: 'line', color: C.niti, width: 1.5, dash: [4, 3], points: loops.strain.map((s, i) => [s, loops.recovered_stress_MPa[i]]) },
    ],
    xlabel: 'Strain ε', ylabel: 'Stress (MPa)', title: '3% noise · true vs. PyTorch recovered', xdec: 3, ydec: 0,
  });
  const tp = diff.true_parameters, rp = diff.recovered_parameters, ci = diff.confidence_intervals_95;
  const rows = PARAM_ORDER.map((n) => `<tr><td>${n}</td><td>${tp[n].toPrecision(5)}</td><td>${rp[n].toPrecision(5)}</td><td>${diff.relative_recovery_error_pct[n].toFixed(2)}%</td><td>${ci[n][0].toPrecision(4)}–${ci[n][1].toPrecision(4)}</td></tr>`).join('');
  document.getElementById('chart-caption').innerHTML = `<table class="param-table"><thead><tr><th>Param</th><th>True</th><th>Recovered</th><th>Err</th><th>95% CI</th></tr></thead><tbody>${rows}</tbody></table>`;
}
function drawActiveChart() {
  if (state.chartTab === 'hysteresis') drawHysteresis();
  else if (state.chartTab === 'pareto') drawPareto();
  else if (state.chartTab === 'diffid') drawDiffID();
}

// -------------------- Telemetry and selection --------------------
function setConfidence(text, cls) {
  const el = document.getElementById('conf-pill');
  el.className = 'badge ' + cls;
  el.textContent = text;
}
function computeMass(model) {
  let mass = 0;
  for (const e of model.elements) {
    const a = model.nodes[e.i], b = model.nodes[e.j];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const s = GROUP_STYLE[e.group];
    const area = e.group === 'GLULAM_DIAGRID' ? (s.radius * 2) * (s.depth || 0.15) : Math.PI * s.radius * s.radius;
    mass += area * len * (s.dens || 500);
  }
  return mass / 1000;
}
function updateMeta() {
  const el = document.getElementById('meta');
  if (!state.model) { el.textContent = '—'; return; }
  const m = state.model, mass = computeMass(m);
  el.textContent = `${m.span.toFixed(1)} × ${m.width.toFixed(1)} m · ${m.nodes.length} nodes · ${m.elements.length} elements · ${mass.toFixed(1)} t`;
}
function updateTelemetry(design) {
  const k = design.kpis[state.fuse], other = design.kpis[state.fuse === 'niti' ? 'steel' : 'niti'];
  document.getElementById('kpi-carbon').textContent = `${k.c_total_per_m2.toFixed(1)} kgCO₂e/m²`;
  document.getElementById('kpi-carbon-sub').textContent = `A1–A3 ${k.c_a1_a3_kgco2e.toFixed(0)} · Repair ${k.c_repair_kgco2e.toFixed(0)} kgCO₂e`;
  document.getElementById('kpi-drift').textContent = `${k.peak_drift_pct.toFixed(2)}% / ${k.abs_residual_drift_pct.toFixed(2)}%`;
  document.getElementById('kpi-drift-sub').textContent = `Δ peak ${(other.peak_drift_pct - k.peak_drift_pct).toFixed(2)} · ${state.fuse} vs ${state.fuse === 'niti' ? 'steel' : 'niti'}`;
  // Max utilization: read raw ratio field (e.g. 1.73476) and format as % without double-multiplying.
  const utils = Object.entries(k.peak_member_stress_utilization).sort((a, b) => b[1] - a[1]);
  const maxUtil = utils[0][1];
  const utilPct = Math.round(maxUtil * 100);
  const kpiUtilEl = document.getElementById('kpi-util');
  kpiUtilEl.textContent = utilPct + '%';
  kpiUtilEl.className = 'kpi-val ' + (maxUtil > 1.0 ? 'danger' : maxUtil >= 0.85 ? 'caution' : 'ok');
  document.getElementById('kpi-util-sub').textContent = `${utils[0][0]} · ${state.fuse}`;
  document.getElementById('conf-sub').textContent = `${design.id} · exact solver`;
  setConfidence('Exact OpenSeesPy', 'badge-exact');
}
function selectDesign(design) {
  state.selectedDesign = design; state.fuse = design.front_case; state.mode = 'pareto';
  state.playing = false; state.playFrame = 0; state.playClock = 0; state.steelYielded = false;
  state.model = designToModel(design);
  rebuildStructure(); resetFuseEmissive();
  fitCamera();
  syncFuseButtons();
  document.getElementById('scrub').value = '0';
  document.getElementById('scrub-time').textContent = '0.00s';
  const t = design.quake_frames[state.fuse].time_s;
  document.getElementById('scrub-end').textContent = `${t[t.length - 1].toFixed(2)}s`;
  updatePlayButton(); updateTelemetry(design); updateMeta(); drawActiveChart();
}
function exploreParams() { const p = {}; for (const s of SLIDERS) p[s.key] = s.value; return p; }
function applyExploreModel() {
  const p = exploreParams(); state.model = buildCanopy(p); rebuildStructure(); resetFuseEmissive(); updateMeta();
  if (!state.data.gp) return;
  const caseData = state.data.gp.cases[state.fuse];
  const features = [p.span, p.arch_rise_ratio, p.diagrid_divisions, p.glulam_depth, p.niti_fuse_area];
  const out = {}; for (const t of TARGETS) out[t] = gpPredict(caseData, t, features);
  const pd = out.peak_drift_pct, ar = out.abs_residual_drift_pct, gd = out.grav_defl_mm, cc = out.c_total_per_m2;
  document.getElementById('kpi-carbon').textContent = `${cc.mean.toFixed(1)} kgCO₂e/m²`;
  document.getElementById('kpi-carbon-sub').textContent = `± ${(1.96 * cc.std).toFixed(1)} · GP lifecycle`;
  document.getElementById('kpi-drift').textContent = `${pd.mean.toFixed(2)}% / ${ar.mean.toFixed(2)}%`;
  document.getElementById('kpi-drift-sub').textContent = `± ${(1.96 * pd.std).toFixed(2)} / ± ${(1.96 * ar.std).toFixed(3)}`;
  const kpiUtilEl = document.getElementById('kpi-util');
  kpiUtilEl.textContent = '—'; kpiUtilEl.className = 'kpi-val';
  document.getElementById('kpi-util-sub').textContent = `gravity ${gd.mean.toFixed(1)} ± ${(1.96 * gd.std).toFixed(1)} mm`;
  document.getElementById('conf-sub').textContent = 'Matérn 5/2 surrogate';
  const ucb = pd.mean + 1.96 * pd.std;
  const lo = caseData.training_domain.minimum, hi = caseData.training_domain.maximum;
  const outside = features.some((v, i) => v < lo[i] || v > hi[i]);
  if (outside) setConfidence('Outside training envelope', 'badge-warn');
  else if (ucb > 2.0) setConfidence(`95% UCB ${ucb.toFixed(2)}% > 2.0%`, 'badge-danger');
  else setConfidence('GP ± 1.96σ', 'badge-gp');
}

// -------------------- Playback, export --------------------
let lastTs = 0;
function updatePlayButton() {
  const b = document.getElementById('play-quake');
  b.textContent = state.playing ? '❚❚ Pause Quake' : state.playFrame > 0 ? '▶ Resume Quake' : '▶ Play 0.40g Quake';
  b.classList.toggle('playing', state.playing);
}
function startPlayback() {
  if (!state.selectedDesign) return;
  state.playing = true;
  if (state.playFrame >= state.selectedDesign.quake_frames[state.fuse].time_s.length - 1) state.playFrame = 0;
  updatePlayButton();
}
function stopPlayback() { state.playing = false; updatePlayButton(); }
function animate(ts) {
  requestAnimationFrame(animate);
  const dt = lastTs ? Math.min(ts - lastTs, 80) : 16; lastTs = ts;
  if (camAnim) {
    const t = Math.min((ts - camAnim.start) / camAnim.duration, 1), e = 1 - Math.pow(1 - t, 3);
    camera.position.lerpVectors(camAnim.fromPos, camAnim.toPos, e);
    controls.target.lerpVectors(camAnim.fromTarget, camAnim.toTarget, e);
    if (t >= 1) camAnim = null;
  }
  if (state.playing && state.selectedDesign) {
    state.playClock += dt;
    if (state.playClock >= 300) {
      state.playClock = 0;
      const frames = state.selectedDesign.quake_frames[state.fuse].time_s.length;
      state.playFrame += 1;
      if (state.playFrame >= frames) { state.playFrame = frames - 1; stopPlayback(); }
      applyFrameDisplacements(state.playFrame);
      document.getElementById('scrub').value = String(state.playFrame);
      document.getElementById('scrub-time').textContent = `${state.selectedDesign.quake_frames[state.fuse].time_s[state.playFrame].toFixed(2)}s`;
      if (state.chartTab === 'hysteresis') drawHysteresis();
    }
  }
  controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}
function exportHighResPNG() {
  const container = document.getElementById('view');
  const w = container.clientWidth, h = container.clientHeight, oldRatio = renderer.getPixelRatio();
  renderer.setPixelRatio(Math.min((window.devicePixelRatio || 1) * 2, 4));
  renderer.setSize(w, h, false); renderer.render(scene, camera);
  const a = document.createElement('a'); a.href = renderer.domElement.toDataURL('image/png'); a.download = 'diagrid_optimization_2x.png'; a.click();
  renderer.setPixelRatio(oldRatio); renderer.setSize(w, h, false);
}

// -------------------- UI wiring and startup --------------------
function formatSliderVal(s) { return s.unit ? `${s.value.toFixed(s.decimals)} ${s.unit}` : s.value.toFixed(s.decimals); }
function buildSliderUI() {
  const host = document.getElementById('sliders'); host.innerHTML = '';
  for (const s of SLIDERS) {
    const row = document.createElement('div'); row.className = 'slider-row';
    const label = document.createElement('label');
    label.innerHTML = `<span><span class="port"></span>${s.label}</span><span class="val">${formatSliderVal(s)}</span>`;
    const input = document.createElement('input');
    input.type = 'range'; input.min = s.min; input.max = s.max; input.step = s.step; input.value = s.value;
    input.addEventListener('input', () => {
      s.value = parseFloat(input.value);
      label.querySelector('.val').textContent = formatSliderVal(s);
      state.mode = 'explore'; state.selectedDesign = null; state.playing = false;
      applyExploreModel();
    });
    row.appendChild(label); row.appendChild(input); host.appendChild(row);
  }
}
function syncFuseButtons() {
  document.querySelectorAll('#fuse-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.fuse === state.fuse));
}
function setFuse(fuse) {
  state.fuse = fuse; state.steelYielded = false; resetFuseEmissive();
  syncFuseButtons();
  if (state.mode === 'explore' || !state.selectedDesign) applyExploreModel();
  else { updateTelemetry(state.selectedDesign); applyFrameDisplacements(state.playFrame); }
  applyDisplayMode(); drawActiveChart();
}
function resetModel() {
  for (const s of SLIDERS) s.value = s.def;
  buildSliderUI();
  state.fuse = 'niti'; state.steelYielded = false;
  state.selectedDesign = null; state.mode = 'explore'; state.playing = false; state.playFrame = 0;
  syncFuseButtons(); resetFuseEmissive();
  applyExploreModel(); fitCamera();
}

function showError(msg) {
  const banner = document.getElementById('error-banner');
  banner.innerHTML = '';
  const span = document.createElement('span'); span.textContent = msg;
  const btn = document.createElement('button'); btn.className = 'dismiss'; btn.textContent = 'Dismiss';
  btn.addEventListener('click', () => { banner.hidden = true; });
  banner.appendChild(span); banner.appendChild(btn);
  banner.hidden = false;
  console.error(msg);
}

function wireUI() {
  document.querySelectorAll('#fuse-toggle button').forEach((b) => b.addEventListener('click', () => setFuse(b.dataset.fuse)));
  document.getElementById('reset-model').addEventListener('click', resetModel);
  document.getElementById('reset-view').addEventListener('click', () => fitCamera());

  document.querySelectorAll('[data-display]').forEach((b) => b.addEventListener('click', () => {
    state.displayMode = b.dataset.display;
    document.querySelectorAll('[data-display]').forEach((x) => x.classList.toggle('active', x === b));
    applyDisplayMode();
  }));

  document.querySelectorAll('#chart-tabs button').forEach((b) => b.addEventListener('click', () => {
    state.chartTab = b.dataset.tab;
    document.querySelectorAll('#chart-tabs button').forEach((x) => x.classList.toggle('active', x === b));
    document.querySelectorAll('.chart-canvas').forEach((c) => c.classList.add('hidden'));
    document.getElementById(`chart-${state.chartTab}`).classList.remove('hidden');
    drawActiveChart();
  }));

  const pc = document.getElementById('chart-pareto');
  pc.addEventListener('mousemove', (ev) => {
    const { best, dist, x, y } = nearestPareto(ev), tip = document.getElementById('pareto-tooltip');
    if (!best || dist > 18) { tip.classList.add('hidden'); return; }
    const d = best.design, k = d.kpis[d.front_case];
    tip.innerHTML = `<strong>${d.id} · ${d.front_case}</strong><br>Span ${d.params.span.toFixed(2)} m · Carbon ${k.c_total_per_m2.toFixed(1)}<br>Peak ${k.peak_drift_pct.toFixed(2)}% · Util ${Math.round(Math.max(...Object.values(k.peak_member_stress_utilization)) * 100)}%`;
    tip.style.left = `${Math.min(x + 12, 210)}px`; tip.style.top = `${Math.max(y - 65, 4)}px`; tip.classList.remove('hidden');
  });
  pc.addEventListener('mouseleave', () => document.getElementById('pareto-tooltip').classList.add('hidden'));
  pc.addEventListener('click', (ev) => { const { best, dist } = nearestPareto(ev); if (best && dist < 40) selectDesign(best.design); });

  document.getElementById('play-quake').addEventListener('click', () => { if (state.playing) stopPlayback(); else startPlayback(); });
  document.getElementById('scrub').addEventListener('input', (ev) => {
    if (!state.selectedDesign) return;
    state.playFrame = parseInt(ev.target.value, 10);
    applyFrameDisplacements(state.playFrame);
    document.getElementById('scrub-time').textContent = `${state.selectedDesign.quake_frames[state.fuse].time_s[state.playFrame].toFixed(2)}s`;
    if (state.chartTab === 'hysteresis') drawHysteresis();
  });

  document.querySelectorAll('#mag button').forEach((b) => b.addEventListener('click', () => {
    state.mag = parseInt(b.dataset.mag, 10);
    document.querySelectorAll('#mag button').forEach((x) => x.classList.toggle('active', x === b));
    if (state.selectedDesign) applyFrameDisplacements(state.playFrame);
  }));

  document.getElementById('toggle-glazing').addEventListener('click', (ev) => { state.glazing = !state.glazing; ev.target.classList.toggle('active', state.glazing); applyDisplayMode(); });
  document.getElementById('toggle-grid').addEventListener('click', (ev) => { state.grid = !state.grid; ev.target.classList.toggle('active', state.grid); if (gridHelper) gridHelper.visible = state.grid; });
  document.getElementById('toggle-axes').addEventListener('click', (ev) => { state.axes = !state.axes; ev.target.classList.toggle('active', state.axes); if (axesHelper) axesHelper.visible = state.axes; });
  document.getElementById('export-png').addEventListener('click', exportHighResPNG);

  const camCurrent = document.getElementById('camera-current');
  const camMenu = document.getElementById('camera-menu');
  camCurrent.addEventListener('click', () => camMenu.classList.toggle('hidden'));
  document.querySelectorAll('#camera-menu button').forEach((b) => b.addEventListener('click', () => { camMenu.classList.add('hidden'); setCameraPreset(b.dataset.cam); }));
  document.addEventListener('click', (ev) => { if (!document.getElementById('camera-vtab').contains(ev.target)) camMenu.classList.add('hidden'); });

  window.addEventListener('resize', () => setTimeout(drawActiveChart, 120));
}

async function init() {
  try { initScene(); } catch (err) { showError('Initialization error: ' + err.message); return; }
  buildSliderUI(); wireUI();
  requestAnimationFrame(animate); // continuous render loop starts immediately → viewport never blank
  try {
    const [gp, pareto, diff] = await Promise.all([loadJSON(DATA_FILES.gp), loadJSON(DATA_FILES.pareto), loadJSON(DATA_FILES.diff)]);
    state.data = { gp, pareto, diff };
    const initial = pareto.designs[0] || pareto.designs.find((d) => d.front_case === state.fuse);
    if (initial) selectDesign(initial);
  } catch (err) { showError('Failed to load artifacts: ' + err.message); }
}
init();







