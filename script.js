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
  { key: 'span', label: 'Clear span (m)', min: 14, max: 28, step: 0.1, value: 20, decimals: 1 },
  { key: 'arch_rise_ratio', label: 'Arch rise ratio', min: 0.08, max: 0.22, step: 0.005, value: 0.15, decimals: 3 },
  { key: 'diagrid_divisions', label: 'Diagrid divisions', min: 6, max: 12, step: 2, value: 8, decimals: 0 },
  { key: 'glulam_depth', label: 'Glulam depth (m)', min: 0.22, max: 0.48, step: 0.01, value: 0.35, decimals: 2 },
  { key: 'niti_fuse_area', label: 'NiTi fuse area (mm²)', min: 60, max: 260, step: 5, value: 100, decimals: 0 },
];
const GROUP_STYLE = {
  GLULAM_DIAGRID: { radius: 0.075, depth: 0.16 },
  STEEL_EDGE: { radius: 0.085 },
  TREE_TRUNK: { radius: 0.16 },
  TREE_BRANCH: { radius: 0.09 },
  SEISMIC_FUSE: { radius: 0.075 },
};
const STEEL_YIELD_STRAIN = 0.001775; // 0.1775%
const state = {
  mode: 'pareto', displayMode: 'architectural', chartTab: 'hysteresis', fuse: 'niti',
  data: { gp: null, pareto: null, diff: null }, selectedDesign: null, model: null,
  playing: false, playFrame: 0, playClock: 0, speed: 1, mag: 10,
  glazing: true, grid: true, axes: true, theme: 'light', steelYielded: false,
};

async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not load ${url} (HTTP ${res.status}). Serve the repo root over http://.`);
  return res.json();
}

// Pure-JS Matérn-5/2 GP, matching the exported scikit-learn parameters exactly.
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
}
function designToModel(design) {
  const { nx, nz } = gridDims(design.params.diagrid_divisions);
  return {
    nodes: design.nodes.map((n) => n.xyz_m),
    elements: design.elements.map((e) => ({ i: e.node_ids[0], j: e.node_ids[1], group: e.group_tag, util: e.group_peak_static_utilization })),
    nx, nz,
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
function makeGridTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 1024;
  const x = c.getContext('2d'); x.clearRect(0, 0, 1024, 1024);
  x.strokeStyle = 'rgba(82,98,122,0.38)'; x.lineWidth = 1;
  for (let i = 0; i <= 1024; i += 64) {
    x.beginPath(); x.moveTo(i, 0); x.lineTo(i, 1024); x.stroke();
    x.beginPath(); x.moveTo(0, i); x.lineTo(1024, i); x.stroke();
  }
  x.globalCompositeOperation = 'destination-in';
  const fade = x.createRadialGradient(512, 512, 100, 512, 512, 510);
  fade.addColorStop(0, 'rgba(0,0,0,0.8)'); fade.addColorStop(0.55, 'rgba(0,0,0,0.38)'); fade.addColorStop(1, 'rgba(0,0,0,0)');
  x.fillStyle = fade; x.fillRect(0, 0, 1024, 1024);
  return new THREE.CanvasTexture(c);
}
function makeContactShadowTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 256;
  const x = c.getContext('2d');
  const g = x.createRadialGradient(128, 128, 5, 128, 128, 126);
  g.addColorStop(0, 'rgba(15,35,66,0.20)'); g.addColorStop(0.55, 'rgba(15,35,66,0.08)'); g.addColorStop(1, 'rgba(15,35,66,0)');
  x.fillStyle = g; x.fillRect(0, 0, 256, 256);
  return new THREE.CanvasTexture(c);
}

let renderer, scene, camera, controls, structureGroup, ground, gridPlane, contactShadow, axesHelper;
let woodTexture, keyLight, hemiLight, rimLight;
const groupMeshes = {}, detailMeshes = [];
let lineMesh = null, nodeMesh = null, glazingMesh = null, glazingBase = null;
const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _mid = new THREE.Vector3();
const _dir = new THREE.Vector3(), _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3(), _mat = new THREE.Matrix4(), _up = new THREE.Vector3(0, 1, 0);
const _col = new THREE.Color();

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

function initScene() {
  const container = document.getElementById('viewport');
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.08;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f6f9);
  scene.fog = new THREE.Fog(0xf4f6f9, 80, 240);
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  camera = new THREE.PerspectiveCamera(42, container.clientWidth / container.clientHeight, 0.1, 600);
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
  gridPlane = new THREE.Mesh(new THREE.PlaneGeometry(240, 240), new THREE.MeshBasicMaterial({ map: makeGridTexture(), transparent: true, depthWrite: false, opacity: 0.62 }));
  gridPlane.rotation.x = -Math.PI / 2; gridPlane.position.y = -0.025; scene.add(gridPlane);
  contactShadow = new THREE.Mesh(new THREE.PlaneGeometry(65, 65), new THREE.MeshBasicMaterial({ map: makeContactShadowTexture(), transparent: true, depthWrite: false }));
  contactShadow.rotation.x = -Math.PI / 2; contactShadow.position.y = -0.015; scene.add(contactShadow);
  axesHelper = new THREE.AxesHelper(4); scene.add(axesHelper);
  woodTexture = makeWoodTexture();
  structureGroup = new THREE.Group(); scene.add(structureGroup);
  window.addEventListener('resize', onResize);
}
// -------------------- Structure, joints, glazing --------------------
function materialFor(group) {
  if (group === 'GLULAM_DIAGRID') return new THREE.MeshStandardMaterial({ map: woodTexture, bumpMap: woodTexture, bumpScale: 0.025, roughness: 0.62, metalness: 0.02 });
  if (group === 'SEISMIC_FUSE') return new THREE.MeshStandardMaterial({ color: 0xbcc7d1, metalness: 0.85, roughness: 0.20, emissive: new THREE.Color(0x0ea5c6), emissiveIntensity: 0.0 });
  return new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.72, roughness: 0.30 });
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
  // Bifurcation collars, baseplate pedestals, and fuse-end clevises.
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
  lineMesh = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x52718f, transparent: true, opacity: 0.8 }));
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
  const mat = new THREE.MeshPhysicalMaterial({ color: 0x9acbe8, roughness: 0.08, metalness: 0.0, transparent: true, opacity: 0.22, transmission: 0.12, thickness: 0.04, clearcoat: 1.0, clearcoatRoughness: 0.12, side: THREE.DoubleSide, depthWrite: false });
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
}
function onResize() {
  const container = document.getElementById('viewport');
  if (!container.clientWidth) return;
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

function utilizationColor(u) {
  const t = Math.min(Math.max((u || 0) / 1.2, 0), 1);
  const c0 = new THREE.Color(0x2b6cb0), c1 = new THREE.Color(0xd97706), c2 = new THREE.Color(0xc2410c);
  return t < 0.5 ? c0.lerp(c1, t / 0.5) : c1.lerp(c2, (t - 0.5) / 0.5);
}
function applyDisplayMode() {
  const mode = state.displayMode;
  if (lineMesh) lineMesh.visible = mode === 'wireframe' || mode === 'technical';
  if (nodeMesh) nodeMesh.visible = mode === 'technical';
  if (glazingMesh) glazingMesh.visible = state.glazing && (mode === 'architectural' || mode === 'utilization');
  for (const m of detailMeshes) m.visible = mode !== 'wireframe';
  for (const [group, { mesh, els }] of Object.entries(groupMeshes)) {
    mesh.visible = mode !== 'wireframe';
    const useUtil = mode === 'utilization' && state.mode === 'pareto';
    if (useUtil) {
      mesh.material.color.set(0xffffff);
      els.forEach((e, idx) => { const u = e.util ? (state.fuse === 'niti' ? e.util.niti : e.util.steel) : 0; mesh.setColorAt(idx, utilizationColor(u)); });
    } else {
      mesh.material.color.set(group === 'SEISMIC_FUSE' ? 0xbcc7d1 : mode === 'technical' ? 0x52718f : group === 'GLULAM_DIAGRID' ? 0xffffff : 0x1e293b);
      els.forEach((e, idx) => mesh.setColorAt(idx, _col.set(0xffffff)));
    }
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
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
      fm.emissive.set(0x0ea5c6); fm.emissiveIntensity = 0.06 + Math.min(2.2, (strain / 0.06) * 2.2);
    } else {
      if (strain > STEEL_YIELD_STRAIN) state.steelYielded = true;
      fm.emissive.set(0xc2410c);
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
}
// -------------------- HiDPI scientific charts --------------------
function cssVar(name) { return getComputedStyle(document.body).getPropertyValue(name).trim(); }
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 320, h = canvas.clientHeight || 240;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, dpr };
}
const linScale = (d0, d1, p0, p1) => (v) => p0 + (v - d0) * ((p1 - p0) / ((d1 - d0) || 1));
function chartColors() {
  return { text: cssVar('--chart-text') || '#52627a', grid: cssVar('--chart-grid') || '#d9e0e8', ink: cssVar('--ink') || '#0f2342', accent: cssVar('--accent') || '#0891b2' };
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
  const series = ['steel', 'niti'].map((fuse) => ({
    type: 'scatter', color: fuse === 'niti' ? '#0891b2' : '#c2410c', radius: 5.5,
    points: designs.filter((d) => d.front_case === fuse).map((d) => [d.params.span, d.kpis[fuse].c_total_per_m2]),
  }));
  const { ctx, sx, sy } = plotXY(canvas, { series, xlabel: 'Clear span (m)', ylabel: 'Carbon', title: 'Lifecycle carbon vs. clear span', ydec: 0 });
  paretoHits = designs.map((d) => ({ design: d, px: sx(d.params.span), py: sy(d.kpis[d.front_case].c_total_per_m2) }));
  for (const h of paretoHits) if (h.design === state.selectedDesign) {
    ctx.beginPath(); ctx.arc(h.px, h.py, 9.5, 0, 2 * Math.PI); ctx.strokeStyle = chartColors().ink; ctx.lineWidth = 2; ctx.stroke();
  }
  document.getElementById('chart-caption').textContent = 'Hover for exact KPIs · click to morph the canopy';
}
function nearestPareto(ev) {
  const rect = ev.target.getBoundingClientRect(), x = ev.clientX - rect.left, y = ev.clientY - rect.top;
  let best = null, bd = 1e9;
  for (const h of paretoHits) { const d = (h.px - x) ** 2 + (h.py - y) ** 2; if (d < bd) { bd = d; best = h; } }
  return { best, dist: Math.sqrt(bd), x, y };
}
document.getElementById('chart-pareto').addEventListener('mousemove', (ev) => {
  const { best, dist, x, y } = nearestPareto(ev), tip = document.getElementById('pareto-tooltip');
  if (!best || dist > 18) { tip.classList.add('hidden'); return; }
  const d = best.design, k = d.kpis[d.front_case];
  tip.innerHTML = `<strong>${d.id} · ${d.front_case}</strong><br>Span ${d.params.span.toFixed(2)} m · Carbon ${k.c_total_per_m2.toFixed(1)}<br>Peak ${k.peak_drift_pct.toFixed(2)}% · Residual ${k.abs_residual_drift_pct.toFixed(3)}%`;
  tip.style.left = `${Math.min(x + 12, 210)}px`; tip.style.top = `${Math.max(y - 65, 4)}px`; tip.classList.remove('hidden');
});
document.getElementById('chart-pareto').addEventListener('mouseleave', () => document.getElementById('pareto-tooltip').classList.add('hidden'));
function drawHysteresis() {
  const canvas = document.getElementById('chart-hysteresis'), design = state.selectedDesign;
  if (!design) { plotXY(canvas, { series: [{ type: 'line', color: '#94a3b8', points: [[0, 0], [1, 1]] }], xlim: [0, 1], ylim: [0, 1], title: 'Select a certified design first' }); return; }
  const frames = design.quake_frames.steel.lead_fuse_strain.length;
  const xvals = [], yvals = [];
  for (const fuse of ['steel', 'niti']) {
    const q = design.quake_frames[fuse];
    for (let i = 0; i < q.lead_fuse_strain.length; i++) { xvals.push(q.lead_fuse_strain[i]); yvals.push(q.lead_fuse_stress_MPa[i]); }
  }
  const xpad = (Math.max(...xvals) - Math.min(...xvals)) * 0.08 || 0.001;
  const ypad = (Math.max(...yvals) - Math.min(...yvals)) * 0.08 || 10;
  const upTo = state.playing || state.playFrame > 0 ? state.playFrame + 1 : frames;
  const other = state.fuse === 'niti' ? 'steel' : 'niti';
  const series = [];
  const ghost = design.quake_frames[other];
  series.push({ type: 'line', color: '#94a3b8', width: 1, alpha: 0.45, points: ghost.lead_fuse_strain.map((e, i) => [e, ghost.lead_fuse_stress_MPa[i]]) });
  const q = design.quake_frames[state.fuse], activeColor = state.fuse === 'niti' ? '#0891b2' : '#c2410c';
  const pts = [];
  for (let i = 0; i < upTo; i++) pts.push([q.lead_fuse_strain[i], q.lead_fuse_stress_MPa[i]]);
  series.push({ type: 'line', color: activeColor, width: 1.9, points: pts });
  const { ctx, sx, sy } = plotXY(canvas, { series, xlim: [Math.min(...xvals) - xpad, Math.max(...xvals) + xpad], ylim: [Math.min(...yvals) - ypad, Math.max(...yvals) + ypad], xlabel: 'Lead fuse strain ε', ylabel: 'Stress σ (MPa)', title: `${state.fuse.toUpperCase()} live · ${other} ghost`, xdec: 4, ydec: 0 });
  const f = Math.min(state.playFrame, frames - 1), ex = q.lead_fuse_strain[f], sig = q.lead_fuse_stress_MPa[f];
  if (state.playing || state.playFrame > 0) {
    ctx.beginPath(); ctx.arc(sx(ex), sy(sig), 5.5, 0, 2 * Math.PI); ctx.fillStyle = activeColor; ctx.shadowColor = activeColor; ctx.shadowBlur = 12; ctx.fill(); ctx.shadowBlur = 0;
  }
  document.getElementById('chart-caption').textContent = `t = ${q.time_s[f].toFixed(2)} s   ·   ε = ${ex.toFixed(5)}   ·   σ = ${sig.toFixed(1)} MPa`;
}
function drawDiffID() {
  const diff = state.data.diff, canvas = document.getElementById('chart-diffid'), loops = diff.loops;
  plotXY(canvas, {
    series: [
      { type: 'scatter', color: '#94a3b8', radius: 1.3, alpha: 0.45, points: loops.strain.map((s, i) => [s, loops.observed_stress_MPa[i]]) },
      { type: 'line', color: chartColors().ink, width: 1.6, points: loops.strain.map((s, i) => [s, loops.true_stress_MPa[i]]) },
      { type: 'line', color: '#0891b2', width: 1.5, dash: [4, 3], points: loops.strain.map((s, i) => [s, loops.recovered_stress_MPa[i]]) },
    ],
    xlabel: 'Strain ε', ylabel: 'Stress (MPa)', title: '3% noise · true vs. PyTorch recovered', xdec: 3, ydec: 0,
  });
  const tp = diff.true_parameters, rp = diff.recovered_parameters, ci = diff.confidence_intervals_95;
  const rows = PARAM_ORDER.map((n) => `<tr><td>${n}</td><td>${tp[n].toPrecision(5)}</td><td>${rp[n].toPrecision(5)}</td><td>${diff.relative_recovery_error_pct[n].toFixed(2)}%</td><td>[${ci[n][0].toPrecision(4)}, ${ci[n][1].toPrecision(4)}]</td></tr>`).join('');
  document.getElementById('chart-caption').innerHTML = `<table class="param-table"><thead><tr><th>Param</th><th>True</th><th>Recovered</th><th>Error</th><th>95% CI</th></tr></thead><tbody>${rows}</tbody></table>`;
}
// -------------------- Telemetry and selection --------------------
function setConfidence(text, kind) {
  const el = document.getElementById('conf-pill'); el.textContent = text; el.className = `pill ${kind}`;
}
function updateTelemetry(design) {
  const k = design.kpis[state.fuse], other = design.kpis[state.fuse === 'niti' ? 'steel' : 'niti'];
  document.getElementById('kpi-carbon').textContent = `${k.c_total_per_m2.toFixed(1)} kgCO₂e/m²`;
  document.getElementById('kpi-carbon-sub').textContent = `A1–A3 ${k.c_a1_a3_kgco2e.toFixed(0)} · Repair ${k.c_repair_kgco2e.toFixed(0)} kgCO₂e`;
  document.getElementById('kpi-drift').textContent = `${k.peak_drift_pct.toFixed(2)} / ${k.abs_residual_drift_pct.toFixed(3)} %`;
  document.getElementById('kpi-drift-sub').textContent = `Δ peak ${(other.peak_drift_pct - k.peak_drift_pct).toFixed(2)} · Δ resid ${(other.abs_residual_drift_pct - k.abs_residual_drift_pct).toFixed(3)}`;
  const utils = Object.entries(k.peak_member_stress_utilization).sort((a, b) => b[1] - a[1]);
  document.getElementById('kpi-util').textContent = utils[0][1].toFixed(2);
  document.getElementById('kpi-util-sub').textContent = `${utils[0][0]} · ${state.fuse}`;
  document.getElementById('conf-sub').textContent = `${design.id} · exact solver`;
  setConfidence('Exact OpenSeesPy', 'pill-ok');
}
function updateReadout() {
  const el = document.getElementById('design-readout'), d = state.selectedDesign;
  if (!d) { el.innerHTML = '<em>No design selected.</em>'; return; }
  const k = d.kpis[state.fuse];
  el.innerHTML = `<table>
    <tr><td>Design</td><td>${d.id} · ${d.front_case}</td></tr>
    <tr><td>Span / divisions</td><td>${d.params.span.toFixed(2)} m · ${d.params.diagrid_divisions}</td></tr>
    <tr><td>Glulam depth</td><td>${d.params.glulam_depth.toFixed(3)} m</td></tr>
    <tr><td>NiTi fuse area</td><td>${d.params.niti_fuse_area.toFixed(0)} mm²</td></tr>
    <tr><td>Peak drift</td><td>${k.peak_drift_pct.toFixed(2)} %</td></tr>
    <tr><td>Residual drift</td><td>${k.abs_residual_drift_pct.toFixed(3)} %</td></tr>
    <tr><td>Carbon</td><td>${k.c_total_per_m2.toFixed(1)} kgCO₂e/m²</td></tr>
    <tr><td>First period</td><td>${k.T1_s.toFixed(3)} s</td></tr>
  </table>`;
}
function selectDesign(design) {
  state.selectedDesign = design; state.fuse = design.front_case;
  state.playing = false; state.playFrame = 0; state.playClock = 0; state.steelYielded = false;
  state.model = designToModel(design);
  rebuildStructure();
  document.querySelectorAll('#replay-fuse button, #explore-fuse button').forEach((b) => b.classList.toggle('active', b.dataset.fuse === state.fuse));
  document.getElementById('scrub').value = '0';
  document.getElementById('scrub-time').textContent = '0.00s';
  updatePlayButton(); updateReadout(); updateTelemetry(design); drawActiveChart();
}
function exploreParams() { const p = {}; for (const s of SLIDERS) p[s.key] = s.value; return p; }
function applyExploreModel() {
  const p = exploreParams(); state.model = buildCanopy(p); rebuildStructure();
  const caseData = state.data.gp.cases[state.fuse];
  const features = [p.span, p.arch_rise_ratio, p.diagrid_divisions, p.glulam_depth, p.niti_fuse_area];
  const out = {}; for (const t of TARGETS) out[t] = gpPredict(caseData, t, features);
  const pd = out.peak_drift_pct, ar = out.abs_residual_drift_pct, gd = out.grav_defl_mm, cc = out.c_total_per_m2;
  document.getElementById('kpi-carbon').textContent = `${cc.mean.toFixed(1)} kgCO₂e/m²`;
  document.getElementById('kpi-carbon-sub').textContent = `± ${(1.96 * cc.std).toFixed(1)} · GP lifecycle`;
  document.getElementById('kpi-drift').textContent = `${pd.mean.toFixed(2)} / ${ar.mean.toFixed(3)} %`;
  document.getElementById('kpi-drift-sub').textContent = `± ${(1.96 * pd.std).toFixed(2)} / ± ${(1.96 * ar.std).toFixed(3)}`;
  document.getElementById('kpi-util').textContent = '—';
  document.getElementById('kpi-util-sub').textContent = `gravity ${gd.mean.toFixed(1)} ± ${(1.96 * gd.std).toFixed(1)} mm`;
  document.getElementById('conf-sub').textContent = 'Matérn 5/2 surrogate';
  const ucb = pd.mean + 1.96 * pd.std;
  const lo = caseData.training_domain.minimum, hi = caseData.training_domain.maximum;
  const outside = features.some((v, i) => v < lo[i] || v > hi[i]);
  if (outside) setConfidence('Outside training envelope', 'pill-warn');
  else if (ucb > 2.0) setConfidence(`95% UCB ${ucb.toFixed(2)}% > 2.0%`, 'pill-danger');
  }
// -------------------- Playback, theme, export --------------------
let lastTs = 0, exploreDirty = false;
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
    const frameMs = 300 / state.speed;
    if (state.playClock >= frameMs) {
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
  if (state.mode === 'explore' && exploreDirty) { exploreDirty = false; applyExploreModel(); }
  controls.update(); renderer.render(scene, camera);
}
function setTheme(theme) {
  state.theme = theme; document.body.className = `theme-${theme}`;
  const dark = theme === 'dark';
  scene.background.set(dark ? 0x0b1120 : 0xf4f6f9); scene.fog.color.set(dark ? 0x0b1120 : 0xf4f6f9);
  ground.material.color.set(dark ? 0x0e1729 : 0xe9edf2);
  gridPlane.material.opacity = dark ? 0.42 : 0.62;
  keyLight.intensity = dark ? 1.8 : 2.6; hemiLight.intensity = dark ? 0.55 : 1.05;
  document.getElementById('toggle-theme').textContent = dark ? 'Light Studio' : 'Dark Lab';
  drawActiveChart();
}
function exportHighResPNG() {
  const container = document.getElementById('viewport'), oldRatio = renderer.getPixelRatio();
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setPixelRatio(Math.min((window.devicePixelRatio || 1) * 2, 4));
  renderer.setSize(w, h, false); renderer.render(scene, camera);
  const a = document.createElement('a'); a.href = renderer.domElement.toDataURL('image/png'); a.download = 'diagrid_studio_2x.png'; a.click();
  }
// -------------------- UI wiring and startup --------------------
function buildSliderUI() {
  const host = document.getElementById('sliders'); host.innerHTML = '';
  for (const s of SLIDERS) {
    const row = document.createElement('div'); row.className = 'slider-row';
    const label = document.createElement('label');
    label.innerHTML = `<span>${s.label}</span><span class="val">${s.value.toFixed(s.decimals)}</span>`;
    const input = document.createElement('input');
    input.type = 'range'; input.min = s.min; input.max = s.max; input.step = s.step; input.value = s.value;
    input.addEventListener('input', () => { s.value = parseFloat(input.value); label.querySelector('.val').textContent = s.value.toFixed(s.decimals); exploreDirty = true; });
    row.appendChild(label); row.appendChild(input); host.appendChild(row);
  }
}
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll('#mode-toggle button').forEach((b) => b.classList.toggle('active', b.dataset.mode === mode));
  document.getElementById('pareto-controls').classList.toggle('hidden', mode !== 'pareto');
  document.getElementById('explore-controls').classList.toggle('hidden', mode !== 'explore');
  if (mode === 'explore') { stopPlayback(); applyExploreModel(); }
  else if (state.selectedDesign) { state.model = designToModel(state.selectedDesign); rebuildStructure(); updateReadout(); updateTelemetry(state.selectedDesign); }
  applyDisplayMode(); drawActiveChart();
}
function setFuse(fuse) {
  state.fuse = fuse; state.steelYielded = false;
  document.querySelectorAll('#replay-fuse button, #explore-fuse button').forEach((b) => b.classList.toggle('active', b.dataset.fuse === fuse));
  if (state.mode === 'explore') applyExploreModel();
  else if (state.selectedDesign) { updateTelemetry(state.selectedDesign); updateReadout(); applyFrameDisplacements(state.playFrame); }
  drawActiveChart();
}
function wireUI() {
  document.querySelectorAll('#mode-toggle button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  document.querySelectorAll('[data-display]').forEach((b) => b.addEventListener('click', () => {
    state.displayMode = b.dataset.display;
    document.querySelectorAll('[data-display]').forEach((x) => x.classList.toggle('active', x === b)); applyDisplayMode();
  }));
  document.querySelectorAll('#chart-tabs button').forEach((b) => b.addEventListener('click', () => {
    state.chartTab = b.dataset.tab;
    document.querySelectorAll('#chart-tabs button').forEach((x) => x.classList.toggle('active', x === b));
    document.querySelectorAll('.chart-canvas').forEach((c) => c.classList.add('hidden'));
    document.getElementById(`chart-${state.chartTab}`).classList.remove('hidden'); drawActiveChart();
  }));
  document.querySelectorAll('#replay-fuse button, #explore-fuse button').forEach((b) => b.addEventListener('click', () => setFuse(b.dataset.fuse)));
  document.getElementById('play-quake').addEventListener('click', () => { if (state.playing) stopPlayback(); else startPlayback(); });
  document.getElementById('scrub').addEventListener('input', (ev) => {
    if (!state.selectedDesign) return;
    state.playFrame = parseInt(ev.target.value, 10); applyFrameDisplacements(state.playFrame);
    document.getElementById('scrub-time').textContent = `${state.selectedDesign.quake_frames[state.fuse].time_s[state.playFrame].toFixed(2)}s`;
    if (state.chartTab === 'hysteresis') drawHysteresis();
  });
  document.querySelectorAll('#speed button').forEach((b) => b.addEventListener('click', () => {
    state.speed = parseFloat(b.dataset.speed); document.querySelectorAll('#speed button').forEach((x) => x.classList.toggle('active', x === b));
  }));
  document.querySelectorAll('#mag button').forEach((b) => b.addEventListener('click', () => {
    state.mag = parseInt(b.dataset.mag, 10); document.querySelectorAll('#mag button').forEach((x) => x.classList.toggle('active', x === b));
    applyFrameDisplacements(state.playFrame);
  }));
  document.querySelectorAll('[data-cam]').forEach((b) => b.addEventListener('click', () => setCameraPreset(b.dataset.cam)));
  document.getElementById('toggle-glazing').addEventListener('click', (ev) => { state.glazing = !state.glazing; ev.target.classList.toggle('active', state.glazing); applyDisplayMode(); });
  document.getElementById('toggle-grid').addEventListener('click', (ev) => { state.grid = !state.grid; ev.target.classList.toggle('active', state.grid); gridPlane.visible = state.grid; });
  document.getElementById('toggle-axes').addEventListener('click', (ev) => { state.axes = !state.axes; ev.target.classList.toggle('active', state.axes); axesHelper.visible = state.axes; });
  document.getElementById('toggle-theme').addEventListener('click', () => setTheme(state.theme === 'light' ? 'dark' : 'light'));
  document.getElementById('export-png').addEventListener('click', exportHighResPNG);
  document.getElementById('rail-collapse').addEventListener('click', () => document.getElementById('rail').classList.toggle('collapsed'));
  window.addEventListener('resize', () => setTimeout(drawActiveChart, 100));
}
async function init() {
  initScene(); buildSliderUI(); wireUI();
  const loading = document.getElementById('loading');
  try {
    const [gp, pareto, diff] = await Promise.all([loadJSON(DATA_FILES.gp), loadJSON(DATA_FILES.pareto), loadJSON(DATA_FILES.diff)]);
    state.data = { gp, pareto, diff }; loading.classList.add('hidden');
  } catch (err) { loading.classList.add('error'); loading.textContent = `Failed to load Day 4 artifacts:\n${err.message}`; return; }
  if (state.data.pareto.designs.length) selectDesign(state.data.pareto.designs[0]);
  requestAnimationFrame(animate);
}
init();









