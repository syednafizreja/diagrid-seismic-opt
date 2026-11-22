# Engineering Specification: diagrid-seismic-opt (`diagrid_opt`)

## 1. Scope

Parametric design, nonlinear seismic analysis, and whole-life carbon optimization of a 3D curved glulam-timber and steel diagrid canopy supported on four branching tree columns, comparing:

- **Case A (Steel):** Conventional yielding steel braces (`Steel02`).
- **Case B (NiTi):** Superelastic NiTi shape-memory alloy self-centering fuses (`SelfCentering`).

## 2. Exact 1D Superelastic NiTi Constitutive Model (Auricchio-Sacco / Souza-Auricchio)

Implement this exact rate-independent 1D model using signed martensite fraction `z` in `[-1, 1]` (handles tension, compression, and partial sub-loops).

### Public Literature Baseline Parameters (DesRoches et al., 2004; Auricchio & Sacco, 1997)

- `E = 40_000.0` MPa (elastic modulus)
- `eps_L = 0.06` (maximum transformation strain)
- `sigma_Ms = 380.0` MPa, `sigma_Mf = 450.0` MPa (`H_f = sigma_Mf - sigma_Ms = 70.0` MPa)
- `sigma_As = 180.0` MPa, `sigma_Af = 110.0` MPa (`H_r = sigma_As - sigma_Af = 70.0` MPa)

### Return-Mapping Driver: `update(eps_new, z_n, params) -> (sigma_new, z_new, E_alg)`

1. **Elastic trial:**
   - `sigma_tr = E * (eps_new - eps_L * z_n)`
   - `s_tr = 1.0 if sigma_tr >= 0.0 else -1.0`
   - `s_n = 1.0 if z_n >= 0.0 else -1.0`
2. **Threshold functions:**
   - `sigma_f(xi) = sigma_Ms + H_f * xi`
   - `sigma_r(xi) = sigma_Af + H_r * xi`
3. **Regime checks:**
   - **Reverse transformation (M -> A):** Active if `abs(z_n) > 0.0` and `s_n * sigma_tr < sigma_r(abs(z_n))`.
     - `dxi_rev = (sigma_r(abs(z_n)) - s_n * sigma_tr) / (E * eps_L + H_r)`
     - If `dxi_rev < abs(z_n)`:
       - `z_new = s_n * (abs(z_n) - dxi_rev)`
       - `sigma_new = E * (eps_new - eps_L * z_new)`
       - `E_alg = (E * H_r) / (E * eps_L + H_r)`
       - Return `(sigma_new, z_new, E_alg)`.
     - Else (fully unloaded to `z = 0.0`):
       - Set `z_n = 0.0`, recompute `sigma_tr = E * eps_new`, `s_tr = 1.0 if sigma_tr >= 0.0 else -1.0`, and continue to Forward check.
   - **Forward transformation (A -> M):** Active if `abs(z_n) < 1.0` and `abs(sigma_tr) > sigma_f(abs(z_n))` and (`z_n == 0.0` or `s_tr == s_n`).
     - `dxi_fwd = (abs(sigma_tr) - sigma_f(abs(z_n))) / (E * eps_L + H_f)`
     - If `abs(z_n) + dxi_fwd < 1.0`:
       - `z_new = s_tr * (abs(z_n) + dxi_fwd)`
       - `sigma_new = E * (eps_new - eps_L * z_new)`
       - `E_alg = (E * H_f) / (E * eps_L + H_f)`
     - Else (saturated martensite `|z| = 1.0`):
       - `z_new = s_tr * 1.0`
       - `sigma_new = E * (eps_new - eps_L * z_new)`
       - `E_alg = E`
     - Return `(sigma_new, z_new, E_alg)`.
   - **Elastic step:** Return `(sigma_tr, z_n, E)`.

### Thermodynamic Consistency (Clausius-Duhem Inequality)

- `sigma_0 = 0.5 * (sigma_Ms + sigma_Af)`, `H_eq = 0.5 * (H_f + H_r)`
- Helmholtz free energy: `psi(eps, z) = 0.5 * E * (eps - eps_L * z)**2 + eps_L * (sigma_0 * abs(z) + 0.5 * H_eq * z**2)`
- Discrete step dissipation: `D_step = 0.5 * (sigma_n + sigma_new) * (eps_new - eps_n) - (psi_new - psi_n)`
- Must satisfy `D_step >= -1e-9` at every increment.

## 3. Exact 3D Parametric Diagrid Canopy Geometry

- Design parameters:
  - `span` (`L` in `[14.0, 28.0]` m along X), `width = 0.5 * L` along Z
  - `arch_rise` (`f` in `[0.08 * L, 0.22 * L]` m)
  - `diagrid_divisions` (`nx` even integer in `[6, 12]`, `nz = max(4, nx // 2)`)
  - `branch_spread` (`s_b` in `[0.12 * L, 0.22 * L]` m)
  - `glulam_depth` (`h_g` in `[0.22, 0.48]` m, width `b_g = 0.16` m)
  - `niti_fuse_area` (`A_niti` in `[60.0, 260.0]` mm^2, baseline `100.0` mm^2)
- Roof surface (`H_clear = 4.5` m):
  - `y(x, z) = H_clear + arch_rise * (1.0 - (2.0 * x / L)**2) * (1.0 - 0.5 * (2.0 * z / width)**2)`
- Roof grid: `(nx + 1) x (nz + 1)` nodes over `x in [-L/2, L/2]`, `z in [-width/2, width/2]`.
- Element groups:
  - `GLULAM_DIAGRID`: X-braced diagonal members across every grid cell `(i,j)-(i+1,j+1)` and `(i+1,j)-(i,j+1)`.
  - `STEEL_EDGE`: perimeter beams along `i = 0, nx` and `j = 0, nz`.
  - `TREE_TRUNK`: 4 vertical steel columns at `(±0.32 * L, 0.0, ±0.25 * width)` rising from `y = 0.0` to bifurcation node `y_b = 0.55 * H_clear`.
  - `TREE_BRANCH`: 4 inclined steel struts per column connecting its bifurcation node to the 4 nearest distinct roof grid nodes.
  - `SEISMIC_FUSE`: 2 diagonal fuse bars per column connecting its bifurcation node to ground anchor nodes at `(x_col ± 1.5, 0.0, z_col)`.

## 4. Day 1 Verification Gates

1. `E_alg` matches central finite differences within `< 1e-7` relative error on smooth branches.
2. 1D Newton-Raphson equilibrium solver achieves `|r| < 1e-12` in `<= 5` iterations (quadratic convergence).
3. `D_step >= -1e-9` across 5 reversed cyclic loops (`±5%` strain).
4. Zero floating nodes, zero zero-length elements, exact 4-quadrant symmetry in 3D canopy geometry.

## 5. Day 2 Specification: 3D OpenSeesPy Multi-Hazard Solver (`solver_opensees.py`)

### Units & Coordinate Convention (Strict SI)
- Length: `m`, Force: `N`, Stress/Modulus: `Pa` (multiply MPa by `1e6`), Mass: `kg`, Time: `s`.
- Global vertical axis is **Y** (gravity acts in `-Y`, wind uplift in `+Y`, horizontal seismic excitation in `X`).
- `ops.model('basic', '-ndm', 3, '-ndf', 6)`
- Boundary conditions:
  - Column base nodes (`TREE_TRUNK` bottom at `y = 0`): fixed `(1, 1, 1, 1, 1, 0)`, releasing `Rz` in the X-Y seismic sway plane while retaining out-of-plane stability.
  - Fuse ground anchor nodes (`SEISMIC_FUSE` bottom at `y = 0`): pinned `(1, 1, 1, 0, 0, 0)`.
- `geomTransf` rule (prevents collinear vector crash):
  - Transf tag `1` (`Linear` / `PDelta`) for vertical members (`|dy/L| > 0.9`): `vecxz = (0.0, 0.0, 1.0)`.
  - Transf tag `2` (`Linear` / `PDelta`) for roof & inclined members (`|dy/L| <= 0.9`): `vecxz = (0.0, 1.0, 0.0)`.

### Structural Sections & Elastic Properties
- `GLULAM_DIAGRID` (GL28h timber): `E = 12.6e9` Pa, `G = 0.65e9` Pa, `rho = 420.0` kg/m³, rectangular `b_g x h_g` (`b_g = 0.16` m, `h_g = glulam_depth`), `f_m_d = 18.0e6` Pa.
- `STEEL_EDGE` (S355 SHS 250x250x10): `E = 210e9` Pa, `G = 81e9` Pa, `rho = 7850.0` kg/m³, `A = 9.49e-3` m², `Iz = Iy = 9.06e-5` m⁴, `J = 1.41e-4` m⁴.
- `TREE_TRUNK` (S355 CHS 323.9x12.5): `E = 210e9` Pa, `G = 81e9` Pa, `A = 1.22e-2` m², `Iz = Iy = 1.48e-4` m⁴, `J = 2.97e-4` m⁴.
- `TREE_BRANCH` (S355 CHS 193.7x8.0): `E = 210e9` Pa, `G = 81e9` Pa, `A = 4.67e-3` m², `Iz = Iy = 2.02e-5` m⁴, `J = 4.03e-5` m⁴.
- Roof dead load cladding/glazing/purlin: `1.50e3` N/m² lumped as nodal gravity loads and translational nodal masses `(m, m, m, 1e-3, 1e-3, 1e-3)` on roof nodes.

### Equal-Strength Seismic Fuse Comparison (`corotTruss` elements)
- **Case B (`fuse_type='niti'`):** `SelfCentering` calibrated directly to Day 1 Auricchio-Sacco parameters:
  - `k1 = 40.0e9` Pa
  - `k2 = (40.0e9 * 70.0e6) / (40.0e9 * 0.06 + 70.0e6)` (`~1.1336e9` Pa)
  - `sigAct = 380.0e6` Pa
  - `beta = (380.0 - 110.0) / 380.0` (`0.7105`)
  - `epsSlip = 0.0`, `epsBear = 0.071`, `rBear = 1.0`
  - Baseline area `A_niti = 100.0` mm² (11.3 mm diameter rod), with `A = A_niti * 1e-6` m².
- **Case A (`fuse_type='steel'`):** `Steel02` sized for **equal design yield force** (`Fy * A_steel = sigAct * A_niti`):
  - `Fy = 355.0e6` Pa, `E0 = 200.0e9` Pa, `b = 0.01`, `R0 = 18.0`, `cR1 = 0.925`, `cR2 = 0.15`
  - `A_steel = (380.0 / 355.0) * (A_niti * 1e-6)` m²

### Multi-Hazard Load Cases
1. **Static ULS Gravity:** `1.35 * G + 1.50 * Q` (`Q = 0.75e3` N/m² downward `-Y`).
2. **Static ULS Wind Uplift (Load Reversal):** `0.90 * G + 1.50 * W_up` (`W_up = +1.20e3` N/m² upward `+Y`).
3. **Nonlinear Seismic Time-History (`0.25g` and `0.40g`):**
   - Apply `1.0 * G` gravity first, hold constant (`loadConst('-time', 0.0)`), apply 5% Rayleigh damping targeting modes 1 & 2, then run `UniformExcitation` along X (`dof = 1`) using `Newmark(0.5, 0.25)` with `dt = 0.01` s.
   - Use 3 deterministic synthetic spectrum-compatible ground motions (`src/diagrid_opt/ground_motions.py`, seeds `101, 202, 303`): 12.0 s strong shaking (broadband multi-frequency modulated by a Jennings-Housner envelope, peak-normalized to `PGA`) followed by **6.0 s of zero acceleration** (free-vibration tail) so permanent **residual roof drift** settles cleanly.

### Day 2 Verification Gates
1. Global static equilibrium error `|sum(F_applied) + sum(R_support)| / |sum(F_applied)| < 1e-4` for both Gravity and Wind Uplift.
2. Fundamental period `T1` is physically plausible (`0.15 s <= T1 <= 1.50 s`).
3. Nonlinear `0.40g` time-history converges across all 3 records for both `steel` and `niti` without NaN/divergence.
4. Mean permanent residual drift of `niti` is strictly lower than `steel` under `0.40g` shaking.

## 6. Day 3 Specification: Whole-Life Carbon and Verified LHS Dataset

### A1-A3 Material Inventory
- Calculate member volumes from each geometry element's actual length and the structural section area used by the OpenSees model. Do not use approximate steel volumes from `CanopyGeometry.volumes_m3`.
- Glulam density is `420.0 kg/m³`; fossil-only A1-A3 factor is `0.512 kgCO2e/kg`. Provide an optional biogenic sensitivity in which the glulam factor is `-0.98 kgCO2e/kg` instead.
- Structural steel density is `7850.0 kg/m³`, with A1-A3 factor `1.55 kgCO2e/kg`.
- NiTi density is `6450.0 kg/m³`, with A1-A3 factor `28.0 kgCO2e/kg`.
- Steel-fuse area is calibrated for equal yield force as specified in Section 5; use the corresponding steel density and carbon factor for its inventory.
- Fixed concrete footings contribute `1612.8 kgCO2e`.

### Repair Carbon for One Design Earthquake
- Steel fuses are replaced when maximum absolute fuse strain is strictly greater than `0.00355` (0.355%); replacement carbon is `1.5 *` the A1-A3 carbon of the installed fuse material.
- NiTi fuses are replaced when maximum absolute fuse strain is strictly greater than `0.06`; below or at this recoverable strain, fuse replacement carbon is zero.
- Glazing and connection inventory is `45 kgCO2e/m²` of shaded roof area. Let `d_peak` be peak roof drift ratio and `r_glazing` be the damage ratio:
  - `r_glazing = 0.0` when `d_peak <= 0.005`.
  - `r_glazing = 0.35 * (d_peak - 0.005) / (0.025 - 0.005)` when `0.005 < d_peak <= 0.025`.
  - `r_glazing = 0.60` when `d_peak > 0.025`.
  - Glazing repair carbon is `45 * shaded_area_m2 * r_glazing`. Apply these literal thresholds, including the specified change above 2.5%.
- Let `d_res` be the permanent residual roof drift ratio and `C_superstructure` be the superstructure A1-A3 carbon excluding footings. Residual repair carbon is `min(1.0, d_res / 0.005) * 0.40 * C_superstructure`.
- `C_repair` is the sum of fuse replacement, glazing/connection repair, and residual repair carbon. Total carbon intensity is `(C_A1-A3 + C_repair) / shaded_area_m2`, in `kgCO2e/m²`.

### Reproducible Verified Dataset
- Generate `N = 80` designs using `scipy.stats.qmc.LatinHypercube(d=5, seed=42)`.
- Map the five dimensions to span `[14.0, 28.0] m`, arch-rise ratio `[0.08, 0.22]`, divisions `{6, 8, 10, 12}`, glulam depth `[0.22, 0.48] m`, and NiTi fuse area `[60.0, 260.0] mm²`. Use an in-range deterministic `branch_spread` because it is not sampled.
- For every design and both `steel` and `niti`, run static gravity and uplift checks and a `0.40g`, seed `101` seismic analysis. A case is verified only if the seismic analysis converges and both static equilibrium errors are below `1e-4`.
- Preserve the signed mean roof drift over the final 1.0 s separately from its absolute magnitude so the Steel/NiTi residual-drift comparison retains direction.
- Run independent design/fuse cases in isolated worker processes; OpenSees model state must not be shared between workers.
- Save paired, fully verified designs to `data/verified_lhs_dataset.csv`. Exclude a design unless both fuse cases pass, keep deterministic design ordering, and report excluded cases with their failure reasons.

## 7. Day 4 Specification: GP Surrogate, NSGA-II Pareto Optimization, and Differentiable NiTi Identification

### 7.1 Matérn-5/2 Gaussian-Process Surrogate (`surrogate_gp.py`)
- Features: `span_m`, `arch_rise_ratio`, `diagrid_divisions`, `glulam_depth_m`, `niti_fuse_area_mm2` (the five LHS design dimensions).
- Targets per fuse case: `peak_drift_pct`, `abs_residual_drift_pct`, `grav_defl_mm`, `c_total_per_m2`.
- Kernel `ConstantKernel * Matern(nu=2.5) + WhiteKernel` with `StandardScaler` feature standardization.
- Grouped 4-fold cross-validation via `GroupKFold` over span quartiles (no leakage between span groups), producing 20/20/20/20 held-out folds.
- Verification gates:
  1. Exported weights (`alpha`, `cholesky`, `constant_value`, `length_scale`, `noise_level`), `scaler`, and `X_train_standardized` reproduce scikit-learn `predict` mean/std to `< 1e-6` (browser-replayable Matérn-5/2 evaluation).
  2. The false-safe peak-drift rate using the 95% upper confidence bound (`mean + 1.96σ`) is strictly lower than the mean-only rate (`ucb_95 < mean`); a design is unsafe when true peak drift exceeds `2.0%`.
  3. Grouped-CV R² is reported per target for `docs/VALIDATION.md`, and out-of-fold peak-drift `(truth, mean, std)` is exported under `validation.oof_peak_drift_pct`.

### 7.2 NSGA-II Multi-Objective Pareto Optimization (`optimize_nsga2.py`)
- Two objectives: minimize lifecycle carbon `c_total_per_m2` and maximize clear span `span` (minimize `-span`).
- Conservative 95% UCB constraints: `peak_drift_pct + 1.96σ <= 2.0%` and `grav_defl_mm + 1.96σ <= span*1000/250`.
- `diagrid_divisions` snapped to the nearest allowed even value `{6, 8, 10, 12}`; `branch_spread = 0.17 * span`.
- NSGA-II, population 60, 60 generations, seed 42, run independently per fuse case.
- Certification: every exported front design is re-run through **exact OpenSeesPy** (static gravity + wind uplift and a 0.40 g, seed-101, 1800-step transient). No surrogate value is substituted for a certified KPI.
- Verification gates:
  1. Exactly **8** certified designs are exported — 4 `steel` + 4 `niti` front cases.
  2. Each design's exact static equilibrium error `< 1e-4` and `converged_steps == 1800`.
  3. **Zero** false-safe designs on the certified front: any candidate whose exact peak drift `> 2.0%` or exact gravity deflection exceeds its limit is rejected and counted in `meta.certification`.
  4. Each design carries `nodes`, `elements` (with per-group static utilizations), `kpis`, and a ~60-frame `quake_frames` (translational nodal displacements + lead-fuse strain/stress), rounded to 5 decimal places and serialized compactly for a Pages-friendly export (< 2 MB).

### 7.3 Differentiable NiTi Parameter Identification (`diff_ident_sciml.py`)
- Smooth, differentiable Auricchio-Sacco surrogate in PyTorch (`smooth_stress_history`); observed data are generated from the exact model with 3% Gaussian sensor noise on a ramped multi-amplitude (±1.5%→±6%) nested cyclic strain history.
- Optimization: Adam (300 steps) followed by L-BFGS (strong-Wolfe line search); parameters bounded via a logit/sigmoid map.
- Uncertainty: 20-member residual-resampling bootstrap → 95% percentile confidence intervals per parameter.
- Verification gates:
  1. Every recovered parameter (`E`, `sigma_Ms`, `sigma_Mf`, `sigma_As`, `sigma_Af`, `eps_L`) is within **< 8%** relative error of the true value (`meta.all_parameters_within_8_percent`).
  2. 95% CIs are reported for all six parameters.
