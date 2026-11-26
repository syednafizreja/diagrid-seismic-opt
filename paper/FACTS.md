# Verified facts — diagrid-seismic-opt

Audit date: 26 Nov 2026. Repository root: `diagrid-seismic-opt` (git `85a698c` on `main`). `paper/REPO_MAP.txt` is not in the tree. Every number below was read from source or from the committed artifacts. OpenSeesPy time histories and the PyTorch identification were not re-run. The constitutive tangent, dissipation, and Newton checks were recomputed from `constitutive_sma.py` because they are a few thousand scalar updates.

Units in the 1D model are MPa and dimensionless strain. The structural model is SI: m, N, Pa, kg, s.

---

## 1. Constitutive model and verification

The implemented model is the rate-independent 1D signed-martensite return map in `src/diagrid_opt/constitutive_sma.py`, with the Helmholtz energy and discrete dissipation in `src/diagrid_opt/thermodynamics.py`. `src/diagrid_opt/niti.py` is a second copy of the same update and energy. The structural time history does not call either function. See the discrepancies.

### Parameters

Default `NiTiParams`, also the identification targets:

| Symbol | Code field | Value |
|---|---|---|
| \(E\) | `E` | 40000 MPa |
| \(\varepsilon_L\) | `eps_L` | 0.06 |
| \(\sigma_{Ms}\) | `sigma_Ms` | 380 MPa |
| \(\sigma_{Mf}\) | `sigma_Mf` | 450 MPa |
| \(\sigma_{As}\) | `sigma_As` | 180 MPa |
| \(\sigma_{Af}\) | `sigma_Af` | 110 MPa |

Hardening moduli:

\[
H_f = \sigma_{Mf}-\sigma_{Ms} = 70\ \mathrm{MPa}, \qquad
H_r = \sigma_{As}-\sigma_{Af} = 70\ \mathrm{MPa}.
\]

Martensite fraction \(z\in[-1,1]\), \(\xi=|z|\). A state with \(|z_n|>1\) raises `ValueError`.

### Free energy

\[
\sigma_0 = \tfrac12(\sigma_{Ms}+\sigma_{Af}) = 245\ \mathrm{MPa}, \qquad
H_{eq} = \tfrac12(H_f+H_r) = 70\ \mathrm{MPa},
\]

\[
\psi(\varepsilon,z)
= \tfrac12 E\,(\varepsilon-\varepsilon_L z)^2
+ \varepsilon_L\left(\sigma_0\,|z| + \tfrac12 H_{eq}\, z^2\right).
\]

There is no temperature, entropy, or heat flux. The energy is a monitor. The return map is not a variational update of this \(\psi\).

### Trial stress and transformation functions

\[
\sigma^{\mathrm{tr}} = E\,(\varepsilon_{n+1}-\varepsilon_L z_n), \quad
s^{\mathrm{tr}}=\mathrm{sign}_+(\sigma^{\mathrm{tr}}), \quad
s_n=\mathrm{sign}_+(z_n),
\]

with \(\mathrm{sign}_+(0)=+1\). Thresholds:

\[
\sigma_f(\xi)=\sigma_{Ms}+H_f\xi, \qquad
\sigma_r(\xi)=\sigma_{Af}+H_r\xi.
\]

Forward and reverse measures used by the code:

\[
\Phi_f = |\sigma^{\mathrm{tr}}|-\sigma_f(\xi_n), \qquad
\Phi_r = \sigma_r(\xi_n)-s_n\sigma^{\mathrm{tr}}.
\]

### Internal-variable update

One closed-form step. No constitutive Newton iteration.

1. Reverse, if \(|z_n|>0\) and \(\Phi_r>0\):

\[
\Delta\xi_r = \frac{\sigma_r(\xi_n)-s_n\sigma^{\mathrm{tr}}}{E\varepsilon_L+H_r}.
\]

If \(\Delta\xi_r<\xi_n\), then \(z_{n+1}=s_n(\xi_n-\Delta\xi_r)\). If \(\Delta\xi_r\ge\xi_n\), set \(z_n=0\), recompute the trial from a clean austenite state, and continue.

2. Forward, if \(\xi_n<1\), \(\Phi_f>0\), and either \(z_n=0\) or \(s^{\mathrm{tr}}=s_n\):

\[
\Delta\xi_f = \frac{|\sigma^{\mathrm{tr}}|-\sigma_f(\xi_n)}{E\varepsilon_L+H_f}.
\]

If \(\xi_n+\Delta\xi_f<1\), then \(z_{n+1}=s^{\mathrm{tr}}(\xi_n+\Delta\xi_f)\). Otherwise \(z_{n+1}=s^{\mathrm{tr}}\) (saturated).

3. Otherwise the step is elastic: \(z_{n+1}=z_n\), \(\sigma= \sigma^{\mathrm{tr}}\).

Updated stress on every branch:

\[
\sigma = E\,(\varepsilon_{n+1}-\varepsilon_L z_{n+1}).
\]

### Algorithmic tangent

Piecewise constant on each open regime:

| Regime | \(E_{\mathrm{alg}}\) | Default value |
|---|---|---|
| Elastic, or saturated \(\lvert z\rvert=1\) | \(E\) | 40000 MPa |
| Partial forward | \(E H_f/(E\varepsilon_L+H_f)\) | 1133.6032388663969 MPa |
| Partial reverse | \(E H_r/(E\varepsilon_L+H_r)\) | 1133.6032388663969 MPa |

The two transformation tangents coincide because \(H_f=H_r\).

Stress-driven Newton (`constitutive_sma.newton_solve`) is \(r=\sigma(\varepsilon)- \sigma_{\mathrm{target}}\), \(\varepsilon\leftarrow\varepsilon-r/E_{\mathrm{alg}}\), tolerance \(10^{-12}\), limit 5 iterations. For \(\sigma_{\mathrm{target}}=500\) MPa and \(z_n=0\), recomputed on this audit: 4 iterations, \(\varepsilon=0.0725\), \(z=1\), \(r=4.547473508864641\times 10^{-13}\).

### Discrete dissipation

\[
D
= \tfrac12(\sigma_n+\sigma_{n+1})(\varepsilon_{n+1}-\varepsilon_n)
- \bigl(\psi_{n+1}-\psi_n\bigr).
\]

The gate in `tests/test_constitutive.py` and `tests/test_niti.py` is \(D\ge -10^{-9}\). Recomputed minima, strain \(0.05\sin\phi\):

| History | Minimum \(D\) (MPa) |
|---|---|
| 5 cycles, 5000 samples (`test_constitutive.py`) | \(-2.0235549347269455\times 10^{-15}\) |
| 5 cycles, 5001 samples (Pareto metadata) | \(-1.700029006457271\times 10^{-15}\) |
| 10 cycles, 5001 samples (`test_niti.py`) | \(-1.97758476261356\times 10^{-15}\) |

`data/pareto_canopies.json` stores `tangent_error_relative_max = 1.8504164817172542e-10` and `minimum_cyclic_dissipation_MPa = -1.700029006457271e-15`. Those match this recomputation.

### Tangent check

Central difference, \(h=10^{-7}\), four states in both constitutive tests. Relative error \(|(E_{\mathrm{alg}}-E_{\mathrm{num}})/E_{\mathrm{num}}|\):

| \((\varepsilon,z_n)\) | Relative error |
|---|---|
| \((0.001,\ 0)\) and \((-0.001,\ 0)\) | \(1.2221789802409672\times 10^{-12}\) |
| \((0.025,\ 0.2)\) and \((-0.025,\ -0.2)\) | \(1.8504164817172542\times 10^{-10}\) |

Gate: \(<10^{-7}\). Worst case on this set: \(1.8504164817172542\times 10^{-10}\).

### Test count

25 `test_*` functions under `tests/`. Three are parametrized over `steel` and `niti` (`test_fuse_replacement_uses_strict_strain_threshold`, `test_a1_a3_uses_physical_section_areas`, `test_static_load_cases_satisfy_equilibrium_and_period_gates`), so a collection is 28 tests. Five of the functions are constitutive (three in `test_constitutive.py`, two in `test_niti.py`). `test_solver.py` repeats the Newton gate through `niti.py`. This audit did not execute `pytest`.

---

## 2. Inverse parameter identification

Source: `src/diagrid_opt/diff_ident_sciml.py`. Saved result: `data/diff_id_uncertainty.json` (67,791 bytes). The fit was not repeated.

### Synthetic experiment

Truth is `NiTiParams()`. Strain, 601 samples:

\[
\phi\in[0,1],\qquad
\varepsilon(\phi)=0.06\,(0.25+0.75\phi)\,\sin(10\pi\phi).
\]

Amplitude grows from 1.5% to 6% over five cycles. Stress observations come from the exact return map plus Gaussian noise

\[
s_{\mathrm{noise}} = 0.03\,\max|\sigma_{\mathrm{true}}|
= 13.04089068825911\ \mathrm{MPa},
\]

seed 42 (`numpy` Generator). Meta field `sensor_noise_fraction` is 0.03.

### What is trained

`smooth_stress_history` is a differentiable stand-in: `tanh` signs, `softplus` driving forces, sigmoid gates, `transition_mpa=0.5`. It is not the exact return map. After the fit, `recovered_stress_MPa` is recomputed with the exact `niti_update` at the recovered parameters.

The optimizer sees an internal vector \((E,\sigma_{Ms},H_f,\sigma_{Af},H_r,\varepsilon_L)\), mapped from an unconstrained logit by a sigmoid into

| Internal | Bounds |
|---|---|
| \(E\) | [20000, 60000] MPa |
| \(\sigma_{Ms}\) | [250, 500] MPa |
| \(H_f\) | [20, 150] MPa |
| \(\sigma_{Af}\) | [50, 200] MPa |
| \(H_r\) | [20, 150] MPa |
| \(\varepsilon_L\) | [0.03, 0.09] |

Then \(\sigma_{Mf}=\sigma_{Ms}+H_f\) and \(\sigma_{As}=\sigma_{Af}+H_r\). The start point is \((38000,\ 360,\ 80,\ 120,\ 80,\ 0.058)\), which decodes to \(\sigma_{Mf}=440\) and \(\sigma_{As}=200\). Unconstrained parameters are clamped to \([-8,8]\) after each Adam step.

### Loss and optimizer

Twenty residual-bootstrap curves are stacked with the original noisy curve (21 curves). Residuals are resampled with replacement from the original noise. One shared loss trains all 21 parameter rows together:

\[
\mathcal{L}
= \mathrm{mean}\left(
\left(\frac{\sigma_{\mathrm{smooth}}(\varepsilon;\theta_b)-\sigma_{\mathrm{obs},b}}{s}\right)^2
\right),
\quad
s=\max|\sigma_{\mathrm{obs}}|.
\]

`torch.float64`, one CPU thread. Adam: learning rate 0.04, 300 steps. Then L-BFGS: `max_iter=80`, `history_size=20`, `tolerance_grad=1e-9`, `tolerance_change=1e-12`, `line_search_fn="strong_wolfe"`. The reported point is row 0. The 95% intervals are the 2.5 and 97.5 percentiles of rows 1 through 20 only.

### Recovered parameters

Relative error is \(|\hat\theta/\theta-1|\times 100\). Worst case is \(\varepsilon_L\) at 7.282%.

| Parameter | True | Recovered | Relative error | 95% bootstrap interval |
|---|---:|---:|---:|---|
| \(E\) (MPa) | 40000 | 39834.217965940596 | 0.414455% | [39384.518719156615, 40482.47987395222] |
| \(\sigma_{Ms}\) (MPa) | 380 | 381.73981037491853 | 0.457845% | [378.159803843313, 380.83015001417454] |
| \(\sigma_{Mf}\) (MPa) | 450 | 452.17299736798634 | 0.482888% | [450.84658363584845, 457.8180625563239] |
| \(\sigma_{As}\) (MPa) | 180 | 186.44724142943846 | 3.581801% | [171.08574203133736, 183.8202532317722] |
| \(\sigma_{Af}\) (MPa) | 110 | 111.71851873546828 | 1.562290% | [110.63563785423547, 117.14235455889079] |
| \(\varepsilon_L\) | 0.06 | 0.06436921016136132 | 7.282017% | [0.0627024932234163, 0.06575754035703371] |

`all_parameters_within_8_percent` is true. Coverage is not: the interval excludes the true \(\sigma_{Mf}\), \(\sigma_{Af}\), and \(\varepsilon_L\), and it excludes the recovered \(\sigma_{Ms}\) and \(\sigma_{As}\). Twenty bootstrap members are a small sample. `docs/VALIDATION.md` rounds this table and does not state the coverage misses.

---

## 3. Structural seismic and carbon benchmark

### Geometry

`GeometryParams` bounds, from `geometry.py`:

| Parameter | Bounds | Role |
|---|---|---|
| `span` \(L\) | [14, 28] m | Plan length in \(X\) |
| `arch_rise` | \([0.08L,\ 0.22L]\) | Crown rise |
| `diagrid_divisions` \(n_x\) | even integer in [6, 12] | Roof divisions in \(X\) |
| `branch_spread` | \([0.12L,\ 0.22L]\) | Stored only. The mesh ignores it |
| `glulam_depth` | [0.22, 0.48] m | Section depth. Width `b_glulam=0.16` m is fixed |
| `niti_fuse_area` | [60, 260] mm² | NiTi area. Default 100 mm² |

Width \(=0.5L\). Clear-height constant \(H=4.5\) m. Roof

\[
y(x,z)=4.5+\mathrm{arch\_rise}\,\bigl(1-(2x/L)^2\bigr)\,\bigl(1-0.5\,(2z/\mathrm{width})^2\bigr).
\]

\(n_z=\max(4,\lfloor n_x/2\rfloor)\). Roof grid \((n_x+1)\times(n_z+1)\).

Members:

- `GLULAM_DIAGRID`: both diagonals of every roof cell.
- `STEEL_EDGE`: the four perimeter lines.
- `TREE_TRUNK`: four columns at \((\pm 0.32L,\ 0,\ \pm 0.25\,\mathrm{width})\), from \(y=0\) to a bifurcation at \(y=0.55\times 4.5=2.475\) m.
- `TREE_BRANCH`: each bifurcation to the four nearest roof nodes. `branch_spread` is not used.
- `SEISMIC_FUSE`: two bars per column, from the bifurcation to ground anchors at \(x_{\mathrm{col}}\pm 1.5\) m, same \(z\).

A divisions-6 canopy has 51 nodes. That is the node count in every certified `quake_frames` record.

### Section and fuse idealization

`sections.py` and `solver_opensees.py`. OpenSeesPy 3.8.0.0 (`pareto_canopies.json` meta). Model: `basic`, `ndm=3`, `ndf=6`. Beams and columns are `elasticBeamColumn` with `geomTransf Linear`. Fuses are `corotTruss`. There are no fiber sections and no `PDelta` transformation.

| Group | Section in code | \(E\) | Density | Capacity used in utilization |
|---|---|---:|---:|---|
| Glulam diagrid | \(0.16\times h_g\), \(G=0.65\) GPa | 12.6 GPa | 420 kg/m³ | 18 MPa |
| Steel edge | area \(9.49\times 10^{-3}\) m², \(I=9.06\times 10^{-5}\) m⁴, \(J=1.41\times 10^{-4}\) m⁴ | 210 GPa | 7850 | \(f_y=355\) MPa |
| Trunk | area \(1.22\times 10^{-2}\) m², \(I=1.48\times 10^{-4}\), \(J=2.97\times 10^{-4}\) | 210 GPa | 7850 | 355 MPa |
| Branch | area \(4.67\times 10^{-3}\) m², \(I=2.02\times 10^{-5}\), \(J=4.03\times 10^{-5}\) | 210 GPa | 7850 | 355 MPa |

\(G_{\mathrm{steel}}=81\) GPa. Trunk and branch radii used for the elastic section modulus are 0.3239/2 m and 0.1937/2 m.

Supports: trunk bases `fix(1,1,1,1,1,0)` (released torsion about global \(Z\)). Fuse anchors `fix(1,1,1,1,1,1)`.

**NiTi fuse**, OpenSees `SelfCentering`, area \(A=A_{\mathrm{NiTi}}\times 10^{-6}\) m², density 6450 kg/m³:

\[
k_1=40\ \mathrm{GPa},\quad
k_2=\frac{40\times 10^9\cdot 70\times 10^6}{40\times 10^9\cdot 0.06+70\times 10^6}
=1\,133\,603\,238.8663967\ \mathrm{Pa},
\]

\[
\sigma_{\mathrm{act}}=380\ \mathrm{MPa},\quad
\beta=(380-110)/380=0.7105263157894737,
\]

`epsSlip=0`, `epsBear=0.071`, `rBear=1`. \(k_2\) equals the 1D forward tangent converted to Pa. The saturation strain 0.071 is not \(\varepsilon_L=0.06\).

**Steel fuse**, `Steel02`, equal activation force, not equal stiffness and not equal base shear:

\[
A_{\mathrm{steel}}=(380/355)\,A_{\mathrm{NiTi}}=1.0704225352112675\,A_{\mathrm{NiTi}}.
\]

\(f_y=355\) MPa, \(E_0=200\) GPa, \(b=0.01\), \(R_0=18\), \(cR_1=0.925\), \(cR_2=0.15\), density 7850 kg/m³.

### Loads, damping, integrator

\(g=9.80665\) m/s². Roof dead load \(G=1.50\) kN/m². Live \(Q=0.75\) kN/m². Wind uplift \(W=1.20\) kN/m².

Static combinations, on separate models: gravity \(1.35G+1.50Q\); uplift \(0.90G+1.50W\). Self-weight of every member is included. Equilibrium error is \(|\sum F_y+\sum R_y|/\max(|\sum F_y|,1)\). On all 80 dataset rows the stored maximum is \(9.37\times 10^{-11}\) (steel) and \(3.45\times 10^{-11}\) (NiTi).

Seismic preload is \(1.0G\) including self-weight, then `loadConst`. Seismic mass is roof dead load plus member mass. Live load is absent from the mass. Rotational masses are \(10^{-3}\). Rayleigh damping at 5% of modes 1 and 2: `rayleigh(α_M, 0, 0, β_K)` with \(\alpha_M\) and \(\beta_K\) from those two frequencies. Excitation is `UniformExcitation` in global \(X\) only. Integrator `Newmark(0.5, 0.25)`, \(dt=0.01\) s. Step logic: Newton, then NewtonLineSearch, then four substeps at \(dt/4\) with ModifiedNewton and a NewtonLineSearch retry. Test `NormDispIncr`, \(10^{-8}\), 30 iterations.

Roof drift ratio is \(\max|u_x|\) over roof nodes, divided by the fixed 4.5 m. It is not divided by the local roof height. Permanent residual drift is the absolute value of the mean signed roof drift over the last 1.0 s. The signed mean is stored separately.

### Ground motions

`ground_motions.py`. Three seeds exist in `tests/test_seismic_opensees.py`: 101, 202, 303. The 80-design dataset and the certified fronts use seed 101 only, at PGA \(0.40g\).

Each record: 48 sinusoids, frequencies `geomspace(0.35, 18.0, 48)` Hz, amplitudes \(1/\sqrt{f}\), phases uniform on \([0,2\pi)\) from the seed. Envelope \((1-e^{-t/0.65})\) times a decay \(e^{-0.55(t-9)}\) for \(t>9\) s. The 12 s strong motion is scaled so its peak is `pga_g * 9.80665`. Then 6 s of zeros, plus one extra zero sample: 1801 samples, \(t\in[0,18]\) s, 1800 analysis steps. The code does not match a code spectrum.

The three-seed test asserts convergence on every record and that the mean NiTi residual is below the mean steel residual. It does not write those six analyses to disk. Those means are not a verified number in this file.

### Carbon

`carbon_lca.py`. Default inventory is fossil, `include_biogenic=False`.

| Item | Factor |
|---|---|
| Glulam, fossil | 0.512 kgCO₂e/kg |
| Glulam, biogenic sensitivity | −0.98 kgCO₂e/kg |
| Steel, including the steel fuse | 1.55 kgCO₂e/kg |
| NiTi | 28.0 kgCO₂e/kg |
| Concrete footings | 1612.8 kgCO₂e, fixed |
| Glazing and connections | 45 kgCO₂e/m² of shaded area, repair only |

Mass is \(\sum \ell\,A\,\rho\) with the OpenSees section areas, not `CanopyGeometry.volumes_m3`.

\[
C_{A1\text{–}A3} = 1612.8 + C_{\mathrm{superstructure}}.
\]

One design earthquake:

- Steel fuse replacement if \(\max|\varepsilon_{\mathrm{fuse}}|>0.00355\): \(1.5\) times that fuse's A1–A3 carbon. At the threshold, replacement is zero.
- NiTi replacement if \(\max|\varepsilon_{\mathrm{fuse}}|>0.06\), same 1.5 factor.
- Glazing ratio \(r=0\) for peak drift \(\le 0.005\); \(r=0.35\,(d-0.005)/0.020\) up to 0.025; \(r=0.60\) above 0.025. Repair \(=45\,A_{\mathrm{shaded}}\,r\).
- Residual repair \(=\min(1,\ d_{\mathrm{res}}/0.005)\times 0.40\times C_{\mathrm{superstructure}}\). Footings are outside \(C_{\mathrm{superstructure}}\).

Lifecycle intensity is \((C_{A1\text{–}A3}+C_{\mathrm{repair}})/A_{\mathrm{shaded}}\).

### Paired 0.40 g comparison (80 designs, seed 101)

Source: `data/verified_lhs_dataset.csv`, 80 rows, 51,388 bytes. Latin hypercube, 5-D, seed 42, 20 designs in each of divisions {6, 8, 10, 12}. `branch_spread_ratio` is 0.17 on every row. Span runs from 14.0535 m to 27.8640 m. Both fuse cases passed 1800 steps and the static equilibrium gate before a row was kept. No baseline \(L=20\) m, rise 3 m, depth 0.35 m row is in the file.

| Quantity | Steel mean | Steel median | NiTi mean | NiTi median |
|---|---:|---:|---:|---:|
| Peak roof drift ratio | 0.017879931 | 0.015304033 | 0.033965356 | 0.026640000 |
| Absolute residual drift ratio | 0.00065878336 | 0.00052343371 | 0.00017462030 | 0.000053369496 |
| Peak base shear (N) | 285325.25 | 283786.20 | 415348.84 | 402566.42 |
| Max fuse strain | 0.0075820884 | 0.0064123085 | 0.015503281 | 0.011569512 |
| A1–A3 (kgCO₂e) | 17669.405 | 17590.935 | 18290.302 | 18234.054 |
| Repair (kgCO₂e) | 2435.2781 | 1694.7496 | 2875.7426 | 2758.3967 |
| Lifecycle intensity (kgCO₂e/m²) | 174.76201 | 165.20734 | 184.23659 | 179.11098 |

Counts on the same 80 geometries:

- NiTi peak drift is higher on 80/80. Ratio of means: 1.8996. Designs above 2% peak drift: steel 17, NiTi 55.
- NiTi absolute residual is lower on 66/80. Ratio of means: 0.2651. Ratio of medians: 0.1020.
- NiTi peak base shear is higher on the mean (ratio of means 1.4557). The equal-strength condition is the fuse yield force, \(355\,A_{\mathrm{steel}}=380\,A_{\mathrm{NiTi}}\).
- NiTi A1–A3 is higher on 80/80. Mean difference: +620.90 kgCO₂e.
- NiTi lifecycle intensity is lower on 14/80 and higher on 66/80. Mean difference: +9.475 kgCO₂e/m².
- Fuse-replacement trigger: steel 53/80, NiTi 1/80.
- Fundamental period above the 1.50 s static-test band: steel 3/80 (maximum 1.7219909 s), NiTi 0/80 (maximum 1.4465167 s). The dataset writer does not reject those periods.

---

## 4. Surrogate screening and multi-objective optimization

### Gaussian process

`surrogate_gp.py`. Training table: the 80-row CSV. Features, in order:

`span_m`, `arch_rise_ratio`, `diagrid_divisions`, `glulam_depth_m`, `niti_fuse_area_mm2`.

Targets, separate models for each fuse, after the stated scaling of the CSV column:

| Target | CSV column suffix | Scale |
|---|---|---|
| `peak_drift_pct` | `peak_roof_drift_ratio` | ×100 |
| `abs_residual_drift_pct` | `permanent_residual_roof_drift_ratio` | ×100 |
| `grav_defl_mm` | `static_gravity_deflection_m` | ×1000 |
| `c_total_per_m2` | `total_intensity_kgco2e_m2` | ×1 |

Kernel: `ConstantKernel * Matern(nu=2.5) + WhiteKernel`. Constant bounds \((10^{-8},10^{8})\), length-scale bounds \((10^{-2},10^{3})\) with five initial scales of 1, white-noise bounds \((10^{-10},10^{4})\). `alpha=1e-10`, `normalize_y=False`, `n_restarts_optimizer=0`. Features are `StandardScaler`'d inside each fold and again on the full fit.

Cross-validation: `GroupKFold(n_splits=4)` on `pandas.qcut` span quartiles. Fold sizes in the artifact: 20, 20, 20, 20. Seed 42, with an offset per fuse, target, and fold. The exported regressors are refit on all 80 rows. The out-of-fold vectors are the held-out predictions.

A held-out row is unsafe when the true peak drift exceeds 2%. The false-safe rate is the fraction of held-out rows that are unsafe and are still called safe. Safe means `mean ≤ 2`, or `mean + 1.96 σ ≤ 2` for the UCB rule. The rate is over all held-out rows, safe and unsafe together.

Pooled held-out count: 160 (80 designs × 2 fuses). Actual unsafe: 72.

| Predictor | False-safe count / 160 | Rate |
|---|---:|---:|
| Mean | 8 | 0.050 |
| 95% UCB | 0 | 0.0 |

Per fuse, the mean rate is 0.050 (4/80) and the UCB rate is 0. Unsafe counts: steel 17, NiTi 55.

Grouped-CV scores from `gp_surrogate.json`:

| Fuse | Target | RMSE | MAE | Mean σ | \(R^2\) |
|---|---|---:|---:|---:|---:|
| steel | peak drift (%) | 1.407151 | 0.622602 | 0.538626 | 0.379154 |
| steel | abs. residual (%) | 0.057670 | 0.044477 | 0.055821 | 0.226659 |
| steel | gravity defl. (mm) | 30.169536 | 13.687478 | 10.464621 | 0.771964 |
| steel | carbon (kgCO₂e/m²) | 13.721841 | 10.511134 | 9.496066 | 0.822833 |
| niti | peak drift (%) | 1.049117 | 0.570317 | 0.691730 | 0.807748 |
| niti | abs. residual (%) | 0.067036 | 0.024045 | 0.028178 | −0.154299 |
| niti | gravity defl. (mm) | 29.115062 | 13.502213 | 8.787581 | 0.807724 |
| niti | carbon (kgCO₂e/m²) | 12.351381 | 8.641187 | 6.936672 | 0.867919 |

The NiTi residual model has a negative \(R^2\). The screening constraint uses peak drift and gravity deflection, not residual drift.

`tests/test_surrogate_gp.py` checks that a JavaScript-style Matérn evaluation of the exported `alpha`, Cholesky factor, length scales, noise, and scaler matches `GaussianProcessRegressor.predict` within `rtol=atol=1e-8` on the first dataset row. That test refits the GP. This audit did not run it. SPEC §7 writes the tolerance as \(<10^{-6}\).

### NSGA-II

`optimize_nsga2.py`. pymoo `NSGA2`, population 60, 60 generations, seed 42, `eliminate_duplicates=True`. One run per fuse. Decision vector \((L,\ \mathrm{rise}/L,\ n_x,\ h_g,\ A_{\mathrm{NiTi}})\), bounds

\[
[14,\ 0.08,\ 6,\ 0.22,\ 60]
\quad\text{to}\quad
[28,\ 0.22,\ 12,\ 0.48,\ 260].
\]

\(n_x\) is snapped to {6, 8, 10, 12}. `branch_spread` is set to \(0.17L\) and is not a free variable.

Objectives: minimize predicted `c_total_per_m2`, and minimize \(-L\). Inequality constraints, both required ≤ 0:

\[
(\mu_{\mathrm{peak}}+1.96\sigma_{\mathrm{peak}})-2.0 \le 0,
\]

\[
(\mu_{\mathrm{grav}}+1.96\sigma_{\mathrm{grav}})-L\cdot 1000/250 \le 0.
\]

Certification re-runs exact OpenSees static gravity, wind uplift, and a 0.40 g seed-101 transient. A point is rejected if exact peak drift exceeds 2% or exact gravity deflection exceeds \(L/250\). The artifact records 8 attempts, 8 accepted, 0 rejected, 0 failed analyses: 4 designs labeled `steel-*` and 4 labeled `niti-*`. Exported floats are rounded to 5 decimal places.

The eight accepted points are two tight clusters, all with `diagrid_divisions=6`. They do not trace a long carbon–span curve.

| Design | Span (m) | NiTi area (mm²) | Depth (m) | Steel peak / residual (%) | NiTi peak / residual (%) | Steel / NiTi base shear (kN) | Steel / NiTi lifecycle (kgCO₂e/m²) |
|---|---:|---:|---:|---|---|---|---|
| steel-1 | 27.20284 | 227.42428 | 0.32537 | 1.45778 / 0.13975 | 4.10272 / 0.00708 | 357.444 / 560.368 | 123.23842 / 136.67642 |
| steel-2 | 27.20284 | 227.43393 | 0.32534 | 1.45984 / 0.14023 | 4.10608 / 0.00707 | 357.384 / 560.650 | 123.29321 / 136.67631 |
| steel-3 | 27.20321 | 227.32948 | 0.32536 | 1.45802 / 0.14002 | 4.10347 / 0.00703 | 357.351 / 560.325 | 123.25845 / 136.66818 |
| steel-4 | 27.28762 | 213.05722 | 0.33466 | 1.46764 / 0.18028 | 4.20198 / 0.00452 | 351.599 / 571.146 | 126.44086 / 136.18770 |
| niti-1 | 21.20139 | 246.09827 | 0.40375 | 0.55142 / 0.00345 | 1.81793 / 0.00079 | 290.376 / 482.563 | 148.62105 / 166.64543 |
| niti-2 | 21.26665 | 251.08835 | 0.42398 | 0.49083 / 0.00407 | 1.73964 / 0.00024 | 265.425 / 491.072 | 149.01803 / 166.81667 |
| niti-3 | 21.31710 | 226.30405 | 0.46380 | 0.49245 / 0.01162 | 1.68723 / 0.00025 | 280.934 / 497.051 | 151.90063 / 167.57828 |
| niti-4 | 21.31671 | 248.72535 | 0.46305 | 0.45202 / 0.00204 | 1.54460 / 0.00019 | 277.905 / 474.641 | 150.85046 / 167.20458 |

On all eight geometries the NiTi peak drift, peak base shear, A1–A3, and lifecycle intensity are higher, and the NiTi absolute residual is lower. The steel-front geometries are exact-unsafe if the fuses are switched to NiTi (peak drift about 4.1% to 4.2%). Static equilibrium error is stored as 0.0 at the 5-decimal export. Each design has 60 quake frames.

---

## 5. Figure data mapping

Four figures can be drawn from committed tables. `scripts/plot_day2.py` re-runs OpenSees and is not a data source. `scripts/plot_day1.py` rebuilds the hysteresis from the constitutive function. That rebuild is a few thousand scalar updates. It is not a stored series.

### Figure A — constitutive loop and identification

File: `data/diff_id_uncertainty.json`, object `loops` (601 samples each).

| Series | Path |
|---|---|
| Strain | `loops.strain` |
| Exact return-map stress (MPa) | `loops.true_stress_MPa` |
| 3% noise observations (MPa) | `loops.observed_stress_MPa` |
| Exact return map at recovered parameters (MPa) | `loops.recovered_stress_MPa` |

Error bars: `true_parameters`, `recovered_parameters`, `relative_recovery_error_pct`, `confidence_intervals_95`. Order used by `scripts/plot_day4.py`: `E`, `sigma_Ms`, `sigma_Mf`, `sigma_As`, `sigma_Af`, `eps_L`.

The Day-1 hysteresis in `figures/day1_niti_constitutive.png` is a different history: five cycles of \(0.05\sin\phi\), 1000 samples per cycle, from `constitutive_sma.niti_update`. It is not in a CSV.

### Figure B — NiTi versus steel seismic comparison

File: `data/verified_lhs_dataset.csv`.

Shared columns: `design_id`, `span_m`, `arch_rise_ratio`, `diagrid_divisions`, `glulam_depth_m`, `niti_fuse_area_mm2`, `branch_spread_ratio`, `shaded_area_m2`.

Per fuse, prefix `steel_` or `niti_`:

`peak_roof_drift_ratio`, `permanent_residual_roof_drift_ratio`, `signed_permanent_residual_roof_drift_ratio`, `peak_base_shear_N`, `max_fuse_strain`, `static_gravity_deflection_m`, `static_wind_uplift_deflection_m`, `T1_s`, `static_equilibrium_error`.

`scripts/plot_day3.py` scatters peak drift (×100) against the signed residual (×100) and uses `span_m` for color.

Frame histories, certified designs only: `data/pareto_canopies.json`, `designs[].quake_frames.steel` and `.niti`.

| Series | Key |
|---|---|
| Time (s) | `time_s` (60) |
| Node tags | `node_ids` |
| Translations (m), shape (60, n_nodes, 3) | `nodal_displacements_m_rad` |
| Lead-fuse strain | `lead_fuse_strain` |
| Lead-fuse stress (MPa) | `lead_fuse_stress_MPa` |

The displacement array is three translations. The key name still says `m_rad`. Roof drift versus time is not stored. It can be reduced from `designs[].nodes[].xyz_m` and the \(x\) translations, divided by 4.5, using the same roof-node set the solver used. Scalar KPIs for the same runs are `designs[].kpis.steel` and `.niti` (`peak_drift_pct`, `abs_residual_drift_pct`, `signed_residual_drift_pct`, `peak_base_shear_kN`, `max_fuse_strain_pct`).

### Figure C — upfront and lifecycle carbon

Same CSV. Plot columns, as in `scripts/plot_day3.py`:

| Axis | Steel | NiTi |
|---|---|---|
| A1–A3 total (kgCO₂e) | `steel_c_a1_a3_kgco2e` | `niti_c_a1_a3_kgco2e` |
| Lifecycle intensity (kgCO₂e/m²) | `steel_total_intensity_kgco2e_m2` | `niti_total_intensity_kgco2e_m2` |
| Shaded area, if a total is rebuilt | `shaded_area_m2` | same |

Repair totals, if a stacked bar is needed: `steel_c_repair_kgco2e`, `niti_c_repair_kgco2e`, `steel_total_kgco2e`, `niti_total_kgco2e`. The CSV does not split repair into fuse, glazing, and residual. Those components exist only inside `assess_lifecycle_carbon`.

### Figure D — grouped-CV parity, false-safe rate, certified front

File: `data/gp_surrogate.json` (1.54 MB). Plot only the `validation` object. Leave `cases.*.models.*.alpha` and `cholesky` unread. They are 80×80 factors for the browser, not figure data.

| Series | Path |
|---|---|
| Out-of-fold peak drift, truth / mean / std (%) | `validation.oof_peak_drift_pct.{steel,niti}.{truth,mean,std}` (80 each) |
| \(R^2\) annotation | `validation.per_case.{steel,niti}.peak_drift_pct.r2` |
| False-safe rates | `validation.false_safe_rate.{mean,ucb_95}` and `validation.per_case.{steel,niti}.false_safe_rate` |
| Fold sizes | `validation.fold_sizes` = [20, 20, 20, 20] |

Certified front, `data/pareto_canopies.json`:

| Series | Path |
|---|---|
| Which front a point belongs to | `designs[].front_case` (`steel` or `niti`) |
| Clear span (m) | `designs[].params.span` |
| Lifecycle carbon of that front's fuse (kgCO₂e/m²) | `designs[].kpis[front_case].c_total_per_m2` |
| Predicted UCB drift and carbon | `designs[].predicted.peak_drift_ucb_pct`, `predicted_total_carbon_per_m2` |

`scripts/plot_day4.py` draws one Pareto point per design, using the fuse named by `front_case`. The paired other fuse is in `kpis` and is the honest same-geometry comparison.

---

## 6. Gaps and limitations in the code

1. The 1D law is isothermal and rate-independent. There is no temperature, no latent heat, and no cyclic degradation or fatigue. Recentering cannot wear out inside this model. The free-energy check only shows that the discrete trapezoidal work stays above \(-10^{-9}\) MPa on the tested loops.

2. The frame does not integrate that return map. NiTi fuses are an OpenSees flag-shaped `SelfCentering` material with a bearing strain of 0.071. Every other member is linear elastic. The geometric transformation on those members is `Linear`. Excitation is one horizontal component. Drift divides roof \(u_x\) by a fixed 4.5 m.

3. The ground-motion set is three seeded synthetic waveforms. The published design table uses one of them (seed 101) scaled to 0.40 g. There is no recorded suite and no target spectrum in the code. A period gate of 0.15–1.50 s is enforced on one baseline static test. Three of the 80 steel analyses have \(T_1>1.50\) s and remain in the dataset.

4. Carbon is a cradle-to-gate factor table plus one design-earthquake repair rule. Transport, construction, and end of life are absent. Glazing carbon appears only as repair. On this factor table NiTi A1–A3 is higher for every paired design, and lifecycle intensity is higher for 66 of 80. The NiTi residual Gaussian process has \(R^2=-0.154\), so a residual objective would not be supported by the surrogate that was trained. The certified fronts collapse to two spans, about 27.2 m for steel and about 21.3 m for NiTi, all with six diagrid divisions.

`branch_spread` does not move a node. The 20-member identification intervals miss the true \(\varepsilon_L\). Those two facts belong in the paper next to the limitations above.

---

## DISCREPANCIES TO RESOLVE

1. `paper/REPO_MAP.txt` was required as the first read. It is not in the repository.

2. `README.md` and `index.html` call the solver a geometric fiber-section analysis. The builder uses `elasticBeamColumn` and `corotTruss`. SPEC §5 says a `Linear` or `PDelta` transformation and pinned fuse anchors `(1,1,1,0,0,0)`. The code uses `Linear` only, and it fixes fuse anchors as `(1,1,1,1,1,1)`.

3. `README.md` calls the 1D model thermo-mechanical Auricchio–Sacco. The code has no temperature. SPEC §2 also names Souza–Auricchio. The implemented object is the 1D threshold map written in that section. OpenSees never calls it.

4. `README.md` and SPEC §5 call the records spectrum-compatible. `generate_ground_motion` scales a seeded sum of 48 sinusoids to the requested PGA. No target spectrum is stored.

5. The README abstract says NiTi cuts residual drift at a comparable or lower whole-life carbon cost than equal-strength steel. On the 80 paired 0.40 g analyses, residual drift is lower for NiTi on 66 geometries, lifecycle intensity is lower on 14, A1–A3 is higher on 80, and peak drift is higher on 80. The same carbon and peak-drift inequalities hold on all 8 certified geometries.

6. `docs/VALIDATION.md` prints the identification table to a rounded 7.28% maximum error. That rounding matches \(\varepsilon_L\). It does not say that the saved 95% intervals miss the true \(\sigma_{Mf}\), \(\sigma_{Af}\), and \(\varepsilon_L\). The 8% point-error gate still passes.

7. `docs/VALIDATION.md` quotes grouped-CV \(R^2\) only for peak drift (steel 0.38, NiTi 0.81). The artifact also has NiTi residual \(R^2=-0.154299\).

8. `index.html` says "Tangent residual < 1e-7". The tangent gate is a relative error below \(10^{-7}\). The Newton gate is \(|r|<10^{-12}\). The recomputed tangent error is \(1.850\times 10^{-10}\). The recomputed Newton residual at 500 MPa is \(4.55\times 10^{-13}\).

9. `scripts/plot_day2.py` multiplies `permanent_residual_roof_drift_ratio` by 100 and labels the result "Signed residual". That field is already an absolute value. The signed series is `signed_permanent_residual_roof_drift_ratio`.

10. `CanopyGeometry.volumes_m3` assigns \(0.01\times 0.01\) m² to every steel member and the NiTi area to the fuses, including when the fuse is the enlarged steel area. SPEC §6 forbids using that field for carbon. `carbon_lca.py` uses `section_properties`. A figure that reads `volumes_m3` will mis-state the steel mass.

11. Two constitutive modules remain: `niti.py` (used by `solver.py` and `test_niti.py`) and `constitutive_sma.py` (used by identification, thermodynamics, and the Pareto metadata). They implement the same update today.

12. SPEC §7 asks the browser Matérn check to agree within \(10^{-6}\). The test asserts \(10^{-8}\). The saved GP was not refit for this audit, so this line is a tolerance mismatch, not a new measurement.
