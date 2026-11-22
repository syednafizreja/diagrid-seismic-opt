# diagrid-seismic-opt

Multi-hazard seismic design and whole-life carbon optimisation of a 3D curved
glulam-timber diagrid canopy, comparing conventional yielding steel fuses with
superelastic **NiTi shape-memory-alloy** self-centering fuses.

**Live demo:** https://syednafizreja.github.io/diagrid-seismic-opt/

## Research question

Can superelastic NiTi fuses reduce post-earthquake residual drift and whole-life
carbon relative to equal-strength steel fuses — and can a Gaussian-process
surrogate + NSGA-II recover a *certified* carbon/span Pareto front, while a
differentiable PyTorch model recovers the NiTi constitutive parameters from
noisy cyclic data?

## Verified results

| Layer | Quantity | Verified value |
|---|---|---|
| Constitutive | Algorithmic tangent vs central finite differences | `< 1.9e-10` rel. (gate `< 1e-7`) |
| Constitutive | Clausius–Duhem step dissipation | `>= -1.7e-15` (gate `>= -1e-9`) |
| Constitutive | Newton equilibrium | `\|r\| < 1e-12` in `<= 5` iterations |
| OpenSeesPy | Static equilibrium (gravity + wind uplift) | `< 1e-4` relative |
| OpenSeesPy | Nonlinear 0.40 g seismic convergence | 1800/1800 steps, all records |
| GP surrogate | Grouped-CV `R²` (peak drift) | steel `0.38` / NiTi `0.81` |
| GP surrogate | False-safe peak-drift rate | `5.0%` → `0.0%` via 95% UCB |
| Pareto | Certified designs (exact OpenSeesPy) | `8` (4 steel, 4 NiTi), `0` false-safe |
| Inverse ID | PyTorch parameter recovery (max error) | `7.28%` (eps_L; gate `< 8%`) |

See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the full verified-numbers dossier.

## What is built

- **Days 1–3** — exact 1D Auricchio–Sacco NiTi constitutive model, 3D parametric
  diagrid geometry, OpenSeesPy static + nonlinear time-history solver, whole-life
  carbon LCA, and an 80-design verified LHS dataset (`data/verified_lhs_dataset.csv`).
- **Day 4** — Matérn-5/2 GP surrogate (`data/gp_surrogate.json`), NSGA-II
  certified Pareto canopies (`data/pareto_canopies.json`), and differentiable
  PyTorch NiTi parameter identification with bootstrap CIs
  (`data/diff_id_uncertainty.json`). Reproduce with the module `main()`s and
  `scripts/plot_day4.py`.
- **Day 5** — this zero-build Three.js viewer, GitHub Actions CI, and the
  validation dossier.

## Repository layout

```
src/diagrid_opt/      constitutive_sma, thermodynamics, geometry, ground_motions,
                      solver_opensees, carbon_lca, surrogate_gp, optimize_nsga2,
                      diff_ident_sciml, dataset
tests/                pytest suites incl. Day 4 JSON-schema/gate checks
scripts/              plot_day1..4.py, generate_dataset.py
data/                 verified_lhs_dataset.csv + Day 4 JSON artifacts (served by the web app)
figures/              day1..day4 figures
index.html, style.css, script.js   the interactive viewer
.github/workflows/ci.yml           Python 3.11 CI (pytest)
docs/VALIDATION.md                 verified-numbers dossier
```

## Local run

```powershell
python -m pip install -e .
python -m pytest -q
python -m http.server 8000      # then open http://localhost:8000
```

> The page must be served over `http://` (browsers block `fetch()` of the JSON
> artifacts from `file://`).

## Interactive viewer

Zero-build static ES modules (Three.js r170 via a jsDelivr `importmap`). It loads
the three Day 4 JSON artifacts and offers:

- **Certified Pareto mode** — clickable Pareto scatter (lifecycle carbon vs clear
  span) and a *Play 0.40 g Earthquake (Steel vs NiTi)* animation driven by the real
  recorded `quake_frames` nodal displacements, synchronised with a live 2D fuse
  hysteresis canvas (displacements visually amplified ×40).
- **Live slider mode** — five structural sliders morph the diagrid at interactive
  frame rates and evaluate the pure-JS Matérn-5/2 GP to show `mean ± 1.96σ` for the
  four KPIs, flagging unsafe (`> 2.0%`) 95%-UCB drift.
- **Display modes** — Architectural, Utilization, Technical, Wireframe.
- **Chart tabs** — Hysteresis, Pareto Front, PyTorch Inverse ID.

## Data & deployment

- Day 4 JSON payloads are compacted for fast GitHub Pages loading:
  `pareto_canopies.json` keeps a 60-frame `quake_frames` but stores only the
  translational DOF and rounds floats to 5 decimal places (~1.3 MB).
- GitHub Pages is served from `main` / root via standard branch deploy; the root
  `.nojekyll` file disables Jekyll processing so static assets (JSON, PNG, ES
  modules) are served as-is.

## Limitations

- The GP surrogate is only trustworthy inside the sampled design envelope
  (`training_domain`); the UCB is a safety margin, not a guarantee.
- Ground motions are deterministic synthetic spectrum-compatible records
  (seeds 101/202/303), not site-specific hazard.
- Carbon factors are A1–A3 cradle-to-gate plus a single design-earthquake repair
  model; they exclude construction, transport and end-of-life.
- The earthquake animation amplifies nodal displacements ×40 for legibility.

## License

MIT.
