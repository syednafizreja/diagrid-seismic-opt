# Validation dossier — diagrid-seismic-opt

Every figure below is produced by the project `.venv` from the committed dataset
and Day 4 JSON artifacts. Reproduce with `python -m pytest -q`, the module
`main()`s, and `scripts/plot_day*.py`.

## 1. Constitutive model (Day 1)

| Check | Gate | Result |
|---|---|---|
| Algorithmic tangent vs central finite differences | rel. error `< 1e-7` | `1.85e-10` |
| Newton–Raphson equilibrium | `\|r\| < 1e-12` in `<= 5` iterations | passes |
| Clausius–Duhem discrete dissipation (5001-step cyclic) | `D_step >= -1e-9` | min `-1.7e-15` |

## 2. OpenSeesPy multi-hazard solver (Day 2)

| Check | Gate | Result |
|---|---|---|
| Global static equilibrium (gravity + wind uplift) | `< 1e-4` | passes, both fuse cases |
| Fundamental period `T1` | `0.15–1.50 s` | passes |
| Nonlinear 0.40 g time-history convergence | 1800/1800 steps | passes, all records, steel + NiTi |
| Mean residual drift: NiTi `<` steel | strict | passes |

## 3. Whole-life carbon + verified dataset (Day 3)

- `data/verified_lhs_dataset.csv` — 80 paired designs verified for gravity,
  wind uplift and a 0.40 g seed-101 seismic analysis in isolated worker processes.

## 4. GP surrogate (Day 4)

| Check | Result |
|---|---|
| Grouped-CV folds (`GroupKFold` by span quartile) | `20 / 20 / 20 / 20` |
| Grouped-CV `R²` (peak roof drift) | steel `0.38`, NiTi `0.81` |
| False-safe peak-drift rate — mean → 95% UCB | `5.0%` → `0.0%` |
| Browser Matérn-5/2 replay vs scikit-learn | `< 1e-8` (`tests/test_surrogate_gp.py`) |

## 5. Certified Pareto front (Day 4)

| Check | Result |
|---|---|
| Certified designs (exact OpenSeesPy re-run) | `8` — 4 steel, 4 NiTi |
| False-safe designs on certified front | `0` |
| Exact static equilibrium error | `< 1e-4`, all designs |
| Seismic convergence | 1800/1800 steps, all designs |
| Tangent / dissipation meta | `1.85e-10` / `-1.7e-15` |

## 6. Differentiable NiTi inverse identification (Day 4)

3%-noisy cyclic data, recovered by PyTorch `Adam → L-BFGS` with a 20-member
residual bootstrap. Gate: **every parameter within 8%**.

| Parameter | True | Recovered | Error | 95% CI |
|---|---|---|---|---|
| `E` (MPa) | 40000 | 39834.2 | 0.41% | [39384.5, 40482.5] |
| `sigma_Ms` (MPa) | 380 | 381.7 | 0.46% | [378.2, 380.8] |
| `sigma_Mf` (MPa) | 450 | 452.2 | 0.48% | [450.8, 457.8] |
| `sigma_As` (MPa) | 180 | 186.4 | 3.58% | [171.1, 183.8] |
| `sigma_Af` (MPa) | 110 | 111.7 | 1.56% | [110.6, 117.1] |
| `eps_L` | 0.060 | 0.0644 | 7.28% | [0.0627, 0.0658] |

Max error **7.28%** (eps_L) `< 8%` gate; every parameter carries a 20-member
bootstrap 95% CI.

## Reproduce

```powershell
python -m diagrid_opt.surrogate_gp        # data/gp_surrogate.json
python -m diagrid_opt.optimize_nsga2      # data/pareto_canopies.json (exact OpenSeesPy)
python -m diagrid_opt.diff_ident_sciml    # data/diff_id_uncertainty.json
python scripts/plot_day4.py               # figures/day4_*.png
python -m pytest -q
python -m http.server 8000                # open http://localhost:8000
```
