from __future__ import annotations

from dataclasses import asdict
from importlib.metadata import version
import json
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize

from .carbon_lca import assess_lifecycle_carbon
from .constitutive_sma import NiTiParams, niti_update
from .geometry import GeometryParams, canopy_geometry
from .solver_opensees import run_seismic_analysis, run_static_checks
from .surrogate_gp import (
    DATASET_PATH,
    EXPORT_PATH,
    FUSE_TYPES,
    SurrogateBundle,
    fit_grouped_cv_surrogates,
)
from .thermodynamics import discrete_dissipation


POPULATION_SIZE = 60
GENERATION_COUNT = 60
SEED = 42
CERTIFIED_PER_CASE = 4
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "pareto_canopies.json"
)
DIVISION_VALUES = (6, 8, 10, 12)


class CanopyOptimizationProblem(ElementwiseProblem):
    def __init__(self, fuse_type: str, surrogate: SurrogateBundle):
        if fuse_type not in FUSE_TYPES:
            raise ValueError(f"unknown fuse type {fuse_type!r}")
        self.fuse_type = fuse_type
        self.surrogate = surrogate
        super().__init__(
            n_var=5,
            n_obj=2,
            n_ieq_constr=2,
            xl=np.array([14.0, 0.08, 6.0, 0.22, 60.0]),
            xu=np.array([28.0, 0.22, 12.0, 0.48, 260.0]),
        )

    @staticmethod
    def decode(values: np.ndarray) -> GeometryParams:
        span, rise_ratio, division_value, depth, fuse_area = map(float, values)
        division_index = int(np.rint((division_value - 6.0) / 2.0))
        divisions = DIVISION_VALUES[
            min(max(division_index, 0), len(DIVISION_VALUES) - 1)
        ]
        return GeometryParams(
            span=span,
            arch_rise=span * rise_ratio,
            diagrid_divisions=divisions,
            branch_spread=0.17 * span,
            glulam_depth=depth,
            niti_fuse_area=fuse_area,
        )

    @staticmethod
    def features(params: GeometryParams) -> np.ndarray:
        return np.asarray(
            [
                params.span,
                params.arch_rise / params.span,
                params.diagrid_divisions,
                params.glulam_depth,
                params.niti_fuse_area,
            ],
            dtype=float,
        )

    def prediction(self, params: GeometryParams) -> dict[str, tuple[float, float]]:
        estimates = self.surrogate.predict(
            self.fuse_type, self.features(params)
        )
        return {
            name: (float(mean[0]), float(std[0]))
            for name, (mean, std) in estimates.items()
        }

    def _evaluate(self, values: np.ndarray, out: dict[str, Any], *args, **kwargs):
        params = self.decode(values)
        prediction = self.prediction(params)
        peak_mean, peak_std = prediction["peak_drift_pct"]
        gravity_mean, gravity_std = prediction["grav_defl_mm"]
        peak_ucb = peak_mean + 1.96 * peak_std
        gravity_ucb = gravity_mean + 1.96 * gravity_std
        out["F"] = [
            prediction["c_total_per_m2"][0],
            -params.span,
        ]
        out["G"] = [
            peak_ucb - 2.0,
            gravity_ucb - params.span * 1000.0 / 250.0,
        ]


def _front_candidates(
    result: Any, problem: CanopyOptimizationProblem
) -> list[dict[str, Any]]:
    if result.X is None:
        raise RuntimeError(
            f"{problem.fuse_type} NSGA-II found no feasible Pareto designs"
        )
    candidates: dict[tuple[float, ...], dict[str, Any]] = {}
    for values in np.atleast_2d(result.X):
        params = problem.decode(values)
        key = (
            round(params.span, 7),
            round(params.arch_rise / params.span, 7),
            float(params.diagrid_divisions),
            round(params.glulam_depth, 7),
            round(params.niti_fuse_area, 7),
        )
        prediction = problem.prediction(params)
        peak_mean, peak_std = prediction["peak_drift_pct"]
        gravity_mean, gravity_std = prediction["grav_defl_mm"]
        peak_ucb = peak_mean + 1.96 * peak_std
        gravity_ucb = gravity_mean + 1.96 * gravity_std
        if (
            peak_ucb <= 2.0
            and gravity_ucb <= params.span * 1000.0 / 250.0
        ):
            candidates[key] = {
                "params": params,
                "prediction": prediction,
                "peak_drift_ucb_pct": peak_ucb,
                "gravity_deflection_ucb_mm": gravity_ucb,
            }
    if len(candidates) < CERTIFIED_PER_CASE:
        raise RuntimeError(
            f"{problem.fuse_type} has only {len(candidates)} distinct "
            "GP-feasible Pareto designs; four are required"
        )
    return sorted(
        candidates.values(),
        key=lambda item: (
            item["params"].span,
            item["prediction"]["c_total_per_m2"][0],
        ),
    )


def _design_key(params: GeometryParams) -> tuple[float, ...]:
    return (
        round(params.span, 7),
        round(params.arch_rise / params.span, 7),
        float(params.diagrid_divisions),
        round(params.glulam_depth, 7),
        round(params.niti_fuse_area, 7),
    )


def _frame_payload(history: dict[str, Any]) -> dict[str, Any]:
    times = np.asarray(history["time_s"], dtype=float)
    if times.size < 2:
        raise RuntimeError("seismic analysis returned too few frames")
    selected = np.linspace(0, times.size - 1, 60, dtype=int)
    return {
        "time_s": times[selected].tolist(),
        "node_ids": np.asarray(history["node_ids"], dtype=int).tolist(),
        "nodal_displacements_m_rad": np.asarray(
            history["nodal_displacements_m_rad"], dtype=float
        )[selected].tolist(),
        "lead_fuse_strain": np.asarray(
            history["lead_fuse_strain"], dtype=float
        )[selected].tolist(),
        "lead_fuse_stress_MPa": (
            np.asarray(history["lead_fuse_stress_Pa"], dtype=float)[selected]
            / 1e6
        ).tolist(),
    }


def _exact_case_results(
    params: GeometryParams,
) -> dict[str, Any]:
    geometry = canopy_geometry(params)
    cases: dict[str, Any] = {}
    frames: dict[str, Any] = {}
    for fuse_type in FUSE_TYPES:
        static = run_static_checks(geometry, fuse_type=fuse_type)
        for load_case, values in static["cases"].items():
            equilibrium = float(values["equilibrium_error"])
            if not isfinite(equilibrium) or equilibrium >= 1e-4:
                raise RuntimeError(
                    f"{fuse_type} static {load_case} equilibrium error "
                    f"{equilibrium:.6g} is not below 1e-4"
                )
        seismic = run_seismic_analysis(
            geometry,
            fuse_type=fuse_type,
            pga_g=0.40,
            seed=101,
            record_frames=True,
        )
        if seismic["converged_steps"] != 1800:
            raise RuntimeError(
                f"{fuse_type} seismic analysis converged "
                f"{seismic['converged_steps']} of 1800 steps"
            )
        seismic_metrics = {
            "peak_roof_drift_ratio": float(
                seismic["peak_roof_drift_ratio"]
            ),
            "permanent_residual_roof_drift_ratio": float(
                seismic["permanent_residual_roof_drift_ratio"]
            ),
            "max_fuse_strain": float(seismic["max_fuse_strain"]),
        }
        carbon = assess_lifecycle_carbon(
            geometry, fuse_type, seismic_metrics
        )
        gravity_deflection_mm = (
            1000.0
            * static["cases"]["gravity"]["max_roof_deflection_m"]
        )
        peak_drift_pct = 100.0 * seismic_metrics["peak_roof_drift_ratio"]
        cases[fuse_type] = {
            "kpis": {
                "peak_drift_pct": peak_drift_pct,
                "abs_residual_drift_pct": (
                    100.0
                    * seismic_metrics["permanent_residual_roof_drift_ratio"]
                ),
                "signed_residual_drift_pct": (
                    100.0
                    * float(
                        seismic["signed_permanent_residual_roof_drift_ratio"]
                    )
                ),
                "grav_defl_mm": gravity_deflection_mm,
                "wind_uplift_defl_mm": (
                    1000.0
                    * static["cases"]["wind_uplift"]["max_roof_deflection_m"]
                ),
                "T1_s": float(static["T1_s"]),
                "peak_base_shear_kN": (
                    float(seismic["peak_base_shear_N"]) / 1000.0
                ),
                "max_fuse_strain_pct": (
                    100.0 * float(seismic["max_fuse_strain"])
                ),
                "static_equilibrium_error": float(
                    static["equilibrium_error"]
                ),
                "peak_member_stress_utilization": static[
                    "peak_member_stress_utilization"
                ],
                "c_a1_a3_kgco2e": carbon.c_a1_a3_kgco2e,
                "c_repair_kgco2e": carbon.c_repair_kgco2e,
                "c_total_per_m2": carbon.total_intensity_kgco2e_m2,
                "peak_drift_safe": peak_drift_pct <= 2.0,
                "gravity_deflection_safe": (
                    gravity_deflection_mm
                    <= params.span * 1000.0 / 250.0
                ),
            },
            "history": _frame_payload(seismic["history"]),
        }
        frames[fuse_type] = cases[fuse_type].pop("history")

    return {
        "geometry": geometry,
        "cases": cases,
        "quake_frames": frames,
    }


def _select_certified(
    fuse_type: str,
    candidates: list[dict[str, Any]],
    exact_cache: dict[tuple[float, ...], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    targets = np.linspace(14.0, 28.0, CERTIFIED_PER_CASE)
    used: set[tuple[float, ...]] = set()
    selected: list[dict[str, Any]] = []
    attempted = 0
    false_safe = 0
    failed = 0
    for target_span in targets:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                abs(candidate["params"].span - target_span),
                candidate["params"].span,
            ),
        )
        for candidate in ordered:
            params = candidate["params"]
            key = _design_key(params)
            if key in used:
                continue
            used.add(key)
            attempted += 1
            try:
                if key not in exact_cache:
                    exact_cache[key] = _exact_case_results(params)
                exact = exact_cache[key]
            except RuntimeError as error:
                failed += 1
                print(
                    f"Skipping exact certification of {fuse_type} span "
                    f"{params.span:.3f} m: {error}"
                )
                continue

            kpis = exact["cases"][fuse_type]["kpis"]
            exact_safe = (
                kpis["peak_drift_safe"]
                and kpis["gravity_deflection_safe"]
            )
            if not exact_safe:
                false_safe += 1
                continue
            selected.append(
                {
                    "fuse_type": fuse_type,
                    "params": params,
                    "prediction": candidate,
                    "exact": exact,
                }
            )
            break
        else:
            raise RuntimeError(
                f"could not select four exactly safe {fuse_type} Pareto designs"
            )
    return selected, {
        "attempted": attempted,
        "false_safe": false_safe,
        "failed": failed,
        "accepted": len(selected),
    }


def _constitutive_metadata() -> dict[str, float]:
    params = NiTiParams()
    tangent_error = 0.0
    step = 1e-7
    for strain, z_n in (
        (0.001, 0.0),
        (-0.001, 0.0),
        (0.025, 0.2),
        (-0.025, -0.2),
    ):
        _, _, tangent = niti_update(strain, z_n, params)
        sigma_plus = niti_update(strain + step, z_n, params)[0]
        sigma_minus = niti_update(strain - step, z_n, params)[0]
        numerical = (sigma_plus - sigma_minus) / (2.0 * step)
        tangent_error = max(
            tangent_error, abs((tangent - numerical) / numerical)
        )

    minimum_dissipation = float("inf")
    strain_history = 0.05 * np.sin(np.linspace(0.0, 10.0 * np.pi, 5001))
    eps_n = z_n = sigma_n = 0.0
    for strain in strain_history:
        sigma, z, _ = niti_update(float(strain), z_n, params)
        dissipation = discrete_dissipation(
            eps_n,
            float(strain),
            z_n,
            z,
            sigma_n,
            sigma,
            params,
        )
        minimum_dissipation = min(minimum_dissipation, dissipation)
        eps_n, z_n, sigma_n = float(strain), z, sigma
    return {
        "tangent_error_relative_max": tangent_error,
        "minimum_cyclic_dissipation_MPa": minimum_dissipation,
    }


def _serialize_design(
    design_id: str,
    selected: dict[str, Any],
) -> dict[str, Any]:
    params: GeometryParams = selected["params"]
    exact = selected["exact"]
    geometry = exact["geometry"]
    group_utilization = {
        fuse_type: exact["cases"][fuse_type]["kpis"][
            "peak_member_stress_utilization"
        ]
        for fuse_type in FUSE_TYPES
    }
    element_rows = []
    element_tag = 1
    for group, members in geometry.elements.items():
        for node_i, node_j in members:
            element_rows.append(
                {
                    "tag": element_tag,
                    "group_tag": group,
                    "node_ids": [int(node_i), int(node_j)],
                    "group_peak_static_utilization": {
                        fuse_type: group_utilization[fuse_type][group]
                        for fuse_type in FUSE_TYPES
                    },
                }
            )
            element_tag += 1
    prediction = selected["prediction"]
    return {
        "id": design_id,
        "front_case": selected["fuse_type"],
        "params": asdict(params),
        "predicted": {
            "peak_drift_ucb_pct": prediction["peak_drift_ucb_pct"],
            "gravity_deflection_ucb_mm": prediction[
                "gravity_deflection_ucb_mm"
            ],
            "predicted_total_carbon_per_m2": prediction["prediction"][
                "c_total_per_m2"
            ][0],
        },
        "nodes": [
            {"id": int(node_id), "xyz_m": np.asarray(xyz, dtype=float).tolist()}
            for node_id, xyz in geometry.nodes.items()
        ],
        "elements": element_rows,
        "kpis": {
            fuse_type: exact["cases"][fuse_type]["kpis"]
            for fuse_type in FUSE_TYPES
        },
        "quake_frames": exact["quake_frames"],
    }


def run_optimization(
    dataset_path: str | Path = DATASET_PATH,
    gp_export_path: str | Path = EXPORT_PATH,
    output_path: str | Path = OUTPUT_PATH,
) -> dict[str, Any]:
    surrogate = fit_grouped_cv_surrogates(
        dataset_path=dataset_path, export_path=gp_export_path, seed=SEED
    )
    optimized: dict[str, list[dict[str, Any]]] = {}
    for fuse_type in FUSE_TYPES:
        problem = CanopyOptimizationProblem(fuse_type, surrogate)
        result = minimize(
            problem,
            NSGA2(pop_size=POPULATION_SIZE, eliminate_duplicates=True),
            termination=("n_gen", GENERATION_COUNT),
            seed=SEED,
            verbose=False,
        )
        optimized[fuse_type] = _front_candidates(result, problem)

    exact_cache: dict[tuple[float, ...], dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    certification_stats: dict[str, Any] = {}
    for fuse_type in FUSE_TYPES:
        certified, stats = _select_certified(
            fuse_type, optimized[fuse_type], exact_cache
        )
        selected.extend(certified)
        certification_stats[fuse_type] = stats

    counts = {fuse_type: 0 for fuse_type in FUSE_TYPES}
    certified_designs = []
    for item in selected:
        fuse_type = item["fuse_type"]
        counts[fuse_type] += 1
        certified_designs.append(
            _serialize_design(f"{fuse_type}-{counts[fuse_type]}", item)
        )
    false_safe_total = sum(
        stats["false_safe"] for stats in certification_stats.values()
    )
    attempted_total = sum(
        stats["attempted"] for stats in certification_stats.values()
    )
    if any(
        sum(design["front_case"] == case for design in certified_designs) != 4
        for case in FUSE_TYPES
    ):
        raise RuntimeError("certification output must contain four designs per fuse type")
    payload = {
        "meta": {
            "solver": {
                "name": "OpenSeesPy",
                "version": version("openseespy"),
            },
            **_constitutive_metadata(),
            "gp_false_safe_rates": surrogate.validation["false_safe_rate"],
            "gp_validation_by_case": surrogate.validation["per_case"],
            "optimization": {
                "algorithm": "NSGA2",
                "population_size": POPULATION_SIZE,
                "generations": GENERATION_COUNT,
                "seed": SEED,
            },
            "certification": {
                "attempted_exact_front_points": attempted_total,
                "false_safe_violations_rejected": false_safe_total,
                "false_safe_violation_rate": (
                    false_safe_total / attempted_total
                    if attempted_total
                    else 0.0
                ),
                "by_case": certification_stats,
            },
        },
        "designs": certified_designs,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    payload = run_optimization()
    print(
        f"Saved {len(payload['designs'])} certified Pareto designs to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
