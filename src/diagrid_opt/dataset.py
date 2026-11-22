from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .carbon_lca import assess_lifecycle_carbon
from .geometry import GeometryParams, canopy_geometry
from .solver_opensees import run_seismic_analysis, run_static_checks


DESIGN_COUNT = 80
DEFAULT_SEED = 42
FUSE_TYPES = ("steel", "niti")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "data" / "verified_lhs_dataset.csv"
)


def sample_lhs_designs(
    n: int = DESIGN_COUNT, seed: int = DEFAULT_SEED
) -> list[GeometryParams]:
    if n < 1:
        raise ValueError("n must be a positive integer")
    samples = qmc.LatinHypercube(d=5, seed=seed).random(n)
    divisions = (6, 8, 10, 12)
    designs = []
    for span_u, rise_u, division_u, depth_u, area_u in samples:
        span = 14.0 + 14.0 * span_u
        division_index = min(int(division_u * len(divisions)), len(divisions) - 1)
        designs.append(
            GeometryParams(
                span=float(span),
                arch_rise=float(span * (0.08 + 0.14 * rise_u)),
                diagrid_divisions=divisions[division_index],
                branch_spread=float(0.17 * span),
                glulam_depth=float(0.22 + 0.26 * depth_u),
                niti_fuse_area=float(60.0 + 200.0 * area_u),
            )
        )
    return designs


def _evaluate_fuse_case(
    task: tuple[int, GeometryParams, str]
) -> dict[str, Any]:
    design_id, params, fuse_type = task
    geometry = canopy_geometry(params)
    static = run_static_checks(geometry, fuse_type=fuse_type)
    for case_name, case in static["cases"].items():
        equilibrium_error = float(case["equilibrium_error"])
        if not np.isfinite(equilibrium_error) or equilibrium_error >= 1e-4:
            raise RuntimeError(
                f"design {design_id} {fuse_type} static {case_name} "
                f"equilibrium error {equilibrium_error:.6g} is not below 1e-4"
            )

    seismic = run_seismic_analysis(
        geometry, fuse_type=fuse_type, pga_g=0.40, seed=101
    )
    if seismic["converged_steps"] != 1800:
        raise RuntimeError(
            f"design {design_id} {fuse_type} converged "
            f"{seismic['converged_steps']} of 1800 seismic steps"
        )
    seismic_values = {
        name: float(seismic[name])
        for name in (
            "peak_roof_drift_ratio",
            "permanent_residual_roof_drift_ratio",
            "signed_permanent_residual_roof_drift_ratio",
            "peak_base_shear_N",
            "max_fuse_strain",
        )
    }
    if not all(np.isfinite(value) for value in seismic_values.values()):
        raise RuntimeError(f"design {design_id} {fuse_type} has non-finite results")

    carbon = assess_lifecycle_carbon(geometry, fuse_type, seismic_values)
    return {
        "design_id": design_id,
        "fuse_type": fuse_type,
        "static_gravity_deflection_m": float(
            static["cases"]["gravity"]["max_roof_deflection_m"]
        ),
        "static_wind_uplift_deflection_m": float(
            static["cases"]["wind_uplift"]["max_roof_deflection_m"]
        ),
        "T1_s": float(static["T1_s"]),
        "static_equilibrium_error": float(static["equilibrium_error"]),
        "shaded_area_m2": float(geometry.shaded_area_m2),
        **seismic_values,
        "c_a1_a3_kgco2e": carbon.c_a1_a3_kgco2e,
        "c_repair_kgco2e": carbon.c_repair_kgco2e,
        "total_kgco2e": carbon.total_kgco2e,
        "total_intensity_kgco2e_m2": carbon.total_intensity_kgco2e_m2,
    }


def generate_verified_dataset(
    n: int = DESIGN_COUNT,
    seed: int = DEFAULT_SEED,
    workers: int | None = None,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    designs = sample_lhs_designs(n, seed)
    worker_count = workers if workers is not None else min(4, os.cpu_count() or 1)
    if worker_count < 1:
        raise ValueError("workers must be a positive integer")

    tasks = [
        (design_id, design, fuse_type)
        for design_id, design in enumerate(designs)
        for fuse_type in FUSE_TYPES
    ]
    results: dict[tuple[int, str], dict[str, Any]] = {}
    failures: dict[int, list[str]] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_evaluate_fuse_case, task): (task[0], task[2])
            for task in tasks
        }
        for future in as_completed(futures):
            design_id, fuse_type = futures[future]
            try:
                result = future.result()
            except RuntimeError as error:
                failures.setdefault(design_id, []).append(
                    f"{fuse_type}: {error}"
                )
            else:
                results[(design_id, fuse_type)] = result

    rows = []
    for design_id, design in enumerate(designs):
        pair = [results.get((design_id, fuse_type)) for fuse_type in FUSE_TYPES]
        if any(result is None for result in pair):
            continue
        row: dict[str, Any] = {
            "design_id": design_id,
            "span_m": design.span,
            "arch_rise_ratio": design.arch_rise / design.span,
            "diagrid_divisions": design.diagrid_divisions,
            "glulam_depth_m": design.glulam_depth,
            "niti_fuse_area_mm2": design.niti_fuse_area,
            "branch_spread_ratio": design.branch_spread / design.span,
            "shaded_area_m2": float(pair[0]["shaded_area_m2"]),
        }
        for fuse_type, result in zip(FUSE_TYPES, pair):
            for name, value in result.items():
                if name not in ("design_id", "fuse_type", "shaded_area_m2"):
                    row[f"{fuse_type}_{name}"] = value
        rows.append(row)

    if not rows:
        raise RuntimeError("no design passed both steel and NiTi verification")
    dataframe = pd.DataFrame(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)
    if failures:
        print(
            f"Excluded {len(failures)} design(s) with failed cases; "
            f"saved {len(dataframe)} verified paired designs to {destination}.",
            file=sys.stderr,
        )
        for design_id, messages in sorted(failures.items()):
            print(f"  design {design_id}: {'; '.join(messages)}", file=sys.stderr)
    return dataframe
