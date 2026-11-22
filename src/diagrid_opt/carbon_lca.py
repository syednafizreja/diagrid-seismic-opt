from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .sections import section_properties


GLULAM_FOSSIL_FACTOR = 0.512
GLULAM_BIOGENIC_FACTOR = -0.98
STEEL_FACTOR = 1.55
NITI_FACTOR = 28.0
CONCRETE_FOOTINGS_KGCO2E = 1612.8
GLAZING_FACTOR_KGCO2E_M2 = 45.0


@dataclass(frozen=True)
class CarbonLCAResult:
    c_a1_a3_kgco2e: float
    c_repair_kgco2e: float
    fuse_replacement_kgco2e: float
    glazing_repair_kgco2e: float
    residual_repair_kgco2e: float
    total_kgco2e: float
    total_intensity_kgco2e_m2: float


def _metric(metrics: dict[str, Any], name: str) -> float:
    try:
        value = float(metrics[name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative number") from error
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def assess_lifecycle_carbon(
    geometry: Any,
    fuse_type: str,
    seismic_metrics: dict[str, Any],
    *,
    include_biogenic: bool = False,
) -> CarbonLCAResult:
    """Assess A1-A3 and the specified single-event repair carbon in kgCO2e."""
    peak_drift = _metric(seismic_metrics, "peak_roof_drift_ratio")
    residual_drift = _metric(
        seismic_metrics, "permanent_residual_roof_drift_ratio"
    )
    max_fuse_strain = _metric(seismic_metrics, "max_fuse_strain")
    shaded_area = float(geometry.shaded_area_m2)
    if not isfinite(shaded_area) or shaded_area <= 0.0:
        raise ValueError("geometry shaded_area_m2 must be positive and finite")

    sections = section_properties(geometry, fuse_type)
    nodes = geometry.nodes
    elements = geometry.elements
    superstructure_carbon = 0.0
    fuse_carbon = 0.0
    for group, members in elements.items():
        section = sections[group]
        member_length = sum(
            float(np.linalg.norm(np.asarray(nodes[b]) - np.asarray(nodes[a])))
            for a, b in members
        )
        mass_kg = member_length * section.area * section.density
        if group == "GLULAM_DIAGRID":
            factor = (
                GLULAM_BIOGENIC_FACTOR
                if include_biogenic
                else GLULAM_FOSSIL_FACTOR
            )
        elif group == "SEISMIC_FUSE":
            factor = NITI_FACTOR if fuse_type == "niti" else STEEL_FACTOR
        else:
            factor = STEEL_FACTOR
        group_carbon = mass_kg * factor
        superstructure_carbon += group_carbon
        if group == "SEISMIC_FUSE":
            fuse_carbon = group_carbon

    c_a1_a3 = CONCRETE_FOOTINGS_KGCO2E + superstructure_carbon
    if fuse_type == "steel" and max_fuse_strain > 0.00355:
        fuse_replacement = 1.5 * fuse_carbon
    elif fuse_type == "niti" and max_fuse_strain > 0.06:
        fuse_replacement = 1.5 * fuse_carbon
    else:
        fuse_replacement = 0.0

    if peak_drift <= 0.005:
        glazing_damage_ratio = 0.0
    elif peak_drift <= 0.025:
        glazing_damage_ratio = 0.35 * (peak_drift - 0.005) / 0.020
    else:
        glazing_damage_ratio = 0.60
    glazing_repair = (
        GLAZING_FACTOR_KGCO2E_M2 * shaded_area * glazing_damage_ratio
    )
    residual_repair_ratio = min(1.0, residual_drift / 0.005)
    residual_repair = 0.40 * superstructure_carbon * residual_repair_ratio
    c_repair = fuse_replacement + glazing_repair + residual_repair
    total = c_a1_a3 + c_repair
    return CarbonLCAResult(
        c_a1_a3_kgco2e=c_a1_a3,
        c_repair_kgco2e=c_repair,
        fuse_replacement_kgco2e=fuse_replacement,
        glazing_repair_kgco2e=glazing_repair,
        residual_repair_kgco2e=residual_repair,
        total_kgco2e=total,
        total_intensity_kgco2e_m2=total / shaded_area,
    )
