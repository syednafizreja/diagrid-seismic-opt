from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Any

import numpy as np
import openseespy.opensees as ops

from .ground_motions import DT_SECONDS, GroundMotion, generate_ground_motion
from .sections import (
    NITI_SIG_ACT,
    STEEL_FY,
    SectionProperties,
    section_properties,
)

G_ACCEL = 9.80665
ROOF_DEAD_LOAD = 1.50e3
LIVE_LOAD = 0.75e3
WIND_UPLIFT = 1.20e3


@dataclass(frozen=True)
class OpenSeesModel:
    node_tags: dict[int, int]
    element_groups: dict[str, list[int]]
    element_nodes: dict[int, tuple[int, int]]
    element_sections: dict[int, SectionProperties]
    roof_node_ids: tuple[int, ...]
    support_node_ids: tuple[int, ...]
    roof_tributary_areas: dict[int, float]
    geometry: Any
    fuse_type: str
    fuse_area: float


def _get(geom: Any, name: str) -> Any:
    try:
        return getattr(geom, name)
    except AttributeError:
        return geom[name]


def _roof_tributary_areas(geom: Any) -> dict[int, float]:
    nodes = _get(geom, "nodes")
    node_ids = _get(geom, "node_ids")
    nx = int(_get(geom, "nx"))
    nz = int(_get(geom, "nz"))
    tributary = {node_id: 0.0 for node_id in node_ids.values()}
    for j in range(nz):
        for i in range(nx):
            ids = (
                node_ids[(i, j)],
                node_ids[(i + 1, j)],
                node_ids[(i + 1, j + 1)],
                node_ids[(i, j + 1)],
            )
            a, b, c, d = (np.asarray(nodes[node_id], dtype=float) for node_id in ids)
            area = 0.5 * (
                np.linalg.norm(np.cross(b - a, c - a))
                + np.linalg.norm(np.cross(c - a, d - a))
            )
            for node_id in ids:
                tributary[node_id] += float(area) / 4.0
    return tributary


def build_opensees_model(geom: Any, fuse_type: str) -> OpenSeesModel:
    """Build the SI-unit 3D frame and selected corotational fuse model."""
    sections = section_properties(geom, fuse_type)
    source_nodes = _get(geom, "nodes")
    source_elements = _get(geom, "elements")
    source_supports = tuple(_get(geom, "support_node_ids"))
    node_tags = {node_id: index + 1 for index, node_id in enumerate(source_nodes)}

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for node_id, xyz in source_nodes.items():
        ops.node(node_tags[node_id], *map(float, xyz))

    roof_ids = tuple(_get(geom, "node_ids").values())
    tributary_areas = _roof_tributary_areas(geom)

    trunk_base_ids = {
        base_id for base_id, _ in source_elements["TREE_TRUNK"]
    }
    for support_id in source_supports:
        if support_id in trunk_base_ids:
            ops.fix(node_tags[support_id], 1, 1, 1, 1, 1, 0)
        else:
            ops.fix(node_tags[support_id], 1, 1, 1, 1, 1, 1)
    fuse_anchor_ids = {
        anchor_id for _, anchor_id in source_elements["SEISMIC_FUSE"]
    }
    for anchor_id in fuse_anchor_ids:
        if anchor_id not in source_supports:
            ops.fix(node_tags[anchor_id], 1, 1, 1, 1, 1, 1)

    ops.geomTransf("Linear", 1, 0.0, 0.0, 1.0)
    ops.geomTransf("Linear", 2, 0.0, 1.0, 0.0)

    material_tag = {"niti": 901, "steel": 902}[fuse_type]
    if fuse_type == "niti":
        k1 = 40.0e9
        k2 = (40.0e9 * 70.0e6) / (40.0e9 * 0.06 + 70.0e6)
        beta = (380.0 - 110.0) / 380.0
        ops.uniaxialMaterial(
            "SelfCentering", material_tag, k1, k2, NITI_SIG_ACT, beta, 0.0, 0.071, 1.0
        )
    else:
        ops.uniaxialMaterial(
            "Steel02", material_tag, STEEL_FY, 200.0e9, 0.01, 18.0, 0.925, 0.15
        )

    element_groups: dict[str, list[int]] = {
        group: [] for group in source_elements
    }
    element_nodes: dict[int, tuple[int, int]] = {}
    element_sections: dict[int, SectionProperties] = {}
    element_tag = 1
    for group, pairs in source_elements.items():
        for source_i, source_j in pairs:
            i_tag = node_tags[source_i]
            j_tag = node_tags[source_j]
            p_i = np.asarray(source_nodes[source_i], dtype=float)
            p_j = np.asarray(source_nodes[source_j], dtype=float)
            delta = p_j - p_i
            length = float(np.linalg.norm(delta))
            if length <= 0.0:
                raise ValueError(f"zero-length element in group {group}")
            transform_tag = 1 if abs(delta[1] / length) > 0.9 else 2
            section = sections[group]
            if group == "SEISMIC_FUSE":
                ops.element(
                    "corotTruss",
                    element_tag,
                    i_tag,
                    j_tag,
                    section.area,
                    material_tag,
                )
            else:
                ops.element(
                    "elasticBeamColumn",
                    element_tag,
                    i_tag,
                    j_tag,
                    section.area,
                    section.E,
                    section.G,
                    section.J,
                    section.Iy,
                    section.Iz,
                    transform_tag,
                )
            element_groups[group].append(element_tag)
            element_nodes[element_tag] = (source_i, source_j)
            element_sections[element_tag] = section
            element_tag += 1

    nodal_mass = {node_id: 0.0 for node_id in source_nodes}
    for source_id, tributary in tributary_areas.items():
        nodal_mass[source_id] += ROOF_DEAD_LOAD * tributary / G_ACCEL
    for group, pairs in source_elements.items():
        section = sections[group]
        for source_i, source_j in pairs:
            length = float(
                np.linalg.norm(
                    np.asarray(source_nodes[source_j], dtype=float)
                    - np.asarray(source_nodes[source_i], dtype=float)
                )
            )
            half_mass = 0.5 * length * section.area * section.density
            nodal_mass[source_i] += half_mass
            nodal_mass[source_j] += half_mass

    for source_id, mass in nodal_mass.items():
        ops.mass(node_tags[source_id], mass, mass, mass, 1e-3, 1e-3, 1e-3)

    return OpenSeesModel(
        node_tags=node_tags,
        element_groups=element_groups,
        element_nodes=element_nodes,
        element_sections=element_sections,
        roof_node_ids=roof_ids,
        support_node_ids=source_supports + tuple(sorted(fuse_anchor_ids)),
        roof_tributary_areas=tributary_areas,
        geometry=geom,
        fuse_type=fuse_type,
        fuse_area=sections["SEISMIC_FUSE"].area,
    )


def _apply_static_load(model: OpenSeesModel, case: str) -> dict[int, float]:
    geom = model.geometry
    source_nodes = _get(geom, "nodes")
    source_elements = _get(geom, "elements")
    factor_g = 1.35 if case == "gravity" else 0.90
    roof_pressure = 1.50 * LIVE_LOAD if case == "gravity" else 1.50 * WIND_UPLIFT
    load_y: dict[int, float] = {node_id: 0.0 for node_id in source_nodes}

    for node_id, area in model.roof_tributary_areas.items():
        load_y[node_id] += (
            -factor_g * ROOF_DEAD_LOAD * area
            + (roof_pressure * area if case == "wind_uplift" else -roof_pressure * area)
        )

    sections = section_properties(geom, model.fuse_type)
    for group, pairs in source_elements.items():
        section = sections[group]
        for node_i, node_j in pairs:
            length = float(
                np.linalg.norm(
                    np.asarray(source_nodes[node_j], dtype=float)
                    - np.asarray(source_nodes[node_i], dtype=float)
                )
            )
            member_weight = section.area * length * section.density * G_ACCEL
            load_y[node_i] -= factor_g * member_weight / 2.0
            load_y[node_j] -= factor_g * member_weight / 2.0

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for node_id, force_y in load_y.items():
        if force_y != 0.0:
            ops.load(model.node_tags[node_id], 0.0, force_y, 0.0, 0.0, 0.0, 0.0)
    return load_y


def _apply_gravity_load(model: OpenSeesModel) -> None:
    geom = model.geometry
    source_nodes = _get(geom, "nodes")
    source_elements = _get(geom, "elements")
    load_y: dict[int, float] = {node_id: 0.0 for node_id in source_nodes}

    for node_id, area in model.roof_tributary_areas.items():
        load_y[node_id] -= ROOF_DEAD_LOAD * area
    sections = section_properties(geom, model.fuse_type)
    for group, pairs in source_elements.items():
        section = sections[group]
        for node_i, node_j in pairs:
            length = float(
                np.linalg.norm(
                    np.asarray(source_nodes[node_j], dtype=float)
                    - np.asarray(source_nodes[node_i], dtype=float)
                )
            )
            half_weight = (
                0.5 * section.area * length * section.density * G_ACCEL
            )
            load_y[node_i] -= half_weight
            load_y[node_j] -= half_weight

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for node_id, force_y in load_y.items():
        if force_y != 0.0:
            ops.load(model.node_tags[node_id], 0.0, force_y, 0.0, 0.0, 0.0, 0.0)


def _stress_utilizations(model: OpenSeesModel) -> dict[str, float]:
    group_utilization = {group: 0.0 for group in model.element_groups}
    for group, tags in model.element_groups.items():
        for tag in tags:
            section = model.element_sections[tag]
            if group == "SEISMIC_FUSE":
                axial_response = ops.eleResponse(tag, "axialForce")
                axial = abs(float(axial_response[0]))
                stress = axial / section.area
            else:
                local_force = ops.eleResponse(tag, "localForce")
                if len(local_force) < 12:
                    raise RuntimeError(f"unexpected local-force response for element {tag}")
                axial = max(abs(float(local_force[0])), abs(float(local_force[6])))
                my = max(abs(float(local_force[4])), abs(float(local_force[10])))
                mz = max(abs(float(local_force[5])), abs(float(local_force[11])))
                stress = (
                    axial / section.area
                    + my * section.c_y / section.Iy
                    + mz * section.c_z / section.Iz
                )
            group_utilization[group] = max(
                group_utilization[group], stress / section.capacity
            )
    return group_utilization


def _analyze_case(model: OpenSeesModel, case: str) -> dict[str, Any]:
    applied_load_y = _apply_static_load(model, case)
    ops.system("UmfPack")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.integrator("LoadControl", 1.0)
    ops.test("NormUnbalance", 1e-7, 30)
    ops.algorithm("Newton")
    ops.analysis("Static")
    code = ops.analyze(1)
    if code != 0:
        raise RuntimeError(f"OpenSees static {case} analysis failed with code {code}")

    roof_deflection = max(
        abs(float(ops.nodeDisp(model.node_tags[node_id], 2)))
        for node_id in model.roof_node_ids
    )
    ops.reactions()
    applied_total_y = sum(applied_load_y.values())
    reaction_total_y = sum(
        float(ops.nodeReaction(model.node_tags[node_id], 2))
        for node_id in model.support_node_ids
    )
    equilibrium_error = abs(applied_total_y + reaction_total_y) / max(
        abs(applied_total_y), 1.0
    )
    utilizations = _stress_utilizations(model)

    eigenvalues = ops.eigen(2)
    positive = [float(value) for value in eigenvalues if float(value) > 0.0]
    if not positive:
        raise RuntimeError("OpenSees did not return a positive eigenvalue")
    t1 = 2.0 * pi / sqrt(min(positive))
    return {
        "max_roof_deflection_m": roof_deflection,
        "peak_member_stress_utilization": utilizations,
        "T1_s": t1,
        "equilibrium_error": equilibrium_error,
    }


def run_static_checks(geom: Any, fuse_type: str = "niti") -> dict[str, Any]:
    """Run factored gravity and wind-uplift static checks on separate models."""
    cases = {}
    for case in ("gravity", "wind_uplift"):
        model = build_opensees_model(geom, fuse_type)
        cases[case] = _analyze_case(model, case)

    return {
        "cases": cases,
        "max_roof_deflection_m": max(
            case["max_roof_deflection_m"] for case in cases.values()
        ),
        "peak_member_stress_utilization": {
            group: max(
                case["peak_member_stress_utilization"][group]
                for case in cases.values()
            )
            for group in cases["gravity"]["peak_member_stress_utilization"]
        },
        "T1_s": cases["gravity"]["T1_s"],
        "equilibrium_error": max(
            case["equilibrium_error"] for case in cases.values()
        ),
    }


def _roof_drifts(model: OpenSeesModel) -> tuple[float, float]:
    roof_x = [
        float(ops.nodeDisp(model.node_tags[node_id], 1))
        for node_id in model.roof_node_ids
    ]
    return max(abs(value) for value in roof_x) / 4.5, float(np.mean(roof_x) / 4.5)


def _support_base_shear(model: OpenSeesModel) -> float:
    ops.reactions()
    return abs(
        sum(
            float(ops.nodeReaction(model.node_tags[node_id], 1))
            for node_id in model.support_node_ids
        )
    )


def _lead_fuse_response(model: OpenSeesModel) -> tuple[float, float]:
    fuse_tag = model.element_groups["SEISMIC_FUSE"][0]
    source_i, source_j = model.element_nodes[fuse_tag]
    nodes = _get(model.geometry, "nodes")
    coordinates = [
        np.asarray(nodes[node_id], dtype=float)
        + np.asarray(
            ops.nodeDisp(model.node_tags[node_id])[:3],
            dtype=float,
        )
        for node_id in (source_i, source_j)
    ]
    initial_length = float(
        np.linalg.norm(
            np.asarray(nodes[source_j], dtype=float)
            - np.asarray(nodes[source_i], dtype=float)
        )
    )
    strain = float(np.linalg.norm(coordinates[1] - coordinates[0]) / initial_length - 1.0)
    axial_force = float(ops.eleResponse(fuse_tag, "axialForce")[0])
    stress = axial_force / model.fuse_area
    return strain, stress


def _max_fuse_strain(model: OpenSeesModel) -> float:
    nodes = _get(model.geometry, "nodes")
    maximum = 0.0
    for element_tag in model.element_groups["SEISMIC_FUSE"]:
        source_i, source_j = model.element_nodes[element_tag]
        initial_length = float(
            np.linalg.norm(
                np.asarray(nodes[source_j], dtype=float)
                - np.asarray(nodes[source_i], dtype=float)
            )
        )
        current_coordinates = [
            np.asarray(nodes[node_id], dtype=float)
            + np.asarray(
                ops.nodeDisp(model.node_tags[node_id])[:3],
                dtype=float,
            )
            for node_id in (source_i, source_j)
        ]
        strain = (
            float(
                np.linalg.norm(current_coordinates[1] - current_coordinates[0])
            )
            / initial_length
            - 1.0
        )
        maximum = max(maximum, abs(strain))
    return maximum


def _seismic_damping(eigenvalues: list[float]) -> tuple[float, float]:
    if len(eigenvalues) < 2 or any(value <= 0.0 for value in eigenvalues[:2]):
        raise RuntimeError("two positive eigenvalues are required for Rayleigh damping")
    omega_1, omega_2 = (sqrt(value) for value in eigenvalues[:2])
    damping_ratio = 0.05
    alpha_m = (
        2.0 * damping_ratio * omega_1 * omega_2 / (omega_1 + omega_2)
    )
    beta_k = 2.0 * damping_ratio / (omega_1 + omega_2)
    return alpha_m, beta_k


def _advance_transient_step(dt_s: float) -> bool:
    ops.test("NormDispIncr", 1e-8, 30)
    ops.algorithm("Newton")
    if ops.analyze(1, dt_s) == 0:
        return False

    ops.algorithm("NewtonLineSearch")
    if ops.analyze(1, dt_s) == 0:
        return True

    ops.algorithm("ModifiedNewton")
    substep = dt_s / 4.0
    for substep_index in range(4):
        if ops.analyze(1, substep) != 0:
            ops.algorithm("NewtonLineSearch")
            if ops.analyze(1, substep) != 0:
                raise RuntimeError(
                    f"transient step failed after Newton, NewtonLineSearch, "
                    f"and dt/4 ModifiedNewton fallback "
                    f"(substep {substep_index + 1}/4)"
                )
    return True


def run_seismic_analysis(
    geom: Any,
    fuse_type: str,
    pga_g: float = 0.40,
    seed: int = 101,
    record_frames: bool = False,
) -> dict[str, Any]:
    """Run gravity-preloaded nonlinear X-direction earthquake analysis."""
    motion: GroundMotion = generate_ground_motion(seed, pga_g, DT_SECONDS)
    model = build_opensees_model(geom, fuse_type)

    _apply_gravity_load(model)
    ops.system("UmfPack")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.integrator("LoadControl", 1.0)
    ops.test("NormUnbalance", 1e-7, 30)
    ops.algorithm("Newton")
    ops.analysis("Static")
    gravity_code = ops.analyze(1)
    if gravity_code != 0:
        raise RuntimeError(
            f"OpenSees seismic gravity preload failed with code {gravity_code}"
        )
    ops.loadConst("-time", 0.0)

    eigenvalues = [float(value) for value in ops.eigen(2)]
    alpha_m, beta_k = _seismic_damping(eigenvalues)
    ops.rayleigh(alpha_m, 0.0, 0.0, beta_k)

    acceleration_tag = 2
    pattern_tag = 2
    ops.timeSeries(
        "Path",
        acceleration_tag,
        "-dt",
        motion.dt_s,
        "-values",
        *motion.acceleration_m_s2.tolist(),
    )
    ops.pattern(
        "UniformExcitation", pattern_tag, 1, "-accel", acceleration_tag
    )
    ops.wipeAnalysis()
    ops.system("UmfPack")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-8, 30)
    ops.algorithm("Newton")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.analysis("Transient")

    steps = motion.acceleration_m_s2.size - 1
    output_interval = 0.02 if record_frames else 0.1
    downsample = max(1, int(round(output_interval / motion.dt_s)))
    sample_times: list[float] = []
    sample_drifts: list[float] = []
    sample_signed_drifts: list[float] = []
    sample_displacements: list[np.ndarray] = []
    sample_fuse_strains: list[float] = []
    sample_fuse_stresses: list[float] = []
    peak_roof_drift = 0.0
    peak_base_shear = 0.0
    max_fuse_strain = 0.0
    fallback_steps = 0
    residual_roof_drifts: list[float] = []
    residual_signed_roof_drifts: list[float] = []
    source_node_ids = tuple(sorted(model.node_tags))
    source_nodes = _get(model.geometry, "nodes")

    for step in range(1, steps + 1):
        try:
            used_fallback = _advance_transient_step(motion.dt_s)
        except RuntimeError as error:
            raise RuntimeError(
                f"OpenSees {fuse_type} transient analysis failed at "
                f"step {step}/{steps}, t={step * motion.dt_s:.2f} s "
                f"(seed={seed}, PGA={pga_g}g): {error}"
            ) from error
        fallback_steps += int(used_fallback)

        drift_ratio, signed_drift_ratio = _roof_drifts(model)
        peak_roof_drift = max(peak_roof_drift, drift_ratio)
        max_fuse_strain = max(max_fuse_strain, _max_fuse_strain(model))
        base_shear = _support_base_shear(model)
        peak_base_shear = max(peak_base_shear, base_shear)
        current_time = float(ops.getTime())
        if current_time >= motion.time_s[-1] - 1.0:
            residual_roof_drifts.append(drift_ratio)
            residual_signed_roof_drifts.append(signed_drift_ratio)

        if step % downsample == 0 or step == steps:
            sample_times.append(current_time)
            sample_drifts.append(drift_ratio)
            sample_signed_drifts.append(signed_drift_ratio)
            if record_frames:
                sample_displacements.append(
                    np.asarray(
                        [
                            ops.nodeDisp(model.node_tags[node_id])
                            for node_id in source_node_ids
                        ],
                        dtype=float,
                    )
                )
                fuse_strain, fuse_stress = _lead_fuse_response(model)
                sample_fuse_strains.append(fuse_strain)
                sample_fuse_stresses.append(fuse_stress)

    signed_permanent_residual = float(np.mean(residual_signed_roof_drifts))
    result: dict[str, Any] = {
        "peak_roof_drift_ratio": peak_roof_drift,
        "permanent_residual_roof_drift_ratio": abs(
            signed_permanent_residual
        ),
        "signed_permanent_residual_roof_drift_ratio": (
            signed_permanent_residual
        ),
        "mean_absolute_tail_roof_drift_ratio": float(
            np.mean(residual_roof_drifts)
        ),
        "peak_base_shear_N": peak_base_shear,
        "max_fuse_strain": max_fuse_strain,
        "pga_g": float(pga_g),
        "seed": seed,
        "converged_steps": steps,
        "fallback_steps": fallback_steps,
    }
    if record_frames:
        result["history"] = {
            "time_s": np.asarray(sample_times, dtype=float),
            "roof_drift_ratio": np.asarray(sample_drifts, dtype=float),
            "signed_roof_drift_ratio": np.asarray(
                sample_signed_drifts, dtype=float
            ),
            "node_ids": np.asarray(source_node_ids, dtype=int),
            "nodal_displacements_m_rad": np.asarray(
                sample_displacements, dtype=float
            ),
            "lead_fuse_strain": np.asarray(sample_fuse_strains, dtype=float),
            "lead_fuse_stress_Pa": np.asarray(sample_fuse_stresses, dtype=float),
        }
    return result
