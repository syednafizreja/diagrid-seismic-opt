from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class GeometryParams:
    span: float
    arch_rise: float
    diagrid_divisions: int
    branch_spread: float
    glulam_depth: float = 0.22
    b_glulam: float = 0.16
    niti_fuse_area: float = 100.0

    def __post_init__(self) -> None:
        if not 14.0 <= self.span <= 28.0:
            raise ValueError("span must lie in [14, 28] m")
        if not 0.08 * self.span <= self.arch_rise <= 0.22 * self.span:
            raise ValueError("arch_rise must lie in [0.08L, 0.22L]")
        if self.diagrid_divisions % 2 != 0:
            raise ValueError("diagrid_divisions must be even")
        if not 6 <= self.diagrid_divisions <= 12:
            raise ValueError("diagrid_divisions must lie in [6, 12]")
        if not 0.12 * self.span <= self.branch_spread <= 0.22 * self.span:
            raise ValueError("branch_spread must lie in [0.12L, 0.22L]")
        if not 0.22 <= self.glulam_depth <= 0.48:
            raise ValueError("glulam_depth must lie in [0.22, 0.48] m")
        if not 60.0 <= self.niti_fuse_area <= 260.0:
            raise ValueError("niti_fuse_area must lie in [60, 260] mm^2")


@dataclass(frozen=True)
class CanopyGeometry:
    nodes: Dict[int, np.ndarray]
    elements: Dict[str, list[tuple[int, int]]]
    support_node_ids: Tuple[int, ...]
    volumes_m3: Dict[str, float]
    shaded_area_m2: float
    span: float
    width: float
    nx: int
    nz: int
    node_ids: Dict[tuple[int, int], int]
    bifurcation_ids: Tuple[int, ...]
    column_positions: np.ndarray
    branch_spread: float
    glulam_depth: float
    b_glulam: float
    niti_fuse_area: float

    def __getitem__(self, key: str):
        return getattr(self, key)


def _roof_height(x: float, z: float, span: float, arch_rise: float) -> float:
    width = 0.5 * span
    return 4.5 + arch_rise * (1.0 - (2.0 * x / span) ** 2) * (
        1.0 - 0.5 * (2.0 * z / width) ** 2
    )


def canopy_geometry(params: GeometryParams) -> CanopyGeometry:
    L = params.span
    width = 0.5 * L
    nx = params.diagrid_divisions
    nz = max(4, nx // 2)

    nodes: dict[int, np.ndarray] = {}
    node_ids: dict[tuple[int, int], int] = {}
    for j in range(nz + 1):
        z = -width / 2.0 + j * width / nz
        for i in range(nx + 1):
            x = -L / 2.0 + i * L / nx
            y = _roof_height(x, z, L, params.arch_rise)
            node_id = len(nodes)
            nodes[node_id] = np.array([x, y, z], dtype=float)
            node_ids[(i, j)] = node_id

    elements: dict[str, list[tuple[int, int]]] = {
        "GLULAM_DIAGRID": [],
        "STEEL_EDGE": [],
        "TREE_TRUNK": [],
        "TREE_BRANCH": [],
        "SEISMIC_FUSE": [],
    }

    for j in range(nz):
        for i in range(nx):
            elements["GLULAM_DIAGRID"].append(
                (node_ids[(i, j)], node_ids[(i + 1, j + 1)])
            )
            elements["GLULAM_DIAGRID"].append(
                (node_ids[(i + 1, j)], node_ids[(i, j + 1)])
            )

    for i in (0, nx):
        for j in range(nz):
            elements["STEEL_EDGE"].append(
                (node_ids[(i, j)], node_ids[(i, j + 1)])
            )
    for j in (0, nz):
        for i in range(nx):
            elements["STEEL_EDGE"].append(
                (node_ids[(i, j)], node_ids[(i + 1, j)])
            )

    roof_nodes = list(node_ids.values())
    roof_xy = np.array([(nodes[n][0], nodes[n][2]) for n in roof_nodes])
    column_positions = [
        (-0.32 * L, -0.25 * width),
        (-0.32 * L, 0.25 * width),
        (0.32 * L, -0.25 * width),
        (0.32 * L, 0.25 * width),
    ]
    y_b = 0.55 * 4.5

    bifurcation_ids: list[int] = []
    for x_col, z_col in column_positions:
        bifurcation_id = len(nodes)
        nodes[bifurcation_id] = np.array([x_col, y_b, z_col], dtype=float)
        bifurcation_ids.append(bifurcation_id)

        ground_id = len(nodes)
        nodes[ground_id] = np.array([x_col, 0.0, z_col], dtype=float)
        elements["TREE_TRUNK"].append((ground_id, bifurcation_id))

        anchor = np.array([x_col, z_col])
        distances = np.linalg.norm(roof_xy - anchor, axis=1)
        nearest_ids = np.array(roof_nodes)[np.argsort(distances, kind="stable")[:4]]
        for roof_id in nearest_ids:
            elements["TREE_BRANCH"].append((bifurcation_id, int(roof_id)))

        for x_anchor in (x_col - 1.5, x_col + 1.5):
            fuse_ground_id = len(nodes)
            nodes[fuse_ground_id] = np.array([x_anchor, 0.0, z_col], dtype=float)
            elements["SEISMIC_FUSE"].append((bifurcation_id, fuse_ground_id))

    column_positions_array = np.array(
        [(x, 0.0, z) for x, z in column_positions], dtype=float
    )

    volumes_m3: dict[str, float] = {
        "GLULAM_DIAGRID": 0.0,
        "STEEL_EDGE": 0.0,
        "TREE_TRUNK": 0.0,
        "TREE_BRANCH": 0.0,
        "SEISMIC_FUSE": 0.0,
    }
    for tag, pairs in elements.items():
        for a, b in pairs:
            length = float(np.linalg.norm(nodes[a] - nodes[b]))
            if tag == "GLULAM_DIAGRID":
                area = params.b_glulam * params.glulam_depth
            elif tag == "SEISMIC_FUSE":
                area = params.niti_fuse_area * 1e-6
            else:
                area = 0.01 * 0.01
            volumes_m3[tag] += length * area

    shaded_area_m2 = 0.0
    for j in range(nz):
        for i in range(nx):
            corners = (
                nodes[node_ids[(i, j)]],
                nodes[node_ids[(i + 1, j)]],
                nodes[node_ids[(i + 1, j + 1)]],
                nodes[node_ids[(i, j + 1)]],
            )
            u = corners[1] - corners[0]
            v = corners[3] - corners[0]
            shaded_area_m2 += 0.5 * float(np.linalg.norm(np.cross(u, v)))

    support_ids = tuple(
        next(
            node_id
            for node_id, coordinate in nodes.items()
            if np.array_equal(coordinate, np.array([x, 0.0, z]))
        )
        for x, z in column_positions
    )

    return CanopyGeometry(
        nodes=nodes,
        elements=elements,
        support_node_ids=support_ids,
        volumes_m3=volumes_m3,
        shaded_area_m2=shaded_area_m2,
        span=L,
        width=width,
        nx=nx,
        nz=nz,
        node_ids=node_ids,
        bifurcation_ids=tuple(bifurcation_ids),
        column_positions=column_positions_array,
        branch_spread=params.branch_spread,
        glulam_depth=params.glulam_depth,
        b_glulam=params.b_glulam,
        niti_fuse_area=params.niti_fuse_area,
    )


def verify_geometry(geometry: dict) -> dict:
    nodes = geometry["nodes"]
    elements = geometry["elements"]
    used = {
        node_id
        for element_pairs in elements.values()
        for pair in element_pairs
        for node_id in pair
    }
    zero_length = sum(
        np.linalg.norm(nodes[a] - nodes[b]) <= 1e-12
        for pair in elements.values()
        for a, b in pair
    )

    nx = geometry["nx"]
    nz = geometry["nz"]
    node_ids = geometry["node_ids"]
    roof_errors = []
    for j in range(nz + 1):
        z = nodes[node_ids[(0, j)]][2]
        for i in range(nx + 1):
            p = nodes[node_ids[(i, j)]]
            mirror = nodes[node_ids[(nx - i, nz - j)]]
            roof_errors.append(
                np.linalg.norm(p - np.array([-mirror[0], mirror[1], -mirror[2]]))
            )

    return {
        "floating_nodes": sorted(set(nodes) - used),
        "zero_length_elements": zero_length,
        "four_quadrant_max_error": float(max(roof_errors, default=0.0)),
    }
