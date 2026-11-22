from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STEEL_FY = 355.0e6
NITI_SIG_ACT = 380.0e6


@dataclass(frozen=True)
class SectionProperties:
    area: float
    E: float
    G: float
    J: float
    Iy: float
    Iz: float
    density: float
    capacity: float
    c_y: float
    c_z: float


def _get(geom: Any, name: str) -> Any:
    try:
        return getattr(geom, name)
    except AttributeError:
        return geom[name]


def section_properties(
    geom: Any, fuse_type: str
) -> dict[str, SectionProperties]:
    depth = float(_get(geom, "glulam_depth"))
    width = float(_get(geom, "b_glulam"))
    niti_area = float(_get(geom, "niti_fuse_area")) * 1e-6
    glulam_area = width * depth
    glulam_iy = width * depth**3 / 12.0
    glulam_iz = depth * width**3 / 12.0
    long_side = max(width, depth)
    short_side = min(width, depth)
    glulam_j = long_side * short_side**3 * (
        1.0 / 3.0
        - 0.21
        * short_side
        / long_side
        * (1.0 - short_side**4 / (12.0 * long_side**4))
    )

    sections = {
        "GLULAM_DIAGRID": SectionProperties(
            glulam_area,
            12.6e9,
            0.65e9,
            glulam_j,
            glulam_iy,
            glulam_iz,
            420.0,
            18.0e6,
            depth / 2.0,
            width / 2.0,
        ),
        "STEEL_EDGE": SectionProperties(
            9.49e-3,
            210e9,
            81e9,
            1.41e-4,
            9.06e-5,
            9.06e-5,
            7850.0,
            STEEL_FY,
            0.125,
            0.125,
        ),
        "TREE_TRUNK": SectionProperties(
            1.22e-2,
            210e9,
            81e9,
            2.97e-4,
            1.48e-4,
            1.48e-4,
            7850.0,
            STEEL_FY,
            0.3239 / 2.0,
            0.3239 / 2.0,
        ),
        "TREE_BRANCH": SectionProperties(
            4.67e-3,
            210e9,
            81e9,
            4.03e-5,
            2.02e-5,
            2.02e-5,
            7850.0,
            STEEL_FY,
            0.1937 / 2.0,
            0.1937 / 2.0,
        ),
    }
    if fuse_type == "niti":
        sections["SEISMIC_FUSE"] = SectionProperties(
            niti_area,
            40.0e9,
            0.0,
            0.0,
            0.0,
            0.0,
            6450.0,
            NITI_SIG_ACT,
            0.0,
            0.0,
        )
    elif fuse_type == "steel":
        steel_area = (NITI_SIG_ACT / STEEL_FY) * niti_area
        sections["SEISMIC_FUSE"] = SectionProperties(
            steel_area,
            200.0e9,
            0.0,
            0.0,
            0.0,
            0.0,
            7850.0,
            STEEL_FY,
            0.0,
            0.0,
        )
    else:
        raise ValueError("fuse_type must be 'niti' or 'steel'")
    return sections
