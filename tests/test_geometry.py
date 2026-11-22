import numpy as np

from diagrid_opt.geometry import (
    CanopyGeometry,
    GeometryParams,
    canopy_geometry,
    verify_geometry,
)


def test_geometry_verification_gates():
    params = GeometryParams(
        span=21.0,
        arch_rise=3.0,
        diagrid_divisions=8,
        branch_spread=3.2,
        glulam_depth=0.35,
        niti_fuse_area=100.0,
    )
    geometry = canopy_geometry(params)
    assert isinstance(geometry, CanopyGeometry)
    result = verify_geometry(geometry)

    assert result["floating_nodes"] == []
    assert result["zero_length_elements"] == 0
    assert result["four_quadrant_max_error"] < 1e-12
    assert set(geometry.elements) == {
        "GLULAM_DIAGRID",
        "STEEL_EDGE",
        "TREE_TRUNK",
        "TREE_BRANCH",
        "SEISMIC_FUSE",
    }
    assert len(geometry.support_node_ids) == 4
    assert geometry.shaded_area_m2 > 0.0
    assert all(volume >= 0.0 for volume in geometry.volumes_m3.values())
    assert len(geometry["elements"]["TREE_TRUNK"]) == 4
    assert len(geometry["elements"]["TREE_BRANCH"]) == 16
    assert len(geometry["elements"]["SEISMIC_FUSE"]) == 8
