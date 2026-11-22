import pytest

from diagrid_opt.geometry import GeometryParams, canopy_geometry
from diagrid_opt.solver_opensees import run_static_checks


@pytest.fixture
def baseline_geometry():
    return canopy_geometry(
        GeometryParams(
            span=20.0,
            arch_rise=3.0,
            diagrid_divisions=6,
            branch_spread=3.2,
            glulam_depth=0.35,
            niti_fuse_area=100.0,
        )
    )


@pytest.mark.parametrize("fuse_type", ["steel", "niti"])
def test_static_load_cases_satisfy_equilibrium_and_period_gates(
    baseline_geometry, fuse_type
):
    result = run_static_checks(baseline_geometry, fuse_type=fuse_type)
    assert set(result["cases"]) == {"gravity", "wind_uplift"}
    for case in result["cases"].values():
        assert case["equilibrium_error"] < 1e-4
        assert 0.15 <= case["T1_s"] <= 1.50
        assert case["max_roof_deflection_m"] > 0.0
        assert set(case["peak_member_stress_utilization"]) == {
            "GLULAM_DIAGRID",
            "STEEL_EDGE",
            "TREE_TRUNK",
            "TREE_BRANCH",
            "SEISMIC_FUSE",
        }
    assert result["equilibrium_error"] < 1e-4
