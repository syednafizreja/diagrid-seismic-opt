import numpy as np
import pytest

from diagrid_opt.carbon_lca import assess_lifecycle_carbon
from diagrid_opt.geometry import GeometryParams, canopy_geometry


@pytest.fixture
def geometry():
    return canopy_geometry(
        GeometryParams(
            span=20.0,
            arch_rise=3.0,
            diagrid_divisions=8,
            branch_spread=3.4,
            glulam_depth=0.35,
            niti_fuse_area=100.0,
        )
    )


def metrics(peak_drift=0.0, residual_drift=0.0, max_fuse_strain=0.0):
    return {
        "peak_roof_drift_ratio": peak_drift,
        "permanent_residual_roof_drift_ratio": residual_drift,
        "max_fuse_strain": max_fuse_strain,
    }


def fuse_material_carbon(geometry, fuse_type):
    fuse_length = sum(
        np.linalg.norm(geometry.nodes[a] - geometry.nodes[b])
        for a, b in geometry.elements["SEISMIC_FUSE"]
    )
    area = geometry.niti_fuse_area * 1e-6
    density, factor = (6450.0, 28.0) if fuse_type == "niti" else (7850.0, 1.55)
    if fuse_type == "steel":
        area *= 380.0 / 355.0
    return fuse_length * area * density * factor


@pytest.mark.parametrize("fuse_type", ["steel", "niti"])
def test_fuse_replacement_uses_strict_strain_threshold(geometry, fuse_type):
    threshold = 0.00355 if fuse_type == "steel" else 0.06
    at_threshold = assess_lifecycle_carbon(
        geometry, fuse_type, metrics(max_fuse_strain=threshold)
    )
    beyond_threshold = assess_lifecycle_carbon(
        geometry, fuse_type, metrics(max_fuse_strain=threshold + 1e-8)
    )

    assert at_threshold.fuse_replacement_kgco2e == 0.0
    assert beyond_threshold.fuse_replacement_kgco2e == pytest.approx(
        1.5 * fuse_material_carbon(geometry, fuse_type)
    )


def test_glazing_damage_boundaries_and_residual_repair(geometry):
    no_damage = assess_lifecycle_carbon(geometry, "niti", metrics(peak_drift=0.005))
    at_upper_boundary = assess_lifecycle_carbon(
        geometry, "niti", metrics(peak_drift=0.025, residual_drift=0.0025)
    )
    above_upper_boundary = assess_lifecycle_carbon(
        geometry, "niti", metrics(peak_drift=0.025001)
    )

    assert no_damage.glazing_repair_kgco2e == 0.0
    assert at_upper_boundary.glazing_repair_kgco2e == pytest.approx(
        45.0 * geometry.shaded_area_m2 * 0.35
    )
    assert above_upper_boundary.glazing_repair_kgco2e == pytest.approx(
        45.0 * geometry.shaded_area_m2 * 0.60
    )
    assert at_upper_boundary.residual_repair_kgco2e == pytest.approx(
        0.5 * 0.40 * (at_upper_boundary.c_a1_a3_kgco2e - 1612.8)
    )


def test_biogenic_sensitivity_changes_only_glulam_factor(geometry):
    fossil = assess_lifecycle_carbon(geometry, "niti", metrics())
    biogenic = assess_lifecycle_carbon(
        geometry, "niti", metrics(), include_biogenic=True
    )
    glulam_length = sum(
        np.linalg.norm(geometry.nodes[a] - geometry.nodes[b])
        for a, b in geometry.elements["GLULAM_DIAGRID"]
    )
    glulam_mass = glulam_length * geometry.b_glulam * geometry.glulam_depth * 420.0

    assert biogenic.c_a1_a3_kgco2e == pytest.approx(
        fossil.c_a1_a3_kgco2e + glulam_mass * (-0.98 - 0.512)
    )
    assert biogenic.total_intensity_kgco2e_m2 == pytest.approx(
        biogenic.total_kgco2e / geometry.shaded_area_m2
    )


@pytest.mark.parametrize("fuse_type", ["steel", "niti"])
def test_a1_a3_uses_physical_section_areas(geometry, fuse_type):
    result = assess_lifecycle_carbon(geometry, fuse_type, metrics())
    lengths = {
        group: sum(
            np.linalg.norm(geometry.nodes[a] - geometry.nodes[b])
            for a, b in geometry.elements[group]
        )
        for group in geometry.elements
    }
    steel_volume = (
        lengths["STEEL_EDGE"] * 9.49e-3
        + lengths["TREE_TRUNK"] * 1.22e-2
        + lengths["TREE_BRANCH"] * 4.67e-3
    )
    glulam_volume = (
        lengths["GLULAM_DIAGRID"] * geometry.b_glulam * geometry.glulam_depth
    )
    expected = (
        1612.8
        + steel_volume * 7850.0 * 1.55
        + glulam_volume * 420.0 * 0.512
        + fuse_material_carbon(geometry, fuse_type)
    )

    assert result.c_a1_a3_kgco2e == pytest.approx(expected)


def test_invalid_seismic_metrics_are_rejected(geometry):
    with pytest.raises(ValueError, match="non-negative"):
        assess_lifecycle_carbon(
            geometry, "niti", metrics(peak_drift=-0.1)
        )
