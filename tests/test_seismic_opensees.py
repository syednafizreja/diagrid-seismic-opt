import numpy as np
import pytest

from diagrid_opt.ground_motions import generate_ground_motion
from diagrid_opt.geometry import GeometryParams, canopy_geometry
from diagrid_opt import solver_opensees
from diagrid_opt.solver_opensees import run_seismic_analysis


def _baseline_geometry():
    return canopy_geometry(
        GeometryParams(
            span=20.0,
            arch_rise=3.0,
            diagrid_divisions=8,
            branch_spread=3.2,
            glulam_depth=0.35,
        )
    )


def test_ground_motions_are_repeatable_pga_scaled_and_have_free_vibration_tail():
    first = generate_ground_motion(seed=101, pga_g=0.40)
    repeat = generate_ground_motion(seed=101, pga_g=0.40)
    other = generate_ground_motion(seed=202, pga_g=0.40)

    assert first.dt_s == 0.01
    assert first.time_s[-1] == 18.0
    assert np.isclose(np.max(np.abs(first.acceleration_m_s2)), 0.40 * 9.80665)
    assert np.array_equal(first.acceleration_m_s2, repeat.acceleration_m_s2)
    assert not np.array_equal(first.acceleration_m_s2, other.acceleration_m_s2)
    assert np.count_nonzero(first.acceleration_m_s2[1200:]) == 0
    assert first.acceleration_m_s2.size == 1801


def test_all_040g_records_converge_and_niti_reduces_mean_residual_drift():
    geometry = _baseline_geometry()
    residuals = {"steel": [], "niti": []}

    for seed in (101, 202, 303):
        for fuse_type in ("steel", "niti"):
            result = run_seismic_analysis(
                geometry,
                fuse_type=fuse_type,
                pga_g=0.40,
                seed=seed,
            )
            residuals[fuse_type].append(
                result["permanent_residual_roof_drift_ratio"]
            )
            assert result["converged_steps"] == 1800
            assert np.isfinite(result["peak_roof_drift_ratio"])
            assert np.isfinite(result["permanent_residual_roof_drift_ratio"])
            assert np.isfinite(result["peak_base_shear_N"])
            assert result["peak_roof_drift_ratio"] > 0.0
            assert result["peak_base_shear_N"] > 0.0

    assert np.mean(residuals["niti"]) < np.mean(residuals["steel"])


def test_record_frames_returns_downsampled_response_histories():
    result = run_seismic_analysis(
        _baseline_geometry(),
        fuse_type="niti",
        pga_g=0.25,
        seed=101,
        record_frames=True,
    )
    history = result["history"]

    assert 1 < len(history["time_s"]) < result["converged_steps"]
    assert history["nodal_displacements_m_rad"].shape == (
        len(history["time_s"]),
        len(history["node_ids"]),
        6,
    )
    assert history["lead_fuse_strain"].shape == history["time_s"].shape
    assert history["lead_fuse_stress_Pa"].shape == history["time_s"].shape
    assert history["signed_roof_drift_ratio"].shape == history["time_s"].shape
    assert result["max_fuse_strain"] >= np.max(
        np.abs(history["lead_fuse_strain"])
    )
    assert abs(result["signed_permanent_residual_roof_drift_ratio"]) == pytest.approx(
        result["permanent_residual_roof_drift_ratio"]
    )


def test_040g_calibration_yields_steel_and_recenters_niti():
    geometry = _baseline_geometry()
    steel = run_seismic_analysis(
        geometry, fuse_type="steel", pga_g=0.40, seed=101, record_frames=True
    )
    niti = run_seismic_analysis(
        geometry, fuse_type="niti", pga_g=0.40, seed=101, record_frames=True
    )

    steel_peak_stress = np.max(
        np.abs(steel["history"]["lead_fuse_stress_Pa"])
    )
    niti_peak_stress = np.max(
        np.abs(niti["history"]["lead_fuse_stress_Pa"])
    )
    assert steel_peak_stress > 355.0e6
    assert niti_peak_stress > 380.0e6
    assert (
        steel["permanent_residual_roof_drift_ratio"]
        > niti["permanent_residual_roof_drift_ratio"]
    )
    assert niti["permanent_residual_roof_drift_ratio"] < 0.0008


def test_transient_step_uses_newton_linesearch_and_dt_quarter_fallback(monkeypatch):
    algorithms = []
    steps = []
    codes = iter((-1, -1, 0, 0, 0, 0))
    monkeypatch.setattr(solver_opensees.ops, "test", lambda *args: None)
    monkeypatch.setattr(
        solver_opensees.ops,
        "algorithm",
        lambda name: algorithms.append(name),
    )
    monkeypatch.setattr(
        solver_opensees.ops,
        "analyze",
        lambda count, dt: steps.append(dt) or next(codes),
    )

    assert solver_opensees._advance_transient_step(0.01)
    assert algorithms == ["Newton", "NewtonLineSearch", "ModifiedNewton"]
    assert steps == [0.01, 0.01, 0.0025, 0.0025, 0.0025, 0.0025]
