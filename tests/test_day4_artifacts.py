"""Schema and verification-gate checks for the Day 4 artifacts.

These read the committed JSON artifacts (also consumed by the web app) and assert
that their structure matches what `script.js` expects and that the SPEC Section 7
gates hold.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def _load(name: str) -> dict:
    path = DATA / name
    assert path.exists(), f"missing Day 4 artifact: {name}"
    return json.loads(path.read_text(encoding="utf-8"))


def test_gp_surrogate_schema_and_gates():
    gp = _load("gp_surrogate.json")
    assert set(gp["cases"]) == {"steel", "niti"}
    assert len(gp["features"]) == 5
    assert len(gp["targets"]) == 4
    n_train = 80
    for fuse in ("steel", "niti"):
        case = gp["cases"][fuse]
        assert len(case["X_train_standardized"]) == n_train
        assert len(case["X_train_standardized"][0]) == 5
        assert len(case["scaler"]["mean"]) == 5
        assert len(case["scaler"]["scale"]) == 5
        assert "minimum" in case["training_domain"] and "maximum" in case["training_domain"]
        for target in gp["targets"]:
            model = case["models"][target]
            assert len(model["alpha"]) == n_train
            assert len(model["cholesky"]) == n_train
            assert len(model["cholesky"][0]) == n_train
            assert len(model["length_scale"]) == 5
    rates = gp["validation"]["false_safe_rate"]
    assert rates["ucb_95"] < rates["mean"]
    assert gp["validation"]["fold_sizes"] == [20, 20, 20, 20]


def test_pareto_canopies_schema_and_certification():
    pareto = _load("pareto_canopies.json")
    designs = pareto["designs"]
    assert len(designs) == 8
    counts = {}
    for design in designs:
        counts[design["front_case"]] = counts.get(design["front_case"], 0) + 1
        assert design["nodes"] and design["elements"]
        assert set(design["kpis"]) == {"steel", "niti"}
        for fuse in ("steel", "niti"):
            q = design["quake_frames"][fuse]
            n_frames = len(q["time_s"])
            assert n_frames >= 50
            assert len(q["nodal_displacements_m_rad"]) == n_frames
            assert len(q["lead_fuse_strain"]) == n_frames
            assert len(q["lead_fuse_stress_MPa"]) == n_frames
            assert len(q["node_ids"]) == len(q["nodal_displacements_m_rad"][0])
    assert counts == {"steel": 4, "niti": 4}
    meta = pareto["meta"]
    cert = meta["certification"]
    assert cert["false_safe_violations_rejected"] == 0
    assert meta["tangent_error_relative_max"] < 1e-7
    assert meta["minimum_cyclic_dissipation_MPa"] >= -1e-9


def test_diff_id_schema_and_eight_percent_gate():
    diff = _load("diff_id_uncertainty.json")
    assert diff["meta"]["all_parameters_within_8_percent"] is True
    for name, error in diff["relative_recovery_error_pct"].items():
        assert error < 8.0, f"{name} exceeds the 8% recovery gate: {error:.2f}%"
    assert set(diff["confidence_intervals_95"]) == set(diff["recovered_parameters"])
    for name, interval in diff["confidence_intervals_95"].items():
        assert interval[0] <= interval[1], f"{name} has an inverted 95% CI"
    loops = diff["loops"]
    sizes = {len(loops[k]) for k in ("strain", "true_stress_MPa", "observed_stress_MPa", "recovered_stress_MPa")}
    assert len(sizes) == 1
