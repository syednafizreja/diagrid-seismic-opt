import numpy as np

from diagrid_opt.optimize_nsga2 import CanopyOptimizationProblem


class FixedSurrogate:
    def predict(self, fuse_type, features):
        assert fuse_type == "steel"
        assert features.shape == (5,)
        return {
            "peak_drift_pct": (np.array([1.0]), np.array([0.1])),
            "grav_defl_mm": (np.array([10.0]), np.array([0.5])),
            "c_total_per_m2": (np.array([200.0]), np.array([2.0])),
            "abs_residual_drift_pct": (np.array([0.1]), np.array([0.02])),
        }


def test_nsga_problem_rounds_divisions_and_uses_conservative_constraints():
    problem = CanopyOptimizationProblem("steel", FixedSurrogate())
    design = problem.decode(np.array([20.0, 0.15, 8.6, 0.35, 150.0]))
    output = {}
    problem._evaluate(
        np.array([20.0, 0.15, 8.6, 0.35, 150.0]), output
    )

    # Nearest allowed even division to 8.6 is 8 (|8.6-8|=0.6 < |8.6-10|=1.4).
    assert design.diagrid_divisions == 8
    np.testing.assert_allclose(output["F"], [200.0, -20.0])
    np.testing.assert_allclose(
        output["G"],
        [
            1.0 + 1.96 * 0.1 - 2.0,
            10.0 + 1.96 * 0.5 - 20.0 * 1000.0 / 250.0,
        ],
    )
