import numpy as np

from diagrid_opt.constitutive_sma import (
    NiTiParams,
    newton_solve,
    niti_update,
)
from diagrid_opt.thermodynamics import (
    discrete_dissipation,
)


def test_algorithmic_tangent_matches_central_difference():
    params = NiTiParams()
    h = 1e-7
    for eps, z_n in ((0.001, 0.0), (-0.001, 0.0), (0.025, 0.2), (-0.025, -0.2)):
        _, _, E_alg = niti_update(eps, z_n, params)
        sigma_plus, _, _ = niti_update(eps + h, z_n, params)
        sigma_minus, _, _ = niti_update(eps - h, z_n, params)
        numerical = (sigma_plus - sigma_minus) / (2.0 * h)
        assert abs((E_alg - numerical) / numerical) < 1e-7


def test_newton_equilibrium_converges_quadratically():
    params = NiTiParams()
    eps, z, residual, iterations = newton_solve(
        target_stress=500.0,
        z_n=0.0,
        params=params,
    )
    assert abs(residual) < 1e-12
    assert iterations <= 5
    assert -1.0 <= z <= 1.0


def test_clausius_duhem_dissipation_across_five_reverse_loops():
    params = NiTiParams()
    eps_n = 0.0
    z_n = 0.0
    sigma_n = 0.0

    cycles = 5
    samples_per_cycle = 1000
    phase = np.linspace(0.0, cycles * 2.0 * np.pi, cycles * samples_per_cycle)
    strain_history = 0.05 * np.sin(phase)

    minimum_dissipation = float("inf")
    for eps_new in strain_history:
        sigma_new, z_new, _ = niti_update(float(eps_new), z_n, params)
        dissipation = discrete_dissipation(
            eps_n,
            float(eps_new),
            z_n,
            z_new,
            sigma_n,
            sigma_new,
            params,
        )
        minimum_dissipation = min(minimum_dissipation, dissipation)
        eps_n, z_n, sigma_n = float(eps_new), z_new, sigma_new

    assert minimum_dissipation >= -1e-9
