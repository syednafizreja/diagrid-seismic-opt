import numpy as np

from diagrid_opt.niti import NiTiParams, helmholtz_free_energy, niti_update


def test_alg_tangent_matches_central_difference_on_smooth_branches():
    params = NiTiParams()
    cases = [
        (0.001, 0.0),
        (-0.001, 0.0),
        (0.025, 0.2),
        (-0.025, -0.2),
    ]
    h = 1e-7
    for eps, z_n in cases:
        _, _, E_alg = niti_update(eps, z_n, params)
        sigma_plus, _, _ = niti_update(eps + h, z_n, params)
        sigma_minus, _, _ = niti_update(eps - h, z_n, params)
        numerical = (sigma_plus - sigma_minus) / (2.0 * h)
        relative_error = abs((E_alg - numerical) / numerical)
        assert relative_error < 1e-7


def test_cyclic_dissipation():
    params = NiTiParams()
    eps_n = 0.0
    z_n = 0.0
    sigma_n = 0.0
    psi_n = helmholtz_free_energy(eps_n, z_n, params)
    minimum = float("inf")

    strain = [0.05 * np.sin(t) for t in np.linspace(0.0, 20.0 * np.pi, 5001)]
    for eps_new in strain:
        sigma_new, z_new, _ = niti_update(float(eps_new), z_n, params)
        psi_new = helmholtz_free_energy(float(eps_new), z_new, params)
        dissipation = (
            0.5 * (sigma_n + sigma_new) * (eps_new - eps_n)
            - (psi_new - psi_n)
        )
        minimum = min(minimum, float(dissipation))
        eps_n, z_n, sigma_n, psi_n = eps_new, z_new, sigma_new, psi_new

    assert minimum >= -1e-9
