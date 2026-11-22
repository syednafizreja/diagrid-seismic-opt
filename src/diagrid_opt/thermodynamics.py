from .constitutive_sma import NiTiParams


def helmholtz_free_energy(
    eps: float,
    z: float,
    params: NiTiParams,
) -> float:
    sigma_0 = 0.5 * (params.sigma_Ms + params.sigma_Af)
    H_eq = 0.5 * (params.H_f + params.H_r)
    return (
        0.5 * params.E * (eps - params.eps_L * z) ** 2
        + params.eps_L
        * (sigma_0 * abs(z) + 0.5 * H_eq * z**2)
    )


def discrete_dissipation(
    eps_n: float,
    eps_new: float,
    z_n: float,
    z_new: float,
    sigma_n: float,
    sigma_new: float,
    params: NiTiParams,
) -> float:
    psi_n = helmholtz_free_energy(eps_n, z_n, params)
    psi_new = helmholtz_free_energy(eps_new, z_new, params)
    work = 0.5 * (sigma_n + sigma_new) * (eps_new - eps_n)
    return work - (psi_new - psi_n)
