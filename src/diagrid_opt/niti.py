from dataclasses import dataclass


@dataclass(frozen=True)
class NiTiParams:
    E: float = 40_000.0
    eps_L: float = 0.06
    sigma_Ms: float = 380.0
    sigma_Mf: float = 450.0
    sigma_As: float = 180.0
    sigma_Af: float = 110.0

    @property
    def H_f(self) -> float:
        return self.sigma_Mf - self.sigma_Ms

    @property
    def H_r(self) -> float:
        return self.sigma_As - self.sigma_Af


def niti_update(
    eps_new: float,
    z_n: float,
    params: NiTiParams,
) -> tuple[float, float, float]:
    """Rate-independent 1D superelastic NiTi return mapping."""
    if not -1.0 <= z_n <= 1.0:
        raise ValueError("z_n must lie in [-1, 1]")

    sigma_tr = params.E * (eps_new - params.eps_L * z_n)
    s_tr = 1.0 if sigma_tr >= 0.0 else -1.0
    s_n = 1.0 if z_n >= 0.0 else -1.0

    if abs(z_n) > 0.0:
        xi_n = abs(z_n)
        sigma_r_xi = params.sigma_Af + params.H_r * xi_n
        if s_n * sigma_tr < sigma_r_xi:
            dxi_rev = (
                sigma_r_xi - s_n * sigma_tr
            ) / (params.E * params.eps_L + params.H_r)
            if dxi_rev < xi_n:
                z_new = s_n * (xi_n - dxi_rev)
                sigma_new = params.E * (eps_new - params.eps_L * z_new)
                E_alg = (params.E * params.H_r) / (
                    params.E * params.eps_L + params.H_r
                )
                return sigma_new, z_new, E_alg

            z_n = 0.0
            sigma_tr = params.E * eps_new
            s_tr = 1.0 if sigma_tr >= 0.0 else -1.0
            s_n = 1.0

    xi_n = abs(z_n)
    sigma_f_xi = params.sigma_Ms + params.H_f * xi_n
    if (
        xi_n < 1.0
        and abs(sigma_tr) > sigma_f_xi
        and (z_n == 0.0 or s_tr == s_n)
    ):
        dxi_fwd = (
            abs(sigma_tr) - sigma_f_xi
        ) / (params.E * params.eps_L + params.H_f)
        if xi_n + dxi_fwd < 1.0:
            z_new = s_tr * (xi_n + dxi_fwd)
            sigma_new = params.E * (eps_new - params.eps_L * z_new)
            E_alg = (params.E * params.H_f) / (
                params.E * params.eps_L + params.H_f
            )
        else:
            z_new = s_tr
            sigma_new = params.E * (eps_new - params.eps_L * z_new)
            E_alg = params.E
        return sigma_new, z_new, E_alg

    z_new = z_n
    return sigma_tr, z_new, params.E


def helmholtz_free_energy(eps: float, z: float, params: NiTiParams) -> float:
    sigma_0 = 0.5 * (params.sigma_Ms + params.sigma_Af)
    H_eq = 0.5 * (params.H_f + params.H_r)
    return (
        0.5 * params.E * (eps - params.eps_L * z) ** 2
        + params.eps_L
        * (sigma_0 * abs(z) + 0.5 * H_eq * z**2)
    )
