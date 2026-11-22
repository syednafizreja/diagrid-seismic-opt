from dataclasses import dataclass
from typing import Tuple


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
) -> Tuple[float, float, float]:
    """Exact signed-martensite fraction return mapping."""
    if not -1.0 <= z_n <= 1.0:
        raise ValueError("z_n must lie in [-1, 1]")

    sigma_tr = params.E * (eps_new - params.eps_L * z_n)
    s_tr = 1.0 if sigma_tr >= 0.0 else -1.0
    s_n = 1.0 if z_n >= 0.0 else -1.0

    if abs(z_n) > 0.0:
        xi_n = abs(z_n)
        sigma_r = params.sigma_Af + params.H_r * xi_n
        if s_n * sigma_tr < sigma_r:
            dxi_rev = (sigma_r - s_n * sigma_tr) / (
                params.E * params.eps_L + params.H_r
            )
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
    sigma_f = params.sigma_Ms + params.H_f * xi_n
    if (
        xi_n < 1.0
        and abs(sigma_tr) > sigma_f
        and (z_n == 0.0 or s_tr == s_n)
    ):
        dxi_fwd = (abs(sigma_tr) - sigma_f) / (
            params.E * params.eps_L + params.H_f
        )
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

    return sigma_tr, z_n, params.E


def newton_solve(
    target_stress: float,
    z_n: float,
    params: NiTiParams,
    eps_initial: float = 0.0,
    tolerance: float = 1e-12,
    max_iterations: int = 5,
) -> Tuple[float, float, float, int]:
    """Solve stress equilibrium with the NiTi algorithmic tangent."""
    eps = eps_initial
    for iterations in range(1, max_iterations + 1):
        sigma, z_new, E_alg = niti_update(eps, z_n, params)
        residual = sigma - target_stress
        if abs(residual) < tolerance:
            return eps, z_new, residual, iterations
        eps -= residual / E_alg

    raise RuntimeError(
        f"Newton-Raphson did not converge in {max_iterations} iterations "
        f"(residual={residual:.3e})"
    )
