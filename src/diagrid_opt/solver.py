import math

from .niti import NiTiParams, niti_update


def newton_solve(
    target_stress: float,
    z_n: float,
    eps_n: float,
    params: NiTiParams,
    eps_initial: float = 0.0,
    tolerance: float = 1e-12,
    max_iterations: int = 5,
) -> tuple[float, float, float, int]:
    """Solve residual(sigma(eps, z_n)-target) with the NiTi algorithmic tangent."""
    eps = eps_initial
    iterations = 0
    sigma = 0.0
    z = z_n
    tangent = params.E

    for iterations in range(1, max_iterations + 1):
        sigma, z, tangent = niti_update(eps, z_n, params)
        residual = sigma - target_stress
        if abs(residual) < tolerance:
            return eps, z, residual, iterations
        eps -= residual / tangent

    raise RuntimeError(
        f"Newton-Raphson did not converge in {max_iterations} iterations "
        f"(residual={residual:.3e})"
    )
