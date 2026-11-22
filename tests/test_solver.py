from diagrid_opt.niti import NiTiParams
from diagrid_opt.solver import newton_solve


def test_newton_converges_in_five_iterations():
    params = NiTiParams()
    eps, z, residual, iterations = newton_solve(
        target_stress=500.0,
        z_n=0.0,
        eps_n=0.0,
        params=params,
    )
    assert abs(residual) < 1e-12
    assert iterations <= 5
