import torch

from diagrid_opt.diff_ident_sciml import smooth_stress_history


def test_smooth_superelastic_history_has_finite_parameter_gradients():
    strain = 0.045 * torch.sin(
        torch.linspace(0.0, 4.0 * torch.pi, 81, dtype=torch.float64)
    )
    raw_parameters = torch.zeros((1, 6), dtype=torch.float64, requires_grad=True)

    stress = smooth_stress_history(strain, raw_parameters)
    gradient = torch.autograd.grad(stress.square().mean(), raw_parameters)[0]

    assert stress.shape == (1, strain.numel())
    assert torch.isfinite(stress).all()
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 6
