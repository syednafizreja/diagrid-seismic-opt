from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from .constitutive_sma import NiTiParams, niti_update


PARAMETER_NAMES = (
    "E",
    "sigma_Ms",
    "sigma_Mf",
    "sigma_As",
    "sigma_Af",
    "eps_L",
)
PARAMETER_BOUNDS = (
    (20_000.0, 60_000.0),
    (250.0, 500.0),
    (20.0, 150.0),
    (50.0, 200.0),
    (20.0, 150.0),
    (0.03, 0.09),
)
INITIAL_INTERNAL_PARAMETERS = (
    38_000.0,
    360.0,
    80.0,
    120.0,
    80.0,
    0.058,
)
BOOTSTRAP_MEMBERS = 20
RANDOM_SEED = 42
OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "diff_id_uncertainty.json"
)


def _decode_parameters(raw: torch.Tensor) -> torch.Tensor:
    bounds = torch.as_tensor(
        PARAMETER_BOUNDS, dtype=raw.dtype, device=raw.device
    )
    unconstrained = bounds[:, 0] + (
        bounds[:, 1] - bounds[:, 0]
    ) * torch.sigmoid(raw)
    modulus, martensite_start, forward_width, austenite_finish, reverse_width, eps_l = (
        unconstrained.unbind(dim=-1)
    )
    martensite_finish = martensite_start + forward_width
    austenite_start = austenite_finish + reverse_width
    return torch.stack(
        (
            modulus,
            martensite_start,
            martensite_finish,
            austenite_start,
            austenite_finish,
            eps_l,
        ),
        dim=-1,
    )


def _encode_initial_parameters(
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    initial = torch.as_tensor(INITIAL_INTERNAL_PARAMETERS, dtype=dtype)
    bounds = torch.as_tensor(PARAMETER_BOUNDS, dtype=dtype)
    fraction = (initial - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return torch.logit(fraction).unsqueeze(0)


def smooth_stress_history(
    strain: torch.Tensor,
    raw_parameters: torch.Tensor,
    transition_mpa: float = 0.5,
) -> torch.Tensor:
    """Differentiable, signed-martensite approximation of the Day 1 update."""
    if strain.ndim != 1:
        raise ValueError("strain must be a one-dimensional history")
    if raw_parameters.ndim == 1:
        raw_parameters = raw_parameters.unsqueeze(0)
    if raw_parameters.ndim != 2 or raw_parameters.shape[1] != 6:
        raise ValueError("raw_parameters must have shape (batch, 6)")
    if transition_mpa <= 0.0:
        raise ValueError("transition_mpa must be positive")

    physical = _decode_parameters(raw_parameters)
    modulus, sigma_ms, sigma_mf, sigma_as, sigma_af, eps_l = (
        physical.unbind(dim=-1)
    )
    forward_width = sigma_mf - sigma_ms
    reverse_width = sigma_as - sigma_af
    z = torch.zeros_like(modulus)
    outputs = []
    martensite_smoothing = 1e-4
    tiny = 1e-12

    for strain_value in strain.unbind():
        sigma_trial = modulus * (strain_value - eps_l * z)
        sign_z = torch.tanh(z / martensite_smoothing)
        abs_z = torch.sqrt(z.square() + tiny)

        reverse_drive = sigma_af + reverse_width * abs_z - sign_z * sigma_trial
        reverse_gate = torch.sigmoid(reverse_drive / transition_mpa) * torch.sigmoid(
            abs_z / martensite_smoothing
        )
        reverse_increment = (
            functional.softplus(reverse_drive / transition_mpa)
            * transition_mpa
            / (modulus * eps_l + reverse_width)
        )
        reverse_amount = abs_z * torch.tanh(
            reverse_gate * reverse_increment / (abs_z + tiny)
        )
        z_reversed = z - sign_z * reverse_amount

        sigma_reversed = modulus * (strain_value - eps_l * z_reversed)
        abs_z_reversed = torch.sqrt(z_reversed.square() + tiny)
        sign_stress = torch.tanh(
            sigma_reversed / (2.0 * transition_mpa)
        )
        abs_stress = torch.sqrt(sigma_reversed.square() + tiny)
        forward_drive = (
            abs_stress - sigma_ms - forward_width * abs_z_reversed
        )
        zero_martensite_gate = torch.exp(
            -abs_z_reversed / (10.0 * martensite_smoothing)
        )
        same_direction_gate = torch.sigmoid(
            (
                sign_stress
                * torch.tanh(z_reversed / martensite_smoothing)
                - 0.25
            )
            / 0.1
        )
        direction_gate = zero_martensite_gate + (
            1.0 - zero_martensite_gate
        ) * same_direction_gate
        forward_increment = (
            functional.softplus(forward_drive / transition_mpa)
            * transition_mpa
            / (modulus * eps_l + forward_width)
            * direction_gate
        )
        available = 1.0 - abs_z_reversed
        forward_amount = available * torch.tanh(
            forward_increment / (available + tiny)
        )
        z = z_reversed + sign_stress * forward_amount
        outputs.append(modulus * (strain_value - eps_l * z))

    return torch.stack(outputs, dim=-1)


def _exact_stress_history(
    strain: np.ndarray, params: NiTiParams
) -> np.ndarray:
    z = 0.0
    stress_history = []
    for strain_value in strain:
        stress, z, _ = niti_update(float(strain_value), z, params)
        stress_history.append(stress)
    return np.asarray(stress_history, dtype=float)


def _fit_curves(
    strain: np.ndarray,
    observations: np.ndarray,
    *,
    adam_steps: int = 300,
    lbfgs_steps: int = 50,
) -> np.ndarray:
    strain_tensor = torch.as_tensor(strain, dtype=torch.float64)
    target_tensor = torch.as_tensor(observations, dtype=torch.float64)
    raw_parameters = torch.nn.Parameter(
        _encode_initial_parameters().repeat(observations.shape[0], 1)
    )
    stress_scale = max(float(np.max(np.abs(observations))), 1.0)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def loss_value() -> torch.Tensor:
        predicted = smooth_stress_history(strain_tensor, raw_parameters)
        return torch.mean(((predicted - target_tensor) / stress_scale).square())

    try:
        adam = torch.optim.Adam([raw_parameters], lr=0.04)
        for _ in range(adam_steps):
            adam.zero_grad()
            loss = loss_value()
            loss.backward()
            adam.step()
            with torch.no_grad():
                raw_parameters.clamp_(-8.0, 8.0)

        lbfgs = torch.optim.LBFGS(
            [raw_parameters],
            max_iter=lbfgs_steps,
            history_size=20,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss = loss_value()
            loss.backward()
            return loss

        lbfgs.step(closure)
        with torch.no_grad():
            return _decode_parameters(raw_parameters).cpu().numpy()
    finally:
        torch.set_num_threads(previous_threads)


def _synthetic_data(seed: int) -> dict[str, Any]:
    params = NiTiParams()
    # Ramped multi-amplitude nested cycles (+/-1.5% -> +/-6%). A single constant
    # amplitude leaves the reverse-transformation parameters and eps_L weakly
    # identifiable; nested loops pin every parameter inside the 8% gate.
    phase = np.linspace(0.0, 1.0, 601)
    strain = 0.06 * (0.25 + 0.75 * phase) * np.sin(2.0 * np.pi * 5.0 * phase)
    true_stress = _exact_stress_history(strain, params)
    noise_sigma = 0.03 * float(np.max(np.abs(true_stress)))
    rng = np.random.default_rng(seed)
    observed = true_stress + rng.normal(0.0, noise_sigma, size=true_stress.shape)
    return {
        "strain": strain,
        "true_stress": true_stress,
        "observed_stress": observed,
        "noise_sigma": noise_sigma,
        "rng": rng,
    }


def run_parameter_identification(
    output_path: str | Path = OUTPUT_PATH,
    *,
    adam_steps: int = 300,
    lbfgs_steps: int = 80,
    bootstrap_members: int = BOOTSTRAP_MEMBERS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    if bootstrap_members < 1:
        raise ValueError("bootstrap_members must be positive")
    synthetic = _synthetic_data(seed)
    strain = synthetic["strain"]
    true_stress = synthetic["true_stress"]
    observed = synthetic["observed_stress"]
    residuals = observed - true_stress
    rng = synthetic["rng"]
    bootstrap_curves = np.stack(
        [
            true_stress
            + rng.choice(residuals, size=residuals.size, replace=True)
            for _ in range(bootstrap_members)
        ],
        axis=0,
    )
    fit_curves = np.concatenate((observed[np.newaxis, :], bootstrap_curves))
    fitted = _fit_curves(
        strain,
        fit_curves,
        adam_steps=adam_steps,
        lbfgs_steps=lbfgs_steps,
    )
    true_params = asdict(NiTiParams())
    recovered = dict(zip(PARAMETER_NAMES, fitted[0].tolist()))
    relative_error = {
        name: abs(recovered[name] / true_params[name] - 1.0) * 100.0
        for name in PARAMETER_NAMES
    }
    bootstrap_values = fitted[1:]
    confidence_intervals = {
        name: np.percentile(
            bootstrap_values[:, index], (2.5, 97.5)
        ).tolist()
        for index, name in enumerate(PARAMETER_NAMES)
    }
    recovered_stress = _exact_stress_history(
        strain,
        NiTiParams(**recovered),
    )
    payload = {
        "meta": {
            "seed": seed,
            "sensor_noise_fraction": 0.03,
            "sensor_noise_sigma_MPa": synthetic["noise_sigma"],
            "bootstrap_method": "resample noisy-curve residuals with replacement",
            "bootstrap_members": bootstrap_members,
            "adam_steps": adam_steps,
            "lbfgs_max_iterations": lbfgs_steps,
            "all_parameters_within_8_percent": all(
                error < 8.0 for error in relative_error.values()
            ),
        },
        "true_parameters": true_params,
        "recovered_parameters": recovered,
        "relative_recovery_error_pct": relative_error,
        "confidence_intervals_95": confidence_intervals,
        "loops": {
            "strain": strain.tolist(),
            "true_stress_MPa": true_stress.tolist(),
            "observed_stress_MPa": observed.tolist(),
            "recovered_stress_MPa": recovered_stress.tolist(),
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    result = run_parameter_identification()
    print("NiTi differentiable parameter identification")
    for name in PARAMETER_NAMES:
        print(
            f"{name:<12} true={result['true_parameters'][name]:>10.4f} "
            f"recovered={result['recovered_parameters'][name]:>10.4f} "
            f"relative error={result['relative_recovery_error_pct'][name]:>7.3f}% "
            f"95% CI={result['confidence_intervals_95'][name]}"
        )
    print(f"Saved {OUTPUT_PATH}")
    if not result["meta"]["all_parameters_within_8_percent"]:
        print("FAIL: at least one recovered parameter exceeds the 8% gate")


if __name__ == "__main__":
    main()
