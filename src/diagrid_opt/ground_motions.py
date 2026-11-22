from dataclasses import dataclass

import numpy as np


DT_SECONDS = 0.01
STRONG_SHAKING_SECONDS = 12.0
FREE_VIBRATION_SECONDS = 6.0
GRAVITY_M_S2 = 9.80665


@dataclass(frozen=True)
class GroundMotion:
    seed: int
    pga_g: float
    dt_s: float
    time_s: np.ndarray
    acceleration_m_s2: np.ndarray


def generate_ground_motion(
    seed: int,
    pga_g: float,
    dt_s: float = DT_SECONDS,
) -> GroundMotion:
    """Create a deterministic broadband synthetic record with a zero tail."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not np.isfinite(pga_g) or pga_g <= 0.0:
        raise ValueError("pga_g must be finite and positive")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    n_strong = int(round(STRONG_SHAKING_SECONDS / dt_s))
    n_tail = int(round(FREE_VIBRATION_SECONDS / dt_s))
    strong_time = np.arange(n_strong, dtype=float) * dt_s
    rng = np.random.default_rng(seed)
    frequencies_hz = np.geomspace(0.35, 18.0, 48)
    phases = rng.uniform(0.0, 2.0 * np.pi, frequencies_hz.size)
    amplitudes = 1.0 / np.sqrt(frequencies_hz)
    signal = np.sum(
        amplitudes[:, None]
        * np.sin(
            2.0 * np.pi * frequencies_hz[:, None] * strong_time[None, :]
            + phases[:, None]
        ),
        axis=0,
    )

    rise = 1.0 - np.exp(-strong_time / 0.65)
    decay = np.ones_like(strong_time)
    decay_start = 9.0
    decaying = strong_time > decay_start
    decay[decaying] = np.exp(-0.55 * (strong_time[decaying] - decay_start))
    envelope = rise * decay
    shaped = signal * envelope
    peak = float(np.max(np.abs(shaped)))
    if peak == 0.0:
        raise RuntimeError("synthetic ground motion has zero peak amplitude")

    acceleration_strong = shaped * (pga_g * GRAVITY_M_S2 / peak)
    acceleration = np.concatenate(
        (acceleration_strong, np.zeros(n_tail + 1, dtype=float))
    )
    time = np.arange(acceleration.size, dtype=float) * dt_s
    return GroundMotion(
        seed=seed,
        pga_g=float(pga_g),
        dt_s=float(dt_s),
        time_s=time,
        acceleration_m_s2=acceleration,
    )
