from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from diagrid_opt.constitutive_sma import NiTiParams, niti_update
from diagrid_opt.geometry import GeometryParams, canopy_geometry
from diagrid_opt.thermodynamics import discrete_dissipation


def plot_niti(output: Path) -> None:
    params = NiTiParams()
    cycles = 5
    samples = 1000
    phase = np.linspace(0.0, cycles * 2.0 * np.pi, cycles * samples)
    strain = 0.05 * np.sin(phase)

    eps_n = 0.0
    z_n = 0.0
    sigma_n = 0.0
    stress = []
    dissipation = []
    cumulative = 0.0
    for eps_new in strain:
        sigma_new, z_new, _ = niti_update(float(eps_new), z_n, params)
        step = discrete_dissipation(
            eps_n, float(eps_new), z_n, z_new,
            sigma_n, sigma_new, params,
        )
        cumulative += max(step, 0.0)
        stress.append(sigma_new)
        dissipation.append(cumulative)
        eps_n, z_n, sigma_n = float(eps_new), z_new, sigma_new

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(strain * 100.0, stress, color="#176B87", linewidth=1.2)
    axes[0].set(xlabel="Strain (%)", ylabel="Stress (MPa)", title="NiTi Hysteresis")
    axes[0].grid(alpha=0.25)

    axes[1].plot(np.arange(len(dissipation)), dissipation, color="#C44E52", linewidth=1.4)
    axes[1].set(
        xlabel="Increment",
        ylabel="Accumulated dissipation (MPa)",
        title="Clausius-Duhem Dissipation",
    )
    axes[1].grid(alpha=0.25)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_geometry(output: Path) -> None:
    geometry = canopy_geometry(
        GeometryParams(
            span=20.0,
            arch_rise=3.0,
            diagrid_divisions=8,
            branch_spread=3.2,
            glulam_depth=0.35,
        )
    )
    colors = {
        "GLULAM_DIAGRID": "#5B8FF9",
        "STEEL_EDGE": "#FFB000",
        "TREE_TRUNK": "#4C956C",
        "TREE_BRANCH": "#A259A4",
        "SEISMIC_FUSE": "#E05252",
    }

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for tag, pairs in geometry.elements.items():
        for a, b in pairs:
            p = geometry.nodes[a]
            q = geometry.nodes[b]
            ax.plot(
                [p[0], q[0]],
                [p[1], q[1]],
                [p[2], q[2]],
                color=colors[tag],
                linewidth=1.3,
                alpha=0.9,
            )
    for tag, color in colors.items():
        ax.plot([], [], [], color=color, linewidth=2.0, label=tag)
    ax.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title="Baseline 20 m Diagrid Canopy",
    )
    ax.view_init(elev=24, azim=-52)
    ax.legend(loc="upper left")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    figures = Path("figures")
    figures.mkdir(parents=True, exist_ok=True)
    plot_niti(figures / "day1_niti_constitutive.png")
    plot_geometry(figures / "day1_diagrid_3d.png")


if __name__ == "__main__":
    main()
