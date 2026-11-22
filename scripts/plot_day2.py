from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from diagrid_opt.geometry import GeometryParams, canopy_geometry
from diagrid_opt.solver_opensees import run_seismic_analysis, run_static_checks


def _baseline_geometry():
    return canopy_geometry(
        GeometryParams(
            span=20.0,
            arch_rise=3.0,
            diagrid_divisions=8,
            branch_spread=3.2,
            glulam_depth=0.35,
        )
    )


def _comparison_results(geometry):
    results = {}
    for fuse_type in ("steel", "niti"):
        results[fuse_type] = {
            "seismic": run_seismic_analysis(
                geometry,
                fuse_type=fuse_type,
                pga_g=0.40,
                seed=101,
                record_frames=True,
            ),
            "static": run_static_checks(geometry, fuse_type=fuse_type),
        }
    return results


def _print_comparison(results):
    print(
        f"{'Fuse':<8} {'Peak drift':>12} {'Mean |tail|':>13} "
        f"{'Signed residual':>17} {'Base shear':>13} {'Max fuse strain':>17} "
        f"{'Gravity defl.':>15} {'Uplift defl.':>14}"
    )
    print("-" * 123)
    for fuse_type in ("steel", "niti"):
        seismic = results[fuse_type]["seismic"]
        max_fuse_strain = (
            100.0 * np.max(np.abs(seismic["history"]["lead_fuse_strain"]))
        )
        signed_residual = 100.0 * seismic["permanent_residual_roof_drift_ratio"]
        static_cases = results[fuse_type]["static"]["cases"]
        print(
            f"{fuse_type:<8} "
            f"{100 * seismic['peak_roof_drift_ratio']:>11.3f}% "
            f"{100 * seismic['mean_absolute_tail_roof_drift_ratio']:>15.4f}% "
            f"{signed_residual:>16.4f}% "
            f"{seismic['peak_base_shear_N'] / 1e3:>12.1f} "
            f"{max_fuse_strain:>16.4f}% "
            f"{1e3 * static_cases['gravity']['max_roof_deflection_m']:>14.2f} mm "
            f"{1e3 * static_cases['wind_uplift']['max_roof_deflection_m']:>13.2f} mm"
        )


def _plot_comparison(results, output: Path):
    colors = {"steel": "#D97706", "niti": "#147D92"}
    labels = {}
    for fuse_type, label in (
        ("steel", "Steel02"),
        ("niti", "NiTi SelfCentering"),
    ):
        residual = 100.0 * results[fuse_type]["seismic"][
            "permanent_residual_roof_drift_ratio"
        ]
        labels[fuse_type] = f"{label} (signed residual {residual:.4f}%)"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for fuse_type in ("steel", "niti"):
        history = results[fuse_type]["seismic"]["history"]
        axes[0].plot(
            history["time_s"],
            100.0 * history["signed_roof_drift_ratio"],
            color=colors[fuse_type],
            linewidth=1.5,
            label=labels[fuse_type],
        )
    axes[0].axhline(0.0, color="black", linewidth=0.7, alpha=0.6)
    axes[0].axvline(12.0, color="gray", linestyle="--", linewidth=0.9)
    axes[0].set(
        xlabel="Time (s)",
        ylabel="Mean signed roof drift (%)",
        title="Roof drift under 0.40g (seed 101)",
        xlim=(0.0, 18.0),
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for fuse_type in ("steel", "niti"):
        history = results[fuse_type]["seismic"]["history"]
        axes[1].plot(
            100.0 * history["lead_fuse_strain"],
            history["lead_fuse_stress_Pa"] / 1e6,
            color=colors[fuse_type],
            linewidth=1.4,
            label=labels[fuse_type],
        )
    axes[1].set(
        xlabel="Lead fuse strain (%)",
        ylabel="Axial stress (MPa)",
        title="Lead fuse stress-strain response",
    )
    activation_stress = {"steel": 355.0e6, "niti": 380.0e6}
    elastic_at_peak = all(
        np.max(np.abs(results[fuse_type]["seismic"]["history"]["lead_fuse_stress_Pa"]))
        < activation_stress[fuse_type]
        for fuse_type in ("steel", "niti")
    )
    if elastic_at_peak:
        axes[1].text(
            0.03,
            0.97,
            "Neither fuse reaches yield/activation in this 0.40g record",
            transform=axes[1].transAxes,
            va="top",
            fontsize=9,
            color="#555555",
        )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    results = _comparison_results(_baseline_geometry())
    _print_comparison(results)
    output = figures / "day2_steel_vs_niti_seismic.png"
    _plot_comparison(results, output)
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
