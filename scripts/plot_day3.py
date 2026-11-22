from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def _read_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    required = [
        "span_m",
        "shaded_area_m2",
        "steel_c_a1_a3_kgco2e",
        "niti_c_a1_a3_kgco2e",
        "steel_total_intensity_kgco2e_m2",
        "niti_total_intensity_kgco2e_m2",
        "steel_peak_roof_drift_ratio",
        "niti_peak_roof_drift_ratio",
        "steel_signed_permanent_residual_roof_drift_ratio",
        "niti_signed_permanent_residual_roof_drift_ratio",
    ]
    missing = [column for column in required if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {', '.join(missing)}")
    values = dataset[required].to_numpy(dtype=float)
    if dataset.empty or not np.isfinite(values).all():
        raise ValueError("dataset must contain verified rows with finite metrics")
    if (dataset["shaded_area_m2"] <= 0.0).any():
        raise ValueError("shaded_area_m2 must be positive")
    return dataset


def _plot(dataset: pd.DataFrame, output: Path) -> None:
    colors = {"steel": "#D97706", "niti": "#147D92"}
    labels = {"steel": "Steel (Steel02)", "niti": "NiTi (SelfCentering)"}
    markers = {"steel": "o", "niti": "^"}
    span_values = dataset["span_m"].to_numpy(dtype=float)
    span_norm = Normalize(vmin=float(span_values.min()), vmax=float(span_values.max()))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2), constrained_layout=True)

    for fuse_type in ("steel", "niti"):
        upfront = (
            dataset[f"{fuse_type}_c_a1_a3_kgco2e"]
            / dataset["shaded_area_m2"]
        )
        total = dataset[f"{fuse_type}_total_intensity_kgco2e_m2"]
        axes[0].scatter(
            upfront,
            total,
            c=span_values,
            cmap="viridis",
            norm=span_norm,
            marker=markers[fuse_type],
            s=42,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )

    limits = axes[0].get_xlim()
    axes[0].plot(
        limits,
        limits,
        color="#555555",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
        label="No repair carbon",
    )
    axes[0].set(
        xlabel="Upfront A1-A3 carbon (kgCO$_2$e/m$^2$)",
        ylabel="Total lifecycle carbon (kgCO$_2$e/m$^2$)",
        title="Upfront vs lifecycle carbon",
    )
    axes[0].grid(alpha=0.22)
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=markers[fuse_type],
                color="none",
                markerfacecolor="#666666",
                markeredgecolor="white",
                markersize=7,
                label=labels[fuse_type],
            )
            for fuse_type in ("steel", "niti")
        ]
        + [
            Line2D(
                [0],
                [0],
                color="#555555",
                linestyle="--",
                linewidth=0.9,
                label="No repair carbon",
            )
        ],
        frameon=False,
        loc="best",
    )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=span_norm, cmap="viridis"),
        ax=axes[0],
        pad=0.02,
    )
    colorbar.set_label("Clear span (m)")

    steel_peak = 100.0 * dataset["steel_peak_roof_drift_ratio"].to_numpy()
    niti_peak = 100.0 * dataset["niti_peak_roof_drift_ratio"].to_numpy()
    steel_residual = (
        100.0
        * dataset["steel_signed_permanent_residual_roof_drift_ratio"].to_numpy()
    )
    niti_residual = (
        100.0
        * dataset["niti_signed_permanent_residual_roof_drift_ratio"].to_numpy()
    )
    for steel_x, steel_y, niti_x, niti_y in zip(
        steel_peak, steel_residual, niti_peak, niti_residual
    ):
        axes[1].plot(
            (steel_x, niti_x),
            (steel_y, niti_y),
            color="#888888",
            linewidth=0.65,
            alpha=0.22,
            zorder=1,
        )
    axes[1].scatter(
        steel_peak,
        steel_residual,
        color=colors["steel"],
        marker=markers["steel"],
        s=38,
        alpha=0.82,
        label=labels["steel"],
        zorder=2,
    )
    axes[1].scatter(
        niti_peak,
        niti_residual,
        color=colors["niti"],
        marker=markers["niti"],
        s=42,
        alpha=0.82,
        label=labels["niti"],
        zorder=2,
    )
    axes[1].axhline(0.0, color="#555555", linewidth=0.8, alpha=0.65)
    axes[1].set(
        xlabel="Peak roof drift (%)",
        ylabel="Signed permanent residual drift (%)",
        title="Seismic drift envelope (paired designs)",
    )
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False, loc="best")

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_path = root / "data" / "verified_lhs_dataset.csv"
    output = root / "figures" / "day3_carbon_seismic_tradeoff.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = _read_dataset(dataset_path)
    _plot(dataset, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
