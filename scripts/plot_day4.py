"""Day 4 figures: GP surrogate + certified Pareto front, and PyTorch NiTi inverse ID.

Reads the freshly generated Day 4 artifacts and renders two 300-DPI figures:
  - figures/day4_gp_and_pareto.png
  - figures/day4_diff_id_niti.png
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
COLORS = {"steel": "#c2410c", "niti": "#0ea5c6"}
LABELS = {"steel": "Steel (Steel02)", "niti": "NiTi (SelfCentering)"}
PARAM_ORDER = ["E", "sigma_Ms", "sigma_Mf", "sigma_As", "sigma_Af", "eps_L"]


def _load(name: str) -> dict:
    path = DATA / name
    if not path.exists():
        raise FileNotFoundError(f"required Day 4 artifact missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_gp_and_pareto(gp: dict, pareto: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.0), constrained_layout=True)

    # (a) Grouped-CV parity for peak roof drift (out-of-fold predictions).
    ax = axes[0]
    validation = gp["validation"]
    oof = validation["oof_peak_drift_pct"]
    per_case = validation["per_case"]
    combined = []
    for fuse in ("steel", "niti"):
        truth = np.asarray(oof[fuse]["truth"], dtype=float)
        mean = np.asarray(oof[fuse]["mean"], dtype=float)
        combined.append(np.concatenate((truth, mean)))
        r2 = per_case[fuse]["peak_drift_pct"]["r2"]
        ax.scatter(
            truth,
            mean,
            s=24,
            alpha=0.82,
            color=COLORS[fuse],
            edgecolors="white",
            linewidths=0.3,
            label=f"{LABELS[fuse]} ($R^2$={r2:.2f})",
        )
    extent = float(np.max(np.abs(np.concatenate(combined)))) * 1.05
    ax.plot([0.0, extent], [0.0, extent], color="#555555", ls="--", lw=0.9, label="Parity")
    ax.axvline(2.0, color="#c2410c", lw=0.8, alpha=0.5)
    ax.axhline(2.0, color="#c2410c", lw=0.8, alpha=0.5)
    ax.set(
        xlim=(0.0, extent),
        ylim=(0.0, extent),
        xlabel="Actual peak roof drift (%)",
        ylabel="GP predicted peak roof drift (%)",
        title="GP grouped-CV parity (peak drift)",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    # (b) False-safe peak-drift rate: mean-only vs 95% UCB.
    ax = axes[1]
    groups = ["steel", "niti", "pooled"]
    mean_rates = []
    ucb_rates = []
    for group in groups:
        rates = (
            validation["false_safe_rate"]
            if group == "pooled"
            else per_case[group]["false_safe_rate"]
        )
        mean_rates.append(rates["mean"])
        ucb_rates.append(rates["ucb_95"])
    xpos = np.arange(len(groups))
    width = 0.38
    ax.bar(xpos - width / 2, mean_rates, width, color="#c2410c", label="Mean only")
    ax.bar(xpos + width / 2, ucb_rates, width, color="#143055", label="95% UCB")
    for x, value in zip(xpos - width / 2, mean_rates):
        ax.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for x, value in zip(xpos + width / 2, ucb_rates):
        ax.text(x, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([g.capitalize() for g in groups])
    ax.set(
        ylabel="False-safe peak-drift rate",
        title="False-safe rate reduced by 95% UCB",
        ylim=(0.0, max(mean_rates) * 1.25 + 1e-6),
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    # (c) Certified Pareto front: lifecycle carbon vs clear span.
    ax = axes[2]
    for fuse in ("steel", "niti"):
        spans = []
        carbon = []
        for design in pareto["designs"]:
            if design["front_case"] != fuse:
                continue
            spans.append(design["params"]["span"])
            carbon.append(design["kpis"][fuse]["c_total_per_m2"])
        order = np.argsort(spans)
        ax.plot(
            np.asarray(spans)[order],
            np.asarray(carbon)[order],
            "-o",
            color=COLORS[fuse],
            markeredgecolor="white",
            markeredgewidth=0.5,
            lw=1.2,
            label=f"{LABELS[fuse]} front",
        )
    ax.set(
        xlabel="Clear span (m)",
        ylabel="Lifecycle carbon (kgCO$_2$e/m$^2$)",
        title="Certified Pareto front (8 exact OpenSeesPy designs)",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_diff_id(diff: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0), constrained_layout=True)
    loops = diff["loops"]
    strain = np.asarray(loops["strain"], dtype=float)
    true = np.asarray(loops["true_stress_MPa"], dtype=float)
    observed = np.asarray(loops["observed_stress_MPa"], dtype=float)
    recovered = np.asarray(loops["recovered_stress_MPa"], dtype=float)

    # (a) Noisy cyclic data vs recovered loop.
    ax = axes[0]
    ax.scatter(strain, observed, s=9, color="#94a3b8", alpha=0.55, label="Observed (3% noise)")
    ax.plot(strain, true, color="#143055", lw=1.9, label="True Auricchio-Sacco")
    ax.plot(strain, recovered, color="#0ea5c6", lw=1.3, ls="--", label="PyTorch recovered")
    ax.set(
        xlabel="Strain",
        ylabel="Stress (MPa)",
        title="NiTi cyclic loop: noisy data vs recovered",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best", fontsize=8)

    # (b) Parameter recovery error with 95% bootstrap CIs.
    ax = axes[1]
    true_params = diff["true_parameters"]
    recovered_params = diff["recovered_parameters"]
    intervals = diff["confidence_intervals_95"]
    centers = []
    lows = []
    highs = []
    for name in PARAM_ORDER:
        scale = true_params[name]
        centers.append((recovered_params[name] / scale - 1.0) * 100.0)
        lows.append((intervals[name][0] / scale - 1.0) * 100.0)
        highs.append((intervals[name][1] / scale - 1.0) * 100.0)
    xpos = np.arange(len(PARAM_ORDER))
    ax.axhspan(-8.0, 8.0, color="#e2e8f0", alpha=0.6, zorder=0)
    ax.axhline(0.0, color="#143055", lw=1.0)
    for x, low, high in zip(xpos, lows, highs):
        ax.vlines(x, low, high, color="#143055", lw=2.2, alpha=0.75, zorder=2)
    ax.scatter(xpos, centers, color="#0ea5c6", s=46, zorder=3, edgecolors="white", linewidths=0.6)
    max_error = max(abs(diff["relative_recovery_error_pct"][n]) for n in PARAM_ORDER)
    ax.set_xticks(xpos)
    ax.set_xticklabels(PARAM_ORDER, rotation=25)
    ax.set(
        ylabel="Recovery error (%)",
        title=f"PyTorch recovery + 95% CIs (max |err| = {max_error:.2f}%)",
    )
    ax.grid(axis="y", alpha=0.2)
    handles = [
        Line2D([0], [0], color="#0ea5c6", marker="o", lw=0, markersize=7, label="Recovered"),
        Line2D([0], [0], color="#143055", lw=2.2, label="95% bootstrap CI"),
        Line2D([0], [0], color="#e2e8f0", lw=6, alpha=0.6, label="±8% gate"),
    ]
    ax.legend(handles=handles, frameon=False, loc="best", fontsize=8)

    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    gp = _load("gp_surrogate.json")
    pareto = _load("pareto_canopies.json")
    diff = _load("diff_id_uncertainty.json")

    out1 = FIGURES / "day4_gp_and_pareto.png"
    out2 = FIGURES / "day4_diff_id_niti.png"
    _plot_gp_and_pareto(gp, pareto, out1)
    print(f"Saved {out1}")
    _plot_diff_id(diff, out2)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
