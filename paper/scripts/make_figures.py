"""Publication figures and booktabs tables from the verified artifacts in paper/FACTS.md.

Reads only:
  data/diff_id_uncertainty.json
  data/verified_lhs_dataset.csv
  data/gp_surrogate.json  (validation object)
  data/pareto_canopies.json  (meta scalars)

The four tangent states and the two extra dissipation minima are the audit
table in FACTS.md section 1. They are not stored as a series. The Pareto
metadata scalar is checked against the worst tabulated tangent error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
FIG = ROOT / "paper" / "figures"
PREVIEW = FIG / "preview"
TAB = ROOT / "paper" / "tables"

NAVY = "#1B3A4B"
BRICK = "#A65D57"
TEAL = "#2A6F6A"
SLATE = "#5C6B73"
GOLD = "#C4A35A"
GRID = "#E6E4DF"
INK = "#1C1917"

# FACTS.md section 1. Relative |(E_alg - E_num) / E_num|.
TANGENT_STATES = [
    (r"$(\pm 0.001,\ 0)$", 1.2221789802409672e-12),
    (r"$(\pm 0.025,\ \pm 0.2)$", 1.8504164817172542e-10),
]
TANGENT_GATE = 1e-7
DISSIPATION_MINIMA = [
    ("5 cyc., 5000", -2.0235549347269455e-15),
    ("5 cyc., 5001", -1.700029006457271e-15),
    ("10 cyc., 5001", -1.97758476261356e-15),
]
DISSIPATION_GATE = -1e-9

PARAM_ORDER = ["E", "sigma_Ms", "sigma_Mf", "sigma_As", "sigma_Af", "eps_L"]
PARAM_ROWS = [
    ("E", r"$E$", "Young modulus", "MPa"),
    ("sigma_Ms", r"$\sigma_{Ms}$", "Martensite start stress", "MPa"),
    ("sigma_Mf", r"$\sigma_{Mf}$", "Martensite finish stress", "MPa"),
    ("sigma_As", r"$\sigma_{As}$", "Austenite start stress", "MPa"),
    ("sigma_Af", r"$\sigma_{Af}$", "Austenite finish stress", "MPa"),
    ("eps_L", r"$\varepsilon_L$", "Transformation strain", "1"),
]


def load() -> dict:
    ident = json.loads((DATA / "diff_id_uncertainty.json").read_text(encoding="utf-8"))
    gp = json.loads((DATA / "gp_surrogate.json").read_text(encoding="utf-8"))
    pareto = json.loads((DATA / "pareto_canopies.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA / "verified_lhs_dataset.csv")
    if len(frame) != 80:
        raise SystemExit(f"expected 80 designs, found {len(frame)}")
    worst = max(err for _, err in TANGENT_STATES)
    stored = float(pareto["meta"]["tangent_error_relative_max"])
    if abs(stored - worst) / worst > 1e-9:
        raise SystemExit("Pareto tangent scalar does not match the FACTS table")
    stored_d = float(pareto["meta"]["minimum_cyclic_dissipation_MPa"])
    if stored_d != DISSIPATION_MINIMA[1][1]:
        raise SystemExit("Pareto dissipation scalar does not match the 5001-sample loop")
    return {"ident": ident, "gp": gp["validation"], "frame": frame, "pareto_meta": pareto["meta"]}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "grid.alpha": 1.0,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", pad=4)


def finish(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(PREVIEW / f"{name}.png", dpi=150)
    plt.close(fig)


def fig1(bundle: dict) -> None:
    loops = bundle["ident"]["loops"]
    strain = np.asarray(loops["strain"], dtype=float)
    stress = np.asarray(loops["true_stress_MPa"], dtype=float)
    if not np.isfinite(strain).all() or not np.isfinite(stress).all():
        raise SystemExit("constitutive loop contains NaN")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45))
    ax = axes[0]
    ax.plot(strain, stress, color=NAVY, lw=0.9, label="Exact return map")
    for level, name in ((380, r"$\sigma_{Ms}$"), (450, r"$\sigma_{Mf}$"), (110, r"$\sigma_{Af}$"), (180, r"$\sigma_{As}$")):
        ax.axhline(level, color=SLATE, lw=0.4, ls=(0, (2, 2)))
        ax.axhline(-level, color=SLATE, lw=0.4, ls=(0, (2, 2)))
    ax.set_xlabel(r"Strain $\varepsilon$")
    ax.set_ylabel(r"Stress $\sigma$ (MPa)")
    ax.set_xlim(-0.072, 0.072)
    ax.set_ylim(-520, 520)
    panel_label(ax, "(a)  Growing five-cycle return map")
    ax.legend(loc="upper left", borderaxespad=0.2)
    finish(ax)

    ax = axes[1]
    labels = [name for name, _ in TANGENT_STATES]
    errors = [err for _, err in TANGENT_STATES]
    ypos = np.arange(len(labels))
    ax.barh(ypos, errors, color=TEAL, height=0.45, zorder=3)
    ax.axvline(TANGENT_GATE, color=BRICK, lw=0.8)
    ax.text(1.2e-8, 1.42, r"gate $10^{-7}$", color=BRICK, fontsize=7, ha="right", va="bottom")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Relative tangent error $|(E_{\mathrm{alg}}-E_{\mathrm{num}})/E_{\mathrm{num}}|$")
    ax.set_xscale("log")
    ax.set_xlim(1e-13, 3e-7)
    ax.set_ylim(-0.7, 1.7)
    d_txt = "\n".join(
        f"min $D$ ({name}) = {val:.2e} MPa" for name, val in DISSIPATION_MINIMA
    )
    d_txt += "\n" + r"gate $D \geq -10^{-9}$ MPa"
    ax.text(0.97, 0.38, d_txt, transform=ax.transAxes, ha="right", va="center", fontsize=6.5, color=INK, linespacing=1.45)
    panel_label(ax, "(b)  Tangent error and dissipation")
    finish(ax)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig1_constitutive_verification")


def fig2(bundle: dict) -> None:
    ident = bundle["ident"]
    loops = ident["loops"]
    strain = np.asarray(loops["strain"], dtype=float)
    obs = np.asarray(loops["observed_stress_MPa"], dtype=float)
    rec = np.asarray(loops["recovered_stress_MPa"], dtype=float)
    err = ident["relative_recovery_error_pct"]
    if not np.isfinite(obs).all() or not np.isfinite(rec).all():
        raise SystemExit("identification curves contain NaN")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.45), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.plot(strain, obs, color=SLATE, lw=0.45, alpha=0.85, label=r"Observed, $3\%$ noise")
    ax.plot(strain, rec, color=NAVY, lw=1.05, label="Recovered return map")
    ax.set_xlabel(r"Strain $\varepsilon$")
    ax.set_ylabel(r"Stress $\sigma$ (MPa)")
    panel_label(ax, "(a)  Noisy observations and recovery")
    ax.legend(loc="upper left")
    finish(ax)

    ax = axes[1]
    names = [r"$E$", r"$\sigma_{Ms}$", r"$\sigma_{Mf}$", r"$\sigma_{As}$", r"$\sigma_{Af}$", r"$\varepsilon_L$"]
    values = [float(err[key]) for key in PARAM_ORDER]
    colors = [TEAL if value < 2 else BRICK for value in values]
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, values, color=colors, height=0.62, zorder=3)
    ax.axvline(8.0, color=SLATE, lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks(ypos)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"Relative recovery error $|\hat\theta/\theta-1|\times 100$ (%)")
    ax.set_xlim(0, 11.6)
    ax.set_ylim(-0.55, 5.55)
    ax.text(8.0, 5.35, r"$8\%$ gate", color=SLATE, fontsize=7, ha="center", va="bottom")
    for y, value in zip(ypos, values):
        if value >= 3:
            ax.text(value - 0.15, y, f"{value:.2f}", va="center", ha="right", fontsize=7, color="white")
        else:
            ax.text(value + 0.15, y, f"{value:.2f}", va="center", ha="left", fontsize=7, color=INK)
    panel_label(ax, "(b)  Six-parameter recovery")
    finish(ax)
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig2_inverse_id")


def fig3(bundle: dict) -> None:
    frame = bundle["frame"]
    steel_peak = frame["steel_peak_roof_drift_ratio"].to_numpy() * 100
    niti_peak = frame["niti_peak_roof_drift_ratio"].to_numpy() * 100
    steel_res = frame["steel_permanent_residual_roof_drift_ratio"].to_numpy() * 100
    niti_res = frame["niti_permanent_residual_roof_drift_ratio"].to_numpy() * 100
    steel_c = frame["steel_total_intensity_kgco2e_m2"].to_numpy()
    niti_c = frame["niti_total_intensity_kgco2e_m2"].to_numpy()
    arrays = [steel_peak, niti_peak, steel_res, niti_res, steel_c, niti_c]
    if any(not np.isfinite(a).all() for a in arrays):
        raise SystemExit("seismic-carbon columns contain NaN")

    peak_ratio = float(np.mean(niti_peak) / np.mean(steel_peak))
    res_ratio = float(np.mean(niti_res) / np.mean(steel_res))
    carbon_gap = float(np.mean(niti_c) - np.mean(steel_c))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    ax = axes[0]
    ax.scatter(steel_peak, steel_res, s=16, c=NAVY, zorder=3, label="Yielding steel")
    ax.scatter(niti_peak, niti_res, s=16, c=BRICK, marker="s", zorder=3, label="NiTi")
    ax.axvline(2.0, color=SLATE, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlabel(r"Peak roof drift ratio $\max|u_x|/4.5$ (%)")
    ax.set_ylabel(r"Absolute residual drift ratio (%)")
    note = (
        f"Mean residual NiTi/steel = {res_ratio:.3f}\n"
        f"Mean peak NiTi/steel = {peak_ratio:.3f}\n"
        f"Mean carbon penalty = {carbon_gap:+.2f} kgCO$_2$e/m$^2$"
    )
    ax.text(0.58, 0.98, note, transform=ax.transAxes, ha="center", va="top", fontsize=7, color=INK, linespacing=1.35)
    panel_label(ax, "(a)  Recentering against peak drift")
    ax.legend(loc="center right", borderpad=0.15)
    finish(ax)

    ax = axes[1]
    ax.scatter(steel_c, niti_c, s=16, c=TEAL, zorder=3)
    lo = min(steel_c.min(), niti_c.min()) - 4
    hi = max(steel_c.max(), niti_c.max()) + 4
    ax.plot([lo, hi], [lo, hi], color=SLATE, lw=0.7, label="Equal intensity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Steel lifecycle intensity (kgCO$_2$e/m$^2$)")
    ax.set_ylabel(r"NiTi lifecycle intensity (kgCO$_2$e/m$^2$)")
    n_higher = int(np.sum(niti_c > steel_c))
    ax.text(
        0.04,
        0.96,
        f"NiTi higher on {n_higher}/80 designs\nMean gap {carbon_gap:+.2f} kgCO$_2$e/m$^2$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    panel_label(ax, "(b)  Lifecycle carbon, same geometry")
    ax.legend(loc="lower right")
    finish(ax)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_seismic_pareto")


def fig4(bundle: dict) -> None:
    oof = bundle["gp"]["oof_peak_drift_pct"]
    per = bundle["gp"]["per_case"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
    false_safe = 0
    held = 0
    for ax, fuse, color, title in (
        (axes[0], "steel", NAVY, "(a)  Steel fuses"),
        (axes[1], "niti", BRICK, "(b)  NiTi fuses"),
    ):
        truth = np.asarray(oof[fuse]["truth"], dtype=float)
        mean = np.asarray(oof[fuse]["mean"], dtype=float)
        std = np.asarray(oof[fuse]["std"], dtype=float)
        if not np.isfinite(truth).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise SystemExit(f"{fuse} out-of-fold peak drift contains NaN")
        ucb = mean + 1.96 * std
        unsafe = truth > 2.0
        false_safe += int(np.sum(unsafe & (ucb <= 2.0)))
        held += truth.size
        ax.errorbar(
            truth,
            mean,
            yerr=1.96 * std,
            fmt="o",
            ms=3.2,
            color=color,
            ecolor=SLATE,
            elinewidth=0.45,
            capsize=0,
            zorder=3,
        )
        lim_hi = max(8.5, float(np.max(truth)), float(np.max(ucb))) + 0.3
        ax.plot([0, lim_hi], [0, lim_hi], color=SLATE, lw=0.6, zorder=1)
        ax.axvline(2.0, color=GOLD, lw=0.7, zorder=2)
        ax.axhline(2.0, color=GOLD, lw=0.7, zorder=2)
        # Held-out rows with true drift above 2% and a 95% upper bound still at or below 2%.
        ax.add_patch(
            Rectangle((2.0, 0.0), lim_hi - 2.0, 2.0, facecolor=BRICK, alpha=0.08, edgecolor="none", zorder=0)
        )
        r2 = per[fuse]["peak_drift_pct"]["r2"]
        rate = per[fuse]["false_safe_rate"]
        ax.text(
            0.04,
            0.96,
            f"$R^2$ = {r2:.3f}\nUCB false-safe = {int(rate['ucb_95'] * rate['held_out_count'])}"
            f"/{int(rate['held_out_count'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
        ax.set_xlim(0, lim_hi)
        ax.set_ylim(0, lim_hi)
        ax.set_xlabel(r"True peak drift (%)")
        ax.set_ylabel(r"Grouped-CV prediction (%)")
        panel_label(ax, title)
        finish(ax)
    if false_safe != 0:
        raise SystemExit(f"expected zero UCB false-safe rows, found {false_safe}")
    if held != 160:
        raise SystemExit(f"expected 160 held-out rows, found {held}")
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SLATE, ms=4, label=r"Mean $\pm 1.96\sigma$"),
        Line2D([0], [0], color=GOLD, lw=0.8, label="2% screen"),
        Rectangle((0, 0), 1, 1, facecolor=BRICK, alpha=0.25, edgecolor="none", label="False-safe region"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 1))
    save(fig, "fig4_gp_screening")


def write_tables(bundle: dict) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    ident = bundle["ident"]
    true = ident["true_parameters"]
    rec = ident["recovered_parameters"]
    err = ident["relative_recovery_error_pct"]
    lines = [
        "% Source: data/diff_id_uncertainty.json. Relative error is |theta_hat/theta - 1| x 100.",
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Symbol & Meaning & True & Recovered & Error (\%) & Unit \\",
        r"\midrule",
    ]
    digits = {"E": 2, "sigma_Ms": 2, "sigma_Mf": 2, "sigma_As": 2, "sigma_Af": 2, "eps_L": 5}
    true_digits = {"E": 0, "sigma_Ms": 0, "sigma_Mf": 0, "sigma_As": 0, "sigma_Af": 0, "eps_L": 2}
    for key, symbol, meaning, unit in PARAM_ROWS:
        lines.append(
            f"{symbol} & {meaning} & {true[key]:.{true_digits[key]}f} & "
            f"{rec[key]:.{digits[key]}f} & {err[key]:.3f} & {unit} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (TAB / "tab1_inverse_params.tex").write_text("\n".join(lines), encoding="utf-8")

    frame = bundle["frame"]
    gp = bundle["gp"]
    rows = []

    def add(quantity, steel, niti, unit):
        rows.append((quantity, steel, niti, unit))

    add(
        r"Peak roof drift, mean",
        f"{frame['steel_peak_roof_drift_ratio'].mean() * 100:.3f}",
        f"{frame['niti_peak_roof_drift_ratio'].mean() * 100:.3f}",
        r"\%",
    )
    add(
        r"Absolute residual drift, mean",
        f"{frame['steel_permanent_residual_roof_drift_ratio'].mean() * 100:.4f}",
        f"{frame['niti_permanent_residual_roof_drift_ratio'].mean() * 100:.4f}",
        r"\%",
    )
    add(
        r"Lifecycle intensity, mean",
        f"{frame['steel_total_intensity_kgco2e_m2'].mean():.2f}",
        f"{frame['niti_total_intensity_kgco2e_m2'].mean():.2f}",
        r"kgCO$_2$e/m$^2$",
    )
    add(
        r"A1--A3 carbon, mean",
        f"{frame['steel_c_a1_a3_kgco2e'].mean():.1f}",
        f"{frame['niti_c_a1_a3_kgco2e'].mean():.1f}",
        r"kgCO$_2$e",
    )
    steel_fs = gp["per_case"]["steel"]["false_safe_rate"]
    niti_fs = gp["per_case"]["niti"]["false_safe_rate"]
    add(
        r"False-safe rate, mean rule",
        f"{steel_fs['mean']:.3f}",
        f"{niti_fs['mean']:.3f}",
        "fraction of 80",
    )
    add(
        r"False-safe rate, 95\% UCB",
        f"{steel_fs['ucb_95']:.3f}",
        f"{niti_fs['ucb_95']:.3f}",
        "fraction of 80",
    )
    add(
        r"Peak-drift grouped-CV $R^2$",
        f"{gp['per_case']['steel']['peak_drift_pct']['r2']:.3f}",
        f"{gp['per_case']['niti']['peak_drift_pct']['r2']:.3f}",
        "--",
    )
    out = [
        "% Means from data/verified_lhs_dataset.csv (80 paired designs, seed 101, 0.40 g).",
        "% Screening rates from data/gp_surrogate.json validation.per_case.",
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Quantity & Yielding steel & NiTi & Unit \\",
        r"\midrule",
    ]
    for quantity, steel, niti, unit in rows:
        out.append(f"{quantity} & {steel} & {niti} & {unit} \\\\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    (TAB / "tab2_niti_vs_steel.tex").write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    style()
    bundle = load()
    fig1(bundle)
    fig2(bundle)
    fig3(bundle)
    fig4(bundle)
    write_tables(bundle)
    required = [
        FIG / "fig1_constitutive_verification.pdf",
        FIG / "fig2_inverse_id.pdf",
        FIG / "fig3_seismic_pareto.pdf",
        FIG / "fig4_gp_screening.pdf",
        PREVIEW / "fig1_constitutive_verification.png",
        PREVIEW / "fig2_inverse_id.png",
        PREVIEW / "fig3_seismic_pareto.png",
        PREVIEW / "fig4_gp_screening.png",
        TAB / "tab1_inverse_params.tex",
        TAB / "tab2_niti_vs_steel.tex",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise SystemExit("empty or missing outputs: " + ", ".join(missing))
    print("wrote", len(required), "files")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise
