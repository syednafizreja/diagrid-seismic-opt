from diagrid_opt.dataset import generate_verified_dataset


def _mean_sd(values, scale=1.0):
    return (
        f"{values.mean() * scale:.3f} ± "
        f"{values.std(ddof=1) * scale:.3f}"
    )


def _print_summary(dataset):
    print(f"\nMean ± SD across {len(dataset)} verified designs")
    print(
        f"{'Fuse':<8} {'A1-A3 kgCO2e/m2':>22} "
        f"{'Total kgCO2e/m2':>22} {'Peak drift %':>18} "
        f"{'Signed residual %':>21} {'Base shear kN':>18}"
    )
    print("-" * 113)
    for fuse_type, label in (("steel", "Steel"), ("niti", "NiTi")):
        area = dataset["shaded_area_m2"]
        print(
            f"{label:<8} "
            f"{_mean_sd(dataset[f'{fuse_type}_c_a1_a3_kgco2e'] / area):>22} "
            f"{_mean_sd(dataset[f'{fuse_type}_total_intensity_kgco2e_m2']):>22} "
            f"{_mean_sd(dataset[f'{fuse_type}_peak_roof_drift_ratio'], 100.0):>18} "
            f"{_mean_sd(dataset[f'{fuse_type}_signed_permanent_residual_roof_drift_ratio'], 100.0):>21} "
            f"{_mean_sd(dataset[f'{fuse_type}_peak_base_shear_N'], 1e-3):>18}"
        )


if __name__ == "__main__":
    dataset = generate_verified_dataset()
    print(
        f"Saved {len(dataset)} verified paired designs to "
        "data/verified_lhs_dataset.csv"
    )
    _print_summary(dataset)
