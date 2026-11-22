from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = (
    "span_m",
    "arch_rise_ratio",
    "diagrid_divisions",
    "glulam_depth_m",
    "niti_fuse_area_mm2",
)
FUSE_TYPES = ("steel", "niti")
TARGET_NAMES = (
    "peak_drift_pct",
    "abs_residual_drift_pct",
    "grav_defl_mm",
    "c_total_per_m2",
)
DATASET_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "verified_lhs_dataset.csv"
)
EXPORT_PATH = Path(__file__).resolve().parents[2] / "data" / "gp_surrogate.json"


@dataclass(frozen=True)
class CaseSurrogate:
    scaler: StandardScaler
    regressors: dict[str, GaussianProcessRegressor]
    training_domain: dict[str, list[float]]


@dataclass(frozen=True)
class SurrogateBundle:
    models: dict[str, CaseSurrogate]
    validation: dict[str, Any]

    def predict(
        self, fuse_type: str, features: np.ndarray | list[float]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if fuse_type not in self.models:
            raise ValueError(f"unknown fuse type {fuse_type!r}")
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_COLUMNS):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_COLUMNS)})"
            )
        case = self.models[fuse_type]
        scaled = case.scaler.transform(values)
        return {
            target: model.predict(scaled, return_std=True)
            for target, model in case.regressors.items()
        }


def _target_columns(fuse_type: str) -> dict[str, tuple[str, float]]:
    return {
        "peak_drift_pct": (f"{fuse_type}_peak_roof_drift_ratio", 100.0),
        "abs_residual_drift_pct": (
            f"{fuse_type}_permanent_residual_roof_drift_ratio",
            100.0,
        ),
        "grav_defl_mm": (
            f"{fuse_type}_static_gravity_deflection_m",
            1000.0,
        ),
        "c_total_per_m2": (
            f"{fuse_type}_total_intensity_kgco2e_m2",
            1.0,
        ),
    }


def _load_dataset(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dataset = pd.read_csv(path)
    if dataset.empty:
        raise ValueError("verified dataset must contain at least one row")
    required = list(FEATURE_COLUMNS)
    for fuse_type in FUSE_TYPES:
        required.extend(column for column, _ in _target_columns(fuse_type).values())
    missing = sorted(set(required) - set(dataset.columns))
    if missing:
        raise ValueError(f"dataset is missing columns: {', '.join(missing)}")
    features = dataset.loc[:, FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("dataset features must be finite")
    return dataset, features, dataset["span_m"].to_numpy(dtype=float)


def _span_quartiles(spans: np.ndarray) -> np.ndarray:
    groups = pd.qcut(spans, q=4, labels=False, duplicates="drop")
    values = np.asarray(groups, dtype=int)
    if len(np.unique(values)) != 4:
        raise ValueError("span values must form four non-empty quartile groups")
    return values


def _new_regressor(target: np.ndarray, seed: int) -> GaussianProcessRegressor:
    variance = max(float(np.var(target)), 1e-8)
    kernel = (
        ConstantKernel(variance, (1e-8, 1e8))
        * Matern(
            length_scale=np.ones(len(FEATURE_COLUMNS)),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=max(variance * 0.01, 1e-10),
            noise_level_bounds=(1e-10, 1e4),
        )
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=False,
        random_state=seed,
        n_restarts_optimizer=0,
    )


def _cv_metrics(
    truth: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> dict[str, float]:
    residual = truth - mean
    total_ss = float(np.sum((truth - np.mean(truth)) ** 2))
    residual_ss = float(np.sum(residual**2))
    if total_ss > 0.0:
        r2 = 1.0 - residual_ss / total_ss
    else:
        r2 = 1.0 if residual_ss == 0.0 else 0.0
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "mean_predictive_std": float(np.mean(std)),
        "r2": float(r2),
    }


def _false_safe_rates(
    truth: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> dict[str, float]:
    actual_unsafe = truth > 2.0
    return {
        "mean": float(np.mean((mean <= 2.0) & actual_unsafe)),
        "ucb_95": float(
            np.mean(((mean + 1.96 * std) <= 2.0) & actual_unsafe)
        ),
        "held_out_count": int(truth.size),
        "actual_unsafe_count": int(np.count_nonzero(actual_unsafe)),
    }


def _kernel_export(model: GaussianProcessRegressor) -> dict[str, Any]:
    kernel = model.kernel_
    matern = kernel.k1.k2
    return {
        "constant_value": float(kernel.k1.k1.constant_value),
        "length_scale": np.asarray(matern.length_scale, dtype=float).tolist(),
        "noise_level": float(kernel.k2.noise_level),
        "alpha": np.asarray(model.alpha_, dtype=float).tolist(),
        "cholesky": np.asarray(model.L_, dtype=float).tolist(),
    }


def _browser_export(bundle: SurrogateBundle) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for fuse_type, case in bundle.models.items():
        model = next(iter(case.regressors.values()))
        cases[fuse_type] = {
            "scaler": {
                "mean": case.scaler.mean_.tolist(),
                "scale": case.scaler.scale_.tolist(),
            },
            "training_domain": case.training_domain,
            "X_train_standardized": np.asarray(
                model.X_train_, dtype=float
            ).tolist(),
            "models": {
                target: _kernel_export(regressor)
                for target, regressor in case.regressors.items()
            },
        }
    return {
        "format_version": 1,
        "features": list(FEATURE_COLUMNS),
        "targets": list(TARGET_NAMES),
        "kernel": "ConstantKernel * Matern(nu=2.5) + WhiteKernel",
        "validation": bundle.validation,
        "cases": cases,
    }


def fit_grouped_cv_surrogates(
    dataset_path: str | Path = DATASET_PATH,
    export_path: str | Path = EXPORT_PATH,
    seed: int = 42,
) -> SurrogateBundle:
    dataset, features, spans = _load_dataset(dataset_path)
    groups = _span_quartiles(spans)
    splitter = GroupKFold(n_splits=4)
    split_indices = list(splitter.split(features, groups=groups))
    fold_sizes = [int(test.size) for _, test in split_indices]
    trained: dict[str, CaseSurrogate] = {}
    per_case_validation: dict[str, Any] = {}
    oof_peak_drift: dict[str, Any] = {}
    all_truth: list[np.ndarray] = []
    all_mean: list[np.ndarray] = []
    all_std: list[np.ndarray] = []

    for case_index, fuse_type in enumerate(FUSE_TYPES):
        target_data = {
            name: dataset[column].to_numpy(dtype=float) * scale
            for name, (column, scale) in _target_columns(fuse_type).items()
        }
        case_validation: dict[str, Any] = {}
        oof_peak: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        for target_index, target_name in enumerate(TARGET_NAMES):
            truth = target_data[target_name]
            mean = np.full(truth.shape, np.nan, dtype=float)
            std = np.full(truth.shape, np.nan, dtype=float)
            for fold_index, (train_indices, test_indices) in enumerate(split_indices):
                scaler = StandardScaler()
                train_x = scaler.fit_transform(features[train_indices])
                test_x = scaler.transform(features[test_indices])
                regressor = _new_regressor(
                    truth[train_indices], seed + case_index * 100 + target_index * 10 + fold_index
                )
                regressor.fit(train_x, truth[train_indices])
                mean[test_indices], std[test_indices] = regressor.predict(
                    test_x, return_std=True
                )
            if not np.isfinite(mean).all() or not np.isfinite(std).all():
                raise RuntimeError(f"non-finite grouped-CV prediction: {fuse_type}/{target_name}")
            case_validation[target_name] = _cv_metrics(truth, mean, std)
            if target_name == "peak_drift_pct":
                oof_peak = truth, mean, std

        assert oof_peak is not None
        case_validation["false_safe_rate"] = _false_safe_rates(*oof_peak)
        oof_peak_drift[fuse_type] = {
            "truth": oof_peak[0].tolist(),
            "mean": oof_peak[1].tolist(),
            "std": oof_peak[2].tolist(),
        }
        all_truth.append(oof_peak[0])
        all_mean.append(oof_peak[1])
        all_std.append(oof_peak[2])

        full_scaler = StandardScaler().fit(features)
        scaled_features = full_scaler.transform(features)
        full_models: dict[str, GaussianProcessRegressor] = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            regressor = _new_regressor(
                target_data[target_name], seed + case_index * 100 + target_index
            )
            regressor.fit(scaled_features, target_data[target_name])
            full_models[target_name] = regressor
        trained[fuse_type] = CaseSurrogate(
            scaler=full_scaler,
            regressors=full_models,
            training_domain={
                "minimum": features.min(axis=0).tolist(),
                "maximum": features.max(axis=0).tolist(),
            },
        )
        per_case_validation[fuse_type] = case_validation

    pooled_truth = np.concatenate(all_truth)
    pooled_mean = np.concatenate(all_mean)
    pooled_std = np.concatenate(all_std)
    validation = {
        "cv": "4-fold GroupKFold by span quartile",
        "fold_sizes": fold_sizes,
        "span_quartile_by_row": groups.tolist(),
        "per_case": per_case_validation,
        "oof_peak_drift_pct": oof_peak_drift,
        "false_safe_rate": _false_safe_rates(
            pooled_truth, pooled_mean, pooled_std
        ),
    }
    bundle = SurrogateBundle(models=trained, validation=validation)
    export = _browser_export(bundle)

    # Export the common standardized design matrix once per case. GP prediction
    # uses each target's own alpha and Cholesky factor.
    for fuse_type, case_export in export["cases"].items():
        case_export["X_train_standardized"] = trained[
            fuse_type
        ].scaler.transform(features).tolist()
    destination = Path(export_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(export, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return bundle


def main() -> None:
    bundle = fit_grouped_cv_surrogates()
    for fuse_type in FUSE_TYPES:
        print(
            f"{fuse_type}: false-safe mean="
            f"{bundle.validation['per_case'][fuse_type]['false_safe_rate']['mean']:.4f}, "
            f"95% UCB="
            f"{bundle.validation['per_case'][fuse_type]['false_safe_rate']['ucb_95']:.4f}"
        )
    print(
        "pooled: false-safe mean="
        f"{bundle.validation['false_safe_rate']['mean']:.4f}, 95% UCB="
        f"{bundle.validation['false_safe_rate']['ucb_95']:.4f}"
    )
    print(f"Saved {EXPORT_PATH}")


if __name__ == "__main__":
    main()
