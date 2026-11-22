import json

import numpy as np
import pandas as pd

from diagrid_opt.surrogate_gp import (
    DATASET_PATH,
    FEATURE_COLUMNS,
    fit_grouped_cv_surrogates,
)


def _browser_matern_prediction(case_export, model_export, features):
    standardized = (
        features - np.asarray(case_export["scaler"]["mean"])
    ) / np.asarray(case_export["scaler"]["scale"])
    training = np.asarray(case_export["X_train_standardized"])
    length_scale = np.asarray(model_export["length_scale"])
    distance = np.sqrt(np.sum(((training - standardized) / length_scale) ** 2, axis=1))
    root5_distance = np.sqrt(5.0) * distance
    covariance = model_export["constant_value"] * (
        1.0 + root5_distance + 5.0 * distance**2 / 3.0
    ) * np.exp(-root5_distance)
    mean = float(covariance @ np.asarray(model_export["alpha"]))
    cholesky = np.asarray(model_export["cholesky"])
    projection = np.linalg.solve(cholesky, covariance)
    variance = (
        model_export["constant_value"]
        + model_export["noise_level"]
        - float(projection @ projection)
    )
    return mean, np.sqrt(max(variance, 0.0))


def test_grouped_cv_exports_gp_parameters_for_exact_browser_prediction(tmp_path):
    export_path = tmp_path / "gp_surrogate.json"
    bundle = fit_grouped_cv_surrogates(export_path=export_path)
    artifact = json.loads(export_path.read_text(encoding="utf-8"))
    dataset = pd.read_csv(DATASET_PATH)
    sample = dataset.loc[0, list(FEATURE_COLUMNS)].to_numpy(dtype=float)

    assert bundle.validation["fold_sizes"] == [20, 20, 20, 20]
    rates = bundle.validation["false_safe_rate"]
    assert rates["ucb_95"] < rates["mean"]
    assert set(artifact["cases"]) == {"steel", "niti"}
    for fuse_type in ("steel", "niti"):
        case_export = artifact["cases"][fuse_type]
        assert len(case_export["X_train_standardized"]) == len(dataset)
        assert len(case_export["scaler"]["mean"]) == len(FEATURE_COLUMNS)
        for target in artifact["targets"]:
            model_export = case_export["models"][target]
            assert len(model_export["alpha"]) == len(dataset)
            assert len(model_export["cholesky"]) == len(dataset)
            predicted_mean, predicted_std = bundle.predict(fuse_type, sample)[target]
            exported_mean, exported_std = _browser_matern_prediction(
                case_export, model_export, sample
            )
            np.testing.assert_allclose(
                predicted_mean[0], exported_mean, rtol=1e-8, atol=1e-8
            )
            np.testing.assert_allclose(
                predicted_std[0], exported_std, rtol=1e-8, atol=1e-8
            )
