import numpy as np
import pandas as pd

from app.services.calib_validation.benchmark import (
    CalibrationConfig,
    benchmark_dataset,
    calculate_metrics,
    load_calibration_dataset,
    make_validation_splits,
)


def _dataset():
    rows = []
    for group, (x, y) in enumerate(((100, 100), (500, 300), (900, 500), (100, 500))):
        for offset in (-0.01, 0.01):
            rows.append(
                {
                    "fixation_id": group,
                    "left_iris_x": 0.3 + group * 0.1 + offset,
                    "left_iris_y": 0.4 + group * 0.05 + offset,
                    "right_iris_x": 0.6 + group * 0.1 + offset,
                    "right_iris_y": 0.4 + group * 0.05 + offset,
                    "screen_x": x,
                    "screen_y": y,
                }
            )
    return pd.DataFrame(rows)


def test_loader_normalizes_notebook_target_names(tmp_path):
    path = tmp_path / "calibration.csv"
    _dataset().to_csv(path, index=False)

    loaded = load_calibration_dataset(path)

    assert {"point_x", "point_y", "fixation_id"}.issubset(loaded.columns)
    assert len(loaded) == 8


def test_validation_splits_keep_fixations_together():
    frame = _dataset().rename(columns={"screen_x": "point_x", "screen_y": "point_y"})

    for train, test in make_validation_splits(frame, n_splits=2):
        assert set(frame.iloc[train].fixation_id).isdisjoint(frame.iloc[test].fixation_id)


def test_metrics_report_pixel_errors_and_precision():
    metrics = calculate_metrics(
        np.array([[0, 0], [10, 10]]),
        np.array([[3, 4], [10, 10]]),
        groups=["a", "b"],
    )

    assert metrics["mean_error_px"] == 2.5
    assert metrics["horizontal_mae_px"] == 1.5
    assert metrics["vertical_mae_px"] == 2.0
    assert metrics["precision_px"] == 0.0


def test_benchmark_supports_initial_models(tmp_path):
    path = tmp_path / "calibration.csv"
    _dataset().to_csv(path, index=False)

    results = benchmark_dataset(
        path,
        CalibrationConfig(
            n_splits=2,
            tune_hyperparameters=False,
            model_names=("linear", "ridge", "lasso"),
        ),
    )

    assert {row["model"] for row in results} == {"linear", "ridge", "lasso"}
    assert all(row["samples"] == 4 for row in results)
