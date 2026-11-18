import numpy as np
from sklearn.pipeline import Pipeline

from app.services.gaze_tracker import (
    squash,
    normalizeData,
    models,
    train_and_predict,
    predict,
    predict_new_data_simple,
)


def test_squash_zero_in_zero_out():
    assert squash(0) == 0.0


def test_squash_bounded_output():
    assert 0 < squash(1.0) < 1.0
    assert -1.0 < squash(-1.0) < 0


def test_squash_symmetry():
    assert abs(squash(0.5) + squash(-0.5)) < 1e-10


def test_squash_saturates():
    assert abs(squash(100.0) - 1.0) < 0.01


def test_squash_respects_limit():
    result = squash(2.0, limit=2.0)
    assert abs(result - np.tanh(1.0)) < 1e-10


def test_squash_handles_arrays():
    result = squash(np.array([-1.0, 0.0, 1.0]))
    assert len(result) == 3
    assert result[1] == 0.0


def test_normalize_maps_to_0_1():
    result = normalizeData(np.array([0.0, 5.0, 10.0]))
    np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])


def test_normalize_handles_negatives():
    result = normalizeData(np.array([-10.0, 0.0, 10.0]))
    np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])


def test_normalize_endpoints():
    result = normalizeData(np.array([1.0, 2.0, 3.0]))
    assert result[0] == 0.0
    assert result[-1] == 1.0


def test_all_models_registered():
    expected = [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net",
        "Bayesian Ridge",
        "SGD Regressor",
        "Support Vector Regressor",
        "Random Forest Regressor",
    ]

    for name in expected:
        assert name in models

    for name, model in models.items():
        assert isinstance(model, Pipeline), f"{name} should be a Pipeline"


def test_train_predict_linear():
    np.random.seed(42)

    X_train = np.random.rand(20, 2)
    y_train = X_train[:, 0] * 100 + X_train[:, 1] * 200

    X_test = np.random.rand(5, 2)
    y_test = X_test[:, 0] * 100 + X_test[:, 1] * 200

    preds = train_and_predict(
        "Linear Regression",
        X_train,
        y_train,
        X_test,
        y_test,
        "X",
    )

    assert len(preds) == 5


def test_train_predict_ridge_with_gridsearch():
    np.random.seed(42)

    X_train = np.random.rand(30, 2)
    y_train = X_train[:, 0] * 100 + X_train[:, 1] * 200

    X_test = np.random.rand(5, 2)
    y_test = X_test[:, 0] * 100 + X_test[:, 1] * 200

    preds = train_and_predict(
        "Ridge Regression",
        X_train,
        y_train,
        X_test,
        y_test,
        "X",
    )

    assert len(preds) == 5


def test_predict_full_pipeline(calib_csv_path):
    result = predict(
        calib_csv_path,
        k=2,
        model_X="Linear Regression",
        model_Y="Linear Regression",
    )

    assert isinstance(result, dict)
    assert "centroids" in result
    assert len(result["centroids"]) == 2

    for centroid in result["centroids"]:
        assert len(centroid) == 2


def test_predict_new_data(
    calib_csv_path,
    predict_csv_path,
    sample_calib_iris_points,
):
    iris_data = [
        {**pt, "timestamp": i * 100}
        for i, pt in enumerate(sample_calib_iris_points)
    ]

    result = predict_new_data_simple(
        calib_csv_path=calib_csv_path,
        predict_csv_path=predict_csv_path,
        iris_data=iris_data,
        screen_width=1000,
        screen_height=600,
    )

    assert isinstance(result, list)
    assert len(result) == len(iris_data)

    for p in result:
        assert isinstance(p["predicted_x"], float)
        assert isinstance(p["predicted_y"], float)
        assert "timestamp" in p


def test_predict_new_data_preserves_screen_size(
    calib_csv_path,
    predict_csv_path,
    sample_calib_iris_points,
):
    iris_data = [
        {**pt, "timestamp": i * 100}
        for i, pt in enumerate(sample_calib_iris_points)
    ]

    result = predict_new_data_simple(
        calib_csv_path=calib_csv_path,
        predict_csv_path=predict_csv_path,
        iris_data=iris_data,
        screen_width=1920,
        screen_height=1080,
    )

    for p in result:
        assert p["screen_width"] == 1920
        assert p["screen_height"] == 1080


def test_predict_new_data_simple_head_pose_invariance():
    rng = np.random.default_rng(42)
    W, H = 1920, 1080
    grid = [(x, y) for x in [200, 960, 1720] for y in [150, 540, 930]]
    fixed_points = []
    for px, py in grid:
        for _ in range(15):
            nose_x = 0.50 + rng.normal(0, 0.002)
            nose_y = 0.50 + rng.normal(0, 0.002)
            gaze_dx = (px - W / 2) / (W / 2) * 0.03
            gaze_dy = (py - H / 2) / (H / 2) * 0.025
            mean_x = nose_x + gaze_dx + rng.normal(0, 0.001)
            mean_y = nose_y + gaze_dy + rng.normal(0, 0.001)
            fixed_points.append({
                "left_iris_x": mean_x + 0.06,
                "right_iris_x": mean_x - 0.06,
                "left_iris_y": mean_y,
                "right_iris_y": mean_y,
                "nose_x": nose_x,
                "nose_y": nose_y,
                "head_yaw": (nose_x - 0.50) * 2.0,
                "head_pitch": (nose_y - 0.50) * 2.0,
                "head_roll": 0.0,
                "point_x": px,
                "point_y": py,
            })

    test_points, targets = [], []
    for px, py in grid:
        head_shift_x = rng.uniform(-0.06, 0.06)
        head_shift_y = rng.uniform(-0.04, 0.04)
        nose_x = 0.50 + head_shift_x
        nose_y = 0.50 + head_shift_y
        gaze_dx = (px - W / 2) / (W / 2) * 0.03
        gaze_dy = (py - H / 2) / (H / 2) * 0.025
        mean_x = nose_x + gaze_dx
        mean_y = nose_y + gaze_dy
        test_points.append({
            "left_iris_x": mean_x + 0.06,
            "right_iris_x": mean_x - 0.06,
            "left_iris_y": mean_y,
            "right_iris_y": mean_y,
            "nose_x": nose_x,
            "nose_y": nose_y,
            "head_yaw": head_shift_x * 2.0,
            "head_pitch": head_shift_y * 2.0,
            "head_roll": 0.0,
            "timestamp": 100,
        })
        targets.append((px, py))

    fixed_no_nose = [{k: v for k, v in r.items() if not k.startswith(("nose_", "head_"))} for r in fixed_points]
    test_no_nose = [{k: v for k, v in r.items() if not k.startswith(("nose_", "head_"))} for r in test_points]

    pred_base = predict_new_data_simple(fixed_no_nose, test_no_nose, screen_width=W, screen_height=H)
    pred_pose = predict_new_data_simple(fixed_points, test_points, screen_width=W, screen_height=H)

    err_base = np.mean([np.hypot(p["predicted_x"] - tx, p["predicted_y"] - ty) for p, (tx, ty) in zip(pred_base, targets)])
    err_pose = np.mean([np.hypot(p["predicted_x"] - tx, p["predicted_y"] - ty) for p, (tx, ty) in zip(pred_pose, targets)])

    assert err_pose < err_base * 0.3
    assert err_pose < 120.0
