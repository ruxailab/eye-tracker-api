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
