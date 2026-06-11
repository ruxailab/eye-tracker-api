import pytest
import csv
import pandas as pd

from app.main import app as flask_app


@pytest.fixture
def app():
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_iris_points():
    points = []
    calibration_targets = [
        (100, 100, 0.30, 0.40, 0.60, 0.40),
        (500, 300, 0.50, 0.50, 0.80, 0.50),
        (900, 500, 0.70, 0.60, 0.90, 0.60),
        (100, 500, 0.20, 0.70, 0.50, 0.70),
        (900, 100, 0.60, 0.30, 0.85, 0.30),
    ]
    for px, py, lx, ly, rx, ry in calibration_targets:
        for jitter in [-0.01, 0.0, 0.01]:
            points.append({
                "left_iris_x": lx + jitter, "left_iris_y": ly + jitter,
                "right_iris_x": rx + jitter, "right_iris_y": ry + jitter,
                "point_x": px, "point_y": py,
            })
    return points


@pytest.fixture
def sample_calib_iris_points():
    return [
        {"left_iris_x": 0.35, "left_iris_y": 0.45, "right_iris_x": 0.65, "right_iris_y": 0.45},
        {"left_iris_x": 0.55, "left_iris_y": 0.55, "right_iris_x": 0.85, "right_iris_y": 0.55},
        {"left_iris_x": 0.45, "left_iris_y": 0.50, "right_iris_x": 0.75, "right_iris_y": 0.50},
    ]


@pytest.fixture
def calib_csv_path(tmp_path, sample_iris_points):
    csv_path = tmp_path / "test_fixed_train_data.csv"
    columns = ["left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y",
               "point_x", "point_y", "screen_height", "screen_width"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in sample_iris_points:
            writer.writerow({**row, "screen_height": 600, "screen_width": 1000})
    return str(csv_path)


@pytest.fixture
def predict_csv_path(tmp_path, sample_calib_iris_points):
    csv_path = tmp_path / "test_predict_data.csv"
    columns = ["left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in sample_calib_iris_points:
            writer.writerow(row)
    return str(csv_path)


@pytest.fixture
def sample_metrics_df():
    return pd.DataFrame({
        "True X": [100, 100, 100, 500, 500],
        "Predicted X": [110, 95, 105, 490, 510],
        "True Y": [200, 200, 200, 400, 400],
        "Predicted Y": [210, 195, 205, 390, 410],
    })
