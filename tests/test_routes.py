import json
from unittest.mock import patch


def test_health_endpoint(client):
    resp = client.get("/api/session/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}
    assert resp.content_type == "application/json"


def test_calib_validation(client, sample_iris_points, sample_calib_iris_points, tmp_path):
    with patch("app.routes.session.Path") as mock_path:
        mock_path.return_value.absolute.return_value = tmp_path

        form_data = {
            "from_ruxailab": json.dumps(False),
            "file_name": json.dumps("test_session_123"),
            "fixed_circle_iris_points": json.dumps(sample_iris_points),
            "calib_circle_iris_points": json.dumps(sample_calib_iris_points),
            "screen_height": json.dumps(600),
            "screen_width": json.dumps(1000),
            "model": json.dumps("Linear Regression"),
            "k": json.dumps(2),
        }
        resp = client.post("/api/session/calib_validation", data=form_data)
        assert resp.status_code == 200

        data = json.loads(resp.data)
        assert "centroids" in data


def test_calib_validation_fails_without_data(client):
    resp = client.post("/api/session/calib_validation", data={})
    assert resp.status_code in (400, 500)


def test_calib_validation_rejects_get(client):
    assert client.get("/api/session/calib_validation").status_code == 405


def test_batch_predict_needs_calibration_points(client):
    payload = {
        "iris_tracking_data": [
            {"left_iris_x": 0.3, "left_iris_y": 0.4,
             "right_iris_x": 0.6, "right_iris_y": 0.4, "timestamp": 100}
        ],
        "screen_width": 1920,
        "screen_height": 1080,
    }
    resp = client.post("/api/session/batch_predict",
                       data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 400


def test_batch_predict_full_flow(client, sample_iris_points):
    payload = {
        "iris_tracking_data": [
            {"left_iris_x": 0.35, "left_iris_y": 0.45,
             "right_iris_x": 0.65, "right_iris_y": 0.45, "timestamp": 100},
            {"left_iris_x": 0.55, "left_iris_y": 0.55,
             "right_iris_x": 0.85, "right_iris_y": 0.55, "timestamp": 200},
        ],
        "screen_width": 1000,
        "screen_height": 600,
        "fixed_points": sample_iris_points,
        "calib_points": sample_iris_points,
    }

    resp = client.post("/api/session/batch_predict",
                       data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 200

    data = resp.get_json()
    assert len(data["predictions"]) == 2
    assert "calibration" in data["metrics"]
    for p in data["predictions"]:
        assert "predicted_x" in p
        assert "predicted_y" in p
        assert "timestamp" in p


def test_batch_predict_rejects_get(client):
    assert client.get("/api/session/batch_predict").status_code == 405


def test_batch_predict_handles_bad_json(client):
    resp = client.post("/api/session/batch_predict",
                       data="not json",
                       content_type="application/json")
    assert resp.status_code == 500
