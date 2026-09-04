# Necesary imports
import os
import re
import time
import json
import csv
import math
import numpy as np
import uuid
from pathlib import Path
import os
import pandas as pd
import traceback
import re
import requests
from flask import Flask, request, Response, send_file, jsonify

# Local imports from app
from app.services.storage import save_file_locally
from app.models.session import Session

# from app.services import database as db
from app.services import gaze_tracker


# Constants

ALLOWED_EXTENSIONS = {"txt", "webm"}
COLLECTION_NAME = "session"

def summarize_numeric_values(values, screen_width=None, screen_height=None):
    numeric_values = [
        value for value in values
        if isinstance(value, (int, float, np.integer, np.floating))
        and np.isfinite(value)
    ]
    array = np.asarray(numeric_values, dtype=float)
    return {
        "min": float(np.min(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
        "mean": float(np.mean(array)) if array.size else None,
        "std": float(np.std(array)) if array.size else None,
        "constant": bool(array.size and np.all(array == array[0])),
        "invalid": len(values) - len(numeric_values),
        "strings": sum(isinstance(value, str) for value in values),
        "outside_screen_range": int(
            np.sum(
                (array < 0) |
                (array > max(screen_width or 0, screen_height or 0))
            )
        ) if array.size else 0,
    }

def summarize_batch_input(iris_data, screen_width=None, screen_height=None):
    fields = ["left_iris_x", "right_iris_x", "left_iris_y", "right_iris_y"]
    timestamps = [
        sample.get("timestamp") for sample in iris_data
        if isinstance(sample, dict)
        and isinstance(sample.get("timestamp"), (int, float))
        and np.isfinite(sample.get("timestamp"))
    ]
    intervals = np.diff(timestamps).tolist() if len(timestamps) > 1 else []
    return {
        "sample_count": len(iris_data),
        "invalid_samples": sum(
            not isinstance(sample, dict) or any(
                not isinstance(sample.get(field), (int, float))
                or not np.isfinite(sample.get(field))
                for field in fields
            )
            for sample in iris_data
        ),
        "timestamp_range": {
            "min": min(timestamps) if timestamps else None,
            "max": max(timestamps) if timestamps else None,
        },
        "timestamp_intervals": summarize_numeric_values(intervals),
        "fields": {
            field: summarize_numeric_values(
                [sample.get(field) if isinstance(sample, dict) else None for sample in iris_data],
                screen_width,
                screen_height,
            )
            for field in fields
        },
    }


def sanitize_filename(name):
    """Sanitize a user-provided filename to prevent path traversal attacks.

    Strips directory components and allows only alphanumeric characters,
    hyphens, underscores, and dots. Raises ValueError if the result is empty.
    """
    # Take only the final path component to strip any directory traversal
    name = os.path.basename(name)
    # Allow only safe characters
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '', name)
    if not name:
        raise ValueError("Invalid filename")
    return name

# Initialize Flask app
app = Flask(__name__)


# Helper function to convert NaN values to None for JSON serialization
def convert_nan_to_none(obj):
    """
    Recursively converts NaN and Inf values to None for proper JSON serialization.
    
    Args:
        obj: Python object (dict, list, float, etc.)
    
    Returns:
        The object with NaN/Inf values converted to None
    """
    if isinstance(obj, dict):
        return {k: convert_nan_to_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_nan_to_none(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    return obj




def calib_results():
    from_ruxailab = json.loads(request.form['from_ruxailab'])
    user_id = json.loads(
        request.form.get('user_id', 'null')
    )
    study_id = json.loads(
        request.form.get('study_id', 'null')
    )
    fallback_file_name = sanitize_filename(json.loads(request.form['file_name']))
    fixed_points = json.loads(request.form['fixed_circle_iris_points'])
    calib_points = json.loads(request.form['calib_circle_iris_points'])
    screen_height = json.loads(request.form['screen_height'])
    screen_width = json.loads(request.form['screen_width'])
    model = json.loads(request.form.get('model', '"Linear Regression"'))
    k = json.loads(request.form['k'])

    timestamp = int(time.time() * 1000)

    if (
        user_id
        and isinstance(user_id, str)
        and user_id.strip()
        and study_id
        and isinstance(study_id, str)
        and study_id.strip()
    ):
        file_name = sanitize_filename(
            f"{user_id}_{study_id}_{timestamp}"
        )
    else:
        file_name = fallback_file_name

    # Generate csv dataset of calibration points
    os.makedirs(
        f"{Path().absolute()}/app/services/calib_validation/csv/data/", exist_ok=True
    )

    # Generate csv of calibration points with following columns
    calib_csv_file = f"{Path().absolute()}/app/services/calib_validation/csv/data/{file_name}_fixed_train_data.csv"
    csv_columns = [
        "left_iris_x",
        "left_iris_y",
        "right_iris_x",
        "right_iris_y",
        "point_x",
        "point_y",
        "screen_height",
        "screen_width",
    ]

    # Save calibration points to CSV file
    try:
        # Open CSV file
        with open(calib_csv_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

            # Write calibration points to CSV file
            for data in fixed_points:
                data["screen_height"] = screen_height
                data["screen_width"] = screen_width
                writer.writerow(data)

    # Handle I/O error
    except IOError:
        print("I/O error")

    # Generate csv of iris points of session
    os.makedirs(
        f"{Path().absolute()}/app/services/calib_validation/csv/data/", exist_ok=True
    )
    predict_csv_file = f"{Path().absolute()}/app/services/calib_validation/csv/data/{file_name}_predict_train_data.csv"
    csv_columns = ["left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"]
    try:
        with open(predict_csv_file, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in calib_points:
                # print(data)
                writer.writerow(data)
    except IOError:
        print("I/O error")

    # Run prediction
    data = gaze_tracker.predict(calib_csv_file, k, model, model)

    if from_ruxailab:
        try:
            payload = {
                "session_id": file_name,
                "user_id": user_id,
                "study_id": study_id,
                "model": model,
                "screen_height": screen_height,
                "screen_width": screen_width,
                "k": k,
                "fixed_points": fixed_points,
                "calib_points": calib_points,
                "timestamp": timestamp,
            }
            
            FUNCTIONS_ENDPOINT_URL = os.getenv('FUNCTIONS_ENDPOINT_URL')
            FUNCTIONS_ENDPOINT_URL+='/receiveCalibration'    

            print(
                "[calibration] sending calibration",
                {
                    "session_id": file_name,
                    "user_id": user_id,
                    "study_id": study_id,
                    "fixed_points": len(fixed_points),
                    "calib_points": len(calib_points),
                }
            )

            resp = requests.post(FUNCTIONS_ENDPOINT_URL, json=payload)
            print("Enviado para RuxaiLab:", resp.status_code, resp.text)
        except Exception as e:
            print("Erro ao enviar para RuxaiLab:", e)

    # Convert NaN values to None before returning JSON
    data = convert_nan_to_none(data)
    return Response(json.dumps(data), status=200, mimetype='application/json')

def batch_predict():
    try:
        data = request.get_json()

        iris_data = data["iris_tracking_data"]
        calib_points = data["calib_points"]
        fixed_points = data["fixed_points"]

        screen_width = data.get("screen_width")
        screen_height = data.get("screen_height")
        model_name = data.get("model_name", "Linear Regression")
        k = data.get("k", 5)
        calib_id = data.get("calib_id")

        print("[eye-tracking-diagnostic] batch input", {
            "screen_width": screen_width,
            "screen_height": screen_height,
            "calib_id": calib_id,
            "model_name": model_name,
            "k": k,
            "calib_points_count": len(calib_points),
            "fixed_points_count": len(fixed_points),
            **summarize_batch_input(
                iris_data,
                screen_width,
                screen_height,
            ),
        })

        if not calib_points:
            return Response("Missing calib_points", status=400)

        if not fixed_points:
            return Response("Missing fixed_points", status=400)

        # ---------------------------------------------------------
        # Predict live/session gaze data
        # ---------------------------------------------------------
        result = gaze_tracker.predict_new_data_simple(
            fixed_points=fixed_points,
            iris_data=iris_data,
            model_name=model_name,
            screen_width=screen_width,
            screen_height=screen_height,
        )

        predicted_x = [
            item.get("predicted_x")
            for item in result
        ]

        predicted_y = [
            item.get("predicted_y")
            for item in result
        ]
        # ---------------------------------------------------------
        # Predict calibration samples for model evaluation
        # ---------------------------------------------------------

        calibration_result = (
            gaze_tracker.predict_new_data_simple(
                fixed_points=fixed_points,
                iris_data=calib_points,
                model_name=model_name,
                screen_width=screen_width,
                screen_height=screen_height,
            )
        )

        calibration_metrics = calculate_calibration_metrics(
            fixed_points=fixed_points,
            calibration_result=calibration_result,
            calib_points_count=len(calib_points),
        )

        print("[eye-tracking-diagnostic] calibration metrics", {
            "samples": calibration_metrics["samples"],
            "points": calibration_metrics["points"],
            "mean_error_px": calibration_metrics["mean_error_px"],
            "median_error_px": calibration_metrics["median_error_px"],
            "rmse_px": calibration_metrics["rmse_px"],
            "horizontal_mae_px": calibration_metrics[
                "horizontal_mae_px"
            ],
            "vertical_mae_px": calibration_metrics[
                "vertical_mae_px"
            ],
            "accuracy_50px": calibration_metrics[
                "accuracy_50px"
            ],
            "accuracy_100px": calibration_metrics[
                "accuracy_100px"
            ],
            "quality": calibration_metrics["quality"],
        })


        print("[eye-tracking-diagnostic] batch response", {
            "prediction_count": len(result),
            "predicted_x": summarize_numeric_values(predicted_x),
            "predicted_y": summarize_numeric_values(predicted_y),
            "screen_width": screen_width,
            "screen_height": screen_height,
        })

        return jsonify(
            convert_nan_to_none({
                "predictions": result,
                "metrics": {
                    "calibration": calibration_metrics,
                    "session": {
                        "prediction_count": len(result),
                    },
                },
            })
        )

    except Exception as e:
        print("Erro batch_predict:", e)
        traceback.print_exc()
        return Response("Erro interno", status=500)
    
def calculate_calibration_metrics(
    fixed_points,
    calibration_result,
    calib_points_count,
):
    """
    Calculate calibration performance metrics by comparing predictions
    from calib_points against their corresponding fixed calibration targets.

    calib_points are expected to be ordered by target point, with the
    same number of samples collected for each calibration target.
    """

    if not fixed_points or not calibration_result:
        return {
            "samples": 0,
            "points": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "rmse_px": None,
            "horizontal_mae_px": None,
            "vertical_mae_px": None,
            "accuracy_50px": None,
            "accuracy_100px": None,
            "quality": "Unknown",
            "points_detail": [],
        }

    # ---------------------------------------------------------
    # Get unique calibration targets preserving their order.
    # ---------------------------------------------------------

    calibration_targets = []
    seen_targets = set()

    for point in fixed_points:
        point_x = point.get("point_x")
        point_y = point.get("point_y")

        if point_x is None or point_y is None:
            continue

        target = (
            float(point_x),
            float(point_y),
        )

        if target not in seen_targets:
            seen_targets.add(target)
            calibration_targets.append(target)

    if not calibration_targets:
        return {
            "samples": 0,
            "points": 0,
            "mean_error_px": None,
            "median_error_px": None,
            "rmse_px": None,
            "horizontal_mae_px": None,
            "vertical_mae_px": None,
            "accuracy_50px": None,
            "accuracy_100px": None,
            "quality": "Unknown",
            "points_detail": [],
        }

    # ---------------------------------------------------------
    # calib_points are ordered by target point.
    # ---------------------------------------------------------

    total_predictions = min(
        len(calibration_result),
        calib_points_count,
    )

    point_count = len(calibration_targets)
    samples_per_point = total_predictions // point_count

    if samples_per_point <= 0:
        return {
            "samples": 0,
            "points": point_count,
            "mean_error_px": None,
            "median_error_px": None,
            "rmse_px": None,
            "horizontal_mae_px": None,
            "vertical_mae_px": None,
            "accuracy_50px": None,
            "accuracy_100px": None,
            "quality": "Unknown",
            "points_detail": [],
        }

    errors = []
    horizontal_errors = []
    vertical_errors = []

    points_detail = []

    # ---------------------------------------------------------
    # Evaluate each calibration point independently.
    # ---------------------------------------------------------

    for point_index, (target_x, target_y) in enumerate(
        calibration_targets
    ):
        start_index = point_index * samples_per_point
        end_index = start_index + samples_per_point

        point_predictions = calibration_result[
            start_index:end_index
        ]

        point_errors = []
        point_horizontal_errors = []
        point_vertical_errors = []

        for prediction in point_predictions:
            predicted_x = prediction.get("predicted_x")
            predicted_y = prediction.get("predicted_y")

            if (
                predicted_x is None
                or predicted_y is None
            ):
                continue

            try:
                predicted_x = float(predicted_x)
                predicted_y = float(predicted_y)

                horizontal_error = abs(
                    predicted_x - target_x
                )

                vertical_error = abs(
                    predicted_y - target_y
                )

                euclidean_error = (
                    (predicted_x - target_x) ** 2
                    + (predicted_y - target_y) ** 2
                ) ** 0.5

                horizontal_errors.append(horizontal_error)
                vertical_errors.append(vertical_error)
                errors.append(euclidean_error)

                point_horizontal_errors.append(
                    horizontal_error
                )
                point_vertical_errors.append(
                    vertical_error
                )
                point_errors.append(euclidean_error)

            except (TypeError, ValueError):
                continue

        # Use the existing numeric summarizer instead of manually
        # calculating min/max/mean/std for each error collection.
        point_error_stats = summarize_numeric_values(
            point_errors
        )

        point_accuracy_50 = (
            np.mean(
                np.asarray(point_errors) <= 50
            ) * 100
            if point_errors
            else None
        )

        point_accuracy_100 = (
            np.mean(
                np.asarray(point_errors) <= 100
            ) * 100
            if point_errors
            else None
        )

        points_detail.append({
            "target_x": target_x,
            "target_y": target_y,
            "samples": len(point_errors),
            "mean_error_px": (
                round(point_error_stats["mean"], 2)
                if point_error_stats["mean"] is not None
                else None
            ),
            "std_error_px": (
                round(point_error_stats["std"], 2)
                if point_error_stats["std"] is not None
                else None
            ),
            "accuracy_50px": (
                round(float(point_accuracy_50), 2)
                if point_accuracy_50 is not None
                else None
            ),
            "accuracy_100px": (
                round(float(point_accuracy_100), 2)
                if point_accuracy_100 is not None
                else None
            ),
        })

    # ---------------------------------------------------------
    # Global metrics
    # ---------------------------------------------------------

    error_stats = summarize_numeric_values(errors)
    horizontal_stats = summarize_numeric_values(
        horizontal_errors
    )
    vertical_stats = summarize_numeric_values(
        vertical_errors
    )

    if not errors:
        return {
            "samples": 0,
            "points": point_count,
            "mean_error_px": None,
            "median_error_px": None,
            "rmse_px": None,
            "horizontal_mae_px": None,
            "vertical_mae_px": None,
            "accuracy_50px": None,
            "accuracy_100px": None,
            "quality": "Unknown",
            "points_detail": points_detail,
        }

    errors_array = np.asarray(errors, dtype=float)

    median_error = float(np.median(errors_array))

    rmse = float(
        np.sqrt(np.mean(np.square(errors_array)))
    )

    accuracy_50 = float(
        np.mean(errors_array <= 50) * 100
    )

    accuracy_100 = float(
        np.mean(errors_array <= 100) * 100
    )

    mean_error = error_stats["mean"]

    # ---------------------------------------------------------
    # Calibration quality
    # ---------------------------------------------------------

    if mean_error <= 40:
        quality = "Excellent"
    elif mean_error <= 75:
        quality = "Good"
    elif mean_error <= 120:
        quality = "Fair"
    else:
        quality = "Poor"

    return {
        "samples": len(errors),
        "points": point_count,
        "mean_error_px": round(mean_error, 2),
        "median_error_px": round(median_error, 2),
        "rmse_px": round(rmse, 2),
        "horizontal_mae_px": round(
            horizontal_stats["mean"], 2
        ),
        "vertical_mae_px": round(
            vertical_stats["mean"], 2
        ),
        "accuracy_50px": round(accuracy_50, 2),
        "accuracy_100px": round(accuracy_100, 2),
        "quality": quality,
        "points_detail": points_detail,
    }