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
    user_id = json.loads(request.form['user_id'])
    study_id = json.loads(request.form['study_id'])
    file_name = sanitize_filename(json.loads(request.form['file_name']))
    fixed_points = json.loads(request.form['fixed_circle_iris_points'])
    calib_points = json.loads(request.form['calib_circle_iris_points'])
    screen_height = json.loads(request.form['screen_height'])
    screen_width = json.loads(request.form['screen_width'])
    model = json.loads(request.form.get('model', '"Linear Regression"'))
    k = json.loads(request.form['k'])

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
                "k": k
            }
            
            FUNCTIONS_ENDPOINT_URL = os.getenv('FUNCTIONS_ENDPOINT_URL')
            FUNCTIONS_ENDPOINT_URL+='/receiveCalibration'    

            print("file_name:", file_name)

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
        screen_width = data.get("screen_width")
        screen_height = data.get("screen_height")
        model_name = data.get("model_name", "Linear Regression")
        calib_id = data.get("calib_id")

        print("[eye-tracking-diagnostic] batch input", {
            "screen_width": screen_width,
            "screen_height": screen_height,
            "calib_id": calib_id,
            "model_name": model_name,
            **summarize_batch_input(iris_data, screen_width, screen_height),
        })

        if not calib_id:
            return Response("Missing calib_id", status=400)

        calib_id = sanitize_filename(calib_id)

        base_path = Path().absolute() / "app/services/calib_validation/csv/data"
        calib_csv_path = base_path / f"{calib_id}_fixed_train_data.csv"
        predict_csv_path = base_path / f"temp_batch_predict_{uuid.uuid4().hex}.csv"

        print("[eye-tracking-diagnostic] calibration artifact", {
            "path": str(calib_csv_path),
            "exists": calib_csv_path.exists(),
        })

        # CSV temporário
        with open(predict_csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"
            ])
            writer.writeheader()
            for item in iris_data:
                writer.writerow({
                    "left_iris_x": item["left_iris_x"],
                    "left_iris_y": item["left_iris_y"],
                    "right_iris_x": item["right_iris_x"],
                    "right_iris_y": item["right_iris_y"],
                })

        result = gaze_tracker.predict_new_data_simple(
            calib_csv_path=calib_csv_path,
            predict_csv_path=predict_csv_path,
            iris_data=iris_data,
            model_name=model_name,
            screen_width=screen_width,
            screen_height=screen_height,
        )
        os.remove(predict_csv_path)

        predicted_x = [item.get("predicted_x") for item in result]
        predicted_y = [item.get("predicted_y") for item in result]
        print("[eye-tracking-diagnostic] batch response", {
            "prediction_count": len(result),
            "predicted_x": summarize_numeric_values(predicted_x),
            "predicted_y": summarize_numeric_values(predicted_y),
            "screen_width": screen_width,
            "screen_height": screen_height,
        })
        
        return jsonify(convert_nan_to_none(result))

    except Exception as e:
        print("Erro batch_predict:", e)
        traceback.print_exc()
        return Response("Erro interno", status=500)