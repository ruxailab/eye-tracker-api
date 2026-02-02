# Necesary imports
import os
import re
import time
import json
import csv
import math
import numpy as np

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
    file_name = json.loads(request.form['file_name'])
    fixed_points = json.loads(request.form['fixed_circle_iris_points'])
    calib_points = json.loads(request.form['calib_circle_iris_points'])
    screen_height = json.loads(request.form['screen_height'])
    screen_width = json.loads(request.form['screen_width'])
    model_X = json.loads(request.form.get('model', '"Linear Regression"'))
    model_Y = json.loads(request.form.get('model', '"Linear Regression"'))
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
    data = gaze_tracker.predict(calib_csv_file, k, model_X, model_Y)

    if from_ruxailab:
        try:
            payload = {
                "session_id": file_name,
                "model": data,
                "screen_height": screen_height,
                "screen_width": screen_width,
                "k": k
            }

            RUXAILAB_WEBHOOK_URL = "https://receivecalibration-ffptzpxikq-uc.a.run.app"

            print("file_name:", file_name)

            resp = requests.post(RUXAILAB_WEBHOOK_URL, json=payload)
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
        model_X = data.get("model_X", "Linear Regression")
        model_Y = data.get("model_Y", "Linear Regression")
        calib_id = data.get("calib_id")

        if not calib_id:
            return Response("Missing calib_id", status=400)

        base_path = Path().absolute() / "app/services/calib_validation/csv/data"
        calib_csv_path = base_path / f"{calib_id}_fixed_train_data.csv"
        predict_csv_path = base_path / "temp_batch_predict.csv"

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
            # model_X="Random Forest Regressor",
            # model_Y="Random Forest Regressor",
            screen_width=screen_width,
            screen_height=screen_height,
        )

        return jsonify(convert_nan_to_none(result))

    except Exception as e:
        print("Erro batch_predict:", e)
        traceback.print_exc()
        return Response("Erro interno", status=500)