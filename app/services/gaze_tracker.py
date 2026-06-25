# Necessary imports
import math
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import time

# Model imports
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt

# Metrics imports
from sklearn.metrics import make_scorer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
)

# Local imports
from app.services.metrics import (
    func_precision_x,
    func_presicion_y,
    func_accuracy_x,
    func_accuracy_y,
    func_total_accuracy,
)
from app.services.config import hyperparameters


# Machine learning models to use
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(2), 
        linear_model.LinearRegression()
    ),
    "Ridge Regression": make_pipeline(PolynomialFeatures(2), linear_model.Ridge()),
    "Lasso Regression": make_pipeline(PolynomialFeatures(2), linear_model.Lasso()),
    "Elastic Net": make_pipeline(
        PolynomialFeatures(2), linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
    ),
    "Bayesian Ridge": make_pipeline(
        PolynomialFeatures(2), linear_model.BayesianRidge()
    ),
    "SGD Regressor": make_pipeline(PolynomialFeatures(2), linear_model.SGDRegressor()),
    "Support Vector Regressor": make_pipeline(
        PolynomialFeatures(2), SVR(kernel="linear")
    ),
    "Random Forest Regressor": make_pipeline(
    RandomForestRegressor(
        n_estimators=200, 
        max_depth=10, 
        min_samples_split=5,
        random_state=42
    )
)}

models_gaze_engineered = {
    "Linear Regression": make_pipeline(
        StandardScaler(),
        PolynomialFeatures(2),
        linear_model.LinearRegression()
    ),
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        linear_model.Ridge()
    ),
    "Lasso Regression": make_pipeline(
        StandardScaler(),
        linear_model.Lasso()
    ),
    "Elastic Net": make_pipeline(
        StandardScaler(),
        linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
    ),
    "Bayesian Ridge": make_pipeline(
        StandardScaler(),
        linear_model.BayesianRidge()
    ),
    "SGD Regressor": make_pipeline(
        StandardScaler(),
        linear_model.SGDRegressor()
    ),
    "Support Vector Regressor": make_pipeline(
        StandardScaler(),
        SVR(kernel="linear")
    ),
    "Random Forest Regressor": make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
    )
}

# Set the scoring metrics for GridSearchCV to r2_score and mean_absolute_error
scoring = {
    "r2": make_scorer(r2_score),
    "mae": make_scorer(mean_absolute_error),
}

def squash(v, limit=1.0):
    """Squash não-linear estilo WebGazer"""
    return np.tanh(v / limit)

def train_and_predict(model_name, X_train, y_train, X_test, y_test, label):
    """
    Helper to train a model (with or without GridSearchCV) and return predictions.
    """
    if model_name == "Linear Regression":
        model = models[model_name]
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        y_pred = model.predict(X_test)
        print(f"Score {label}: {r2_score(y_test, y_pred)}")
        print(f"Time {label}: {end_time - start_time}")
        return y_pred
    else:
        pipeline = models[model_name]
        param_grid = hyperparameters[model_name]["param_grid"]
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring=scoring,
            refit="r2",
            return_train_score=True,
        )
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        print(f"Time {label}: {end_time - start_time}")
        return y_pred


def predict(data, k, model_X, model_Y):
    """
    Predicts the gaze coordinates using machine learning models.

    Args:
        - data (str): The path to the CSV file containing the training data.
        - k (int): The number of clusters for KMeans clustering.
        - model_X: The machine learning model to use for prediction on the X coordinate.
        - model_Y: The machine learning model to use for prediction on the Y coordinate.

    Returns:
        dict: A dictionary containing the predicted gaze coordinates, precision, accuracy, and cluster centroids.
    """


    # Load data from csv file and drop unnecessary columns
    df = pd.read_csv(data)
    df = df.drop(["screen_height", "screen_width"], axis=1)
    print(df.head())
    # Create groups (point_x, point_y)
    df["group"] = list(zip(df["point_x"], df["point_y"]))

    # Data for X axis
    X_x = df[["left_iris_x", "right_iris_x"]]
    X_y = df["point_x"]
    # groups = df["group"]
    # Data for Y axis
    X_feature_y = df[["left_iris_y", "right_iris_y"]]
    y_y = df["point_y"]
    # Split data into training and testing sets then Normalize data using standard scaler
    (
        X_train_x, X_test_x,
        y_train_x, y_test_x,
        X_train_y, X_test_y,
        y_train_y, y_test_y
    )= train_test_split(
        X_x,
        X_y,
        X_feature_y,
        y_y,
        test_size=0.2,
        random_state=42,
        stratify=df["group"]
    )
    
    # Scaling (fit on train only)
    scaler_x = StandardScaler()
    X_train_x = scaler_x.fit_transform(X_train_x)
    X_test_x  = scaler_x.transform(X_test_x)
    
    y_pred_x = train_and_predict(model_X, X_train_x, y_train_x, X_test_x, y_test_x, "X")
    
    # Scaling (fit on train only)
    scaler_y = StandardScaler()
    X_train_y = scaler_y.fit_transform(X_train_y)
    X_test_y  = scaler_y.transform(X_test_y)

    
    y_pred_y = train_and_predict(model_Y, X_train_y, y_train_y, X_test_y, y_test_y, "Y")
    
    # Convert the predictions to a numpy array and apply KMeans clustering
    data = np.array([y_pred_x, y_pred_y]).T
    model = KMeans(n_clusters=k, n_init="auto", init="k-means++")
    y_kmeans = model.fit_predict(data)

    # Create a dataframe with the truth and predicted values
    data = {
        "True X": y_test_x,
        "Predicted X": y_pred_x,
        "True Y": y_test_y,
        "Predicted Y": y_pred_y,
    }
    df_data = pd.DataFrame(data)
    df_data["True XY"] = list(zip(df_data["True X"], df_data["True Y"]))
    
    # Filter out negative values
    df_data = df_data[(df_data["Predicted X"] >= 0) & (df_data["Predicted Y"] >= 0)]

    # Calculate the precision and accuracy for each 
    precision_x = df_data.groupby("True XY").apply(func_precision_x)
    precision_y = df_data.groupby("True XY").apply(func_presicion_y)

    # Calculate the average precision 
    precision_xy = (precision_x + precision_y) / 2
    
    # Calculate the average accuracy (eculidian distance)
    accuracy_xy = df_data.groupby("True XY").apply(func_total_accuracy)
    
    # Create a dictionary to store the data
    data = {}
    grouped = df_data.groupby("True XY")

    for (true_x, true_y), group in grouped:

        # keys
        outer_key = str(true_x).split(".")[0]
        inner_key = str(true_y).split(".")[0]

        # create outer key if missing
        if outer_key not in data:
            data[outer_key] = {}

        # fill data
        data[outer_key][inner_key] = {
            "predicted_x": group["Predicted X"].tolist(),
            "predicted_y": group["Predicted Y"].tolist(),
            "PrecisionSD": precision_xy[(true_x, true_y)],
            "Accuracy": accuracy_xy[(true_x, true_y)],
        }
    # Centroids of the clusters
    data["centroids"] = model.cluster_centers_.tolist()

    # Return the data
    return data

def predict_new_data_simple(
    calib_csv_path,
    predict_csv_path,
    iris_data,
    model_name="Linear Regression",
    screen_width=None,
    screen_height=None,
):  
    # ============================
    # CONFIG (WebGazer-inspired)
    # ============================
    BASELINE_ALPHA = 0.01
    SQUASH_LIMIT_X = 1.0
    SQUASH_LIMIT_Y = 1.0
    Y_GAIN = 1.2  # adjustment to compensate for vertical bias

    # ============================
    # LOAD TRAIN
    # ============================
    df_train = pd.read_csv(calib_csv_path)

    x_center = screen_width / 2
    y_center = screen_height / 2

    # normalize targets to [-1, 1] space
    y_train_x = (df_train["point_x"].values.astype(float) - x_center) / (screen_width / 2)
    y_train_y = (df_train["point_y"].values.astype(float) - y_center) / (screen_height / 2)

    # ensure laterality
    if df_train["left_iris_x"].mean() < df_train["right_iris_x"].mean():
        df_train["left_iris_x"], df_train["right_iris_x"] = (
            df_train["right_iris_x"].copy(),
            df_train["left_iris_x"].copy(),
        )
    if df_train["left_iris_y"].mean() < df_train["right_iris_y"].mean():
        df_train["left_iris_y"], df_train["right_iris_y"] = (
            df_train["right_iris_y"].copy(),
            df_train["left_iris_y"].copy(),
        )

    left_x = df_train["left_iris_x"].values.astype(float)
    right_x = df_train["right_iris_x"].values.astype(float)
    left_y = df_train["left_iris_y"].values.astype(float)
    right_y = df_train["right_iris_y"].values.astype(float)

    mean_x = (left_x + right_x) / 2
    diff_x = left_x - right_x
    mean_y = (left_y + right_y) / 2
    diff_y = left_y - right_y

    # baseline inicial (WebGazer)
    ref_mean_x = np.mean(mean_x)
    ref_mean_y = np.mean(mean_y)

    rel_x = mean_x - ref_mean_x
    rel_y = mean_y - ref_mean_y

    # ============================
    # PHYSICAL NORMALIZATION Y
    # ============================
    iris_y_scale = np.std(mean_y) + 1e-6
    diff_y_norm = diff_y / iris_y_scale
    rel_y_norm = rel_y / iris_y_scale

    # ============================
    # FEATURES
    # ============================
    X_train_x = np.column_stack([
        left_x, right_x, mean_x, diff_x, rel_x
    ])

    X_train_y = np.column_stack([
        diff_y_norm, rel_y_norm
    ])

    # ============================
    # MODELS
    # ============================

    model_x=models_gaze_engineered.get(model_name,models_gaze_engineered['Linear Regression'])
    model_y=models.get(model_name,models['Linear Regression'])

    model_x.fit(X_train_x, y_train_x)
    model_y.fit(X_train_y, y_train_y)

    # ============================
    # Real scale (calibration) - normalize predicted values to screen coordinates
    # ============================
    x_range = np.percentile(y_train_x, 95) - np.percentile(y_train_x, 5)
    y_range = np.percentile(y_train_y, 95) - np.percentile(y_train_y, 5)

    x_scale = max(x_range / 2, 1e-6) * (screen_width / 2)
    y_scale = max(y_range / 2, 1e-6) * (screen_height / 2)

    # ============================
    # LOAD PREDICT
    # ============================
    df_pred = pd.read_csv(predict_csv_path)

    if df_pred["left_iris_x"].mean() < df_pred["right_iris_x"].mean():
        df_pred["left_iris_x"], df_pred["right_iris_x"] = (
            df_pred["right_iris_x"].copy(),
            df_pred["left_iris_x"].copy(),
        )
    if df_pred["left_iris_y"].mean() < df_pred["right_iris_y"].mean():
        df_pred["left_iris_y"], df_pred["right_iris_y"] = (
            df_pred["right_iris_y"].copy(),
            df_pred["left_iris_y"].copy(),
        )

    left_px = df_pred["left_iris_x"].values.astype(float)
    right_px = df_pred["right_iris_x"].values.astype(float)
    left_py = df_pred["left_iris_y"].values.astype(float)
    right_py = df_pred["right_iris_y"].values.astype(float)

    mean_px = (left_px + right_px) / 2
    diff_px = left_px - right_px
    mean_py = (left_py + right_py) / 2
    diff_py = left_py - right_py

    # baseline relativo
    rel_px = mean_px - ref_mean_x
    rel_py = mean_py - ref_mean_y

    diff_py_norm = diff_py / iris_y_scale
    rel_py_norm = rel_py / iris_y_scale

    X_pred_x = np.column_stack([
        left_px, right_px, mean_px, diff_px, rel_px
    ])

    X_pred_y = np.column_stack([
        diff_py_norm, rel_py_norm
    ])

    y_pred_x = model_x.predict(X_pred_x)
    y_pred_y = model_y.predict(X_pred_y)

    # remove bias vertical
    y_pred_y = y_pred_y - np.mean(y_pred_y)
    
    y_pred_y = y_pred_y * Y_GAIN


    # ============================
    # PREDICTION LOOP (WebGazer)
    # ============================
    predictions = []
    window_size = 30


    for i in range(len(y_pred_x)):
        # baseline dinâmico
        ref_mean_x = BASELINE_ALPHA * mean_px[i] + (1 - BASELINE_ALPHA) * ref_mean_x
        ref_mean_y = BASELINE_ALPHA * mean_py[i] + (1 - BASELINE_ALPHA) * ref_mean_y

        # squash não-linear
        sx = squash(y_pred_x[i], SQUASH_LIMIT_X)
        sy = squash(y_pred_y[i], SQUASH_LIMIT_Y)

        px = x_center + float(sx) * x_scale
        py = y_center + float(sy) * y_scale

        predictions.append({
            "timestamp": iris_data[i].get("timestamp"),
            "predicted_x": px,
            "predicted_y": py,
            "screen_width": screen_width,
            "screen_height": screen_height,
        })

    # ============================
    # LOGS
    # ============================
    print("====== MODEL DEBUG ======")
    print(f"y_pred_x: {np.min(y_pred_x):.3f} → {np.max(y_pred_x):.3f}")
    print(f"y_pred_y: {np.min(y_pred_y):.3f} → {np.max(y_pred_y):.3f}")
    print("=========================")

    print("====== PIXEL SAMPLE ======")
    for p in predictions[:15]:
        print(f"x: {p['predicted_x']:.1f}, y: {p['predicted_y']:.1f}")

    return predictions


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
