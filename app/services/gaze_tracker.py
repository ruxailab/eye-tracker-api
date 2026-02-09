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
from sklearn.pipeline import make_pipeline

# Model imports
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

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
)
from app.services.config import hyperparameters


# Machine learning models to use
models = {
    "Linear Regression": make_pipeline(
        PolynomialFeatures(2), linear_model.LinearRegression()
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

# Set the scoring metrics for GridSearchCV to r2_score and mean_absolute_error
scoring = {
    "r2": make_scorer(r2_score),
    "mae": make_scorer(mean_absolute_error),
}


def squash(v, limit=1.0):
    """Squash não-linear estilo WebGazer"""
    return np.tanh(v / limit)

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
    # Inicialize standard scaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    # Load data from csv file and drop unnecessary columns
    df = pd.read_csv(data)
    df = df.drop(["screen_height", "screen_width"], axis=1)
    
    # Drop rows with NaN values to prevent sklearn errors
    df = df.dropna()

    # Create a stratification key based on (point_x, point_y) to ensure all calibration
    # points are represented in both training and test sets
    df["stratify_key"] = df["point_x"].astype(str) + "_" + df["point_y"].astype(str)
    
    # Perform a single stratified split to ensure all points are in both train and test sets
    train_indices, test_indices = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=df["stratify_key"]
    )
    
    df_train = df.loc[train_indices]
    df_test = df.loc[test_indices]

    # Data for X axis
    X_train_x = sc_x.fit_transform(df_train[["left_iris_x", "right_iris_x"]])
    X_test_x = sc_x.transform(df_test[["left_iris_x", "right_iris_x"]])
    y_train_x = df_train["point_x"]
    y_test_x = df_test["point_x"]

    if (
        model_X == "Linear Regression"
        or model_X == "Elastic Net"
        or model_X == "Support Vector Regressor"
    ):
        model = models[model_X]

        # Fit the model and make predictions
        model.fit(X_train_x, y_train_x)
        y_pred_x = model.predict(X_test_x)

    else:
        pipeline = models[model_X]
        param_grid = hyperparameters[model_X]["param_grid"]

        # Initialize GridSearchCV with the pipeline and parameter grid
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring=scoring,
            refit="r2",
            return_train_score=True,
        )

        # Fit the GridSearchCV to the training data for X
        grid_search.fit(X_train_x, y_train_x)

        # Use the best estimator to predict the values and calculate the R2 score
        best_model_x = grid_search.best_estimator_
        y_pred_x = best_model_x.predict(X_test_x)

    # Data for Y axis (use same train/test split as X for consistency)
    X_train_y = sc_y.fit_transform(df_train[["left_iris_y", "right_iris_y"]])
    X_test_y = sc_y.transform(df_test[["left_iris_y", "right_iris_y"]])
    y_train_y = df_train["point_y"]
    y_test_y = df_test["point_y"]

    if (
        model_Y == "Linear Regression"
        or model_Y == "Elastic Net"
        or model_Y == "Support Vector Regressor"
    ):
        model = models[model_Y]

        # Fit the model and make predictions
        model.fit(X_train_y, y_train_y)
        y_pred_y = model.predict(X_test_y)

    else:
        pipeline = models[model_Y]
        param_grid = hyperparameters[model_Y]["param_grid"]

        # Initialize GridSearchCV with the pipeline and parameter grid
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring=scoring,
            refit="r2",
            return_train_score=True,
        )

        # Fit the GridSearchCV to the training data for X
        grid_search.fit(X_train_y, y_train_y)

        # Use the best estimator to predict the values and calculate the R2 score
        best_model_y = grid_search.best_estimator_
        y_pred_y = best_model_y.predict(X_test_y)

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

    # Calculate the average precision and accuracy
    precision_xy = (precision_x + precision_y) / 2
    precision_xy = precision_xy / np.mean(precision_xy)

    # Calculate the accuracy for each axis
    accuracy_x = df_data.groupby("True XY").apply(func_accuracy_x)
    accuracy_y = df_data.groupby("True XY").apply(func_accuracy_y)

    # Calculate the average accuracy
    accuracy_xy = (accuracy_x + accuracy_y) / 2
    accuracy_xy = accuracy_xy / np.mean(accuracy_xy)

    # Create a dictionary to store the data
    data = {}

    # Iterate over the dataframe and store the data
    for index, row in df_data.iterrows():

        # Get the outer and inner keys
        outer_key = str(row["True X"]).split(".")[0]
        inner_key = str(row["True Y"]).split(".")[0]

        # If the outer key is not in the dictionary, add it
        if outer_key not in data:
            data[outer_key] = {}

        # Add the data to the dictionary
        data[outer_key][inner_key] = {
            "predicted_x": df_data[
                (df_data["True X"] == row["True X"])
                & (df_data["True Y"] == row["True Y"])
            ]["Predicted X"].values.tolist(),
            "predicted_y": df_data[
                (df_data["True X"] == row["True X"])
                & (df_data["True Y"] == row["True Y"])
            ]["Predicted Y"].values.tolist(),
            "PrecisionSD": precision_xy[(row["True X"], row["True Y"])],
            "Accuracy": accuracy_xy[(row["True X"], row["True Y"])],
        }

    # Centroids of the clusters
    data["centroids"] = model.cluster_centers_.tolist()

    # Return the data
    return data


def predict_new_data_simple(
    calib_csv_path,
    predict_csv_path,
    iris_data,
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
    model_x = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model_y = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

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
