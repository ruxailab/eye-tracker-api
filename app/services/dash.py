# Necessary imports
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# Sklearn imports
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
    max_error,
)


# ---------------------------------------------------------------------------
# Data directory — resolved relative to this file so it works on every OS
# and regardless of the working directory from which streamlit is launched.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
data_dir = BASE_DIR / "calib_validation" / "csv" / "data"

# Guard against missing data folder with a clear, actionable error message
if not data_dir.exists():
    st.error(
        f"Data directory not found: `{data_dir}`\n\n"
        "Please make sure the `calib_validation/csv/data` folder exists "
        "and contains calibration CSV files."
    )
    st.stop()

# Get all the files in the data directory
files = os.listdir(data_dir)

# Extract the prefixes from the file names
prefixes = [
    file.split("_fixed_train_data.csv")[0]
    for file in files
    if file.endswith("_fixed_train_data.csv")
]

# Set the page configuration for the Streamlit app and set the title
st.set_page_config(page_title="Streamlit Dashboard📊", layout="wide")
st.title("Streamlit Dashboard📊")

# Prefix for the calibration data to identify the correct file
st.subheader("Select from your collected data")
prefix = st.selectbox("Select the prefix for the calibration data", prefixes)

# Load the dataset — path built with pathlib for cross-platform compatibility
dataset_train_path = data_dir / f"{prefix}_fixed_train_data.csv"
try:
    raw_dataset = pd.read_csv(dataset_train_path)
# File not found error handling
except FileNotFoundError:
    st.error("File not found. Please make sure the file path is correct.")
    st.stop()
else:
    st.success("Data loaded successfully!")


def evaluate_models(X, Y, axis_label, models, model_names):
    """
    Trains multiple regression models to predict gaze coordinates along one axis
    and displays a suite of comparison charts via Streamlit.

    Replaces the previous ``model_for_mouse_x`` / ``model_for_mouse_y`` pair,
    which were near-identical (~120 duplicated lines).  A single generic
    function now handles both axes, reducing duplication and making future
    maintenance easier.

    Args:
        - X (array-like): The input features (iris coordinates).
        - Y (array-like): The target variable (screen coordinate for this axis).
        - axis_label (str): Human-readable axis identifier, e.g. ``"X"`` or ``"Y"``.
        - models (list): A list of machine learning models to be trained.
        - model_names (list): A list of model names corresponding to *models*.

    Returns: None
    """
    # Split dataset into train and test sets (80/20; fixed seed for reproducibility)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    metrics_list = []

    for model, model_name in zip(models, model_names):
        # Train the model
        model.fit(X_train, Y_train)

        # Predict the target variable for the test set
        Y_pred_test = model.predict(X_test)

        # Filter out negative predicted values
        non_negative_indices = Y_pred_test >= 0
        Y_pred_test_filtered = Y_pred_test[non_negative_indices]
        Y_test_filtered = Y_test[non_negative_indices]

        # Compute metrics for the test set with filtered predictions
        metrics_data_test = {
            "Model": model_name,
            "Mean Absolute Error (MAE)": mean_absolute_error(
                Y_test_filtered, Y_pred_test_filtered
            ),
            "Median Absolute Error": median_absolute_error(
                Y_test_filtered, Y_pred_test_filtered
            ),
            "Mean Squared Error (MSE)": mean_squared_error(
                Y_test_filtered, Y_pred_test_filtered
            ),
            "Mean Log Squared Error (MSLE)": mean_squared_log_error(
                Y_test_filtered, Y_pred_test_filtered
            ),
            "Root Mean Squared Error (RMSE)": np.sqrt(
                mean_squared_error(Y_test_filtered, Y_pred_test_filtered)
            ),
            "Explained Variance Score": explained_variance_score(
                Y_test_filtered, Y_pred_test_filtered
            ),
            "Max Error": max_error(Y_test_filtered, Y_pred_test_filtered),
            f"MODEL {axis_label} SCORE R2": r2_score(
                Y_test_filtered, Y_pred_test_filtered
            ),
        }

        metrics_list.append(metrics_data_test)

    # Convert metrics data to DataFrame
    metrics_df_test = pd.DataFrame(metrics_list)

    # Display metrics using Streamlit
    st.subheader(f"Metrics for the test set - {axis_label}")
    st.dataframe(metrics_df_test, use_container_width=True)

    # Bar charts for visualization
    for metric in metrics_df_test.columns[1:]:
        st.subheader(f"Comparison of {metric}")
        fig = px.bar(metrics_df_test.set_index("Model"), y=metric)
        st.plotly_chart(fig)

    # Line chart for visualizing the metrics
    st.subheader("Line Chart Comparison")
    fig = px.line(metrics_df_test.set_index("Model"))
    st.plotly_chart(fig)

    # Box plot for distribution of errors
    st.subheader("Box Plot of Model Errors")
    errors_df = pd.DataFrame(
        {
            "Model": np.repeat(model_names, len(Y_test)),
            "Actual": np.tile(Y_test, len(models)),
            "Predicted": np.concatenate([model.predict(X_test) for model in models]),
        }
    )
    errors_df["Error"] = errors_df["Actual"] - errors_df["Predicted"]

    # Create the box plot
    st.dataframe(errors_df, use_container_width=True)
    fig = px.box(errors_df, x="Model", y="Error")
    st.plotly_chart(fig)

    # Radar chart for model comparison
    st.subheader("Radar Chart Comparison")

    # Normalize the metric values for better comparison
    metrics_normalized = metrics_df_test.copy()
    for col in metrics_normalized.columns[1:]:
        col_min = metrics_normalized[col].min()
        col_max = metrics_normalized[col].max()
        denom = col_max - col_min
        metrics_normalized[col] = (
            (metrics_normalized[col] - col_min) / denom if denom != 0 else 0
        )

    # Create the radar chart
    fig = go.Figure()
    for i in range(len(models)):
        fig.add_trace(
            go.Scatterpolar(
                r=metrics_normalized.iloc[i, 1:].values,
                theta=metrics_normalized.columns[1:],
                fill="toself",
                name=metrics_normalized.iloc[i, 0],
            )
        )

    # Update the layout
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True
    )

    # Display the radar chart
    st.plotly_chart(fig)


# ---------------------------------------------------------------------------
# Main dashboard — tabs for raw data and model metrics
# ---------------------------------------------------------------------------

# Set the title of the app and the tabs
st.subheader("Eye Tracker Calibration Data Analysis and Prediction")
st.write(f"Select the tab to view the data and metrics for [{prefix}] data")
tab1, tab2 = st.tabs(["Raw Data", "Metrics"])

# With the first tab
with tab1:
    # Display the raw dataset
    st.subheader("Data Obtained from Calibration")
    st.dataframe(raw_dataset, use_container_width=True)

    # Two columns for the plots
    col1, col2 = st.columns(2)

    with col1:
        # Subheader
        st.subheader("Left Eye")
        df = raw_dataset

        # Create the scatter plot
        fig_left = px.scatter(
            df,
            x="left_iris_x",
            y="left_iris_y",
            color="left_iris_y",
            color_continuous_scale="reds",
        )

        # Display the plot
        st.plotly_chart(fig_left, theme="streamlit", use_container_width=True)

    with col2:
        # Subheader
        st.subheader("Right Eye")

        # Create the scatter plot
        fig_right = px.scatter(
            df,
            x="right_iris_x",
            y="right_iris_y",
            color="right_iris_y",
            color_continuous_scale="reds",
        )

        # Display the plot
        st.plotly_chart(fig_right, theme="streamlit", use_container_width=True)

    # Create the line plot
    fig3 = px.line(
        raw_dataset,
        y=["left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"],
        title="Left and Right Iris Position",
    )
    # Display the plot
    st.plotly_chart(fig3, theme="streamlit", use_container_width=True)


# With the second tab
with tab2:
    st.subheader("Model Performance Comparison")
    # Create a list of models to be trained
    models = [
        make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression()),
        make_pipeline(PolynomialFeatures(2), linear_model.Lasso(alpha=0.1)),
        make_pipeline(PolynomialFeatures(2), linear_model.Ridge(alpha=0.5)),
        make_pipeline(
            PolynomialFeatures(2), linear_model.ElasticNet(alpha=1.0, l1_ratio=0.5)
        ),
        make_pipeline(PolynomialFeatures(2), linear_model.BayesianRidge()),
        make_pipeline(
            PolynomialFeatures(2),
            linear_model.SGDRegressor(random_state=42, penalty="elasticnet"),
        ),
        make_pipeline(PolynomialFeatures(2), SVR(kernel="linear")),
    ]
    model_names = [
        "Linear Regression",
        "Lasso Regression",
        "Ridge Regression",
        "Elastic Net",
        "Bayesian Ridge",
        "SGD Regressor",
        "Support Vector Regressor",
    ]

    # Drop the columns that are not needed
    X = raw_dataset.drop(["screen_height", "screen_width"], axis=1)

    # Split the dataset into input features and target variables
    X1 = X[["left_iris_x", "right_iris_x"]]
    X2 = X[["left_iris_y", "right_iris_y"]]

    # Standardize the input features
    sc = StandardScaler()
    X1 = sc.fit_transform(X1)
    X2 = sc.fit_transform(X2)

    # Target variables
    Y1 = raw_dataset.point_x
    Y2 = raw_dataset.point_y

    # Train and evaluate models for both axes using the unified function
    evaluate_models(X1, Y1, "X", models, model_names)
    evaluate_models(X2, Y2, "Y", models, model_names)
