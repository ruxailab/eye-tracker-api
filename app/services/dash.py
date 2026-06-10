# Necessary imports
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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


# Get the data directory (relative to this file's location)
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "calib_validation", "csv", "data")

# Get all files in the data directory
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

# Load the dataset
dataset_train_path = os.path.join(data_dir, f"{prefix}_fixed_train_data.csv")
try:
    raw_dataset = pd.read_csv(dataset_train_path)
    st.success(f"Loaded: {prefix}_fixed_train_data.csv")
# File not found error handling
except FileNotFoundError:
    st.error("File not found. Please make sure the file path is correct.")
    st.stop()


def model_for_mouse_x(X1, Y1, models, model_names):
    """
    Trains multiple models to predict the X coordinate based on the given features and compares their performance.

    Args:
        - X1 (array-like): The input features.
        - Y1 (array-like): The target variable (X coordinate).
        - models (list): A list of machine learning models to be trained.
        - model_names (list): A list of model names corresponding to the models.

    Returns: None
    """
    # Split dataset into train and test sets (80/20 where 20 is for test)
    X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)

    metrics_list = []

    for model, model_name in zip(models, model_names):
        # Train the model
        model.fit(X1_train, Y1_train)

        # Predict the target variable for the test set
        Y1_pred_test = model.predict(X1_test)

        # Filter out the negative predicted values
        non_negative_indices = Y1_pred_test >= 0
        Y1_pred_test_filtered = Y1_pred_test[non_negative_indices]
        Y1_test_filtered = Y1_test[non_negative_indices]

        # Compute the metrics for the test set with filtered predictions
        metrics_data_test = {
            "Model": model_name,
            "Mean Absolute Error (MAE)": mean_absolute_error(
                Y1_test_filtered, Y1_pred_test_filtered
            ),
            "Median Absolute Error": median_absolute_error(
                Y1_test_filtered, Y1_pred_test_filtered
            ),
            "Mean Squared Error (MSE)": mean_squared_error(
                Y1_test_filtered, Y1_pred_test_filtered
            ),
            "Mean Log Squared Error (MSLE)": mean_squared_log_error(
                Y1_test_filtered, Y1_pred_test_filtered
            ),
            "Root Mean Squared Error (RMSE)": np.sqrt(
                mean_squared_error(Y1_test_filtered, Y1_pred_test_filtered)
            ),
            "Explained Variance Score": explained_variance_score(
                Y1_test_filtered, Y1_pred_test_filtered
            ),
            "Max Error": max_error(Y1_test_filtered, Y1_pred_test_filtered),
            "MODEL X SCORE R2": r2_score(Y1_test_filtered, Y1_pred_test_filtered),
        }

        metrics_list.append(metrics_data_test)

    # Convert metrics data to DataFrame
    metrics_df_test = pd.DataFrame(metrics_list)

    # Display metrics using Streamlit
    st.subheader("Metrics for the test set - X")
    st.dataframe(metrics_df_test, width="stretch")

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
            "Model": np.repeat(model_names, len(Y1_test)),
            "Actual": np.tile(Y1_test, len(models)),
            "Predicted": np.concatenate([model.predict(X1_test) for model in models]),
        }
    )
    errors_df["Error"] = errors_df["Actual"] - errors_df["Predicted"]

    # Create the box plot
    st.dataframe(errors_df, width="stretch")
    fig = px.box(errors_df, x="Model", y="Error")
    st.plotly_chart(fig)

    # Radar chart for model comparison
    st.subheader("Radar Chart Comparison")

    # Normalize the metric values for better comparison
    metrics_normalized = metrics_df_test.copy()
    for col in metrics_normalized.columns[1:]:
        metrics_normalized[col] = (
            metrics_normalized[col] - metrics_normalized[col].min()
        ) / (metrics_normalized[col].max() - metrics_normalized[col].min())

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


def model_for_mouse_y(X2, Y2, models, model_names):
    """
    Trains multiple models to predict the Y coordinate based on the given features and compares their performance.

    Args:
        - X2 (array-like): The input features.
        - Y2 (array-like): The target variable (Y coordinate).
        - models (list): A list of machine learning models to be trained.
        - model_names (list): A list of model names corresponding to the models.

    Returns: None
    """
    # Split dataset into train and test sets (80/20 where 20 is for test)
    X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.2, random_state=42)

    # Initialize empty lists to store the metrics data
    metrics_list = []

    for model, model_name in zip(models, model_names):
        # Train the model
        model.fit(X2_train, Y2_train)

        # Predict the target variable for the test set
        Y2_pred_test = model.predict(X2_test)

        # Filter out the negative predicted values
        non_negative_indices = Y2_pred_test >= 0
        Y2_pred_test_filtered = Y2_pred_test[non_negative_indices]
        Y2_test_filtered = Y2_test[non_negative_indices]

        # Compute the metrics for the test set with filtered predictions
        metrics_data_test = {
            "Model": model_name,
            "Mean Absolute Error (MAE)": mean_absolute_error(
                Y2_test_filtered, Y2_pred_test_filtered
            ),
            "Median Absolute Error": median_absolute_error(
                Y2_test_filtered, Y2_pred_test_filtered
            ),
            "Mean Squared Error (MSE)": mean_squared_error(
                Y2_test_filtered, Y2_pred_test_filtered
            ),
            "Mean Log Squared Error (MSLE)": mean_squared_log_error(
                Y2_test_filtered, Y2_pred_test_filtered
            ),
            "Root Mean Squared Error (RMSE)": np.sqrt(
                mean_squared_error(Y2_test_filtered, Y2_pred_test_filtered)
            ),
            "Explained Variance Score": explained_variance_score(
                Y2_test_filtered, Y2_pred_test_filtered
            ),
            "Max Error": max_error(Y2_test_filtered, Y2_pred_test_filtered),
            "MODEL Y SCORE R2": r2_score(Y2_test_filtered, Y2_pred_test_filtered),
        }

        metrics_list.append(metrics_data_test)

    # Convert metrics data to DataFrame
    metrics_df_test = pd.DataFrame(metrics_list)

    # Display metrics using Streamlit
    st.subheader("Metrics for the test set - Y")
    st.dataframe(metrics_df_test, width="stretch")

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
            "Model": np.repeat(model_names, len(Y2_test)),
            "Actual": np.tile(Y2_test, len(models)),
            "Predicted": np.concatenate([model.predict(X2_test) for model in models]),
        }
    )
    errors_df["Error"] = errors_df["Actual"] - errors_df["Predicted"]

    # Create the box plot
    st.dataframe(errors_df, width="stretch")
    fig = px.box(errors_df, x="Model", y="Error")
    st.plotly_chart(fig)

    # Radar chart for model comparison
    st.subheader("Radar Chart Comparison")

    # Normalize the metric values for better comparison
    metrics_normalized = metrics_df_test.copy()
    for col in metrics_normalized.columns[1:]:
        metrics_normalized[col] = (
            metrics_normalized[col] - metrics_normalized[col].min()
        ) / (metrics_normalized[col].max() - metrics_normalized[col].min())

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


# Set the title of the app and the tabs
st.subheader("Eye Tracker Calibration Data Analysis and Prediction")
st.write(f"Select the tab to view the data and metrics for [{prefix}] data")
tab1, tab2 = st.tabs(["Raw Data", "Metrics"])

# With the first tab
with tab1:
    # Display the raw dataset
    st.subheader("Data Obtained from Calibration")
    st.dataframe(raw_dataset, width="stretch")

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
        st.plotly_chart(fig_left, theme="streamlit", width="stretch")

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
        st.plotly_chart(fig_right, theme="streamlit", width="stretch")

    # Create the line plot
    fig3 = px.line(
        raw_dataset,
        y=["left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"],
        title="Left and Right Iris Position",
    )
    # Display the plot
    st.plotly_chart(fig3, theme="streamlit", width="stretch")


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

    # Train the models
    model_for_mouse_x(X1, Y1, models, model_names)
    model_for_mouse_y(X2, Y2, models, model_names)
