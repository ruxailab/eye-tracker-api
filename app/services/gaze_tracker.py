# Necessary imports
import warnings

warnings.filterwarnings("ignore")
from sklearn.base import clone
import numpy as np
import pandas as pd
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
}

# Set the scoring metrics for GridSearchCV to r2_score and mean_absolute_error
scoring = {
    "r2": make_scorer(r2_score),
    "mae": make_scorer(mean_absolute_error),
}


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
    sc = StandardScaler()

    # Load data from csv file and drop unnecessary columns
    df = pd.read_csv(data)
    df = df.drop(["screen_height", "screen_width"], axis=1)

    # Data for X axis
    X_x = df[["left_iris_x", "right_iris_x"]]
    X_y = df["point_x"]

    # Normalize data using standard scaler and split data into training and testing sets
    X_x = sc.fit_transform(X_x)
    X_train_x, X_test_x, y_train_x, y_test_x = train_test_split(
        X_x, X_y, test_size=0.2, random_state=42
    )

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

    # Data for Y axis
    X_y = df[["left_iris_y", "right_iris_y"]]
    y_y = df["point_y"]

    # Normalize data using standard scaler and split data into training and testing sets
    X_y = sc.fit_transform(X_y)
    X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(
        X_y, y_y, test_size=0.2, random_state=42
    )

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

def predict_new_data_simple(calib_csv_path, predict_csv_path, model_X, model_Y, k=3):
    """
    Versão simplificada de predict_new_data.
    Treina modelos nos dados de calibração e prevê coordenadas nos novos dados.
    Retorna o mesmo formato que a função `predict`.
    """
    # -------------------- SCALERS --------------------
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    # -------------------- TREINO --------------------
    df_train = pd.read_csv(calib_csv_path).drop(["screen_height", "screen_width"], axis=1)

    X_train_x = df_train[["left_iris_x", "right_iris_x"]].values
    y_train_x = df_train["point_x"].values
    X_train_y = df_train[["left_iris_y", "right_iris_y"]].values
    y_train_y = df_train["point_y"].values

    X_train_x_scaled = sc_x.fit_transform(X_train_x)
    X_train_y_scaled = sc_y.fit_transform(X_train_y)

    # Modelos
    model_fit_x = clone(models[model_X]).fit(X_train_x_scaled, y_train_x)
    model_fit_y = clone(models[model_Y]).fit(X_train_y_scaled, y_train_y)
    # -------------------- NOVOS DADOS --------------------
    df_predict = pd.read_csv(predict_csv_path)
    X_pred_x = sc_x.transform(df_predict[["left_iris_x", "right_iris_x"]].values)
    X_pred_y = sc_y.transform(df_predict[["left_iris_y", "right_iris_y"]].values)

    y_pred_x = model_fit_x.predict(X_pred_x)
    y_pred_y = model_fit_y.predict(X_pred_y)

    # Garantir valores não-negativos
    y_pred_x = np.clip(y_pred_x, 0, None)
    y_pred_y = np.clip(y_pred_y, 0, None)

    # -------------------- KMEANS --------------------
    data_pred = np.array([y_pred_x, y_pred_y]).T
    kmeans_model = KMeans(n_clusters=k, n_init="auto", init="k-means++")
    y_kmeans = kmeans_model.fit_predict(data_pred)

    # -------------------- FORMATA DADOS --------------------
    df_data = pd.DataFrame({
        "Predicted X": y_pred_x,
        "Predicted Y": y_pred_y,
        "True X": df_predict["point_x"] if "point_x" in df_predict else y_pred_x,
        "True Y": df_predict["point_y"] if "point_y" in df_predict else y_pred_y
    })

    # Calcular métricas
    precision_x = df_data.groupby(["True X", "True Y"]).apply(func_precision_x)
    precision_y = df_data.groupby(["True X", "True Y"]).apply(func_presicion_y)
    precision_xy = (precision_x + precision_y) / 2
    precision_xy /= np.mean(precision_xy)

    accuracy_x = df_data.groupby(["True X", "True Y"]).apply(func_accuracy_x)
    accuracy_y = df_data.groupby(["True X", "True Y"]).apply(func_accuracy_y)
    accuracy_xy = (accuracy_x + accuracy_y) / 2
    accuracy_xy /= np.mean(accuracy_xy)

    # Estrutura final
    data = {}
    for index, row in df_data.iterrows():
        outer_key = str(int(row["True X"]))
        inner_key = str(int(row["True Y"]))
        if outer_key not in data:
            data[outer_key] = {}
        data[outer_key][inner_key] = {
            "predicted_x": df_data[
                (df_data["True X"] == row["True X"]) &
                (df_data["True Y"] == row["True Y"])
            ]["Predicted X"].tolist(),
            "predicted_y": df_data[
                (df_data["True X"] == row["True X"]) &
                (df_data["True Y"] == row["True Y"])
            ]["Predicted Y"].tolist(),
            "PrecisionSD": precision_xy[(row["True X"], row["True Y"])],
            "Accuracy": accuracy_xy[(row["True X"], row["True Y"])],
        }

    data["centroids"] = kmeans_model.cluster_centers_.tolist()
    return data


def train_to_validate_calib(calib_csv_file, predict_csv_file):
    dataset_train_path = calib_csv_file
    dataset_predict_path = predict_csv_file

    # Carregue os dados de treinamento a partir do CSV
    data = pd.read_csv(dataset_train_path)

    # Para evitar que retorne valores negativos: Aplicar uma transformação logarítmica aos rótulos (point_x e point_y)
    # data['point_x'] = np.log(data['point_x'])
    # data['point_y'] = np.log(data['point_y'])

    # Separe os recursos (X) e os rótulos (y)
    X = data[["left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"]]
    y = data[["point_x", "point_y"]]

    # Crie e ajuste um modelo de regressão linear
    model = linear_model.LinearRegression()
    model.fit(X, y)

    # Carregue os dados de teste a partir de um novo arquivo CSV
    dados_teste = pd.read_csv(dataset_predict_path)

    # Faça previsões
    previsoes = model.predict(dados_teste)

    # Para evitar que retorne valores negativos: Inverter a transformação logarítmica nas previsões
    # previsoes = np.exp(previsoes)

    # Exiba as previsões
    print("Previsões de point_x e point_y:")
    print(previsoes)
    return previsoes.tolist()


def train_model(session_id):
    # Download dataset
    dataset_train_path = (
        f"{Path().absolute()}/public/training/{session_id}/train_data.csv"
    )
    dataset_session_path = (
        f"{Path().absolute()}/public/sessions/{session_id}/session_data.csv"
    )

    # Importing data from csv
    raw_dataset = pd.read_csv(dataset_train_path)
    session_dataset = pd.read_csv(dataset_session_path)

    train_stats = raw_dataset.describe()
    train_stats = train_stats.transpose()

    dataset_t = raw_dataset
    dataset_s = session_dataset.drop(["timestamp"], axis=1)

    # Drop the columns that will be predicted
    X = dataset_t.drop(["timestamp", "mouse_x", "mouse_y"], axis=1)

    Y1 = dataset_t.mouse_x
    Y2 = dataset_t.mouse_y
    # print('Y1 is the mouse_x column ->', Y1)
    # print('Y2 is the mouse_y column ->', Y2)

    MODEL_X = model_for_mouse_x(X, Y1)
    MODEL_Y = model_for_mouse_y(X, Y2)

    GAZE_X = MODEL_X.predict(dataset_s)
    GAZE_Y = MODEL_Y.predict(dataset_s)

    GAZE_X = np.abs(GAZE_X)
    GAZE_Y = np.abs(GAZE_Y)

    return {"x": GAZE_X, "y": GAZE_Y}


def model_for_mouse_x(X, Y1):
    print("-----------------MODEL FOR X------------------")
    # split dataset into train and test sets (80/20 where 20 is for test)
    X_train, X_test, Y1_train, Y1_test = train_test_split(X, Y1, test_size=0.2)

    model = linear_model.LinearRegression()
    model.fit(X_train, Y1_train)

    Y1_pred_train = model.predict(X_train)
    Y1_pred_test = model.predict(X_test)

    Y1_test = normalizeData(Y1_test)
    Y1_pred_test = normalizeData(Y1_pred_test)

    print(f"Mean absolute error MAE = {mean_absolute_error(Y1_test, Y1_pred_test)}")
    print(f"Mean squared error MSE = {mean_squared_error(Y1_test, Y1_pred_test)}")
    print(
        f"Mean squared log error MSLE = {mean_squared_log_error(Y1_test, Y1_pred_test)}"
    )
    print(f"MODEL X SCORE R2 = {model.score(X, Y1)}")

    # print(f'TRAIN{Y1_pred_train}')
    # print(f'TEST{Y1_pred_test}')
    return model


def model_for_mouse_y(X, Y2):
    print("-----------------MODEL FOR Y------------------")
    # split dataset into train and test sets (80/20 where 20 is for test)
    X_train, X_test, Y2_train, Y2_test = train_test_split(X, Y2, test_size=0.2)

    model = linear_model.LinearRegression()
    model.fit(X_train, Y2_train)

    Y2_pred_train = model.predict(X_train)
    Y2_pred_test = model.predict(X_test)

    Y2_test = normalizeData(Y2_test)
    Y2_pred_test = normalizeData(Y2_pred_test)

    print(f"Mean absolute error MAE = {mean_absolute_error(Y2_test, Y2_pred_test)}")
    print(f"Mean squared error MSE = {mean_squared_error(Y2_test, Y2_pred_test)}")
    print(
        f"Mean squared log error MSLE = {mean_squared_log_error(Y2_test, Y2_pred_test)}"
    )
    print(f"MODEL X SCORE R2 = {model.score(X, Y2)}")

    # print(f'TRAIN{Y2_pred_train}')
    print(f"TEST{Y2_pred_test}")
    return model


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
