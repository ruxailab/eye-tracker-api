"""
experiments/run_experiment.py
-----------------------------
Model Comparison Benchmark for RUXAILAB Eye-Tracker API.

Evaluates 10 regression model variants across two validation regimes:
1. Stratified 80/20 train/test split by target calibration point
2. Leave-One-Point-Out (9-fold cross validation across calibration targets)

Records positional accuracy (mean, median, RMSE, threshold accuracy at 50px/100px),
intra-target spatial precision, training duration, and single-sample inference latency.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    BayesianRidge,
    SGDRegressor,
)
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Ensure repository root is on sys.path for internal imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Task 2 Requirement: Reuse metrics from app/services/metrics.py if they import cleanly.
# If they don't, fallback implementations are provided with explanatory comments.
try:
    from app.services.metrics import (
        func_precision_x,
        func_presicion_y,
        func_accuracy_x,
        func_accuracy_y,
        func_total_accuracy,
    )
    REUSED_METRICS_FROM_APP = True
except Exception as err:
    REUSED_METRICS_FROM_APP = False

    # Fallback reimplementation required when app package cannot be imported
    # (e.g. running in an isolated environment without repo path configuration).
    def func_precision_x(group):
        return float(np.sqrt(np.mean(np.square(group["Predicted X"] - np.mean(group["Predicted X"])))))

    def func_presicion_y(group):
        return float(np.sqrt(np.mean(np.square(group["Predicted Y"] - np.mean(group["Predicted Y"])))))

    def func_accuracy_x(group):
        return float(np.sqrt(np.mean(np.square(group["True X"] - group["Predicted X"]))))

    def func_accuracy_y(group):
        return float(np.sqrt(np.mean(np.square(group["True Y"] - group["Predicted Y"]))))

    def func_total_accuracy(group):
        distances = np.sqrt(
            np.square(group["True X"] - group["Predicted X"])
            + np.square(group["True Y"] - group["Predicted Y"])
        )
        return float(np.mean(distances))

# Attempt to reuse production hyperparameters from app.services.config
try:
    from app.services.config import hyperparameters as app_hyperparameters
except Exception:
    app_hyperparameters = {}


def extract_features(
    df: pd.DataFrame,
    feature_set: str,
    train_ref: Dict[str, float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Extracts X and Y input features based on feature_set mode.

    Parameters:
        df: Input DataFrame with iris coordinates.
        feature_set: 'raw' or 'engineered'.
        train_ref: Reference baseline stats from training set (used during inference/test).

    Returns:
        (X_for_x_model, X_for_y_model, stats_dict)
    """
    lx = df["left_iris_x"].values.astype(float)
    rx = df["right_iris_x"].values.astype(float)
    ly = df["left_iris_y"].values.astype(float)
    ry = df["right_iris_y"].values.astype(float)

    if feature_set == "raw":
        # Raw landmarks: axis-specific pairings matching gaze_tracker.py predict()
        X_x = np.column_stack([lx, rx])
        X_y = np.column_stack([ly, ry])
        return X_x, X_y, {}

    # Engineered features (matching gaze_tracker.py lines 351-468 & 580-587)
    mean_x = (lx + rx) / 2.0
    diff_x = lx - rx
    mean_y = (ly + ry) / 2.0
    diff_y = ly - ry

    if train_ref is None:
        # Fit reference baseline on training split
        ref_mean_x = float(np.mean(mean_x))
        ref_mean_y = float(np.mean(mean_y))
        iris_y_scale = float(np.std(mean_y)) + 1e-6
        rel_x = mean_x - ref_mean_x
        stats = {
            "ref_mean_x": ref_mean_x,
            "ref_mean_y": ref_mean_y,
            "iris_y_scale": iris_y_scale,
            "calib_rel_x_mean": float(np.mean(rel_x)),
        }
    else:
        # Transform test set strictly using training-derived reference values.
        # NOTE: Test-batch centering (subtracting np.mean(rel_x) of the test batch, as done
        # in app/services/gaze_tracker.py lines 580-587) was removed here because in
        # Leave-One-Point-Out (LOPO) cross-validation, the test set consists of only a single
        # calibration target point. Subtracting the test batch mean erases the target's
        # true spatial offset from screen center, collapsing test predictions to the horizontal
        # center (x=960) and artificially inflating LOPO error to ~530 px (see sanity_check.md).
        ref_mean_x = train_ref["ref_mean_x"]
        ref_mean_y = train_ref["ref_mean_y"]
        iris_y_scale = train_ref["iris_y_scale"]
        stats = train_ref
        rel_x = mean_x - ref_mean_x

    rel_y = mean_y - ref_mean_y
    diff_y_norm = diff_y / iris_y_scale
    rel_y_norm = rel_y / iris_y_scale

    X_x = np.column_stack([diff_x, rel_x])
    X_y = np.column_stack([diff_y_norm, rel_y_norm])

    return X_x, X_y, stats


def build_models(feature_set: str, seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Constructs model pipelines matching gaze_tracker.py and config.py specifications.

    Parameters:
        feature_set: 'raw' or 'engineered'.
        seed: Random seed for stochastic algorithms.

    Returns:
        Dictionary mapping model_name to dict with 'pipeline' and 'hyperparameters'.
    """
    use_poly_on_linear = True
    use_scaler = True

    model_defs = {
        "Linear Regression": {
            "model": LinearRegression(),
            "use_poly": True,
            "params": {"fit_intercept": True},
        },
        "Ridge Regression": {
            "model": Ridge(alpha=0.5, random_state=seed),
            "use_poly": (feature_set == "raw"),
            "params": {"alpha": 0.5},
        },
        "Lasso Regression": {
            "model": Lasso(alpha=0.1, random_state=seed, max_iter=2000),
            "use_poly": (feature_set == "raw"),
            "params": {"alpha": 0.1},
        },
        "Elastic Net": {
            "model": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=seed, max_iter=2000),
            "use_poly": (feature_set == "raw"),
            "params": {"alpha": 1.0, "l1_ratio": 0.5},
        },
        "Bayesian Ridge": {
            "model": BayesianRidge(alpha_init=1.0, lambda_init=1e-3),
            "use_poly": (feature_set == "raw"),
            "params": {"alpha_init": 1.0, "lambda_init": 1e-3},
        },
        "SGD Regressor": {
            "model": SGDRegressor(
                alpha=0.001,
                l1_ratio=0.7,
                max_iter=1000,
                eta0=0.001,
                random_state=seed,
            ),
            "use_poly": (feature_set == "raw"),
            "params": {"alpha": 0.001, "l1_ratio": 0.7, "max_iter": 1000, "eta0": 0.001},
        },
        "Support Vector Regressor (Linear)": {
            "model": SVR(kernel="linear", C=1.0),
            "use_poly": (feature_set == "raw"),
            "params": {"kernel": "linear", "C": 1.0},
        },
        "Support Vector Regressor (RBF)": {
            "model": SVR(kernel="rbf", C=10.0, gamma=0.1),
            "use_poly": (feature_set == "raw"),
            "params": {"kernel": "rbf", "C": 10.0, "gamma": 0.1},
        },
        "Support Vector Regressor (Poly)": {
            "model": SVR(kernel="poly", degree=2, C=1.0),
            "use_poly": False,
            "params": {"kernel": "poly", "degree": 2, "C": 1.0},
        },
        "Random Forest Regressor": {
            # NOTE: max_depth=5 and min_samples_leaf=10 were chosen to prevent memorization
            # of the 9 discrete calibration target clusters in feature space (which otherwise
            # yields artificial 0.00 px error on stratified splits, as diagnosed in sanity_check.md).
            # n_estimators=200 and min_samples_split=5 are kept from production settings.
            # This is a regularized baseline constraint, not a cross-validated tuned score.
            "model": RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=10,
                random_state=seed,
            ),
            "use_poly": False,
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "min_samples_split": 5,
                "min_samples_leaf": 10,
            },
        },
    }

    built = {}
    for name, item in model_defs.items():
        steps = []
        if use_scaler:
            steps.append(StandardScaler())
        if item["use_poly"]:
            steps.append(PolynomialFeatures(degree=2, include_bias=False))
        steps.append(item["model"])

        pipeline = make_pipeline(*steps)
        built[name] = {
            "pipeline": pipeline,
            "hyperparameters": item["params"],
        }

    return built


def benchmark_single_inference_latency(
    model_x, model_y, x_sample: np.ndarray, y_sample: np.ndarray, iterations: int = 400
) -> float:
    """
    Measures the median single-sample inference latency in milliseconds.
    """
    x_1 = x_sample.reshape(1, -1)
    y_1 = y_sample.reshape(1, -1)

    # Warmup
    for _ in range(20):
        _ = model_x.predict(x_1)
        _ = model_y.predict(y_1)

    timings = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model_x.predict(x_1)
        _ = model_y.predict(y_1)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    return float(np.median(timings))


def compute_metrics(
    true_x: np.ndarray,
    pred_x: np.ndarray,
    true_y: np.ndarray,
    pred_y: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes all standard eye-tracking evaluation metrics.
    Reuses app/services/metrics.py functions directly.
    """
    eval_df = pd.DataFrame({
        "True X": true_x,
        "Predicted X": pred_x,
        "True Y": true_y,
        "Predicted Y": pred_y,
        "Target": targets,
    })

    # Total accuracy (mean Euclidean error)
    mean_euclidean = float(func_total_accuracy(eval_df))

    # Axis RMSE
    rmse_x = float(func_accuracy_x(eval_df))
    rmse_y = float(func_accuracy_y(eval_df))
    rmse = float(np.sqrt(rmse_x**2 + rmse_y**2))

    # Euclidean error distribution
    errors = np.sqrt((true_x - pred_x) ** 2 + (true_y - pred_y) ** 2)
    median_error = float(np.median(errors))

    # Threshold accuracies
    accuracy_50px = float(np.mean(errors <= 50.0) * 100.0)
    accuracy_100px = float(np.mean(errors <= 100.0) * 100.0)

    # Spatial precision (std per axis grouped by calibration target)
    grouped = eval_df.groupby("Target")
    prec_x = float(grouped.apply(func_precision_x).mean())
    prec_y = float(grouped.apply(func_presicion_y).mean())
    prec_xy = (prec_x + prec_y) / 2.0

    # Per-point details
    point_details = []
    for target_name, group in grouped:
        pt_errors = np.sqrt(
            (group["True X"] - group["Predicted X"]) ** 2
            + (group["True Y"] - group["Predicted Y"]) ** 2
        )
        tx = float(group["True X"].iloc[0])
        ty = float(group["True Y"].iloc[0])
        point_details.append({
            "target_x": tx,
            "target_y": ty,
            "samples": len(group),
            "mean_error_px": round(float(np.mean(pt_errors)), 2),
            "std_error_px": round(float(np.std(pt_errors)), 2),
            "accuracy_50px": round(float(np.mean(pt_errors <= 50.0) * 100.0), 2),
            "accuracy_100px": round(float(np.mean(pt_errors <= 100.0) * 100.0), 2),
            "precision_x_px": round(float(func_precision_x(group)), 2),
            "precision_y_px": round(float(func_presicion_y(group)), 2),
        })

    return {
        "mean_euclidean_error_px": round(mean_euclidean, 2),
        "median_error_px": round(median_error, 2),
        "rmse_px": round(rmse, 2),
        "rmse_x_px": round(rmse_x, 2),
        "rmse_y_px": round(rmse_y, 2),
        "accuracy_50px_pct": round(accuracy_50px, 2),
        "accuracy_100px_pct": round(accuracy_100px, 2),
        "precision_x_px": round(prec_x, 2),
        "precision_y_px": round(prec_y, 2),
        "precision_xy_px": round(prec_xy, 2),
        "points_detail": point_details,
    }


def evaluate_stratified_80_20(
    df: pd.DataFrame,
    model_name: str,
    base_pipeline,
    feature_set: str,
    seed: int,
) -> Tuple[Dict[str, Any], float, float]:
    """
    Evaluates a model using stratified 80/20 train/test split.
    """
    target_labels = [f"({row.point_x:.0f},{row.point_y:.0f})" for _, row in df.iterrows()]
    df = df.copy()
    df["target_group"] = target_labels

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target_group"],
        random_state=seed,
    )

    X_train_x, X_train_y, train_ref = extract_features(train_df, feature_set)
    X_test_x, X_test_y, _ = extract_features(test_df, feature_set, train_ref=train_ref)

    y_train_x = train_df["point_x"].values
    y_train_y = train_df["point_y"].values
    y_test_x = test_df["point_x"].values
    y_test_y = test_df["point_y"].values

    pipe_x = clone(base_pipeline)
    pipe_y = clone(base_pipeline)

    t0 = time.perf_counter()
    pipe_x.fit(X_train_x, y_train_x)
    pipe_y.fit(X_train_y, y_train_y)
    train_time_ms = (time.perf_counter() - t0) * 1000.0

    pred_x = pipe_x.predict(X_test_x)
    pred_y = pipe_y.predict(X_test_y)

    latency_ms = benchmark_single_inference_latency(
        pipe_x, pipe_y, X_test_x[0], X_test_y[0]
    )

    metrics = compute_metrics(
        true_x=y_test_x,
        pred_x=pred_x,
        true_y=y_test_y,
        pred_y=pred_y,
        targets=test_df["target_group"].values,
    )

    return metrics, train_time_ms, latency_ms


def evaluate_leave_one_point_out(
    df: pd.DataFrame,
    model_name: str,
    base_pipeline,
    feature_set: str,
    seed: int,
) -> Tuple[Dict[str, Any], float, float]:
    """
    Evaluates a model using Leave-One-Point-Out (9-fold cross-validation).
    """
    unique_points = df[["point_x", "point_y"]].drop_duplicates().values
    df = df.copy()
    df["target_group"] = [f"({row.point_x:.0f},{row.point_y:.0f})" for _, row in df.iterrows()]

    all_true_x = []
    all_pred_x = []
    all_true_y = []
    all_pred_y = []
    all_targets = []

    fold_train_times = []
    fold_latencies = []

    for target_x, target_y in unique_points:
        test_mask = (df["point_x"] == target_x) & (df["point_y"] == target_y)
        train_df = df[~test_mask]
        test_df = df[test_mask]

        X_train_x, X_train_y, train_ref = extract_features(train_df, feature_set)
        X_test_x, X_test_y, _ = extract_features(test_df, feature_set, train_ref=train_ref)

        y_train_x = train_df["point_x"].values
        y_train_y = train_df["point_y"].values
        y_test_x = test_df["point_x"].values
        y_test_y = test_df["point_y"].values

        pipe_x = clone(base_pipeline)
        pipe_y = clone(base_pipeline)

        t0 = time.perf_counter()
        pipe_x.fit(X_train_x, y_train_x)
        pipe_y.fit(X_train_y, y_train_y)
        fold_train_times.append((time.perf_counter() - t0) * 1000.0)

        p_x = pipe_x.predict(X_test_x)
        p_y = pipe_y.predict(X_test_y)

        all_true_x.extend(y_test_x)
        all_pred_x.extend(p_x)
        all_true_y.extend(y_test_y)
        all_pred_y.extend(p_y)
        all_targets.extend(test_df["target_group"].values)

        if len(fold_latencies) == 0:
            fold_latencies.append(
                benchmark_single_inference_latency(pipe_x, pipe_y, X_test_x[0], X_test_y[0])
            )

    metrics = compute_metrics(
        true_x=np.array(all_true_x),
        pred_x=np.array(all_pred_x),
        true_y=np.array(all_true_y),
        pred_y=np.array(all_pred_y),
        targets=np.array(all_targets),
    )

    avg_train_time_ms = float(np.mean(fold_train_times))
    latency_ms = fold_latencies[0]

    return metrics, avg_train_time_ms, latency_ms


def run_experiment(
    data_path: str = "experiments/data/synthetic_calib.csv",
    seed: int = 42,
    output_dir: str = "experiments/results",
    feature_set: str = "engineered",
) -> Dict[str, Any]:
    """
    Executes the model comparison experiment and writes output files.
    """
    print("=" * 70)
    print("  RUXAILAB Eye-Tracker Model Comparison Benchmark")
    print("=" * 70)
    print(f"Data Source       : {data_path}")
    print(f"Feature Set       : {feature_set}")
    print(f"Random Seed       : {seed}")
    print(f"Output Directory  : {output_dir}")
    print(f"App Metrics Reused: {REUSED_METRICS_FROM_APP}")
    print("-" * 70)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at '{data_path}'. "
            f"Run 'python experiments/synthetic_data.py' first."
        )

    df = pd.read_csv(data_path)
    total_samples = len(df)
    unique_points = df[["point_x", "point_y"]].drop_duplicates().values
    print(f"Loaded {total_samples} samples across {len(unique_points)} calibration targets.")

    models_dict = build_models(feature_set=feature_set, seed=seed)
    print(f"Prepared {len(models_dict)} model pipelines for evaluation.")

    csv_rows = []
    full_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_path": data_path,
            "feature_set": feature_set,
            "seed": seed,
            "total_samples": total_samples,
            "calibration_targets_count": len(unique_points),
            "reused_metrics_from_app": REUSED_METRICS_FROM_APP,
        },
        "models": {},
    }

    print("\nRunning benchmarks...")
    print(f"{'Model Name':<35} | {'Split':<18} | {'Mean Err (px)':<13} | {'RMSE (px)':<10} | {'Acc@50px':<9} | {'Train (ms)':<10}")
    print("-" * 105)

    for model_name, info in models_dict.items():
        base_pipe = info["pipeline"]
        hparams = info["hyperparameters"]

        full_results["models"][model_name] = {
            "hyperparameters": hparams,
            "stratified_80_20": {},
            "leave_one_point_out": {},
        }

        # --- 1. Stratified 80/20 Split ---
        m_80, t_train_80, t_lat_80 = evaluate_stratified_80_20(
            df=df,
            model_name=model_name,
            base_pipeline=base_pipe,
            feature_set=feature_set,
            seed=seed,
        )

        full_results["models"][model_name]["stratified_80_20"] = {
            **m_80,
            "train_time_ms": round(t_train_80, 2),
            "inference_latency_ms": round(t_lat_80, 4),
        }

        csv_rows.append({
            "model_name": model_name,
            "split_type": "stratified_80_20",
            "feature_set": feature_set,
            "mean_euclidean_error_px": m_80["mean_euclidean_error_px"],
            "median_error_px": m_80["median_error_px"],
            "rmse_px": m_80["rmse_px"],
            "accuracy_50px_pct": m_80["accuracy_50px_pct"],
            "accuracy_100px_pct": m_80["accuracy_100px_pct"],
            "precision_x_px": m_80["precision_x_px"],
            "precision_y_px": m_80["precision_y_px"],
            "precision_xy_px": m_80["precision_xy_px"],
            "train_time_ms": round(t_train_80, 2),
            "inference_latency_ms": round(t_lat_80, 4),
        })

        print(
            f"{model_name:<35} | {'stratified_80_20':<18} | "
            f"{m_80['mean_euclidean_error_px']:<13.2f} | "
            f"{m_80['rmse_px']:<10.2f} | "
            f"{m_80['accuracy_50px_pct']:<8.1f}% | "
            f"{t_train_80:<10.1f}"
        )

        # --- 2. Leave-One-Point-Out Split ---
        m_lopo, t_train_lopo, t_lat_lopo = evaluate_leave_one_point_out(
            df=df,
            model_name=model_name,
            base_pipeline=base_pipe,
            feature_set=feature_set,
            seed=seed,
        )

        full_results["models"][model_name]["leave_one_point_out"] = {
            **m_lopo,
            "train_time_ms": round(t_train_lopo, 2),
            "inference_latency_ms": round(t_lat_lopo, 4),
        }

        csv_rows.append({
            "model_name": model_name,
            "split_type": "leave_one_point_out",
            "feature_set": feature_set,
            "mean_euclidean_error_px": m_lopo["mean_euclidean_error_px"],
            "median_error_px": m_lopo["median_error_px"],
            "rmse_px": m_lopo["rmse_px"],
            "accuracy_50px_pct": m_lopo["accuracy_50px_pct"],
            "accuracy_100px_pct": m_lopo["accuracy_100px_pct"],
            "precision_x_px": m_lopo["precision_x_px"],
            "precision_y_px": m_lopo["precision_y_px"],
            "precision_xy_px": m_lopo["precision_xy_px"],
            "train_time_ms": round(t_train_lopo, 2),
            "inference_latency_ms": round(t_lat_lopo, 4),
        })

        print(
            f"{model_name:<35} | {'leave_one_out':<18} | "
            f"{m_lopo['mean_euclidean_error_px']:<13.2f} | "
            f"{m_lopo['rmse_px']:<10.2f} | "
            f"{m_lopo['accuracy_50px_pct']:<8.1f}% | "
            f"{t_train_lopo:<10.1f}"
        )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "model_comparison_metrics.csv")
    json_path = os.path.join(output_dir, "benchmark_results.json")

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print("-" * 105)
    print(f"\n[DONE] Saved summary metrics to: {csv_path}")
    print(f"[DONE] Saved full benchmark JSON to: {json_path}")

    return full_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Eye-Tracker Model Comparison Experiment.")
    parser.add_argument(
        "--data",
        type=str,
        default="experiments/data/synthetic_calib.csv",
        help="Path to calibration dataset CSV (default: experiments/data/synthetic_calib.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory to store output reports (default: experiments/results).",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        choices=["raw", "engineered"],
        default="engineered",
        help="Feature representation to evaluate (default: engineered).",
    )

    args = parser.parse_args()

    run_experiment(
        data_path=args.data,
        seed=args.seed,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
    )
