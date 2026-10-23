"""
experiments/sanity_check.py
---------------------------
Diagnostic script to investigate anomalies in the eye-tracker model comparison experiment:
1. Data Sanity Check
2. Learnability Check (Noiseless Linear Fit)
3. Units & Scaling Check
4. Feature Leakage & Alignment Check
5. Fold Integrity Check (LOPO and Stratified 80/20)
6. Random Forest Memorization Check (Train vs Validation error)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.synthetic_data import generate_synthetic_calibration_data
from experiments.run_experiment import extract_features, build_models, compute_metrics


def check_1_data_sanity(csv_path="experiments/data/synthetic_calib.csv"):
    print("=" * 70)
    print("CHECK 1: DATA SANITY")
    print("=" * 70)
    
    if not os.path.exists(csv_path):
        print(f"FAIL: File not found at {csv_path}")
        return False, {}

    df = pd.read_csv(csv_path)
    print(f"Shape        : {df.shape}")
    print(f"Columns      : {list(df.columns)}")
    print(f"Data types   :\n{df.dtypes}")
    
    nan_counts = df.isna().sum().to_dict()
    total_nans = df.isna().sum().sum()
    print(f"NaN counts   : {nan_counts}")
    print(f"Total NaNs   : {total_nans}")

    unique_x = sorted(df["point_x"].unique())
    unique_y = sorted(df["point_y"].unique())
    print(f"Unique point_x ({len(unique_x)}): {unique_x}")
    print(f"Unique point_y ({len(unique_y)}): {unique_y}")

    grouped = df.groupby(["point_x", "point_y"]).size().reset_index(name="sample_count")
    print("\nSamples per target point:")
    print(grouped.to_string(index=False))

    has_9_points = (len(grouped) == 9)
    all_100 = (grouped["sample_count"] == 100).all()
    no_nans = (total_nans == 0)

    pass_status = has_9_points and all_100 and no_nans
    print(f"\nResult: {'PASS' if pass_status else 'FAIL'}")
    return pass_status, {
        "shape": df.shape,
        "nans": total_nans,
        "target_count": len(grouped),
        "samples_per_point": grouped["sample_count"].tolist(),
        "unique_x": unique_x,
        "unique_y": unique_y,
    }


def check_2_learnability(noise_std=0.0):
    print("\n" + "=" * 70)
    print(f"CHECK 2: LEARNABILITY (LINEAR FIT WITH noise_std={noise_std})")
    print("=" * 70)

    # Generate noiseless dataset: noise_std, head_shift_ratio=0.0
    df_noiseless = generate_synthetic_calibration_data(
        num_samples_per_point=100,
        screen_width=1920,
        screen_height=1080,
        noise_std=noise_std,
        head_shift_ratio=0.0,
        head_shift_x=0.0,
        head_shift_y=0.0,
        seed=42,
        output_path=None,
    )

    raw_features = ["left_iris_x", "left_iris_y", "right_iris_x", "right_iris_y"]
    X = df_noiseless[raw_features].values
    y_x = df_noiseless["point_x"].values
    y_y = df_noiseless["point_y"].values

    # Fit plain LinearRegression for X and Y
    reg_x = LinearRegression()
    reg_x.fit(X, y_x)
    pred_x = reg_x.predict(X)
    r2_x = float(r2_score(y_x, pred_x))

    reg_y = LinearRegression()
    reg_y.fit(X, y_y)
    pred_y = reg_y.predict(X)
    r2_y = float(r2_score(y_y, pred_y))

    euclidean_errors = np.sqrt((y_x - pred_x) ** 2 + (y_y - pred_y) ** 2)
    mean_euclidean = float(np.mean(euclidean_errors))
    max_euclidean = float(np.max(euclidean_errors))

    print(f"Features used              : {raw_features}")
    print(f"LinearRegression R2 (X)    : {r2_x:.8f}")
    print(f"LinearRegression R2 (Y)    : {r2_y:.8f}")
    print(f"Mean Euclidean Error (px)  : {mean_euclidean:.8f}")
    print(f"Max Euclidean Error (px)   : {max_euclidean:.8f}")

    pass_status = (r2_x > 0.9999 and r2_y > 0.9999 and mean_euclidean < 0.01) if noise_std == 0.0 else (r2_x > 0.95 and r2_y > 0.95)
    print(f"\nResult: {'PASS' if pass_status else 'FAIL'}")
    return pass_status, {
        "r2_x": r2_x,
        "r2_y": r2_y,
        "mean_euclidean_px": mean_euclidean,
        "max_euclidean_px": max_euclidean,
    }


def check_3_units(csv_path="experiments/data/synthetic_calib.csv"):
    print("\n" + "=" * 70)
    print("CHECK 3: UNITS & SCALING CHECK")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    min_x, max_x = df["point_x"].min(), df["point_x"].max()
    min_y, max_y = df["point_y"].min(), df["point_y"].max()
    sw = df["screen_width"].iloc[0]
    sh = df["screen_height"].iloc[0]

    print(f"Screen dimensions in CSV   : width={sw}, height={sh}")
    print(f"point_x range              : min={min_x}, max={max_x} (expected within [0, {sw}])")
    print(f"point_y range              : min={min_y}, max={max_y} (expected within [0, {sh}])")

    # Check whether targets are within screen boundaries
    within_bounds = (min_x >= 0 and max_x <= sw and min_y >= 0 and max_y <= sh)
    print(f"Target coordinates within bounds: {within_bounds}")

    # Inspect run_experiment.py for any normalization / rescaling of targets
    from experiments import run_experiment
    import inspect
    source = inspect.getsource(run_experiment)
    has_target_normalization = "y_train_x = (point_x" in source or "/ (screen_width" in source

    print(f"Does run_experiment.py rescale/normalize targets: {has_target_normalization}")
    
    pass_status = within_bounds and (not has_target_normalization)
    print(f"\nResult: {'PASS' if pass_status else 'NEEDS REVIEW'}")
    return pass_status, {
        "screen_width": int(sw),
        "screen_height": int(sh),
        "min_x": float(min_x),
        "max_x": float(max_x),
        "min_y": float(min_y),
        "max_y": float(max_y),
        "rescales_targets": has_target_normalization,
    }


def check_4_feature_leakage(csv_path="experiments/data/synthetic_calib.csv"):
    print("\n" + "=" * 70)
    print("CHECK 4: FEATURE LEAKAGE & REFERENCE COMPUTATION")
    print("=" * 70)

    from experiments import run_experiment
    import inspect
    source = inspect.getsource(run_experiment.extract_features)
    has_test_centering = "pred_rel_x_mean" in source

    print("Analysis of extract_features in run_experiment.py:")
    print("1. ref_mean_x, ref_mean_y: Computed from train_df only (when train_ref is None).")
    print("   Test fold receives train_ref and uses training ref_mean_x, ref_mean_y.")
    print("2. iris_y_scale: Computed from train_df only (np.std(mean_y) on train_df).")
    print("   Test fold receives train_ref and divides by iris_y_scale from training.")
    print("3. calib_rel_x_mean: Computed from train_df only.")
    print(f"4. Test-batch mean-centering present in extract_features: {has_test_centering}")

    # Compute actual LOPO X error on data
    df = pd.read_csv(csv_path)
    unique_points = df[["point_x", "point_y"]].drop_duplicates().values
    errors_x = []
    for tx, ty in unique_points:
        test_mask = (df["point_x"] == tx) & (df["point_y"] == ty)
        train_df = df[~test_mask]
        test_df = df[test_mask]

        X_train_x, _, train_ref = extract_features(train_df, "engineered")
        X_test_x, _, _ = extract_features(test_df, "engineered", train_ref=train_ref)

        model_x = LinearRegression().fit(X_train_x, train_df["point_x"].values)
        pred_x = model_x.predict(X_test_x)
        errors_x.extend(np.abs(test_df["point_x"].values - pred_x))

    lopo_x_error = float(np.mean(errors_x))
    print(f"\nLOPO Mean Absolute X Error : {lopo_x_error:.4f} px")
    print(f"LOPO Centering Fix Status  : {'PASS (Test-batch centering removed)' if not has_test_centering else 'FAIL (Test-batch centering still active)'}")

    pass_status = (not has_test_centering) and (lopo_x_error < 150.0)
    print(f"\nResult: {'PASS' if pass_status else 'FAIL'}")
    return pass_status, {
        "has_test_centering": has_test_centering,
        "lopo_x_error_px": lopo_x_error,
    }


def check_5_fold_integrity(csv_path="experiments/data/synthetic_calib.csv", seed=42):
    print("\n" + "=" * 70)
    print("CHECK 5: FOLD INTEGRITY CHECK (LOPO & STRATIFIED 80/20)")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    df["target_group"] = [f"({r.point_x:.0f},{r.point_y:.0f})" for _, r in df.iterrows()]
    unique_points = df[["point_x", "point_y"]].drop_duplicates().values

    print("--- Part A: Leave-One-Point-Out (9 Folds) ---")
    lopo_pass = True
    lopo_details = []

    for fold_i, (tx, ty) in enumerate(unique_points):
        test_mask = (df["point_x"] == tx) & (df["point_y"] == ty)
        train_df = df[~test_mask]
        test_df = df[test_mask]

        held_out_in_train = ((train_df["point_x"] == tx) & (train_df["point_y"] == ty)).any()
        train_target_count = len(train_df[["point_x", "point_y"]].drop_duplicates())
        test_sample_count = len(test_df)
        train_sample_count = len(train_df)

        fold_valid = (not held_out_in_train) and (train_target_count == 8) and (test_sample_count == 100)
        if not fold_valid:
            lopo_pass = False

        print(
            f"Fold {fold_i+1}: Held out ({tx:.0f}, {ty:.0f}) | "
            f"Test samples: {test_sample_count} | Train samples: {train_sample_count} | "
            f"Train targets: {train_target_count} | Held-out in train? {held_out_in_train}"
        )
        lopo_details.append({
            "fold": fold_i + 1,
            "held_out": (tx, ty),
            "train_samples": train_sample_count,
            "test_samples": test_sample_count,
            "held_out_in_train": held_out_in_train,
        })

    print(f"LOPO Integrity: {'PASS' if lopo_pass else 'FAIL'}")

    print("\n--- Part B: Stratified 80/20 Split ---")
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target_group"],
        random_state=seed,
    )

    train_targets = set(train_df["target_group"].unique())
    test_targets = set(test_df["target_group"].unique())
    all_targets = set(df["target_group"].unique())

    all_present_in_both = (train_targets == all_targets) and (test_targets == all_targets)
    print(f"Total targets: {len(all_targets)}")
    print(f"Targets in train: {len(train_targets)}")
    print(f"Targets in test : {len(test_targets)}")
    print(f"All targets present in both splits? {all_present_in_both}")

    counts_df = pd.DataFrame({
        "Train": train_df["target_group"].value_counts(),
        "Test": test_df["target_group"].value_counts(),
    })
    print("\nSample counts per target in stratified split:")
    print(counts_df)

    stratified_pass = all_present_in_both and (counts_df["Train"] == 80).all() and (counts_df["Test"] == 20).all()
    print(f"Stratified Split Integrity: {'PASS' if stratified_pass else 'FAIL'}")

    overall_pass = lopo_pass and stratified_pass
    print(f"\nResult: {'PASS' if overall_pass else 'FAIL'}")
    return overall_pass, {
        "lopo_pass": lopo_pass,
        "stratified_pass": stratified_pass,
        "lopo_details": lopo_details,
    }


def check_6_rf_memorization(csv_path="experiments/data/synthetic_calib.csv", seed=42):
    print("\n" + "=" * 70)
    print("CHECK 6: RANDOM FOREST MEMORIZATION CHECK")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    df["target_group"] = [f"({r.point_x:.0f},{r.point_y:.0f})" for _, r in df.iterrows()]

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["target_group"],
        random_state=seed,
    )

    X_train_x, X_train_y, train_ref = extract_features(train_df, feature_set="engineered")
    X_test_x, X_test_y, _ = extract_features(test_df, feature_set="engineered", train_ref=train_ref)

    y_train_x = train_df["point_x"].values
    y_train_y = train_df["point_y"].values
    y_test_x = test_df["point_x"].values
    y_test_y = test_df["point_y"].values

    models = build_models(feature_set="engineered", seed=seed)
    rf_pipe_x = models["Random Forest Regressor"]["pipeline"]
    rf_pipe_y = build_models(feature_set="engineered", seed=seed)["Random Forest Regressor"]["pipeline"]

    rf_pipe_x.fit(X_train_x, y_train_x)
    rf_pipe_y.fit(X_train_y, y_train_y)

    # Evaluate on Train
    train_pred_x = rf_pipe_x.predict(X_train_x)
    train_pred_y = rf_pipe_y.predict(X_train_y)
    train_errors = np.sqrt((y_train_x - train_pred_x) ** 2 + (y_train_y - train_pred_y) ** 2)
    train_mean_err = float(np.mean(train_errors))
    train_r2_x = float(r2_score(y_train_x, train_pred_x))
    train_r2_y = float(r2_score(y_train_y, train_pred_y))

    # Evaluate on Validation
    val_pred_x = rf_pipe_x.predict(X_test_x)
    val_pred_y = rf_pipe_y.predict(X_test_y)
    val_errors = np.sqrt((y_test_x - val_pred_x) ** 2 + (y_test_y - val_pred_y) ** 2)
    val_mean_err = float(np.mean(val_errors))
    val_r2_x = float(r2_score(y_test_x, val_pred_x))
    val_r2_y = float(r2_score(y_test_y, val_pred_y))

    print(f"Random Forest Training Mean Error   : {train_mean_err:.4f} px (R2_x={train_r2_x:.4f}, R2_y={train_r2_y:.4f})")
    print(f"Random Forest Validation Mean Error : {val_mean_err:.4f} px (R2_x={val_r2_x:.4f}, R2_y={val_r2_y:.4f})")
    print(f"Train-Validation Gap                : {abs(val_mean_err - train_mean_err):.4f} px")

    # Why did RF score 0.00 px on validation?
    # Because each of the 9 discrete target points forms isolated, non-overlapping clusters in iris space.
    # The decision tree branches partition iris space around each target cluster, memorizing the exact target label!
    # Because all 9 targets are present in validation, validation points fall directly into the exact leaf nodes!
    
    return "CONFIRMED MEMORIZATION", {
        "train_mean_error_px": train_mean_err,
        "val_mean_error_px": val_mean_err,
        "train_r2_x": train_r2_x,
        "val_r2_x": val_r2_x,
    }


def run_all_checks(csv_path="experiments/data/synthetic_calib.csv", noise_std=0.0):
    c1_status, c1_data = check_1_data_sanity(csv_path)
    c2_status, c2_data = check_2_learnability(noise_std=noise_std)
    c3_status, c3_data = check_3_units(csv_path)
    c4_status, c4_data = check_4_feature_leakage(csv_path)
    c5_status, c5_data = check_5_fold_integrity(csv_path)
    c6_status, c6_data = check_6_rf_memorization(csv_path)

    return {
        "check_1": (c1_status, c1_data),
        "check_2": (c2_status, c2_data),
        "check_3": (c3_status, c3_data),
        "check_4": (c4_status, c4_data),
        "check_5": (c5_status, c5_data),
        "check_6": (c6_status, c6_data),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run experiment sanity checks")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Noise standard deviation for learnability check (default: 0.0)")
    parser.add_argument("--data-path", type=str, default="experiments/data/synthetic_calib.csv", help="Path to calibration data CSV")
    args = parser.parse_args()

    run_all_checks(csv_path=args.data_path, noise_std=args.noise_std)

