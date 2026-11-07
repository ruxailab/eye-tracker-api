"""Reproducible, leakage-aware calibration model benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = (
    "left_iris_x",
    "left_iris_y",
    "right_iris_x",
    "right_iris_y",
)
TARGET_COLUMNS = ("point_x", "point_y")
MODEL_FACTORIES = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
}
MODEL_ALIASES = {
    "linear regression": "linear",
    "ridge regression": "ridge",
    "lasso regression": "lasso",
}


@dataclass(frozen=True)
class CalibrationConfig:
    """Options defining one reproducible benchmark configuration."""

    feature_representation: str = "raw"
    test_size: float = 0.2
    random_state: int = 42
    n_splits: int = 5
    tune_hyperparameters: bool = True
    model_names: tuple[str, ...] = ("linear", "ridge", "lasso")
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    lasso_alphas: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.1, 1.0)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_dataset_paths() -> list[Path]:
    data_dirs = (
        _repository_root() / "app" / "services" / "calib_validation" / "csv" / "data",
        _repository_root() / "app" / "services" / "calib_validation" / "csv",
    )
    return [
        path
        for directory in data_dirs
        if directory.exists()
        for path in sorted(directory.glob("*.csv"))
    ]


def load_calibration_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate one calibration CSV using a path independent of CWD."""

    requested = Path(path)
    candidates = [requested] if requested.is_absolute() else [
        Path.cwd() / requested,
        _repository_root() / requested,
    ]
    resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(f"Calibration dataset not found: {path}")

    frame = pd.read_csv(resolved)
    aliases = {"screen_x": "point_x", "screen_y": "point_y"}
    frame = frame.rename(columns={key: value for key, value in aliases.items()})
    required = set(FEATURE_COLUMNS + TARGET_COLUMNS)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Calibration dataset is missing columns: {', '.join(missing)}")

    numeric_columns = list(FEATURE_COLUMNS + TARGET_COLUMNS)
    frame.loc[:, numeric_columns] = frame.loc[:, numeric_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    result = frame.dropna(subset=numeric_columns).reset_index(drop=True)
    if result.empty:
        raise ValueError(f"Calibration dataset contains no usable rows: {resolved}")
    return result


def _groups(frame: pd.DataFrame) -> np.ndarray:
    for column in ("fixation_id", "group", "sample_id"):
        if column in frame:
            return frame[column].to_numpy()
    return frame.loc[:, list(TARGET_COLUMNS)].astype(str).agg("|".join, axis=1).to_numpy()


def make_validation_splits(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic group splits so one fixation cannot cross folds."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    groups = _groups(frame)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two calibration groups are required")

    if n_splits == 1:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        return list(splitter.split(frame, groups=groups))

    folds = min(n_splits, len(unique_groups))
    if folds < 2:
        raise ValueError("At least two calibration groups are required")
    splitter = GroupKFold(n_splits=folds)
    return list(splitter.split(frame, groups=groups))


def _features(frame: pd.DataFrame, representation: str) -> pd.DataFrame:
    if representation == "raw":
        return frame.loc[:, list(FEATURE_COLUMNS)]
    if representation == "mean_diff":
        return pd.DataFrame(
            {
                "mean_x": (frame.left_iris_x + frame.right_iris_x) / 2,
                "diff_x": frame.left_iris_x - frame.right_iris_x,
                "mean_y": (frame.left_iris_y + frame.right_iris_y) / 2,
                "diff_y": frame.left_iris_y - frame.right_iris_y,
            },
            index=frame.index,
        )
    raise ValueError("feature_representation must be 'raw' or 'mean_diff'")


def calculate_metrics(
    y_true: Sequence[Sequence[float]] | np.ndarray,
    y_pred: Sequence[Sequence[float]] | np.ndarray,
    groups: Iterable[object] | None = None,
) -> dict[str, float | int | None]:
    """Calculate comparable pixel error and within-fixation stability metrics."""

    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError("y_true and y_pred must both have shape (n_samples, 2)")
    errors = prediction - truth
    distances = np.linalg.norm(errors, axis=1)
    result: dict[str, float | int | None] = {
        "samples": int(len(truth)),
        "mean_error_px": float(np.mean(distances)),
        "median_error_px": float(np.median(distances)),
        "horizontal_mae_px": float(mean_absolute_error(truth[:, 0], prediction[:, 0])),
        "vertical_mae_px": float(mean_absolute_error(truth[:, 1], prediction[:, 1])),
        "precision_px": None,
        "stability_std_px": None,
    }
    if groups is not None:
        group_values = np.asarray(list(groups))
        if len(group_values) != len(truth):
            raise ValueError("groups must have one value per sample")
        centers = pd.DataFrame(prediction).groupby(group_values).transform("mean").to_numpy()
        spread = np.linalg.norm(prediction - centers, axis=1)
        result["precision_px"] = float(np.mean(spread))
        result["stability_std_px"] = float(np.std(spread))
    return result


def _pipeline(model_name: str) -> Pipeline:
    model_key = MODEL_ALIASES.get(model_name.lower(), model_name.lower())
    if model_key not in MODEL_FACTORIES:
        raise ValueError(f"Unsupported model: {model_name}")
    return Pipeline(
        [("scaler", StandardScaler()), ("model", MODEL_FACTORIES[model_key]())]
    )


def _fit_axis(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: np.ndarray,
    config: CalibrationConfig,
) -> Pipeline:
    estimator = _pipeline(model_name)
    model_key = MODEL_ALIASES.get(model_name.lower(), model_name.lower())
    if not config.tune_hyperparameters or model_key == "linear":
        return estimator.fit(x_train, y_train)
    unique_groups = len(np.unique(groups))
    if unique_groups < 2:
        return estimator.fit(x_train, y_train)
    cv = GroupKFold(n_splits=min(config.n_splits, unique_groups))
    values = config.ridge_alphas if model_key == "ridge" else config.lasso_alphas
    search = GridSearchCV(estimator, {"model__alpha": values}, cv=cv, scoring="neg_mean_absolute_error")
    search.fit(x_train, y_train, groups=groups)
    return search.best_estimator_


def benchmark_dataset(
    dataset: str | Path | pd.DataFrame,
    config: CalibrationConfig | None = None,
) -> list[dict[str, object]]:
    """Evaluate configured models on every leakage-safe validation split."""

    config = config or CalibrationConfig()
    frame = load_calibration_dataset(dataset) if not isinstance(dataset, pd.DataFrame) else dataset.copy()
    x = _features(frame, config.feature_representation)
    y = frame.loc[:, list(TARGET_COLUMNS)]
    groups = _groups(frame)
    splits = make_validation_splits(
        frame, config.test_size, config.random_state, config.n_splits
    )
    results: list[dict[str, object]] = []
    for model_name in config.model_names:
        fold_metrics = []
        for train_indices, test_indices in splits:
            train_groups = groups[train_indices]
            x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            x_model = _fit_axis(model_name, x_train, y_train.point_x, train_groups, config)
            y_model = _fit_axis(model_name, x_train, y_train.point_y, train_groups, config)
            prediction = np.column_stack(
                [x_model.predict(x_test), y_model.predict(x_test)]
            )
            fold_metrics.append(calculate_metrics(y_test, prediction, groups[test_indices]))
        numeric_keys = [key for key, value in fold_metrics[0].items() if isinstance(value, (int, float))]
        summary = {
            key: float(np.mean([metric[key] for metric in fold_metrics if metric[key] is not None]))
            for key in numeric_keys
        }
        results.append(
            {
                "dataset": str(dataset) if not isinstance(dataset, pd.DataFrame) else "<dataframe>",
                "model": model_name,
                "feature_representation": config.feature_representation,
                "folds": len(splits),
                **summary,
            }
        )
    return results


def benchmark_datasets(
    datasets: Iterable[str | Path] | None = None,
    config: CalibrationConfig | None = None,
) -> pd.DataFrame:
    """Run a configuration across datasets and return a results table."""

    paths = list(datasets) if datasets is not None else _default_dataset_paths()
    if not paths:
        raise FileNotFoundError("No calibration CSV files were found")
    rows = [row for path in paths for row in benchmark_dataset(path, config)]
    return pd.DataFrame(rows)


__all__ = [
    "CalibrationConfig",
    "benchmark_dataset",
    "benchmark_datasets",
    "calculate_metrics",
    "load_calibration_dataset",
    "make_validation_splits",
]
