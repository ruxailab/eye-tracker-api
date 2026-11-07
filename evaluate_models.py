"""Run the offline calibration benchmark.

Examples:
    python evaluate_models.py
    python evaluate_models.py --dataset app/services/calib_validation/csv/data/01_fixed_train_data.csv --json results.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.services.calib_validation.benchmark import (
    CalibrationConfig,
    benchmark_datasets,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare calibration regression models.")
    parser.add_argument("--dataset", action="append", dest="datasets", help="CSV path; repeatable")
    parser.add_argument("--feature-representation", choices=("raw", "mean_diff"), default="raw")
    parser.add_argument("--no-tuning", action="store_true", help="Skip grouped hyperparameter tuning")
    parser.add_argument("--csv", type=Path, help="Write results to CSV")
    parser.add_argument("--json", type=Path, help="Write results to JSON")
    args = parser.parse_args()

    config = CalibrationConfig(
        feature_representation=args.feature_representation,
        tune_hyperparameters=not args.no_tuning,
    )
    results = benchmark_datasets(args.datasets, config)
    print(results.to_string(index=False))
    if args.csv:
        results.to_csv(args.csv, index=False)
    if args.json:
        results.to_json(args.json, orient="records", indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
