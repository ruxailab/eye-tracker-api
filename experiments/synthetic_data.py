"""
experiments/synthetic_data.py
-----------------------------
Generates a realistic synthetic 9-point eye-tracker calibration dataset
matching the schema defined in tests/conftest.py and app/routes/session.py.

Schema columns:
    left_iris_x, left_iris_y, right_iris_x, right_iris_y,
    point_x, point_y, screen_height, screen_width
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_synthetic_calibration_data(
    num_samples_per_point: int = 100,
    screen_width: int = 1920,
    screen_height: int = 1080,
    noise_std: float = 1.0,
    head_shift_ratio: float = 0.5,
    head_shift_x: float = 2.8,
    head_shift_y: float = -1.6,
    seed: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Generate synthetic iris coordinates for a 9-point calibration grid.

    Parameters:
        num_samples_per_point: Number of frames recorded for each calibration point.
        screen_width: Screen width in pixels (e.g. 1920).
        screen_height: Screen height in pixels (e.g. 1080).
        noise_std: Standard deviation of Gaussian sensor/landmark jitter in pixels (default: 1.0).
        head_shift_ratio: Fraction of samples per target subjected to head drift.
        head_shift_x: Horizontal head drift offset applied to iris landmarks.
        head_shift_y: Vertical head drift offset applied to iris landmarks.
        seed: Random seed for reproducibility.
        output_path: Optional CSV filepath to write dataset to.

    Returns:
        pd.DataFrame containing the generated calibration dataset.
    """
    np.random.seed(seed)

    # 3x3 Calibration target grid (points at 15%, 50%, 85% of screen width and height)
    grid_x = [
        int(round(0.15 * screen_width)),
        int(round(0.50 * screen_width)),
        int(round(0.85 * screen_width)),
    ]
    grid_y = [
        int(round(0.15 * screen_height)),
        int(round(0.50 * screen_height)),
        int(round(0.85 * screen_height)),
    ]

    targets = [(x, y) for y in grid_y for x in grid_x]

    # Baseline iris positions in camera coordinate frame (640x480 resolution)
    # MediaPipe face mesh landmarks typical coordinate space:
    # Left eye center ~ 360px, Right eye center ~ 280px (IPD ~ 80px), Y ~ 240px
    base_left_x = 360.0
    base_right_x = 280.0
    base_left_y = 240.0
    base_right_y = 240.0

    # Sensitivity coefficients: pixels of iris movement per pixel of screen gaze
    # Horizontal gaze rotation: ~0.018 iris px / screen px
    # Vertical gaze rotation: ~0.012 iris px / screen px
    kx_left = 0.0185
    kx_right = 0.0180
    ky_left = 0.0120
    ky_right = 0.0122

    center_screen_x = screen_width / 2.0
    center_screen_y = screen_height / 2.0

    records = []

    for pt_idx, (target_x, target_y) in enumerate(targets):
        # Base gaze offset from screen center
        dx_screen = target_x - center_screen_x
        dy_screen = target_y - center_screen_y

        nominal_lx = base_left_x + dx_screen * kx_left
        nominal_rx = base_right_x + dx_screen * kx_right
        nominal_ly = base_left_y + dy_screen * ky_left
        nominal_ry = base_right_y + dy_screen * ky_right

        head_shift_count = int(round(num_samples_per_point * head_shift_ratio))

        for sample_i in range(num_samples_per_point):
            # Apply head-shift offset to the second half of samples
            if sample_i >= (num_samples_per_point - head_shift_count):
                hs_x = head_shift_x
                hs_y = head_shift_y
            else:
                hs_x = 0.0
                hs_y = 0.0

            # Add zero-mean Gaussian jitter
            jitter_lx = np.random.normal(0, noise_std)
            jitter_ly = np.random.normal(0, noise_std)
            jitter_rx = np.random.normal(0, noise_std)
            jitter_ry = np.random.normal(0, noise_std)

            records.append({
                "left_iris_x": nominal_lx + hs_x + jitter_lx,
                "left_iris_y": nominal_ly + hs_y + jitter_ly,
                "right_iris_x": nominal_rx + hs_x + jitter_rx,
                "right_iris_y": nominal_ry + hs_y + jitter_ry,
                "point_x": float(target_x),
                "point_y": float(target_y),
                "screen_height": int(screen_height),
                "screen_width": int(screen_width),
            })

    # Expected column order matching conftest.py and session route
    columns = [
        "left_iris_x",
        "left_iris_y",
        "right_iris_x",
        "right_iris_y",
        "point_x",
        "point_y",
        "screen_height",
        "screen_width",
    ]

    df = pd.DataFrame(records)[columns]

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} synthetic calibration samples across {len(targets)} targets.")
        print(f"Saved dataset to: {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic eye-tracking calibration data.")
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/data/synthetic_calib.csv",
        help="Target output CSV file path.",
    )
    parser.add_argument(
        "--samples-per-point",
        type=int,
        default=100,
        help="Number of calibration frames per target dot (default: 100).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=1.0,
        help="Standard deviation of Gaussian sensor/landmark jitter in pixels (default: 1.0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Screen width in pixels (default: 1920).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Screen height in pixels (default: 1080).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    args = parser.parse_args()

    generate_synthetic_calibration_data(
        num_samples_per_point=args.samples_per_point,
        screen_width=args.width,
        screen_height=args.height,
        noise_std=args.noise_std,
        seed=args.seed,
        output_path=args.output,
    )
