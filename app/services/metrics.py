import numpy as np


def func_precision_x(group):
    """
    Calculate the precision for the X axis.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the X axis.

    Returns:
        float: The precision value.
    """
    return np.sqrt(
        np.mean(np.square(group["Predicted X"] - np.mean(group["Predicted X"])))
    )

def func_presicion_y(group):
    """
    Calculate the precision for the Y axis.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the Y axis.

    Returns:
        float: The precision value.
    """
    return np.sqrt(
        np.mean(np.square(group["Predicted Y"] - np.mean(group["Predicted Y"])))
    )


def func_accuracy_x(group):
    """
    Calculate the accuracy for the X axis.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the X axis.

    Returns:
        float: The accuracy value.
    """
    
    return np.sqrt(np.mean(np.square(group["True X"] - group["Predicted X"])))



def func_accuracy_y(group):
    """
    Calculate the accuracy for the Y axis.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the Y axis.

    Returns:
        float: The accuracy value.
    """
    return np.sqrt(np.mean(np.square(group["True Y"] - group["Predicted Y"])))

def func_total_accuracy(group):
    """
    Calculate the total accuracy for the X and Y axes.

    Args:
        group (pandas.DataFrame): A group of data containing the predicted and true values for the X and Y axes.

    Returns:
        float: The total accuracy value(eculidean distance).

    """
    distances = np.sqrt(
        np.square(group["True X"] - group["Predicted X"]) + 
        np.square(group["True Y"] - group["Predicted Y"])
    )
    return np.mean(distances) # Returns average error in pixels


"""
EyeTrackingBenchmark

Implements:
- Spatial accuracy (Euclidean pixel error)
- Angular accuracy (visual angle degrees)
- Temporal precision (frame-to-frame RMS)
- Data quality (sample loss percentage)
- Per-target spatial breakdown

Designed for benchmarking eye-tracking systems across devices and setups.
"""


class EyeTrackingBenchmark:

    def __init__(self, df, screen_width_px, screen_width_cm, viewing_distance_cm):
        required_cols = {"True X", "True Y", "Predicted X", "Predicted Y"}

        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Expected columns: {required_cols}"
            )

        self.df = df
        self.screen_width_px = screen_width_px
        self.screen_width_cm = screen_width_cm
        self.viewing_distance_cm = viewing_distance_cm

    def _euclidean_error_px(self):
        dx = self.df["True X"] - self.df["Predicted X"]
        dy = self.df["True Y"] - self.df["Predicted Y"]
        return np.sqrt(dx**2 + dy**2)

    def accuracy_metrics(self):
        errors = self._euclidean_error_px()
        return {
            "mean_accuracy_error_px": float(np.mean(errors)),
            "median_accuracy_error_px": float(np.median(errors)),
            "p95_accuracy_error_px": float(np.percentile(errors, 95)),
            "mean_accuracy_error_deg": float(self._mean_error_deg(errors))
        }


    def _mean_error_deg(self, errors_px):
        pixel_size_cm = self.screen_width_cm / self.screen_width_px
        errors_cm = errors_px * pixel_size_cm
        errors_deg = 2 * np.degrees(
            np.arctan(errors_cm / (2 * self.viewing_distance_cm))
        )
        return np.mean(errors_deg)

    def precision_metrics(self):
        grouped = self.df.groupby(["True X", "True Y"])

        rms_values = []

        for (_, _), group in grouped:
            x = group["Predicted X"].values
            y = group["Predicted Y"].values

            dx = x - np.mean(x)
            dy = y - np.mean(y)

            rms = np.sqrt(np.mean(dx ** 2 + dy ** 2))
            rms_values.append(rms)

        return {
            "mean_rms_px": float(np.mean(rms_values)),
            "median_rms_px": float(np.median(rms_values))
        }

    def data_quality_metrics(self):
        total = len(self.df)
        missing = self.df["Predicted X"].isna().sum()
        return {
            "data_loss_percent": float((missing / total) * 100),
            "num_samples": int(total)
        }

    def evaluate(self):
        return {
            "accuracy": self.accuracy_metrics(),
            "precision": self.precision_metrics(),
            "data_quality": self.data_quality_metrics()
        }

    def evaluate_per_target(self):
        grouped = self.df.groupby(["True X", "True Y"])

        results = []

        for (tx, ty), group in grouped:
            bench = EyeTrackingBenchmark(
                group,
                self.screen_width_px,
                self.screen_width_cm,
                self.viewing_distance_cm
            )

            metrics = bench.accuracy_metrics()

            results.append({
                "true_x": float(tx),
                "true_y": float(ty),
                **metrics
            })

        return results