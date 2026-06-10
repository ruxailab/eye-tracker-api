import numpy as np
import pandas as pd

from app.services.metrics import (
    func_precision_x,
    func_presicion_y,
    func_accuracy_x,
    func_accuracy_y,
    func_total_accuracy,
)


def test_precision_is_zero_when_predictions_are_identical():
    df = pd.DataFrame({"Predicted X": [100.0, 100.0, 100.0]})
    assert func_precision_x(df) == 0.0

    df_y = pd.DataFrame({"Predicted Y": [200.0, 200.0, 200.0]})
    assert func_presicion_y(df_y) == 0.0


def test_precision_grows_with_spread():
    tight = pd.DataFrame({"Predicted X": [99.0, 100.0, 101.0]})
    wide = pd.DataFrame({"Predicted X": [80.0, 100.0, 120.0]})
    assert func_precision_x(tight) < func_precision_x(wide)


def test_precision_known_rms():
    df = pd.DataFrame({"Predicted X": [90.0, 100.0, 110.0]})
    expected = np.sqrt(np.mean([100.0, 0.0, 100.0]))
    assert abs(func_precision_x(df) - expected) < 1e-10


def test_accuracy_zero_when_perfect():
    df = pd.DataFrame({"True X": [100.0, 200.0], "Predicted X": [100.0, 200.0]})
    assert func_accuracy_x(df) == 0.0


def test_accuracy_rmse_value():
    df = pd.DataFrame({"True X": [100.0, 200.0], "Predicted X": [110.0, 190.0]})
    assert func_accuracy_x(df) == 10.0


def test_accuracy_y():
    df = pd.DataFrame({"True Y": [100.0, 200.0], "Predicted Y": [105.0, 195.0]})
    assert func_accuracy_y(df) == 5.0


def test_total_accuracy_uses_euclidean_distance():
    df = pd.DataFrame({
        "True X": [100.0], "Predicted X": [103.0],
        "True Y": [200.0], "Predicted Y": [204.0],
    })
    assert abs(func_total_accuracy(df) - 5.0) < 1e-10


def test_total_accuracy_averages_multiple_points():
    df = pd.DataFrame({
        "True X": [0.0, 0.0], "Predicted X": [3.0, 0.0],
        "True Y": [0.0, 0.0], "Predicted Y": [4.0, 5.0],
    })
    assert abs(func_total_accuracy(df) - 5.0) < 1e-10


def test_total_accuracy_on_sample_data(sample_metrics_df):
    assert func_total_accuracy(sample_metrics_df) > 0
