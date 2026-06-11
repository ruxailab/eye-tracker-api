import numpy as np

from app.routes.session import convert_nan_to_none


def test_nan_and_inf_become_none():
    assert convert_nan_to_none(float("nan")) is None
    assert convert_nan_to_none(float("inf")) is None
    assert convert_nan_to_none(float("-inf")) is None


def test_regular_values_pass_through():
    assert convert_nan_to_none(3.14) == 3.14
    assert convert_nan_to_none(0.0) == 0.0
    assert convert_nan_to_none(42) == 42
    assert convert_nan_to_none("hello") == "hello"
    assert convert_nan_to_none(None) is None


def test_numpy_nan_and_inf():
    assert convert_nan_to_none(np.float64("nan")) is None
    assert convert_nan_to_none(np.float64("inf")) is None


def test_numpy_types_converted_to_python():
    result_f = convert_nan_to_none(np.float64(2.5))
    assert result_f == 2.5
    assert isinstance(result_f, float)

    result_i = convert_nan_to_none(np.int64(10))
    assert result_i == 10
    assert isinstance(result_i, int)


def test_handles_dicts_and_lists():
    assert convert_nan_to_none({"a": 1.0, "b": float("nan")}) == {"a": 1.0, "b": None}
    assert convert_nan_to_none([1.0, float("nan"), float("inf")]) == [1.0, None, None]


def test_handles_nested_structures():
    obj = {
        "outer": {
            "values": [1.0, float("nan")],
            "score": float("inf"),
        },
        "name": "test",
    }
    result = convert_nan_to_none(obj)
    assert result["outer"]["values"] == [1.0, None]
    assert result["outer"]["score"] is None
    assert result["name"] == "test"


def test_empty_containers_unchanged():
    assert convert_nan_to_none({}) == {}
    assert convert_nan_to_none([]) == []
