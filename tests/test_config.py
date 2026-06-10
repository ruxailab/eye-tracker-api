from app.services.config import hyperparameters


TUNABLE_MODELS = [
    "Lasso Regression", "Ridge Regression", "Elastic Net",
    "Bayesian Ridge", "SGD Regressor",
    "Support Vector Regressor", "Random Forest Regressor",
]


def test_all_tunable_models_present():
    for model in TUNABLE_MODELS:
        assert model in hyperparameters


def test_param_grids_are_valid():
    for name, config in hyperparameters.items():
        assert "param_grid" in config
        grid = config["param_grid"]
        assert len(grid) > 0
        for param, values in grid.items():
            assert isinstance(values, list), f"{name}.{param} should be a list"


def test_ridge_alpha_values():
    alphas = hyperparameters["Ridge Regression"]["param_grid"]["ridge__alpha"]
    assert min(alphas) > 0
    assert len(alphas) >= 5


def test_linear_regression_not_included():
    assert "Linear Regression" not in hyperparameters
