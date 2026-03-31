# Hyperparameters for the models
hyperparameters = {
    "Lasso Regression": {
        "param_grid": {
            "lasso__alpha": [10, 20, 30, 40, 45, 50, 55, 100, 200, 500]
        }
    },
    "Ridge Regression": {
        "param_grid": {
            "ridge__alpha": [ 1e-3, 0.005, 0.01, 0.1, 0.5, 1.0, 10, 20, 50, 100]
        }
    },
    "Elastic Net": {
        "param_grid": {
            "elasticnet__alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
            "elasticnet__l1_ratio": [0.5, 0.7, 0.8, 0.9, 1.0],
        }
    },
    "Bayesian Ridge": {
        "param_grid": {
            "bayesianridge__alpha_init": [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.9],
            "bayesianridge__lambda_init": [1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        }
    },
    "SGD Regressor": {
        "param_grid": {
            "sgdregressor__alpha": [0.0001, 0.001],
            "sgdregressor__l1_ratio": [0.5, 0.7, 0.8, 1],
            "sgdregressor__max_iter": [1000],
            "sgdregressor__eta0": [0.0001, 0.001],
        }
    },
    "Support Vector Regressor": {
        "param_grid": {
            "svr__C": [50, 100, 200, 500, 1000, 2000],
            "svr__gamma": [0.1, 0.5, 1, 2, 5],
            "svr__kernel": ["rbf"],
        }
    },
    "Random Forest Regressor": {
        "param_grid": {
            "randomforestregressor__n_estimators": [100],
            "randomforestregressor__max_depth": [10],
            "randomforestregressor__min_samples_split": [2, 5, 10],
        }
    },
}
