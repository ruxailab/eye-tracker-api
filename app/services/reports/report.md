# 🎯 Eye-Gaze Calibration Regression Study

This project evaluates multiple regression models for **eye-gaze calibration**, aiming to map iris landmark coordinates to screen positions.

The goal is to determine which regression model provides the best trade-off between:

- 🎯 Prediction accuracy (screen point error)
- 📏 Precision
- ⚡ Execution speed
- 🧠 Generalization (overfitting behavior)

---

## 📊 Dataset Overview

- Total samples: **900**
- Features:
  - `left_iris_x`, `left_iris_y`
  - `right_iris_x`, `right_iris_y`
- Targets:
  - `point_x`, `point_y`
- Calibration points detected: **9 unique points (k = 9)**

---

## 🧠 Models Evaluated

The following regression models were tested:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Bayesian Ridge
- SGD Regressor
- Support Vector Regressor (SVR)
- Random Forest Regressor

---

## ⚙️ Evaluation Metrics

The pipeline reports:

- **Avg Accuracy** → positional error (lower is better)
- **Avg Precision**
- **Execution Time**
- Axis-wise R² scores during calibration

Hyperparameter tuning was performed using **GridSearchCV** when available.

---

# 🧪 Experiment 1 — Baseline Results

Initial run before pipeline modifications.

### 🔎 Key Observations

- Linear & Ridge produced stable baseline performance.
- Elastic Net was very fast but less precise.
- SVR achieved strong R² values.
- Random Forest failed due to missing configuration.

### 📋 Performance Summary

| Model | Avg Accuracy | Avg Precision | Time (s) |
|------|-------------|---------------|---------|
| Linear Regression | 235.37 | 49.11 | 1.56 |
| Ridge Regression | 236.13 | 48.81 | 0.67 |
| Lasso Regression | 290.82 | 45.88 | 0.77 |
| Elastic Net | 298.97 | 38.66 | 0.11 |
| Bayesian Ridge | 235.39 | 48.12 | 1.69 |
| SGD Regressor | 241.21 | 46.88 | 15.31 |
| Support Vector Regressor | 266.18 | 48.07 | 0.13 |
| Random Forest | ❌ Error | — | — |

---

# 🧪 Experiment 2 — Pipeline Improvements

Changes made:

- Added Random Forest configuration
- Expanded hyperparameter grids
- Improved pipeline stability

### 📋 Updated Performance Summary

| Model | Avg Accuracy | Avg Precision | Time (s) |
|------|-------------|---------------|---------|
| Linear Regression | 235.37 | 49.11 | 1.50 |
| Ridge Regression | 235.87 | 48.71 | 0.39 |
| Lasso Regression | 292.79 | 45.10 | 0.41 |
| Elastic Net | 298.97 | 38.66 | 0.13 |
| Bayesian Ridge | 235.39 | 48.12 | 1.68 |
| SGD Regressor | 241.24 | 46.94 | 3.78 |
| Support Vector Regressor | 266.18 | 48.07 | 0.36 |
| Random Forest | **52.85** | 31.96 | 4.43 |

### 💡 Insights

- Random Forest dramatically reduced positional error.
- Precision dropped, suggesting sensitivity or instability.
- Ridge became faster after optimization.
- SVR remained a strong non-linear alternative.

---

# 🧪 Experiment 3 — Train/Validation Split (Overfitting Check)

To evaluate generalization:

- Training: **765 samples**
- Validation: **135 samples**

Each model went through:

1. Calibration phase (internal split)
2. Validation phase (hold-out set)

---

## 📋 Train vs Validation Results

| Model | Calib Acc | Calib Prec | Valid Acc | Valid Prec | Time (s) |
|------|-----------|------------|-----------|------------|---------|
| Linear Regression | 245.73 | 49.84 | 229.45 | 51.55 | 1.55 |
| Ridge Regression | 247.09 | 49.94 | 229.64 | 51.80 | 0.42 |
| Lasso Regression | 251.61 | 47.05 | 265.08 | 44.32 | 0.44 |
| Elastic Net | 301.30 | 41.36 | 294.61 | 38.25 | 0.09 |
| Bayesian Ridge | 246.86 | 50.26 | 229.11 | 50.94 | 1.69 |
| SGD Regressor | 247.04 | 49.60 | 239.26 | 48.56 | 3.98 |
| Support Vector Regressor | 281.02 | 49.79 | 257.47 | 47.05 | 0.22 |
| Random Forest | **69.50** | 46.49 | **46.65** | 29.15 | 6.57 |

---

## 🔍 Overfitting Analysis

### ✅ Stable Models
- Linear Regression
- Ridge Regression
- Bayesian Ridge

These models showed consistent calibration and validation behavior.

### ⚠️ Potential Overfitting
- Random Forest achieved lowest error but suffered large precision drop.
- Indicates high capacity and sensitivity to dataset structure.

### ⚡ Best Balance
- SVR provided a strong balance between:
  - Accuracy
  - Speed
  - Generalization

---

# 🏆 Final Findings

### 🥇 Best Raw Accuracy
**Random Forest Regressor**

- Lowest positional error
- Higher computation cost
- Possible overfitting

### 🥈 Most Stable Models
- Linear Regression
- Ridge Regression

### 🥉 Best Overall Trade-off
**Support Vector Regressor (SVR)**

---

# 🚀 Future Improvements

Possible next steps:

- Feature scaling & normalization experiments
- Temporal smoothing for gaze stability
- Ensemble methods (Linear + Non-linear)
- Neural network-based gaze regression
- Real-time latency benchmarking

---

# 📌 Conclusion

This study shows that:

- Non-linear models improve gaze estimation accuracy.
- Random Forest can greatly reduce error but may overfit.
- Linear models remain strong baselines for robustness.
- Proper train/validation splits are essential for realistic performance evaluation.

---

