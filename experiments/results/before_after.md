# 🔄 Benchmark Before vs. After Comparison Report

**Target:** RUXAILAB Eye-Tracker API Model Benchmark (`experiments/`)  
**Evaluation Dates:** Previous Benchmark (Before fixes) vs. Current Benchmark (After fixes)  
**Evaluated Models:** 10 regression pipelines across Stratified 80/20 and Leave-One-Point-Out (LOPO) splits.

---

## 📝 What Changed and Why

Three targeted corrections were implemented to resolve the synthetic benchmark anomalies identified during the diagnostic audit:
1. **Removed Test-Batch Centering in Feature Extraction (`experiments/run_experiment.py`):** In the original implementation (inherited from `app/services/gaze_tracker.py`), test features were normalized by subtracting the test batch's mean relative position. In Leave-One-Point-Out evaluation, each test fold contains samples from only a single screen target; subtracting the single-point batch mean collapsed its coordinates to screen center, forcing all models to predict near $(960, 540)$ and causing artificial LOPO errors of $448$–$570\text{ px}$. With test features now referenced strictly against the baseline learned during training (`ref_mean_x`, `ref_mean_y`), LOPO error across linear and regularized models dropped dramatically from $\approx 535\text{ px}$ to $\approx 114\text{ px}$ (and from $448\text{ px}$ to $8.58\text{ px}$ for Random Forest).
2. **Regularized Random Forest (`experiments/run_experiment.py`):** The decision tree ensemble was constrained to `max_depth=5` and `min_samples_leaf=10` (from `max_depth=10`, `min_samples_leaf=1`). Because calibration targets form 9 isolated spatial clusters in feature space, unconstrained trees memorized the discrete coordinates like a lookup table, yielding an artificial $0.00\text{ px}$ error on stratified hold-out sets. Constraining tree depth prevents trivial single-cluster leaf formation, restoring a realistic train-validation gap ($7.14\text{ px}$ train vs. $2.54\text{ px}$ validation).
3. **Increased Landmark Jitter (`experiments/synthetic_data.py`):** The default MediaPipe landmark Gaussian noise was raised from $\sigma=0.35\text{ px}$ to $\sigma=1.0\text{ px}$ (with a configurable `--noise-std` argument), reflecting real-world webcam jitter and eliminating artificially sharp cluster boundaries. Consequently, baseline stratified errors across linear models shifted from $\approx 101\text{ px}$ to $\approx 115\text{ px}$ without harming noiseless learnability ($R^2 = 1.000$ at $\sigma=0$).

---

## 📊 Comparison Table: Stratified 80/20 Split

| Model | Mean Error (Before) | Mean Error (After) | RMSE (Before) | RMSE (After) | Acc @ 50px (Before) | Acc @ 50px (After) | Acc @ 100px (Before) | Acc @ 100px (After) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.00 px** | **2.54 px** | 0.00 px | 11.44 px | 100.00% | 98.89% | 100.00% | 99.44% |
| **SGD Regressor** | 100.90 px | 115.23 px | 102.81 px | 120.85 px | 0.00% | 2.22% | 48.89% | 33.33% |
| **Ridge Regression** | 100.95 px | 115.25 px | 102.83 px | 120.86 px | 0.00% | 2.22% | 48.89% | 33.89% |
| **Bayesian Ridge** | 101.01 px | 115.28 px | 102.84 px | 120.87 px | 0.00% | 2.22% | 48.33% | 33.89% |
| **Lasso Regression** | 101.01 px | 115.29 px | 102.85 px | 120.88 px | 0.00% | 2.22% | 48.33% | 33.89% |
| **Linear Regression** | 100.87 px | 115.37 px | 102.75 px | 121.04 px | 0.00% | 2.22% | 48.89% | 35.00% |
| **Support Vector Regressor (RBF)** | 112.17 px | 127.29 px | 127.78 px | 142.93 px | 14.44% | 14.44% | 36.67% | 31.11% |
| **Support Vector Regressor (Linear)** | 119.25 px | 132.56 px | 133.66 px | 145.90 px | 11.67% | 8.89% | 38.33% | 32.22% |
| **Elastic Net** | 205.86 px | 219.62 px | 223.21 px | 235.26 px | 3.89% | 2.22% | 14.44% | 12.22% |
| **Support Vector Regressor (Poly)** | 575.98 px | 575.95 px | 629.48 px | 629.45 px | 11.11% | 11.11% | 11.11% | 11.11% |

---

## 📊 Comparison Table: Leave-One-Point-Out (LOPO) Split

| Model | Mean Error (Before) | Mean Error (After) | RMSE (Before) | RMSE (After) | Acc @ 50px (Before) | Acc @ 50px (After) | Acc @ 100px (Before) | Acc @ 100px (After) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **448.00 px** | **8.58 px** | 548.69 px | 44.77 px | 33.33% | 96.78% | 33.33% | 97.89% |
| **Ridge Regression** | **534.68 px** | **114.14 px** | 620.13 px | 122.95 px | 0.11% | 7.56% | 16.78% | 40.00% |
| **SGD Regressor** | **534.53 px** | **114.14 px** | 619.96 px | 122.96 px | 0.11% | 7.56% | 17.00% | 39.89% |
| **Bayesian Ridge** | **534.78 px** | **114.14 px** | 620.24 px | 122.93 px | 0.11% | 7.33% | 16.67% | 40.00% |
| **Lasso Regression** | **534.81 px** | **114.15 px** | 620.28 px | 122.93 px | 0.11% | 7.56% | 16.67% | 40.00% |
| **Linear Regression** | **537.46 px** | **116.18 px** | 623.25 px | 125.00 px | 0.00% | 6.33% | 16.11% | 38.67% |
| **Support Vector Regressor (RBF)** | **525.25 px** | **134.45 px** | 603.72 px | 149.27 px | 2.89% | 10.67% | 12.11% | 31.56% |
| **Support Vector Regressor (Linear)** | **522.52 px** | **145.37 px** | 604.03 px | 160.00 px | 4.22% | 6.44% | 15.22% | 31.11% |
| **Elastic Net** | **529.62 px** | **248.77 px** | 598.34 px | 267.03 px | 2.44% | 2.22% | 8.00% | 9.22% |
| **Support Vector Regressor (Poly)** | **571.74 px** | **564.88 px** | 625.72 px | 617.14 px | 11.11% | 11.11% | 11.11% | 11.11% |

---

## 🏆 Top 3 Model Ranking Dynamics

### Stratified 80/20 Split:
- **Before:** #1 Random Forest ($0.00\text{ px}$ - memorized), #2 Linear Regression ($100.87\text{ px}$), #3 SGD Regressor ($100.90\text{ px}$).
- **After:** #1 Random Forest ($2.54\text{ px}$ - regularized), #2 SGD Regressor ($115.23\text{ px}$), #3 Ridge Regression ($115.25\text{ px}$).
- **Change:** Linear Regression fell from #2 to #6 (though separated from #2 by less than $0.15\text{ px}$), while SGD Regressor rose to #2 and Ridge Regression entered the Top 3 at #3.

### Leave-One-Point-Out (LOPO) Split:
- **Before:** #1 Random Forest ($448.00\text{ px}$), #2 SVR Linear ($522.52\text{ px}$), #3 SVR RBF ($525.25\text{ px}$).
- **After:** #1 Random Forest ($8.58\text{ px}$), #2 Ridge Regression ($114.14\text{ px}$), #3 SGD Regressor ($114.14\text{ px}$).
- **Change:** Under the previous centering bug, SVR models appeared falsely superior because their bounded predictions clustered nearer to screen center. Once test-batch centering was removed, the linear regularized family (Ridge, SGD, Bayesian Ridge, Lasso) cleanly outperformed SVR models, causing SVR Linear to plummet from #2 to #8 ($145.37\text{ px}$) and SVR RBF to fall from #3 to #7 ($134.45\text{ px}$). Ridge Regression and SGD Regressor surged into the Top 3 alongside Random Forest.
