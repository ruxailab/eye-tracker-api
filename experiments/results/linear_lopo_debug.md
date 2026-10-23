# 🔬 Linear Model LOPO Evaluation Diagnostic Audit

**Target:** RUXAILAB Eye-Tracker API Experiment Suite (`experiments/`)  
**Scope:** Investigation of Linear Model ~114–116 px Leave-One-Point-Out (LOPO) errors on synthetic calibration data.  
**Diagnostic Script:** [experiments/sanity_check.py](../sanity_check.py)

---

## 📌 Executive Summary

The investigation confirmed that **the evaluation pipeline in `experiments/run_experiment.py` has no mathematical or alignment bugs**. 
When evaluated on a noiseless synthetic dataset (`noise_std=0.0`, `head_shift_ratio=0.0`), the LOPO evaluation for Linear Regression achieves an exact **0.000000 px** mean Euclidean error across all 9 held-out folds.

The ~114–116 px LOPO error observed on the benchmark dataset is not an evaluation pipeline artifact; it is the mathematically expected physical response of a 2-feature linear eye tracker to two specific properties of the synthetic data generator:
1. **Geometric Sensitivity Ratio:** The simulated iris sensitivity ($k_x \approx 0.01825\text{ px/px}$, $k_y \approx 0.0121\text{ px/px}$) has an inverse magnification factor of $\approx 54.8\text{ screen px / iris px}$ in X and $\approx 82.6\text{ screen px / iris px}$ in Y. A $1.0\text{ px}$ landmark sensor noise directly creates an error floor of $\approx 39\text{ px}$ in X and $\approx 60\text{ px}$ in Y.
2. **Unobservable Head Shift:** The generator injects a $+2.8\text{ px}$ head translation into 50% of the frames. Because a 2-feature model (`diff_x`, `rel_x`) cannot distinguish head translation from eye rotation under noise, this creates an unobserved $\pm 76.7\text{ px}$ bimodal displacement in X.

---

## 🔍 Detailed Diagnostic Checks

### Section 1: Task 1 — Noiseless LOPO Check
* **Status:** **PASS**
* **Procedure:** Generated a noiseless synthetic dataset (`num_samples_per_point=100`, `noise_std=0.0`, `head_shift_ratio=0.0`) and executed `evaluate_leave_one_point_out()` using `Linear Regression` with engineered features.
* **Evidence:**
  * **Mean Euclidean Error:** `0.000000 px`
  * **RMSE:** `0.000000 px` (`RMSE_X = 0.000000 px`, `RMSE_Y = 0.000000 px`)
  * **Accuracy @ 50px:** `100.00%`
  * **Accuracy @ 100px:** `100.00%`
  * **Sample Held-Out Point `(288.0, 162.0)` Predictions:**
    * Sample 0: true = $(288.0, 162.0)$, pred = $(288.0000, 162.0000)$, error = $0.0000\text{ px}$
    * Sample 1: true = $(288.0, 162.0)$, pred = $(288.0000, 162.0000)$, error = $0.0000\text{ px}$
    * Sample 2: true = $(288.0, 162.0)$, pred = $(288.0000, 162.0000)$, error = $0.0000\text{ px}$
* **Conclusion:** The evaluation pipeline correctly trains, isolates folds, transforms features, and predicts held-out coordinates with zero error under deterministic linear conditions.

---

### Section 2: Task 2 — Noisy LOPO with Linear Model Only
* **Status:** **PASS**
* **Procedure:** Executed LOPO on the standard benchmark dataset (`noise_std=1.0`, `head_shift_ratio=0.5`). Separated X and Y errors across all 9 folds and decomposed the contribution of sensor noise vs. head shift.
* **Evidence:**
  * **Mean Euclidean Error:** `116.18 px` (with degree-2 polynomial expansion; `114.14 px` with plain linear regression)
  * **Overall RMSE:** `125.00 px`
  * **Horizontal Errors (X):**
    * `RMSE_X`: `86.43 px`
    * `Mean Absolute Error (X)`: `77.81 px` ($\sigma = 37.63\text{ px}$)
  * **Vertical Errors (Y):**
    * `RMSE_Y`: `90.30 px`
    * `Mean Absolute Error (Y)`: `74.04 px` ($\sigma = 51.70\text{ px}$)
  * **Ablation Sweep (Linear Regression):**
    * `noise=0.0, hs=0.0`: Mean Error = `0.00 px` (`RMSE_X = 0.00 px`, `RMSE_Y = 0.00 px`)
    * `noise=0.0, hs=0.5`: Mean Error = `0.00 px` (In noiseless data, $hs_x$ cancels out in `diff_x = lx - rx`, allowing perfect fit)
    * `noise=1.0, hs=0.0`: Mean Error = `62.49 px` (`RMSE_X = 37.26 px`, `RMSE_Y = 60.63 px`)
    * `noise=1.0, hs=0.5`: Mean Error = `116.18 px` (`RMSE_X = 86.43 px`, `RMSE_Y = 90.30 px`)
* **Explanation of Difference:**
  1. Landmark noise $\sigma = 1.0\text{ px}$ destroys the micro-signal in `diff_x` (whose entire screen span is only $0.672\text{ px}$). The linear model shifts weight entirely to `rel_x = mean_x - ref_mean_x`.
  2. Because `rel_x` directly absorbs the $+2.8\text{ px}$ head translation present in 50% of frames, head-shifted frames suffer a $+97.7\text{ px}$ offset while non-head-shifted frames suffer a $-72.1\text{ px}$ offset, establishing the $\approx 78\text{ px}$ horizontal error floor.

---

### Section 3: Task 3 — Feature Transform Check
* **Status:** **PASS**
* **Procedure:** Inspected feature matrices for held-out corner point $(288, 162)$ and verified training references vs. test transformations. Checked feature boundaries across all 9 folds.
* **Evidence:**
  * **Training Reference Stats (Fold `(288, 162)`):**
    * `ref_mean_x`: `322.947412` (mean of training `(lx + rx)/2`)
    * `ref_mean_y`: `239.811525` (mean of training `(ly + ry)/2`)
    * `iris_y_scale`: `3.727168` (std of training `mean_y` $+ 10^{-6}$)
    * `calib_rel_x_mean`: `0.000000`
  * **Test Feature Formulation:**
    * $\text{diff}_x = lx_{\text{test}} - rx_{\text{test}}$
    * $\text{rel}_x = \text{mean\_x}_{\text{test}} - \text{ref\_mean\_x}_{\text{train}}$
    * $\text{diff\_y\_norm} = (ly_{\text{test}} - ry_{\text{test}}) / \text{iris\_y\_scale}_{\text{train}}$
    * $\text{rel\_y\_norm} = (\text{mean\_y}_{\text{test}} - \text{ref\_mean\_y}_{\text{train}}) / \text{iris\_y\_scale}_{\text{train}}$
    * **Confirmed:** No test-batch mean or test-batch scaling is computed. Test features are 100% referenced to the training fold baseline.
  * **Feature Extrapolation Range Audit (Fold `(288, 162)`):**
    * `diff_x`: Train $[75.52, 84.60]$, Test $[77.16, 83.31]$ $\rightarrow$ **Inside training range**
    * `rel_x`: Train $[-16.5346, 14.1392]$, Test $[-16.5392, -10.6092]$ $\rightarrow$ **Marginal boundary excursion ($0.0046\text{ px}$)**
    * `diff_y_norm`: Train $[-1.2465, 1.1704]$, Test $[-0.6613, 0.7599]$ $\rightarrow$ **Inside training range**
    * `rel_y_norm`: Train $[-2.2214, 1.8333]$, Test $[-2.0727, -0.6167]$ $\rightarrow$ **Inside training range**
  * **Across all 9 folds:** Marginal sensor jitter excursions ($0.005$–$0.15\text{ px}$) occur at the extreme outer screen borders ($x=288, 1632$ and $y=162, 918$), requiring slight linear extrapolation at screen boundaries, but within expected Gaussian noise limits.

---

### Section 4: Task 4 — Y Normalization Check
* **Status:** **PASS**
* **Procedure:** Verified that `iris_y_scale` ($\sigma_y$) is derived strictly from training data and applied identically to test data.
* **Evidence:**
  * Training fold: $\sigma_{y,\text{train}} = \text{std}(\text{mean\_y}_{\text{train}}) + 10^{-6} = 3.727168$.
  * Reference dictionary: `train_ref['iris_y_scale'] = 3.727168`.
  * Manual calculation on test fold:
    * `test_diff_y_norm = (ly_test - ry_test) / 3.727168` matches `X_test_y[:, 0]` with max absolute difference `< 1e-15`.
    * `test_rel_y_norm = (mean_y_test - 239.811525) / 3.727168` matches `X_test_y[:, 1]` with max absolute difference `< 1e-15`.
* **Conclusion:** Scaling is constant and strictly computed from the training split only.

---

### Section 5: Task 5 — Target Check
* **Status:** **PASS**
* **Procedure:** Inspected ground-truth targets and prediction vectors in training and test routines.
* **Evidence:**
  * Training targets: `y_train_x` in $[288.0, 1632.0]\text{ px}$, `y_train_y` in $[162.0, 918.0]\text{ px}$ (`float64`).
  * Test targets: `y_test_x` in $[288.0, 1632.0]\text{ px}$, `y_test_y` in $[162.0, 918.0]\text{ px}` (`float64`).
  * Model predictions: `pred_x`, `pred_y` output in screen pixels (range $\approx [200, 1700]\text{ px}$).
  * Metric evaluation: `compute_metrics()` computes Euclidean distance directly between raw targets and raw predictions: $\sqrt{(x_{\text{true}} - x_{\text{pred}})^2 + (y_{\text{true}} - y_{\text{pred}})^2}$.
  * No target normalization, division by screen width, or one-sided scaling is applied.
* **Conclusion:** Target and prediction units are identical screen pixels.

---

## 🛠️ Identified Inconsistencies & Proposed Adjustments (Awaiting Confirmation)

No bugs were found in the LOPO evaluation mechanism itself. However, one model pipeline discrepancy was identified:

| Issue | File & Line | Observation | Proposed Adjustment (Pending Confirmation) |
| :--- | :--- | :--- | :--- |
| **Polynomial Features in Linear Regression** | [experiments/run_experiment.py:L169](../run_experiment.py#L169) | `Linear Regression` has `"use_poly": True`, while all other linear models (`Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`, `SGD`) have `"use_poly": (feature_set == "raw")` (which evaluates to `False` for engineered features). Degree-2 expansion slightly overfits landmark noise, raising LOPO error from $114.14\text{ px}$ to $116.18\text{ px}$. | Change `"use_poly": (feature_set == "raw")` for `Linear Regression` to align with the rest of the linear model family. |
