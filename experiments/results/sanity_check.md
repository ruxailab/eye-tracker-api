# 🩺 Sanity Check & Diagnostic Audit Report

**Date:** 2026-10-23  
**Target:** RUXAILAB Eye-Tracker API Experiment Suite (`experiments/`)  
**Scope:** Investigation of Random Forest 0.00 px error and LOPO 448–570 px errors.  
**Diagnostic Script:** [experiments/sanity_check.py](../sanity_check.py)

---

## 📌 Executive Summary

The investigation revealed that the synthetic data generator is mathematically sound and 100% learnable, but the experimental evaluation pipeline suffers from two structural flaws:

1. **The LOPO Error Explosion (~530 px error across models):** Caused by **test-time batch mean-centering** in [experiments/run_experiment.py:L124-L127](../run_experiment.py#L124-L127) (borrowed from [app/services/gaze_tracker.py:L580-L587](../../app/services/gaze_tracker.py#L580-L587)). In Leave-One-Point-Out cross-validation, the test set consists of samples from a *single* held-out target point. Subtracting the test batch mean erases the target's spatial offset from screen center, forcing all models to predict the screen center ($x=960$). Removing this single artifact drops LOPO horizontal error from **528.8 px down to 75.5 px**.
2. **The Random Forest 0.00 px Error (Stratified 80/20 Split):** Caused by **discrete cluster memorization**. Because calibration data consists of only 9 discrete screen coordinates, and the iris landmark clusters for these 9 targets are non-overlapping in feature space, an unconstrained tree ensemble (`max_depth=10`) isolates each cluster into a pure leaf node. In a stratified 80/20 split, test samples fall into the exact same leaves as their training counterparts, resulting in a discrete lookup table with $0.00\text{ px}$ error.

---

## 🔍 Detailed Checks

### Check 1: Data Sanity
* **Result:** **PASS**
* **Evidence:**
  * **File:** [experiments/data/synthetic_calib.csv](../data/synthetic_calib.csv)
  * **Shape:** $(900, 8)$
  * **Columns:** `['left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y', 'point_x', 'point_y', 'screen_height', 'screen_width']`
  * **Dtypes:** 6 columns `float64`, 2 columns `int64` (`screen_height`, `screen_width`).
  * **Missing Values:** Exactly **0 NaNs** across all columns.
  * **Targets:** Exactly 9 unique target points ($3 \times 3$ grid), with exactly **100 samples per target**:
    * Unique `point_x`: `[288.0, 960.0, 1632.0]` (at 15%, 50%, 85% of screen width 1920)
    * Unique `point_y`: `[162.0, 540.0, 918.0]` (at 15%, 50%, 85% of screen height 1080)
* **Bug Location:** None.

---

### Check 2: Learnability Check (Noiseless Linear Fit)
* **Result:** **PASS**
* **Evidence:**
  * Generated a noiseless copy of the synthetic calibration data (`noise_std=0.0`, `head_shift_ratio=0.0`) using [generate_synthetic_calibration_data](../synthetic_data.py#L18).
  * Fitted a standard `sklearn.linear_model.LinearRegression` on the 4 raw iris columns (`left_iris_x`, `left_iris_y`, `right_iris_x`, `right_iris_y`) to predict `point_x` and `point_y`:
    * **$R^2$ Score (X):** `1.00000000`
    * **$R^2$ Score (Y):** `1.00000000`
    * **Mean Euclidean Error:** `0.00000000 px`
    * **Max Euclidean Error:** `0.00000000 px`
  * **Conclusion:** The synthetic data generator's underlying transformation is strictly linear and mathematically coherent. It does not introduce non-invertible singularities.
* **Bug Location:** None in data generation mathematics.

---

### Check 3: Units & Scaling Check
* **Result:** **PASS**
* **Evidence:**
  * Target columns `point_x` and `point_y` in [experiments/data/synthetic_calib.csv](../data/synthetic_calib.csv) are stored directly in screen pixels:
    * `point_x`: min = `288.0 px`, max = `1632.0 px` (bounded inside `screen_width=1920`).
    * `point_y`: min = `162.0 px`, max = `918.0 px` (bounded inside `screen_height=1080`).
  * In [experiments/run_experiment.py:L186-L189](../run_experiment.py#L186-L189), target vectors are assigned directly without normalization:
    ```python
    y_train_x = train_df["point_x"].values
    y_train_y = train_df["point_y"].values
    ```
  * In [compute_metrics()](../run_experiment.py#L134-L149), Euclidean error is computed directly on raw target and prediction vectors:
    $\sqrt{(x_{\text{true}} - x_{\text{pred}})^2 + (y_{\text{true}} - y_{\text{pred}})^2}$.
  * The production rescaling bug from [app/services/gaze_tracker.py:L439-L447](../../app/services/gaze_tracker.py#L439-L447) (`y_train_x = (point_x - center) / (dim / 2)`) was not propagated into `run_experiment.py`. Reported errors are in unscaled screen pixels.
* **Bug Location:** None.

---

### Check 4: Feature Leakage & Reference Computation
* **Result:** **NEEDS REVIEW (Critical Alignment Bug Found)**
* **Evidence:**
  * **Feature Parameters:**
    * $\text{ref\_mean}_x, \text{ref\_mean}_y$: Computed strictly from `train_df` only ([run_experiment.py:L109-L113](../run_experiment.py#L109-L113)). Passed to test evaluation via `train_ref` dictionary. No pre-split leakage.
    * $\sigma_y$ (`iris_y_scale`): Computed strictly from `train_df` only ([run_experiment.py:L112](../run_experiment.py#L112)).
    * $\overline{\text{rel}}_{x,\text{calib}}$: Computed strictly from `train_df` only.
  * **Bug Identified in Test-Time Feature Alignment:**
    * **Code:** [experiments/run_experiment.py:L124-L127](../run_experiment.py#L124-L127):
      ```python
      pred_rel_x_mean = float(np.mean(rel_x))
      rel_x = rel_x - pred_rel_x_mean + train_ref.get("calib_rel_x_mean", 0.0)
      ```
    * **Mechanism:** This logic was originally written in [app/services/gaze_tracker.py:L580-L587](../../app/services/gaze_tracker.py#L580-L587) to center entire free-browsing user sessions where a participant sits off-center.
    * **Failure in LOPO:** In Leave-One-Point-Out cross-validation, `test_df` contains **only one target point**.
      Computing `pred_rel_x_mean = np.mean(rel_x)` on a single target point calculates that target's own position.
      Subtracting `pred_rel_x_mean` sets $\text{rel}_x \approx 0$, collapsing every held-out target point to screen center ($x=960$).
    * **Numerical Proof:**
      * LOPO X error with test-batch centering: **528.8 px**
      * LOPO X error without test-batch centering: **75.5 px**
* **Bug Location:** [experiments/run_experiment.py:L124-L127](../run_experiment.py#L124-L127).

---

### Check 5: Fold Integrity Check
* **Result:** **PASS**
* **Evidence:**
  * **Leave-One-Point-Out (LOPO):**
    * Evaluated all 9 folds.
    * In every fold $k$, the held-out point $(T_x, T_y)$ has exactly **100 test samples**.
    * The training set has exactly **800 samples** spanning the remaining 8 targets.
    * In all 9 folds, `held_out_in_train == False`. Fold isolation is 100% verified.
  * **Stratified 80/20 Split:**
    * Stratification by target group `(point_x, point_y)` ensures every one of the 9 targets is present in both splits.
    * Training set contains exactly **80 samples per target** (720 total).
    * Validation set contains exactly **20 samples per target** (180 total).
* **Bug Location:** None in folding or data slicing.

---

### Check 6: Random Forest Memorization Check
* **Result:** **CONFIRMED MEMORIZATION**
* **Evidence:**
  * Evaluated Random Forest (`n_estimators=100`, `max_depth=10`, `min_samples_split=5`) on the stratified 80/20 split:
    * **Training Mean Error:** `0.0000 px` ($R^2_x = 1.0000$, $R^2_y = 1.0000$)
    * **Validation Mean Error:** `0.0000 px` ($R^2_x = 1.0000$, $R^2_y = 1.0000$)
    * **Train-Validation Gap:** `0.0000 px`
  * **Explanation:**
    * In standard regression datasets, validation error is non-zero because the target variable is continuous.
    * In calibration datasets, the target variable $(P_x, P_y)$ only takes **9 discrete values**.
    * In iris feature space, the 9 target clusters are separated by $\approx 12.4\text{ px}$, while the sample jitter ($\sigma=0.35\text{ px}$) and head shift ($2.8\text{ px}$) keep each cluster within a $\pm 3.5\text{ px}$ envelope.
    * As a result, the 9 target clusters in feature space have **zero overlap**.
    * With `max_depth=10`, decision trees partition the feature space into 9 pure leaf nodes. In a stratified split, validation samples for target $k$ fall into the identical leaf nodes as training samples for target $k$, outputting the exact constant target coordinate.
    * Under LOPO (where target $k$ is not in the training set), Random Forest cannot extrapolate outside its axis-aligned training boxes, causing error to spike to $448.0\text{ px}$.
* **Bug Location:** Methodological mismatch between discrete calibration targets and unconstrained decision tree depth.

---

## 🛠️ Proposed Fixes (Awaiting Confirmation)

| Issue | File & Line | Proposed Fix |
| :--- | :--- | :--- |
| **1. LOPO Centering Collapse** | [experiments/run_experiment.py:L124-L127](../run_experiment.py#L124-L127) | Remove test-time batch mean-centering (`pred_rel_x_mean`). Test samples should use the fixed reference baseline $(\text{ref\_mean}_x, \text{ref\_mean}_y)$ established during training without subtracting their own batch mean. This immediately restores LOPO error to realistic values ($\approx 75\text{ px}$). |
| **2. Random Forest Memorization** | [experiments/run_experiment.py:L170-L177](../run_experiment.py#L170-L177) | Regularize Random Forest to prevent pure single-cluster leaf formation: restrict `max_depth=4` or `max_depth=5`, increase `min_samples_leaf=10`, and evaluate continuous interpolation behavior. |
| **3. Realistic Landmark Jitter** | [experiments/synthetic_data.py:L21](../synthetic_data.py#L21) | Increase Gaussian landmark noise from $\sigma=0.35\text{ px}$ to $\sigma=0.8-1.2\text{ px}$ to reflect real-world MediaPipe Face Mesh iris jitter across webcams, preventing trivial decision-tree separability. |
| **4. Raw Feature Mode SVR Scaling** | [experiments/run_experiment.py:L142-L178](../run_experiment.py#L142-L178) | Ensure `StandardScaler()` is consistently applied across all raw-feature polynomial pipelines so degree-2 polynomial expansion does not cause numerical scale explosion on raw pixel inputs (range 200–600). |
