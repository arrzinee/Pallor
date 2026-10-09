# Pallor — Model Experiment Record

> **Research prototype. Not a validated diagnostic device. Do not use it to diagnose or rule out anemia.**
> No clinical validation has been performed. Dataset licence and provenance are **pending verification** ([`docs/DATASET.md`](../docs/DATASET.md)).

This document is the technical record of every model trained for Pallor. It covers why each was tried, how it was implemented, how it performed, and what we concluded. Every number is copied from a result file in [`outputs/`](../outputs/), and every configuration is taken from the scripts in [`scripts/`](../scripts/). Anything that was not recorded is marked as such.

*Last updated: 2026-10-09. The broader narrative report is [`docs/REPORT.md`](../docs/REPORT.md); this document goes deeper on the models.*

**How to read this document.** Results come from three kinds of evaluation, and they are kept separate throughout:

| Label | Meaning |
|---|---|
| **[Nested CV]** | Nested cross-validation on the 200-image training split. The 7 classical classifiers were all scored on the same 25 outer folds. |
| **[CV, other protocol]** | Non-nested or 5-fold cross-validation on the training split (original baseline, CNNs, Ridge). Not directly comparable with [Nested CV]. |
| **[Test]** | One scoring of a frozen model on the 51-image held-out test set. |

Numbers are rounded to 3 decimals, half up, from the 4-decimal values stored in the result files (exact fractions are used where they are known, for example accuracy 37/51 = 0.725).

---

## Contents

1. [Project context](#1-project-context)
2. [Experimental methodology](#2-experimental-methodology)
3. [Models tested](#3-models-tested)
   - [3.1 Logistic Regression](#31-logistic-regression)
   - [3.2 RBF Support Vector Machine](#32-rbf-support-vector-machine)
   - [3.3 Random Forest](#33-random-forest)
   - [3.4 Extra Trees](#34-extra-trees)
   - [3.5 HistGradientBoostingClassifier](#35-histgradientboostingclassifier)
   - [3.6 XGBoost](#36-xgboost)
   - [3.7 CatBoost](#37-catboost)
   - [3.8 Frozen MobileNetV3-Large with Logistic Regression](#38-frozen-mobilenetv3-large-with-logistic-regression)
   - [3.9 Frozen EfficientNet-B0 with Logistic Regression](#39-frozen-efficientnet-b0-with-logistic-regression)
   - [3.10 Fine-tuned MobileNetV3-Large](#310-fine-tuned-mobilenetv3-large)
   - [3.11 Ridge Regression for hemoglobin](#311-ridge-regression-for-hemoglobin)
4. [Comparison tables](#4-comparison-tables)
5. [Feature importance and dataset bias](#5-feature-importance-and-dataset-bias)
6. [Final model decision](#6-final-model-decision)
7. [Reproducibility and project files](#7-reproducibility-and-project-files)
8. [Limitations and ethical interpretation](#8-limitations-and-ethical-interpretation)

---

## 1. Project context

### What Pallor is

Pallor is an experimental research prototype built for the ML Empowerment Build Challenge 3.0. It asks whether **photographs of the inner lower eyelid (palpebral conjunctiva) contain colour information that is statistically associated with anemia** in one labelled dataset.

### Why conjunctival colour might carry a signal

Clinicians sometimes look at the conjunctiva for pallor (paleness) as a physical sign of anemia. The red colour of this thin, well-vascularised tissue comes partly from hemoglobin in the blood beneath it, so lower hemoglobin could plausibly appear as less red, paler tissue. This is the motivating hypothesis only; **our experiments do not show that any model has learned a physiologically meaningful pallor signal** (see [Section 5](#5-feature-importance-and-dataset-bias)). Image colour is also strongly affected by lighting, camera and white balance, which are uncontrolled in this dataset.

### Dataset facts

All of these were verified by [`scripts/inspect_dataset.py`](../scripts/inspect_dataset.py) and are recorded in `derived/inspection_summary.json` (gitignored).

| Fact | Value |
|---|---|
| Source | Downloaded from Mendeley Data. DOI, licence and citation **unrecorded — pending verification** |
| Images used | **251** PNG files: **110** in `data/Anaemic/`, **141** in `data/Non-Anaemic/` |
| Not used | `data/All Categories/` (251 byte-identical copies of the two class folders) |
| Image type | Small, pre-segmented RGBA crops of conjunctiva. The background is fully transparent and stored as RGB (0,0,0). Median size 213 × 76 px; 246 distinct sizes |
| Labels | Per-image hemoglobin (Hb, g/dL) in `Cp-Anaemic Dataset.xlsx` (3 sheets, all consistent). Hb range 4.2–15.0, mean 11.0 |
| Hb measurement method | **Unrecorded** |
| Patient identifiers | **None.** The workbook's `group_id` is unique per image and does not identify patients |
| Integrity | Every file maps to exactly one workbook row; 0 unreadable images; 0 exact duplicates |

### Classification target and the Hb < 11 g/dL convention

The binary target is `anemic = 1` if Hb < 11.0 g/dL, else 0. This is the **dataset's own labelling convention**: every image in the Anaemic folder has Hb ≤ 10.8 and every image in the Non-Anaemic folder has Hb ≥ 11.0, so the rule reproduces the folder labels exactly (251/251).

**This threshold is not a universal diagnostic rule.** Clinical anemia cut-offs differ by age, sex, pregnancy status and other factors. 11 g/dL matches some guideline values (for example, for young children or pregnant women), but the population behind this dataset is undocumented.

### Classification versus hemoglobin regression

- **Binary classification** predicts the category (Hb < 11 or not). It is the **primary task**.
- **Regression** predicts the continuous Hb value in g/dL. It was tried once ([Section 3.11](#311-ridge-regression-for-hemoglobin)) and did not beat predicting the training-set mean. A category label does not imply that per-image Hb can be estimated.

### Why the experiments ran in this order

1. **Engineered colour features + logistic regression** came first. They are transparent, cheap, hard to overfit on 200 images, and directly encode the hypothesis that colour matters.
2. **Pretrained CNNs** came next, to check whether generic image representations capture more than hand-made colour statistics. They did not.
3. **Other classical classifiers** followed, to check whether non-linear models extract more from the same 35 features. Several scored higher in cross-validation, but no improvement was established on the test set.
4. **XGBoost and CatBoost** completed the search. Neither beat Extra Trees. **The model search is now closed.**

---

## 2. Experimental methodology

### 2.1 Feature extraction (classical models)

Code: [`scripts/features.py`](../scripts/features.py). All features are per-image statistics with **no learned parameters**, so computing them for all 251 images before splitting leaks nothing.

- **Tissue mask:** only fully opaque pixels count as tissue (`alpha == 255`, [`features.py:25`](../scripts/features.py#L25)). This excludes the transparent background (stored as black) and conservatively drops the soft mask boundary (0 < alpha < 255, about 5% of non-transparent pixels).
- **The 35 features:**

| Group | Features |
|---|---|
| RGB (15) | mean, std, and 10th/50th/90th percentiles of R, G and B (scaled to 0–1) |
| Chromaticity (2) | mean of R/(R+G+B) and G/(R+G+B) |
| HSV (6) | circular hue mean as `hue_sin_mean` and `hue_cos_mean` (red sits at both 0° and 360°, so hue is averaged as a vector); S mean and std; V mean and std |
| CIELAB (7) | L\*, a\*, b\* mean and std (sRGB → XYZ → Lab, D65, implemented in NumPy); `a_over_L` = mean a\* / mean L\* |
| Redness (4) | `erythema_index` = mean of 100·log10(R/G); `rg_norm_diff` = mean (R−G)/(R+G); `r_over_g_mean`; `frac_red_dominant` (fraction of pixels with R > G and R > B) |
| Size (1) | `n_tissue_px`: number of opaque pixels (a crop-size feature; see [Section 5](#5-feature-importance-and-dataset-bias)) |

The extractor was unit-tested during development on a synthetic image, against reference HSV and Lab values, and for the hue wrap-around. Those tests were run interactively and are **not stored as a test file** in the repository.

### 2.2 Image preprocessing (CNN models)

Code: [`scripts/cnn_common.py`](../scripts/cnn_common.py).

1. **Compositing:** RGBA is converted to RGB by standard alpha compositing over the ImageNet mean colour: `rgb·a + mean·(1−a)` ([`cnn_common.py:85`](../scripts/cnn_common.py#L85)). RGB is stored un-premultiplied (checked during development), so this is correct compositing, and after normalisation the **background is exactly 0**.
2. **Letterboxing:** each crop is padded to a square with the same mean colour (aspect ratio preserved), then resized to 224 × 224 (bilinear, antialiased).
3. **Normalisation:** fixed ImageNet mean/std constants. Nothing is fitted to our data.
4. **Augmentation (training folds of fine-tuning only):** horizontal flip (p = 0.5), rotation ±10°, translation ±5%, scale 0.9–1.1, with fill value 0 (the background). There are **no colour, brightness, hue or saturation transforms**, because tissue colour is the hypothesised signal. There are no vertical flips, which would not be a realistic view.

### 2.3 Train/test split

Code: [`scripts/train_baseline.py`](../scripts/train_baseline.py).

- **80/20 split, `SEED = 42`** ([`train_baseline.py:39`](../scripts/train_baseline.py#L39)), stratified on Hb bins with edges 8.0, 9.5, 11.0, 12.0 and 13.0 g/dL ([line 43](../scripts/train_baseline.py#L43)). Because 11.0 is a bin edge, the class balance is stratified too.

| Split | n | Anemic | Non-anemic | % anemic | Hb mean | Hb range |
|---|---|---|---|---|---|---|
| Train | 200 | 88 | 112 | 44.0% | 11.00 | 4.2–15.0 |
| Test | 51 | 22 | 29 | 43.1% | 11.05 | 5.9–14.9 |

- **Locked split:** the SHA-256 hash of `derived/split.csv` is stored in `derived/split.lock`. `train_baseline.py`, all CNN scripts, `diagnose_shortcut.py`, `classical_select.py`, `classical_evaluate.py` and `boosting_select.py` check this lock and abort on mismatch (`verify_split_lock()`, [`train_baseline.py:64`](../scripts/train_baseline.py#L64); [`cnn_common.py:57`](../scripts/cnn_common.py#L57)). `train_baseline.py` refuses to overwrite a locked split. **Exception:** `evaluate.py` (the baseline test evaluation) predates the lock. It only checks that `split.csv` matches the manifest's image order, not the hash.

### 2.4 Cross-validation protocols

| Protocol | Used for | Outer loop (performance) | Hyperparameter selection |
|---|---|---|---|
| **Non-nested repeated CV** | Original LR baseline; Ridge; frozen CNNs; size diagnostic | `RepeatedStratifiedKFold(5 splits × 5 repeats, seed 42)` = 25 folds | Grid search scored on the **same** 25 folds. The best mean is reported, which is mildly optimistic |
| **5-fold CV (repeat 1)** | Fine-tuned CNN | The first 5 of those 25 folds | Epoch count chosen from {10, 20, 30} on the same 5 folds |
| **Nested CV** | All 7 classical classifiers | The same 25 outer folds | An **inner** `StratifiedKFold(5, shuffle, seed 42)` grid search inside each outer training fold, so the outer fold never influences tuning ([`classical_select.py:95-106`](../scripts/classical_select.py#L95-L106)) |

All classification experiments reproduce `train_baseline.py`'s training-row order and stratify on the class label, so **the 25 outer folds are identical across every classification experiment**. **Exception:** Ridge regression stratifies its folds on Hb bins, so its 25 folds are different (checked). `boosting_select.py` checks this by re-running the nested LR and asserting that its 25 per-fold AUCs match the saved values ([`boosting_select.py:73`](../scripts/boosting_select.py#L73)).

**Preprocessing inside CV.** `StandardScaler` sits inside an sklearn `Pipeline` for LR, SVM and Ridge, so it is re-fitted on every (inner and outer) training fold. Tree models use unscaled features. CNN normalisation uses fixed ImageNet constants.

### 2.5 Decision thresholds

For every test-evaluated classifier, the threshold that turns a score into "anemic / not anemic" was chosen on the **training split only**: the maximum Youden's J (sensitivity + specificity − 1) on 5-fold out-of-fold (OOF) training predictions. The test set played no part in choosing any threshold.

### 2.6 Metrics

| Metric | Meaning here |
|---|---|
| **ROC-AUC** | Probability that a randomly chosen anemic image scores higher than a randomly chosen non-anemic one. Threshold-independent; 0.5 = chance |
| **Sensitivity** (recall) | Fraction of anemic images flagged: TP / (TP + FN) |
| **Specificity** | Fraction of non-anemic images correctly not flagged: TN / (TN + FP) |
| **Precision** | Fraction of flagged images that are truly anemic: TP / (TP + FP) |
| **F1** | Harmonic mean of precision and sensitivity |
| **Accuracy** | Fraction of all images classified correctly |
| **Confusion matrix** | TP, FN, FP, TN counts; positive = anemic |
| **MAE / RMSE** (regression) | Mean absolute / root-mean-square error in g/dL |
| **R²** | Variance explained relative to predicting the test-set mean; ≤ 0 means no better than the mean |

**Uncertainty:** [`scripts/evaluate.py`](../scripts/evaluate.py) computes **2000-resample percentile bootstrap 95% CIs** ([`evaluate.py:51-65`](../scripts/evaluate.py#L51-L65)). For classification, resampling is done within each class so that AUC is always defined. **Paired bootstrap** comparisons (candidate − LR) resample the *same* indices for both models, which is confirmed in [`classical_evaluate.py:68`](../scripts/classical_evaluate.py#L68) and [`cnn_evaluate.py:79`](../scripts/cnn_evaluate.py#L79). CV results are reported as mean ± SD across folds. The folds overlap, so this SD describes spread; it is not a standard error.

### 2.7 Leakage prevention

- Only the class folders are used; the duplicate `All Categories` folder is excluded.
- The test split is locked by hash and is never read by the selection scripts (`cnn_select.py`, `classical_select.py`, `boosting_select.py`).
- Each test evaluation was declared in advance with a fixed adoption rule, scored **once**, and its script refuses to run again (`cnn_evaluate.py`, `classical_evaluate.py`).
- Fitted preprocessing lives inside the CV pipelines, and thresholds come from training OOF predictions only.
- **Not prevented: patient-level leakage.** With no patient identifiers, images from one person may fall on both sides of the split ([Section 5](#5-feature-importance-and-dataset-bias)).

### 2.8 Preservation of results

- `outputs/baseline_colour/` is a **read-only snapshot** of the original baseline artifacts, with a `SHA256SUMS` file.
- New experiments write to new `cnn_*`, `classical_*` and `boosting_*` files. `classical_select.py` and `boosting_select.py` refuse to overwrite existing results.

### 2.9 The held-out test set has been used three times

The same 51 test images were used for three one-shot evaluations: the LR baseline, the frozen MobileNetV3, and Extra Trees. Each was pre-declared and run once, but the test set is **no longer an untouched final evaluation**: picking whichever experiment looks best on it would be selecting on the test set. XGBoost and CatBoost were therefore deliberately **not** test-evaluated.

---

## 3. Models tested

Classical models (3.1–3.7) all use the **35 colour features** and balanced class weights unless noted. The hyperparameter grids listed are the exact grids in the scripts.

### 3.1 Logistic Regression

**Status: primary model.** Family: linear classifier.

**How it works.** It learns one weight per (standardised) feature and passes the weighted sum through a sigmoid to produce a probability. The L2 penalty strength is controlled by C: smaller C means stronger regularisation.

**Why we used it.** It is the simplest credible classifier for 35 continuous colour features on 200 images. Its coefficients are inspectable, it has a single hyperparameter, and it serves as the reference for every comparison.

**Implementation**

| Item | Original baseline ([`train_baseline.py`](../scripts/train_baseline.py)) | Nested-CV reference ([`classical_select.py`](../scripts/classical_select.py)) |
|---|---|---|
| Pipeline | `StandardScaler` → `LogisticRegression(class_weight="balanced", max_iter=10000)` | same |
| C grid | 17 values, 10⁻³ to 10⁵ (`np.logspace(-3, 5, 17)`) | same |
| Tuning | Grid search on the 25 outer folds (non-nested) | Inner 5-fold grid search per outer fold |
| Selected C | **100** | Varied by outer fold, from 0.003 to 10⁵ (most frequent: 10⁵ in 6 folds, 100 in 5) |
| Threshold | **0.502** (Youden on 5-fold OOF training probabilities) | — (not test-evaluated) |
| Saved model | `models/classifier.joblib` (+ snapshot copy) | none |

**Results**

| Evaluation | ROC-AUC | Other metrics |
|---|---|---|
| [CV, other protocol] non-nested, 25 folds | 0.741 ± 0.079 | — |
| **[Nested CV]** 25 outer folds | **0.718 ± 0.077** (fold range 0.527–0.875) | Without `n_tissue_px`: 0.708 ± 0.070 |
| **[Test]** n = 51, model with C = 100 | **0.724** (95% CI 0.567–0.865) | Sensitivity **0.727** (0.545–0.909); specificity **0.690** (0.517–0.862); precision **0.640** (0.500–0.792); F1 **0.681** (0.533–0.809); accuracy 0.706 (0.569–0.824). Confusion matrix: **TP 16, FN 6, FP 9, TN 20** |

Note: the test result comes from the **original baseline model** (non-nested tuning, C = 100). The nested-CV figure is an estimate of the *procedure*, not of that specific saved model.

**Observed advantages.** Test AUC (0.724) is consistent with its CV estimates. It has the highest test sensitivity among the test-evaluated models (16/22). Its behaviour is stable and interpretable.

**Observed limitations.** It ranks last in nested CV. The tuned C was unstable across folds, which suggests a flat optimisation landscape with highly correlated features. The non-nested CV figure (0.741) was about 0.02 optimistic compared with nested CV.

**Interpretation.** It remains the **primary model** under the pre-declared rule ([Section 6](#6-final-model-decision)). That status reflects the evidence standard, not the best score.

### 3.2 RBF Support Vector Machine

**Status: promising CV candidate; not test-evaluated.** Family: kernel method.

**How it works.** It finds a maximum-margin boundary in a space implicitly defined by a radial-basis-function kernel, which allows a smooth non-linear boundary. C trades margin width against training errors; γ sets how local each training point's influence is.

**Why we used it.** It captures non-linear interactions between colour features without the variance of deep trees, and it is a standard strong baseline for small tabular datasets.

**Implementation** ([`classical_select.py:66`](../scripts/classical_select.py#L66)): `StandardScaler` → `SVC(kernel="rbf", class_weight="balanced")`. Grid: C ∈ {0.1, 1, 10, 100}, γ ∈ {"scale", 0.001, 0.01, 0.1} (16 combinations). Scores come from `decision_function`, because `probability=True` was **not** enabled, so this configuration produces no calibrated probabilities. Most frequent inner choices: C = 10 with γ = 0.1 (6 folds) or γ = "scale" (6 folds). No final model was fitted.

**Results.** **[Nested CV]** **0.781 ± 0.077** (fold range 0.652–0.934). Without `n_tissue_px`: 0.763 ± 0.082. Paired vs LR: +0.063 ± 0.078 mean ΔAUC, better in 21 of 25 folds. **[Test]** Not evaluated.

**Observed advantages.** Second-highest nested-CV AUC, and the fastest to evaluate (6.3 s for the full nested run).

**Observed limitations.** Removing the size feature costs 0.018 AUC, more than for Extra Trees (0.002). It has no probability output in this configuration, and it was not test-evaluated.

**Interpretation.** A credible alternative candidate supported only by CV evidence.

### 3.3 Random Forest

**Status: not selected.** Family: bagged decision-tree ensemble.

**How it works.** Many decision trees are each trained on a bootstrap sample, each considering a random subset of features at every split. Their votes are averaged.

**Why we used it.** It handles non-linear effects and feature interactions with little tuning and is robust to correlated features.

**Implementation** ([`classical_select.py:59-60`](../scripts/classical_select.py#L59-L60)): `RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)`. Grid: max_depth ∈ {None, 4}, min_samples_leaf ∈ {1, 3, 10}, max_features ∈ {"sqrt", 0.5} (12 combinations). Most frequent inner choice: max_depth None, "sqrt", leaf 1 (11 of 25 folds).

**Results.** **[Nested CV]** **0.743 ± 0.063** (fold range 0.596–0.890). Without `n_tissue_px`: 0.749 ± 0.060. Paired vs LR: +0.025 ± 0.079, better in 17 of 25 folds. **[Test]** Not evaluated.

**Interpretation.** It improved modestly over LR in CV but was clearly behind Extra Trees and the SVM.

### 3.4 Extra Trees

**Status: strongest experimental candidate.** Family: randomised decision-tree ensemble.

**How it works.** Like Random Forest, but split thresholds are drawn **at random** rather than optimised, and by default each tree sees the full training set. The extra randomness reduces variance, which can help on small, noisy datasets.

**Why we used it.** It is a lower-variance alternative to Random Forest, suited to 200 samples with correlated features.

**Implementation**

| Item | Value |
|---|---|
| Estimator | `ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42)` |
| Grid | Same as Random Forest (12 combinations) |
| Nested-CV inner choices | max_depth None, "sqrt", leaf 1 in **19 of 25** outer folds (fairly stable) |
| Final fit (full training split) | Inner 5-fold grid search → max_depth None, max_features "sqrt", min_samples_leaf 1. Inner-CV AUC on the full training set 0.815 (selection-biased; not a performance estimate) |
| Threshold | **0.44** (Youden on 5-fold OOF training probabilities; training OOF sensitivity 0.773, specificity 0.768) |
| Saved model | `models/classical_final.joblib` |

**Results**

| Evaluation | Result |
|---|---|
| **[Nested CV]** | **0.789 ± 0.061** (fold range 0.639–0.925). Without `n_tissue_px`: 0.787 ± 0.057. Paired vs LR: +0.071 ± 0.069, better in **23 of 25** folds |
| **[Test]** n = 51 | ROC-AUC **0.800** (0.668–0.914); sensitivity **0.682** (0.500–0.864); specificity **0.759** (0.586–0.897); precision **0.682** (0.536–0.850); F1 **0.682** (0.522–0.818); accuracy 0.725 (0.608–0.843). Confusion matrix: **TP 15, FN 7, FP 7, TN 22** |
| **[Test] paired vs LR** (ET − LR, 95% CI) | ΔAUC **+0.076 (−0.063 to 0.219)**; Δsensitivity −0.045 (−0.227 to 0.136); Δspecificity +0.069 (−0.103 to 0.241); Δprecision +0.042 (−0.112 to 0.193); ΔF1 +0.001 (−0.149 to 0.142); Δaccuracy +0.020 (−0.098 to 0.137) |

**Observed advantages.** Highest nested-CV AUC, the most consistent fold-level advantage over LR, essentially no dependence on the size feature, and a test AUC that agrees with its CV estimate.

**Observed limitations.** The test advantage over LR is **not statistically established** (the ΔAUC CI includes 0). At the chosen threshold it has one more false negative and two fewer false positives than LR, so its sensitivity and F1 are essentially the same. Its decisions are less interpretable than LR's. The saved model is larger (about 4.5 MB vs 3 KB).

**Interpretation.** Retained as the **leading experimental candidate**. Adopting it as primary would require new independent test data.

### 3.5 HistGradientBoostingClassifier

**Status: not selected.** Family: gradient-boosted decision trees (histogram-based, scikit-learn).

**How it works.** Shallow trees are added one after another, each correcting the errors of the current ensemble. Feature values are binned into histograms for speed.

**Implementation** ([`classical_select.py:71`](../scripts/classical_select.py#L71)): `HistGradientBoostingClassifier(class_weight="balanced", l2_regularization=1.0, early_stopping=False, random_state=42)`. Grid: learning_rate ∈ {0.03, 0.1}, max_depth ∈ {2, 3}, max_iter ∈ {100, 300}, min_samples_leaf ∈ {10, 20} (16 combinations).

**Results.** **[Nested CV]** **0.720 ± 0.070** (fold range 0.563–0.862). Without `n_tissue_px`: 0.715 ± 0.064. Paired vs LR: +0.002 ± 0.066, better in 14 of 25 folds. **[Test]** Not evaluated.

**Interpretation.** No meaningful gain over LR. Boosting did not help with this sample size and feature set.

### 3.6 XGBoost

**Status: not selected.** Family: gradient-boosted decision trees.

**Environment.** XGBoost's macOS wheel requires the OpenMP runtime. It could not load during the first classical comparison (that run's results file records it as unavailable). After the project owner approved it, `libomp` was installed system-wide with Homebrew, and **XGBoost 3.4.1** was installed into the project's virtual environment.

**Implementation** ([`boosting_select.py:40`](../scripts/boosting_select.py#L40)): `XGBClassifier(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss", random_state=42, n_jobs=1)`. **No class weighting.** Grid: n_estimators ∈ {100, 300}, max_depth ∈ {2, 3}, learning_rate ∈ {0.03, 0.1} (8 combinations). Same 35 features and the same nested protocol (fold identity verified).

**Results.** **[Nested CV]** **0.724 ± 0.073** (fold range 0.552–0.905). Without `n_tissue_px`: 0.707 ± 0.061. Paired vs LR: +0.006 ± 0.072, better in 14 of 25 folds. Paired vs Extra Trees: −0.065 ± 0.046, better in 2 of 25 folds. **[Test]** **Intentionally not evaluated**, because the test set had already been reused.

**Interpretation.** Comparable to LR and clearly behind Extra Trees.

### 3.7 CatBoost

**Status: not selected.** Family: gradient-boosted decision trees (ordered boosting, symmetric trees).

**Environment.** **CatBoost 1.2.10** was installed into the virtual environment; it needs no system dependency.

**Implementation** ([`boosting_select.py:44`](../scripts/boosting_select.py#L44)): `CatBoostClassifier(l2_leaf_reg=3.0, random_seed=42, thread_count=1, verbose=0, allow_writing_files=False)`. **No class weighting.** Grid: iterations ∈ {200, 500}, depth ∈ {3, 5}, learning_rate ∈ {0.03, 0.1} (8 combinations). Most frequent inner choice: depth 5, 200 iterations.

**Results.** **[Nested CV]** **0.762 ± 0.071** (fold range 0.604–0.898). Without `n_tissue_px`: 0.749 ± 0.056. Paired vs LR: +0.044 ± 0.075, better in 20 of 25 folds. Paired vs Extra Trees: −0.026 ± 0.031, better in 6 of 25 folds. **[Test]** **Intentionally not evaluated.**

**Interpretation.** The best of the boosting models, but it did not beat Extra Trees.

### 3.8 Frozen MobileNetV3-Large with Logistic Regression

**Status: tested once on the test set; not adopted.** Family: pretrained CNN feature extractor + linear classifier.

**How it works.** A convolutional network pretrained on ImageNet turns each image into a 960-dimensional feature vector (the globally pooled output of its convolutional layers). The network itself is **frozen** (no weights updated), and a logistic regression is trained on these vectors.

**Why we used it.** It is the most conservative form of transfer learning for 200 images: no CNN weights are trained, so there is little room to overfit at the network level. MobileNetV3 is small (about 5.5M parameters including its original classifier, measured during development) and ran quickly on Apple-silicon MPS.

**Implementation** ([`cnn_common.py`](../scripts/cnn_common.py), [`cnn_select.py`](../scripts/cnn_select.py))

| Item | Value |
|---|---|
| Weights | torchvision `MobileNet_V3_Large_Weights.IMAGENET1K_V2` (pretrained) |
| Backbone | Frozen; `features` → `avgpool` → 960-d vector ([`cnn_common.py:112-119`](../scripts/cnn_common.py#L112-L119)) |
| Input | Preprocessing as in [Section 2.2](#22-image-preprocessing-cnn-models); no augmentation |
| Classifier | `StandardScaler` → `LogisticRegression(class_weight="balanced", max_iter=20000)`; C grid `np.logspace(-5, 1, 13)` |
| Tuning | Grid search on the same 25 folds (**non-nested**); selected C = **0.001** |
| Threshold | **0.533** (Youden on 5-fold OOF training probabilities; training OOF sensitivity 0.511, specificity 0.804) |
| Selection | Chosen as the CNN representative by the pre-declared rule (best repeat-1 mean AUC among CNN candidates) |
| Saved model | `models/cnn_final.joblib`, `models/cnn_final_meta.json` |

**Results**

| Evaluation | Result |
|---|---|
| [CV, other protocol] non-nested, 25 folds | **0.716 ± 0.087** (repeat-1 5-fold: 0.681). The colour-feature LR on the same folds and protocol scored **0.741 ± 0.079** (repeat-1: 0.705) |
| **[Test]** n = 51 | ROC-AUC **0.660** (0.508–0.803); sensitivity **0.364** (0.182–0.591); specificity **0.759** (0.586–0.897); precision 0.533 (0.308–0.750); F1 **0.432** (0.229–0.619); accuracy 0.588 (0.451–0.706). Confusion matrix: **TP 8, FN 14, FP 7, TN 22** |
| **[Test] paired vs LR** | ΔAUC −0.064 (−0.257 to 0.129); Δsensitivity **−0.364 (−0.591 to −0.136)**; ΔF1 −0.248 (−0.453 to −0.051) |

**Interpretation.** It was worse than the colour baseline in CV and failed the adoption rule. On the test set it missed 14 of 22 anemic images, and its sensitivity CI lies entirely below the baseline's. Possible contributing factors (interpretations, not tested): a 960-dimensional generic embedding gives more room to overfit 200 images than 35 targeted features; ImageNet features are trained to be fairly invariant to colour and lighting; and letterboxing thin crops makes most of each input background.

### 3.9 Frozen EfficientNet-B0 with Logistic Regression

**Status: not selected; not test-evaluated.** Family: pretrained CNN feature extractor + linear classifier.

**Implementation.** Identical to 3.8 except for the backbone: torchvision `EfficientNet_B0_Weights.IMAGENET1K_V1`, frozen, 1280-d pooled features. Selected C = **3.16**.

**Results.** [CV, other protocol] non-nested, 25 folds: **0.649 ± 0.067** (repeat-1: 0.608 ± 0.055). **[Test]** Not evaluated.

**Interpretation.** It was the weakest frozen backbone, so it was not used as the base for fine-tuning.

### 3.10 Fine-tuned MobileNetV3-Large

**Status: not selected; not test-evaluated.** Family: partially fine-tuned pretrained CNN.

**How it works.** It starts from the pretrained MobileNetV3 and updates the weights of its last two convolutional blocks plus a new classification head, using our images, so the representation can adapt slightly to conjunctiva colour.

**Implementation** (`FT_CONFIG`, [`cnn_select.py:48`](../scripts/cnn_select.py#L48); `FineTuneNet`, [`cnn_common.py:122`](../scripts/cnn_common.py#L122))

| Item | Value |
|---|---|
| Weights | ImageNet `IMAGENET1K_V2`. **Last 2 feature blocks + new head trainable; rest frozen** |
| Head | Dropout(0.3) → Linear(960 → 1) |
| BatchNorm | Frozen (eval mode), because batches of 16 are too small to re-estimate its statistics |
| Optimiser | AdamW; learning rate 1e-4 (backbone) and 1e-3 (head); weight decay 0.05; batch size 16 |
| Loss | `BCEWithLogitsLoss`, pos_weight = n_neg / n_pos of the training fold |
| Augmentation | As in [Section 2.2](#22-image-preprocessing-cnn-models) (geometric only) |
| Protocol | 5-fold CV (repeat-1 folds), 30 epochs per fold; epochs chosen from {10, 20, 30} by mean validation AUC |
| Device | Apple MPS; **not bit-for-bit reproducible** |
| Trainable parameters | About 0.95M of 2.97M (backbone without the original classifier). Measured in a development check; **not stored in the result files** |
| Saved model | **None.** It was not the selected candidate, so no final fine-tuned model was trained |

**Results.** [CV, other protocol] 5-fold: **0.639 ± 0.091** at 30 epochs (per fold 0.616, 0.624, 0.735, 0.489, 0.734). Mean validation AUC at 10/20/30 epochs: 0.639 / 0.634 / 0.639. **[Test]** Not evaluated.

**Interpretation.** It performed worse than frozen features and varied widely between folds (one fold near chance). We interpret this as **consistent with overfitting** at about 160 training images per fold. Training-set loss and AUC were not recorded, so overfitting was not measured directly.

### 3.11 Ridge Regression for hemoglobin

**Status: did not establish useful Hb estimation.** Family: linear regression with L2 penalty.

**How it works.** It fits a linear function of the standardised features to predict Hb in g/dL, shrinking coefficients toward zero by an amount set by α.

**Why we used it.** It is the simplest test of whether colour features carry enough information to estimate Hb continuously.

**Implementation** ([`train_baseline.py:136-137`](../scripts/train_baseline.py#L136-L137)): `StandardScaler` → `Ridge`. α grid `np.logspace(-2, 4, 13)`, chosen by CV MAE on 25 folds stratified on Hb bins; selected α = **1**. Comparator: `DummyRegressor` predicting the training-set mean (10.998 g/dL). Saved model: `models/regressor.joblib`.

**Results**

| Evaluation | Ridge | Mean predictor |
|---|---|---|
| [CV, other protocol] MAE (25 folds) | 1.592 ± 0.115 g/dL | 1.717 ± 0.042 g/dL |
| **[Test]** MAE | **1.639** (1.290–2.017) | 1.693 |
| **[Test]** RMSE | 2.101 (1.658–2.520) | — |
| **[Test]** R² | **−0.036** (−0.386 to 0.183) | — |
| **[Test]** Pearson r | 0.175 (−0.139 to 0.467) | — |
| **[Test]** MAE improvement | +0.054 g/dL (−0.153 to 0.259) | — |

Test predictions were compressed to 8.50–13.23 g/dL, against true values of 5.9–14.9, so Hb was systematically over-estimated for the most anemic images (that is, their severity was under-estimated). Secondary result, for context only: thresholding the Ridge predictions at 11 g/dL gave test AUC 0.655, sensitivity 0.591 and specificity 0.655, worse than the direct classifier.

**Interpretation.** It is no meaningful improvement over predicting the mean. **Per-image Hb estimation is not supported** by these features and data.

---

## 4. Comparison tables

**Do not compare columns from different evaluation types as though they were the same procedure.** CV AUC and test AUC are different measurements.

### A. Classical models: comparable nested CV

[Nested CV]: same 25 outer folds, same 35 features, training split only (n = 200). Source: `outputs/classical_cv_results.json`, `outputs/boosting_cv_results.json`.

| Rank | Model | Family | Input | Nested-CV AUC | SD | Without size feature | Folds better than LR | Main limitation |
|---|---|---|---|---|---|---|---|---|
| 1 | **Extra Trees** | Randomised tree ensemble | 35 colour features | **0.789** | 0.061 | 0.787 | 23 / 25 | Test gain over LR not established |
| 2 | RBF-SVM | Kernel method | 35 colour features (scaled) | 0.781 | 0.077 | 0.763 | 21 / 25 | Not test-evaluated; no probabilities |
| 3 | CatBoost | Gradient boosting | 35 colour features | 0.762 | 0.071 | 0.749 | 20 / 25 | Not test-evaluated |
| 4 | Random Forest | Bagged trees | 35 colour features | 0.743 | 0.063 | 0.749 | 17 / 25 | Not test-evaluated |
| 5 | XGBoost | Gradient boosting | 35 colour features | 0.724 | 0.073 | 0.707 | 14 / 25 | Not test-evaluated |
| 6 | HistGradientBoosting | Gradient boosting | 35 colour features | 0.720 | 0.070 | 0.715 | 14 / 25 | Not test-evaluated |
| 7 | **Logistic Regression** | Linear | 35 colour features (scaled) | 0.718 | 0.077 | 0.708 | — | Lowest nested-CV AUC |

Ranking seven models on the same folds has a winner's-curse effect: the top model's estimate is likely somewhat optimistic.

### B. Models evaluated on the held-out test set

[Test]: n = 51 (22 anemic, 29 non-anemic); one scoring each; thresholds from training data. 95% CIs in parentheses.

| Model | Input | Test AUC | Sensitivity | Specificity | Precision | F1 | Confusion (TP/FN/FP/TN) | Paired ΔAUC vs LR |
|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | Colour features | 0.724 (0.567–0.865) | **0.727** | 0.690 | 0.640 | **0.681** | 16 / 6 / 9 / 20 | — |
| Extra Trees | Colour features | **0.800** (0.668–0.914) | 0.682 | **0.759** | **0.682** | **0.682** | 15 / 7 / 7 / 22 | +0.076 (−0.063 to 0.219) |
| Frozen MobileNetV3 + LR | 960-d ImageNet embedding | 0.660 (0.508–0.803) | 0.364 | **0.759** | 0.533 | 0.432 | 8 / 14 / 7 / 22 | −0.064 (−0.257 to 0.129) |

All other models: **Not evaluated** on the test set.

### C. CNN experiments (their actual protocols)

| Model | Pretrained | Backbone | CV protocol | CV AUC | SD | Test AUC | Limitation |
|---|---|---|---|---|---|---|---|
| Frozen MobileNetV3-Large + LR | ImageNet V2 | Frozen | Non-nested, 25 folds | 0.716 | 0.087 | 0.660 | Below colour LR on the same protocol (0.741); low sensitivity |
| Frozen EfficientNet-B0 + LR | ImageNet V1 | Frozen | Non-nested, 25 folds | 0.649 | 0.067 | Not evaluated | Weakest frozen backbone |
| Fine-tuned MobileNetV3-Large | ImageNet V2 | Last 2 blocks + head trained | 5-fold (repeat 1) | 0.639 | 0.091 | Not evaluated | Unstable across folds (0.49–0.73); MPS non-deterministic |
| *Reference: colour-feature LR* | — | — | Non-nested, 25 folds | 0.741 | 0.079 | 0.724 | — |

### D. Hemoglobin regression

| Model | Input | CV MAE (g/dL) | Test MAE | Test R² | Mean-predictor test MAE | Limitation |
|---|---|---|---|---|---|---|
| Ridge | 35 colour features (scaled) | 1.592 ± 0.115 | 1.639 | −0.036 | 1.693 | Improvement CI includes 0; predictions compressed toward the mean |

### E. Final recommendation

| Role | Model | Basis |
|---|---|---|
| **Primary** | Logistic Regression | Pre-declared rule; no challenger met the adoption criterion |
| Strongest experimental candidate | Extra Trees | Best nested CV (0.789; 23/25 folds over LR); test AUC 0.800, but the paired CI includes 0 |
| Secondary CV candidate | RBF-SVM | Nested CV 0.781; CV evidence only |
| Not recommended | CatBoost, XGBoost, Random Forest, HistGradientBoosting | Below Extra Trees in nested CV |
| Not recommended | All CNN variants | Did not exceed the colour baseline |
| Not supported | Ridge Hb regression | No meaningful gain over the mean |

---

## 5. Feature importance and dataset bias

### Feature importance

**No feature-importance analysis was performed.** We did not compute LR coefficient rankings, tree impurity importances, permutation importance or SHAP values. The only feature-level evidence is the size-feature ablation below. Claims about *which* colour features drive the predictions are therefore **unrecorded**.

### The crop-size shortcut investigation

All of this analysis used **training data only**. Sources: `outputs/cnn_cv_results.json` (controls), `outputs/shortcut_diagnostic.json`, and the nested results.

- **Crop size differs between classes.** Median tissue pixel count is 3,745 for anemic and 4,430 for non-anemic training images.
- **Size and shape alone carry signal:**

| Input (training CV, 25 folds) | AUC |
|---|---|
| `n_tissue_px` only (LR, non-nested) | 0.648 ± 0.093 |
| Crop geometry: tissue pixels, width, height, aspect (LR, non-nested) | 0.635 ± 0.094 |
| Silhouette only → frozen MobileNetV3 / EfficientNet-B0 (non-nested) | 0.571 / 0.589 |

- **Removing `n_tissue_px`** has little effect on the colour models:

| Model | With | Without | Protocol |
|---|---|---|---|
| Logistic Regression | 0.741 | 0.740 | Non-nested ([`diagnose_shortcut.py`](../scripts/diagnose_shortcut.py)) |
| Logistic Regression | 0.718 | 0.708 | Nested |
| Extra Trees | 0.789 | 0.787 | Nested |
| RBF-SVM | 0.781 | 0.763 | Nested |

**Interpretation.** The colour models do not depend on the *explicit* size feature. This is evidence about **one** possible shortcut; it does **not** show that the models are free of bias. Colour statistics could still correlate with crop size, camera, lighting, session or person. The cause of the size difference (acquisition procedure, cropping, or something physiological) is **unknown**.

### Other sources of bias and dependence

- **Small sample:** 200 training and 51 test images, so CIs are wide (for example, LR test AUC 0.567–0.865).
- **Unknown patient independence:** there are no patient IDs, and image numbers run from 4 to 710 with gaps, which suggests a subset of a larger collection. If one person contributed several images (for example, both eyes or repeat visits), they could appear in both train and test. That would inflate test metrics, probably more for higher-capacity models, and would make CV folds non-independent.
- **Acquisition and lighting:** white balance and brightness vary strongly between images, and there is no colour reference in frame. A model could learn device or lighting differences that happen to correlate with the label.
- **Provenance:** licence, collection site, population, devices and Hb method are **unverified** ([`docs/DATASET.md`](../docs/DATASET.md)).

**Predicting this dataset's labels does not show that a model has learned a medically meaningful pallor signal.**

---

## 6. Final model decision

Three different questions are involved:

| Question | Answer |
|---|---|
| Which model has the **highest observed score**? | Extra Trees (nested CV 0.789; test AUC 0.800) |
| Is any improvement over LR **statistically established**? | **No.** Extra Trees' paired test ΔAUC is +0.076 with 95% CI −0.063 to 0.219, which includes zero |
| Which model is **primary under our pre-declared rule**? | **Logistic Regression.** The rule required a challenger to beat LR in CV *and* to have a paired test ΔAUC CI entirely above 0 |

**Why we do not simply choose the highest test score.**
1. With 51 test images, an AUC difference of this size cannot be distinguished from chance; the CI spans about ±0.14.
2. The test set has been used for three evaluations. Choosing the best of them would make the test set part of model selection, and the selected score would then overstate future performance.
3. The rule was fixed **before** each test run, specifically to stop results from changing the decision criteria after the fact.
4. "Not demonstrated" is not "no difference". Extra Trees' consistent CV advantage (23 of 25 folds) makes it the leading candidate. Confirming it needs new, independent data.

**Current decision**
- **Logistic Regression** remains the primary model.
- **Extra Trees** is the strongest experimental candidate. If shown in a demo, it must be labelled *experimental*.
- **RBF-SVM** is another promising candidate, based on CV evidence only.
- **CatBoost** and **XGBoost** did not beat Extra Trees.
- **CNNs** did not establish an improvement over the colour baseline.
- **Ridge regression** did not establish useful Hb estimation.
- The model search is **closed**. Further progress depends on data: colour calibration, patient-level splits and external validation.

---

## 7. Reproducibility and project files

### Scripts (all present in the repository)

| File | Purpose | Reads the test set? |
|---|---|---|
| [`scripts/inspect_dataset.py`](../scripts/inspect_dataset.py) | Validates every image-to-label mapping; writes `derived/manifest.csv`; exits non-zero (and writes no manifest) on any failure | N/A (no model) |
| [`scripts/features.py`](../scripts/features.py) | Extracts the 35 colour features from opaque tissue pixels | — |
| [`scripts/train_baseline.py`](../scripts/train_baseline.py) | Creates or verifies the locked split; tunes LR and Ridge with CV; fits final baseline models | No |
| [`scripts/evaluate.py`](../scripts/evaluate.py) | Baseline test evaluation with bootstrap CIs; shared metric and bootstrap helpers | Yes (baseline) |
| [`scripts/cnn_common.py`](../scripts/cnn_common.py) | Image preprocessing, augmentation, backbones, `FineTuneNet`, split-lock check | — |
| [`scripts/cnn_select.py`](../scripts/cnn_select.py) | CNN candidates, controls, and selection on training data | No |
| [`scripts/cnn_evaluate.py`](../scripts/cnn_evaluate.py) | One-shot CNN test evaluation with paired comparison; refuses to re-run | Yes (once) |
| [`scripts/diagnose_shortcut.py`](../scripts/diagnose_shortcut.py) | LR with and without `n_tissue_px` (training CV) | No |
| [`scripts/classical_select.py`](../scripts/classical_select.py) | Nested CV for LR, SVM, RF, ET and HGB; selects and fits Extra Trees; refuses to overwrite | No |
| [`scripts/classical_evaluate.py`](../scripts/classical_evaluate.py) | One-shot Extra Trees test evaluation with paired comparison; refuses to re-run | Yes (once) |
| [`scripts/boosting_select.py`](../scripts/boosting_select.py) | Nested CV for XGBoost and CatBoost, with fold-identity check; refuses to overwrite | No |

### Aggregate results (tracked in Git; no per-image data)

| File | Contents |
|---|---|
| [`outputs/cv_results.json`](../outputs/cv_results.json) | Baseline LR and Ridge CV, grids, threshold, feature list |
| [`outputs/metrics.json`](../outputs/metrics.json) | Baseline test metrics (classification and regression) with CIs |
| [`outputs/cnn_cv_results.json`](../outputs/cnn_cv_results.json) | CNN candidates, controls, fine-tuning curves, selection |
| [`outputs/cnn_metrics.json`](../outputs/cnn_metrics.json) | Frozen MobileNetV3 test metrics and paired comparison |
| [`outputs/shortcut_diagnostic.json`](../outputs/shortcut_diagnostic.json) | Size-feature ablation |
| [`outputs/classical_cv_results.json`](../outputs/classical_cv_results.json) | Nested CV for 5 classical models, inner choices, selection |
| [`outputs/classical_metrics.json`](../outputs/classical_metrics.json) | Extra Trees test metrics and paired comparison |
| [`outputs/boosting_cv_results.json`](../outputs/boosting_cv_results.json) | Nested CV for XGBoost and CatBoost |

### Local-only artifacts (gitignored)

- **Data:** `data/` (the raw dataset, never modified).
- **Derived:** `derived/` (manifest, features, split and split lock).
- **Saved models:** everything in `models/` except this README: `classifier.joblib`, `regressor.joblib`, `cnn_final.joblib`, `cnn_final_meta.json`, `classical_final.joblib`.
- **Per-image prediction CSVs:** in `outputs/`.
- **Baseline snapshot:** `outputs/baseline_colour/` (read-only, with `SHA256SUMS`).

**The raw dataset, per-image records and per-image predictions must stay out of Git** unless the licence, permission and privacy questions have been resolved.

### Environment

- **Python:** 3.13 virtual environment (`.venv/`).
- **Packages:** pinned in [`requirements.txt`](../requirements.txt): scikit-learn 1.9.1, torch 2.14.1, torchvision 0.29.1, xgboost 3.4.1, catboost 1.2.10, and others.
- **XGBoost on macOS:** needs `brew install libomp`.
- **ImageNet weights:** downloaded by torchvision to `~/.cache/torch`.

### Run order

```bash
.venv/bin/python scripts/inspect_dataset.py
.venv/bin/python scripts/train_baseline.py      # verifies split.lock
.venv/bin/python scripts/evaluate.py
.venv/bin/python scripts/cnn_select.py
.venv/bin/python scripts/cnn_evaluate.py        # one-shot; refuses if results exist
.venv/bin/python scripts/diagnose_shortcut.py
.venv/bin/python scripts/classical_select.py    # refuses if results exist
.venv/bin/python scripts/classical_evaluate.py  # one-shot; refuses if results exist
.venv/bin/python scripts/boosting_select.py     # refuses if results exist
```

**Reproducibility notes**
- A re-run of the baseline was verified to reproduce `outputs/metrics.json` byte for byte.
- All classical models use fixed seeds.
- Frozen CNN embeddings on MPS were checked to be identical across two runs.
- Fine-tuning on MPS is not bit-for-bit reproducible.

---

## 8. Limitations and ethical interpretation

- **Pallor is an experimental research prototype.** It is **not** a validated diagnostic or screening device.
- **It must not be used to diagnose anemia or rule it out.** A non-anemic prediction is not reassurance; a blood test is the reference standard.
- **No clinical validation** of any kind has been performed: no prospective study, no external dataset, no comparison with clinicians.
- **The dataset is small** (251 images), and **patient independence is unknown**, so reported metrics may be optimistic.
- **The Hb < 11 g/dL threshold is this dataset's labelling convention**, not a universal medical rule.
- **Performance here does not generalise** to other people, cameras, lighting conditions, skin tones, ages or populations. None of these were tested.
- The best test AUC observed (0.800, Extra Trees) still means many misclassifications: 7 of 22 anemic and 7 of 29 non-anemic test images. That is not adequate for clinical use.
- **Stronger claims would need** independent, properly labelled data with patient identifiers, documented Hb measurement, controlled or calibrated imaging, and appropriate ethical approval.
- Dataset licence and provenance must be verified before any redistribution or publication.
