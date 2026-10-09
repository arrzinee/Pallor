# Pallor — Experiment Report

> **Research prototype. Not clinically validated. Not for diagnosis or treatment decisions.**
> Dataset licence and provenance are **pending verification** (see [DATASET.md](DATASET.md)).

**Current status.**
- Primary model: logistic regression on 35 colour features (test AUC 0.724).
- Frozen and fine-tuned CNNs underperformed it (Section 5).
- Extra Trees led in nested CV (0.789 vs 0.718) and scored a higher test AUC (0.800), but the paired test ΔAUC CI includes 0, so it was not adopted under the pre-declared rule (Section 6).
- XGBoost (nested-CV AUC 0.724) and CatBoost (0.762) did not beat Extra Trees and were not test-evaluated (Section 7). Model search is closed; see the summary of all models in Section 8.
- Hb regression did not beat a mean predictor (Section 3).

## 1. Dataset

| | |
|---|---|
| Images used | 251 PNG conjunctiva crops (`data/Anaemic` 110, `data/Non-Anaemic` 141) |
| Excluded | `data/All Categories` (all 251 files are byte-identical copies) |
| Labels | Per-image Hb (g/dL) from the workbook. Anemic := Hb < 11 g/dL, which matches the folder label for 251/251 images |
| Hb range | 4.2–15.0 g/dL, mean 11.01 |
| Validation | Every file maps to exactly one workbook row; all 3 sheets agree; 0 unreadable images; 0 exact duplicates |
| Patient IDs | **None available** |

The manifest is built reproducibly by `scripts/inspect_dataset.py`, which writes `derived/manifest.csv` (gitignored). The script exits non-zero and writes no manifest if any check fails. This was tested by injecting a wrong threshold: it flagged 35 inconsistent rows and wrote nothing.

## 2. Method

**Features (35).** Computed on fully opaque pixels only (alpha = 255). Transparent background pixels are stored as RGB (0,0,0) and are excluded. The soft mask boundary (0 < alpha < 255, about 5% of non-transparent pixels) is also excluded, as a conservative choice. *(Correction: an earlier version said these edge pixels were stored darkened. Their RGB is actually stored un-premultiplied, i.e. roughly true colour. Excluding them is unchanged, so the baseline results are unaffected.)* Features:
- RGB mean, std, and 10/50/90th percentiles
- rg chromaticity
- HSV, with hue averaged circularly (sin/cos)
- CIELAB mean and std, and a\*/L\*
- Redness indices: erythema index 100·log10(R/G), (R−G)/(R+G), R/G, and the fraction of red-dominant pixels
- Tissue pixel count

The extractor is unit-tested on a synthetic image and against reference HSV and Lab values.

**Split.** One fixed 80/20 split, seed 42, stratified on Hb bins (edges 8, 9.5, 11, 12, 13 g/dL). Because 11 is an edge, the class is stratified as well.

| Split | n | Anemic | Non-anemic | % anemic | Hb mean | Hb range |
|---|---|---|---|---|---|---|
| Train | 200 | 88 | 112 | 44.0% | 11.00 | 4.2–15.0 |
| Test | 51 | 22 | 29 | 43.1% | 11.05 | 5.9–14.9 |

**Model selection (training split only).**
- CV scheme: 5-fold × 5-repeat stratified cross-validation.
- Scaling is fitted inside each fold (sklearn `Pipeline`).
- Classification: `StandardScaler + LogisticRegression(class_weight="balanced")`. C was chosen by CV ROC-AUC from 10⁻³ to 10⁵; selected C = 100.
- Decision threshold: the value that maximises Youden's J on 5-fold out-of-fold training probabilities, which gave **0.502**.
- Regression: `StandardScaler + Ridge`, with α chosen by CV MAE; selected α = 1.
- Baseline for regression: predict the training-set mean Hb (11.00 g/dL).

**Test set.** Test labels are read only by `scripts/evaluate.py`, after every choice above was frozen. The test set was evaluated once with the final configuration.

## 3. Results

### Cross-validation (training split, n = 200)

| Task | Model | CV result (mean ± SD over 25 folds) |
|---|---|---|
| Classification | Logistic regression | ROC-AUC 0.741 ± 0.079 |
| Regression | Ridge | MAE 1.592 ± 0.115 g/dL |
| Regression | Train-mean baseline | MAE 1.717 ± 0.042 g/dL |

### Held-out test (n = 51; 95% bootstrap CIs, 2000 resamples)

**A. Classification (positive = anemic)**

| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | **0.724** | 0.567 – 0.865 |
| Sensitivity | 0.727 (16/22) | 0.545 – 0.909 |
| Specificity | 0.690 (20/29) | 0.517 – 0.862 |
| Precision | 0.640 (16/25) | 0.500 – 0.792 |
| F1 | 0.681 | 0.533 – 0.809 |
| Accuracy | 0.706 | 0.569 – 0.824 |

Confusion matrix: TP 16, FN 6, FP 9, TN 20.

**B. Hb regression (g/dL)**

| Metric | Ridge | Train-mean baseline |
|---|---|---|
| MAE | **1.639** (CI 1.290 – 2.017) | 1.693 |
| RMSE | 2.101 | — |
| R² | −0.036 (CI −0.386 – 0.183) | 0 by construction on train |
| Pearson r | 0.175 (CI −0.139 – 0.467) | — |
| MAE improvement vs baseline | 0.054 g/dL (CI −0.153 – 0.259) | — |

Secondary result, for context only: thresholding the ridge Hb prediction at 11 g/dL gives test AUC 0.655, sensitivity 0.591, and specificity 0.655. That is worse than the direct classifier.

## 4. Interpretation

- **Classification shows a modest signal.** Test AUC (0.72) is consistent with CV AUC (0.74 ± 0.08). The 95% CI lower bound is 0.57, so the result is above chance but imprecise. Roughly 3 in 10 anemic and 3 in 10 non-anemic test images are misclassified. Performance at this level is not suitable for screening use.
- **Hb regression does not meaningfully beat the trivial baseline on the test set.** MAE improves by only 0.05 g/dL, the CI includes zero, and R² is slightly negative. In CV the gain was small (about 0.13 g/dL). Ridge predictions are compressed toward the mean (test predictions range 8.5–13.2 g/dL against a true range of 5.9–14.9), so severe anemia is systematically underestimated. **With these colour features, per-image Hb estimation is not supported.**
- Likely limiting factors:
  - Uncontrolled lighting and white balance, with no colour reference in frame
  - The small sample
  - Summary colour statistics discard texture and vascular pattern

## 5. CNN transfer-learning experiment

**Outcome: no CNN variant beat the colour baseline, so the colour-feature logistic regression stays the primary model.**

### Setup
- **Backbones:** ImageNet-pretrained MobileNetV3-Large and EfficientNet-B0 (torchvision). Both run comfortably on an Apple M4 via MPS: frozen feature extraction takes seconds, and 5-fold partial fine-tuning took about 3.5 minutes.
- **RGBA handling:** each crop is alpha-composited over the ImageNet mean colour. RGB is stored un-premultiplied, so this is standard compositing. After normalisation the background is exactly 0. Crops are letterbox-padded (aspect ratio kept) and resized to 224×224. ImageNet mean/std are fixed constants, so no preprocessing is fitted to our data.
- **Augmentation (training folds only):** horizontal flip; rotation ±10°; shift ±5%; scale 0.9–1.1. There are **no colour, brightness, hue or saturation transforms**, because tissue colour is the diagnostic signal and altering it would change the image's medical meaning. There are no vertical flips.
- **Candidates, declared before any results:**
  - F: frozen backbone features + logistic regression, with C tuned by CV.
  - FT: fine-tuning of the last 2 blocks + a dropout(0.3) head on the better frozen backbone. AdamW (lr 1e-4 backbone / 1e-3 head, weight decay 0.05), BatchNorm frozen, class-weighted loss, epochs chosen from {10, 20, 30}.
- **Selection:** on the training split only, using **the same CV folds as the colour baseline**, by a pre-declared rule (best repeat-1 mean AUC; ties go to the simpler model). The decision threshold uses Youden's J on out-of-fold training predictions, the same rule as the baseline.
- **Test:** scored **once**. `cnn_evaluate.py` refuses to run a second time. An adoption rule was fixed before scoring: the CNN replaces the baseline only if it beat the baseline in CV **and** the paired test ΔAUC 95% CI lies above 0.

### Training-split cross-validation (n = 200, identical folds)

| Model | 25-fold AUC (mean ± SD) | Repeat-1 5-fold AUC |
|---|---|---|
| **Colour baseline** | **0.741 ± 0.079** | **0.705** |
| Frozen MobileNetV3-Large + LR (C = 0.001) | 0.716 ± 0.087 | 0.681 |
| Frozen EfficientNet-B0 + LR (C = 3.16) | 0.649 ± 0.067 | 0.608 |
| Fine-tuned MobileNetV3-Large (30 epochs) | — | 0.639 ± 0.091 (per fold 0.49–0.73) |

Fine-tuning was worse than frozen features and unstable across folds, consistent with overfitting at about 160 training images per fold. Mean CV AUC was flat across 10/20/30 epochs (0.639 / 0.634 / 0.639). The chosen CNN was the frozen MobileNetV3, with threshold 0.533.

### Held-out test (n = 51: 22 anemic, 29 non-anemic; same images as the baseline)

| Metric | Frozen MobileNetV3 + LR (95% CI) | Colour baseline | Paired Δ (CNN − baseline), 95% CI |
|---|---|---|---|
| ROC-AUC | 0.660 (0.508 – 0.803) | **0.724** | −0.064 (−0.257 – 0.129) |
| Sensitivity | 0.364 (0.182 – 0.591) | **0.727** | −0.364 (−0.591 – −0.136) |
| Specificity | **0.759** (0.586 – 0.897) | 0.690 | +0.069 (−0.069 – 0.241) |
| Precision | 0.533 (0.308 – 0.750) | **0.640** | −0.107 (−0.316 – 0.094) |
| F1 | 0.432 (0.229 – 0.619) | **0.681** | −0.248 (−0.453 – −0.051) |
| Accuracy | 0.588 (0.451 – 0.706) | **0.706** | −0.118 (−0.255 – 0.020) |

Confusion matrices:
- CNN: TP 8, FN 14, FP 7, TN 22.
- Baseline: TP 16, FN 6, FP 9, TN 20.

CIs come from 2000 class-stratified bootstrap resamples; the Δ column is a paired bootstrap on identical resamples.

**Reading the results.** The AUC difference is inconclusive (its CI spans 0), but the CNN is clearly worse at detecting anemic cases. It missed 14 of 22, and its sensitivity CI lies entirely below the baseline's. The CNN failed both adoption conditions, so **the baseline is retained.**

Likely reasons:
1. With only 200 training images, a 960-dimensional generic embedding gives the classifier much more room to overfit than 35 targeted colour features.
2. ImageNet features are built to be fairly insensitive to exact colour and lighting, and colour is what carries the pallor signal here.
3. Padding thin, elongated crops to a square leaves most of the input as background.

### Shortcut check: crop geometry (training CV only)

| Input | 25-fold CV AUC |
|---|---|
| Crop geometry only (tissue pixels, width, height, aspect) | 0.635 ± 0.094 |
| Tissue pixel count only | 0.648 ± 0.093 |
| Silhouette only → frozen MobileNetV3 / EfficientNet-B0 | 0.571 / 0.589 |
| Colour baseline, all 35 features | 0.741 ± 0.079 |
| Colour baseline without `n_tissue_px` | 0.740 ± 0.068 |

Crop **size and shape alone predict the label better than chance**: anemic crops are smaller (median 3,745 vs 4,430 tissue pixels). The cause is unknown. It could be an acquisition or cropping artefact, or something physiological. Removing the size feature does not change the baseline's CV AUC, so the baseline does not depend on the explicit size feature. Colour features could still correlate with size indirectly, and any future model should be checked the same way.

## 6. Classical ML model comparison

**Outcome: Extra Trees was the strongest model in nested cross-validation and scored higher on the test set, but its test advantage is not statistically conclusive. Under the pre-declared rule, logistic regression remains the primary model.**

### Setup
- **Features:** the same 35 colour features as the baseline.
- **Nested cross-validation, training split only (n = 200):**
  - Outer loop: the same 25 folds (5×5 repeated stratified) used for the baseline and CNN comparisons. It measures performance.
  - Inner loop: a 5-fold stratified grid search inside every outer training fold. It tunes hyperparameters.
  - Scaling is inside the pipeline for LR and SVM.
  - Nested CV was used because plain CV, which tunes and scores on the same folds, favours models with larger grids.
- **Models and grids:**
  - LR: C from 10⁻³ to 10⁵.
  - RBF-SVM: C ∈ {0.1, 1, 10, 100}, γ ∈ {scale, 0.001, 0.01, 0.1}.
  - Random Forest and Extra Trees: 300 trees, max_depth ∈ {None, 4}, min_samples_leaf ∈ {1, 3, 10}, max_features ∈ {√, 0.5}.
  - HistGradientBoosting: learning rate ∈ {0.03, 0.1}, depth ∈ {2, 3}, iterations ∈ {100, 300}, min_samples_leaf ∈ {10, 20}, L2 = 1.
  - All use balanced class weights.
- **XGBoost: not evaluated in this comparison.** At the time, its macOS wheel could not load without the system OpenMP library (`libomp`). XGBoost was evaluated later, after `libomp` was installed with approval (Section 7). HistGradientBoosting covers the same model family here.
- **Selection rule (declared before results):** the single non-LR model with the highest nested-CV mean AUC goes to one test evaluation. It replaces LR only if it beat LR in nested CV **and** the paired test ΔAUC 95% CI lies above 0. The decision threshold uses Youden's J on 5-fold OOF training probabilities.

### Nested cross-validation (training split only)

| Model | Nested-CV AUC (mean ± SD, 25 outer folds) | Without `n_tissue_px` | Paired ΔAUC vs LR (mean ± SD) | Outer folds better than LR |
|---|---|---|---|---|
| Logistic regression (reference) | 0.718 ± 0.077 | 0.708 | — | — |
| SVM (RBF) | 0.781 ± 0.077 | 0.763 | +0.063 ± 0.078 | 21 / 25 |
| Random Forest | 0.743 ± 0.063 | 0.749 | +0.025 ± 0.079 | 17 / 25 |
| **Extra Trees** | **0.789 ± 0.061** | **0.787** | **+0.071 ± 0.069** | **23 / 25** |
| HistGradientBoosting | 0.720 ± 0.070 | 0.715 | +0.002 ± 0.066 | 14 / 25 |

- LR's nested-CV AUC (0.718) is lower than its earlier non-nested CV AUC (0.741). That gap is the optimism of tuning and scoring on the same folds.
- LR's tuned C was unstable across outer folds, ranging from 0.003 to 10⁵.
- Removing the crop-size feature barely changes Extra Trees (0.789 → 0.787), so its advantage does not come from the size shortcut identified in Section 5.
- Outer folds overlap, so the fold-level SDs and "folds better" counts are descriptive, not independent tests.

**Candidate: Extra Trees.** Inner-CV-selected settings on the full training split: max_depth = None, max_features = √, min_samples_leaf = 1. Threshold 0.44 (training OOF sensitivity 0.773, specificity 0.768).

### Held-out test, one evaluation (n = 51: 22 anemic, 29 non-anemic; same images as the baseline)

| Metric | Extra Trees (95% CI) | LR baseline | Paired Δ (ET − LR), 95% CI |
|---|---|---|---|
| ROC-AUC | **0.800** (0.668 – 0.914) | 0.724 | +0.076 (−0.063 – 0.219) |
| Sensitivity | 0.682 (0.500 – 0.864) | **0.727** | −0.045 (−0.227 – 0.136) |
| Specificity | **0.759** (0.586 – 0.897) | 0.690 | +0.069 (−0.103 – 0.241) |
| Precision | **0.682** (0.536 – 0.850) | 0.640 | +0.042 (−0.112 – 0.193) |
| F1 | 0.682 (0.522 – 0.818) | 0.681 | +0.001 (−0.149 – 0.142) |
| Accuracy | **0.725** (0.608 – 0.843) | 0.706 | +0.020 (−0.098 – 0.137) |

Confusion matrices:
- Extra Trees: TP 15, FN 7, FP 7, TN 22.
- LR: TP 16, FN 6, FP 9, TN 20.

The two models differ by one more missed anemic case and two fewer false alarms for Extra Trees. CIs come from 2000 class-stratified bootstrap resamples; Δ is a paired bootstrap on identical resamples.

**Reading the results.** Extra Trees' test AUC point estimate (0.80) is higher and agrees with its nested-CV estimate (0.79). The paired ΔAUC CI still includes 0, so the adoption rule is **not met**, and LR stays primary. With only 51 test images, a true AUC gain of about 0.07 is unlikely to reach significance. So the outcome means "not demonstrated", not "no difference". The strongest evidence for Extra Trees is its consistent nested-CV advantage (23 of 25 outer folds). At the chosen thresholds, sensitivity and F1 are essentially the same as LR.

## 7. XGBoost and CatBoost (final model-search experiment)

**Outcome: neither model beat Extra Trees in nested CV. Per the plan, this closes the model search.**

### Setup
- **Same protocol as Section 6:** the 35 colour features, training split only, the same 25 outer folds, and a 5-fold inner grid search.
- **Fold identity check:** before running, the LR reference was re-run and reproduced all 25 of its saved per-fold AUCs exactly, so the paired comparisons below use identical folds.
- **Limited grids (8 combinations each):**
  - XGBoost 3.4.1: n_estimators ∈ {100, 300}, max_depth ∈ {2, 3}, learning_rate ∈ {0.03, 0.1}; subsample and colsample 0.8, λ = 1.
  - CatBoost 1.2.10: iterations ∈ {200, 500}, depth ∈ {3, 5}, learning_rate ∈ {0.03, 0.1}; l2_leaf_reg = 3.
  - Both seeded, single-threaded, and without class weighting (AUC is threshold-free).
- **Held-out test set NOT used.** It has already served three one-shot evaluations and no new test data exist, so these models have CV evidence only, and no final model was fitted.
- **Environment:** XGBoost on macOS needs the OpenMP runtime, which was installed system-wide with `brew install libomp` after the project owner approved it. CatBoost needs no system dependency.

### Nested cross-validation (training split only, n = 200)

| Model | Nested-CV AUC (mean ± SD) | Without `n_tissue_px` | Paired Δ vs LR (folds better) | Paired Δ vs Extra Trees (folds better) |
|---|---|---|---|---|
| XGBoost | 0.724 ± 0.073 | 0.707 | +0.006 (14 / 25) | −0.065 (2 / 25) |
| CatBoost | 0.762 ± 0.071 | 0.749 | +0.044 (20 / 25) | −0.026 (6 / 25) |

## 8. Summary of all models

### Comparable cross-validation: nested CV, same 25 outer folds, same 35 colour features, training split only

| Rank | Model | Nested-CV AUC (mean ± SD) | Without size feature | Folds better than LR | Held-out test AUC |
|---|---|---|---|---|---|
| 1 | **Extra Trees** | **0.789 ± 0.061** | 0.787 | 23 / 25 | 0.800 (one-shot) |
| 2 | SVM (RBF) | 0.781 ± 0.077 | 0.763 | 21 / 25 | not tested |
| 3 | CatBoost | 0.762 ± 0.071 | 0.749 | 20 / 25 | not tested |
| 4 | Random Forest | 0.743 ± 0.063 | 0.749 | 17 / 25 | not tested |
| 5 | XGBoost | 0.724 ± 0.073 | 0.707 | 14 / 25 | not tested |
| 6 | HistGradientBoosting | 0.720 ± 0.070 | 0.715 | 14 / 25 | not tested |
| 7 | **Logistic regression (primary)** | 0.718 ± 0.077 | 0.708 | — | **0.724** (one-shot) |

### Not directly comparable: non-nested or a different protocol

| Model | CV result | Held-out test AUC | Note |
|---|---|---|---|
| LR, non-nested (original baseline) | 0.741 ± 0.079 (25 folds) | 0.724 | Hyperparameter tuned and scored on the same folds, so optimistic |
| Frozen MobileNetV3 + LR | 0.716 ± 0.087 (25 folds, non-nested) | 0.660 (one-shot) | Compare with LR non-nested 0.741 |
| Frozen EfficientNet-B0 + LR | 0.649 ± 0.067 (25 folds, non-nested) | not tested | |
| Fine-tuned MobileNetV3 | 0.639 ± 0.091 (5 folds) | not tested | Epochs chosen on the same folds |
| Ridge Hb regression | CV MAE 1.59 vs 1.72 (mean predictor) | test MAE 1.64 vs 1.69 | Did not meaningfully beat the mean |

### Recommended model

- **Primary model: logistic regression on colour features.** Under the selection and adoption rules declared before each test, no challenger met the bar of a test ΔAUC CI lying above 0. LR is also the simplest and most interpretable model, and it was the first one to be tested.
- **Strongest candidate: Extra Trees.** It ranks first in nested CV, beats LR in 23 of 25 outer folds, does not rely on the crop-size feature, and its one-shot test AUC (0.800) agrees with its CV estimate. Its advantage is consistent but **not demonstrated on held-out data** (test ΔAUC +0.076, 95% CI −0.063 to 0.219).
- Switching the primary model would require **new, independent test data**, ideally with patient identifiers. Re-using the 51-image test set to decide would be test-set selection.
- If a demo shows Extra Trees, it should be labelled **experimental** next to the LR result.
- The model search ends here. Further gains are more likely to come from better data than from more model types: colour calibration, patient-level splits, and external validation.

## 9. Threats to validity

1. **No patient identifiers.** Multiple images from the same person (for example, both eyes or repeat visits) could fall on both sides of the split. That would inflate the test metrics, and nothing in the data lets us rule it out. Image numbering (4–710, with gaps) suggests this is a subset of a larger collection. This applies to every model here. Higher-capacity models such as CNNs can memorise person-specific appearance, so leakage would tend to inflate their test scores more than the baseline's. Even so, the CNN did not outperform the baseline. Patient-level splitting will only be possible if identifiers can be recovered from the original data source.
2. **Small test set (n = 51).** This applies to every model; all were evaluated on the same 51 images. The CIs are wide, and a different seed would give noticeably different point estimates.
3. **Single dataset, single source.** No external validation. Population, devices, and Hb measurement method are undocumented here, pending provenance verification.
4. **Threshold selection.** The threshold was chosen on OOF probabilities from the same training folds used to choose C, which is mildly optimistic for the threshold only. The test set played no part.
5. **Crop-geometry signal.** Crop size and shape alone reach CV AUC of about 0.64 (Section 5), so models may partly learn acquisition or cropping characteristics rather than pallor.
6. **CNN non-determinism.** Fine-tuning on MPS is not bit-for-bit reproducible, so its CV score (0.639) could shift slightly on a re-run. It trailed the frozen MobileNetV3 by 0.04 AUC and the colour baseline by 0.07. The chosen frozen-feature model and its test result are deterministic.
7. **Model-search multiplicity.** Seven model families were compared in nested CV. The top-ranked model's CV advantage may be partly selection luck (winner's curse), which is one more reason to require independent test data before switching.
8. **Repeated use of one test set.** The same 51 test images have now been used for three one-shot evaluations (LR baseline, frozen CNN, Extra Trees). XGBoost and CatBoost were deliberately not tested on them. Each was pre-declared and scored once, but choosing whichever experiment looks best on these images would itself be test-set selection. Confirming any model change needs new, independent data.
9. **Label definition.** Anemic is defined as Hb < 11 g/dL. This cutoff is appropriate for some populations (for example, young children and pregnant women) but not for others.

## 10. Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/inspect_dataset.py   # -> derived/manifest.csv
.venv/bin/python scripts/train_baseline.py    # -> models/, outputs/cv_results.json
.venv/bin/python scripts/evaluate.py          # -> outputs/metrics.json, outputs/predictions.csv
.venv/bin/python scripts/cnn_select.py        # training split only -> outputs/cnn_cv_results.json, models/cnn_final.*
.venv/bin/python scripts/cnn_evaluate.py      # ONE-SHOT test evaluation -> outputs/cnn_metrics.json
.venv/bin/python scripts/diagnose_shortcut.py # training split only -> outputs/shortcut_diagnostic.json
.venv/bin/python scripts/classical_select.py  # training split only, nested CV -> outputs/classical_cv_results.json
.venv/bin/python scripts/classical_evaluate.py # ONE-SHOT test evaluation -> outputs/classical_metrics.json
.venv/bin/python scripts/boosting_select.py   # training split only, nested CV -> outputs/boosting_cv_results.json
```

On macOS, XGBoost requires `brew install libomp`. `cnn_evaluate.py`, `classical_evaluate.py`, `classical_select.py` and `boosting_select.py` refuse to overwrite their existing results. The baseline, `cnn_select.py` and `diagnose_shortcut.py` overwrite their outputs on re-run, and the baseline's re-run is byte-identical.

The train/test split is frozen by `derived/split.lock`. Every training and selection script, and the CNN and classical test evaluations, abort if `derived/split.csv` changes. The baseline `evaluate.py` predates the lock and checks only that the split matches the manifest's image order. A read-only, checksummed snapshot of the original baseline results is kept in `outputs/baseline_colour/`.

A re-run reproduces the baseline `outputs/metrics.json` byte for byte. ImageNet weights are downloaded by torchvision to `~/.cache/torch` (about 21 MB each).
