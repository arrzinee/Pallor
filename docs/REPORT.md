# Pallor — Baseline Experiment Report

> **Research prototype. Not clinically validated. Not for diagnosis or treatment decisions.**
> Dataset licence and provenance are **pending verification** (see [DATASET.md](DATASET.md)).

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
| Fine-tuned MobileNetV3-Large (30 epochs) | — | 0.639 ± 0.091 (per fold 0.49–0.74) |

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

## 6. Threats to validity

1. **No patient identifiers.** Multiple images from the same person (for example, both eyes or repeat visits) could fall on both sides of the split. That would inflate the test metrics, and nothing in the data lets us rule it out. Image numbering (4–710, with gaps) suggests this is a subset of a larger collection. This applies to every model here. Higher-capacity models such as CNNs can memorise person-specific appearance, so leakage would tend to inflate their test scores more than the baseline's. Even so, the CNN did not outperform the baseline. Patient-level splitting will only be possible if identifiers can be recovered from the original data source.
2. **Small test set (n = 51).** This applies to both the baseline and the CNN, which were evaluated on the same 51 images. The CIs are wide, and a different seed would give noticeably different point estimates.
3. **Single dataset, single source.** No external validation. Population, devices, and Hb measurement method are undocumented here, pending provenance verification.
4. **Threshold selection.** The threshold was chosen on OOF probabilities from the same training folds used to choose C, which is mildly optimistic for the threshold only. The test set played no part.
5. **Crop-geometry signal.** Crop size and shape alone reach CV AUC of about 0.64 (Section 5), so models may partly learn acquisition or cropping characteristics rather than pallor.
6. **CNN non-determinism.** Fine-tuning on MPS is not bit-for-bit reproducible, so its CV score (0.639) could shift slightly on a re-run. It trailed the frozen MobileNetV3 by 0.04 AUC and the colour baseline by 0.07. The chosen frozen-feature model and its test result are deterministic.
7. **Label definition.** Anemic is defined as Hb < 11 g/dL. This cutoff is appropriate for some populations (for example, young children and pregnant women) but not for others.

## 7. Reproduce

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/inspect_dataset.py   # -> derived/manifest.csv
.venv/bin/python scripts/train_baseline.py    # -> models/, outputs/cv_results.json
.venv/bin/python scripts/evaluate.py          # -> outputs/metrics.json, outputs/predictions.csv
.venv/bin/python scripts/cnn_select.py        # training split only -> outputs/cnn_cv_results.json, models/cnn_final.*
.venv/bin/python scripts/cnn_evaluate.py      # ONE-SHOT test evaluation -> outputs/cnn_metrics.json
.venv/bin/python scripts/diagnose_shortcut.py # training split only -> outputs/shortcut_diagnostic.json
```

The train/test split is frozen by `derived/split.lock`; every script aborts if `derived/split.csv` changes. A read-only, checksummed snapshot of the original baseline results is kept in `outputs/baseline_colour/`.

A re-run reproduces the baseline `outputs/metrics.json` byte for byte. ImageNet weights are downloaded by torchvision to `~/.cache/torch` (about 21 MB each).
