# Pallor

**Conjunctival colour research prototype.** Pallor explores whether colour features from pre-segmented images
of the inner eyelid (the palpebral conjunctiva) are statistically associated with anaemia labels in one
research dataset. It includes a reproducible ML pipeline, documented experiments and a local Gradio
application.

> **Pallor is an experimental research prototype, not a medical device.** Its results are not a diagnosis
> or clinically validated anaemia screening. It cannot measure haemoglobin, and its model score is **not**
> a probability of anaemia. If anaemia is a concern, consult a qualified healthcare professional; a blood
> test is needed to measure haemoglobin.

Built for the ML Empowerment Build Challenge 3.0.

---

## Contents
1. [Project overview](#1-project-overview)
2. [The problem Pallor explores](#2-the-problem-pallor-explores)
3. [What the application does](#3-what-the-application-does)
4. [How the application works](#4-how-the-application-works)
5. [Architecture and data flow](#5-architecture-and-data-flow)
6. [Dataset overview and provenance limitations](#6-dataset-overview-and-provenance-limitations)
7. [Image preprocessing and the 35 colour features](#7-image-preprocessing-and-the-35-colour-features)
8. [Training and evaluation methodology](#8-training-and-evaluation-methodology)
9. [Evaluation results](#9-evaluation-results)
10. [Why Logistic Regression remains the primary model](#10-why-logistic-regression-remains-the-primary-model)
11. [Comparison with Extra Trees and CNN experiments](#11-comparison-with-extra-trees-and-cnn-experiments)
12. [Installation and setup](#12-installation-and-setup)
13. [Running the local application](#13-running-the-local-application)
14. [Running the automated tests](#14-running-the-automated-tests)
15. [Supported input format](#15-supported-input-format)
16. [Limitations and ethical considerations](#16-limitations-and-ethical-considerations)
17. [Future improvements](#17-future-improvements)
18. [Project status](#18-project-status)

---

## 1. Project overview

| Component | Location |
|---|---|
| Dataset inspection and validated manifest | `scripts/inspect_dataset.py` |
| 35 engineered colour features | `scripts/features.py` |
| Baseline training (Logistic Regression, Ridge) and evaluation | `scripts/train_baseline.py`, `scripts/evaluate.py` |
| CNN, classical-ML and boosting experiments | `scripts/cnn_*.py`, `scripts/classical_*.py`, `scripts/boosting_select.py`, `scripts/diagnose_shortcut.py` |
| Aggregate results (no per-image data) | `outputs/*.json` |
| Local research application | `app/app.py`, `app/inference.py` |
| Automated tests | `tests/test_app_inference.py` |
| Research report and model record | [`docs/REPORT.md`](docs/REPORT.md), [`models/README.md`](models/README.md), [`docs/DATASET.md`](docs/DATASET.md) |

## 2. The problem Pallor explores

Anaemia means having too little haemoglobin, and it's measured with a blood test. Clinicians sometimes look
for paleness ("pallor") of the inner eyelid as a physical sign, because this thin, well-vascularised tissue
gets part of its red colour from the blood beneath it.

Pallor tests a narrow research question: **in one labelled dataset, do numerical colour features of
pre-segmented eyelid crops separate images labelled anaemic from those that are not?**

Image colour also depends on lighting, camera and white balance, so a large part of the work is about not
fooling ourselves (see [Section 16](#16-limitations-and-ethical-considerations)).

## 3. What the application does

The local Gradio app has four tabs:

| Tab | Purpose |
|---|---|
| **Analyse** | Upload a pre-segmented conjunctiva crop (transparent-background PNG). The app validates it, extracts the 35 colour features, applies the saved Logistic Regression model, and explains the output in plain language. **Try the demo** loads a synthetic test image into the same workflow, so first-time users can run the real pipeline without preparing an image. |
| **Examples** | Synthetic, code-generated images showing accepted, rejected and out-of-range inputs. Each states its expected behaviour and is compared with what actually happened. No dataset images are shown, because the dataset's licence and sharing permissions are unresolved. |
| **Performance** | Held-out test results, the model-selection rationale and cross-validation results, in separate sections. Every figure is read at runtime from the aggregate files in `outputs/`. |
| **Limitations** | Data, model, input and clinical limitations. |

A result is shown in four parts:
1. **Model output:** the predicted dataset label (for example, "Resembles the dataset's anaemic-labelled
   examples" or "Did not cross the research threshold"), marked as not a medical diagnosis, plus the model
   score with the note *"not a calibrated probability of anaemia"*.
2. **Plain-English explanation:** what the output means, the feature groups used, and the recommended next
   step (a blood test if anaemia is a concern).
3. **Input warnings:** whether the image falls outside the training data's ranges.
4. **Technical details:** in a collapsed section.

Synthetic test images (the demo and the Examples tab) are recognised by their file contents. They show the
actual model output in neutral wording ("Score above/below the research threshold"), without a dataset label,
because a code-generated image has no medical meaning.

## 4. How the application works

1. **Model integrity check:** at start-up, the SHA-256 hash of `models/classifier.joblib` must match the
   baseline snapshot's `SHA256SUMS`. If it doesn't, the app refuses to start.
2. **Input validation:** the file must be a real PNG (checked by its signature bytes, not its extension),
   with usable transparency. Anything else is rejected with a reason ([Section 15](#15-supported-input-format)).
3. **Feature extraction:** the unmodified training-time extractor (`scripts/features.py`) reads the uploaded
   file directly, so its alpha channel is preserved. Only fully opaque pixels count as tissue.
4. **Scoring:** the 35 features are arranged in the saved training order and passed to the saved
   `StandardScaler → LogisticRegression` pipeline.
5. **Classification:** the model score is compared with the stored research threshold (0.502, chosen on
   training data only).
6. **Range warnings:** if the image size, tissue-pixel count or any colour feature lies outside the range
   seen in the 200 training images, the result is flagged *"treat this result as unreliable"*.
7. **Presentation:** a plain-language result first, then secondary technical details.

The app never retrains anything, and it runs locally only: bound to `127.0.0.1:7860`, no share link, Gradio
analytics disabled, and system fonts only, so no external font requests.

## 5. Architecture and data flow

```
Conjunctiva crop (transparent-background PNG)
  -> Input validation        app/inference.py  validate_png()
  -> Colour feature extraction  scripts/features.py  extract()   (35 features, opaque pixels only)
  -> Saved Logistic Regression  models/classifier.joblib         (integrity-checked; not retrained)
  -> Research classification    model score vs stored threshold
  -> Out-of-range checks        against training-split feature ranges
  -> Plain-language result      app/app.py  (Gradio, 127.0.0.1 only)
```

The research pipeline that produced the model:

```
data/ (local only)
  -> scripts/inspect_dataset.py   validated manifest      (derived/, local only)
  -> scripts/train_baseline.py    fixed 80/20 split, CV model selection, saved models
  -> scripts/evaluate.py          one held-out evaluation -> outputs/metrics.json
  -> further experiments          CNNs, classical ML, boosting -> outputs/*.json
```

## 6. Dataset overview and provenance limitations

| Item | Value |
|---|---|
| Source | Downloaded from Mendeley Data. **Licence, DOI and citation are pending verification** ([`docs/DATASET.md`](docs/DATASET.md)) |
| Images used | 251 PNG conjunctiva crops: 110 in `Anaemic/`, 141 in `Non-Anaemic/` |
| Excluded | `All Categories/`, which holds byte-identical copies of the two class folders |
| Labels | Per-image haemoglobin (Hb, g/dL) from the accompanying workbook, range 4.2–15.0 |
| Label rule | "Anaemic" means Hb < 11 g/dL, which matches the folder labels for all 251 images. This is the dataset's labelling convention, **not a universal diagnostic threshold** |
| Patient identifiers | None |

**The dataset is not included in this repository** and must not be redistributed until its licence is
verified. Per-image records, predictions and model binaries are excluded from Git as well.

**Known concerns:**
- Small sample size.
- A single source.
- An undocumented Hb measurement method and population.
- Uncontrolled lighting and cameras.
- **Possible patient-level leakage.** There are no patient IDs, so images from one person (for example,
  both eyes) could fall into both training and test data.

## 7. Image preprocessing and the 35 colour features

The images are pre-segmented RGBA crops: background pixels are fully transparent and stored as black.
**Only fully opaque pixels (alpha = 255) are treated as tissue**, which excludes the background and the
soft mask edge.

| Group | Count | Features |
|---|---|---|
| RGB | 15 | Mean, standard deviation, and 10th/50th/90th percentiles of red, green and blue |
| Chromaticity | 2 | R/(R+G+B) and G/(R+G+B) |
| HSV | 6 | Hue (averaged as an angle: sine and cosine), saturation mean/std, value mean/std |
| CIELAB | 7 | L\*, a\*, b\* mean and std; a\*/L\* |
| Redness indices | 4 | Erythema index 100·log10(R/G), (R−G)/(R+G), R/G, fraction of red-dominant pixels |
| Tissue size | 1 | Number of opaque pixels (`n_tissue_px`) |

The features are per-image statistics with no learned parameters. All fitted preprocessing (scaling) is done
inside the cross-validation pipelines.

## 8. Training and evaluation methodology

- **Fixed split:**
  - 80/20, seed 42, stratified on Hb bins (edges 8, 9.5, 11, 12, 13 g/dL).
  - **200 training images** (88 anaemic-labelled) and **51 held-out test images** (22 anaemic-labelled).
  - The split is locked by a SHA-256 fingerprint.
- **Model selection on training data only:**
  - Repeated stratified cross-validation (5 folds × 5 repeats = 25 folds).
  - For the classical model comparison, **nested** CV: an inner loop tunes the hyperparameters and an outer
    loop estimates performance on folds the tuning never saw.
- **Decision threshold:** chosen to maximise Youden's J (sensitivity + specificity − 1) on out-of-fold
  *training* predictions.
- **Held-out test:**
  - Each test-evaluated model was declared in advance and scored **once**; the scripts refuse to re-run.
  - Uncertainty is reported as 95% bootstrap intervals (2,000 resamples); comparisons use a **paired**
    bootstrap on the same images.

**Glossary:**
- **ROC-AUC:** the probability that a randomly chosen anaemic-labelled image scores higher than a randomly
  chosen non-anaemic-labelled one. 0.5 is chance; 1.0 is perfect ranking.
- **Sensitivity:** the share of anaemic-labelled images placed above the threshold.
- **Specificity:** the share of non-anaemic-labelled images placed below the threshold.
- **Model score:** the model's raw output between 0 and 1. It is **not calibrated** and is **not a
  probability of anaemia**.

## 9. Evaluation results

### Held-out test set (51 images, scored once)

| Model | ROC-AUC (95% CI) | Sensitivity | Specificity | Precision | F1 | TP / FN / FP / TN |
|---|---|---|---|---|---|---|
| **Logistic Regression (primary, used by the app)** | **0.724** (0.567–0.865) | 0.727 (16 of 22) | 0.690 (20 of 29) | 0.640 | 0.681 | 16 / 6 / 9 / 20 |
| Extra Trees (experimental candidate) | 0.800 (0.668–0.914) | 0.682 (15 of 22) | 0.759 (22 of 29) | 0.682 | 0.682 | 15 / 7 / 7 / 22 |
| Frozen MobileNetV3 + LR (not adopted) | 0.660 (0.508–0.803) | 0.364 (8 of 22) | 0.759 (22 of 29) | 0.533 | 0.432 | 8 / 14 / 7 / 22 |

The same 51 test images were used for these three one-shot evaluations, so they're no longer an untouched
final test.

### Cross-validation on the training split (not held-out results)

Nested CV, the same 25 outer folds and the same 35 features for every model:

| Model | Nested-CV ROC-AUC (mean ± SD) |
|---|---|
| Extra Trees | 0.789 ± 0.061 |
| RBF-SVM | 0.781 ± 0.077 |
| CatBoost | 0.762 ± 0.071 |
| Random Forest | 0.743 ± 0.063 |
| XGBoost | 0.724 ± 0.073 |
| HistGradientBoosting | 0.720 ± 0.070 |
| Logistic Regression | 0.718 ± 0.077 |

### Haemoglobin regression

Ridge regression was tested to see whether Hb itself could be estimated. Its test mean absolute error was
1.639 g/dL, against 1.693 for always predicting the training mean (R² −0.036). That isn't a meaningful
improvement, so **Pallor does not estimate haemoglobin**.

## 10. Why Logistic Regression remains the primary model

A selection rule was fixed **before** each test evaluation: a challenger replaces Logistic Regression only
if it beats it in cross-validation **and** its paired test ROC-AUC difference has a 95% interval entirely
above zero.

- Extra Trees' paired difference was **+0.076 (95% CI −0.063 to 0.219)**. The interval includes zero, so
  the improvement is **not established** on 51 images.
- Choosing whichever model scored highest on a repeatedly used test set would make that test set part of
  model selection and overstate future performance.
- Logistic Regression is also the simplest and most transparent of the models tested.

## 11. Comparison with Extra Trees and CNN experiments

- **Extra Trees** had the best nested-CV score and beat Logistic Regression in 23 of 25 outer folds. Its
  held-out ROC-AUC (0.800) agrees with its CV estimate. It is the strongest **experimental candidate**;
  confirming it needs new, independent data.
- **CNNs:**
  - Frozen ImageNet backbones: MobileNetV3-Large 0.716 ± 0.087 and EfficientNet-B0 0.649 ± 0.067, both
    non-nested CV.
  - Partially fine-tuned MobileNetV3-Large: 0.639 ± 0.091 (5-fold CV).
  - All were below the colour-feature Logistic Regression on the same protocol (0.741 ± 0.079).
  - The frozen MobileNetV3 scored 0.660 on the held-out test and missed 14 of 22 anaemic-labelled images.
  - With 200 training images, the 35 targeted colour features generalised better than generic image embeddings.
- **Shortcut check:** crop size alone predicts the labels better than chance (CV ROC-AUC about 0.65).
  Removing the size feature barely changes the colour models (Logistic Regression 0.741 → 0.740; Extra Trees
  0.789 → 0.787), but this rules out only one possible shortcut.

Full details: [`docs/REPORT.md`](docs/REPORT.md) and [`models/README.md`](models/README.md).

## 12. Installation and setup

Developed with Python 3.13 on macOS (Apple silicon). Pinned dependencies are in `requirements.txt`; the
saved model requires **scikit-learn 1.9.1**.

```bash
git clone https://github.com/arrzinee/Pallor.git
cd Pallor
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt   # app, experiments and pytest
```

On macOS, XGBoost (used only by `scripts/boosting_select.py`) also needs `brew install libomp`.

**Local artifacts.** The app needs files that aren't in this repository because they're derived from the
dataset: `models/classifier.joblib`, `outputs/baseline_colour/SHA256SUMS` and the files in `derived/`.
Producing them requires the dataset, placed in `data/` as described in [`docs/DATASET.md`](docs/DATASET.md),
and the pipeline:

```bash
.venv/bin/python scripts/inspect_dataset.py   # validated manifest -> derived/
.venv/bin/python scripts/train_baseline.py    # split, CV selection, saved models
.venv/bin/python scripts/evaluate.py          # held-out evaluation -> outputs/metrics.json
```

In the original research run, the baseline snapshot (`outputs/baseline_colour/`, with `SHA256SUMS`) and
`derived/split.lock` were then created as separate steps. The scripts don't create them, and this fresh-clone
path hasn't been re-verified.

## 13. Running the local application

```bash
.venv/bin/python app/app.py
```

Then open **http://127.0.0.1:7860**, and press Ctrl+C to stop. The app listens on 127.0.0.1 only, never
creates a share link, and has Gradio analytics disabled.

## 14. Running the automated tests

```bash
.venv/bin/python -m pytest tests/test_app_inference.py -v
```

There are 33 tests, covering:
- model integrity (including a tampered-model test)
- feature order, and exact parity with the saved held-out predictions
- rejection of malformed and unsupported files, and out-of-range warnings
- transparency preservation through a real local upload
- dashboard values read from the result files, with test and CV results kept separate
- the disclaimer and the wording that the score is not a probability
- local-only launch settings
- no modification of research artifacts

Tests that need the local dataset artifacts are skipped automatically when those files are absent.

## 15. Supported input format

| Requirement | Detail |
|---|---|
| Format | PNG (verified by file signature), at most 10 MB, at most 4,096 px per side, not animated |
| Transparency | Required. The background must be fully transparent, and only conjunctiva tissue opaque, as in the training crops |
| Tissue | At least 100 fully opaque pixels |
| Not supported | Ordinary photographs, JPEGs, and images without a segmented transparent background |
| Warnings | Inputs outside the training ranges are flagged as unreliable. The ranges are width 79–507 px, height 34–273 px, 1,280–37,158 tissue pixels, and each colour feature's observed range |

## 16. Limitations and ethical considerations

- **Not a medical device; no clinical validation.** Pallor must not be used to diagnose anaemia or rule it out.
- **The model score is not a calibrated probability.** A score of 0.86 does not mean an 86% chance of anaemia.
- **Small, single-source dataset** (251 images; 51 held out), so the confidence intervals are wide.
- **Unresolved provenance.** The dataset licence and citation are pending verification.
- **Possible patient-level leakage.** There are no patient identifiers.
- **Possible dataset-specific confounding.** Crop size predicts the labels, and lighting and cameras vary.
  The models may partly learn such shortcuts rather than pallor.
- **No evidence of generalisation** to other people, cameras, lighting conditions, skin tones, ages or
  populations.
- **The Hb < 11 g/dL label is a dataset convention,** not a universal diagnostic threshold.
- **Strict input requirement:** pre-segmented, transparent-background PNG crops only.

## 17. Future improvements

- Verify the dataset licence, DOI and citation.
- Independent data with patient identifiers and documented haemoglobin measurement, for patient-level
  splits and external validation.
- Colour-calibrated image capture (for example, a reference card) to reduce lighting effects.
- Re-evaluate Extra Trees on new, independent test data.
- Calibrate scores only if suitable data becomes available.
- Make the fresh-clone setup fully scripted, including the baseline snapshot and split lock.
- Separate the app's runtime dependencies from the heavier experiment dependencies.

## 18. Project status

- **ML experiments:** complete. The model search is closed.
- **Primary model:** Logistic Regression. Extra Trees is the experimental candidate.
- **Application:** working locally, with 33 automated tests.
- **Research status:** experimental prototype; **not clinically validated**.
- **Outstanding:** dataset licence verification; independent validation data.
