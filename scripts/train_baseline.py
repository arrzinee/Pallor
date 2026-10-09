"""Train colour-feature baselines on the training split only.

  A. Classification: anemic (Hb < 11 g/dL), scaled logistic regression.
  B. Regression: Hb in g/dL, scaled ridge regression, vs a training-mean baseline.

Steps:
  1. Build features from derived/manifest.csv (cached at derived/features.csv).
  2. Make one fixed, stratified train/test split (derived/split.csv).
  3. On the training split only: choose hyperparameters by repeated stratified
     CV, then choose the classification threshold from out-of-fold training
     probabilities.
  4. Fit the final models on the full training split and save them to models/.

The test split is never touched here; scripts/evaluate.py is the only place
that reads test labels.
"""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_curve
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DERIVED, MODELS, OUTPUTS = ROOT / "derived", ROOT / "models", ROOT / "outputs"
SEED = 42
TEST_SIZE = 0.2
HB_THRESHOLD = 11.0
# Hb strata for splitting; 11.0 is an edge, so every stratum is single-class.
HB_BIN_EDGES = [-np.inf, 8.0, 9.5, 11.0, 12.0, 13.0, np.inf]


def hb_strata(hb):
    return np.digitize(hb, HB_BIN_EDGES[1:-1])


def load_features():
    manifest = pd.read_csv(DERIVED / "manifest.csv")
    cache = DERIVED / "features.csv"
    if cache.exists():
        cached = pd.read_csv(cache)
        if list(cached["image_id"]) == list(manifest["image_id"]):
            return manifest, cached.drop(columns="image_id")
    rows = [features.extract(ROOT / p) for p in manifest["path"]]
    feats = pd.DataFrame(rows)
    feats.insert(0, "image_id", manifest["image_id"])
    feats.to_csv(cache, index=False)
    return manifest, feats.drop(columns="image_id")


def verify_split_lock():
    """Abort unless derived/split.csv matches the SHA-256 recorded in derived/split.lock."""
    lock = DERIVED / "split.lock"
    if not lock.exists():
        return
    digest = hashlib.sha256((DERIVED / "split.csv").read_bytes()).hexdigest()
    if digest != lock.read_text().strip():
        sys.exit(f"derived/split.csv does not match derived/split.lock ({digest}); refusing to continue")


def split_counts(df):
    return {"n": len(df), "n_anemic": int(df["anemic"].sum()),
            "n_non_anemic": int((df["anemic"] == 0).sum()),
            "frac_anemic": round(float(df["anemic"].mean()), 3),
            "hb_mean": round(float(df["hb_g_dl"].mean()), 3),
            "hb_min": float(df["hb_g_dl"].min()), "hb_max": float(df["hb_g_dl"].max())}


def main():
    manifest, X = load_features()
    feature_names = list(X.columns)
    strata = hb_strata(manifest["hb_g_dl"].to_numpy())

    idx_train, idx_test = train_test_split(np.arange(len(manifest)), test_size=TEST_SIZE,
                                           stratify=strata, random_state=SEED)
    split = pd.Series("train", index=manifest.index)
    split.iloc[idx_test] = "test"
    split_df = pd.DataFrame({"image_id": manifest["image_id"], "split": split})
    if (DERIVED / "split.lock").exists():
        # The split is frozen: the regenerated split must be identical to the locked file.
        verify_split_lock()
        if not split_df.equals(pd.read_csv(DERIVED / "split.csv")):
            sys.exit("regenerated split differs from locked derived/split.csv; refusing to overwrite")
    else:
        split_df.to_csv(DERIVED / "split.csv", index=False)

    tr = manifest.iloc[idx_train]
    X_tr = X.iloc[idx_train].to_numpy()
    y_tr, hb_tr, s_tr = tr["anemic"].to_numpy(), tr["hb_g_dl"].to_numpy(), strata[idx_train]
    print("split counts:", json.dumps({"train": split_counts(tr), "test": split_counts(manifest.iloc[idx_test])}))

    # Repeated stratified CV on the training split only. Regression folds are
    # stratified on Hb strata (which also stratify the class).
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    cv_cls = list(rskf.split(X_tr, y_tr))
    cv_reg = list(rskf.split(X_tr, s_tr))

    # --- A. Classification ---------------------------------------------------
    clf_search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=10000)),
        {"logisticregression__C": np.logspace(-3, 5, 17)},
        scoring="roc_auc", cv=cv_cls, n_jobs=-1)
    clf_search.fit(X_tr, y_tr)
    best = clf_search.best_index_
    cls_cv_auc = clf_search.cv_results_
    print(f"[cls] best C={clf_search.best_params_['logisticregression__C']:.4g} "
          f"CV AUC={cls_cv_auc['mean_test_score'][best]:.3f} ± {cls_cv_auc['std_test_score'][best]:.3f}")

    # Threshold from out-of-fold training probabilities (Youden's J).
    # Note: C was chosen on these same training folds, so the OOF estimate is
    # mildly optimistic; it is used only to place the threshold.
    oof_prob = cross_val_predict(clf_search.best_estimator_, X_tr, y_tr,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                 method="predict_proba")[:, 1]
    fpr, tpr, thr = roc_curve(y_tr, oof_prob)
    j = tpr - fpr
    threshold = float(thr[np.argmax(j)])
    print(f"[cls] threshold={threshold:.3f} (train OOF sens={tpr[np.argmax(j)]:.3f}, spec={1 - fpr[np.argmax(j)]:.3f})")

    clf = clf_search.best_estimator_  # refit on the full training split by GridSearchCV

    # --- B. Regression -------------------------------------------------------
    reg_search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                              {"ridge__alpha": np.logspace(-2, 4, 13)},
                              scoring="neg_mean_absolute_error", cv=cv_reg, n_jobs=-1)
    reg_search.fit(X_tr, hb_tr)
    rb = reg_search.best_index_
    reg_cv_mae = -reg_search.cv_results_["mean_test_score"][rb]
    reg_cv_sd = reg_search.cv_results_["std_test_score"][rb]
    dummy_cv = cross_validate(DummyRegressor(strategy="mean"), X_tr, hb_tr, cv=cv_reg,
                              scoring="neg_mean_absolute_error")
    dummy_cv_mae = -dummy_cv["test_score"].mean()
    print(f"[reg] best alpha={reg_search.best_params_['ridge__alpha']:.4g} CV MAE={reg_cv_mae:.3f} ± {reg_cv_sd:.3f} "
          f"vs train-mean baseline CV MAE={dummy_cv_mae:.3f} ± {dummy_cv['test_score'].std():.3f}")
    oof_hb = cross_val_predict(reg_search.best_estimator_, X_tr, hb_tr,
                               cv=list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_tr, s_tr)))
    reg = reg_search.best_estimator_
    baseline = DummyRegressor(strategy="mean").fit(X_tr, hb_tr)

    # --- Save ----------------------------------------------------------------
    MODELS.mkdir(exist_ok=True)
    OUTPUTS.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "threshold": threshold, "feature_names": feature_names}, MODELS / "classifier.joblib")
    joblib.dump({"model": reg, "baseline": baseline, "feature_names": feature_names}, MODELS / "regressor.joblib")

    pd.DataFrame({"image_id": tr["image_id"].to_numpy(), "split": "train_oof",
                  "hb_true": hb_tr, "anemic_true": y_tr,
                  "prob_anemic": oof_prob, "pred_anemic": (oof_prob >= threshold).astype(int),
                  "hb_pred_ridge": oof_hb}).to_csv(OUTPUTS / "train_oof_predictions.csv", index=False)

    cv_summary = {
        "seed": SEED, "test_size": TEST_SIZE, "hb_bin_edges": [str(e) for e in HB_BIN_EDGES],
        "cv_scheme": "RepeatedStratifiedKFold(5 splits x 5 repeats) on training split only",
        "n_features": len(feature_names), "feature_names": feature_names,
        "classification": {
            "model": "StandardScaler + LogisticRegression(class_weight=balanced)",
            "selected_C": float(clf_search.best_params_["logisticregression__C"]),
            "cv_auc_mean": float(cls_cv_auc["mean_test_score"][best]),
            "cv_auc_std": float(cls_cv_auc["std_test_score"][best]),
            "grid": {f"{c:.4g}": round(float(m), 4) for c, m in
                     zip(cls_cv_auc["param_logisticregression__C"], cls_cv_auc["mean_test_score"])},
            "threshold": threshold, "threshold_rule": "max Youden J on 5-fold OOF training probabilities",
        },
        "regression": {
            "model": "StandardScaler + Ridge",
            "selected_alpha": float(reg_search.best_params_["ridge__alpha"]),
            "cv_mae_mean": float(reg_cv_mae), "cv_mae_std": float(reg_cv_sd),
            "baseline_cv_mae_mean": float(dummy_cv_mae), "baseline_cv_mae_std": float(dummy_cv["test_score"].std()),
            "grid": {f"{a:.4g}": round(float(-m), 4) for a, m in
                     zip(reg_search.cv_results_["param_ridge__alpha"], reg_search.cv_results_["mean_test_score"])},
        },
    }
    (OUTPUTS / "cv_results.json").write_text(json.dumps(cv_summary, indent=2))
    print("saved models/ and outputs/cv_results.json, outputs/train_oof_predictions.csv")


if __name__ == "__main__":
    main()
