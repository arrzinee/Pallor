"""Classical-ML model comparison on the TRAINING split only (nested CV).

Test rows are never used by this script.

Features: the same 35 colour features as the baseline (derived/features.csv).

Nested cross-validation:
  outer: RepeatedStratifiedKFold(5, 5, seed 42) over the training order of
         train_baseline.py, i.e. the same 25 folds used for the baseline and CNN
         comparisons. Used only to estimate performance.
  inner: StratifiedKFold(5, shuffle, seed 42) grid search inside each outer
         training fold. Used only to tune hyperparameters.
Scaling is inside the pipeline for LR and SVM, so it is re-fitted on every
inner and outer training fold. Tree models need no scaling.

Selection rule (declared before any results were seen):
  The candidate is the single non-reference model with the highest nested-CV
  mean ROC-AUC. Ties (equal at 3 decimals) go to the simpler model, in the
  order SVM < RF = ET < HGB = XGB. Exactly one candidate goes to the one-shot
  test evaluation (classical_evaluate.py). It is adopted over the logistic
  regression baseline only if it beat LR in nested CV AND the paired test
  ΔAUC 95% CI lies above 0.

Diagnostic (never used for selection): the same nested CV without
`n_tissue_px`, because crop size alone is predictive (shortcut_diagnostic.json).

Outputs: outputs/classical_cv_results.json, outputs/classical_train_oof_predictions.csv,
         models/classical_final.joblib
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import (DERIVED, MODELS, OUTPUTS, SEED, TEST_SIZE, hb_strata,  # noqa: E402
                            verify_split_lock)
from sklearn.model_selection import train_test_split  # noqa: E402

REFERENCE = "logistic_regression"
COMPLEXITY = {"logistic_regression": 0, "svm_rbf": 1, "random_forest": 2, "extra_trees": 2,
              "hist_gradient_boosting": 3, "xgboost": 3}


def model_specs():
    trees = {"n_estimators": 300, "class_weight": "balanced", "random_state": SEED, "n_jobs": 1}
    tree_grid = {"max_depth": [None, 4], "min_samples_leaf": [1, 3, 10], "max_features": ["sqrt", 0.5]}
    specs = {
        "logistic_regression": (
            make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=10000)),
            {"logisticregression__C": np.logspace(-3, 5, 17)}),
        "svm_rbf": (
            make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
            {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.001, 0.01, 0.1]}),
        "random_forest": (RandomForestClassifier(**trees), tree_grid),
        "extra_trees": (ExtraTreesClassifier(**trees), tree_grid),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(class_weight="balanced", l2_regularization=1.0,
                                           early_stopping=False, random_state=SEED),
            {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "max_iter": [100, 300],
             "min_samples_leaf": [10, 20]}),
    }
    unavailable = {}
    try:
        from xgboost import XGBClassifier
        specs["xgboost"] = (
            XGBClassifier(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                          random_state=SEED, n_jobs=1),
            {"n_estimators": [100, 300], "max_depth": [2, 3], "learning_rate": [0.03, 0.1]})
    except Exception as exc:  # noqa: BLE001 - record why it was skipped
        unavailable["xgboost"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return specs, unavailable


def scores(estimator, X):
    """Continuous score for ROC/threshold: probability if available, else SVM decision value."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def inner_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def nested_cv(estimator, grid, X, y, outer):
    search = GridSearchCV(estimator, grid, scoring="roc_auc", cv=inner_cv(), n_jobs=1)
    res = cross_validate(search, X, y, cv=outer, scoring="roc_auc", return_estimator=True, n_jobs=-1)
    chosen = [{k: (v if isinstance(v, (str, type(None))) else float(v)) for k, v in e.best_params_.items()}
              for e in res["estimator"]]
    return res["test_score"], chosen


def main():
    t0 = time.time()
    verify_split_lock()
    # The published results are frozen (and xgboost, now importable, would join the
    # candidate pool on a re-run), so never overwrite them.
    if (OUTPUTS / "classical_cv_results.json").exists():
        sys.exit("outputs/classical_cv_results.json already exists; refusing to overwrite prior results.")
    manifest = pd.read_csv(DERIVED / "manifest.csv")
    split = pd.read_csv(DERIVED / "split.csv")
    assert list(split["image_id"]) == list(manifest["image_id"])
    # Reproduce train_baseline.py's training order so the outer folds are identical.
    idx_train, _ = train_test_split(np.arange(len(manifest)), test_size=TEST_SIZE,
                                    stratify=hb_strata(manifest["hb_g_dl"].to_numpy()), random_state=SEED)
    assert set(manifest["image_id"].iloc[idx_train]) == set(split.loc[split["split"] == "train", "image_id"])
    tr = manifest.iloc[idx_train].reset_index(drop=True)
    y = tr["anemic"].to_numpy()
    feats = pd.read_csv(DERIVED / "features.csv").set_index("image_id").loc[tr["image_id"]]
    feature_names = list(feats.columns)
    X = feats.to_numpy()
    X_nosize = feats.drop(columns="n_tissue_px").to_numpy()
    outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y))
    print(f"training rows: {len(y)} (anemic {y.sum()}, non-anemic {(y == 0).sum()}); "
          f"{len(feature_names)} features; outer folds: {len(outer)}")

    specs, unavailable = model_specs()
    for name, why in unavailable.items():
        print(f"[{name}] UNAVAILABLE: {why}")

    results, fold_auc = {}, {}
    for name, (est, grid) in specs.items():
        t = time.time()
        auc, chosen = nested_cv(est, grid, X, y, outer)
        auc_ns, _ = nested_cv(est, grid, X_nosize, y, outer)
        fold_auc[name] = auc
        results[name] = {
            "nested_cv_auc_mean": float(auc.mean()), "nested_cv_auc_std": float(auc.std()),
            "per_fold_auc": [round(float(v), 4) for v in auc],
            "grid": {k: [v if isinstance(v, (str, type(None))) else float(v) for v in vals] for k, vals in grid.items()},
            "inner_selected_params_per_outer_fold": chosen,
            "diagnostic_without_n_tissue_px": {"nested_cv_auc_mean": float(auc_ns.mean()),
                                               "nested_cv_auc_std": float(auc_ns.std())},
            "runtime_s": round(time.time() - t, 1),
        }
        print(f"[{name:24s}] nested-CV AUC {auc.mean():.3f} ± {auc.std():.3f} | without n_tissue_px "
              f"{auc_ns.mean():.3f} | {time.time() - t:.0f}s")

    ref = fold_auc[REFERENCE]
    for name in results:
        d = fold_auc[name] - ref
        results[name]["paired_vs_lr"] = {"mean_delta_auc": float(d.mean()), "sd_delta_auc": float(d.std()),
                                         "folds_better_than_lr": int((d > 0).sum()), "n_folds": len(d)}

    # --- Pre-declared selection rule -----------------------------------------
    pool = [n for n in results if n != REFERENCE]
    candidate = sorted(pool, key=lambda n: (-round(results[n]["nested_cv_auc_mean"], 3), COMPLEXITY[n]))[0]
    beat_lr = results[candidate]["nested_cv_auc_mean"] > results[REFERENCE]["nested_cv_auc_mean"]
    print(f"CANDIDATE: {candidate} (nested-CV AUC {results[candidate]['nested_cv_auc_mean']:.3f} vs LR "
          f"{results[REFERENCE]['nested_cv_auc_mean']:.3f}; beat LR in CV: {beat_lr})")

    # --- Final fit of the candidate on the full training split ---------------
    est, grid = specs[candidate]
    final_search = GridSearchCV(est, grid, scoring="roc_auc", cv=inner_cv(), n_jobs=-1).fit(X, y)
    best = final_search.best_estimator_
    method = "predict_proba" if hasattr(best, "predict_proba") else "decision_function"
    oof = cross_val_predict(best, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method=method)
    oof = oof[:, 1] if oof.ndim == 2 else oof
    fpr, tpr, thr = roc_curve(y, oof)
    j = int(np.argmax(tpr - fpr))
    threshold = float(thr[j])
    MODELS.mkdir(exist_ok=True)
    joblib.dump({"name": candidate, "estimator": best, "threshold": threshold, "score": method,
                 "feature_names": feature_names}, MODELS / "classical_final.joblib")
    pd.DataFrame({"image_id": tr["image_id"], "split": "train_oof", "anemic_true": y, "score": oof,
                  "pred_anemic": (oof >= threshold).astype(int)}).to_csv(
        OUTPUTS / "classical_train_oof_predictions.csv", index=False)

    out = {
        "scope": "training split only (n=200); test rows never used",
        "features": f"{len(feature_names)} colour features (same as baseline)",
        "outer_cv": "RepeatedStratifiedKFold(5,5,seed=42), same folds as baseline/CNN comparisons",
        "inner_cv": "StratifiedKFold(5, shuffle, seed=42) grid search, ROC-AUC",
        "selection_rule": "best non-reference nested-CV mean AUC (3 dp); ties -> simpler "
                          "(SVM < RF = ET < HGB = XGB); exactly one candidate tested once",
        "adoption_rule": "adopt over LR only if candidate beat LR in nested CV AND paired test ΔAUC 95% CI > 0",
        "unavailable_models": unavailable,
        "models": results,
        "candidate": candidate,
        "candidate_beat_lr_in_cv": bool(beat_lr),
        "candidate_final": {"params": {k: (v if isinstance(v, (str, type(None))) else float(v))
                                       for k, v in final_search.best_params_.items()},
                            "inner_cv_auc_full_train": float(final_search.best_score_),
                            "score_type": method, "threshold": threshold,
                            "threshold_rule": "max Youden J on 5-fold OOF training scores",
                            "train_oof_sensitivity": float(tpr[j]), "train_oof_specificity": float(1 - fpr[j])},
        "runtime_s": round(time.time() - t0, 1),
    }
    (OUTPUTS / "classical_cv_results.json").write_text(json.dumps(out, indent=2))
    print(f"final params {out['candidate_final']['params']}; threshold {threshold:.4f} "
          f"(train OOF sens {tpr[j]:.3f}, spec {1 - fpr[j]:.3f}); total {out['runtime_s']}s")


if __name__ == "__main__":
    main()
