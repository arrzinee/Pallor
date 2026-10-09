"""Evaluate the frozen baseline models on the held-out test split.

This is the only script that reads test labels. Models, hyperparameters and
the classification threshold are all fixed by scripts/train_baseline.py.

Outputs:
  outputs/metrics.json      split counts, CV summary, test metrics with 95% bootstrap CIs
  outputs/predictions.csv   per-image test predictions + training OOF predictions,
                            alongside reference Hb and labels (gitignored)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_baseline import DERIVED, HB_THRESHOLD, MODELS, OUTPUTS, SEED, load_features, split_counts  # noqa: E402

N_BOOT = 2000


def cls_metrics(y, prob, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y, prob),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": (tp + tn) / len(y),
    }


def reg_metrics(hb, pred, base):
    return {
        "mae": mean_absolute_error(hb, pred),
        "rmse": float(np.sqrt(mean_squared_error(hb, pred))),
        "r2": r2_score(hb, pred),
        "baseline_mae": mean_absolute_error(hb, base),
        "mae_improvement_vs_baseline": mean_absolute_error(hb, base) - mean_absolute_error(hb, pred),
        "pearson_r": float(np.corrcoef(hb, pred)[0, 1]),
    }


def bootstrap(fn, n, rng, stratify=None):
    """Percentile 95% CI of every metric returned by fn(indices)."""
    draws = []
    for _ in range(N_BOOT):
        if stratify is None:
            idx = rng.integers(0, n, n)
        else:  # resample within class so AUC is always defined
            idx = np.concatenate([rng.choice(np.flatnonzero(stratify == c), (stratify == c).sum())
                                  for c in np.unique(stratify)])
        draws.append(fn(idx))
    out = {}
    for k in draws[0]:
        vals = np.array([d[k] for d in draws], dtype=float)
        out[k] = [round(float(np.nanpercentile(vals, 2.5)), 4), round(float(np.nanpercentile(vals, 97.5)), 4)]
    return out


def rounded(d):
    return {k: round(float(v), 4) for k, v in d.items()}


def main():
    manifest, X = load_features()
    split = pd.read_csv(DERIVED / "split.csv")
    assert list(split["image_id"]) == list(manifest["image_id"]), "split.csv does not match manifest"
    clf_bundle = joblib.load(MODELS / "classifier.joblib")
    reg_bundle = joblib.load(MODELS / "regressor.joblib")
    assert clf_bundle["feature_names"] == list(X.columns) == reg_bundle["feature_names"]

    test_mask = (split["split"] == "test").to_numpy()
    te = manifest[test_mask]
    X_te = X[test_mask].to_numpy()
    y, hb = te["anemic"].to_numpy(), te["hb_g_dl"].to_numpy()

    prob = clf_bundle["model"].predict_proba(X_te)[:, 1]
    pred = (prob >= clf_bundle["threshold"]).astype(int)
    hb_pred = reg_bundle["model"].predict(X_te)
    hb_base = reg_bundle["baseline"].predict(X_te)

    rng = np.random.default_rng(SEED)
    cls = cls_metrics(y, prob, pred)
    reg = reg_metrics(hb, hb_pred, hb_base)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # Secondary, reported for context only (not used for any selection):
    # thresholding the regression output at 11 g/dL.
    reg_as_cls = cls_metrics(y, HB_THRESHOLD - hb_pred, (hb_pred < HB_THRESHOLD).astype(int))

    cv = json.loads((OUTPUTS / "cv_results.json").read_text())
    metrics = {
        "status": "Research baseline on a small dataset. NOT clinically validated.",
        "caveats": [
            "No patient identifiers: multiple images from one person may fall in both train and test, "
            "which would make these test metrics optimistic.",
            f"Test set is small (n={len(te)}); see the 95% bootstrap CIs.",
            "Dataset licence and provenance pending verification (docs/DATASET.md).",
        ],
        "splits": {
            "train": split_counts(manifest[~test_mask]),
            "test": split_counts(te),
        },
        "cross_validation_on_train": {
            "scheme": cv["cv_scheme"],
            "classification": {k: cv["classification"][k] for k in
                               ("model", "selected_C", "cv_auc_mean", "cv_auc_std", "threshold", "threshold_rule")},
            "regression": {k: cv["regression"][k] for k in
                           ("model", "selected_alpha", "cv_mae_mean", "cv_mae_std",
                            "baseline_cv_mae_mean", "baseline_cv_mae_std")},
        },
        "test": {
            "classification": {
                "label": f"anemic = Hb < {HB_THRESHOLD} g/dL; positive class = anemic",
                "threshold": clf_bundle["threshold"],
                **rounded(cls),
                "confusion_matrix": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
                "ci95_bootstrap": bootstrap(lambda i: cls_metrics(y[i], prob[i], pred[i]), len(y), rng, stratify=y),
            },
            "regression": {
                "units": "g/dL",
                "baseline": "predict training-set mean Hb "
                            f"({reg_bundle['baseline'].constant_.ravel()[0]:.3f} g/dL)",
                **rounded(reg),
                "ci95_bootstrap": bootstrap(lambda i: reg_metrics(hb[i], hb_pred[i], hb_base[i]), len(hb), rng),
            },
            "secondary_regression_thresholded_at_11": {
                "note": "Context only; not used for model selection.",
                **rounded(reg_as_cls),
            },
        },
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }
    (OUTPUTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    test_preds = pd.DataFrame({
        "image_id": te["image_id"].to_numpy(), "split": "test",
        "hb_true": hb, "anemic_true": y, "prob_anemic": prob, "pred_anemic": pred,
        "hb_pred_ridge": hb_pred, "hb_pred_mean_baseline": hb_base,
    })
    oof = pd.read_csv(OUTPUTS / "train_oof_predictions.csv")
    pd.concat([test_preds, oof], ignore_index=True).to_csv(OUTPUTS / "predictions.csv", index=False)

    t = metrics["test"]
    print(json.dumps({"splits": metrics["splits"], "test_classification": t["classification"],
                      "test_regression": t["regression"],
                      "secondary": t["secondary_regression_thresholded_at_11"]}, indent=2))


if __name__ == "__main__":
    main()
