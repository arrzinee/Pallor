"""XGBoost and CatBoost: nested-CV comparison on the TRAINING split only.

This is the final model-search experiment. These candidates are NOT evaluated
on the 51-image held-out test set: it has already been used for three one-shot
evaluations, and no new test data are available. No final model is fitted.
The result is cross-validation evidence only.

Protocol, identical to classical_select.py:
  features: the same 35 colour features
  outer:    RepeatedStratifiedKFold(5, 5, seed 42) over train_baseline.py's
            training order, i.e. the same 25 folds as every earlier comparison
  inner:    StratifiedKFold(5, shuffle, seed 42) grid search, ROC-AUC
Fold identity is verified by re-running the logistic-regression reference and
asserting that its per-fold AUCs equal those saved in classical_cv_results.json.

Limited grids (8 combinations each). No class weighting; AUC is
threshold-free, and the other models' balanced weights mainly affect the threshold.
Diagnostic: the same nested CV without `n_tissue_px`.

Output: outputs/boosting_cv_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classical_select import model_specs, nested_cv  # noqa: E402
from train_baseline import DERIVED, OUTPUTS, SEED, TEST_SIZE, hb_strata, verify_split_lock  # noqa: E402

SPECS = {
    "xgboost": (
        XGBClassifier(subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, eval_metric="logloss",
                      random_state=SEED, n_jobs=1),
        {"n_estimators": [100, 300], "max_depth": [2, 3], "learning_rate": [0.03, 0.1]}),
    "catboost": (
        CatBoostClassifier(l2_leaf_reg=3.0, random_seed=SEED, thread_count=1, verbose=0,
                           allow_writing_files=False),
        {"iterations": [200, 500], "depth": [3, 5], "learning_rate": [0.03, 0.1]}),
}


def main():
    t0 = time.time()
    verify_split_lock()
    out_path = OUTPUTS / "boosting_cv_results.json"
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists; refusing to overwrite prior results.")
    manifest = pd.read_csv(DERIVED / "manifest.csv")
    split = pd.read_csv(DERIVED / "split.csv")
    assert list(split["image_id"]) == list(manifest["image_id"])
    idx_train, _ = train_test_split(np.arange(len(manifest)), test_size=TEST_SIZE,
                                    stratify=hb_strata(manifest["hb_g_dl"].to_numpy()), random_state=SEED)
    assert set(manifest["image_id"].iloc[idx_train]) == set(split.loc[split["split"] == "train", "image_id"])
    tr = manifest.iloc[idx_train].reset_index(drop=True)
    y = tr["anemic"].to_numpy()
    feats = pd.read_csv(DERIVED / "features.csv").set_index("image_id").loc[tr["image_id"]]
    X, X_nosize = feats.to_numpy(), feats.drop(columns="n_tissue_px").to_numpy()
    outer = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y))
    prior = json.loads((OUTPUTS / "classical_cv_results.json").read_text())["models"]

    # Fold-identity check: the LR reference must reproduce its saved per-fold AUCs exactly.
    lr_est, lr_grid = model_specs()[0]["logistic_regression"]
    lr_auc, _ = nested_cv(lr_est, lr_grid, X, y, outer)
    saved = np.array(prior["logistic_regression"]["per_fold_auc"])
    assert np.allclose(np.round(lr_auc, 4), saved), "outer folds differ from classical_select.py"
    print(f"fold check OK: LR reproduces its 25 saved per-fold AUCs (mean {lr_auc.mean():.3f}); "
          f"training rows {len(y)}, test rows not loaded")

    results = {}
    for name, (est, grid) in SPECS.items():
        t = time.time()
        auc, chosen = nested_cv(est, grid, X, y, outer)
        auc_ns, _ = nested_cv(est, grid, X_nosize, y, outer)
        paired = {}
        for ref in ("logistic_regression", "extra_trees"):
            d = auc - np.array(prior[ref]["per_fold_auc"])
            paired[ref] = {"mean_delta_auc": float(d.mean()), "sd_delta_auc": float(d.std()),
                           "folds_better": int((d > 0).sum()), "n_folds": len(d)}
        results[name] = {
            "nested_cv_auc_mean": float(auc.mean()), "nested_cv_auc_std": float(auc.std()),
            "per_fold_auc": [round(float(v), 4) for v in auc],
            "grid": {k: [float(v) for v in vals] for k, vals in grid.items()},
            "inner_selected_params_per_outer_fold": chosen,
            "diagnostic_without_n_tissue_px": {"nested_cv_auc_mean": float(auc_ns.mean()),
                                               "nested_cv_auc_std": float(auc_ns.std())},
            "paired_vs": paired,
            "runtime_s": round(time.time() - t, 1),
        }
        print(f"[{name:9s}] nested-CV AUC {auc.mean():.3f} ± {auc.std():.3f} | without n_tissue_px "
              f"{auc_ns.mean():.3f} | vs LR {paired['logistic_regression']['mean_delta_auc']:+.3f} "
              f"({paired['logistic_regression']['folds_better']}/25) | vs ET "
              f"{paired['extra_trees']['mean_delta_auc']:+.3f} ({paired['extra_trees']['folds_better']}/25) | "
              f"{time.time() - t:.0f}s")

    import catboost
    import xgboost
    out = {
        "scope": "training split only (n=200); held-out test set NOT used; no final model fitted",
        "protocol": "same as classical_select.py (35 features, same 25 outer folds, inner 5-fold grid search)",
        "fold_identity_check": "passed (LR per-fold AUCs reproduced exactly)",
        "versions": {"xgboost": xgboost.__version__, "catboost": catboost.__version__},
        "class_weighting": "none (AUC is threshold-free)",
        "models": results,
        "runtime_s": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"saved {out_path.name}; total {out['runtime_s']}s")


if __name__ == "__main__":
    main()
