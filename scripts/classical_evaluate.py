"""ONE-SHOT held-out test evaluation of the classical-ML candidate vs the LR baseline.

The candidate, its hyperparameters and threshold were all fixed by
scripts/classical_select.py on the training split. This script refuses to run
if outputs/classical_metrics.json already exists, so the test set is not
consulted repeatedly. Delete that file only to reproduce the identical
evaluation; never use it to retune.

Adoption rule (declared in classical_select.py before results were seen):
  The candidate replaces the logistic-regression baseline only if it beat LR
  in nested CV AND the paired-bootstrap 95% CI of test ΔAUC (candidate − LR)
  lies above 0.

Outputs: outputs/classical_metrics.json, outputs/classical_predictions.csv (gitignored)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import N_BOOT, bootstrap, cls_metrics, rounded  # noqa: E402
from train_baseline import DERIVED, MODELS, OUTPUTS, SEED, verify_split_lock  # noqa: E402

BASELINE_PREDICTIONS = OUTPUTS / "baseline_colour" / "predictions.csv"


def confusion(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def main():
    out = OUTPUTS / "classical_metrics.json"
    if out.exists():
        sys.exit(f"{out.name} exists: the one-shot test evaluation has already been run.")
    verify_split_lock()
    manifest = pd.read_csv(DERIVED / "manifest.csv")
    split = pd.read_csv(DERIVED / "split.csv")
    assert list(split["image_id"]) == list(manifest["image_id"])
    te = manifest[split["split"] == "test"].reset_index(drop=True)
    y = te["anemic"].to_numpy()

    cv = json.loads((OUTPUTS / "classical_cv_results.json").read_text())
    bundle = joblib.load(MODELS / "classical_final.joblib")
    assert bundle["name"] == cv["candidate"]
    X = pd.read_csv(DERIVED / "features.csv").set_index("image_id").loc[te["image_id"], bundle["feature_names"]]
    est = bundle["estimator"]
    score = est.predict_proba(X.to_numpy())[:, 1] if bundle["score"] == "predict_proba" \
        else est.decision_function(X.to_numpy())
    pred = (score >= bundle["threshold"]).astype(int)

    base = pd.read_csv(BASELINE_PREDICTIONS)
    base = base[base["split"] == "test"].set_index("image_id").loc[te["image_id"]]
    assert (base["anemic_true"].to_numpy() == y).all()
    bprob, bpred = base["prob_anemic"].to_numpy(), base["pred_anemic"].to_numpy()

    cand_m, base_m = cls_metrics(y, score, pred), cls_metrics(y, bprob, bpred)

    def delta(i):
        a, b = cls_metrics(y[i], score[i], pred[i]), cls_metrics(y[i], bprob[i], bpred[i])
        return {k: a[k] - b[k] for k in a}
    paired = bootstrap(delta, len(y), np.random.default_rng(SEED + 1), stratify=y)
    adopt = bool(cv["candidate_beat_lr_in_cv"] and paired["roc_auc"][0] > 0)

    metrics = {
        "status": "Research experiment on a small dataset. NOT clinically validated.",
        "one_shot": "Test set scored once with the configuration frozen by classical_select.py.",
        "caveats": [
            "No patient identifiers: images from the same person may appear in both train and test.",
            f"Small test set (n={len(y)}); see the bootstrap 95% CIs.",
            "Dataset licence and provenance pending verification (docs/DATASET.md).",
        ],
        "test_counts": {"n": int(len(y)), "n_anemic": int(y.sum()), "n_non_anemic": int((y == 0).sum())},
        "candidate": bundle["name"],
        "candidate_params": cv["candidate_final"]["params"],
        "candidate_metrics": {"threshold": bundle["threshold"], "score_type": bundle["score"], **rounded(cand_m),
                              "confusion_matrix": confusion(y, pred),
                              "ci95_bootstrap": bootstrap(lambda i: cls_metrics(y[i], score[i], pred[i]), len(y),
                                                          np.random.default_rng(SEED), stratify=y)},
        "baseline_lr": {**rounded(base_m), "confusion_matrix": confusion(y, bpred)},
        "paired_difference_candidate_minus_lr": {**rounded({k: cand_m[k] - base_m[k] for k in cand_m}),
                                                 "ci95_paired_bootstrap": paired},
        "adoption_rule": cv["adoption_rule"],
        "candidate_beat_lr_in_cv": cv["candidate_beat_lr_in_cv"],
        "candidate_adopted": adopt,
        "primary_model": bundle["name"] if adopt else "baseline_colour (logistic regression on colour features)",
        "n_bootstrap": N_BOOT,
    }
    out.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"image_id": te["image_id"], "split": "test", "anemic_true": y,
                  f"score_{bundle['name']}": score, f"pred_{bundle['name']}": pred,
                  "prob_anemic_baseline": bprob, "pred_anemic_baseline": bpred}).to_csv(
        OUTPUTS / "classical_predictions.csv", index=False)
    print(json.dumps({k: metrics[k] for k in ("candidate", "candidate_params", "candidate_metrics", "baseline_lr",
                                              "paired_difference_candidate_minus_lr", "candidate_adopted",
                                              "primary_model")}, indent=2))


if __name__ == "__main__":
    main()
