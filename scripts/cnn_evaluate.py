"""ONE-SHOT held-out test evaluation of the frozen CNN, compared with the colour baseline.

Everything (candidate, hyperparameters, epochs, threshold) was fixed by
scripts/cnn_select.py on the training split. This script refuses to run if
outputs/cnn_metrics.json already exists, so the test set is not consulted
repeatedly. Delete that file only to reproduce the identical evaluation;
never use it to retune.

Pre-declared adoption rule (written before the test set was scored):
  The CNN replaces the colour baseline as the primary model only if
  (a) it beat the baseline in training CV (repeat-1 mean AUC), AND
  (b) the paired-bootstrap 95% CI of test ΔAUC (CNN − baseline) lies above 0.
  Otherwise the simpler colour baseline is retained.

Outputs: outputs/cnn_metrics.json, outputs/cnn_predictions.csv (gitignored)
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_common as C  # noqa: E402
from evaluate import N_BOOT, bootstrap, cls_metrics, rounded  # noqa: E402

BASELINE_PREDICTIONS = C.OUTPUTS / "baseline_colour" / "predictions.csv"


def confusion(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def main():
    out = C.OUTPUTS / "cnn_metrics.json"
    if out.exists():
        sys.exit(f"{out.relative_to(C.ROOT)} exists: the one-shot test evaluation has already been run.")

    manifest = C.load_split()  # verifies split.lock
    te = manifest[manifest["split"] == "test"].reset_index(drop=True)
    y = te["anemic"].to_numpy()
    cv = json.loads((C.OUTPUTS / "cnn_cv_results.json").read_text())
    meta = json.loads((C.MODELS / "cnn_final_meta.json").read_text())
    assert meta["chosen"] == cv["chosen"]
    threshold = meta["threshold"]
    chosen = cv["candidates"][cv["chosen"]]

    images = C.load_images(te["path"])
    if chosen["type"] == "frozen":
        bundle = joblib.load(C.MODELS / "cnn_final.joblib")
        prob = bundle["pipeline"].predict_proba(C.embed(bundle["backbone"], images))[:, 1]
    else:
        from cnn_select import predict
        ckpt = torch.load(C.MODELS / "cnn_final.pt", weights_only=False)
        net = C.FineTuneNet(ckpt["backbone"], ckpt["config"]["n_trainable_blocks"], ckpt["config"]["dropout"])
        net.load_state_dict(ckpt["state_dict"])
        prob = predict(net.to(C.device()), images)
    pred = (prob >= threshold).astype(int)

    # Colour baseline predictions on the same test images (frozen snapshot).
    base = pd.read_csv(BASELINE_PREDICTIONS)
    base = base[base["split"] == "test"].set_index("image_id").loc[te["image_id"]]
    assert (base["anemic_true"].to_numpy() == y).all()
    bprob, bpred = base["prob_anemic"].to_numpy(), base["pred_anemic"].to_numpy()

    rng = np.random.default_rng(C.SEED)
    cnn_m, base_m = cls_metrics(y, prob, pred), cls_metrics(y, bprob, bpred)

    # Paired bootstrap: identical resampled indices for both models.
    def delta(i):
        a, b = cls_metrics(y[i], prob[i], pred[i]), cls_metrics(y[i], bprob[i], bpred[i])
        return {k: a[k] - b[k] for k in a}
    paired = bootstrap(delta, len(y), np.random.default_rng(C.SEED + 1), stratify=y)

    beat_in_cv = chosen["auc_mean_rep1"] > cv["baseline_rep1_auc"]
    delta_auc_ci = paired["roc_auc"]
    adopt = bool(beat_in_cv and delta_auc_ci[0] > 0)

    metrics = {
        "status": "Research experiment on a small dataset. NOT clinically validated.",
        "one_shot": "Test set scored once with the configuration frozen by cnn_select.py.",
        "caveats": [
            "No patient identifiers: images from the same person may appear in both train and test, "
            "which would make test metrics optimistic for both models.",
            f"Small test set (n={len(y)}); see the bootstrap 95% CIs.",
            "Dataset licence and provenance pending verification (docs/DATASET.md).",
        ],
        "test_counts": {"n": int(len(y)), "n_anemic": int(y.sum()), "n_non_anemic": int((y == 0).sum())},
        "chosen_cnn": cv["chosen"],
        "cnn": {"threshold": threshold, **rounded(cnn_m), "confusion_matrix": confusion(y, pred),
                "ci95_bootstrap": bootstrap(lambda i: cls_metrics(y[i], prob[i], pred[i]), len(y), rng, stratify=y)},
        "baseline_colour": {**rounded(base_m), "confusion_matrix": confusion(y, bpred)},
        "paired_difference_cnn_minus_baseline": {
            **rounded({k: cnn_m[k] - base_m[k] for k in cnn_m}), "ci95_paired_bootstrap": paired},
        "adoption_rule": "CNN adopted only if it beat the baseline in training CV AND the paired test "
                         "ΔAUC 95% CI lower bound > 0",
        "cnn_beat_baseline_in_cv": bool(beat_in_cv),
        "cnn_adopted": adopt,
        "primary_model": cv["chosen"] if adopt else "baseline_colour (logistic regression on colour features)",
        "n_bootstrap": N_BOOT,
    }
    out.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"image_id": te["image_id"], "split": "test", "hb_true": te["hb_g_dl"], "anemic_true": y,
                  "prob_anemic_cnn": prob, "pred_anemic_cnn": pred,
                  "prob_anemic_baseline": bprob, "pred_anemic_baseline": bpred}).to_csv(
        C.OUTPUTS / "cnn_predictions.csv", index=False)
    print(json.dumps({k: metrics[k] for k in ("chosen_cnn", "cnn", "baseline_colour",
                                              "paired_difference_cnn_minus_baseline", "cnn_adopted",
                                              "primary_model")}, indent=2))


if __name__ == "__main__":
    main()
