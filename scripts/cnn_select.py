"""CNN model selection on the TRAINING split only.

Test images are never loaded by this script.

Candidates (declared before any results were seen):
  F-mobilenet_v3_large  frozen ImageNet MobileNetV3-Large features + logistic regression
  F-efficientnet_b0     frozen ImageNet EfficientNet-B0 features + logistic regression
  FT-<best frozen>      partial fine-tuning (last 2 blocks + new head) of the
                        better frozen backbone, with geometric augmentation only

All candidates are compared with the colour-feature baseline on the SAME
cross-validation folds as train_baseline.py:
RepeatedStratifiedKFold(5, 5, seed 42) over the identical training order.
Fine-tuning is too slow for 25 folds, so it uses the first repeat (5 folds).

Selection rule: highest mean ROC-AUC over the 5 repeat-1 folds, which every
candidate shares; ties go to the simpler model. The fine-tuning epoch count
is chosen from {10, 20, 30} on those same folds (mildly optimistic, documented).

Controls (diagnostic only, never used for selection):
  - silhouette-only images through each frozen backbone
  - crop geometry (tissue pixel count, width, height, aspect) -> logistic regression
If these beat chance, crop shape or size is a potential shortcut.

Outputs: outputs/cnn_cv_results.json, models/cnn_final.*, outputs/cnn_train_oof_predictions.csv
"""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_common as C  # noqa: E402
from train_baseline import TEST_SIZE, hb_strata  # noqa: E402

FT_CONFIG = {"n_trainable_blocks": 2, "dropout": 0.3, "lr_backbone": 1e-4, "lr_head": 1e-3,
             "weight_decay": 0.05, "batch_size": 16, "max_epochs": 30, "epoch_choices": [10, 20, 30],
             "augmentation": "hflip p=0.5; affine rot±10°, translate±5%, scale 0.9–1.1; no colour ops",
             "loss": "BCEWithLogits, pos_weight = n_neg/n_pos of training fold",
             "batchnorm": "frozen (eval mode)"}
C_GRID = np.logspace(-5, 1, 13)


def training_order(manifest):
    """Reproduce train_baseline.py's training-row order so CV folds are identical."""
    idx_train, _ = train_test_split(np.arange(len(manifest)), test_size=TEST_SIZE,
                                    stratify=hb_strata(manifest["hb_g_dl"].to_numpy()), random_state=C.SEED)
    assert set(manifest["image_id"].iloc[idx_train]) == set(manifest.loc[manifest["split"] == "train", "image_id"])
    return idx_train


def youden_threshold(y, prob):
    fpr, tpr, thr = roc_curve(y, prob)
    i = int(np.argmax(tpr - fpr))
    return float(thr[i]), float(tpr[i]), float(1 - fpr[i])


def logreg_search(X, y, folds):
    search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=20000)),
                          {"logisticregression__C": C_GRID}, scoring="roc_auc", cv=folds, n_jobs=-1)
    search.fit(X, y)
    b = search.best_index_
    per_fold = np.array([search.cv_results_[f"split{k}_test_score"][b] for k in range(len(folds))])
    return search, per_fold


def summarise(per_fold):
    return {"auc_mean_25": float(per_fold.mean()), "auc_std_25": float(per_fold.std()),
            "auc_mean_rep1": float(per_fold[:5].mean()), "auc_std_rep1": float(per_fold[:5].std()),
            "per_fold_auc": [round(float(v), 4) for v in per_fold]}


def train_finetune(name, X, y, epochs, val=None, seed=C.SEED):
    """Train FineTuneNet; if val=(Xv, yv) is given, return val probabilities after every epoch."""
    C.seed_everything(seed)
    dev = C.device()
    net = C.FineTuneNet(name, FT_CONFIG["n_trainable_blocks"], FT_CONFIG["dropout"]).to(dev)
    backbone_params = [p for p in net.features.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": FT_CONFIG["lr_backbone"]},
                             {"params": net.head.parameters(), "lr": FT_CONFIG["lr_head"]}],
                            weight_decay=FT_CONFIG["weight_decay"])
    pos_weight = torch.tensor((y == 0).sum() / max((y == 1).sum(), 1), dtype=torch.float32, device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    yt = torch.tensor(y, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed)
    val_probs = []
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), FT_CONFIG["batch_size"]):
            idx = perm[i:i + FT_CONFIG["batch_size"]]
            xb = torch.stack([C.TRAIN_AUGMENT(X[j]) for j in idx]).to(dev)  # per-sample random augmentation
            opt.zero_grad()
            loss_fn(net(xb), yt[idx].to(dev)).backward()
            opt.step()
        if val is not None:
            val_probs.append(predict(net, val[0]))
    return net, val_probs


@torch.no_grad()
def predict(net, X, batch=64):
    net.eval()
    dev = next(net.parameters()).device
    return torch.cat([torch.sigmoid(net(X[i:i + batch].to(dev))).cpu() for i in range(0, len(X), batch)]).numpy()


def main():
    t0 = time.time()
    C.seed_everything()
    manifest = C.load_split()
    idx_train = training_order(manifest)
    tr = manifest.iloc[idx_train].reset_index(drop=True)
    assert (tr["split"] == "train").all()
    y = tr["anemic"].to_numpy()
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=C.SEED).split(np.zeros(len(y)), y))
    rep1 = folds[:5]
    print(f"training images: {len(tr)} (anemic {y.sum()}, non-anemic {(y == 0).sum()}); device: {C.device()}")

    results = {"selection_rule": "max mean ROC-AUC over repeat-1 folds (shared by all candidates); ties -> simpler",
               "cv_scheme": "RepeatedStratifiedKFold(5,5,seed=42) on the training order of train_baseline.py",
               "n_train": int(len(tr)), "candidates": {}, "controls": {}}

    # --- Colour baseline on the same folds (fixed C from its own CV) ---------
    base_cv = json.loads((C.OUTPUTS / "cv_results.json").read_text())
    feats = pd.read_csv(C.DERIVED / "features.csv").set_index("image_id").loc[tr["image_id"]].to_numpy()
    base_pipe = make_pipeline(StandardScaler(), LogisticRegression(
        C=base_cv["classification"]["selected_C"], class_weight="balanced", max_iter=10000))
    base_auc = []
    for trn, val in folds:
        base_pipe.fit(feats[trn], y[trn])
        base_auc.append(roc_auc_score(y[val], base_pipe.predict_proba(feats[val])[:, 1]))
    results["baseline_colour"] = summarise(np.array(base_auc))
    print(f"[baseline colour] 25-fold AUC {np.mean(base_auc):.3f} ± {np.std(base_auc):.3f} | rep1 {np.mean(base_auc[:5]):.3f}")

    # --- Frozen-feature candidates ------------------------------------------
    images = C.load_images(tr["path"])
    masks = C.load_images(tr["path"], mask_only=True)
    embeddings = {}
    for name in C.BACKBONES:
        E = C.embed(name, images)
        embeddings[name] = E
        search, per_fold = logreg_search(E, y, folds)
        results["candidates"][f"F-{name}"] = {"type": "frozen", "backbone": name, "embedding_dim": int(E.shape[1]),
                                              "selected_C": float(search.best_params_["logisticregression__C"]),
                                              **summarise(per_fold)}
        print(f"[F-{name}] C={search.best_params_['logisticregression__C']:.3g} "
              f"25-fold AUC {per_fold.mean():.3f} ± {per_fold.std():.3f} | rep1 {per_fold[:5].mean():.3f}")
        # Control: silhouette only
        _, ctrl = logreg_search(C.embed(name, masks), y, folds)
        results["controls"][f"silhouette-{name}"] = summarise(ctrl)
        print(f"  [control: silhouette only] 25-fold AUC {ctrl.mean():.3f} ± {ctrl.std():.3f}")
    geom = np.column_stack([tr["n_opaque_px"], tr["width"], tr["height"], tr["width"] / tr["height"]]).astype(float)
    _, gctrl = logreg_search(np.log(geom), y, folds)
    results["controls"]["crop-geometry"] = summarise(gctrl)
    print(f"[control: crop geometry] 25-fold AUC {gctrl.mean():.3f} ± {gctrl.std():.3f}")

    # --- Fine-tuning the better frozen backbone ------------------------------
    ft_backbone = max(C.BACKBONES, key=lambda n: results["candidates"][f"F-{n}"]["auc_mean_25"])
    print(f"[FT-{ft_backbone}] 5-fold CV, {FT_CONFIG['max_epochs']} epochs per fold ...")
    curves = np.zeros((5, FT_CONFIG["max_epochs"]))
    oof_by_epoch = np.zeros((FT_CONFIG["max_epochs"], len(y)))
    for k, (trn, val) in enumerate(rep1):
        _, vp = train_finetune(ft_backbone, images[trn], y[trn], FT_CONFIG["max_epochs"], val=(images[val], y[val]))
        for e, p in enumerate(vp):
            curves[k, e] = roc_auc_score(y[val], p)
            oof_by_epoch[e, val] = p
        print(f"  fold {k + 1}: val AUC @10/20/30 = {curves[k, 9]:.3f} / {curves[k, 19]:.3f} / {curves[k, 29]:.3f} "
              f"({time.time() - t0:.0f}s)")
    choice_means = {e: float(curves[:, e - 1].mean()) for e in FT_CONFIG["epoch_choices"]}
    best_epochs = max(choice_means, key=choice_means.get)
    ft_fold_auc = curves[:, best_epochs - 1]
    results["candidates"][f"FT-{ft_backbone}"] = {
        "type": "finetune", "backbone": ft_backbone, "config": FT_CONFIG, "selected_epochs": best_epochs,
        "auc_by_epoch_choice": choice_means, "auc_mean_rep1": float(ft_fold_auc.mean()),
        "auc_std_rep1": float(ft_fold_auc.std()), "per_fold_auc": [round(float(v), 4) for v in ft_fold_auc],
        "mean_val_auc_curve": [round(float(v), 4) for v in curves.mean(0)]}
    print(f"[FT-{ft_backbone}] epochs={best_epochs} rep1 AUC {ft_fold_auc.mean():.3f} ± {ft_fold_auc.std():.3f} "
          f"(by epoch: {choice_means})")

    # --- Apply the pre-declared selection rule -------------------------------
    complexity = {"frozen": 0, "finetune": 1}
    ranked = sorted(results["candidates"].items(),
                    key=lambda kv: (-round(kv[1]["auc_mean_rep1"], 4), complexity[kv[1]["type"]]))
    chosen_name, chosen = ranked[0]
    results["chosen"] = chosen_name
    results["baseline_rep1_auc"] = results["baseline_colour"]["auc_mean_rep1"]
    print(f"CHOSEN: {chosen_name} (rep1 AUC {chosen['auc_mean_rep1']:.3f} vs colour baseline "
          f"{results['baseline_rep1_auc']:.3f})")

    # --- Threshold from OOF training predictions, then final fit -------------
    C.MODELS.mkdir(exist_ok=True)
    if chosen["type"] == "frozen":
        E = embeddings[chosen["backbone"]]
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=chosen["selected_C"], class_weight="balanced",
                                                                  max_iter=20000))
        oof = cross_val_predict(pipe, E, y, cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED),
                                method="predict_proba")[:, 1]
        pipe.fit(E, y)
        joblib.dump({"type": "frozen", "backbone": chosen["backbone"], "pipeline": pipe}, C.MODELS / "cnn_final.joblib")
    else:
        oof = oof_by_epoch[best_epochs - 1]
        net, _ = train_finetune(chosen["backbone"], images, y, best_epochs)
        torch.save({"type": "finetune", "backbone": chosen["backbone"], "config": FT_CONFIG,
                    "epochs": best_epochs, "state_dict": net.cpu().state_dict()}, C.MODELS / "cnn_final.pt")
    thr, sens, spec = youden_threshold(y, oof)
    results["threshold"] = thr
    results["threshold_rule"] = "max Youden J on out-of-fold training probabilities"
    results["train_oof_at_threshold"] = {"sensitivity": sens, "specificity": spec, "auc": float(roc_auc_score(y, oof))}
    (C.MODELS / "cnn_final_meta.json").write_text(json.dumps({"chosen": chosen_name, "threshold": thr}, indent=2))
    pd.DataFrame({"image_id": tr["image_id"], "split": "train_oof", "anemic_true": y, "prob_anemic": oof,
                  "pred_anemic": (oof >= thr).astype(int)}).to_csv(C.OUTPUTS / "cnn_train_oof_predictions.csv",
                                                                   index=False)
    results["runtime_s"] = round(time.time() - t0, 1)
    results["device"] = str(C.device())
    results["torch_version"] = torch.__version__
    (C.OUTPUTS / "cnn_cv_results.json").write_text(json.dumps(results, indent=2))
    print(f"threshold={thr:.3f} (train OOF sens {sens:.3f}, spec {spec:.3f}); saved; {results['runtime_s']}s")


if __name__ == "__main__":
    main()
