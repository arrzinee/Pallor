"""Training-split-only diagnostic: does the colour baseline rely on crop size?

cnn_select.py showed that crop geometry alone predicts the label in training
CV, a potential acquisition/cropping shortcut. The colour baseline includes
`n_tissue_px`, so this script re-runs the baseline's exact CV procedure
(same 25 folds, same C grid) with and without the size feature.

It never touches the test split and does not change the frozen baseline.
Output: outputs/shortcut_diagnostic.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnn_common as C  # noqa: E402
from cnn_select import training_order  # noqa: E402


def cv_auc(X, y, folds):
    s = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=10000)),
                     {"logisticregression__C": np.logspace(-3, 5, 17)}, scoring="roc_auc", cv=folds, n_jobs=-1)
    s.fit(X, y)
    return {"auc_mean": round(float(s.cv_results_["mean_test_score"][s.best_index_]), 4),
            "auc_std": round(float(s.cv_results_["std_test_score"][s.best_index_]), 4),
            "selected_C": float(s.best_params_["logisticregression__C"])}


def main():
    manifest = C.load_split()
    tr = manifest.iloc[training_order(manifest)].reset_index(drop=True)
    y = tr["anemic"].to_numpy()
    feats = pd.read_csv(C.DERIVED / "features.csv").set_index("image_id").loc[tr["image_id"]]
    folds = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=C.SEED).split(np.zeros(len(y)), y))
    variants = {
        "all_35_features (frozen baseline)": feats,
        "colour_only (drop n_tissue_px)": feats.drop(columns="n_tissue_px"),
        "n_tissue_px_only": np.log(feats[["n_tissue_px"]]),
    }
    out = {"scope": "training split only (n=200), RepeatedStratifiedKFold(5,5,seed=42)",
           "variants": {k: cv_auc(np.asarray(v, dtype=float), y, folds) for k, v in variants.items()}}
    # Direction of the size association, for interpretation.
    out["median_tissue_px"] = {"anemic": float(tr.loc[y == 1, "n_opaque_px"].median()),
                               "non_anemic": float(tr.loc[y == 0, "n_opaque_px"].median())}
    (C.OUTPUTS / "shortcut_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
