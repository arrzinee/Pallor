"""Pallor research app (Gradio). Logistic-regression baseline only.

Run:  .venv/bin/python app/app.py      -> http://127.0.0.1:7860

Local by default: bound to 127.0.0.1. On Render (RENDER=true) it binds 0.0.0.0 on $PORT so the
platform can route traffic (see launch_settings). No share link in any environment, Gradio
analytics off, system fonts only (no web-font requests). No dataset images, Hb values or
per-image predictions are displayed: the dataset's licence and sharing
permission are unresolved. Every metric shown is read at runtime from the
tracked aggregate result files in outputs/. This module only changes
presentation; inference.py and the saved model are untouched.
"""

import html
import json
import os
import sys
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Disable telemetry before Gradio is imported.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inference  # noqa: E402

OUTPUTS = inference.ROOT / "outputs"
RESULT_FILES = {
    "baseline": "metrics.json", "baseline_cv": "cv_results.json",
    "classical_test": "classical_metrics.json", "classical_cv": "classical_cv_results.json",
    "boosting_cv": "boosting_cv_results.json", "cnn_test": "cnn_metrics.json",
    "cnn_cv": "cnn_cv_results.json", "shortcut": "shortcut_diagnostic.json",
}

DISCLAIMER = (
    "Pallor is an experimental research prototype, not a medical device. Its results are not a diagnosis "
    "or clinically validated anaemia screening. The model may misclassify images and has not been "
    "validated on an independent clinical population."
)
NEXT_STEPS = (
    "This result cannot establish whether someone has anaemia. If anaemia is a concern, consult a "
    "qualified healthcare professional. A blood test is needed to measure haemoglobin."
)
SCORE_NOTE = "This score is produced by the research model. It is not a calibrated probability of anaemia."

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression", "svm_rbf": "RBF-SVM", "random_forest": "Random Forest",
    "extra_trees": "Extra Trees", "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost", "catboost": "CatBoost",
}

FEATURE_GROUPS = [  # (group, description), in display order
    ("RGB", "red, green and blue intensity statistics and chromaticity"),
    ("HSV", "hue, saturation and brightness"),
    ("CIELAB", "perceptual lightness plus red–green and yellow–blue axes"),
    ("Redness indices", "engineered ratios such as an erythema index"),
    ("Tissue size", "the number of visible tissue pixels"),
]


def feature_group(name):
    if name == "n_tissue_px":
        return "Tissue size"
    if name.startswith(("R_", "G_", "B_", "chrom_")):
        return "RGB"
    if name.startswith(("hue_", "S_", "V_")):
        return "HSV"
    if "_lab_" in name or name == "a_over_L":
        return "CIELAB"
    return "Redness indices"


def r3(x):
    """Round half up to 3 decimals, the convention used in docs/REPORT.md and models/README.md."""
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def pct(x):
    """Proportion -> percentage with one decimal, rounded half up."""
    return str((Decimal(str(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) + "%"


def ci(pair):
    return f"{r3(pair[0])} – {r3(pair[1])}"


def esc(x):
    return html.escape(str(x))


def load_results(outputs_dir=OUTPUTS):
    """Read the aggregate result files and build dashboard data. Nothing is hardcoded."""
    j = {k: json.loads((Path(outputs_dir) / f).read_text()) for k, f in RESULT_FILES.items()}

    def test_row(model, role, m, source):
        c, b = m["confusion_matrix"], m.get("ci95_bootstrap", {})
        return {"Model": model, "Role": role, "ROC-AUC": r3(m["roc_auc"]),
                "AUC 95% CI": ci(b["roc_auc"]) if "roc_auc" in b else "—",
                "Sensitivity": r3(m["sensitivity"]), "Specificity": r3(m["specificity"]),
                "Precision": r3(m["precision"]), "F1": r3(m["f1"]),
                "TP / FN / FP / TN": f"{c['tp']} / {c['fn']} / {c['fp']} / {c['tn']}", "Source": source}

    base = j["baseline"]["test"]["classification"]
    test_models = [
        ("Logistic Regression", "Primary — used by this app", base, RESULT_FILES["baseline"]),
        ("Extra Trees", "Experimental candidate — not used", j["classical_test"]["candidate_metrics"],
         RESULT_FILES["classical_test"]),
        ("Frozen MobileNetV3 + LR", "Not adopted — not used", j["cnn_test"]["cnn"], RESULT_FILES["cnn_test"]),
    ]
    test = pd.DataFrame([test_row(*t) for t in test_models])
    et_pair = j["classical_test"]["paired_difference_candidate_minus_lr"]
    cnn_pair = j["cnn_test"]["paired_difference_cnn_minus_baseline"]
    paired = pd.DataFrame([
        {"Comparison": "Extra Trees − Logistic Regression", "ΔAUC": r3(et_pair["roc_auc"]),
         "95% CI (paired bootstrap)": ci(et_pair["ci95_paired_bootstrap"]["roc_auc"])},
        {"Comparison": "Frozen MobileNetV3 − Logistic Regression", "ΔAUC": r3(cnn_pair["roc_auc"]),
         "95% CI (paired bootstrap)": ci(cnn_pair["ci95_paired_bootstrap"]["roc_auc"])},
    ])

    nested_models = {**j["classical_cv"]["models"], **j["boosting_cv"]["models"]}
    rows = []
    for key, m in nested_models.items():
        pv = m.get("paired_vs_lr") or m.get("paired_vs", {}).get("logistic_regression", {})
        better = pv.get("folds_better_than_lr", pv.get("folds_better"))
        rows.append({"Model": MODEL_NAMES.get(key, key), "_mean": m["nested_cv_auc_mean"],
                     "Nested-CV AUC": r3(m["nested_cv_auc_mean"]), "SD": r3(m["nested_cv_auc_std"]),
                     "Without size feature": r3(m["diagnostic_without_n_tissue_px"]["nested_cv_auc_mean"]),
                     "Folds better than LR": "—" if key == "logistic_regression" else f"{better} / {len(m['per_fold_auc'])}"})
    nested_sorted = sorted(rows, key=lambda r: -r["_mean"])
    nested = pd.DataFrame(nested_sorted).drop(columns="_mean").reset_index(drop=True)

    bcv, ccv = j["baseline_cv"]["classification"], j["cnn_cv"]["candidates"]
    ft = ccv["FT-mobilenet_v3_large"]
    other_cv = pd.DataFrame([
        {"Model": "Logistic Regression (original baseline)", "Protocol": "non-nested, 25 folds",
         "CV AUC": r3(bcv["cv_auc_mean"]), "SD": r3(bcv["cv_auc_std"])},
        {"Model": "Frozen MobileNetV3-Large + LR", "Protocol": "non-nested, 25 folds",
         "CV AUC": r3(ccv["F-mobilenet_v3_large"]["auc_mean_25"]), "SD": r3(ccv["F-mobilenet_v3_large"]["auc_std_25"])},
        {"Model": "Frozen EfficientNet-B0 + LR", "Protocol": "non-nested, 25 folds",
         "CV AUC": r3(ccv["F-efficientnet_b0"]["auc_mean_25"]), "SD": r3(ccv["F-efficientnet_b0"]["auc_std_25"])},
        {"Model": "Fine-tuned MobileNetV3-Large", "Protocol": "5 folds; epochs chosen on the same folds",
         "CV AUC": r3(ft["auc_mean_rep1"]), "SD": r3(ft["auc_std_rep1"])},
    ])

    reg, rcv = j["baseline"]["test"]["regression"], j["baseline_cv"]["regression"]
    regression = pd.DataFrame([{
        "Model": "Ridge (Hb in g/dL)", "CV MAE (training)": f"{r3(rcv['cv_mae_mean'])} ± {r3(rcv['cv_mae_std'])}",
        "Test MAE": r3(reg["mae"]), "Test MAE 95% CI": ci(reg["ci95_bootstrap"]["mae"]),
        "Mean-predictor test MAE": r3(reg["baseline_mae"]), "Test R²": r3(reg["r2"])}])

    sc = j["shortcut"]["variants"]
    shortcut = pd.DataFrame([{"Input (training CV, LR, non-nested)": k, "AUC": r3(v["auc_mean"]), "SD": r3(v["auc_std"])}
                             for k, v in sc.items()])

    raw = {  # unformatted values for the charts (same sources as the tables)
        "test_models": [{"model": name, "role": role, "auc": m["roc_auc"],
                         "ci": m.get("ci95_bootstrap", {}).get("roc_auc")} for name, role, m, _ in test_models],
        "et_pair": {"delta": et_pair["roc_auc"], "ci": et_pair["ci95_paired_bootstrap"]["roc_auc"]},
        "nested": [{"model": r["Model"], "mean": r["_mean"], "sd": float(r["SD"])} for r in nested_sorted],
        "n_outer_folds": len(nested_models["logistic_regression"]["per_fold_auc"]),
    }
    counts = j["baseline"]["splits"]
    return {"test": test, "paired": paired, "nested": nested, "other_cv": other_cv,
            "regression": regression, "shortcut": shortcut, "splits": counts,
            "baseline_test": base, "raw": raw}


# --- Synthetic explorer examples (code-generated; no dataset content) ---------

def make_synthetic_examples(directory):
    """Write deterministic synthetic PNGs that demonstrate each input rule."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def crop(w, h, mode="RGBA"):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).ellipse((2, 2, w - 3, h - 3), fill=255)
        base = np.array([196, 118, 112], dtype=np.float64)
        noise = rng.normal(0, 6, (h, w, 3))
        rgb = np.clip(base + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.dstack([rgb, np.asarray(mask)]), "RGBA")
        return img.convert("RGB") if mode == "RGB" else img

    examples = {
        "Synthetic crop within training size range": ("valid.png", crop(220, 80),
            "Pink ellipse on a transparent background, sized like the training crops. Accepted, though its "
            "artificial texture may be flagged as outside the training colour range. Its score is meaningless "
            "medically: it is not a real eye."),
        "Oversized synthetic crop (out of range)": ("oversized.png", crop(1400, 500),
            "Accepted but flagged: geometry and tissue-pixel count exceed the training range."),
        "No transparency (plain RGB PNG)": ("rgb.png", crop(220, 80, mode="RGB"),
            "Rejected: no alpha channel, like an ordinary photo."),
        "Fully opaque RGBA (no background removed)": ("opaque.png",
            Image.new("RGBA", (220, 80), (196, 118, 112, 255)), "Rejected: has alpha but nothing is transparent."),
        "Too little tissue (64 opaque pixels)": ("tiny.png", None, "Rejected: fewer than 100 tissue pixels."),
        "JPEG renamed to .png": ("fake.png", None, "Rejected: not actually a PNG."),
    }
    out = {}
    for label, (name, img, note) in examples.items():
        path = d / name
        if name == "tiny.png":
            img = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
            img.paste((196, 118, 112, 255), (20, 10, 28, 18))
        if name == "fake.png":
            crop(220, 80, mode="RGB").save(path, format="JPEG")
        else:
            img.save(path, format="PNG")
        out[label] = (str(path), note)
    return out


# --- Result presentation ------------------------------------------------------

ICON_INFO = ('<svg class="p-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8.25" '
             'fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M10 9v5M10 6.2v.1" stroke="currentColor" '
             'stroke-width="1.6" stroke-linecap="round"/></svg>')
ICON_NOTICE = ('<svg class="p-icon" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 3.2 17.4 16H2.6Z" fill="none" '
               'stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M10 8.2v3.6M10 13.9v.1" '
               'stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>')
ICON_CHECK = ('<svg class="p-icon" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8.25" fill="none" '
              'stroke="currentColor" stroke-width="1.5"/><path d="m6.5 10.2 2.3 2.3 4.7-4.9" fill="none" '
              'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>')

# The approved synthetic example loaded by "Try the demo" (one of make_synthetic_examples' outputs).
DEMO_EXAMPLE = "Synthetic crop within training size range"

# Short global notice shown once, quietly, in the page footer (the full limitations live in their own tab).
GLOBAL_NOTE = ("Pallor is an experimental research prototype — not a diagnosis or clinically validated anaemia "
               "screening. It cannot measure haemoglobin.")


def placeholder_card():
    return ('<div class="p-card p-placeholder"><p class="p-state-title">No image analysed yet</p>'
            '<p>Upload a supported PNG on the left, then select <strong>Analyse image</strong>. '
            'Images are analysed by the computer running Pallor; when Pallor is hosted online, '
            'that is a remote server, not your device.</p></div>')


def ready_card(synthetic=False):
    if synthetic:
        return ('<div class="p-card p-placeholder"><p class="p-state-title">Synthetic demo image loaded</p>'
                '<p>Select <strong>Analyse image</strong> to run the real validation and model on this synthetic '
                'test image. The output shows software behaviour, not medical performance.</p></div>')
    return ('<div class="p-card p-placeholder"><p class="p-state-title">Ready to analyse</p>'
            '<p>The image passed the input checks. Select <strong>Analyse image</strong> to run the research model.</p></div>')


SYNTHETIC_BANNER = ('<div class="p-synthetic"><span class="p-tag">Synthetic</span><span>Code-generated test image, not a '
                    'real conjunctiva. This output demonstrates software behaviour, not medical performance.</span></div>')


LOADING_CARD = ('<div class="p-card p-placeholder" role="status" aria-live="polite" aria-busy="true">'
                '<span class="p-spinner" aria-hidden="true"></span><p class="p-state-title">Analysing…</p>'
                '<p>Checking the image, extracting colour features and applying the research model.</p></div>')


TECH_PLACEHOLDER = '<p class="p-muted">Technical details appear here after an analysis.</p>'


def footnote():
    return f'<p class="p-footnote">{ICON_INFO}<span>{esc(DISCLAIMER)}</span></p>'


def influence_section():
    names = inference.default_model()["feature_names"]
    counts = {g: sum(feature_group(n) == g for n in names) for g, _ in FEATURE_GROUPS}
    items = "".join(f'<li title="{esc(desc)}">{esc(g)} <em>{counts[g]}</em></li>'
                    for g, desc in FEATURE_GROUPS if counts[g])
    return ('<div class="p-sub"><h5>What influenced this result?</h5>'
            f'<p>The model evaluates {len(names)} engineered colour features measured from the visible tissue pixels '
            'of the supplied crop, grouped below. Its output reflects patterns learned from one training dataset. '
            'This version does not provide a validated medical explanation for an individual result.</p>'
            f'<ul class="p-groups" aria-label="Feature groups and counts">{items}</ul></div>')


def format_result(result, results, synthetic=False):
    """Result card: A model output, B plain-English explanation, C input warnings (D: technical details panel).
    For synthetic test images the actual model output is shown in neutral wording, without a dataset label."""
    banner = SYNTHETIC_BANNER if synthetic else ""
    if result.status == "rejected":
        reasons = "".join(f"<li>{esc(r)}</li>" for r in result.reasons)
        return (f'<div class="p-card p-result is-rejected" role="status" aria-live="polite">{banner}<div class="p-status">'
                '<span class="p-dot" aria-hidden="true"></span><span class="p-eyebrow">Not analysed</span></div>'
                "<h3>This image can't be analysed</h3>"
                f'<ul class="p-reasons">{reasons}</ul>'
                '<p class="p-muted">Pallor only supports pre-segmented conjunctiva crops saved as PNG with a transparent '
                'background, like its training images. Ordinary photographs, JPEGs, selfies and unsegmented eye images '
                'are not supported.</p>'
                f'{footnote()}</div>')

    above = result.above_threshold
    state = "is-above" if above else "is-below"
    if synthetic:
        eyebrow = "Model output · synthetic test image"
        headline = f"Score {'above' if above else 'below'} the research threshold"
        label_note = ("This is the model's actual output for a synthetic image. It carries no dataset label and no "
                      "medical meaning.")
        explain = ("The real validation checks, feature extraction and model ran on this code-generated image, "
                   "showing the full pipeline working end to end. Because the image is not a real conjunctiva, "
                   "where its score falls says nothing about anaemia.")
    elif above:
        eyebrow = "Model output · predicted dataset label"
        headline = "Resembles the dataset's anaemic-labelled examples"
        label_note = "This is the label the research model predicts for its dataset. It is not a medical diagnosis."
        explain = ("The model found colour-feature patterns that fall on the side of the research threshold "
                   "associated with images labelled anaemic in this dataset.")
    else:
        eyebrow = "Model output · predicted dataset label"
        headline = "Did not cross the research threshold"
        label_note = "This is the label the research model predicts for its dataset. It is not a medical diagnosis."
        explain = ("The model's colour-feature score fell below the threshold used in this experiment. "
                   "This does not establish that the person is free from anaemia.")

    position = min(max(result.probability, 0.0), 1.0) * 100
    marker = min(max(result.threshold, 0.0), 1.0) * 100
    output = ('<section class="p-block p-output">'
              '<div class="p-status"><span class="p-dot" aria-hidden="true"></span>'
              f'<span class="p-eyebrow">{esc(eyebrow)}</span></div>'
              f'<h3>{esc(headline)}</h3>'
              f'<p class="p-label-note">{esc(label_note)}</p>'
              '<div class="p-score"><div class="p-score-head"><span>Model score</span>'
              f'<strong>{r3(result.probability)}</strong></div>'
              f'<div class="p-track" aria-hidden="true"><span class="p-track-threshold" style="left:{marker:.1f}%"></span>'
              f'<span class="p-track-dot" style="left:{position:.1f}%"></span></div>'
              f'<div class="p-track-legend"><span>Lower</span><span class="p-track-label" style="left:{marker:.1f}%">'
              'Research threshold</span><span>Higher</span></div>'
              f'<p class="p-muted">{esc(SCORE_NOTE)}</p></div></section>')

    next_step = ('' if synthetic else
                 f'<div class="p-next"><h5>What should I do with this result?</h5><p>{esc(NEXT_STEPS)}</p></div>')
    meaning = ('<section class="p-block"><h4>What this means</h4>'
               f'<p>{esc(explain)}</p>{influence_section()}{next_step}</section>')

    if result.warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in result.warnings)
        warn_body = (f'<div class="p-notice" role="note">{ICON_NOTICE}<div><strong>Outside the training data — treat '
                     f'this result as unreliable.</strong><ul>{items}</ul></div></div>')
    else:
        warn_body = (f'<p class="p-ok">{ICON_CHECK}<span>No input warnings. The image size, tissue area and colour '
                     'features are within the ranges seen in the training data.</span></p>')
    warnings = f'<section class="p-block"><h4>Input warnings</h4>{warn_body}</section>'

    base, n_test = results["baseline_test"], results["splits"]["test"]["n"]
    cm = base["confusion_matrix"]
    reliability = (f'<p class="p-muted p-reliability">On {n_test} held-out test images, the model placed '
                   f'{pct(base["sensitivity"])} of anaemic-labelled images ({cm["tp"]} of {cm["tp"] + cm["fn"]}) above '
                   f'the threshold and {pct(base["specificity"])} of non-anaemic-labelled images ({cm["tn"]} of '
                   f'{cm["tn"] + cm["fp"]}) below it, so it can be wrong in either direction.</p>')

    return (f'<div class="p-card p-result {state}" role="status" aria-live="polite">{banner}{output}{meaning}{warnings}'
            f'{reliability}{footnote()}</div>')


def fmt_value(v):
    """Readable feature values: counts with thousands separators, others to 4 significant figures."""
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"


def format_technical(result, path):
    """Secondary details for the collapsible panel. No hashes or internal identifiers."""
    if result.status != "ok":
        info = result.info
        rows = []
        if "width" in info:
            rows.append(("Image size", f"{info['width']} × {info['height']} px"))
        if "mode" in info:
            rows.append(("Colour mode", info["mode"]))
        if "file_bytes" in info:
            rows.append(("File size", f"{info['file_bytes'] / 1024:.1f} KB"))
        body = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in rows)
        return f'<div class="p-tech"><table>{body}</table><p class="p-muted">No model score: the input was not analysed.</p></div>'
    info, ref = result.info, inference.default_reference()
    side = "at or above" if result.above_threshold else "below"
    summary = [
        ("Image size", f"{info['width']} × {info['height']} px (training range {ref.width[0]}–{ref.width[1]} × "
                       f"{ref.height[0]}–{ref.height[1]} px)"),
        ("Colour mode", info["mode"]),
        ("Visible tissue pixels", f"{info['n_opaque_px']:,} (training range {ref.n_tissue_px[0]:,}–{ref.n_tissue_px[1]:,})"),
        ("Transparent background", f"{info['transparent_fraction'] * 100:.1f}% of the image"),
        ("Model score", f"{r3(result.probability)} — not a calibrated probability"),
        ("Research threshold", f"{r3(result.threshold)}, fixed, chosen on training data only"),
        ("Outcome", f"Score is {side} the threshold"),
        ("Out-of-range notes", str(len(result.warnings))),
    ]
    table = "".join(f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in summary)
    feats = inference.features.extract(path)  # display only; the score above came from inference.analyse
    groups = []
    for g, _ in FEATURE_GROUPS:
        names = [n for n in inference.default_model()["feature_names"] if feature_group(n) == g]
        if not names:
            continue
        rows = ""
        for n in names:
            lo, hi = ref.feature_ranges[n]
            flag = '' if lo <= feats[n] <= hi else ' class="is-out"'
            rows += (f"<tr{flag}><td><code>{esc(n)}</code></td><td>{fmt_value(feats[n])}</td>"
                     f"<td>{fmt_value(lo)} – {fmt_value(hi)}</td></tr>")
        groups.append(f'<h5>{esc(g)}</h5><table class="p-features"><thead><tr><th>Feature</th><th>Value</th>'
                      f'<th>Training range</th></tr></thead><tbody>{rows}</tbody></table>')
    return (f'<div class="p-tech"><table>{table}</table><h4>Feature values</h4>'
            '<p class="p-muted">Computed from fully opaque pixels only, in the order the model was trained on. '
            'Values outside the training range are highlighted.</p>' + "".join(groups) + "</div>")


def upload_status(path):
    """Instant feedback on upload, using the same validation as the analysis (no model is run here)."""
    if not path:
        return '<p class="p-upload-status">No image selected.</p>', gr.update(value=None, visible=False)
    reasons, info = inference.validate_png(path)
    if reasons:
        return (f'<p class="p-upload-status is-error">{ICON_NOTICE}<span>{esc(reasons[0])}</span></p>',
                gr.update(value=None, visible=False))
    msg = (f"Ready to analyse · {info['width']} × {info['height']} px PNG with a transparent background "
           f"· {info['n_opaque_px']:,} visible tissue pixels")
    return (f'<p class="p-upload-status is-ready"><span class="p-dot"></span><span>{esc(msg)}</span></p>',
            gr.update(value=inference.preview_image(path), visible=True))


def is_synthetic(path, synthetic_hashes):
    """True only for the app's own code-generated examples (matched by SHA-256, never by filename)."""
    return bool(path) and inference.sha256_file(path) in synthetic_hashes


def make_upload_handler(results, synthetic_hashes=frozenset()):
    """Upload feedback: status, preview, Analyse-button state, result area, and demo controls.
    Invalid files show the rejection immediately (validation only); valid files enable the button."""
    def on_upload(path):
        status, preview = upload_status(path)
        demo = is_synthetic(path, synthetic_hashes)
        demo_ui = (gr.update(visible=demo), gr.update(visible=not demo))  # (demo note + exit, try-demo offer)
        if not path:
            return (status, preview, gr.update(interactive=False), placeholder_card(), TECH_PLACEHOLDER, {}, *demo_ui)
        reasons, _ = inference.validate_png(path)
        if reasons:
            rejected = inference.analyse(path)  # returns the rejection before any feature extraction
            return (status, preview, gr.update(interactive=False), format_result(rejected, results, synthetic=demo),
                    format_technical(rejected, path), rejected.as_dict(), *demo_ui)
        return (status, preview, gr.update(interactive=True), ready_card(synthetic=demo), TECH_PLACEHOLDER, {},
                *demo_ui)
    return on_upload


def make_handler(results, synthetic_hashes=frozenset()):
    """Returns (preview, result card HTML, technical details HTML, raw details dict). Always runs real inference."""
    def handle(path):
        if not path:
            return (gr.update(value=None, visible=False), '<div class="p-card p-placeholder"><p class="p-state-title">'
                    'Choose an image first</p><p>Select a transparent-background PNG, then press Analyse image.</p>'
                    f'{footnote()}</div>', TECH_PLACEHOLDER, {})
        result = inference.analyse(path)
        ok = result.status == "ok"
        preview = gr.update(value=inference.preview_image(path) if ok else None, visible=ok)
        card = format_result(result, results, synthetic=is_synthetic(path, synthetic_hashes))
        return preview, card, format_technical(result, path), result.as_dict()
    return handle


# --- Research example explorer -------------------------------------------------

# Purpose and expected behaviour of each synthetic example (keys match make_synthetic_examples labels).
EXAMPLE_INFO = {
    "Synthetic crop within training size range": (
        "Shows a supported input: a transparent-background crop sized like the training images.",
        "Accepted and analysed. Its smooth artificial texture may trigger a colour-range warning.", "accepted"),
    "Oversized synthetic crop (out of range)": (
        "Checks that inputs larger than anything in the training data are flagged.",
        "Accepted, with out-of-range warnings marking the result as unreliable.", "warned"),
    "No transparency (plain RGB PNG)": (
        "Imitates an ordinary photo: a PNG with no transparent background.",
        "Rejected before analysis, with an explanation.", "rejected"),
    "Fully opaque RGBA (no background removed)": (
        "Has an alpha channel, but nothing is transparent, so no background was removed.",
        "Rejected before analysis, with an explanation.", "rejected"),
    "Too little tissue (64 opaque pixels)": (
        "Contains too few tissue pixels to measure colour reliably.",
        "Rejected: at least 100 tissue pixels are required.", "rejected"),
    "JPEG renamed to .png": (
        "A JPEG disguised with a .png extension.",
        "Rejected: the file signature shows it is not a PNG.", "rejected"),
}


def example_panel(label):
    purpose, expected, _ = EXAMPLE_INFO.get(label, ("", "", ""))
    return (f'<div class="p-example"><span class="p-tag">Synthetic</span><p class="p-example-name">{esc(label)}</p>'
            f'<dl><dt>Why it exists</dt><dd>{esc(purpose)}</dd><dt>Expected behaviour</dt><dd>{esc(expected)}</dd></dl></div>')


def example_verdict(label, details):
    """Software-testing view: compare observed behaviour (the analysis result dict) with the expected behaviour."""
    expected = EXAMPLE_INFO.get(label, ("", "", ""))[2]
    warnings = details.get("warnings") or []
    if details.get("status") == "rejected":
        observed, ok = "Rejected before analysis.", expected == "rejected"
    elif warnings:
        observed = f"Accepted, with {len(warnings)} out-of-range warning(s)."
        ok = expected in ("warned", "accepted")
    else:
        observed, ok = "Accepted, with no warnings.", expected == "accepted"
    verdict = "Behaves as designed" if ok else "Unexpected behaviour"
    return (f'<div class="p-verdict {"is-ok" if ok else "is-bad"}">{ICON_CHECK if ok else ICON_NOTICE}<div>'
            f'<strong>{verdict}</strong><p>Observed: {esc(observed)} This checks how the software handles the input; '
            'the score of a synthetic image has no medical meaning.</p></div></div>')


# --- Dashboard presentation ---------------------------------------------------

AXIS = ('<div class="p-bar-row p-axis-row" aria-hidden="true"><span></span><div class="p-axis">'
        '<span style="left:0%">0.5 · chance</span><span style="left:50%">0.75</span><span style="left:100%">1.0</span>'
        '</div><span></span></div>')


def minus(text):
    """Typographic minus sign for display."""
    return str(text).replace("-", "−")


def _bar(value, lo=0.5, hi=1.0):
    return min(max((value - lo) / (hi - lo), 0.0), 1.0) * 100


def _bar_row(name, role, value, lo, hi, value_note, primary=False, compact=False):
    role_html = f"<span>{esc(role)}</span>" if role else ""
    return (f'<div class="p-bar-row{" compact" if compact else ""}{" is-primary" if primary else ""}">'
            f'<div class="p-bar-label"><strong>{esc(name)}</strong>{role_html}</div><div class="p-bar-track">'
            f'<span class="p-bar-fill" style="width:{_bar(value):.1f}%"></span>'
            f'<span class="p-bar-ci" style="left:{_bar(lo):.1f}%;width:{_bar(hi) - _bar(lo):.1f}%"></span></div>'
            f'<div class="p-bar-value">{r3(value)}<small>{value_note}</small></div></div>')


def dashboard_html(results):
    base, s = results["baseline_test"], results["splits"]
    cm, b = base["confusion_matrix"], base["ci95_bootstrap"]
    pos, neg = cm["tp"] + cm["fn"], cm["tn"] + cm["fp"]
    tiles = [
        ("ROC-AUC", r3(base["roc_auc"]), f"95% CI {ci(b['roc_auc'])}",
         "How well the model ranks the two dataset-labelled groups across all possible thresholds. 0.5 is chance; 1.0 is perfect."),
        ("Sensitivity", pct(base["sensitivity"]), f"{cm['tp']} of {pos} anaemia-labelled test images",
         "The share of anaemia-labelled test images the model classified as positive (above the research threshold)."),
        ("Specificity", pct(base["specificity"]), f"{cm['tn']} of {neg} non-anaemia-labelled test images",
         "The share of non-anaemia-labelled test images the model classified as negative (below the research threshold)."),
        ("Held-out test set", f"{s['test']['n']} images", f"{pos} anaemia-labelled · {neg} not",
         f"Set aside before training and scored once. A further {s['train']['n']} images were used for training."),
    ]
    tile_html = "".join(f'<div class="p-tile"><span class="p-tile-label">{esc(t)}</span><span class="p-tile-value">'
                        f'{esc(v)}</span><span class="p-tile-sub">{esc(sub)}</span><p>{esc(d)}</p></div>'
                        for t, v, sub, d in tiles)

    rows = ""
    for m in results["raw"]["test_models"]:
        lo, hi = m["ci"] or (m["auc"], m["auc"])
        rows += _bar_row(m["model"], m["role"], m["auc"], lo, hi, f"95% CI {ci((lo, hi))}",
                         primary=m["role"].startswith("Primary"))
    et = results["raw"]["et_pair"]
    established = et["ci"][0] > 0
    legend = ('<div class="p-legend" aria-hidden="true"><span><i class="p-key-bar"></i>ROC-AUC</span>'
              '<span><i class="p-key-ci"></i>95% interval</span><span><i class="p-key-primary"></i>Used by this app</span></div>')
    compare = (f'<div class="p-chart"><div class="p-chart-head"><p class="p-chart-title">Held-out test ROC-AUC '
               f'(n = {s["test"]["n"]})</p>{legend}</div>{rows}{AXIS}</div>'
               '<p class="p-muted">Longer intervals mean more uncertainty. The three intervals overlap substantially.</p>')
    lr = next(m for m in results["raw"]["test_models"] if m["role"].startswith("Primary"))
    et_model = next(m for m in results["raw"]["test_models"] if m["model"] == "Extra Trees")
    decision = (f'<div class="p-decision"><h4>Why Logistic Regression is the deployed model</h4><dl>'
                f'<div><dt>Deployed</dt><dd>Logistic Regression · held-out ROC-AUC {r3(lr["auc"])}</dd></div>'
                f'<div><dt>Experimental candidate</dt><dd>Extra Trees · held-out ROC-AUC {r3(et_model["auc"])}</dd></div>'
                f'<div><dt>Paired difference</dt><dd>{"+" if et["delta"] >= 0 else ""}{minus(r3(et["delta"]))} '
                f'(95% CI {minus(ci(et["ci"]))}): the interval {"excludes" if established else "includes"} zero, so the '
                f'improvement is {"supported" if established else "inconclusive"}</dd></div>'
                '<div><dt>Rule, fixed before testing</dt><dd>A challenger replaces the deployed model only if its paired '
                'improvement interval lies entirely above zero.</dd></div></dl></div>')

    cv_rows = ""
    for m in results["raw"]["nested"]:
        role = {"Logistic Regression": "Used by this app", "Extra Trees": "Experimental candidate"}.get(m["model"], "")
        cv_rows += _bar_row(m["model"], role, m["mean"], m["mean"] - m["sd"], m["mean"] + m["sd"],
                            f"± {r3(m['sd'])} SD", primary=m["model"] == "Logistic Regression", compact=True)
    n_folds = results["raw"]["n_outer_folds"]
    cv_legend = ('<div class="p-legend" aria-hidden="true"><span><i class="p-key-bar"></i>Mean ROC-AUC</span>'
                 '<span><i class="p-key-ci"></i>±1 standard deviation across folds</span></div>')

    return f"""
<div class="p-page">
  <header class="p-intro">
    <h2>Research performance</h2>
    <p>These metrics describe how the model's predictions matched the dataset's own labels. They are not measures
    of clinical accuracy.</p>
  </header>
  <section>
    <div class="p-section-head"><h3>Held-out test results</h3><span class="p-pill">Logistic Regression · used by this app</span></div>
    <div class="p-tiles">{tile_html}</div>
    <div class="p-explain">
      <p><strong>Uncertainty.</strong> The test set is small, so every figure is uncertain. The 95% interval for
      ROC-AUC runs from {r3(b['roc_auc'][0])} to {r3(b['roc_auc'][1])}: values anywhere in that range are consistent
      with these {s['test']['n']} images.</p>
      <p class="p-callout">{ICON_INFO}<span>The evaluation uses a small, single-source dataset. These results do not
      establish clinical performance or generalisation to new patients. The same test images were used for three
      one-shot evaluations, so they are no longer an untouched final test.</span></p>
    </div>
  </section>
  <section>
    <div class="p-section-head"><h3>Compared on the same test images</h3></div>
    {compare}
    {decision}
  </section>
  <section class="p-separated">
    <div class="p-section-head"><h3>Cross-validation</h3><span class="p-pill quiet">Training data only · not test results</span></div>
    <p class="p-muted">Nested cross-validation on the {s['train']['n']} training images: every model was tuned and scored
    on the same {n_folds} outer folds. These estimates guided model selection. They are not held-out test results,
    and the spread shows how much each estimate varied from fold to fold.</p>
    <div class="p-chart"><div class="p-chart-head"><p class="p-chart-title">Nested-CV ROC-AUC across {n_folds} outer
    folds</p>{cv_legend}</div>{cv_rows}{AXIS}</div>
  </section>
</div>"""


LIMITATION_GROUPS = [
    ("Data limitations", [
        ("Small dataset", "A few hundred images in total, with a small held-out test set, so uncertainty is wide."),
        ("Single data source", "All images come from one dataset whose licence and provenance are still being verified."),
        ("Uncertain patient independence", "No patient identifiers exist, so one person's images could appear in both "
                                           "training and test data, inflating reported performance."),
        ("Possible confounding", "Crop size alone predicts the labels better than chance, and lighting and cameras vary. "
                                 "The model may learn such shortcuts rather than pallor."),
    ]),
    ("Model limitations", [
        ("Limited generalisation", "The model has not been tested on other people, cameras, lighting, skin tones, ages "
                                   "or populations."),
        ("Scores are not probabilities", "The model score is not a calibrated probability of anaemia."),
        ("Dataset label convention", "The anaemic label means Hb < 11 g/dL in this dataset: a labelling convention, "
                                     "not a universal diagnostic threshold."),
    ]),
    ("Input limitations", [
        ("Strict input requirement", "Only pre-segmented conjunctiva crops saved as transparent-background PNGs are "
                                     "supported. Ordinary photographs, JPEGs, selfies and unsegmented eye images are not."),
        ("Out-of-range inputs", "Images whose size or colour features fall outside the training data are flagged, and "
                                "their results should be treated as unreliable."),
    ]),
    ("Clinical limitations", [
        ("Not a diagnostic tool", "Pallor is a research prototype, not a medical device. It cannot diagnose anaemia or "
                                  "rule it out."),
        ("Cannot measure haemoglobin", "It classifies colour features against a dataset label. Only a blood test "
                                       "measures haemoglobin."),
        ("No independent clinical validation", "No prospective study, external validation or comparison with clinicians "
                                               "has been carried out."),
    ]),
]

LIMITATIONS = (
    f'<div class="p-page"><header class="p-intro"><h2>Limitations</h2><p>{esc(DISCLAIMER)}</p></header>'
    '<div class="p-limit-groups">'
    + "".join(f'<section class="p-limit-group"><h3>{esc(g)}</h3>'
              + "".join(f'<div class="p-limit"><h4>{esc(t)}</h4><p>{esc(d)}</p></div>' for t, d in items)
              + '</section>' for g, items in LIMITATION_GROUPS)
    + '</div><p class="p-muted">Full methodology: <code>docs/REPORT.md</code> and <code>models/README.md</code>.</p></div>'
)


# --- Theme and styling (system fonts only: no external font requests) ---------

SYSTEM_FONT = ("-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif")
MONO_FONT = ("SF Mono", "ui-monospace", "Menlo", "Consolas", "monospace")

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.stone,
    font=SYSTEM_FONT, font_mono=MONO_FONT,
)
# The light design is the only theme: dark-mode variants mirror the light values, so the app looks the same
# whatever the operating system's appearance setting.
_LIGHT = dict(
    body_background_fill="#F7F6F3", body_text_color="#1C1C1E", body_text_color_subdued="#5A5A60",
    background_fill_primary="#FFFFFF", background_fill_secondary="#F7F6F3", block_background_fill="#FFFFFF",
    block_border_color="#E4E2DD", block_label_text_color="#5A5A60", border_color_primary="#E4E2DD",
    input_border_color_focus="#1E6B74", button_primary_background_fill="#1E6B74",
    button_primary_background_fill_hover="#17565E", button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="#F0EEEA", color_accent="#1E6B74", color_accent_soft="#E7F1F2",
    input_background_fill="#FFFFFF", link_text_color="#1E6B74",
)
THEME = THEME.set(**_LIGHT, **{f"{k}_dark": v for k, v in _LIGHT.items() if hasattr(THEME, f"{k}_dark")},
                  block_border_width="1px", block_shadow="none", block_radius="12px", container_radius="12px",
                  input_radius="6px", button_large_radius="6px", button_small_radius="6px")

CSS = """
/* ---- Design tokens ------------------------------------------------------
   Light theme is the designed experience (forced on load; see FORCE_LIGHT_JS).
   Spacing: 8-point scale (4 for fine adjustments). Type: 12/13/15/17/20/24/30 px.
   Radii: 6 (controls, tags) and 12 (panels). No shadows, glows or decorative motion. */
:root, .dark {
  --s-half: 4px; --s-1: 8px; --s-2: 16px; --s-3: 24px; --s-4: 32px; --s-5: 40px; --s-6: 48px; --s-8: 64px;
  --fs-cap: 12px; --fs-meta: 13px; --fs-body: 15px; --fs-lead: 17px; --fs-h3: 20px; --fs-h2: 24px; --fs-h1: 30px;
  --lh-body: 1.55; --lh-tight: 1.25;
  --r-sm: 6px; --r-md: 12px;
  --c-bg: #F7F6F3; --c-surface: #FFFFFF; --c-text: #1C1C1E; --c-muted: #5A5A60; --c-line: #E4E2DD;
  --c-soft: #F0EEEA; --c-accent: #1E6B74; --c-accent-strong: #17565E; --c-accent-soft: #E7F1F2;
  --c-warn: #8A5A12; --c-warn-soft: #FBF3E4; --c-ok: #2C6B4F; --c-ok-soft: #EAF4EE;
}

/* ---- Page frame ---------------------------------------------------------- */
body, .gradio-container { background: var(--c-bg) !important; color: var(--c-text); }
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; padding: 0 var(--s-4) var(--s-5) !important;
  -webkit-font-smoothing: antialiased; overflow-x: clip; }
footer { display: none !important; }
.html-container { padding: 0 !important; }
.p-card, .p-tech, .p-page, .p-req, .p-example { overflow-wrap: anywhere; }
.p-icon { flex: none; width: 16px; height: 16px; margin-top: 2px; }
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, select:focus-visible,
[tabindex]:focus-visible, a:focus-visible, label:focus-within {
  outline: 2px solid var(--c-accent) !important; outline-offset: 2px !important; }

/* ---- Header -------------------------------------------------------------- */
#p-header { padding: var(--s-3) 0 0; }
.p-brand { display: flex; align-items: baseline; gap: var(--s-2); flex-wrap: wrap; }
.p-wordmark { font-size: 32px; line-height: 1; font-weight: 700; letter-spacing: -0.03em; color: var(--c-text); }
.p-subtitle { font-size: var(--fs-body); color: var(--c-muted); }
.p-badge { align-self: center; margin-left: auto; padding: var(--s-half) var(--s-1); border: 1px solid var(--c-line); border-radius: var(--r-sm);
  background: var(--c-surface); font-size: var(--fs-cap); font-weight: 500; color: var(--c-muted); }

/* ---- Navigation (tabs) --------------------------------------------------- */
/* Spacing lives inside each tab (padding) because Gradio's fit check sums button widths and ignores gap. */
.tabs { margin-top: var(--s-2); }
[role="tablist"], .tab-container { gap: 0 !important; margin-left: calc(-1 * var(--s-1)) !important; }
[role="tablist"] { border-bottom: 1px solid var(--c-line) !important; }
.tab-container button { padding: var(--s-1) var(--s-1) !important; margin-right: var(--s-1) !important;
  background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important;
  border-radius: 0 !important; font-size: var(--fs-body) !important; font-weight: 500 !important; color: var(--c-muted) !important; }
[role="tab"]:hover { color: var(--c-text) !important; }
[role="tab"][aria-selected="true"] { color: var(--c-text) !important; font-weight: 600 !important;
  border-bottom-color: var(--c-accent) !important; }
.tabitem { padding: 0 !important; border: none !important; background: transparent !important; }

/* ---- Page intro (shared by every tab) ----------------------------------- */
.p-intro { padding: var(--s-4) 0 var(--s-2); max-width: 680px; }
.p-intro p.p-cannot { margin-top: var(--s-1); font-size: var(--fs-meta); font-weight: 500; color: var(--c-text); }
.p-intro h1, .p-intro h2 { margin: 0 0 var(--s-1); font-size: var(--fs-h1); line-height: var(--lh-tight);
  font-weight: 650; letter-spacing: -0.02em; color: var(--c-text); }
.p-intro p { margin: 0; font-size: var(--fs-lead); line-height: var(--lh-body); color: var(--c-muted); }

/* ---- Analyse: three-step workflow ---------------------------------------- */
.p-steps { list-style: none; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: var(--s-3);
  margin: 0 0 var(--s-3); padding: 0; }
.p-steps li { padding-top: var(--s-2); border-top: 2px solid var(--c-line); }
.p-steps li.is-first { border-top-color: var(--c-accent); }
.p-step-num { display: block; font-size: var(--fs-meta); font-weight: 600; color: var(--c-accent); font-variant-numeric: tabular-nums; }
.p-step-title { display: block; margin: var(--s-half) 0; font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-steps p { margin: 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }

/* ---- Analyse: upload panel ----------------------------------------------- */
#p-upload-col { gap: var(--s-2); padding: var(--s-3) !important; background: var(--c-surface);
  border: 1px solid var(--c-line); border-radius: var(--r-md); }
#p-result-col { gap: var(--s-2); scroll-margin-top: var(--s-2); }
#p-analyse-row { align-items: flex-start !important; }
@media (min-width: 861px) { #p-upload-col { position: sticky; top: var(--s-2); align-self: flex-start; } }
.p-panel-text { margin: var(--s-half) 0 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
#p-demo-offer, #p-demo-note { gap: var(--s-1); padding: var(--s-2) !important; border-radius: var(--r-sm);
  background: var(--c-accent-soft); border: 1px solid transparent; }
#p-demo-note { background: var(--c-bg); border-color: var(--c-line); }
#p-try-demo, #p-exit-demo { border-radius: var(--r-sm) !important; font-weight: 600 !important; }
#p-try-demo { background: var(--c-surface) !important; color: var(--c-accent) !important; border: 1px solid var(--c-accent) !important; }
#p-try-demo:hover { background: var(--c-accent) !important; color: #FFFFFF !important; }
.p-upload-title { padding-top: var(--s-1); }
.p-synthetic { display: flex; gap: var(--s-1); align-items: baseline; margin: 0 0 var(--s-3); padding: var(--s-2);
  border-radius: var(--r-sm); background: var(--c-accent-soft); font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-text); }
.p-decision { margin-top: var(--s-3); padding: var(--s-3); background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--r-md); }
.p-decision h4 { margin: 0 0 var(--s-2); font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-decision dl { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s-2) var(--s-3); margin: 0; }
.p-decision dt { font-size: var(--fs-cap); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--c-muted); }
.p-decision dd { margin: var(--s-half) 0 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-text); }
.p-panel-title { margin: 0; font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-req { display: grid; gap: var(--s-1); margin: 0; padding: 0; list-style: none; }
.p-req li { display: flex; gap: var(--s-1); font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-req li::before { content: ""; flex: none; width: 6px; height: 6px; margin-top: 7px; border-radius: 50%; background: var(--c-accent); }
.p-req li.no::before { width: 5px; height: 5px; background: transparent; border: 1.5px solid var(--c-muted); }
.p-req strong { font-weight: 600; color: var(--c-text); }
#p-upload { border: 1px dashed var(--c-muted) !important; border-radius: var(--r-md) !important; background: var(--c-bg) !important; }
#p-upload:hover { border-color: var(--c-accent) !important; }
#p-analyse { min-height: 44px; border-radius: var(--r-sm) !important; font-size: var(--fs-body) !important; font-weight: 600 !important; }
#p-analyse:disabled, #p-ex-run:disabled { background: var(--c-soft) !important; color: var(--c-muted) !important;
  border-color: var(--c-line) !important; cursor: not-allowed; }
.p-upload-status { display: flex; gap: var(--s-1); margin: 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-upload-status.is-ready { color: var(--c-text); }
.p-upload-status.is-ready .p-dot { margin-top: 6px; background: var(--c-ok); }
.p-upload-status.is-error { color: var(--c-warn); }
#p-preview { border-radius: var(--r-md) !important; overflow: hidden; }

/* ---- Result card and states ---------------------------------------------- */
.p-card { padding: var(--s-4); background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--r-md); }
.p-placeholder { padding: var(--s-4) var(--s-3); text-align: center; background: transparent; border-style: dashed; }
.p-state-title { margin: 0 0 var(--s-1); font-size: var(--fs-lead); font-weight: 600; color: var(--c-text); }
.p-placeholder p:not(.p-state-title) { max-width: 420px; margin: 0 auto; font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-muted); }
.p-spinner { display: block; width: 20px; height: 20px; margin: 0 auto var(--s-2); border: 2px solid var(--c-line);
  border-top-color: var(--c-accent); border-radius: 50%; animation: p-spin .9s linear infinite; }
@keyframes p-spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .p-spinner { animation: none; border-top-color: var(--c-line); } }
.p-block + .p-block { margin-top: var(--s-3); padding-top: var(--s-3); border-top: 1px solid var(--c-line); }
.p-block h4 { margin: 0 0 var(--s-1); font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-block > p { margin: 0; font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-text); }
.p-status { display: flex; align-items: center; gap: var(--s-1); margin-bottom: var(--s-1); }
.p-dot { flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--c-muted); }
.p-eyebrow { font-size: var(--fs-meta); font-weight: 500; color: var(--c-muted); }
.p-result h3 { margin: 0; font-size: var(--fs-h2); line-height: var(--lh-tight); font-weight: 650; letter-spacing: -0.015em; color: var(--c-text); }
.p-result.is-above .p-dot { background: var(--c-accent); }
.p-result.is-below .p-dot { width: 6px; height: 6px; background: transparent; border: 2px solid var(--c-accent); }
.p-label-note { margin: var(--s-1) 0 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-score { margin-top: var(--s-3); padding: var(--s-2); background: var(--c-bg); border-radius: var(--r-sm); }
.p-score-head { display: flex; justify-content: space-between; align-items: baseline; gap: var(--s-2); }
.p-score-head span { font-size: var(--fs-meta); font-weight: 500; color: var(--c-muted); }
.p-score-head strong { font-size: var(--fs-h3); font-weight: 600; font-variant-numeric: tabular-nums; color: var(--c-text); }
.p-track { position: relative; height: 4px; margin: var(--s-2) 0 var(--s-1); border-radius: 2px; background: var(--c-line); }
.p-track-threshold { position: absolute; top: -6px; width: 2px; height: 16px; margin-left: -1px; background: var(--c-muted); }
.p-track-dot { position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%;
  background: var(--c-surface); border: 2px solid var(--c-accent); }
.p-track-legend { position: relative; display: flex; justify-content: space-between; font-size: var(--fs-cap); color: var(--c-muted); }
.p-track-label { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; }
.p-score .p-muted { margin-top: var(--s-2); }
.p-muted { margin: var(--s-1) 0 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-sub { margin-top: var(--s-2); }
.p-sub h5, .p-next h5 { margin: 0 0 var(--s-half); font-size: var(--fs-meta); font-weight: 600; color: var(--c-text); }
.p-sub p { margin: 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-groups { display: flex; flex-wrap: wrap; gap: var(--s-1); margin: var(--s-1) 0 0; padding: 0; list-style: none; }
.p-groups li { padding: var(--s-half) var(--s-1); border: 1px solid var(--c-line); border-radius: var(--r-sm);
  font-size: var(--fs-meta); color: var(--c-text); }
.p-groups em { font-style: normal; color: var(--c-muted); font-variant-numeric: tabular-nums; }
.p-next { margin-top: var(--s-2); padding: var(--s-2); border-left: 2px solid var(--c-accent); background: var(--c-accent-soft);
  border-radius: 0 var(--r-sm) var(--r-sm) 0; }
.p-next p { margin: 0; font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-text); }
.p-notice { display: flex; gap: var(--s-1); padding: var(--s-2); border-radius: var(--r-sm); background: var(--c-warn-soft);
  color: var(--c-warn); font-size: var(--fs-meta); line-height: var(--lh-body); }
.p-notice ul { margin: var(--s-half) 0 0; padding-left: var(--s-2); color: var(--c-text); }
.p-ok { display: flex; gap: var(--s-1); margin: 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-ok); }
.p-ok span { color: var(--c-text); }
.p-reasons { margin: var(--s-1) 0 var(--s-2); padding-left: var(--s-3); font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-text); }
.p-result.is-rejected h3 { margin-bottom: var(--s-1); }
.p-reliability { margin-top: var(--s-3); padding-top: var(--s-2); border-top: 1px solid var(--c-line); }
.p-footnote { display: flex; gap: var(--s-1); margin: var(--s-2) 0 0; font-size: var(--fs-cap); line-height: var(--lh-body); color: var(--c-muted); }
.p-footnote .p-icon { width: 14px; height: 14px; }

/* ---- Technical details --------------------------------------------------- */
#p-tech-acc, #p-tables, #p-ex-tech { border-radius: var(--r-md) !important; }
.p-tech table { width: 100%; border-collapse: collapse; font-size: var(--fs-meta); table-layout: fixed; }
.p-tech th, .p-tech td { padding: var(--s-1) 0; text-align: left; vertical-align: top; border-bottom: 1px solid var(--c-line); }
.p-tech th { width: 40%; padding-right: var(--s-2); font-weight: 500; color: var(--c-muted); }
.p-tech td { color: var(--c-text); font-variant-numeric: tabular-nums; }
.p-tech h4 { margin: var(--s-3) 0 0; font-size: var(--fs-body); font-weight: 600; }
.p-tech h5 { margin: var(--s-2) 0 0; font-size: var(--fs-cap); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--c-muted); }
.p-features thead th { font-size: var(--fs-cap); font-weight: 500; }
.p-features th:nth-child(1), .p-features td:nth-child(1) { width: 40%; }
.p-features th:nth-child(2), .p-features td:nth-child(2) { width: 22%; }
.p-features td { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.p-features code { font-family: var(--font-mono); font-size: var(--fs-cap); background: none; padding: 0; }
.p-features tr.is-out td { color: var(--c-warn); font-weight: 500; }

/* ---- Example explorer ---------------------------------------------------- */
#p-ex-col { gap: var(--s-2); padding: var(--s-3) !important; background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--r-md); }
.p-tag { display: inline-block; flex: none; white-space: nowrap; padding: 2px var(--s-1); border-radius: var(--r-sm); background: var(--c-accent-soft);
  font-size: var(--fs-cap); font-weight: 600; color: var(--c-accent); }
.p-example-name { margin: var(--s-1) 0 var(--s-2); font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-example dl { margin: 0; display: grid; gap: var(--s-1); }
.p-example dt { font-size: var(--fs-cap); font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; color: var(--c-muted); }
.p-example dd { margin: 0 0 var(--s-1); font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-text); }
.p-verdict { display: flex; gap: var(--s-1); margin-bottom: var(--s-2); padding: var(--s-2); border-radius: var(--r-sm);
  font-size: var(--fs-meta); line-height: var(--lh-body); }
.p-verdict.is-ok { background: var(--c-ok-soft); color: var(--c-ok); }
.p-verdict.is-bad { background: var(--c-warn-soft); color: var(--c-warn); }
.p-verdict strong { font-size: var(--fs-body); }
.p-verdict p { margin: var(--s-half) 0 0; color: var(--c-text); }
.p-explorer-note { margin: 0 0 var(--s-3); max-width: 680px; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }

/* ---- Dashboard ----------------------------------------------------------- */
.p-page section { margin-bottom: var(--s-5); }
.p-section-head { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s-2); flex-wrap: wrap; margin-bottom: var(--s-2); }
.p-section-head h3 { margin: 0; font-size: var(--fs-h3); font-weight: 600; letter-spacing: -0.01em; color: var(--c-text); }
.p-pill { padding: var(--s-half) var(--s-1); border: 1px solid var(--c-line); border-radius: var(--r-sm); background: var(--c-surface);
  font-size: var(--fs-cap); font-weight: 500; color: var(--c-muted); }
.p-tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; overflow: hidden;
  background: var(--c-line); border: 1px solid var(--c-line); border-radius: var(--r-md); }
.p-tile { display: flex; flex-direction: column; gap: var(--s-half); padding: var(--s-3); background: var(--c-surface); }
.p-tile-label { font-size: var(--fs-meta); font-weight: 500; color: var(--c-muted); }
.p-tile-value { font-size: var(--fs-h1); font-weight: 650; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; color: var(--c-text); }
.p-tile-sub { font-size: var(--fs-meta); font-weight: 500; color: var(--c-accent); }
.p-tile p { margin: var(--s-1) 0 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-explain { margin-top: var(--s-2); max-width: 760px; }
.p-explain > p { margin: 0 0 var(--s-1); font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-text); }
.p-callout { display: flex; gap: var(--s-1); margin: 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted) !important; }
.p-chart { padding: var(--s-2) var(--s-3); background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--r-md); }
.p-chart-head { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s-2); flex-wrap: wrap;
  padding-bottom: var(--s-1); border-bottom: 1px solid var(--c-line); }
.p-chart-title { margin: 0; font-size: var(--fs-meta); font-weight: 600; color: var(--c-text); }
.p-legend { display: flex; gap: var(--s-2); flex-wrap: wrap; font-size: var(--fs-cap); color: var(--c-muted); }
.p-legend span { display: inline-flex; align-items: center; gap: var(--s-half); }
.p-legend i { display: inline-block; }
.p-key-bar { width: 16px; height: 8px; border-radius: 4px; background: var(--c-muted); opacity: .45; }
.p-key-primary { width: 16px; height: 8px; border-radius: 4px; background: var(--c-accent); }
.p-key-ci { width: 16px; height: 2px; background: var(--c-text); }
.p-bar-row { display: grid; grid-template-columns: 216px minmax(0, 1fr) 128px; gap: var(--s-2); align-items: center;
  padding: var(--s-2) 0; border-bottom: 1px solid var(--c-line); }
.p-bar-row.compact { padding: var(--s-1) 0; }
.p-bar-label strong { display: block; font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-bar-label span { font-size: var(--fs-cap); color: var(--c-muted); }
.p-bar-track { position: relative; height: 8px; border-radius: 4px; background: var(--c-soft); }
.p-bar-fill { position: absolute; inset: 0 auto 0 0; border-radius: 4px; background: var(--c-muted); opacity: .45; }
.p-bar-row.is-primary .p-bar-fill { background: var(--c-accent); opacity: 1; }
.p-bar-ci { position: absolute; top: 50%; height: 2px; margin-top: -1px; background: var(--c-text); }
.p-bar-ci::before, .p-bar-ci::after { content: ""; position: absolute; top: -4px; width: 2px; height: 10px; background: inherit; }
.p-bar-ci::before { left: 0; } .p-bar-ci::after { right: 0; }
.p-bar-value { text-align: right; font-size: var(--fs-body); font-weight: 600; font-variant-numeric: tabular-nums; color: var(--c-text); }
.p-bar-value small { display: block; font-size: var(--fs-cap); font-weight: 400; color: var(--c-muted); }
.p-axis-row { padding: var(--s-1) 0 0; border-bottom: none; }
.p-axis { position: relative; height: 16px; font-size: var(--fs-cap); color: var(--c-muted); }
.p-axis span { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; }
.p-axis span:first-child { transform: none; }
.p-axis span:last-child { transform: translateX(-100%); }
.p-page .p-section-head + .p-muted, .p-separated > .p-muted { margin: 0 0 var(--s-2); max-width: 760px; }
.p-separated { padding-top: var(--s-5); border-top: 1px solid var(--c-line); }

/* ---- Limitations --------------------------------------------------------- */
.p-limit-groups { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: var(--s-3); margin-bottom: var(--s-3); }
.p-limit-group { margin: 0 !important; padding: var(--s-3); background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--r-md); }
.p-limit-group h3 { margin: 0 0 var(--s-1); font-size: var(--fs-body); font-weight: 600; color: var(--c-accent); }
.p-limit { padding: var(--s-1) 0; border-top: 1px solid var(--c-line); }
.p-limit:first-of-type { border-top: none; }
.p-limit h4 { margin: 0 0 2px; font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-limit p { margin: 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }

/* ---- Global footer notice ------------------------------------------------ */
#disclaimer { margin-top: var(--s-5) !important; padding: var(--s-2) 0 0 !important; background: transparent !important;
  border: none !important; border-top: 1px solid var(--c-line) !important; border-radius: 0 !important; }
#disclaimer p { margin: 0; font-size: var(--fs-cap); line-height: var(--lh-body); color: var(--c-muted); }

/* ---- Narrow screens ------------------------------------------------------ */
@media (max-width: 860px) {
  .gradio-container { padding: 0 var(--s-2) var(--s-4) !important; }
  .p-badge { margin-left: 0; }
  .p-intro { padding-top: var(--s-3); }
  .p-intro h1, .p-intro h2 { font-size: var(--fs-h2); }
  .p-steps { grid-template-columns: 1fr; gap: var(--s-2); }
  .p-card { padding: var(--s-3); }
  .p-result h3 { font-size: var(--fs-h3); }
  .p-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .p-limit-groups { grid-template-columns: 1fr; }
  .p-decision dl { grid-template-columns: 1fr; }
  .p-wordmark { font-size: 28px; }
  .p-bar-row { grid-template-columns: minmax(0, 1fr) 112px; }
  .p-bar-track { grid-column: 1 / -1; grid-row: 2; }
  .p-axis-row > span { display: none; }
  .p-axis { grid-column: 1 / -1; }
  .tab-container button { font-size: var(--fs-meta) !important; margin-right: 0 !important; }
}
@media (max-width: 480px) { .p-tiles { grid-template-columns: 1fr; } }
"""

# Brings the result into view after analysis when it is off screen (mainly on narrow screens).
SCROLL_TO_RESULT_JS = ("() => { const el = document.querySelector('#p-result-col'); if (!el) return; "
                       "const r = el.getBoundingClientRect(); if (r.top < 0 || r.top > window.innerHeight * 0.6) "
                       "el.scrollIntoView({ block: 'start' }); }")

# Light is the designed theme: drop Gradio's system-driven dark class on load.
FORCE_LIGHT_JS = "() => { document.querySelectorAll('.dark').forEach((el) => el.classList.remove('dark')); }"

LAUNCH_KWARGS = {"server_name": "127.0.0.1", "server_port": 7860, "share": False, "inbrowser": False,
                 "show_error": True, "max_file_size": "10mb", "enable_monitoring": False,
                 "mcp_server": False, "pwa": False, "ssr_mode": False,
                 "theme": THEME, "css": CSS, "footer_links": [], "js": FORCE_LIGHT_JS}


def launch_settings(env=os.environ):
    """Launch arguments for the current environment.

    Locally: 127.0.0.1 (unchanged local-only default). On Render, which always sets RENDER=true, bind 0.0.0.0
    so the platform can route traffic. The port comes from PORT (Render's assigned port), defaulting to 7860.
    Public share links stay disabled in every environment.
    """
    on_render = env.get("RENDER") == "true"
    return {**LAUNCH_KWARGS, "server_name": "0.0.0.0" if on_render else "127.0.0.1",
            "server_port": int(env.get("PORT", "7860")), "share": False}


def busy_state():
    return gr.update(value="Analysing…", interactive=False), LOADING_CARD


def idle_state(label):
    return gr.update(value=label, interactive=True)


def build_app(results=None, example_dir=None):
    results = results or load_results()
    inference.default_model()       # fail fast on integrity problems
    inference.default_reference()
    example_dir = example_dir or tempfile.mkdtemp(prefix="pallor_synthetic_")
    examples = make_synthetic_examples(example_dir)
    synthetic_hashes = frozenset(inference.sha256_file(p) for p, _ in examples.values())
    handle = make_handler(results, synthetic_hashes)
    on_upload = make_upload_handler(results, synthetic_hashes)
    n_features = len(inference.default_model()["feature_names"])
    first_example = next(iter(examples))
    demo_path = examples[DEMO_EXAMPLE][0]

    def run_example(label):
        preview, card, tech, details = handle(examples[label][0])
        return preview, example_verdict(label, details) + card, tech, details

    with gr.Blocks(title="Pallor — conjunctiva colour research", analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:
        gr.HTML('<div class="p-brand"><span class="p-wordmark">Pallor</span>'
                '<span class="p-subtitle">Conjunctiva colour research</span>'
                '<span class="p-badge">Experimental research prototype</span></div>', elem_id="p-header")

        with gr.Tabs():
            with gr.Tab("Analyse"):
                gr.HTML('<header class="p-intro"><h1>Explore conjunctiva colour patterns</h1><p>Pallor analyses colour '
                        'features in a supported conjunctiva image and compares them with patterns learned from a '
                        'research dataset.</p><p class="p-cannot">It cannot diagnose anaemia or measure haemoglobin.</p>'
                        '</header><ol class="p-steps">'
                        '<li class="is-first"><span class="p-step-num">01</span><span class="p-step-title">Upload an image</span>'
                        '<p>A pre-segmented conjunctiva crop saved as a transparent-background PNG, or try the synthetic demo.</p></li>'
                        '<li><span class="p-step-num">02</span><span class="p-step-title">Analyse colour features</span>'
                        f'<p>Pallor checks the image and measures {n_features} colour features from the visible tissue.</p></li>'
                        '<li><span class="p-step-num">03</span><span class="p-step-title">Review the model output</span>'
                        '<p>See the predicted dataset label, what it means and any input warnings.</p></li></ol>')
                with gr.Row(equal_height=False, elem_id="p-analyse-row"):
                    with gr.Column(scale=5, min_width=300, elem_id="p-upload-col"):
                        with gr.Column(elem_id="p-demo-offer") as demo_offer:
                            gr.HTML('<p class="p-panel-title">New to Pallor?</p><p class="p-panel-text">Run the real '
                                    'pipeline on a synthetic test image. No image of your own is needed.</p>')
                            try_demo = gr.Button("Try the demo", variant="secondary", elem_id="p-try-demo", size="md")
                        with gr.Column(elem_id="p-demo-note", visible=False) as demo_note:
                            gr.HTML('<p class="p-panel-title"><span class="p-tag">Synthetic</span> Demo image loaded</p>'
                                    '<p class="p-panel-text">A code-generated test image, not a real conjunctiva. It '
                                    'demonstrates software behaviour, not medical performance.</p>')
                            exit_demo = gr.Button("Use my own image instead", variant="secondary",
                                                  elem_id="p-exit-demo", size="md")
                        gr.HTML('<p class="p-panel-title p-upload-title">Upload an image</p><ul class="p-req">'
                                '<li><span><strong>Supported:</strong> a pre-segmented conjunctiva crop (inner-eyelid '
                                'tissue only) saved as a PNG with a transparent background.</span></li>'
                                '<li class="no"><span><strong>Not supported:</strong> ordinary photographs, JPEGs, selfies '
                                'and unsegmented eye images.</span></li></ul>')
                        upload = gr.File(label="Conjunctiva PNG", file_types=[".png"], file_count="single",
                                         type="filepath", elem_id="p-upload", height=120)
                        preview = gr.Image(label="Preview", show_label=False, interactive=False, height=120,
                                           elem_id="p-preview", buttons=[], visible=False)
                        status = gr.HTML('<p class="p-upload-status">No image selected.</p>')
                        run = gr.Button("Analyse image", variant="primary", elem_id="p-analyse", interactive=False)
                    with gr.Column(scale=7, min_width=320, elem_id="p-result-col"):
                        result_card = gr.HTML(placeholder_card())
                        with gr.Accordion("Technical details", open=False, elem_id="p-tech-acc"):
                            tech = gr.HTML(TECH_PLACEHOLDER)
                details = gr.JSON(visible=False)  # raw result for the API; never displayed
                upload.change(on_upload, inputs=upload,
                              outputs=[status, preview, run, result_card, tech, details, demo_note, demo_offer],
                              api_visibility="private")
                # Demo: load an approved synthetic example into the normal upload flow (real validation and inference).
                try_demo.click(lambda: demo_path, outputs=upload, api_visibility="private")
                exit_demo.click(lambda: None, outputs=upload, api_visibility="private")
                # Visible loading state, then the real analysis, then restore the button and bring the result into view.
                run.click(busy_state, outputs=[run, result_card], api_visibility="private").then(
                    handle, inputs=upload, outputs=[preview, result_card, tech, details], api_name="analyse"
                ).then(lambda: idle_state("Analyse image"), outputs=run, api_visibility="private").then(
                    None, js=SCROLL_TO_RESULT_JS, api_visibility="private")

            with gr.Tab("Examples"):
                gr.HTML('<header class="p-intro"><h2>Research example explorer</h2><p>Synthetic test images show how '
                        'Pallor handles supported, unsupported and out-of-range inputs. They demonstrate software '
                        'behaviour, not medical performance.</p></header>'
                        '<p class="p-explorer-note">Dataset images are not shown. The dataset\'s licence and sharing '
                        'permissions are unresolved, so this app displays no dataset images, patient information, '
                        'haemoglobin values or per-image predictions.</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=300, elem_id="p-ex-col"):
                        choice = gr.Dropdown(list(examples), value=first_example, label="Choose a synthetic example")
                        note = gr.HTML(example_panel(first_example))
                        ex_run = gr.Button("Run example", variant="primary", elem_id="p-ex-run")
                        ex_preview = gr.Image(show_label=False, interactive=False, height=160, buttons=[], visible=False)
                    with gr.Column(scale=7, min_width=320):
                        ex_card = gr.HTML('<div class="p-card p-placeholder"><p class="p-state-title">Choose an example '
                                          'and select Run example</p><p>The result shows the expected behaviour, what '
                                          'actually happened and the app\'s normal output.</p></div>')
                        with gr.Accordion("Technical details", open=False, elem_id="p-ex-tech"):
                            ex_tech = gr.HTML(TECH_PLACEHOLDER)
                ex_details = gr.JSON(visible=False)
                choice.change(example_panel, inputs=choice, outputs=note, api_visibility="private")
                ex_run.click(busy_state, outputs=[ex_run, ex_card], api_visibility="private").then(
                    run_example, inputs=choice, outputs=[ex_preview, ex_card, ex_tech, ex_details],
                    api_name="synthetic_example"
                ).then(lambda: idle_state("Run example"), outputs=ex_run, api_visibility="private")

            with gr.Tab("Performance"):
                gr.HTML(dashboard_html(results))
                with gr.Accordion("Detailed result tables", open=False, elem_id="p-tables"):
                    gr.Markdown("**Held-out test (95% bootstrap CIs)** — each model scored once on the same test images.")
                    gr.Dataframe(results["test"], interactive=False, wrap=True)
                    gr.Dataframe(results["paired"], interactive=False, label="Paired test comparison vs Logistic Regression")
                    gr.Markdown("**Cross-validation — training data only, not held-out test results.**")
                    gr.Dataframe(results["nested"], interactive=False, label="Nested CV on the training split")
                    gr.Dataframe(results["other_cv"], interactive=False,
                                 label="Other CV protocols (not directly comparable with nested CV)")
                    gr.Dataframe(results["regression"], interactive=False,
                                 label="Haemoglobin regression (did not meaningfully beat predicting the mean)")
                    gr.Dataframe(results["shortcut"], interactive=False,
                                 label="Crop-size diagnostic (training CV): a possible confound")

            with gr.Tab("Limitations"):
                gr.HTML(LIMITATIONS)

        gr.Markdown(GLOBAL_NOTE, elem_id="disclaimer")
    return demo


def main():
    settings = launch_settings()
    assert settings["share"] is False
    assert settings["server_name"] == ("0.0.0.0" if os.environ.get("RENDER") == "true" else "127.0.0.1")
    demo = build_app()
    demo.launch(**settings)


if __name__ == "__main__":
    main()
