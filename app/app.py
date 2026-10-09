"""Pallor local research app (Gradio). Logistic-regression baseline only.

Run:  .venv/bin/python app/app.py      -> http://127.0.0.1:7860

Local-only by design: bound to 127.0.0.1, no share link, Gradio analytics off,
system fonts only (no web-font requests). No dataset images, Hb values or
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


def placeholder_card():
    return ('<div class="p-card p-placeholder"><h3>Your result will appear here</h3>'
            '<p>Choose a transparent-background PNG and select <strong>Analyse</strong>. '
            'The image is processed on this computer only.</p></div>')


LOADING_CARD = ('<div class="p-card p-placeholder" role="status" aria-live="polite" aria-busy="true">'
                '<h3>Analysing…</h3><p>Checking the image, extracting colour features and applying the '
                'research model.</p></div>')


TECH_PLACEHOLDER = '<p class="p-muted">Technical details appear here after an analysis.</p>'


def footnote():
    return f'<p class="p-footnote">{ICON_INFO}<span>{esc(DISCLAIMER)}</span></p>'


def influence_section():
    names = inference.default_model()["feature_names"]
    counts = {g: sum(feature_group(n) == g for n in names) for g, _ in FEATURE_GROUPS}
    items = "".join(f'<li title="{esc(desc)}">{esc(g)} <em>{counts[g]}</em></li>'
                    for g, desc in FEATURE_GROUPS if counts[g])
    return ('<section class="p-section"><h4>What influenced this result?</h4>'
            f'<p>The model evaluates {len(names)} engineered colour features measured from the visible tissue '
            'pixels of the supplied crop, grouped below. Its output reflects patterns learned from one training '
            'dataset. This version does not provide a validated medical explanation for an individual result.</p>'
            f'<ul class="p-groups" aria-label="Feature groups and counts">{items}</ul></section>')


def format_result(result, results):
    """Plain-English result card (HTML). The interpretation comes first; numbers are secondary."""
    if result.status == "rejected":
        reasons = "".join(f"<li>{esc(r)}</li>" for r in result.reasons)
        return ('<div class="p-card p-result is-rejected"><div class="p-status">'
                '<span class="p-dot"></span><span class="p-eyebrow">Not analysed</span></div>'
                "<h3>This image can't be analysed</h3>"
                f'<ul class="p-reasons">{reasons}</ul>'
                '<p class="p-muted">Pallor only supports pre-segmented conjunctiva crops saved as PNG with a '
                'transparent background, like its training images. Ordinary photographs are not supported.</p>'
                f'{footnote()}</div>')

    above = result.above_threshold
    if above:
        state, eyebrow = "is-above", "Research classification · above the threshold"
        headline = "Resembles the dataset's anaemic-labelled examples"
        explain = ("The model found colour-feature patterns that fall on the side of the research threshold "
                   "associated with images labelled anaemic in this dataset.")
    else:
        state, eyebrow = "is-below", "Research classification · below the threshold"
        headline = "Did not cross the research threshold"
        explain = ("The model's colour-feature score fell below the threshold used in this experiment. "
                   "This does not establish that the person is free from anaemia.")

    warn = ""
    if result.warnings:
        items = "".join(f"<li>{esc(w)}</li>" for w in result.warnings)
        warn = (f'<div class="p-notice" role="note">{ICON_NOTICE}<div><strong>Outside the training data — '
                f'treat this result as unreliable.</strong><ul>{items}</ul></div></div>')

    position = min(max(result.probability, 0.0), 1.0) * 100
    marker = min(max(result.threshold, 0.0), 1.0) * 100
    score = ('<section class="p-section p-score"><div class="p-score-head"><h4>Model score</h4>'
             f'<strong>{r3(result.probability)}</strong></div>'
             f'<div class="p-track" aria-hidden="true"><span class="p-track-threshold" style="left:{marker:.1f}%"></span>'
             f'<span class="p-track-dot" style="left:{position:.1f}%"></span></div>'
             f'<div class="p-track-legend"><span>Lower</span><span class="p-track-label" style="left:{marker:.1f}%">'
             'Research threshold</span><span>Higher</span></div>'
             f'<p class="p-muted">{esc(SCORE_NOTE)}</p></section>')

    base, n_test = results["baseline_test"], results["splits"]["test"]["n"]
    cm = base["confusion_matrix"]
    limits = ('<section class="p-section"><h4>Limitations of this result</h4>'
              f'<p>On {n_test} held-out test images, the model placed {pct(base["sensitivity"])} of anaemic-labelled '
              f'images ({cm["tp"]} of {cm["tp"] + cm["fn"]}) above the threshold and {pct(base["specificity"])} of '
              f'non-anaemic-labelled images ({cm["tn"]} of {cm["tn"] + cm["fp"]}) below it, so it can be wrong in '
              f'either direction.</p><p>{esc(DISCLAIMER)}</p></section>')
    next_step = (f'<section class="p-next"><h4>What should I do with this result?</h4><p>{esc(NEXT_STEPS)}</p></section>')

    return (f'<div class="p-card p-result {state}" role="status" aria-live="polite"><div class="p-status">'
            f'<span class="p-dot" aria-hidden="true"></span><span class="p-eyebrow">{eyebrow}</span></div>'
            f'<h3>{esc(headline)}</h3><p class="p-lead">{esc(explain)}</p>{warn}{next_step}{score}'
            f'{influence_section()}{limits}</div>')


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
    """Instant feedback on upload, using the same validation as the analysis."""
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


def make_handler(results):
    """Returns (preview, result card HTML, technical details HTML, raw details dict)."""
    def handle(path):
        if not path:
            return (gr.update(value=None, visible=False), '<div class="p-card p-placeholder"><h3>Choose an '
                    'image first</h3><p>Select a transparent-background PNG, then press Analyse.</p>'
                    f'{footnote()}</div>', TECH_PLACEHOLDER, {})
        result = inference.analyse(path)
        ok = result.status == "ok"
        preview = gr.update(value=inference.preview_image(path) if ok else None, visible=ok)
        return preview, format_result(result, results), format_technical(result, path), result.as_dict()
    return handle


# --- Dashboard presentation ---------------------------------------------------

AXIS = ('<div class="p-bar-row p-axis-row" aria-hidden="true"><span></span><div class="p-axis">'
        '<span style="left:0%">0.5 · chance</span><span style="left:50%">0.75</span><span style="left:100%">1.0</span>'
        '</div><span></span></div>')


def minus(text):
    """Typographic minus sign for display."""
    return str(text).replace("-", "−")


def _bar(value, lo=0.5, hi=1.0):
    return min(max((value - lo) / (hi - lo), 0.0), 1.0) * 100


def dashboard_html(results):
    base, s = results["baseline_test"], results["splits"]
    cm, b = base["confusion_matrix"], base["ci95_bootstrap"]
    pos, neg = cm["tp"] + cm["fn"], cm["tn"] + cm["fp"]
    tiles = [
        ("ROC-AUC", r3(base["roc_auc"]), f"95% CI {ci(b['roc_auc'])}",
         "How well the model ranks the two dataset-labelled groups across all possible thresholds. 0.5 is chance; 1.0 is perfect."),
        ("Sensitivity", pct(base["sensitivity"]), f"{cm['tp']} of {pos} anaemic-labelled images",
         "The proportion of anaemic-labelled test images placed above the research threshold."),
        ("Specificity", pct(base["specificity"]), f"{cm['tn']} of {neg} non-anaemic-labelled images",
         "The proportion of non-anaemic-labelled test images placed below the research threshold."),
        ("Test set", f"{s['test']['n']}", f"{pos} anaemic-labelled · {neg} not",
         f"Images held out before training and scored once. A further {s['train']['n']} images were used for training."),
    ]
    tile_html = "".join(f'<div class="p-tile"><span class="p-tile-label">{esc(t)}</span><span class="p-tile-value">'
                        f'{esc(v)}</span><span class="p-tile-sub">{esc(sub)}</span><p>{esc(d)}</p></div>'
                        for t, v, sub, d in tiles)

    rows = ""
    for m in results["raw"]["test_models"]:
        lo, hi = m["ci"] or (m["auc"], m["auc"])
        primary = " is-primary" if m["role"].startswith("Primary") else ""
        rows += (f'<div class="p-bar-row{primary}"><div class="p-bar-label"><strong>{esc(m["model"])}</strong>'
                 f'<span>{esc(m["role"])}</span></div><div class="p-bar-track">'
                 f'<span class="p-bar-fill" style="width:{_bar(m["auc"]):.1f}%"></span>'
                 f'<span class="p-bar-ci" style="left:{_bar(lo):.1f}%;width:{_bar(hi) - _bar(lo):.1f}%"></span></div>'
                 f'<div class="p-bar-value">{r3(m["auc"])}<small>{ci((lo, hi))}</small></div></div>')
    et = results["raw"]["et_pair"]
    established = et["ci"][0] > 0
    compare = (f'<div class="p-chart"><p class="p-chart-title">Held-out test ROC-AUC with 95% bootstrap interval '
               f'(n = {s["test"]["n"]})</p>{rows}{AXIS}</div>'
               f'<p class="p-muted">Bars show ROC-AUC; thin lines show 95% bootstrap intervals. Extra Trees scored '
               f'higher, but the paired difference ({"+" if et["delta"] >= 0 else ""}{minus(r3(et["delta"]))}, 95% CI '
               f'{minus(ci(et["ci"]))}) {"excludes" if established else "includes"} zero, so an improvement is '
               f'{"supported" if established else "not established"}. Logistic Regression remains the model used here.</p>')

    cv_rows = ""
    for m in results["raw"]["nested"]:
        primary = " is-primary" if m["model"] == "Logistic Regression" else ""
        cv_rows += (f'<div class="p-bar-row compact{primary}"><div class="p-bar-label"><strong>{esc(m["model"])}</strong>'
                    f'</div><div class="p-bar-track"><span class="p-bar-fill muted" style="width:{_bar(m["mean"]):.1f}%">'
                    f'</span></div><div class="p-bar-value">{r3(m["mean"])}<small>± {r3(m["sd"])}</small></div></div>')

    return f"""
<div class="p-page">
  <header class="p-intro">
    <h2>Research performance</h2>
    <p>The model was scored once on {s['test']['n']} held-out test images that were set aside before training.
    These figures describe how often its dataset classifications matched the dataset's own labels.
    They are not measures of clinical accuracy.</p>
  </header>
  <section>
    <div class="p-section-head"><h3>Held-out test results</h3><span class="p-pill">Logistic Regression · used by this app</span></div>
    <div class="p-tiles">{tile_html}</div>
    <p class="p-callout">{ICON_INFO}<span>The evaluation uses a small, single-source dataset. These results do not
    establish clinical performance or generalisation to new patients. The same test images were used for three
    one-shot evaluations, so they are no longer an untouched final test.</span></p>
  </section>
  <section>
    <div class="p-section-head"><h3>Compared on the same test images</h3></div>
    {compare}
  </section>
  <section class="p-separated">
    <div class="p-section-head"><h3>Cross-validation</h3><span class="p-pill quiet">Training data only · not test results</span></div>
    <p class="p-muted">Nested cross-validation on the {s['train']['n']} training images: every model was tuned and
    scored on the same {results['raw']['n_outer_folds']} outer folds. These estimates guided model selection and must
    not be read as held-out test results.</p>
    <div class="p-chart"><p class="p-chart-title">Mean nested-CV ROC-AUC ± standard deviation across
    {results['raw']['n_outer_folds']} outer folds</p>{cv_rows}{AXIS}</div>
  </section>
</div>"""


LIMITATION_ITEMS = [
    ("Not a medical device", "Pallor is a research prototype. It cannot diagnose anaemia or rule it out."),
    ("Cannot measure haemoglobin", "It classifies colour features against a dataset label. Only a blood test measures haemoglobin."),
    ("Scores are not probabilities", "The model score is not a calibrated probability of anaemia."),
    ("Small dataset", "A few hundred images in total, with a small held-out test set, so uncertainty is wide."),
    ("Single source", "All images come from one dataset whose licence and provenance are still being verified."),
    ("Possible confounding", "Crop size alone predicts the labels better than chance, and lighting and cameras vary. "
                             "The model may learn such shortcuts rather than pallor."),
    ("Unknown patient independence", "No patient identifiers exist, so one person's images could appear in both "
                                     "training and test data, inflating reported performance."),
    ("Strict input requirement", "Only pre-segmented conjunctiva crops saved as transparent-background PNGs are supported."),
    ("No external validation", "The model has not been tested on other people, cameras, lighting, skin tones, ages or populations."),
    ("No clinical validation", "No prospective study or comparison with clinicians has been carried out."),
]

LIMITATIONS = (
    f'<div class="p-page"><header class="p-intro"><h2>Limitations</h2><p>{esc(DISCLAIMER)}</p></header>'
    '<div class="p-limits">'
    + "".join(f'<div class="p-limit"><h4>{esc(t)}</h4><p>{esc(d)}</p></div>' for t, d in LIMITATION_ITEMS)
    + '</div><p class="p-muted">The anaemic label means Hb &lt; 11 g/dL in this dataset — a labelling convention, '
      'not a universal diagnostic threshold. Full methodology: <code>docs/REPORT.md</code> and '
      '<code>models/README.md</code>.</p></div>'
)


# --- Theme and styling (system fonts only: no external font requests) ---------

SYSTEM_FONT = ("-apple-system", "BlinkMacSystemFont", "SF Pro Text", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif")
MONO_FONT = ("SF Mono", "ui-monospace", "Menlo", "Consolas", "monospace")

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.stone,
    font=SYSTEM_FONT, font_mono=MONO_FONT,
).set(
    body_background_fill="#F6F6F4", body_background_fill_dark="#121214",
    body_text_color="#1B1B1D", body_text_color_dark="#EDEDEF",
    body_text_color_subdued="#5C5C62", body_text_color_subdued_dark="#A6A6AD",
    background_fill_primary="#FFFFFF", background_fill_primary_dark="#1B1B1E",
    background_fill_secondary="#F6F6F4", background_fill_secondary_dark="#121214",
    block_background_fill="#FFFFFF", block_background_fill_dark="#1B1B1E",
    block_border_color="#E3E2DE", block_border_color_dark="#2E2E33", block_border_width="1px",
    block_shadow="none", block_radius="12px", container_radius="12px", input_radius="6px",
    button_large_radius="6px", button_small_radius="6px",
    block_label_text_color="#5C5C62", block_label_text_color_dark="#A6A6AD",
    border_color_primary="#E3E2DE", border_color_primary_dark="#2E2E33",
    input_border_color_focus="#1F6E78", input_border_color_focus_dark="#79C3CC",
    button_primary_background_fill="#1F6E78", button_primary_background_fill_hover="#195C64",
    button_primary_background_fill_dark="#79C3CC", button_primary_background_fill_hover_dark="#8FCFD7",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#0E2326",
    button_secondary_background_fill="#EFEEEA", button_secondary_background_fill_dark="#232327",
    color_accent="#1F6E78", color_accent_soft="#E6F0F1", color_accent_soft_dark="#19302F",
    input_background_fill="#FFFFFF", input_background_fill_dark="#1B1B1E",
    link_text_color="#1F6E78", link_text_color_dark="#79C3CC",
)

CSS = """
/* ---- Design tokens ------------------------------------------------------
   Spacing: 8-point scale (4 for hairline adjustments only).
   Type: 12 / 13 / 15 / 17 / 20 / 28 / 36 px; weights 400 / 500 / 600.
   Radii: 6 (small controls, tags) and 12 (panels). No shadows, no motion. */
:root {
  --s-half: 4px; --s-1: 8px; --s-2: 16px; --s-3: 24px; --s-4: 32px; --s-5: 40px; --s-6: 48px; --s-8: 64px;
  --fs-cap: 12px; --fs-meta: 13px; --fs-body: 15px; --fs-lead: 17px; --fs-h3: 20px; --fs-h2: 28px; --fs-h1: 36px;
  --lh-body: 1.55; --lh-tight: 1.25;
  --r-sm: 6px; --r-md: 12px;
  --c-bg: #F6F6F4; --c-surface: #FFFFFF; --c-text: #1B1B1D; --c-muted: #5C5C62; --c-line: #E3E2DE;
  --c-soft: #EFEEEA; --c-accent: #1F6E78; --c-accent-soft: #E6F0F1; --c-warn: #8A5A12; --c-warn-soft: #FBF3E4;
}
.dark {
  --c-bg: #121214; --c-surface: #1B1B1E; --c-text: #EDEDEF; --c-muted: #A6A6AD; --c-line: #2E2E33;
  --c-soft: #232327; --c-accent: #79C3CC; --c-accent-soft: #19302F; --c-warn: #E6BC74; --c-warn-soft: #2A2218;
}

/* ---- Page frame ---------------------------------------------------------- */
.gradio-container { max-width: 1120px !important; margin: 0 auto !important; padding: 0 var(--s-4) var(--s-8) !important;
  -webkit-font-smoothing: antialiased; overflow-x: hidden; }
footer { display: none !important; }
.html-container { padding: 0 !important; }
.p-card, .p-tech, .p-page, .p-req { overflow-wrap: anywhere; }
.p-icon { flex: none; width: 16px; height: 16px; margin-top: 2px; }

/* Focus: always visible for keyboard users */
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, select:focus-visible,
[tabindex]:focus-visible, a:focus-visible, label:focus-within {
  outline: 2px solid var(--c-accent) !important; outline-offset: 2px !important; }

/* ---- Header and navigation ---------------------------------------------- */
#p-header { padding: var(--s-5) 0 0; }
.p-brand { display: flex; align-items: baseline; gap: var(--s-2); flex-wrap: wrap; }
.p-wordmark { font-size: var(--fs-h3); font-weight: 600; letter-spacing: -0.02em; color: var(--c-text); }
.p-subtitle { font-size: var(--fs-meta); color: var(--c-muted); }
#disclaimer { margin: var(--s-1) 0 0; padding: 0 !important; }
#disclaimer p { display: flex; gap: var(--s-1); margin: 0; max-width: 760px; color: var(--c-muted);
  font-size: var(--fs-meta); line-height: var(--lh-body); }
#disclaimer p::before { content: ""; flex: none; width: 8px; height: 8px; margin-top: 6px; border-radius: 50%;
  border: 1.5px solid currentColor; }
.tabs { margin-top: var(--s-3); }
/* Spacing lives inside each tab (padding) because Gradio's fit check sums button widths and ignores gap. */
[role="tablist"], .tab-container { gap: 0 !important; margin-left: calc(-1 * var(--s-1)) !important; }
[role="tablist"] { border-bottom: 1px solid var(--c-line) !important; }
/* Style both the visible tabs and Gradio's hidden measuring copy, so its overflow logic measures true widths. */
.tab-container button { padding: var(--s-1) var(--s-1) !important; margin-right: var(--s-1) !important; background: transparent !important; border: none !important;
  border-bottom: 2px solid transparent !important; border-radius: 0 !important; font-size: var(--fs-body) !important;
  font-weight: 500 !important; color: var(--c-muted) !important; }
[role="tab"]:hover { color: var(--c-text) !important; }
[role="tab"][aria-selected="true"] { color: var(--c-text) !important; border-bottom-color: var(--c-accent) !important; }
.tabitem { padding: 0 !important; border: none !important; background: transparent !important; }

/* ---- Page intro (shared by every tab) ----------------------------------- */
.p-intro { padding: var(--s-5) 0 var(--s-3); max-width: 720px; }
.p-intro h1, .p-intro h2 { margin: 0 0 var(--s-1); font-size: var(--fs-h1); line-height: var(--lh-tight); font-weight: 600;
  letter-spacing: -0.025em; color: var(--c-text); }
.p-intro p { margin: 0; font-size: var(--fs-lead); line-height: var(--lh-body); color: var(--c-muted); }

/* ---- Analyse: input column ----------------------------------------------- */
#p-upload-col, #p-result-col { gap: var(--s-2); }
.p-req { list-style: none; margin: 0; padding: 0; display: grid; gap: var(--s-1); }
.p-req li { display: flex; gap: var(--s-1); font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-muted); }
.p-req li::before { content: ""; flex: none; width: 6px; height: 6px; margin-top: 9px; border-radius: 50%; background: var(--c-accent); }
.p-req li.no::before { background: transparent; border: 1.5px solid var(--c-muted); width: 5px; height: 5px; }
.p-req strong { font-weight: 600; color: var(--c-text); }
#p-upload { border: 1px dashed var(--c-muted) !important; border-radius: var(--r-md) !important; background: var(--c-surface) !important; }
#p-upload:hover { border-color: var(--c-accent) !important; }
#p-analyse { min-height: 48px; border-radius: var(--r-sm) !important; font-size: var(--fs-body) !important; font-weight: 600 !important; }
#p-analyse:disabled { opacity: .6; cursor: progress; }
.p-upload-status { display: flex; gap: var(--s-1); margin: 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-upload-status.is-ready { color: var(--c-text); }
.p-upload-status.is-ready .p-dot { margin-top: 6px; background: var(--c-accent); }
.p-upload-status.is-error { color: var(--c-warn); }
#p-preview { border-radius: var(--r-md) !important; overflow: hidden; }

/* ---- Result card --------------------------------------------------------- */
.p-card { background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--r-md); padding: var(--s-4); }
.p-placeholder { padding: var(--s-6) var(--s-4); text-align: center; background: transparent; border-style: dashed; }
.p-placeholder h3 { margin: 0 0 var(--s-1); font-size: var(--fs-lead); font-weight: 600; color: var(--c-text); }
.p-placeholder p { margin: 0 auto; max-width: 400px; font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-muted); }
.p-status { display: flex; align-items: center; gap: var(--s-1); margin-bottom: var(--s-1); }
.p-dot { flex: none; width: 8px; height: 8px; border-radius: 50%; background: var(--c-muted); }
.p-eyebrow { font-size: var(--fs-meta); font-weight: 500; color: var(--c-muted); }
.p-result h3 { margin: 0 0 var(--s-1); font-size: var(--fs-h2); line-height: var(--lh-tight); font-weight: 600;
  letter-spacing: -0.02em; color: var(--c-text); }
/* Status: filled accent = above threshold; hollow = below; grey = not analysed */
.p-result.is-above .p-dot { background: var(--c-accent); }
.p-result.is-below .p-dot { background: transparent; border: 2px solid var(--c-accent); width: 6px; height: 6px; }
.p-lead { margin: 0; font-size: var(--fs-lead); line-height: var(--lh-body); color: var(--c-text); }
.p-muted { margin: var(--s-1) 0 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-reasons { margin: var(--s-1) 0 var(--s-2); padding-left: var(--s-3); font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-text); }
.p-notice { display: flex; gap: var(--s-1); margin: var(--s-3) 0 0; padding: var(--s-2); border-radius: var(--r-sm);
  background: var(--c-warn-soft); color: var(--c-warn); font-size: var(--fs-meta); line-height: var(--lh-body); }
.p-notice ul { margin: var(--s-half) 0 0; padding-left: var(--s-2); color: var(--c-text); }
.p-next { margin: var(--s-3) 0 0; padding: var(--s-2) var(--s-2) var(--s-2) var(--s-2); border-left: 2px solid var(--c-accent);
  background: var(--c-accent-soft); border-radius: 0 var(--r-sm) var(--r-sm) 0; }
.p-next h4 { margin: 0 0 var(--s-half); font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-next p { margin: 0; font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-text); }
.p-section { margin-top: var(--s-3); padding-top: var(--s-3); border-top: 1px solid var(--c-line); }
.p-section h4 { margin: 0 0 var(--s-1); font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-section > p { margin: 0 0 var(--s-1); font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-muted); }
.p-score-head { display: flex; justify-content: space-between; align-items: baseline; gap: var(--s-2); }
.p-score-head h4 { margin: 0; }
.p-score-head strong { font-size: var(--fs-h3); font-weight: 600; font-variant-numeric: tabular-nums; color: var(--c-text); }
.p-track { position: relative; height: 4px; margin: var(--s-2) 0 var(--s-1); border-radius: 2px; background: var(--c-soft); }
.p-track-threshold { position: absolute; top: -6px; width: 2px; height: 16px; margin-left: -1px; background: var(--c-muted); }
.p-track-dot { position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%;
  background: var(--c-surface); border: 2px solid var(--c-text); }
.p-track-legend { position: relative; display: flex; justify-content: space-between; font-size: var(--fs-cap); color: var(--c-muted); }
.p-track-label { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; }
.p-score .p-muted { margin-top: var(--s-2); }
.p-page .p-section-head + .p-muted, .p-separated > .p-muted { margin: 0 0 var(--s-2); }
.p-groups { display: flex; flex-wrap: wrap; gap: var(--s-1); list-style: none; margin: var(--s-1) 0 0; padding: 0; }
.p-groups li { padding: var(--s-half) var(--s-1); border: 1px solid var(--c-line); border-radius: var(--r-sm);
  font-size: var(--fs-meta); color: var(--c-text); }
.p-groups em { font-style: normal; color: var(--c-muted); font-variant-numeric: tabular-nums; }
.p-footnote { display: flex; gap: var(--s-1); margin: var(--s-3) 0 0; padding-top: var(--s-2); border-top: 1px solid var(--c-line);
  font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }

/* ---- Technical details (secondary) -------------------------------------- */
#p-tech-acc, #p-tables { border-radius: var(--r-md) !important; }
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

/* ---- Dashboard ----------------------------------------------------------- */
.p-page section { margin-bottom: var(--s-5); }
.p-section-head { display: flex; align-items: baseline; justify-content: space-between; gap: var(--s-2); flex-wrap: wrap; margin-bottom: var(--s-2); }
.p-section-head h3 { margin: 0; font-size: var(--fs-h3); font-weight: 600; letter-spacing: -0.01em; color: var(--c-text); }
.p-pill { padding: var(--s-half) var(--s-1); border: 1px solid var(--c-line); border-radius: var(--r-sm);
  font-size: var(--fs-meta); color: var(--c-muted); }
.p-pill.quiet { color: var(--c-muted); }
.p-tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: var(--c-line);
  border: 1px solid var(--c-line); border-radius: var(--r-md); overflow: hidden; }
.p-tile { display: flex; flex-direction: column; gap: var(--s-half); padding: var(--s-3); background: var(--c-surface); }
.p-tile-label { font-size: var(--fs-meta); font-weight: 500; color: var(--c-muted); }
.p-tile-value { font-size: var(--fs-h2); font-weight: 600; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; color: var(--c-text); }
.p-tile-sub { font-size: var(--fs-meta); color: var(--c-text); }
.p-tile p { margin: var(--s-1) 0 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-callout { display: flex; gap: var(--s-1); margin: var(--s-2) 0 0; font-size: var(--fs-meta); line-height: var(--lh-body); color: var(--c-muted); }
.p-chart { padding: var(--s-1) var(--s-3) var(--s-2); background: var(--c-surface); border: 1px solid var(--c-line); border-radius: var(--r-md); }
.p-chart-title { margin: var(--s-1) 0 0; font-size: var(--fs-meta); font-weight: 500; color: var(--c-muted); }
.p-bar-row { display: grid; grid-template-columns: 224px minmax(0, 1fr) 120px; gap: var(--s-2); align-items: center;
  padding: var(--s-2) 0; border-bottom: 1px solid var(--c-line); }
.p-bar-row.compact { padding: var(--s-1) 0; }
.p-bar-label strong { display: block; font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-bar-label span { font-size: var(--fs-meta); color: var(--c-muted); }
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
.p-separated { padding-top: var(--s-5); border-top: 1px solid var(--c-line); }

/* ---- Explorer and limitations -------------------------------------------- */
.p-explorer-note { margin: 0 0 var(--s-3); max-width: 720px; font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-muted); }
.p-limits { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); column-gap: var(--s-5); margin-bottom: var(--s-3); }
.p-limit { padding: var(--s-2) 0; border-top: 1px solid var(--c-line); }
.p-limit h4 { margin: 0 0 var(--s-half); font-size: var(--fs-body); font-weight: 600; color: var(--c-text); }
.p-limit p { margin: 0; font-size: var(--fs-body); line-height: var(--lh-body); color: var(--c-muted); }

/* ---- Narrow screens ------------------------------------------------------ */
@media (max-width: 860px) {
  .gradio-container { padding: 0 var(--s-2) var(--s-6) !important; }
  .p-intro { padding-top: var(--s-4); }
  .p-intro h1, .p-intro h2 { font-size: var(--fs-h2); }
  .p-card { padding: var(--s-3); }
  .p-result h3 { font-size: var(--fs-h3); }
  .p-tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .p-limits { grid-template-columns: 1fr; }
  .p-bar-row { grid-template-columns: minmax(0, 1fr) 96px; }
  .p-bar-track { grid-column: 1 / -1; grid-row: 2; }
  .p-axis-row > span { display: none; }
  .p-axis { grid-column: 1 / -1; }
  .tab-container button { font-size: var(--fs-meta) !important; margin-right: 0 !important; }
}
@media (max-width: 480px) { .p-tiles { grid-template-columns: 1fr; } }
"""

LAUNCH_KWARGS = {"server_name": "127.0.0.1", "server_port": 7860, "share": False, "inbrowser": False,
                 "show_error": True, "max_file_size": "10mb", "enable_monitoring": False,
                 "mcp_server": False, "pwa": False, "ssr_mode": False,
                 "theme": THEME, "css": CSS, "footer_links": []}


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
    handle = make_handler(results)

    with gr.Blocks(title="Pallor — conjunctival colour research", analytics_enabled=False,
                   delete_cache=(3600, 3600)) as demo:
        gr.HTML('<div class="p-brand"><span class="p-wordmark">Pallor</span>'
                '<span class="p-subtitle">Conjunctival colour research</span></div>', elem_id="p-header")
        gr.Markdown(DISCLAIMER, elem_id="disclaimer")
        n_features = len(inference.default_model()["feature_names"])

        with gr.Tabs():
            with gr.Tab("Analyse"):
                gr.HTML('<header class="p-intro"><h1>Conjunctiva colour analysis</h1><p>Upload a pre-segmented '
                        f'conjunctiva crop. Pallor measures {n_features} colour features from the visible tissue and '
                        'compares them with patterns a research model learned from one labelled dataset.</p></header>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320, elem_id="p-upload-col"):
                        gr.HTML('<ul class="p-req"><li><span><strong>Pre-segmented conjunctiva crop</strong> — only '
                                'the inner-eyelid tissue is visible.</span></li><li><span><strong>PNG with a transparent '
                                'background</strong>, like the training images.</span></li><li class="no"><span>Ordinary '
                                'photographs are not supported.</span></li></ul>')
                        upload = gr.File(label="Conjunctiva PNG", file_types=[".png"], file_count="single",
                                         type="filepath", elem_id="p-upload", height=150)
                        status = gr.HTML('<p class="p-upload-status">No image selected.</p>')
                        run = gr.Button("Analyse", variant="primary", elem_id="p-analyse")
                        preview = gr.Image(label="Preview", show_label=False, interactive=False, height=170,
                                           elem_id="p-preview", buttons=[], visible=False)
                    with gr.Column(scale=7, min_width=360, elem_id="p-result-col"):
                        result_card = gr.HTML(placeholder_card())
                        with gr.Accordion("Technical details", open=False, elem_id="p-tech-acc"):
                            tech = gr.HTML(TECH_PLACEHOLDER)
                details = gr.JSON(visible=False)  # raw result for the API; never displayed
                upload.change(upload_status, inputs=upload, outputs=[status, preview], api_visibility="private")
                upload.change(lambda _: (placeholder_card(), TECH_PLACEHOLDER, {}), inputs=upload,
                              outputs=[result_card, tech, details], api_visibility="private")
                # Visible loading state: disable the button and show a working message, then restore.
                run.click(busy_state, outputs=[run, result_card], api_visibility="private").then(
                    handle, inputs=upload, outputs=[preview, result_card, tech, details], api_name="analyse"
                ).then(lambda: idle_state("Analyse"), outputs=run, api_visibility="private")

            with gr.Tab("Examples"):
                gr.HTML('<header class="p-intro"><h2>Research example explorer</h2><p>Synthetic images that show how '
                        'Pallor handles valid, unsupported and out-of-range inputs. They demonstrate input validation, '
                        'not medical performance.</p></header>')
                gr.HTML('<p class="p-explorer-note">Dataset examples are not shown. The dataset\'s licence and sharing '
                        'permissions are unresolved, so this app displays no dataset images, patient information, '
                        'haemoglobin values or per-image predictions. Scores for these synthetic images have no '
                        'medical meaning.</p>')
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        choice = gr.Dropdown(list(examples), value=next(iter(examples)), label="Synthetic example")
                        note = gr.Markdown(examples[next(iter(examples))][1])
                        ex_run = gr.Button("Run example", variant="primary")
                        ex_preview = gr.Image(show_label=False, interactive=False, height=170, buttons=[], visible=False)
                    with gr.Column(scale=7, min_width=360):
                        ex_card = gr.HTML(placeholder_card())
                        with gr.Accordion("Technical details", open=False):
                            ex_tech = gr.HTML(TECH_PLACEHOLDER)
                ex_details = gr.JSON(visible=False)
                choice.change(lambda c: examples[c][1], inputs=choice, outputs=note, api_visibility="private")
                ex_run.click(busy_state, outputs=[ex_run, ex_card], api_visibility="private").then(
                    lambda c: handle(examples[c][0]), inputs=choice,
                    outputs=[ex_preview, ex_card, ex_tech, ex_details], api_name="synthetic_example"
                ).then(lambda: idle_state("Run example"), outputs=ex_run, api_visibility="private")

            with gr.Tab("Performance"):
                gr.HTML(dashboard_html(results))
                with gr.Accordion("All result tables", open=False, elem_id="p-tables"):
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
    return demo


def main():
    demo = build_app()
    assert LAUNCH_KWARGS["server_name"] == "127.0.0.1" and LAUNCH_KWARGS["share"] is False
    demo.launch(**LAUNCH_KWARGS)


if __name__ == "__main__":
    main()
