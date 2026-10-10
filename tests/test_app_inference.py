"""Tests for the Pallor local research app (logistic-regression baseline only).

The parity tests read local, gitignored files (data/, derived/, the baseline
snapshot's predictions) and skip if they are absent. This file contains no
dataset images, image IDs, Hb values or predictions. No experiment or one-shot
evaluation script is run.
"""

import hashlib
import importlib.util
import json
import re
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import inference  # noqa: E402

spec = importlib.util.spec_from_file_location("pallor_app", ROOT / "app" / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

HAVE_PRIVATE = all(p.exists() for p in [ROOT / "data", ROOT / "derived" / "split.csv",
                                         ROOT / "outputs" / "baseline_colour" / "predictions.csv"])
needs_private = pytest.mark.skipif(not HAVE_PRIVATE, reason="local gitignored dataset files not present")


# --- helpers -----------------------------------------------------------------

def crop_png(path, w=220, h=80, mode="RGBA", seed=0):
    rng = np.random.default_rng(seed)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((2, 2, w - 3, h - 3), fill=255)
    rgb = np.clip(np.array([196, 118, 112]) + rng.normal(0, 6, (h, w, 3)), 0, 255).astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb, np.asarray(mask)]), "RGBA")
    (img.convert("RGB") if mode == "RGB" else img).save(path, format="PNG")
    return path


def held_out_rows():
    manifest = pd.read_csv(ROOT / "derived" / "manifest.csv")
    split = pd.read_csv(ROOT / "derived" / "split.csv")
    return manifest[split["split"] == "test"]


def protected_digest():
    files = []
    for top in ("data", "derived", "models", "outputs", "scripts"):
        files += [p for p in (ROOT / top).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.name != ".DS_Store"]
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(str(p.relative_to(ROOT)).encode() + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest(), len(files)


# --- model bundle and feature extraction -------------------------------------

def test_model_bundle_integrity_and_structure():
    bundle = inference.load_model()
    assert inference.sha256_file(inference.MODEL_PATH) == inference.expected_model_sha256()
    assert len(bundle["feature_names"]) == 35
    stored = json.loads((ROOT / "outputs" / "cv_results.json").read_text())["classification"]["threshold"]
    assert bundle["threshold"] == stored  # the app uses the saved threshold, not a new one


def test_tampered_model_is_refused(tmp_path):
    copy = tmp_path / "classifier.joblib"
    data = bytearray(inference.MODEL_PATH.read_bytes())
    data[-1] ^= 0xFF
    copy.write_bytes(bytes(data))
    with pytest.raises(inference.ModelIntegrityError):
        inference.load_model(copy)


@needs_private
def test_feature_order_matches_training(tmp_path):
    bundle = inference.default_model()
    cached_columns = list(pd.read_csv(ROOT / "derived" / "features.csv", nrows=1).columns[1:])
    assert bundle["feature_names"] == cached_columns
    feats = inference.features.extract(crop_png(tmp_path / "c.png"))
    assert set(feats) == set(bundle["feature_names"])


@needs_private
def test_parity_with_saved_baseline_predictions():
    """Raw held-out PNGs -> app pipeline must reproduce the stored baseline predictions."""
    saved = pd.read_csv(ROOT / "outputs" / "baseline_colour" / "predictions.csv")
    saved = saved[saved["split"] == "test"].set_index("image_id")
    rows = held_out_rows()
    assert len(rows) == 51
    worst, mismatched_decisions = 0.0, 0
    for image_id, path in zip(rows["image_id"], rows["path"]):
        res = inference.analyse(ROOT / path)
        assert res.status == "ok"
        worst = max(worst, abs(res.probability - saved.loc[image_id, "prob_anemic"]))
        mismatched_decisions += int(res.above_threshold) != int(saved.loc[image_id, "pred_anemic"])
    assert worst < 1e-12, f"max |Δp| = {worst}"
    assert mismatched_decisions == 0


@needs_private
def test_features_match_cached_training_features():
    cached = pd.read_csv(ROOT / "derived" / "features.csv").set_index("image_id")
    rows = held_out_rows().head(10)
    for image_id, path in zip(rows["image_id"], rows["path"]):
        live = inference.features.extract(ROOT / path)
        ref = cached.loc[image_id]
        assert max(abs(live[k] - ref[k]) for k in ref.index) < 1e-9


# --- input validation ----------------------------------------------------------

def test_valid_synthetic_crop_is_accepted(tmp_path):
    res = inference.analyse(crop_png(tmp_path / "ok.png"))
    assert res.status == "ok" and 0.0 <= res.probability <= 1.0
    assert not any("geometry" in w or "Tissue-pixel" in w for w in res.warnings)


def test_palette_png_with_transparency_is_accepted(tmp_path):
    img = Image.new("P", (220, 80), 0)
    img.putpalette([0, 0, 0, 196, 118, 112] + [0] * 762)
    ImageDraw.Draw(img).ellipse((2, 2, 217, 77), fill=1)
    img.save(tmp_path / "pal.png", transparency=0)  # palette index 0 is transparent
    with Image.open(tmp_path / "pal.png") as im:
        assert im.mode == "P" and "transparency" in im.info
    assert inference.validate_png(tmp_path / "pal.png")[0] == []
    assert inference.analyse(tmp_path / "pal.png").status == "ok"


@pytest.mark.parametrize("case, expected", [
    ("rgb", "no transparency"),
    ("opaque", "no fully transparent pixels"),
    ("transparent", "fully opaque (tissue) pixels"),
    ("tiny", "fully opaque (tissue) pixels"),
    ("jpeg_renamed", "not a PNG"),
    ("truncated", "could not be decoded"),
    ("empty", "empty"),
    ("random_bytes", "not a PNG"),
    ("too_large_side", "maximum supported side"),
    ("animated", "Animated"),
    ("missing", "No file"),
])
def test_unsupported_inputs_are_rejected_with_reason(tmp_path, case, expected):
    p = tmp_path / f"{case}.png"
    if case == "rgb":
        crop_png(p, mode="RGB")
    elif case == "opaque":
        Image.new("RGBA", (200, 80), (196, 118, 112, 255)).save(p)
    elif case == "transparent":
        Image.new("RGBA", (200, 80), (0, 0, 0, 0)).save(p)
    elif case == "tiny":
        img = Image.new("RGBA", (60, 40), (0, 0, 0, 0))
        img.paste((196, 118, 112, 255), (20, 10, 28, 18))
        img.save(p)
    elif case == "jpeg_renamed":
        Image.new("RGB", (200, 80), (196, 118, 112)).save(p, format="JPEG")
    elif case == "truncated":
        crop_png(tmp_path / "full.png")
        p.write_bytes((tmp_path / "full.png").read_bytes()[:200])
    elif case == "empty":
        p.write_bytes(b"")
    elif case == "random_bytes":
        p.write_bytes(np.random.default_rng(1).bytes(4096))
    elif case == "too_large_side":
        Image.new("RGBA", (5000, 10), (0, 0, 0, 0)).save(p)
    elif case == "animated":
        frames = [Image.new("RGBA", (50, 50), (0, 0, 0, 0)), Image.new("RGBA", (50, 50), (200, 100, 100, 255))]
        frames[0].save(p, save_all=True, append_images=frames[1:])
    res = inference.analyse(p)
    assert res.status == "rejected" and res.probability is None
    assert any(expected.lower() in r.lower() for r in res.reasons), res.reasons


def test_out_of_range_geometry_and_tissue_pixels_warn(tmp_path):
    res = inference.analyse(crop_png(tmp_path / "big.png", w=1400, h=500))
    assert res.status == "ok"
    assert any("geometry" in w for w in res.warnings)
    assert any("Tissue-pixel count" in w for w in res.warnings)
    md = app.format_result(res, app.load_results())
    assert "unreliable" in md and 'class="p-notice"' in md  # prominent notice block (was an emoji)


def test_different_inputs_give_different_scores(tmp_path):
    a = inference.analyse(crop_png(tmp_path / "a.png", seed=1)).probability
    b = inference.analyse(crop_png(tmp_path / "b.png", w=300, h=110, seed=2)).probability
    assert a != b


# --- dashboard ---------------------------------------------------------------

def test_dashboard_reads_actual_result_files():
    r = app.load_results()
    base = json.loads((ROOT / "outputs" / "metrics.json").read_text())["test"]["classification"]
    et = json.loads((ROOT / "outputs" / "classical_metrics.json").read_text())["candidate_metrics"]
    test = r["test"].set_index("Model")
    assert test.loc["Logistic Regression", "ROC-AUC"] == app.r3(base["roc_auc"])
    assert test.loc["Extra Trees", "Sensitivity"] == app.r3(et["sensitivity"])
    nested = json.loads((ROOT / "outputs" / "boosting_cv_results.json").read_text())["models"]["catboost"]
    assert r["nested"].set_index("Model").loc["CatBoost", "Nested-CV AUC"] == app.r3(nested["nested_cv_auc_mean"])


def test_dashboard_values_follow_the_files_not_hardcoded(tmp_path):
    for f in app.RESULT_FILES.values():
        shutil.copy(ROOT / "outputs" / f, tmp_path / f)
    m = json.loads((tmp_path / "metrics.json").read_text())
    m["test"]["classification"]["roc_auc"] = 0.1234
    (tmp_path / "metrics.json").write_text(json.dumps(m))
    r = app.load_results(outputs_dir=tmp_path)
    assert r["test"].set_index("Model").loc["Logistic Regression", "ROC-AUC"] == "0.123"


def test_dashboard_keeps_test_and_cv_separate():
    r = app.load_results()
    assert list(r["test"]["Model"]) == ["Logistic Regression", "Extra Trees", "Frozen MobileNetV3 + LR"]
    assert "Nested-CV AUC" in r["nested"].columns and "ROC-AUC" not in r["nested"].columns
    assert len(r["nested"]) == 7


def test_no_hardcoded_metrics_or_predictions_in_app_code():
    for f in ("app.py", "inference.py"):
        src = (ROOT / "app" / f).read_text().replace('Decimal("0.001")', "")  # rounding quantum, not a metric
        assert not re.findall(r"\b0\.\d{3,}\b", src), f"metric-like literal in {f}"
        assert not re.search(r"\b(37158|1280)\b", src), f"training range hardcoded in {f}"


# --- disclaimer, privacy, launch settings ------------------------------------

def test_disclaimer_visible_globally_and_in_every_result(tmp_path):
    demo = app.build_app(example_dir=tmp_path)
    md = [b for b in demo.blocks.values() if type(b).__name__ == "Markdown"]
    top = [b for b in md if b.elem_id == "disclaimer"]
    assert len(top) == 1 and top[0].visible and "not a diagnosis" in top[0].value
    assert type(top[0].parent).__name__ != "Tab"  # shown above every tab, not inside one
    handle = app.make_handler(app.load_results())
    ok = handle(crop_png(tmp_path / "ok.png"))[1]
    bad = handle(crop_png(tmp_path / "rgb.png", mode="RGB"))[1]
    for text in (ok, bad, app.LIMITATIONS):
        assert app.DISCLAIMER in text
    assert "not a diagnosis" in ok.lower()


def test_explorer_shows_no_dataset_content(tmp_path):
    examples = app.make_synthetic_examples(tmp_path)
    data_dir = str(ROOT / "data")
    assert all(not p.startswith(data_dir) for p, _ in examples.values())
    assert all(Path(p).parent == tmp_path for p, _ in examples.values())


def test_launch_settings_are_local_only():
    k = app.LAUNCH_KWARGS
    assert k["server_name"] == "127.0.0.1" and k["share"] is False
    assert k["mcp_server"] is False and k["enable_monitoring"] is False
    import os
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"


def test_upload_through_local_server_preserves_alpha(tmp_path):
    """End to end: upload via a real local Gradio server; the handler must receive identical bytes."""
    from gradio_client import Client, handle_file
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    demo = app.build_app(example_dir=tmp_path / "ex")
    kwargs = {**app.LAUNCH_KWARGS, "server_port": port, "prevent_thread_lock": True, "quiet": True}
    demo.launch(**kwargs)
    try:
        assert demo.server_name == "127.0.0.1" and demo.share_url is None
        png = crop_png(tmp_path / "upload.png")
        direct = inference.analyse(png)
        _, card, _tech, details = Client(f"http://127.0.0.1:{port}/", verbose=False).predict(
            handle_file(str(png)), api_name="/analyse")
        assert details["info"]["input_sha256"] == inference.sha256_file(png)  # bytes unchanged
        assert details["info"]["mode"] == "RGBA"
        assert details["probability"] == direct.probability
        assert app.DISCLAIMER in card
    finally:
        demo.close()


@needs_private
def test_app_does_not_modify_protected_artifacts(tmp_path):
    before = protected_digest()
    for path in held_out_rows()["path"]:
        inference.analyse(ROOT / path)
    app.load_results()
    app.build_app(example_dir=tmp_path)
    assert protected_digest() == before


# --- result presentation (redesign) ------------------------------------------

def visible_text(markup):
    """Text a user can see: tags (and their attributes, e.g. CSS positions) removed, whitespace collapsed."""
    import html as _html
    return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", markup)).split())


def _fake(prob, above, warnings=()):
    """A presentation-only result object; values are arbitrary test inputs, not model outputs."""
    return inference.AnalysisResult(status="ok", probability=prob, threshold=0.5, above_threshold=above,
                                    warnings=list(warnings), info={})


def test_plain_english_interpretation_comes_first():
    r = app.load_results()
    above = app.format_result(_fake(0.9, True), r)
    below = app.format_result(_fake(0.1, False), r)
    assert "Resembles the dataset's anaemic-labelled examples" in visible_text(above)
    assert "Did not cross the research threshold" in below
    assert "does not establish that the person is free from anaemia" in below
    for card in (above, below):
        assert card.index("<h3>") < card.index("Model score")  # interpretation before numbers
        assert app.NEXT_STEPS in card and app.DISCLAIMER in card and "What influenced this result?" in card


def test_score_is_never_presented_as_a_probability(tmp_path):
    r = app.load_results()
    texts = [visible_text(t) for t in (app.format_result(_fake(0.862, True), r),
             app.format_result(_fake(0.2, False), r), app.dashboard_html(r), app.LIMITATIONS)]
    res = inference.analyse(crop_png(tmp_path / "ok.png"))
    texts.append(visible_text(app.format_technical(res, tmp_path / "ok.png")))
    banned = re.compile(r"(\d+(\.\d+)?\s*%\s*(chance|probability|risk))|probability of anaemia(?! *\.)"
                        r"|risk of anaemia|chance of anaemia", re.I)
    for t in texts:
        assert not banned.search(t.replace("not a calibrated probability of anaemia", "")), banned.search(t)
    assert app.SCORE_NOTE in texts[0] and "Model score" in texts[0]
    assert "86.2%" not in texts[0]


def test_technical_details_hide_hashes_and_show_threshold(tmp_path):
    p = crop_png(tmp_path / "ok.png")
    res = inference.analyse(p)
    tech = app.format_technical(res, p)
    assert res.info["input_sha256"] not in tech and "sha256" not in tech.lower()
    assert app.r3(res.threshold) in tech and app.r3(res.probability) in tech
    assert all(name in tech for name in inference.default_model()["feature_names"])
    card = app.format_result(res, app.load_results())
    assert res.info["input_sha256"] not in card


def test_dashboard_leads_with_test_metrics_read_from_files():
    r = app.load_results()
    html_out = visible_text(app.dashboard_html(r))
    base = json.loads((ROOT / "outputs" / "metrics.json").read_text())["test"]["classification"]
    cm = base["confusion_matrix"]
    assert app.r3(base["roc_auc"]) in html_out and app.pct(base["sensitivity"]) in html_out
    assert f"{cm['tp']} of {cm['tp'] + cm['fn']}" in html_out and f"{cm['tn']} of {cm['tn'] + cm['fp']}" in html_out
    assert html_out.index("Held-out test results") < html_out.index("Cross-validation")
    assert "Training data only" in html_out and "not test results" in html_out
    assert "do not establish clinical performance" in html_out


# --- guided demo (synthetic examples through the real pipeline) ---------------

def test_demo_example_exists_and_passes_validation(tmp_path):
    examples = app.make_synthetic_examples(tmp_path)
    assert app.DEMO_EXAMPLE in examples
    assert inference.validate_png(examples[app.DEMO_EXAMPLE][0])[0] == []


def test_demo_runs_actual_inference_with_neutral_synthetic_wording(tmp_path):
    examples = app.make_synthetic_examples(tmp_path)
    hashes = frozenset(inference.sha256_file(p) for p, _ in examples.values())
    path = examples[app.DEMO_EXAMPLE][0]
    _, card, tech, details = app.make_handler(app.load_results(), hashes)(path)
    direct = inference.analyse(path)
    assert details["probability"] == direct.probability            # the real model output, not simulated
    assert details["above_threshold"] == direct.above_threshold
    assert details["warnings"] == direct.warnings
    text = visible_text(card)
    assert "Synthetic" in text and "not medical performance" in text
    assert app.r3(direct.probability) in text and "Model score" in text
    assert "anaemic-labelled examples" not in text and "Did not cross" not in text   # no dataset/medical label
    assert app.r3(direct.threshold) in visible_text(tech)


def test_synthetic_detection_uses_content_not_filename(tmp_path):
    examples = app.make_synthetic_examples(tmp_path / "ex")
    hashes = frozenset(inference.sha256_file(p) for p, _ in examples.values())
    renamed = tmp_path / "valid.png"                                # same name as a synthetic example
    crop_png(renamed, w=230, h=90, seed=7)
    assert not app.is_synthetic(str(renamed), hashes)
    real_style = visible_text(app.make_handler(app.load_results(), hashes)(str(renamed))[1])
    assert "Synthetic" not in real_style and ("anaemic-labelled" in real_style or "Did not cross" in real_style)
