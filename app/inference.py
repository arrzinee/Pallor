"""Inference for the Pallor research app: the frozen logistic-regression baseline.

Nothing here retrains or alters a model. The pipeline is:
  1. Verify models/classifier.joblib against the SHA-256 recorded in the
     read-only baseline snapshot (outputs/baseline_colour/SHA256SUMS), then load it.
  2. Validate the upload: a real PNG with usable transparency, as in the
     training data (pre-segmented crops with a transparent background).
  3. Extract features with the unmodified training extractor
     (scripts/features.py) directly from the uploaded file, so the alpha
     channel is never converted away, and order them as the saved bundle does.
  4. Score with the saved pipeline and compare with the saved threshold.
  5. Warn when the input lies outside the training split's observed ranges.

Outputs are experimental research results, not diagnoses.
"""

import hashlib
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import features  # noqa: E402  - the exact training-time feature extractor

MODEL_PATH = ROOT / "models" / "classifier.joblib"
SNAPSHOT_SUMS = ROOT / "outputs" / "baseline_colour" / "SHA256SUMS"
DERIVED = ROOT / "derived"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_SIDE_PX = 4096
MIN_OPAQUE_PX = 100  # same minimum as scripts/inspect_dataset.py


class ModelIntegrityError(RuntimeError):
    """The model file does not match the checksummed baseline snapshot."""


class ReferenceDataError(RuntimeError):
    """Training-range reference data is missing or does not match the locked split."""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expected_model_sha256(sums_path=SNAPSHOT_SUMS):
    for line in Path(sums_path).read_text().splitlines():
        digest, _, name = line.partition("  ")
        if name.strip() == "classifier.joblib":
            return digest.strip()
    raise ModelIntegrityError(f"classifier.joblib not listed in {sums_path}")


def load_model(path=MODEL_PATH, expected_sha256=None):
    """Load the saved LR bundle after verifying its checksum. Never retrains."""
    expected = expected_sha256 or expected_model_sha256()
    actual = sha256_file(path)
    if actual != expected:
        raise ModelIntegrityError(f"{Path(path).name} SHA-256 {actual} != snapshot {expected}; refusing to load")
    bundle = joblib.load(path)
    if set(bundle) != {"model", "threshold", "feature_names"}:
        raise ModelIntegrityError(f"unexpected bundle keys: {sorted(bundle)}")
    pipeline = bundle["model"]
    steps = [type(s).__name__ for s in pipeline.named_steps.values()]
    if steps != ["StandardScaler", "LogisticRegression"]:
        raise ModelIntegrityError(f"unexpected pipeline steps: {steps}")
    if list(pipeline.classes_) != [0, 1] or pipeline.n_features_in_ != len(bundle["feature_names"]):
        raise ModelIntegrityError("pipeline classes or feature count do not match the bundle")
    return bundle


@dataclass(frozen=True)
class TrainingReference:
    """Observed ranges on the 200-image training split (aggregates only)."""
    width: tuple
    height: tuple
    n_tissue_px: tuple
    feature_ranges: dict


def load_training_reference(derived=DERIVED):
    split_csv, lock = Path(derived) / "split.csv", Path(derived) / "split.lock"
    if not (split_csv.exists() and lock.exists()):
        raise ReferenceDataError("derived/split.csv or split.lock is missing; run scripts/inspect_dataset.py "
                                 "and scripts/train_baseline.py locally")
    if sha256_file(split_csv) != lock.read_text().strip():
        raise ReferenceDataError("derived/split.csv does not match derived/split.lock")
    manifest = pd.read_csv(Path(derived) / "manifest.csv")
    split = pd.read_csv(split_csv)
    if list(split["image_id"]) != list(manifest["image_id"]):
        raise ReferenceDataError("split.csv does not match the manifest order")
    train = manifest[split["split"] == "train"]
    feats = pd.read_csv(Path(derived) / "features.csv").set_index("image_id").loc[train["image_id"]]
    return TrainingReference(
        width=(int(train["width"].min()), int(train["width"].max())),
        height=(int(train["height"].min()), int(train["height"].max())),
        n_tissue_px=(int(feats["n_tissue_px"].min()), int(feats["n_tissue_px"].max())),
        feature_ranges={c: (float(feats[c].min()), float(feats[c].max())) for c in feats.columns},
    )


@lru_cache(maxsize=1)
def default_model():
    return load_model()


@lru_cache(maxsize=1)
def default_reference():
    return load_training_reference()


@dataclass
class AnalysisResult:
    status: str                       # "ok" or "rejected"
    reasons: list = field(default_factory=list)   # why the input was rejected
    warnings: list = field(default_factory=list)  # out-of-range notes (status "ok" only)
    probability: float = None         # model P(label = Hb < 11 g/dL), research output
    threshold: float = None
    above_threshold: bool = None
    info: dict = field(default_factory=dict)

    def as_dict(self):
        return {"status": self.status, "reasons": self.reasons, "warnings": self.warnings,
                "probability": self.probability, "threshold": self.threshold,
                "above_threshold": self.above_threshold, "info": self.info}


def validate_png(path):
    """Return (reasons, info). An empty reasons list means the input is usable."""
    path = Path(path)
    if not path.is_file():
        return ["No file was provided."], {}
    size = path.stat().st_size
    info = {"file_bytes": size, "input_sha256": sha256_file(path)}
    if size == 0:
        return ["The file is empty."], info
    if size > MAX_FILE_BYTES:
        return [f"The file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB."], info
    with open(path, "rb") as fh:
        if fh.read(8) != PNG_SIGNATURE:
            return ["The file is not a PNG image (the PNG signature is missing), whatever its extension."], info
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                width, height = im.size
                info.update(width=width, height=height, mode=im.mode)
                if max(width, height) > MAX_SIDE_PX:
                    return [f"The image is {width}×{height} px; the maximum supported side is {MAX_SIDE_PX} px."], info
                if getattr(im, "n_frames", 1) > 1:
                    return ["Animated PNGs are not supported."], info
                im.verify()
            with Image.open(path) as im:
                im.load()
                has_alpha = "A" in im.getbands() or "transparency" in im.info
                alpha = np.asarray(im.convert("RGBA"))[..., 3] if has_alpha else None
    except Exception as exc:  # noqa: BLE001 - any decode failure means unusable input
        return [f"The PNG could not be decoded ({type(exc).__name__})."], info
    if alpha is None:
        return ["The PNG has no transparency (alpha channel). The model was trained on pre-segmented "
                "conjunctiva crops with a transparent background; without one, skin, lashes and "
                "background would be analysed as tissue."], info
    n_transparent, n_opaque = int((alpha == 0).sum()), int((alpha == 255).sum())
    info.update(n_transparent_px=n_transparent, n_opaque_px=n_opaque,
                transparent_fraction=round(n_transparent / alpha.size, 4))
    reasons = []
    if n_transparent == 0:
        reasons.append("The PNG has an alpha channel but no fully transparent pixels, so there is no "
                       "segmented background; every pixel would be treated as tissue.")
    if n_opaque < MIN_OPAQUE_PX:
        reasons.append(f"Only {n_opaque} fully opaque (tissue) pixels; at least {MIN_OPAQUE_PX} are required.")
    return reasons, info


def _range_warnings(feats, info, ref):
    out = []
    w, h = info["width"], info["height"]
    if not (ref.width[0] <= w <= ref.width[1] and ref.height[0] <= h <= ref.height[1]):
        out.append(f"Image geometry {w}×{h} px is outside the training range "
                   f"(width {ref.width[0]}–{ref.width[1]}, height {ref.height[0]}–{ref.height[1]} px).")
    n = int(feats["n_tissue_px"])
    if not ref.n_tissue_px[0] <= n <= ref.n_tissue_px[1]:
        out.append(f"Tissue-pixel count {n:,} is outside the training range "
                   f"({ref.n_tissue_px[0]:,}–{ref.n_tissue_px[1]:,}). The model includes a size feature, "
                   f"so the score may be heavily distorted.")
    colour_out = [k for k, (lo, hi) in ref.feature_ranges.items()
                  if k != "n_tissue_px" and not lo <= feats[k] <= hi]
    if colour_out:
        out.append(f"{len(colour_out)} colour feature(s) fall outside the training range "
                   f"({', '.join(colour_out)}); lighting or camera differences are likely.")
    return out


def analyse(path, bundle=None, reference=None):
    """Validate, extract features with the training extractor, and score with the saved model."""
    bundle = bundle or default_model()
    reference = reference or default_reference()
    reasons, info = validate_png(path)
    if reasons:
        return AnalysisResult(status="rejected", reasons=reasons, info=info)
    feats = features.extract(path)  # identical code path to training; reads the file's own RGBA
    names = bundle["feature_names"]
    if set(feats) != set(names):
        raise ModelIntegrityError("feature extractor output no longer matches the saved feature names")
    x = np.array([[feats[n] for n in names]], dtype=np.float64)
    probability = float(bundle["model"].predict_proba(x)[0, 1])
    threshold = float(bundle["threshold"])
    return AnalysisResult(status="ok", warnings=_range_warnings(feats, info, reference),
                          probability=probability, threshold=threshold,
                          above_threshold=bool(probability >= threshold), info=info)


def preview_image(path, tile=8):
    """Display-only RGB preview over a checkerboard; never used for features."""
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // tile + xx // tile) % 2)[..., None] == 0, 0.92, 0.78).astype(np.float32)
    rgb = rgba[..., :3] * rgba[..., 3:] + board * (1 - rgba[..., 3:])
    return (rgb * 255).round().astype(np.uint8)
