"""Shared CNN utilities: image preprocessing, augmentation, backbones.

Preprocessing (no learned parameters, so nothing is fitted to our data):
  1. RGBA -> RGB by standard alpha compositing over the ImageNet mean colour.
     RGB is stored un-premultiplied, so out = rgb * a + mean * (1 - a).
     After ImageNet normalisation the background is exactly 0, so the
     transparent region carries no colour signal and edges fade to neutral,
     not black.
  2. Letterbox-pad to a square with the same mean colour (aspect ratio
     preserved; no stretching of tissue), then resize to 224 x 224.
  3. Normalise with the fixed ImageNet mean/std.

Augmentation (training folds only) is geometric and anatomically plausible:
horizontal flip (left/right eye mirror), small rotation, scale and shift.
There is deliberately NO colour, brightness, hue or saturation jitter: tissue
colour (pallor) is the diagnostic signal, and altering it would change the
medical meaning of the image. There are no vertical flips (not a realistic
view).
"""

import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as tm
from PIL import Image
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
DERIVED, MODELS, OUTPUTS = ROOT / "derived", ROOT / "models", ROOT / "outputs"
SEED = 42
SIZE = 224
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

BACKBONES = {
    "mobilenet_v3_large": (tm.mobilenet_v3_large, tm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (tm.efficientnet_b0, tm.EfficientNet_B0_Weights.IMAGENET1K_V1),
}


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def verify_split_lock():
    digest = hashlib.sha256((DERIVED / "split.csv").read_bytes()).hexdigest()
    expected = (DERIVED / "split.lock").read_text().strip()
    if digest != expected:
        sys.exit(f"derived/split.csv hash {digest} != derived/split.lock {expected}; refusing to continue")


def load_split():
    """Return manifest rows joined with the locked split."""
    verify_split_lock()
    manifest = pd.read_csv(DERIVED / "manifest.csv")
    split = pd.read_csv(DERIVED / "split.csv")
    assert list(split["image_id"]) == list(manifest["image_id"]), "split.csv does not match manifest"
    manifest["split"] = split["split"]
    return manifest


def load_image(path, mask_only=False):
    """Load an RGBA crop as a normalised 3x224x224 tensor (background == 0).

    mask_only=True replaces tissue colour with white, leaving only the crop
    silhouette. It is used as a shortcut-detection control.
    """
    with Image.open(path) as im:
        rgba = torch.from_numpy(np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    rgb, alpha = rgba[:3], rgba[3:4]
    if mask_only:
        rgb = torch.ones_like(rgb)
    x = rgb * alpha + IMAGENET_MEAN * (1 - alpha)
    _, h, w = x.shape
    side = max(h, w)
    canvas = IMAGENET_MEAN.expand(3, side, side).clone()
    top, left = (side - h) // 2, (side - w) // 2
    canvas[:, top:top + h, left:left + w] = x
    x = F.interpolate(canvas[None], size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)[0]
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def load_images(paths, mask_only=False):
    return torch.stack([load_image(ROOT / p, mask_only) for p in paths])


# Operates on normalised tensors, so fill=0 equals the (neutral) background.
TRAIN_AUGMENT = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0.0),
])


def build_backbone(name):
    fn, weights = BACKBONES[name]
    return fn(weights=weights)


@torch.no_grad()
def embed(name, x, batch=64):
    """Frozen ImageNet backbone -> globally pooled features (no classifier head)."""
    model = build_backbone(name).eval().to(device())
    out = []
    for i in range(0, len(x), batch):
        f = model.avgpool(model.features(x[i:i + batch].to(device())))
        out.append(torch.flatten(f, 1).cpu())
    return torch.cat(out).numpy()


class FineTuneNet(torch.nn.Module):
    """Pretrained backbone with only the last `n_trainable_blocks` feature blocks
    and a new dropout + linear head trainable. BatchNorm statistics stay frozen
    (eval mode) because batches of 16 are too small to re-estimate them."""

    def __init__(self, name, n_trainable_blocks=2, dropout=0.3):
        super().__init__()
        base = build_backbone(name)
        self.features, self.pool = base.features, base.avgpool
        for p in self.features.parameters():
            p.requires_grad = False
        for p in self.features[-n_trainable_blocks:].parameters():
            p.requires_grad = True
        n_feat = {"mobilenet_v3_large": 960, "efficientnet_b0": 1280}[name]
        self.head = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(n_feat, 1))

    def train(self, mode=True):
        super().train(mode)
        for m in self.features.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.features(x)), 1)).squeeze(1)
