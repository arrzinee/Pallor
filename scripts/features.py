"""Colour features computed on conjunctiva tissue pixels only.

The PNGs are pre-segmented crops: background pixels have alpha == 0 and are
stored as RGB (0, 0, 0). A thin boundary of pixels has 0 < alpha < 255
(soft mask edge; RGB is stored un-premultiplied, i.e. roughly true tissue
colour). Only fully opaque pixels (alpha == 255) are treated as tissue: this
excludes the black background entirely and conservatively drops the uncertain
mask boundary, which would darken toward black if alpha were ignored.

Features are per-image statistics with no learned parameters, so computing
them before the train/test split leaks nothing. All fitted preprocessing
(scaling) lives inside the sklearn pipelines.
"""

import numpy as np
from PIL import Image

EPS = 1e-6


def tissue_pixels(path):
    """Return an (N, 3) float array in [0, 1] of fully opaque RGB pixels."""
    with Image.open(path) as im:
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float64)
    mask = rgba[..., 3] == 255
    if not mask.any():
        raise ValueError(f"{path}: no fully opaque pixels")
    return rgba[mask][:, :3] / 255.0


def srgb_to_lab(rgb):
    """sRGB in [0, 1] -> CIE L*a*b* (D65)."""
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[:, 1] - 16,
                     500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def rgb_to_hsv(rgb):
    """RGB in [0, 1] -> hue in radians, saturation and value in [0, 1]."""
    r, g, b = rgb.T
    mx, mn = rgb.max(1), rgb.min(1)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 0
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / d[rm]) % 6
    h[gm] = (b - r)[gm] / d[gm] + 2
    h[bm] = (r - g)[bm] / d[bm] + 4
    s = np.where(mx > 0, d / np.maximum(mx, EPS), 0)
    return h * (np.pi / 3), s, mx


def extract(path):
    rgb = tissue_pixels(path)
    r, g, b = rgb.T
    feats = {}
    for name, ch in zip("RGB", (r, g, b)):
        feats[f"{name}_mean"] = ch.mean()
        feats[f"{name}_std"] = ch.std()
        feats[f"{name}_p10"], feats[f"{name}_p50"], feats[f"{name}_p90"] = np.percentile(ch, [10, 50, 90])

    total = r + g + b + EPS
    feats["chrom_r_mean"] = (r / total).mean()
    feats["chrom_g_mean"] = (g / total).mean()

    # Hue is circular (red sits at both 0 and 2*pi), so average it as a vector.
    h, s, v = rgb_to_hsv(rgb)
    feats["hue_sin_mean"] = np.sin(h).mean()
    feats["hue_cos_mean"] = np.cos(h).mean()
    feats["S_mean"], feats["S_std"] = s.mean(), s.std()
    feats["V_mean"], feats["V_std"] = v.mean(), v.std()

    lab = srgb_to_lab(rgb)
    for i, name in enumerate(("L", "a", "b")):
        feats[f"{name}_lab_mean"] = lab[:, i].mean()
        feats[f"{name}_lab_std"] = lab[:, i].std()
    feats["a_over_L"] = lab[:, 1].mean() / (lab[:, 0].mean() + EPS)

    # Redness indices
    feats["erythema_index"] = (100 * np.log10((r + EPS) / (g + EPS))).mean()  # 100*log10(R/G)
    feats["rg_norm_diff"] = ((r - g) / (r + g + EPS)).mean()
    feats["r_over_g_mean"] = ((r + EPS) / (g + EPS)).mean()
    feats["frac_red_dominant"] = ((r > g) & (r > b)).mean()
    feats["n_tissue_px"] = float(len(rgb))
    return feats
