"""Inspect the raw Pallor dataset and write a validated manifest.

Reads (never writes) data/. Uses only data/Anaemic and data/Non-Anaemic;
data/All Categories is checked to confirm it duplicates them, then ignored.

Outputs (gitignored):
  derived/manifest.csv            one row per usable image
  derived/inspection_summary.json counts and validation results

Exits with status 1 if any validation check fails.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DERIVED = ROOT / "derived"
WORKBOOK = DATA / "Cp-Anaemic Dataset.xlsx"
CLASS_FOLDERS = {"Anaemic": "Anemic", "Non-Anaemic": "Non-anemic"}  # folder -> workbook label
HB_THRESHOLD = 11.0  # g/dL; anemic := Hb < 11


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    errors = []
    summary = {}

    # --- Directory structure -------------------------------------------------
    folders = {}
    for d in sorted(p for p in DATA.iterdir() if p.is_dir()):
        files = sorted(f for f in d.iterdir() if f.is_file() and not f.name.startswith("."))
        folders[d.name] = files
        summary.setdefault("folders", {})[d.name] = {
            "n_files": len(files),
            "extensions": dict(Counter(f.suffix.lower() for f in files)),
        }
    for folder in CLASS_FOLDERS:
        if folder not in folders:
            errors.append(f"missing folder data/{folder}")

    # --- Workbook: all sheets must agree -------------------------------------
    sheets = pd.read_excel(WORKBOOK, sheet_name=None)
    summary["workbook_sheets"] = {name: list(df.columns) for name, df in sheets.items()}
    master = sheets["Cp-Anaemic Dataset"]
    per_class = pd.concat([sheets["Anaemic"], sheets["Non-Anaemic"]]).rename(
        columns={"image_id": "Image_ID", "filename": "File_Name",
                 "folder_label": "Folder_Label", "hb_g_dl": "Hb_g_dl"})
    joined = master.merge(per_class, on="Image_ID", how="outer", suffixes=("", "_cls"), indicator=True)
    if (joined["_merge"] != "both").any():
        errors.append(f"master vs per-class sheets: {(joined['_merge'] != 'both').sum()} unmatched rows")
    for col in ["File_Name", "Folder_Label", "Hb_g_dl"]:
        n = int((joined[col] != joined[f"{col}_cls"]).sum())
        if n:
            errors.append(f"master vs per-class sheets disagree on {col} in {n} rows")
    if master["File_Name"].duplicated().any():
        errors.append("duplicate File_Name in workbook")
    if master[["File_Name", "Folder_Label", "Hb_g_dl"]].isna().any().any():
        errors.append("missing values in workbook")

    # --- Map each image file to exactly one workbook row ---------------------
    rows = []
    hashes = {}
    by_name = master.set_index("File_Name")
    for folder, expected_label in CLASS_FOLDERS.items():
        on_disk = {f.name: f for f in folders.get(folder, []) if f.suffix.lower() == ".png"}
        in_sheet = set(master.loc[master["Folder_Label"] == expected_label, "File_Name"])
        for name in sorted(set(on_disk) - in_sheet):
            errors.append(f"{folder}/{name}: file has no workbook row with label {expected_label}")
        for name in sorted(in_sheet - set(on_disk)):
            errors.append(f"{folder}/{name}: workbook row has no file")

        for name in sorted(set(on_disk) & in_sheet):
            path = on_disk[name]
            rec = by_name.loc[name]
            hb = float(rec["Hb_g_dl"])
            anemic = int(hb < HB_THRESHOLD)
            if anemic != int(expected_label == "Anemic"):
                errors.append(f"{name}: Hb {hb} inconsistent with folder label {expected_label}")
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im.load()
                    mode, (w, h) = im.mode, im.size
                    alpha = np.asarray(im.getchannel("A")) if "A" in im.getbands() else None
            except Exception as exc:  # noqa: BLE001 - report every unreadable file
                errors.append(f"{folder}/{name}: unreadable ({exc})")
                continue
            n_opaque = int((alpha == 255).sum()) if alpha is not None else w * h
            if n_opaque < 100:
                errors.append(f"{folder}/{name}: only {n_opaque} fully opaque pixels")
            digest = sha256(path)
            if digest in hashes:
                errors.append(f"{name}: byte-identical to {hashes[digest]}")
            hashes[digest] = name
            rows.append({
                "image_id": rec["Image_ID"],
                "path": str(path.relative_to(ROOT)),
                "hb_g_dl": hb,
                "anemic": anemic,
                "folder_label": expected_label,
                "mode": mode,
                "width": w,
                "height": h,
                "n_opaque_px": n_opaque,
                "sha256": digest,
            })

    manifest = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if len(manifest) != len(master):
        errors.append(f"manifest has {len(manifest)} rows, workbook has {len(master)}")

    # --- All Categories: confirm it only duplicates class folders -----------
    allcat = folders.get("All Categories", [])
    allcat_dupes = sum(1 for f in allcat if f.suffix.lower() == ".png" and sha256(f) in hashes)
    summary["all_categories"] = {
        "n_png": sum(f.suffix.lower() == ".png" for f in allcat),
        "n_byte_identical_to_class_folders": allcat_dupes,
        "used": False,
    }

    summary["manifest"] = {
        "n_images": len(manifest),
        "n_anemic": int(manifest["anemic"].sum()),
        "n_non_anemic": int((1 - manifest["anemic"]).sum()),
        "hb_min": float(manifest["hb_g_dl"].min()),
        "hb_max": float(manifest["hb_g_dl"].max()),
        "hb_mean": round(float(manifest["hb_g_dl"].mean()), 3),
        "image_modes": dict(Counter(manifest["mode"])),
        "label_rule": f"anemic = Hb < {HB_THRESHOLD} g/dL (matches folder label for all rows: {not any('inconsistent' in e for e in errors)})",
        "patient_ids_available": False,
    }
    summary["errors"] = errors
    summary["valid"] = not errors

    DERIVED.mkdir(exist_ok=True)
    (DERIVED / "inspection_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "workbook_sheets"}, indent=2))
    if errors:
        # Never leave a manifest on disk that failed validation.
        (DERIVED / "manifest.csv").unlink(missing_ok=True)
        print(f"\nVALIDATION FAILED: {len(errors)} error(s); manifest not written", file=sys.stderr)
        return 1
    manifest.to_csv(DERIVED / "manifest.csv", index=False)
    print(f"\nOK: wrote {len(manifest)} rows to {(DERIVED / 'manifest.csv').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
