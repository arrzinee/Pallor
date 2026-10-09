# Dataset

## Provenance — ⚠️ PENDING VERIFICATION

| Item | Status |
|---|---|
| Source | Downloaded from **Mendeley Data** (reported by the project owner). |
| Dataset title / DOI | **Not yet recorded — pending verification.** |
| Licence | **Unknown — pending verification.** No licence file is included in the local copy. |
| Publication / citation | **Unknown — pending verification.** No citation is asserted here. |
| Collection population, site, Hb measurement method | **Unknown — pending verification.** |

Until the Mendeley Data record has been checked:

- Do **not** redistribute the images or the workbook, in any form.
- Do **not** commit `data/`, `derived/`, or per-image outputs to Git (enforced by `.gitignore`).
- Do **not** describe the population, collection protocol, or Hb measurement method in any write-up.

Action: open the Mendeley Data record the files came from, then fill in the DOI, licence, and required citation above. Do not guess them.

## Local layout (read-only)

```
data/
  Anaemic/              110 PNG   ← used
  Non-Anaemic/          141 PNG   ← used
  All Categories/       251 PNG   ← NOT used: byte-identical copies of the two folders above
  Cp-Anaemic Dataset.xlsx         ← 3 sheets; per-image Hb (g/dL) and folder label
```

The original files are never modified. `scripts/inspect_dataset.py` reads them and writes a manifest to `derived/` (gitignored).

## Known limitations

- **No patient identifiers.** Image numbers run from 4 to 710 with gaps, so this appears to be a subset of a larger collection. The `group_id` column is unique per image and appears to be a row index, not a patient ID. **We cannot rule out multiple images from the same person** (for example, both eyes or repeat visits). If that happens, a random train/test split can put the same person on both sides, which would make test metrics optimistic.
- **Small sample** (n = 251). Test-set metrics have wide confidence intervals.
- **Label = Hb threshold.** The folder label is exactly `Hb < 11 g/dL` for every image. The workbook does not record how Hb was measured.
- **Uncontrolled acquisition.** Lighting and white balance vary strongly between images, and there is no colour reference in frame.
- Images are pre-segmented conjunctiva crops. Background pixels are fully transparent and stored as RGB (0,0,0).
