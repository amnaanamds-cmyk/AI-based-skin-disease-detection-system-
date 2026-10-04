# Data

This folder holds the images the model learns from. The app never reads it; only the training module does.

```
data/
├── raw/            <- YOUR dataset goes here (never modified by the code)
├── processed/      <- created automatically: resized copies + manifest.csv (safe to delete; rebuilt on next run)
├── splits.csv      <- created automatically: which images are train / val / test (keep it!)
└── README.md
```

## 1. Where to put images

Put your dataset in `data/raw/`. Two layouts are supported. Use whichever is easier, or both at once.

### Layout A: one folder per class (simplest)

```
data/raw/
├── mel/
│   ├── IMG_0001.jpg
│   └── IMG_0002.jpg
├── nv/
│   └── ...
├── bcc/
│   └── ...
└── metadata.csv        (optional, see section 4)
```

The folder name is the label. Images can also be in sub-folders inside a class folder.

### Layout B: a label file

```
data/raw/
├── images/
│   ├── IMG_0001.jpg
│   └── IMG_0002.jpg
└── labels.csv
```

`labels.csv` needs an `image` column and a `label` column:

```csv
image,label,age,sex,localization,lesion_id
IMG_0001.jpg,mel,63,female,back,patient12_spot1
IMG_0002.jpg,nv,,,,
```

`image` may omit the extension (`IMG_0001` finds `IMG_0001.jpg` / `.jpeg` / `.png`). `dx`, `class` or
`diagnosis` are accepted instead of `label`, and `image_id` or `filename` instead of `image`.

The bundled ISIC 2019 dataset uses layout B. Get it with `python scripts/prepare_isic2019.py`.

## 2. Class names

Use these codes so the app shows the right medical description and urgency:

| Code | Condition | Counts as malignant |
|---|---|---|
| `mel` | Melanoma | yes |
| `bcc` | Basal cell carcinoma | yes |
| `scc` | Squamous cell carcinoma | yes |
| `akiec` | Actinic keratosis / intraepithelial carcinoma | yes (pre-malignant) |
| `nv` | Melanocytic nevus (mole) | no |
| `bkl` | Benign keratosis | no |
| `df` | Dermatofibroma | no |
| `vasc` | Vascular lesion | no |

Common full names are converted automatically. For example, `Melanoma/`, `melanoma/`, `Basal Cell Carcinoma/`,
`nevus/` and `mole/` all work. Names are lower-cased and spaces or dashes become `_`.

**New classes** (e.g. `eczema/`, `psoriasis/`) are allowed. The model learns any class you give it. To give a new
class a proper name, urgency and advice in the app, add it to `class_info` in `training/config.py`; if it is a
cancer, also add it to `malignant_classes`. No app code changes are needed.

## 3. File formats and image quality

- Formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif`, `.tiff`
- Any size. Images are resized automatically, but use at least about 300×300 px of the lesion.
- One lesion per image, roughly centred, in focus. Dermoscopy and close-up phone photos both work, but the
  model works best on the kind of photo it was trained on.
- Unreadable, tiny (<64 px) and duplicate-with-conflicting-label files are skipped automatically and listed
  in the training output. Your files are never changed.

## 4. Optional extra information (improves accuracy)

For layout A, add `data/raw/metadata.csv`. For layout B, add the columns to `labels.csv`.

| Column | Example | Notes |
|---|---|---|
| `image` | `IMG_0001.jpg` | file name, to match the row to the image (layout A) |
| `age` | `54` | years |
| `sex` | `female` | `male` / `female` |
| `localization` | `back` | one of: abdomen, acral, back, chest, ear, face, foot, genital, hand, lower extremity, neck, scalp, trunk, upper extremity |
| `lesion_id` | `patient7_mole2` | **important if you have several photos of the same lesion**: they will always be kept in the same split, so the test score stays honest |
| `source` | `clinic_A` | optional; the evaluation then also reports accuracy per source |

Missing values are fine. The model is trained to cope without them.

## 5. How much data?

| Images per class | What to expect |
|---|---|
| < 20 | refused by default (`min_images_per_class` in `training/config.py`) |
| 20–100 | it trains, but results will be unreliable. Use `--init-from current` to fine-tune the existing model |
| 100–1,000 | reasonable, especially when fine-tuning |
| 1,000+ | good |

Keep classes roughly balanced if you can. Rare classes are automatically up-weighted, but more real examples always
help more.

## 6. Adding more data later

1. Copy the new images into the same structure (new files into the class folders, or new rows in `labels.csv`).
2. Run `python training/train.py --init-from current` (fine-tune the current model) or `python training/train.py`
   (train fresh from ImageNet weights).

Images already used keep their train/val/test assignment (stored in `data/splits.csv`, identified by file
content). New images are added to the splits, so the old test images are never trained on and versions stay
comparable. Do not delete `data/splits.csv` unless you deliberately want a brand-new split (`--resplit`).

## Licences

ISIC 2019 images are licensed CC BY-NC 4.0 (non-commercial). Models trained on them inherit that restriction.
For a commercial product, train on data you have rights to use commercially.
