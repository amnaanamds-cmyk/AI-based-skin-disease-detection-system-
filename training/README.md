# Training module

Everything needed to train, evaluate and manage the skin-lesion model lives in this folder.
The web app (`app.py`, `dermaai/`) never trains. It only loads the model that this module produces.

```
your dataset (data/raw) ──► training/train.py ──► models/trained/model_vN ──► app.py uses models/trained/CURRENT
```

## Quick start

```bash
pip install -r requirements.txt          # once

python training/train.py                 # train on data/raw with the settings in training/config.py
python training/manage_models.py list    # see versions and scores; * marks the one the app uses
python app.py                            # start the app (it uses the newest promoted model)
```

Run commands from the project folder (or anywhere; paths are resolved relative to the project).
On Windows use `python` the same way; `py training\train.py` also works.

## Files

| File | What it does | Edit it? |
|---|---|---|
| `config.py` | **All settings**: data paths, model type, image size, epochs, batch size, learning rate, splits, seed, class settings | yes, this is the place |
| `train.py` | The training run, in 8 numbered steps (see below) | only to change the training procedure |
| `dataset.py` | Finds images in `data/raw`, maps class names, checks counts, keeps train/val/test splits stable | rarely |
| `preprocessing.py` | Checks, auto-rotates and caches images in `data/processed` | rarely |
| `calibration.py` | Makes confidence scores honest and fits the "unusual photo" detector | rarely |
| `evaluate.py` | Metrics and reports; can also be run on its own | rarely |
| `manage_models.py` | List versions, switch / roll back the app's model, import a checkpoint | no |

The image → tensor step (resize, crop, normalise) is in `dermaai/preprocessing.py`. It is shared with the app
so that prediction always uses exactly the preprocessing the model was trained with. Its parameters are
saved inside every model.

## What `train.py` does

1. **Load** `data/raw` (class folders and/or `labels.csv`). Prints images per class and stops with a clear
   message if a class has too few.
2. **Preprocess**: opens every image, fixes rotation, skips unreadable or tiny files, flags duplicates with
   conflicting labels, and caches resized copies in `data/processed`. Only new files are processed on later runs.
3. **Split** into train (70%) / validation (15%) / test (15%), stratified by class and grouped by `lesion_id`.
   The split is saved in `data/splits.csv` and reused, so test images stay test images forever.
4. **Model**: an EfficientNet (or another timm CNN) pretrained on ImageNet, plus a small network for age, sex and
   body site. With `--init-from`, it starts from one of our earlier models instead.
5. **Train**: AdamW, warm-up then cosine learning-rate decay, class balancing, augmentation (flips, rotation,
   colour, blur, random erasing), and metadata dropout. After each epoch it scores the validation set and keeps the
   best epoch. It stops early if the score stops improving.
6. **Calibrate**: temperature scaling (so "80% confident" means right ~80% of the time) and the
   out-of-distribution detector.
7. **Evaluate** on the test split: accuracy, balanced accuracy, precision, recall, F1 per class, confusion matrix,
   AUCs, calibration error, and melanoma specificity at 90% sensitivity.
8. **Save** `models/trained/model_vN/` and **promote** it to the app if it is at least as good as the current model
   on the same test images (`promote` setting).

Each version folder contains:

| File | Contents |
|---|---|
| `model.pt` | weights + classes + preprocessing + calibration + OOD detector: everything the app needs |
| `model.json` | the same information in readable form, plus training settings and dataset summary |
| `report.html` | open in a browser: scores, per-class table, confusion matrix, training history |
| `evaluation.txt`, `metrics.json`, `confusion_matrix.csv`, `training_history.csv` | the same results as text / data |
| `dataset_used.csv` | every image used and its split |

## Changing settings

Edit the defaults in `config.py`, or override them for one run. Every setting has a `--option`:

```bash
python training/train.py --epochs 20 --batch-size 16 --learning-rate 1e-4
python training/train.py --model-type efficientnet_b2 --img-size 260     # bigger model (use a GPU)
python training/train.py --data D:/datasets/my_clinic                    # another dataset folder
python training/train.py --class-names mel nv bcc                        # train on a subset of classes
python training/train.py --promote never                                 # experiment without touching the app
python training/train.py --help                                          # every option
```

Most useful settings:

| Setting | Default | When to change |
|---|---|---|
| `epochs` | 10 | more for large datasets / GPU; early stopping protects against overfitting |
| `img_size` | 192 | 224–300 gives more detail but is slower (CPU: keep 192) |
| `model_type` | `efficientnet_b0` | `efficientnet_b2`, `resnet50`, `mobilenetv3_large_100` (fast, for phones) |
| `batch_size` | 32 | lower it if you run out of memory |
| `learning_rate` | 2e-4 | lower (5e-5) when fine-tuning on a small new dataset |
| `max_hours` | 0 (no limit) | cap training time; the run then finishes cleanly with the best epoch so far |

## Retraining: fresh vs fine-tuning

| | Command | Starts from | Use when |
|---|---|---|---|
| **Fresh** | `python training/train.py` | ImageNet weights | you have plenty of data, changed `model_type`, or want a clean result |
| **Fine-tune** | `python training/train.py --init-from current` | the app's current model (or `--init-from model_v2`) | adding data, small datasets, adding a class. Faster, keeps what was learned |

Both create a new version; nothing is overwritten. With `--init-from`, every layer whose shape still fits is
reused. If you added or removed classes, only the final class layer starts fresh.

Fine-tuning is safe with respect to evaluation: `data/splits.csv` guarantees that images the earlier model was tested
on are still only in the test set.

## Interrupted training

Progress is saved after every epoch in `models/checkpoints/last.pt`. Continue with the same command plus `--resume`:

```bash
python training/train.py --resume
```

It refuses to resume if the data changed in the meantime.

## Model versions and rollback

```bash
python training/manage_models.py list            # versions, dates, scores; * = used by the app
python training/manage_models.py use model_v1    # roll back (the running app switches on its next request)
python training/manage_models.py info model_v2   # all details of one version
```

`models/trained/CURRENT` is a one-line text file with the version name the app serves.

## Evaluating separately

```bash
python training/evaluate.py                          # current model, test split of data/raw
python training/evaluate.py --model model_v1 --out reports/v1
python training/evaluate.py --data path/to/external_dataset --all   # every image of another dataset
```

Evaluating on an **external** dataset (another hospital or camera) is the best check of how the model will do
in real use.

## Adding a new class

1. Create `data/raw/<new_class>/` with images (at least 20; 100+ recommended).
2. Optional: describe it for the app in `class_info` in `config.py`, and add it to `malignant_classes` if it is a
   cancer.
3. `python training/train.py --init-from current`

Nothing else changes. The class list is stored in the model and the app reads it from there.

## Changing the model architecture

- **Another backbone**: set `model_type` in `config.py` to any timm CNN name. That is all.
- **Different network design** (layers, heads, fusion): edit `dermaai/model.py` (`DermNet`). It is shared by
  training and the app, so saved models keep loading. Retrain afterwards. Old versions trained with the
  previous design may no longer load if you change layer names or shapes.
- **Different augmentation**: `train_transform` in `dermaai/preprocessing.py`.
- **Different input preprocessing** (normalisation, crop): `eval_transform` / `PreprocessingParams` in
  `dermaai/preprocessing.py`. New values are saved with each new model automatically.

## Hardware

Without a GPU, training on all 25k ISIC images at 192 px takes about 20 minutes per epoch on 4 CPU cores. A GPU is
10–50× faster and is picked automatically (`--device auto`). Apple Silicon (`mps`) is supported.
