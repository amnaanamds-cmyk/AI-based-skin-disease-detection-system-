"""Find, check and split the training data in ``data/raw``.

Two layouts are supported, and you can use both at once (see data/README.md):

A. One folder per class (simplest)::

       data/raw/mel/img001.jpg
       data/raw/nv/img002.jpg
       data/raw/metadata.csv        (optional: image, age, sex, localization, lesion_id)

B. A label file::

       data/raw/images/img001.jpg
       data/raw/labels.csv          (columns: image, label  + optional age, sex, localization, lesion_id)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler

from dermaai.metadata import encode_metadata
from dermaai.preprocessing import is_image_file, load_image

LABEL_FILE = "labels.csv"
METADATA_FILE = "metadata.csv"
LABEL_FILE_IMAGE_DIR = "images"

# Folder / label names people commonly use -> the class codes the app knows about.
# Names not listed here are kept as they are and become new classes.
CLASS_ALIASES = {
    "melanoma": "mel", "nevus": "nv", "naevus": "nv", "mole": "nv", "melanocytic_nevus": "nv",
    "basal_cell_carcinoma": "bcc", "squamous_cell_carcinoma": "scc", "actinic_keratosis": "akiec",
    "ak": "akiec", "benign_keratosis": "bkl", "seborrheic_keratosis": "bkl", "dermatofibroma": "df",
    "vascular_lesion": "vasc", "vascular": "vasc",
}


def normalize_class_name(name: str) -> str:
    """'Basal Cell Carcinoma' -> 'bcc', 'My-New Class' -> 'my_new_class'."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return CLASS_ALIASES.get(slug, slug)


def _read_label_file(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / LABEL_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    image_col = next((c for c in ("image", "image_id", "filename", "file") if c in df.columns), None)
    label_col = next((c for c in ("label", "dx", "class", "diagnosis") if c in df.columns), None)
    if not image_col or not label_col:
        raise ValueError(f"{data_dir / LABEL_FILE} needs an 'image' column and a 'label' column; found {list(df.columns)}")
    img_dir = data_dir / LABEL_FILE_IMAGE_DIR
    paths = []
    for ref in df[image_col].astype(str):
        p = img_dir / ref
        if not p.suffix:  # ids without extension, e.g. ISIC_0000000
            p = next((img_dir / f"{ref}{ext}" for ext in (".jpg", ".jpeg", ".png") if (img_dir / f"{ref}{ext}").exists()), p)
        paths.append(p)
    out = pd.DataFrame({"path": paths, "label": df[label_col].map(normalize_class_name)})
    for col in ("age", "sex", "localization", "lesion_id", "source"):
        if col in df.columns:
            out[col] = df[col].values
    missing = ~out["path"].map(Path.exists)
    if missing.any():
        print(f"[data] WARNING: {int(missing.sum())} images listed in {LABEL_FILE} were not found in {img_dir} and are skipped")
    return out[~missing]


def _read_class_folders(data_dir: Path) -> pd.DataFrame:
    rows = []
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir() and p.name != LABEL_FILE_IMAGE_DIR
                         and not p.name.startswith((".", "_"))):
        label = normalize_class_name(folder.name)
        for f in sorted(folder.rglob("*")):
            if f.is_file() and is_image_file(f):
                rows.append({"path": f, "label": label})
    df = pd.DataFrame(rows, columns=["path", "label"])
    meta_file = data_dir / METADATA_FILE
    if len(df) and meta_file.exists():
        meta = pd.read_csv(meta_file)
        meta.columns = [c.strip().lower() for c in meta.columns]
        key = next((c for c in ("image", "filename", "file", "image_id") if c in meta.columns), None)
        if key:
            meta["_stem"] = meta[key].astype(str).map(lambda s: Path(s).stem)
            df["_stem"] = df["path"].map(lambda p: p.stem)
            cols = [c for c in ("age", "sex", "localization", "lesion_id", "source") if c in meta.columns]
            df = df.merge(meta.drop_duplicates("_stem")[["_stem", *cols]], on="_stem", how="left").drop(columns="_stem")
    return df


def load_dataset(data_dir: Path, class_names: list[str] | None = None) -> pd.DataFrame:
    """Collect every labelled image under ``data_dir`` into one table."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder {data_dir} does not exist. See data/README.md for how to add data.")
    parts = []
    if (data_dir / LABEL_FILE).exists():
        parts.append(_read_label_file(data_dir))
    parts.append(_read_class_folders(data_dir))
    df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
    if df.empty:
        raise ValueError(f"No labelled images found in {data_dir}. Put images in one folder per class "
                         f"(e.g. {data_dir}/mel/*.jpg) or add {LABEL_FILE}. See data/README.md.")
    for col in ("age", "sex", "localization", "lesion_id", "source"):
        if col not in df.columns:
            df[col] = None
    # Photos of the same lesion must stay in the same split; without ids, each image is its own lesion.
    df["lesion_id"] = df["lesion_id"].where(df["lesion_id"].notna() & (df["lesion_id"].astype(str) != ""),
                                            df["path"].map(lambda p: Path(p).stem))
    if class_names:
        wanted = [normalize_class_name(c) for c in class_names]
        df = df[df["label"].isin(wanted)]
    return df.reset_index(drop=True)


def check_dataset(df: pd.DataFrame, min_per_class: int) -> list[str]:
    """Return the class list (sorted) after checking every class has enough images."""
    counts = df["label"].value_counts().sort_index()
    print("[data] images per class:")
    for label, n in counts.items():
        flag = "  <-- too few" if n < min_per_class else "  (more data recommended)" if n < 100 else ""
        print(f"         {label:<24} {n:>6}{flag}")
    too_few = counts[counts < min_per_class]
    if len(too_few):
        raise ValueError(f"Classes {list(too_few.index)} have fewer than {min_per_class} images. Add more images, "
                         f"remove those folders, or lower min_images_per_class in training/config.py.")
    if len(counts) < 2:
        raise ValueError("At least two classes are needed to train a classifier.")
    return sorted(counts.index)


def split_dataset(df: pd.DataFrame, val_split: float, test_split: float, seed: int) -> pd.DataFrame:
    """Add a 'split' column (train / val / test), stratified by class and grouped by lesion_id."""
    def holdout(frame: pd.DataFrame, frac: float) -> tuple[np.ndarray, np.ndarray]:
        n_splits = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return next(sgkf.split(frame, frame["label"], frame["lesion_id"]))

    df = df.copy()
    rest_idx, test_idx = holdout(df, test_split)
    rest = df.iloc[rest_idx]
    train_idx, val_idx = holdout(rest, val_split / (1 - test_split))
    df["split"] = "test"
    df.loc[rest.index[train_idx], "split"] = "train"
    df.loc[rest.index[val_idx], "split"] = "val"
    return df


def _hash_split(lesion_id: str, val_split: float, test_split: float, seed: int) -> str:
    """Stable pseudo-random split for one lesion: the same id always lands in the same split."""
    u = int(hashlib.sha1(f"{seed}:{lesion_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test_split else "val" if u < test_split + val_split else "train"


def assign_splits(df: pd.DataFrame, split_file: Path, val_split: float, test_split: float, seed: int,
                  resplit: bool = False) -> pd.DataFrame:
    """Give every image a split that never changes between training runs.

    Images are identified by content (md5), so renaming or moving files keeps their split. When you add
    data later, existing images keep their split, new photos of an already-known lesion join that
    lesion's split, and other new images are assigned by a stable hash. This keeps the test set
    untouched across retraining, so scores of model_v1, v2, ... stay comparable and honest.
    """
    df = df.copy()
    df["split"] = None
    known = pd.read_csv(split_file) if split_file.exists() and not resplit else pd.DataFrame(columns=["md5", "lesion_id", "split"])
    if len(known) and df["md5"].isin(known["md5"]).any():
        df["split"] = df["md5"].map(dict(zip(known["md5"], known["split"])))
        by_lesion = df.dropna(subset=["split"]).drop_duplicates("lesion_id").set_index("lesion_id")["split"]
        df["split"] = df["split"].fillna(df["lesion_id"].map(by_lesion))
        new = df["split"].isna()
        df.loc[new, "split"] = [_hash_split(str(l), val_split, test_split, seed) for l in df.loc[new, "lesion_id"]]
        if new.any():
            print(f"[split] kept existing splits from {split_file.name}; {int(new.sum())} new images assigned")
    else:
        df["split"] = split_dataset(df, val_split, test_split, seed)["split"]
        print(f"[split] created a new stratified split and saved it to {split_file}")
    known = pd.concat([known, df[["md5", "lesion_id", "split"]].dropna(subset=["md5"])]).drop_duplicates("md5", keep="first")
    split_file.parent.mkdir(parents=True, exist_ok=True)
    known.to_csv(split_file, index=False)
    return df


class LesionDataset(Dataset):
    """Yields (image tensor, metadata vector, class index) for one split."""

    def __init__(self, df: pd.DataFrame, classes: list[str], transform, meta_dropout: float = 0.0,
                 image_column: str = "processed_path"):
        self.df = df.reset_index(drop=True)
        self.class_index = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.meta_dropout = meta_dropout
        self.image_column = image_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        image = load_image(row[self.image_column])
        fields = [row.get("age"), row.get("sex"), row.get("localization")]
        if self.meta_dropout:  # teach the model to cope when users skip these questions
            fields = [None if np.random.random() < self.meta_dropout else f for f in fields]
        return self.transform(image), encode_metadata(*fields), self.class_index[row["label"]]


def class_weights(labels: np.ndarray, num_classes: int, power: float) -> torch.Tensor:
    """Inverse-frequency weights softened by ``power`` (0 = all equal, 1 = fully inverse)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    w[counts == 0] = 0.0
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


def balanced_sampler(labels: np.ndarray, num_classes: int, power: float) -> WeightedRandomSampler:
    """Draw rare classes more often so each batch is less dominated by the common ones."""
    w = class_weights(labels, num_classes, power).numpy()[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)
