"""Dataset loading with leakage-free, lesion-grouped splits.

Accepts the HAM10000 ``HAM10000_metadata.csv`` directly, or any CSV with
columns ``image`` (path or id), ``dx`` (label) and optional ``lesion_id``,
``age``, ``sex``, ``localization``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, WeightedRandomSampler

from .config import CLASSES
from .metadata import encode_metadata

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _resolve_image(ref: str, image_dirs: list[Path]) -> Path | None:
    p = Path(ref)
    if p.is_absolute() and p.exists():
        return p
    for d in image_dirs:
        cand = d / ref
        if cand.exists():
            return cand
        if not p.suffix:
            for ext in IMG_EXTS:
                if (cand := d / f"{ref}{ext}").exists():
                    return cand
    return None


def load_table(csv_path: str | Path, image_dirs: list[str | Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "image" not in df.columns:
        if "image_id" not in df.columns:
            raise ValueError("CSV needs an 'image' or 'image_id' column")
        df = df.rename(columns={"image_id": "image"})
    if "dx" not in df.columns:
        raise ValueError("CSV needs a 'dx' label column")
    df["dx"] = df["dx"].str.strip().str.lower()
    unknown = sorted(set(df["dx"]) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown labels {unknown}; expected subset of {CLASSES}")
    if "lesion_id" not in df.columns:
        df["lesion_id"] = df["image"]
    dirs = [Path(d) for d in image_dirs] or [Path(csv_path).parent]
    df["path"] = [_resolve_image(str(r), dirs) for r in df["image"]]
    missing = int(df["path"].isna().sum())
    if missing:
        raise FileNotFoundError(f"{missing} images not found in {[str(d) for d in dirs]}")
    df["label"] = df["dx"].map(CLASSES.index)
    return df.reset_index(drop=True)


def split_by_lesion(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Stratified split grouped by lesion_id.

    HAM10000 contains several photos of the same lesion; splitting by image
    leaks near-duplicates into the test set and inflates accuracy.
    """
    def _split(frame: pd.DataFrame, frac: float):
        n = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        rest_idx, hold_idx = next(sgkf.split(frame, frame["label"], frame["lesion_id"]))
        return frame.iloc[rest_idx].reset_index(drop=True), frame.iloc[hold_idx].reset_index(drop=True)

    rest, test = _split(df, test_frac)
    train, val = _split(rest, val_frac / (1 - test_frac))
    return train, val, test


class LesionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, meta_dropout: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        # Users often skip age/sex/site; dropping fields during training keeps the model
        # accurate when they are missing instead of relying on them.
        self.meta_dropout = meta_dropout

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        fields = [row.get("age"), row.get("sex"), row.get("localization")]
        if self.meta_dropout:
            fields = [None if np.random.random() < self.meta_dropout else f for f in fields]
        meta = encode_metadata(*fields)
        return self.transform(img), meta, int(row["label"])


def class_weights(labels: np.ndarray, num_classes: int = len(CLASSES), power: float = 0.5) -> torch.Tensor:
    """Smoothed inverse-frequency weights (power<1 avoids over-boosting rare classes)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    w[counts == 0] = 0.0
    return torch.tensor(w / w[counts > 0].mean(), dtype=torch.float32)


def balanced_sampler(labels: np.ndarray, power: float = 0.5) -> WeightedRandomSampler:
    w = class_weights(labels, power=power).numpy()[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)
