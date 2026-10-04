"""Prepare raw images for training (runs automatically from train.py).

For every image in the dataset table this:
  1. opens it and checks it is a readable, reasonably sized photo (bad files are skipped and listed),
  2. fixes rotation from the camera (EXIF) and converts to RGB,
  3. saves a resized copy in data/processed/ so later epochs and later runs load fast,
  4. flags exact duplicate files that carry different labels.

Raw data in data/raw is never modified. Re-running only processes new or changed images.

The image -> tensor transform used by the model (resize, crop, normalise) is NOT here:
it lives in dermaai/preprocessing.py because the app must apply exactly the same steps.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

from dermaai.preprocessing import load_image, resize_for_cache

MIN_SIDE = 64  # images smaller than this are almost certainly thumbnails or icons


def _cache_name(src: Path, size: int) -> str:
    """Stable file name that changes when the source file or the cache size changes."""
    st = src.stat()
    key = f"{src.resolve()}|{st.st_size}|{int(st.st_mtime)}|{size}"
    return f"{src.stem[:60]}_{hashlib.sha1(key.encode()).hexdigest()[:10]}.jpg"


def _process_one(src: Path, out_dir: Path, size: int) -> tuple[str | None, str | None, str | None]:
    """Returns (processed path, content hash, error)."""
    try:
        dst = out_dir / _cache_name(src, size)
        digest = hashlib.md5(src.read_bytes()).hexdigest()
        if not dst.exists():
            img = load_image(src)
            if min(img.size) < MIN_SIDE:
                return None, digest, f"too small ({img.width}x{img.height})"
            tmp = dst.with_suffix(".tmp")
            resize_for_cache(img, size).save(tmp, "JPEG", quality=93)
            tmp.replace(dst)
        return str(dst), digest, None
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        return None, None, f"unreadable ({type(e).__name__})"


def prepare_images(df: pd.DataFrame, processed_dir: Path, size: int, workers: int = 8) -> pd.DataFrame:
    """Add 'processed_path', 'md5' and 'usable' columns. Unusable images are kept in the table
    (marked usable=False) so that excluding a file never reshuffles the train/test split."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_dirs = {label: processed_dir / label for label in df["label"].unique()}
    for d in out_dirs.values():
        d.mkdir(exist_ok=True)
    jobs = [(Path(p), out_dirs[label], size) for p, label in zip(df["path"], df["label"])]
    print(f"[prep] checking and caching {len(jobs)} images in {processed_dir} ...", flush=True)
    with ThreadPoolExecutor(max(1, workers)) as pool:
        results = list(pool.map(lambda j: _process_one(*j), jobs))

    df = df.copy()
    df["processed_path"] = [r[0] for r in results]
    df["md5"] = [r[1] for r in results]
    df["usable"] = [r[0] is not None for r in results]
    errors = [(str(p), r[2]) for p, r in zip(df["path"], results) if r[2]]
    if errors:
        print(f"[prep] skipped {len(errors)} unusable images, e.g.:")
        for path, why in errors[:10]:
            print(f"         {path}: {why}")

    # The same photo filed under two different classes would confuse training and evaluation.
    dup = df.dropna(subset=["md5"]).groupby("md5")["label"].nunique()
    conflicts = dup[dup > 1].index
    if len(conflicts):
        bad = df[df["md5"].isin(conflicts)]
        print(f"[prep] WARNING: {len(conflicts)} identical images have conflicting labels and are skipped, e.g.:")
        for path in bad["path"].head(6):
            print(f"         {path}")
        df.loc[df["md5"].isin(conflicts), "usable"] = False
    # Exact copies share one lesion id, so they can never be split across train and test.
    has_md5 = df["md5"].notna()
    df.loc[has_md5, "lesion_id"] = df[has_md5].groupby("md5")["lesion_id"].transform("first")
    print(f"[prep] {int(df['usable'].sum())} of {len(df)} images ready", flush=True)
    return df.reset_index(drop=True)


def write_manifest(df: pd.DataFrame, processed_dir: Path) -> Path:
    """Save the exact list of images, labels and splits used, for reproducibility."""
    path = processed_dir / "manifest.csv"
    cols = [c for c in ("path", "processed_path", "label", "split", "usable", "lesion_id", "md5", "age", "sex", "localization", "source")
            if c in df.columns]
    df[cols].to_csv(path, index=False)
    return path
