"""Versioned storage for trained models.

Layout (created by the training module, read by the app)::

    models/trained/
    ├── CURRENT              <- one line, e.g. "model_v2": the version the app serves
    ├── model_v1/
    │   ├── model.pt         <- weights + everything needed to predict
    │   ├── model.json       <- human-readable: classes, preprocessing, metrics, training settings
    │   └── ...              <- evaluation reports (metrics.json, confusion matrix, history)
    └── model_v2/

Versions are never overwritten: each training run creates the next ``model_vN``.
The app always loads whatever ``CURRENT`` points to, so retraining or rolling
back never requires changing application code.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT_FILE = "CURRENT"
MODEL_FILE = "model.pt"
INFO_FILE = "model.json"
_VERSION_RE = re.compile(r"^model_v(\d+)$")


def models_dir(path: str | Path | None = None) -> Path:
    """Where versions live: argument > $DERMAAI_MODELS_DIR > <repo>/models/trained."""
    p = Path(path or os.getenv("DERMAAI_MODELS_DIR") or ROOT / "models" / "trained")
    return p if p.is_absolute() else ROOT / p


def list_versions(base: str | Path | None = None) -> list[str]:
    """Version names that contain a model file, oldest first."""
    d = models_dir(base)
    if not d.exists():
        return []
    found = [(int(m.group(1)), p.name) for p in d.iterdir()
             if p.is_dir() and (m := _VERSION_RE.match(p.name)) and (p / MODEL_FILE).exists()]
    return [name for _, name in sorted(found)]


def next_version_name(base: str | Path | None = None) -> str:
    d = models_dir(base)
    nums = [int(m.group(1)) for p in (d.iterdir() if d.exists() else []) if (m := _VERSION_RE.match(p.name))]
    return f"model_v{max(nums, default=0) + 1}"


def current_version(base: str | Path | None = None) -> str | None:
    """The version named in CURRENT; falls back to the newest version if CURRENT is missing or stale."""
    d = models_dir(base)
    versions = list_versions(d)
    f = d / CURRENT_FILE
    if f.exists():
        name = f.read_text().strip()
        if name in versions:
            return name
    return versions[-1] if versions else None


def version_dir(name: str, base: str | Path | None = None) -> Path:
    if not _VERSION_RE.match(name):
        raise ValueError(f"Invalid version name {name!r}; expected e.g. 'model_v3'")
    return models_dir(base) / name


def model_path(name: str | None = None, base: str | Path | None = None) -> Path | None:
    """Path to a version's model.pt (the current version by default), or None if there is none."""
    name = name or current_version(base)
    if not name:
        return None
    p = version_dir(name, base) / MODEL_FILE
    return p if p.exists() else None


def read_info(name: str, base: str | Path | None = None) -> dict:
    p = version_dir(name, base) / INFO_FILE
    return json.loads(p.read_text()) if p.exists() else {}


def set_current(name: str, base: str | Path | None = None) -> None:
    """Make the app serve ``name``. Written atomically so a running app never sees a half-written file."""
    if name not in list_versions(base):
        raise ValueError(f"{name} does not exist in {models_dir(base)}. Available: {list_versions(base)}")
    d = models_dir(base)
    tmp = d / (CURRENT_FILE + ".tmp")
    tmp.write_text(name + "\n")
    tmp.replace(d / CURRENT_FILE)
