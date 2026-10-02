"""Encoding of optional patient metadata into a fixed-length vector."""

from __future__ import annotations

import math

import torch

from .config import LOCALIZATIONS, META_DIM, SEXES


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() in {"", "unknown"})


def encode_metadata(age=None, sex=None, localization=None) -> torch.Tensor:
    vec = torch.zeros(META_DIM, dtype=torch.float32)
    if _missing(age):
        vec[1] = 1.0
    else:
        vec[0] = min(max(float(age), 0.0), 100.0) / 100.0
    off = 2
    sex = None if _missing(sex) else str(sex).strip().lower()
    vec[off + (SEXES.index(sex) if sex in SEXES else len(SEXES))] = 1.0
    off += len(SEXES) + 1
    loc = None if _missing(localization) else str(localization).strip().lower()
    vec[off + (LOCALIZATIONS.index(loc) if loc in LOCALIZATIONS else len(LOCALIZATIONS))] = 1.0
    return vec
