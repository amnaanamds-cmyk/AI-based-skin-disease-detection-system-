"""Out-of-distribution scoring used at prediction time.

The detector itself is fitted by the training module (training/calibration.py).
"""

from __future__ import annotations

import torch


def mahalanobis_score(feats: torch.Tensor, density: dict) -> torch.Tensor:
    """Squared Mahalanobis distance to the closest class mean (higher = less like the training data)."""
    diff = feats[:, None, :] - density["means"][None].to(feats)
    d = torch.einsum("bkd,de,bke->bk", diff, density["precision"].to(feats), diff)
    return d.min(dim=1).values
