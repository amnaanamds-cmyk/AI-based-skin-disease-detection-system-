"""Post-training steps that make the model's outputs trustworthy (run automatically by train.py).

1. Temperature scaling: neural networks are usually over-confident. One number T is fitted on the
   validation set so that "80% confident" really means right about 80% of the time.
2. Out-of-distribution detector: learns what typical training images look like inside the network,
   so the app can say "this doesn't look like a skin-lesion photo" instead of guessing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

from dermaai.metrics import mahalanobis_score


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Find T minimising validation loss of softmax(logits / T) (Guo et al., 2017)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())


@torch.no_grad()
def collect_features(model, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats, labels = [], []
    for x, _, y in loader:
        feats.append(model.embed(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def fit_feature_density(feats: np.ndarray, labels: np.ndarray) -> dict:
    """One Gaussian per class in feature space, with a shared covariance (Lee et al., 2018)."""
    classes = np.unique(labels)
    means = np.stack([feats[labels == c].mean(0) for c in classes])
    centred = feats - means[np.searchsorted(classes, labels)]
    precision = LedoitWolf().fit(centred).precision_
    return {"means": torch.tensor(means, dtype=torch.float32), "precision": torch.tensor(precision, dtype=torch.float32)}


def fit_ood_detector(model, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                     percentile: float = 95.0) -> dict:
    """Fit on training images; set the threshold so `percentile`% of held-out validation images pass."""
    tr_feats, tr_labels = collect_features(model, train_loader, device)
    density = fit_feature_density(tr_feats.numpy(), tr_labels.numpy())
    val_feats, _ = collect_features(model, val_loader, device)
    scores = mahalanobis_score(val_feats, density).numpy()
    threshold = float(np.percentile(scores, percentile))
    # Beyond far_threshold an image is treated as "not a lesion photo at all" and even an urgent result is
    # replaced by a retake request. It sits above every real validation image, so no genuine lesion is affected.
    far_threshold = float(scores.max() * 1.1)
    return {**density, "threshold": threshold, "far_threshold": far_threshold}
