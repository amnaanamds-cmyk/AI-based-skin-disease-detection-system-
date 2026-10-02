"""Clinically meaningful evaluation metrics and temperature calibration."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from .config import CLASSES, MALIGNANT


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(1)
    present = np.unique(labels)
    out: dict = {
        "n": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=present, zero_division=0)),
        "ece": expected_calibration_error(probs, labels),
        "per_class_recall": {},
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(CLASSES)))).tolist(),
    }
    for c in present:
        out["per_class_recall"][CLASSES[c]] = float((preds[labels == c] == c).mean())
    if len(present) > 1:
        try:
            out["macro_auc"] = float(roc_auc_score(labels, probs[:, present] / probs[:, present].sum(1, keepdims=True),
                                                   multi_class="ovr", labels=present))
        except ValueError:
            pass

    mal_idx = [CLASSES.index(c) for c in MALIGNANT]
    is_mal = np.isin(labels, mal_idx)
    mal_score = probs[:, mal_idx].sum(1)
    if 0 < is_mal.sum() < len(labels):
        out["malignant_auc"] = float(roc_auc_score(is_mal, mal_score))
    mel = CLASSES.index("mel")
    is_mel = labels == mel
    if 0 < is_mel.sum() < len(labels):
        out["melanoma_auc"] = float(roc_auc_score(is_mel, probs[:, mel]))
        # Operating point at >=90% melanoma sensitivity — what matters for screening.
        thr = float(np.quantile(probs[is_mel, mel], 0.10))
        flagged = probs[:, mel] >= thr
        out["melanoma_at_90_sensitivity"] = {
            "threshold": thr,
            "sensitivity": float(flagged[is_mel].mean()),
            "specificity": float((~flagged[~is_mel]).mean()),
        }
    return out


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al., 2017): makes confidences trustworthy."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0).item())
