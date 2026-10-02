"""Grad-CAM visual explanations."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def grad_cam(model, x: torch.Tensor, meta: torch.Tensor | None, class_idx: int) -> np.ndarray:
    """Return an HxW heatmap in [0, 1] for ``class_idx`` (Selvaraju et al., 2017)."""
    with torch.enable_grad():
        logits, fmap = model(x, meta, return_features=True)
        grads = torch.autograd.grad(logits[0, class_idx], fmap)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / cam.max().clamp_min(1e-8)).detach().cpu().numpy()


def overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def to_data_url(rgb: np.ndarray, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt, quality=88)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()
