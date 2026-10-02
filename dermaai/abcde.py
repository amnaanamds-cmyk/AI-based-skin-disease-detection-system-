"""Classical computer-vision lesion segmentation and ABCDE features.

Dermatologists use the ABCDE rule (Asymmetry, Border, Colour, Diameter,
Evolution). Computing it independently of the neural network gives an
interpretable second opinion that clinicians already trust.
"""

from __future__ import annotations

import cv2
import numpy as np

# Reference dermoscopic colours in CIE-Lab (OpenCV 8-bit scaling).
REFERENCE_COLORS = {
    "white": (235, 128, 128),
    "red": (120, 175, 155),
    "light brown": (150, 140, 160),
    "dark brown": (75, 140, 150),
    "blue-gray": (110, 125, 115),
    "black": (25, 128, 128),
}


def segment_lesion(rgb: np.ndarray, work_size: int = 256) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    s = work_size / max(h, w)
    img = cv2.resize(rgb, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # Lesions are darker and usually more saturated than surrounding skin.
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    feat = (255 - lab[..., 0]).astype(np.float32) + 1.5 * chroma
    feat = cv2.normalize(feat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    feat = cv2.GaussianBlur(feat, (7, 7), 0)
    _, mask = cv2.threshold(feat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    ih, iw = mask.shape
    best, best_score = None, 0.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.01 * ih * iw:
            continue
        cx, cy = cents[i]
        dist = np.hypot((cx - iw / 2) / iw, (cy - ih / 2) / ih)
        touches = (x == 0) + (y == 0) + (x + bw >= iw) + (y + bh >= ih)
        # Prefer large, central components; penalise dark vignette corners.
        score = area * (1 - dist) / (1 + 1.5 * touches)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None
    lesion = (labels == best).astype(np.uint8)
    contours, _ = cv2.findContours(lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(lesion)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def _asymmetry(mask: np.ndarray) -> float:
    m = mask.astype(np.uint8)
    mom = cv2.moments(m, binaryImage=True)
    if mom["m00"] == 0:
        return 0.0
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    angle = 0.5 * np.degrees(np.arctan2(2 * mom["mu11"], mom["mu20"] - mom["mu02"]))
    h, w = m.shape
    size = int(2 * max(h, w))
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[:, 2] += (size / 2 - cx, size / 2 - cy)
    aligned = cv2.warpAffine(m, M, (size, size), flags=cv2.INTER_NEAREST).astype(bool)
    scores = []
    for flipped in (aligned[::-1, :], aligned[:, ::-1]):
        inter = np.logical_and(aligned, flipped).sum()
        union = np.logical_or(aligned, flipped).sum()
        scores.append(1 - inter / max(union, 1))
    return float(np.mean(scores))


def compute_abcde(rgb: np.ndarray) -> dict | None:
    mask = segment_lesion(rgb)
    if mask is None:
        return None
    h, w = mask.shape
    area = float(mask.sum())
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    compactness = perimeter ** 2 / (4 * np.pi * max(area, 1.0))

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[mask].astype(np.float32)
    if len(lab) > 20000:
        lab = lab[np.random.default_rng(0).choice(len(lab), 20000, replace=False)]
    refs = np.array(list(REFERENCE_COLORS.values()), dtype=np.float32)
    nearest = np.argmin(((lab[:, None, :] - refs[None]) ** 2).sum(-1), axis=1)
    frac = np.bincount(nearest, minlength=len(refs)) / max(len(nearest), 1)
    colors = [name for name, f in zip(REFERENCE_COLORS, frac) if f >= 0.05]

    asym = _asymmetry(mask)
    border = float(np.clip((compactness - 1.0) / 1.5, 0, 1))
    color_score = float(np.clip((len(colors) - 1) / 4, 0, 1))
    diameter_frac = float(2 * np.sqrt(area / np.pi) / max(h, w))

    eps = 0.004 * perimeter
    poly = cv2.approxPolyDP(contour, eps, True)[:, 0, :]
    return {
        "asymmetry": round(asym, 3),
        "border_irregularity": round(border, 3),
        "color_variegation": round(color_score, 3),
        "colors_detected": colors,
        "diameter_relative": round(diameter_frac, 3),
        "lesion_area_fraction": round(area / (h * w), 4),
        # Weighted like the dermoscopic TDS score, normalised to 0-1.
        "suspicion_score": round(float(np.clip(0.4 * asym + 0.25 * border + 0.35 * color_score, 0, 1)), 3),
        "outline": [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in poly],
    }
