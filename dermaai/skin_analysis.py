"""Cosmetic skin analysis of a face or skin-area photo.

Classical image measurements on the skin pixels only: redness, blemishes,
uneven pigmentation, texture and shine. These are image-based estimates for
skincare personalisation and progress tracking, not a medical assessment.
"""

from __future__ import annotations

import cv2
import numpy as np

from .explain import to_data_url

CONCERNS = {
    "redness": "Redness",
    "blemishes": "Blemishes / breakouts",
    "pigmentation": "Uneven tone & dark spots",
    "texture": "Texture & pores",
    "shine": "Oiliness / shine",
}


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    """Skin pixels via YCrCb + HSV rules, cleaned and restricted to the largest regions."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cr, cb, v = ycrcb[..., 1], ycrcb[..., 2], hsv[..., 2]
    # Broad bounds so every Fitzpatrick type is included.
    m = ((cr >= 133) & (cr <= 178) & (cb >= 70) & (cb <= 130) & (v >= 35)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m.astype(bool)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.05 * m.size]
    if not keep:
        keep = [1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
    m = np.isin(labels, keep)
    # Erode so hairline, eyebrows and lip borders don't count as skin features.
    return cv2.erode(m.astype(np.uint8), k, iterations=2).astype(bool)


def _score(x: float, lo: float, hi: float) -> int:
    """Map a raw measurement onto a 0-100 concern score."""
    return int(round(100 * float(np.clip((x - lo) / (hi - lo), 0, 1))))


def analyze_skin(rgb: np.ndarray, work_size: int = 640) -> dict:
    h, w = rgb.shape[:2]
    s = work_size / max(h, w)
    if s < 1:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    mask = skin_mask(rgb)
    coverage = float(mask.mean())
    if coverage < 0.08 or mask.sum() < 2000:
        return {"ok": False, "skin_coverage": round(coverage, 3),
                "message": "Not enough visible skin. Use a well-lit, front-facing close-up without filters."}

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A = lab[..., 0], lab[..., 1]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    scale = max(rgb.shape[:2]) / 640
    big = int(31 * scale) | 1
    small = max(3, int(3 * scale) | 1)

    # Redness: a* above the person's own median (so skin tone itself is not penalised).
    a_med = float(np.median(A[mask]))
    red_frac = float((A[mask] > a_med + 8).mean())
    red_mean = float(np.clip(A[mask] - 128, 0, None).mean())

    # Blemishes: small, local a* peaks (inflamed spots) relative to surroundings.
    a_local = A - cv2.GaussianBlur(A, (big, big), 0)
    spots = ((a_local > 7) & mask).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(spots)
    area_px = mask.sum()
    blemish_count = int(sum(1 for i in range(1, n) if 4 * scale ** 2 <= st[i, cv2.CC_STAT_AREA] <= 0.004 * area_px))

    # Pigmentation: dark patches relative to local brightness + overall tone evenness.
    l_local = L - cv2.GaussianBlur(L, (big, big), 0)
    dark_frac = float(((l_local < -9) & mask).sum() / area_px)
    l_flat = L - cv2.GaussianBlur(L, (int(101 * scale) | 1,) * 2, 0)
    unevenness = float(np.std(l_flat[mask]))

    # Texture: high-frequency energy (pores, roughness) after removing lighting.
    hf = gray - cv2.GaussianBlur(gray, (small * 2 + 1,) * 2, 0)
    texture = float(np.std(hf[mask]))

    # Shine: specular highlights (bright, desaturated) on skin.
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # Highlights fall outside the skin colour rule, so look inside the closed skin region.
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (big, big))
    region = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kc).astype(bool)
    shine = float(((hsv[..., 2] >= 200) & (hsv[..., 1] < 50) & region).sum() / region.sum())

    scores = {
        "redness": int(round(0.6 * _score(red_frac, 0.03, 0.30) + 0.4 * _score(red_mean, 10, 30))),
        "blemishes": _score(blemish_count, 1, 25),
        "pigmentation": int(round(0.5 * _score(dark_frac, 0.005, 0.06) + 0.5 * _score(unevenness, 2.5, 9))),
        "texture": _score(texture, 2.0, 9.0),
        "shine": _score(shine, 0.003, 0.05),
    }
    weights = {"redness": 0.2, "blemishes": 0.25, "pigmentation": 0.25, "texture": 0.15, "shine": 0.15}
    health = int(round(100 - sum(scores[k] * wt for k, wt in weights.items())))

    # Visual: tint detected spots so users can see what was measured.
    vis = rgb.copy()
    vis[(a_local > 7) & mask] = (0.5 * vis[(a_local > 7) & mask] + [127, 0, 0]).astype(np.uint8)
    vis[(l_local < -9) & mask] = (0.5 * vis[(l_local < -9) & mask] + [0, 0, 110]).astype(np.uint8)
    edge = mask.astype(np.uint8) - cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    vis[edge.astype(bool)] = (34, 211, 238)

    return {
        "ok": True,
        "skin_health_score": health,
        "concerns": [{"code": k, "name": CONCERNS[k], "score": v,
                      "level": "high" if v >= 60 else "moderate" if v >= 30 else "low"}
                     for k, v in sorted(scores.items(), key=lambda kv: -kv[1])],
        "scores": scores,
        "skin_type_hint": "oily" if scores["shine"] >= 55 else "normal/dry" if scores["shine"] < 15 else "combination",
        "measurements": {
            "skin_coverage": round(coverage, 3), "blemish_count": blemish_count,
            "redness_fraction": round(red_frac, 4), "dark_spot_fraction": round(dark_frac, 4),
            "tone_unevenness": round(unevenness, 2), "texture_energy": round(texture, 2),
            "shine_fraction": round(shine, 4),
        },
        "visualization": to_data_url(vis),
        "note": "Image-based cosmetic estimates. Lighting, camera and make-up affect results — "
                "compare photos taken in the same conditions.",
    }
