"""Lesion evolution: compare two photos of the same spot over time.

Change ("E" in ABCDE) is the strongest single warning sign of melanoma.
Camera distance is unknown, so size is compared as a share of the frame;
users are told to photograph from the same distance, ideally with a coin or
ruler next to the lesion.
"""

from __future__ import annotations

import numpy as np

from .abcde import compute_abcde


def compare_lesions(before_rgb: np.ndarray, after_rgb: np.ndarray, days_between: int | None = None) -> dict:
    a, b = compute_abcde(before_rgb), compute_abcde(after_rgb)
    if a is None or b is None:
        return {"ok": False, "message": "Could not find the lesion in one of the photos. Centre it and fill more of the frame."}

    area_change = (b["lesion_area_fraction"] - a["lesion_area_fraction"]) / max(a["lesion_area_fraction"], 1e-6)
    diameter_change = (b["diameter_relative"] - a["diameter_relative"]) / max(a["diameter_relative"], 1e-6)
    new_colors = sorted(set(b["colors_detected"]) - set(a["colors_detected"]))
    deltas = {
        "area_change": round(area_change, 3),
        "diameter_change": round(diameter_change, 3),
        "asymmetry_change": round(b["asymmetry"] - a["asymmetry"], 3),
        "border_change": round(b["border_irregularity"] - a["border_irregularity"], 3),
        "color_change": round(b["color_variegation"] - a["color_variegation"], 3),
        "new_colors": new_colors,
    }

    findings = []
    if area_change > 0.25:
        findings.append(f"The lesion appears {area_change:.0%} larger.")
    elif area_change < -0.25:
        findings.append(f"The lesion appears {-area_change:.0%} smaller (check the photos were taken from the same distance).")
    if deltas["asymmetry_change"] > 0.12:
        findings.append("The shape has become more asymmetric.")
    if deltas["border_change"] > 0.15:
        findings.append("The border has become more irregular.")
    if new_colors:
        findings.append("New colour(s) appeared: " + ", ".join(new_colors) + ".")

    worrying = sum([area_change > 0.25, deltas["asymmetry_change"] > 0.12, deltas["border_change"] > 0.15, bool(new_colors)])
    if worrying >= 2 or (new_colors and any(c in {"black", "blue-gray", "red"} for c in new_colors)):
        level, message = "significant", "Several signs of change were found. Show both photos to a dermatologist soon."
    elif worrying == 1:
        level, message = "possible", "Some change was detected. Re-photograph in the same conditions in 2-4 weeks, or see a doctor if you are worried."
    else:
        level, message = "stable", "No meaningful change detected between the two photos."

    rate = None
    if days_between and days_between > 0:
        rate = round(area_change / (days_between / 30.0), 3)
    return {
        "ok": True,
        "change_level": level,
        "message": message,
        "findings": findings,
        "deltas": deltas,
        "monthly_area_growth": rate,
        "before": {k: v for k, v in a.items() if k != "outline"},
        "after": {k: v for k, v in b.items() if k != "outline"},
        "outlines": {"before": a["outline"], "after": b["outline"]},
        "note": "Size is relative to the photo frame. Keep the same distance and lighting, and include a coin or ruler for scale.",
    }
