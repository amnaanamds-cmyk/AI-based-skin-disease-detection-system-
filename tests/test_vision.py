import cv2
import numpy as np

from dermaai.abcde import compute_abcde, segment_lesion
from dermaai.quality import assess_quality


def codes(q):
    return {i["code"] for i in q["issues"]}


def test_good_image_passes(lesion_rgb):
    q = assess_quality(lesion_rgb)
    assert q["acceptable"] and q["score"] >= 80


def test_blurry_image_rejected(lesion_rgb):
    blurred = cv2.GaussianBlur(lesion_rgb, (0, 0), 12)
    q = assess_quality(blurred)
    assert "blurry" in codes(q) and not q["acceptable"]


def test_dark_and_tiny_images_rejected(lesion_rgb):
    assert "too_dark" in codes(assess_quality((lesion_rgb * 0.1).astype(np.uint8)))
    assert "low_resolution" in codes(assess_quality(cv2.resize(lesion_rgb, (90, 90))))


def test_segmentation_finds_circle(lesion_rgb):
    mask = segment_lesion(lesion_rgb)
    assert mask is not None
    expected = np.pi * 80 ** 2 / (400 * 400)
    assert abs(mask.mean() - expected) < 0.03


def test_abcde_symmetric_vs_irregular(lesion_rgb):
    regular = compute_abcde(lesion_rgb)
    img = np.full((400, 400, 3), (205, 160, 140), np.uint8)
    pts = np.array([(200, 90), (260, 170), (330, 160), (280, 250), (300, 330), (200, 280), (120, 320), (140, 220), (70, 150)], np.int32)
    cv2.fillPoly(img, [pts], (60, 35, 30))
    cv2.circle(img, (250, 230), 30, (110, 110, 140), -1)
    irregular = compute_abcde(img)
    assert regular["asymmetry"] < 0.1
    assert irregular["asymmetry"] > regular["asymmetry"]
    assert irregular["border_irregularity"] > regular["border_irregularity"]
    assert irregular["suspicion_score"] > regular["suspicion_score"]
