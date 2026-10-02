import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
# Small backbone keeps the test suite fast on CPU.
os.environ.setdefault("DERMAAI_DEMO_ARCH", "resnet18")
os.environ.setdefault("DERMAAI_DEVICE", "cpu")


@pytest.fixture
def lesion_rgb() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((400, 400, 3), (205, 160, 140), np.float32) + rng.normal(0, 6, (400, 400, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.circle(img, (200, 200), 80, (95, 60, 45), -1)
    return img
