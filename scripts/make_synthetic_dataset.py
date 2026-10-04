"""Generate a small synthetic lesion dataset in the class-folder layout of data/README.md.

Only for smoke-testing the training pipeline end to end — synthetic images
carry no medical meaning. Real training needs HAM10000 / ISIC data.

    python scripts/make_synthetic_dataset.py --out data/synthetic --per-class 20
    python training/train.py --data data/synthetic --model-type resnet18 --img-size 64 --epochs 2 --promote never
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LESION_RGB = {
    "akiec": (170, 90, 80), "bcc": (200, 140, 150), "bkl": (140, 100, 60), "df": (150, 100, 90),
    "mel": (50, 30, 35), "nv": (120, 80, 55), "vasc": (180, 40, 60),
}
LOCS = ["back", "face", "lower extremity", "trunk", "upper extremity", "abdomen"]


def make_image(dx: str, rng: np.random.Generator, size: int = 192) -> np.ndarray:
    skin = np.array([rng.integers(150, 235), rng.integers(110, 190), rng.integers(90, 170)], np.float32)
    img = np.ones((size, size, 3), np.float32) * skin
    img += rng.normal(0, 5, img.shape)
    c = (int(size / 2 + rng.integers(-15, 15)), int(size / 2 + rng.integers(-15, 15)))
    r = int(rng.integers(size // 7, size // 3.5))
    color = np.array(LESION_RGB[dx], np.float32) + rng.normal(0, 12, 3)
    if dx == "mel":  # irregular, multi-coloured
        pts = [(c[0] + int(r * rng.uniform(0.6, 1.4) * np.cos(a)), c[1] + int(r * rng.uniform(0.6, 1.4) * np.sin(a)))
               for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        cv2.fillPoly(img, [np.array(pts, np.int32)], color.tolist())
        cv2.circle(img, (c[0] + r // 3, c[1]), r // 3, (90, 100, 130), -1)
    else:
        cv2.ellipse(img, c, (r, int(r * rng.uniform(0.7, 1.0))), float(rng.uniform(0, 180)), 0, 360, color.tolist(), -1)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--per-class", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    rows = []
    for dx in CLASSES:
        for i in range(a.per_class):
            image_id = f"SYN_{dx}_{i:04d}"
            (out / dx).mkdir(exist_ok=True)
            cv2.imwrite(str(out / dx / f"{image_id}.jpg"), cv2.cvtColor(make_image(dx, rng), cv2.COLOR_RGB2BGR))
            rows.append({"image": image_id, "lesion_id": f"L_{dx}_{i // 2:04d}",
                         "age": int(rng.integers(20, 85)), "sex": rng.choice(["male", "female"]),
                         "localization": rng.choice(LOCS)})
    with open(out / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} images to {out}")


if __name__ == "__main__":
    main()
