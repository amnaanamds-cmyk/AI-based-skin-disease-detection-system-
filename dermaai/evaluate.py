"""Evaluate a checkpoint on a labelled CSV (e.g. an external test set).

    python -m dermaai.evaluate --checkpoint models/dermaai.pt --csv test.csv --images test_images/
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .data import LesionDataset, load_table
from .metrics import compute_metrics
from .model import load_checkpoint
from .train import collect_logits, pick_device
from .transforms import eval_transform


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--images", nargs="*", default=[])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--out", help="write metrics JSON here")
    a = p.parse_args()
    device = pick_device()
    model, info = load_checkpoint(a.checkpoint)
    model.to(device)
    df = load_table(a.csv, a.images)
    loader = DataLoader(LesionDataset(df, eval_transform(int(info.get("img_size", 224)))), batch_size=a.batch_size)
    logits, labels = collect_logits(model, loader, device, tta=not a.no_tta)
    probs = torch.softmax(logits / float(info.get("temperature") or 1.0), 1).numpy()
    metrics = compute_metrics(probs, labels.numpy())
    text = json.dumps(metrics, indent=2)
    print(text)
    if a.out:
        open(a.out, "w").write(text)


if __name__ == "__main__":
    main()
