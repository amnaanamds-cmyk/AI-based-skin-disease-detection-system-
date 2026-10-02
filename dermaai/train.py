"""Train a DermNet model.

Example (HAM10000):
    python -m dermaai.train --csv data/HAM10000_metadata.csv \
        --images data/HAM10000_images_part_1 data/HAM10000_images_part_2 \
        --arch efficientnet_b3 --img-size 300 --epochs 30 --pretrained
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import CLASSES
from .data import LesionDataset, balanced_sampler, class_weights, load_table, split_by_lesion
from .metrics import compute_metrics, fit_temperature
from .model import DermNet, save_checkpoint
from .transforms import eval_transform, train_transform


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device, tta: bool = False):
    model.eval()
    all_logits, all_labels = [], []
    for x, meta, y in loader:
        x, meta = x.to(device), meta.to(device)
        logits = model(x, meta)
        if tta:
            views = [x.flip(3), x.flip(2), x.flip(2).flip(3)]
            logits = torch.stack([logits] + [model(v, meta) for v in views]).mean(0)
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(args.csv, args.images)
    train_df, val_df, test_df = split_by_lesion(df, args.val_frac, args.test_frac, args.seed)
    print(f"[data] train={len(train_df)} val={len(val_df)} test={len(test_df)} device={device}")
    print("[data] train class counts:", train_df["dx"].value_counts().to_dict())

    pin = device.type == "cuda"
    train_labels = train_df["label"].to_numpy()
    train_loader = DataLoader(
        LesionDataset(train_df, train_transform(args.img_size)), batch_size=args.batch_size,
        sampler=balanced_sampler(train_labels, args.balance_power), num_workers=args.workers,
        pin_memory=pin, drop_last=len(train_df) > args.batch_size,
    )
    ev = eval_transform(args.img_size)
    val_loader = DataLoader(LesionDataset(val_df, ev), batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(LesionDataset(test_df, ev), batch_size=args.batch_size, num_workers=args.workers, pin_memory=pin)

    model = DermNet(args.arch, len(CLASSES), pretrained=args.pretrained, use_meta=not args.no_meta).to(device)
    # Sampler handles most of the imbalance; a mild loss weight covers the rest.
    weights = class_weights(train_labels, power=args.loss_weight_power).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr}, {"params": head_params, "lr": args.lr * 10}],
        weight_decay=args.weight_decay,
    )
    steps = args.epochs * max(1, len(train_loader))
    scheduler = cosine_with_warmup(optimizer, int(0.05 * steps), steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_epoch, history = -1.0, -1, []
    best_path = out_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, meta, y in train_loader:
            x, meta, y = x.to(device, non_blocking=True), meta.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x, meta), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)

        logits, labels = collect_logits(model, val_loader, device)
        m = compute_metrics(torch.softmax(logits, 1).numpy(), labels.numpy())
        # Model selection on balanced accuracy + melanoma AUC: never trade away melanoma recall.
        score = m["balanced_accuracy"] + 0.5 * m.get("melanoma_auc", 0.0)
        history.append({"epoch": epoch, "loss": running / max(1, seen), **{k: m[k] for k in ("balanced_accuracy", "macro_f1")},
                        "melanoma_auc": m.get("melanoma_auc")})
        print(f"[epoch {epoch:02d}] loss={running / max(1, seen):.4f} val_bacc={m['balanced_accuracy']:.4f} "
              f"val_f1={m['macro_f1']:.4f} mel_auc={m.get('melanoma_auc', float('nan')):.4f} ({time.time() - t0:.0f}s)")
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(best_path, model, classes=CLASSES, img_size=args.img_size)
        elif epoch - best_epoch >= args.patience:
            print(f"[train] early stopping at epoch {epoch} (best {best_epoch})")
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["state_dict"])
    val_logits, val_labels = collect_logits(model, val_loader, device, tta=True)
    temperature = fit_temperature(val_logits, val_labels)
    test_logits, test_labels = collect_logits(model, test_loader, device, tta=True)
    test_metrics = compute_metrics(torch.softmax(test_logits / temperature, 1).numpy(), test_labels.numpy())
    print(f"[test] temperature={temperature:.3f} " + json.dumps({k: v for k, v in test_metrics.items()
                                                                    if k not in ("confusion_matrix",)}, indent=2))

    final = out_dir / "dermaai.pt"
    save_checkpoint(final, model, classes=CLASSES, img_size=args.img_size, temperature=temperature,
                    metrics=test_metrics, best_epoch=best_epoch, version=args.version, trained_at=time.strftime("%Y-%m-%d"))
    (out_dir / "metrics.json").write_text(json.dumps({"test": test_metrics, "history": history,
                                                      "temperature": temperature}, indent=2))
    print(f"[done] saved {final}")
    return test_metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--images", nargs="*", default=[], help="directories containing images")
    p.add_argument("--out", default="models")
    p.add_argument("--arch", default="efficientnet_b0")
    p.add_argument("--pretrained", action="store_true", help="start from ImageNet weights (recommended)")
    p.add_argument("--no-meta", action="store_true", help="disable metadata fusion")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--balance-power", type=float, default=0.5)
    p.add_argument("--loss-weight-power", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--version", default="1.0.0")
    return p


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
