"""Train (or retrain) the skin-lesion classifier.

    python training/train.py                         # use data/raw and the settings in training/config.py
    python training/train.py --data path/to/dataset # another dataset folder
    python training/train.py --init-from current     # fine-tune the app's current model on (more) data
    python training/train.py --epochs 3 --promote never   # quick experiment that doesn't change the app

What happens:
  1. load the dataset from data/raw and check it            (training/dataset.py)
  2. check, fix and cache the images in data/processed      (training/preprocessing.py)
  3. split into train / validation / test, grouped by lesion
  4. build the model (ImageNet weights, or an earlier version of ours)
  5. train, keeping the epoch with the best validation score
  6. calibrate confidences and fit the unusual-photo detector (training/calibration.py)
  7. evaluate on the untouched test split                    (training/evaluate.py)
  8. save everything as models/trained/model_vN, and make it the app's model if it is at least
     as good as the current one (see `promote` in training/config.py)
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in (None, ""):  # allow `python training/train.py` from any folder
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dermaai import model_store
from dermaai.config import CONDITIONS
from dermaai.device import pick_device
from dermaai.model import GITHUB_WEIGHTS, DermNet, load_checkpoint, save_checkpoint
from dermaai.preprocessing import PreprocessingParams, eval_transform, train_transform
from training.calibration import fit_ood_detector, fit_temperature
from training.config import ROOT, TrainingConfig, parse_config
from training.dataset import (
    LesionDataset,
    assign_splits,
    balanced_sampler,
    check_dataset,
    class_weights,
    load_dataset,
    normalize_class_name,
)
from training.evaluate import compute_metrics, format_summary, predict_loader, write_reports
from training.preprocessing import prepare_images, write_manifest


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Learning rate rises for the first 5% of steps, then decays smoothly to zero."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def load_initial_weights(model: DermNet, init_from: str, models_dir: Path) -> str:
    """Fine-tuning: copy every weight whose shape still fits (all of them unless classes changed)."""
    version = model_store.current_version(models_dir) if init_from == "current" else init_from
    path = model_store.model_path(version, models_dir) if version else None
    if not path:
        raise SystemExit(f"--init-from {init_from}: no such model in {models_dir} ({model_store.list_versions(models_dir)})")
    old, info = load_checkpoint(path)
    if old.arch != model.arch:
        raise SystemExit(f"--init-from {version} uses {old.arch}, but model_type is {model.arch}. Use the same model_type.")
    own = model.state_dict()
    usable = {k: v for k, v in old.state_dict().items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(usable, strict=False)
    skipped = [k for k in own if k not in usable]
    note = f" (re-initialised {len(skipped)} tensors, e.g. the class layer, because the classes changed)" if skipped else ""
    print(f"[model] starting from {version}{note}")
    return version


def project_relative(path) -> str:
    """Store paths relative to the project so saved records stay valid on another machine."""
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def data_signature(df) -> str:
    """Fingerprint of the exact images/labels/splits, so --resume refuses to continue on different data."""
    key = "|".join(f"{p}:{l}:{s}" for p, l, s in zip(df["processed_path"], df["label"], df["split"]))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def train(cfg: TrainingConfig) -> dict:
    started = time.time()
    seed_everything(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    device = pick_device(cfg.device)
    models_dir = cfg.path(cfg.models_dir)
    ckpt_dir = cfg.path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1-3. data ------------------------------------------------------------------------------
    print(f"[data] reading {cfg.path(cfg.data_dir)}")
    df = load_dataset(cfg.path(cfg.data_dir), cfg.class_names)
    df = prepare_images(df, cfg.path(cfg.processed_dir), max(cfg.processed_size, cfg.img_size))
    classes = check_dataset(df[df["usable"]], cfg.min_images_per_class)  # clear message before anything else
    df = assign_splits(df, cfg.path(cfg.split_file), cfg.val_split, cfg.test_split, cfg.seed, cfg.resplit)
    write_manifest(df, cfg.path(cfg.processed_dir))
    df = df[df["usable"]].reset_index(drop=True)
    if cfg.class_names:  # keep the order the user asked for
        classes = [c for c in map(normalize_class_name, cfg.class_names) if c in classes]
    print("[data] split sizes: " + ", ".join(f"{s}={int((df['split'] == s).sum())}" for s in ("train", "val", "test")))
    new_classes = [c for c in classes if c not in CONDITIONS]
    if new_classes:
        print(f"[data] new classes the app has no description for (shown with a generic one): {new_classes}")

    prep = PreprocessingParams(img_size=cfg.img_size)
    train_df, val_df, test_df = (df[df["split"] == s] for s in ("train", "val", "test"))
    train_labels = train_df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    pin = device.type == "cuda"
    loader_args = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=pin,
                       persistent_workers=cfg.num_workers > 0)
    train_loader = DataLoader(LesionDataset(train_df, classes, train_transform(prep), cfg.meta_dropout),
                              sampler=balanced_sampler(train_labels, len(classes), cfg.balance_power),
                              drop_last=len(train_df) > cfg.batch_size, **loader_args)
    val_loader = DataLoader(LesionDataset(val_df, classes, eval_transform(prep)), **loader_args)
    test_loader = DataLoader(LesionDataset(test_df, classes, eval_transform(prep)), **loader_args)

    # 4. model -------------------------------------------------------------------------------
    try:
        model = DermNet(cfg.model_type, len(classes), pretrained=cfg.pretrained and not cfg.init_from,
                        use_meta=cfg.use_metadata).to(device)
    except Exception as e:  # usually: no internet access to download ImageNet weights
        if not cfg.pretrained or cfg.init_from:
            raise
        raise SystemExit(f"Could not download ImageNet weights for '{cfg.model_type}' ({type(e).__name__}: {e}).\n"
                         f"Check your internet connection, pick a model_type with mirrored weights "
                         f"({', '.join(GITHUB_WEIGHTS)}), or train without them using --no-pretrained (much less accurate).")
    started_from = load_initial_weights(model, cfg.init_from, models_dir) if cfg.init_from else (
        "ImageNet" if cfg.pretrained else "random")
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_labels, len(classes), cfg.loss_weight_power).to(device),
                                    label_smoothing=cfg.label_smoothing)
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg.learning_rate},
                                   {"params": head_params, "lr": cfg.learning_rate * 10}], weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * max(1, len(train_loader))
    scheduler = cosine_with_warmup(optimizer, int(0.05 * total_steps), total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 5. training loop -----------------------------------------------------------------------
    signature = data_signature(df)
    best_path, last_path = ckpt_dir / "best.pt", ckpt_dir / "last.pt"
    best_score, best_epoch, history, start_epoch = -1.0, 0, [], 1
    if cfg.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("data_signature") != signature or state.get("classes") != classes:
            raise SystemExit("--resume: the dataset or classes changed since the interrupted run. Start a new run instead.")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_score, best_epoch, history, start_epoch = state["best_score"], state["best_epoch"], state["history"], state["epoch"] + 1
        print(f"[train] resuming after epoch {state['epoch']}")
    deadline = started + cfg.max_hours * 3600 if cfg.max_hours else None

    print(f"[train] {cfg.model_type} on {device}, {len(classes)} classes, {cfg.epochs} epochs, "
          f"{len(train_loader)} batches/epoch")
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        t0, loss_sum, seen = time.time(), 0.0, 0
        for step, (x, meta, y) in enumerate(train_loader, 1):
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
            loss_sum += loss.item() * len(y)
            seen += len(y)
            if cfg.log_every and step % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch} batch {step}/{len(train_loader)} loss={loss_sum / seen:.4f} "
                      f"({seen / elapsed:.1f} img/s, ~{(len(train_loader) - step) * elapsed / step / 60:.1f} min left)", flush=True)

        logits, labels = predict_loader(model, val_loader, device, tta=False)
        val = compute_metrics(torch.softmax(logits, 1).numpy(), labels.numpy(), classes, cfg.malignant_classes)
        # Best epoch = best balanced accuracy, with melanoma ranking as a tie-breaker that protects cancer detection.
        score = val["balanced_accuracy"] + 0.5 * val.get("melanoma_auc", val.get("malignant_auc", 0.0))
        history.append({"epoch": epoch, "train_loss": round(loss_sum / max(1, seen), 5),
                        "val_balanced_accuracy": round(val["balanced_accuracy"], 5), "val_macro_f1": round(val["macro_f1"], 5),
                        "val_melanoma_auc": round(val["melanoma_auc"], 5) if "melanoma_auc" in val else None,
                        "minutes": round((time.time() - t0) / 60, 1)})
        improved = score > best_score
        if improved:
            best_score, best_epoch = score, epoch
            torch.save(model.state_dict(), best_path)
        print(f"[epoch {epoch}] loss={loss_sum / max(1, seen):.4f} val_balanced_acc={val['balanced_accuracy']:.4f} "
              f"val_f1={val['macro_f1']:.4f}{' (best so far)' if improved else ''} [{(time.time() - t0) / 60:.1f} min]", flush=True)
        torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "history": history,
                    "data_signature": signature, "classes": classes}, last_path)
        if epoch - best_epoch >= cfg.patience:
            print(f"[train] no improvement for {cfg.patience} epochs, stopping early")
            break
        if deadline and time.time() + (time.time() - t0) > deadline:
            print(f"[train] max_hours reached, stopping after epoch {epoch}")
            break

    # 6. calibration ---------------------------------------------------------------------------
    model.load_state_dict(torch.load(best_path, map_location=device))
    print(f"[calibrate] using epoch {best_epoch}; fitting confidence calibration and unusual-photo detector ...", flush=True)
    val_logits, val_labels = predict_loader(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    fit_loader = DataLoader(LesionDataset(train_df, classes, eval_transform(prep)), batch_size=cfg.batch_size,
                            num_workers=cfg.num_workers)
    ood = fit_ood_detector(model, fit_loader, val_loader, device)

    # 7. evaluation on the test split -----------------------------------------------------------
    test_logits, test_labels = predict_loader(model, test_loader, device)
    probs = torch.softmax(test_logits / temperature, 1).numpy()
    metrics = compute_metrics(probs, test_labels.numpy(), classes, cfg.malignant_classes)
    if test_df["source"].notna().any():
        metrics["per_source"] = {}
        for src in sorted(test_df["source"].dropna().unique()):
            mask = (test_df["source"] == src).to_numpy()
            if mask.sum() >= 20:
                m = compute_metrics(probs[mask], test_labels.numpy()[mask], classes, cfg.malignant_classes)
                metrics["per_source"][str(src)] = {k: m[k] for k in ("n_images", "balanced_accuracy", "accuracy") if k in m}
    print("\n" + format_summary(metrics, "Test-set evaluation (images never used for training)"))

    # 8. save as a new version, then decide whether the app should use it ----------------------
    version = model_store.next_version_name(models_dir)
    out_dir = model_store.version_dir(version, models_dir)
    meta = {
        "version_name": version,
        "classes": classes,
        "malignant_classes": [c for c in cfg.malignant_classes if c in classes],
        "class_info": {normalize_class_name(k): v for k, v in cfg.class_info.items() if normalize_class_name(k) in classes},
        "preprocessing": prep.to_dict(),
        "img_size": prep.img_size,
        "temperature": temperature,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "started_from": started_from,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_minutes": round((time.time() - started) / 60, 1),
        "dataset": {"path": project_relative(cfg.path(cfg.data_dir)), "images": len(df),
                    "per_split": {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")},
                    "per_class": df["label"].value_counts().to_dict(), "signature": signature},
        "training_config": asdict(cfg),
        "metrics": {k: v for k, v in metrics.items() if k not in ("confusion_matrix",)},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    used = df[["path", "label", "split", "lesion_id"]].assign(path=df["path"].map(project_relative))
    used.to_csv(out_dir / "dataset_used.csv", index=False)  # exact record of the data behind this version
    save_checkpoint(out_dir / model_store.MODEL_FILE, model, ood=ood, **meta)
    (out_dir / model_store.INFO_FILE).write_text(json.dumps(meta, indent=2, default=str))
    write_reports(metrics, out_dir, history, {"version": version, "started_from": started_from,
                                              "best_epoch": best_epoch, "temperature": round(temperature, 4),
                                              "model_type": cfg.model_type, "img_size": cfg.img_size})
    print(f"\n[save] {out_dir}")
    promoted = decide_promotion(cfg, version, metrics, models_dir, test_df, classes, device)
    return {"version": version, "promoted": promoted, "metrics": metrics, "path": str(out_dir)}


def decide_promotion(cfg, version, metrics, models_dir, test_df, classes, device) -> bool:
    """Switch the app to the new model if policy allows; old versions are always kept."""
    current = model_store.current_version(models_dir)
    reason = ""
    if cfg.promote == "never":
        promote, reason = False, "promote=never"
    elif cfg.promote == "always" or current is None or current == version:
        promote, reason = True, "promote=always" if cfg.promote == "always" else "first trained model"
    else:
        old, info = load_checkpoint(model_store.model_path(current, models_dir))
        if info.get("classes") != classes:
            promote, reason = True, f"classes changed ({current} cannot be compared on this test set)"
        else:
            old.to(device)
            prep = PreprocessingParams.from_dict(info.get("preprocessing"), img_size=info.get("img_size"))
            loader = DataLoader(LesionDataset(test_df, classes, eval_transform(prep)), batch_size=cfg.batch_size)
            logits, labels = predict_loader(old, loader, device)
            old_m = compute_metrics(torch.softmax(logits / float(info.get("temperature") or 1), 1).numpy(),
                                    labels.numpy(), classes, info.get("malignant_classes"))
            new_b, old_b = metrics["balanced_accuracy"], old_m["balanced_accuracy"]
            promote = new_b >= old_b - 0.005
            reason = (f"balanced accuracy on this test set: {version} {new_b:.4f} vs {current} {old_b:.4f}")
            if not promote:
                reason += " (new model is worse, keeping the current one)"
    if promote:
        model_store.set_current(version, models_dir)
        print(f"[promote] the app now uses {version} — {reason}")
    else:
        print(f"[promote] the app keeps using {current} — {reason}. "
              f"To switch anyway: python training/manage_models.py use {version}")
    return promote


def main(argv: list[str] | None = None) -> None:
    cfg = parse_config(argv, description=__doc__)
    result = train(cfg)
    print(f"\nDone. {result['version']} saved in {result['path']}. Open report.html there for a readable summary.")


if __name__ == "__main__":
    main()
