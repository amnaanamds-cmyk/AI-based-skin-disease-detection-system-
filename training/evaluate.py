"""Measure how well a model performs, and write easy-to-read reports.

Runs automatically at the end of train.py. You can also run it on its own::

    python training/evaluate.py                       # current model, test split of data/raw
    python training/evaluate.py --model model_v1      # a specific version
    python training/evaluate.py --data path/to/other_dataset --all   # every image of another dataset

Metrics (this is single-label image classification, so these are the relevant ones):
  accuracy            share of images whose top prediction is correct (misleading when classes are unbalanced)
  balanced accuracy   average recall over classes — the main score; every class counts equally
  precision / recall / F1 per class, and macro averages
  confusion matrix    rows = true class, columns = predicted class
  AUC                 ranking quality, per class and malignant-vs-benign
  ECE                 calibration: do 80%-confidence predictions turn out right ~80% of the time?
  melanoma @ 90% sensitivity   specificity when the threshold is set to catch 9 in 10 melanomas
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python training/evaluate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dermaai.preprocessing import eval_transform


@torch.no_grad()
def predict_loader(model, loader: DataLoader, device: torch.device, tta: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw scores (logits) for every image. TTA averages 5 flipped/rotated views, exactly like the app."""
    model.eval()
    all_logits, all_labels = [], []
    for x, meta, y in loader:
        x, meta = x.to(device), meta.to(device)
        logits = model(x, meta)
        if tta:
            views = [x.flip(3), x.flip(2), x.flip(2).flip(3), x.transpose(2, 3)]
            logits = torch.stack([logits] + [model(v, meta) for v in views]).mean(0)
        all_logits.append(logits.float().cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    conf, correct = probs.max(1), probs.argmax(1) == labels
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def compute_metrics(probs: np.ndarray, labels: np.ndarray, classes: list[str], malignant: list[str] | None = None) -> dict:
    preds = probs.argmax(1)
    present = np.unique(labels)
    prec, rec, f1, support = precision_recall_fscore_support(labels, preds, labels=list(range(len(classes))), zero_division=0)
    out: dict = {
        "n_images": int(len(labels)),
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_precision": float(prec[present].mean()),
        "macro_recall": float(rec[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "ece": expected_calibration_error(probs, labels),
        "per_class": {},
        "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(len(classes)))).tolist(),
        "classes": list(classes),
    }
    for i, c in enumerate(classes):
        entry = {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
        if 0 < (labels == i).sum() < len(labels):
            entry["auc"] = float(roc_auc_score(labels == i, probs[:, i]))
        out["per_class"][c] = entry
    aucs = [v["auc"] for v in out["per_class"].values() if "auc" in v]
    if aucs:
        out["macro_auc"] = float(np.mean(aucs))

    mal_idx = [classes.index(c) for c in (malignant or []) if c in classes]
    is_mal = np.isin(labels, mal_idx)
    if mal_idx and 0 < is_mal.sum() < len(labels):
        out["malignant_auc"] = float(roc_auc_score(is_mal, probs[:, mal_idx].sum(1)))
    if "mel" in classes:
        mel = classes.index("mel")
        is_mel = labels == mel
        if 0 < is_mel.sum() < len(labels):
            out["melanoma_auc"] = float(roc_auc_score(is_mel, probs[:, mel]))
            thr = float(np.quantile(probs[is_mel, mel], 0.10))
            flagged = probs[:, mel] >= thr
            out["melanoma_at_90_sensitivity"] = {"threshold": thr, "sensitivity": float(flagged[is_mel].mean()),
                                                 "specificity": float((~flagged[~is_mel]).mean())}
    return out


def format_summary(m: dict, title: str = "Evaluation") -> str:
    """Plain-text report: headline scores, per-class table and confusion matrix."""
    lines = [title, "=" * len(title), f"images: {m['n_images']}",
             f"balanced accuracy: {m['balanced_accuracy']:.3f}   (main score; 1/{len(m['classes'])} = "
             f"{1 / len(m['classes']):.3f} would be random guessing)",
             f"accuracy: {m['accuracy']:.3f}   macro F1: {m['macro_f1']:.3f}   macro AUC: {m.get('macro_auc', float('nan')):.3f}"]
    if "malignant_auc" in m:
        lines.append(f"malignant-vs-benign AUC: {m['malignant_auc']:.3f}")
    if "melanoma_at_90_sensitivity" in m:
        s = m["melanoma_at_90_sensitivity"]
        lines.append(f"melanoma AUC: {m['melanoma_auc']:.3f}   specificity at 90% melanoma sensitivity: {s['specificity']:.3f}")
    lines.append(f"calibration error (ECE, lower is better): {m['ece']:.3f}")
    lines += ["", f"{'class':<16}{'precision':>10}{'recall':>9}{'F1':>8}{'AUC':>8}{'images':>8}"]
    for c, v in m["per_class"].items():
        lines.append(f"{c:<16}{v['precision']:>10.3f}{v['recall']:>9.3f}{v['f1']:>8.3f}"
                     f"{v.get('auc', float('nan')):>8.3f}{v['support']:>8}")
    cls = m["classes"]
    lines += ["", "confusion matrix (rows = true class, columns = predicted)", " " * 10 + "".join(f"{c[:7]:>8}" for c in cls)]
    for c, row in zip(cls, m["confusion_matrix"]):
        lines.append(f"{c[:9]:<10}" + "".join(f"{v:>8}" for v in row))
    return "\n".join(lines)


def write_reports(m: dict, out_dir: Path, history: list[dict] | None = None, extra: dict | None = None) -> None:
    """metrics.json (machine-readable), evaluation.txt, confusion_matrix.csv and a report.html you can open in a browser."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps({**m, **(extra or {})}, indent=2))
    (out_dir / "evaluation.txt").write_text(format_summary(m, "Test-set evaluation") + "\n")
    cls = m["classes"]
    rows = ["true\\predicted," + ",".join(cls)] + [f"{c}," + ",".join(map(str, r)) for c, r in zip(cls, m["confusion_matrix"])]
    (out_dir / "confusion_matrix.csv").write_text("\n".join(rows) + "\n")
    if history:
        keys = list(history[0])
        (out_dir / "training_history.csv").write_text(
            ",".join(keys) + "\n" + "\n".join(",".join("" if h.get(k) is None else str(h.get(k)) for k in keys) for h in history) + "\n")
    (out_dir / "report.html").write_text(_html_report(m, history, extra or {}))


def _html_report(m: dict, history: list[dict] | None, extra: dict) -> str:
    cls, cm = m["classes"], np.array(m["confusion_matrix"])
    row_tot = cm.sum(1, keepdims=True).clip(min=1)
    e = html.escape

    def cell(v, share):
        return f'<td style="background:rgba(15,118,110,{share:.2f});color:{"#fff" if share > .5 else "inherit"}">{v}</td>'

    cm_rows = "".join(f"<tr><th>{e(c)}</th>" + "".join(cell(v, v / row_tot[i, 0]) for v in cm[i]) + "</tr>" for i, c in enumerate(cls))
    pc_rows = "".join(f"<tr><th>{e(c)}</th><td>{v['precision']:.3f}</td><td>{v['recall']:.3f}</td><td>{v['f1']:.3f}</td>"
                      f"<td>{v.get('auc', float('nan')):.3f}</td><td>{v['support']}</td></tr>" for c, v in m["per_class"].items())
    hist = ""
    if history:
        keys = list(history[0])
        hist = ("<h2>Training history</h2><table><tr>" + "".join(f"<th>{e(k)}</th>" for k in keys) + "</tr>"
                + "".join("<tr>" + "".join(f"<td>{h[k]:.4f}</td>" if isinstance(h.get(k), float) else f"<td>{e(str(h.get(k)))}</td>"
                                           for k in keys) + "</tr>" for h in history) + "</table>")
    info = "".join(f"<tr><th>{e(str(k))}</th><td>{e(str(v))}</td></tr>" for k, v in extra.items() if not isinstance(v, (dict, list)))
    head = [("Balanced accuracy", m["balanced_accuracy"]), ("Accuracy", m["accuracy"]), ("Macro F1", m["macro_f1"]),
            ("Macro AUC", m.get("macro_auc")), ("Malignant AUC", m.get("malignant_auc")), ("Melanoma AUC", m.get("melanoma_auc"))]
    tiles = "".join(f'<div class="tile"><div>{k}</div><b>{v:.3f}</b></div>' for k, v in head if v is not None)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model evaluation</title><style>
body{{font:14px/1.5 system-ui,sans-serif;margin:24px;color:#12201f;background:#fff}} table{{border-collapse:collapse;margin:8px 0 20px}}
th,td{{border:1px solid #dbe4e2;padding:4px 8px;text-align:right}} th{{background:#eef3f2}}
.tiles{{display:flex;gap:10px;flex-wrap:wrap}} .tile{{border:1px solid #dbe4e2;border-radius:10px;padding:10px 14px}} .tile b{{font-size:22px}}
</style></head><body><h1>Model evaluation ({m['n_images']} test images)</h1><div class="tiles">{tiles}</div>
<p>Balanced accuracy is the main score: the average share of each class recognised correctly.
Random guessing would score {1 / len(cls):.3f}.</p>
<h2>Per class</h2><table><tr><th>class</th><th>precision</th><th>recall</th><th>F1</th><th>AUC</th><th>images</th></tr>{pc_rows}</table>
<h2>Confusion matrix</h2><p>Rows are the true class, columns the predicted class. Darker = larger share of that row.</p>
<table><tr><th>true \\ predicted</th>{"".join(f"<th>{e(c)}</th>" for c in cls)}</tr>{cm_rows}</table>
{hist}<h2>Details</h2><table>{info}</table></body></html>"""


def main(argv: list[str] | None = None) -> None:
    import argparse

    from dermaai import model_store
    from dermaai.device import pick_device
    from dermaai.model import load_checkpoint
    from dermaai.preprocessing import PreprocessingParams
    from training.config import TrainingConfig
    from training.dataset import LesionDataset, assign_splits, load_dataset
    from training.preprocessing import prepare_images

    cfg = TrainingConfig()
    p = argparse.ArgumentParser(description="Evaluate a trained model.")
    p.add_argument("--model", default=None, help="version name (default: the app's current model)")
    p.add_argument("--data", default=cfg.data_dir, help="dataset folder (default: data/raw)")
    p.add_argument("--all", action="store_true", help="evaluate every image instead of only the test split")
    p.add_argument("--models-dir", default=cfg.models_dir)
    p.add_argument("--batch-size", type=int, default=cfg.batch_size)
    p.add_argument("--out", default=None, help="folder for reports (default: print only)")
    a = p.parse_args(argv)

    path = model_store.model_path(a.model, cfg.path(a.models_dir))
    if not path:
        raise SystemExit(f"No trained model found in {cfg.path(a.models_dir)}. Train one with: python training/train.py")
    model, info = load_checkpoint(path)
    device = pick_device()
    model.to(device)
    classes = info["classes"]
    df = load_dataset(cfg.path(a.data))
    unknown = sorted(set(df["label"]) - set(classes))
    if unknown:
        print(f"[eval] ignoring classes this model does not know: {unknown}")
        df = df[df["label"].isin(classes)]
    df = prepare_images(df, cfg.path(cfg.processed_dir), max(cfg.processed_size, info.get("img_size", 0)))
    if not a.all:
        tc = info.get("training_config", {})
        df = assign_splits(df, cfg.path(tc.get("split_file", cfg.split_file)), tc.get("val_split", cfg.val_split),
                           tc.get("test_split", cfg.test_split), tc.get("seed", cfg.seed))
        df = df[df["split"] == "test"]
    df = df[df["usable"]]
    params = PreprocessingParams.from_dict(info.get("preprocessing"), img_size=info.get("img_size"))
    loader = DataLoader(LesionDataset(df, classes, eval_transform(params)), batch_size=a.batch_size)
    logits, labels = predict_loader(model, loader, device)
    probs = torch.softmax(logits / float(info.get("temperature") or 1.0), 1).numpy()
    m = compute_metrics(probs, labels.numpy(), classes, info.get("malignant_classes"))
    print(format_summary(m, f"Evaluation of {path.parent.name} on {len(df)} images"))
    if a.out:
        write_reports(m, Path(a.out))
        print(f"\nreports written to {a.out}")


if __name__ == "__main__":
    main()
