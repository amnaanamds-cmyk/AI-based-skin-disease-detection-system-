"""See and switch trained model versions.

    python training/manage_models.py list              # all versions, their scores, and which one the app uses
    python training/manage_models.py use model_v1      # make the app use model_v1 (e.g. to roll back)
    python training/manage_models.py info model_v2     # full details of one version
    python training/manage_models.py import path/to/checkpoint.pt   # register a model trained elsewhere

The running app notices a switch automatically on its next request; no restart or code change needed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dermaai import model_store
from training.config import TrainingConfig


def cmd_list(models_dir: Path) -> None:
    versions = model_store.list_versions(models_dir)
    if not versions:
        print(f"No trained models in {models_dir} yet. Train one with: python training/train.py")
        return
    current = model_store.current_version(models_dir)
    print(f"{'':2}{'version':<12}{'trained':<18}{'images':>8}{'classes':>9}{'bal.acc':>9}{'mel AUC':>9}  started from")
    for v in versions:
        info = model_store.read_info(v, models_dir)
        m = info.get("metrics", {})
        fmt = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "-"  # noqa: E731
        print(f"{'*' if v == current else ' '} {v:<12}{str(info.get('trained_at', '?')):<18}"
              f"{info.get('dataset', {}).get('images', '?'):>8}{len(info.get('classes', [])):>9}"
              f"{fmt(m.get('balanced_accuracy')):>9}{fmt(m.get('melanoma_auc')):>9}  {info.get('started_from', '?')}")
    print("\n* = used by the app. Scores are on each version's own test split.")


def cmd_import(src: Path, models_dir: Path) -> str:
    """Copy a checkpoint produced outside training/train.py into the version store."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("arch", "state_dict", "classes"):
        if key not in ckpt:
            raise SystemExit(f"{src} is not a DermaAI checkpoint (missing '{key}')")
    version = model_store.next_version_name(models_dir)
    out = model_store.version_dir(version, models_dir)
    out.mkdir(parents=True)
    shutil.copy2(src, out / model_store.MODEL_FILE)
    info = {k: v for k, v in ckpt.items() if k not in ("state_dict", "ood")}
    info.update(version_name=version, imported_from=str(src), imported_at=time.strftime("%Y-%m-%d %H:%M"))
    (out / model_store.INFO_FILE).write_text(json.dumps(info, indent=2, default=str))
    print(f"imported {src} as {version}")
    return version


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["list", "use", "info", "import"])
    p.add_argument("target", nargs="?", help="version name (use/info) or checkpoint path (import)")
    p.add_argument("--models-dir", default=TrainingConfig.models_dir)
    p.add_argument("--set-current", action="store_true", help="with import: also make it the app's model")
    a = p.parse_args(argv)
    models_dir = TrainingConfig().path(a.models_dir)

    if a.command == "list":
        cmd_list(models_dir)
    elif a.command == "use":
        if not a.target:
            p.error("which version? e.g. `use model_v1`")
        model_store.set_current(a.target, models_dir)
        print(f"the app now uses {a.target}")
    elif a.command == "info":
        name = a.target or model_store.current_version(models_dir)
        print(json.dumps(model_store.read_info(name, models_dir), indent=2))
    elif a.command == "import":
        if not a.target:
            p.error("which checkpoint file?")
        version = cmd_import(Path(a.target), models_dir)
        if a.set_current or model_store.current_version(models_dir) == version:
            model_store.set_current(version, models_dir)
            print(f"the app now uses {version}")


if __name__ == "__main__":
    main()
