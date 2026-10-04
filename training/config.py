"""All training settings in one place.

Change a default here, or override any setting for a single run on the command
line, e.g.::

    python training/train.py --epochs 20 --img-size 224 --model-type efficientnet_b2

Every setting below has a matching ``--option`` (underscores become dashes).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root, so paths work from any working directory


@dataclass
class TrainingConfig:
    # ---------------- data ----------------
    data_dir: str = "data/raw"                 # your dataset (see data/README.md for the layout)
    processed_dir: str = "data/processed"      # resized copies + manifest, created automatically
    processed_size: int = 288                  # cached images: shorter side in pixels (>= img_size)

    # Class names come from your folder names / label column. Leave empty to use every class found,
    # or list names to train on just those classes (in that order).
    class_names: list[str] = field(default_factory=list)
    min_images_per_class: int = 20             # classes with fewer images stop training with an error

    # Classes the app should treat as cancer or pre-cancer when deciding how urgent a result is.
    malignant_classes: list[str] = field(default_factory=lambda: ["mel", "bcc", "akiec", "scc"])

    # How the app should describe classes it doesn't already know (the 8 standard ones are built in).
    # Saved inside the trained model, so the app needs no changes. Example:
    #   "eczema": {"name": "Eczema", "malignancy": "benign", "risk": "low",
    #              "summary": "Itchy, inflamed skin.", "action": "See a doctor if it spreads or weeps."}
    class_info: dict[str, dict] = field(default_factory=dict)

    # ---------------- splits ----------------
    val_split: float = 0.15                    # share of data used to pick the best epoch
    test_split: float = 0.15                   # share held out for the final, unbiased evaluation
    seed: int = 42                             # same seed + same data = same split and same results
    split_file: str = "data/splits.csv"        # remembers each image's split across runs (don't delete)
    resplit: bool = False                      # ignore split_file and make a fresh split (breaks comparability)

    # ---------------- model ----------------
    model_type: str = "efficientnet_b0"        # any timm CNN, e.g. efficientnet_b2, resnet50, mobilenetv3_large_100
    pretrained: bool = True                    # start from ImageNet weights (strongly recommended)
    use_metadata: bool = True                  # also learn from age / sex / body site when provided
    img_size: int = 192                        # model input size; larger = more detail but slower

    # Retraining: "" trains a fresh model from ImageNet weights; "current" or "model_v2" starts from
    # an existing trained model (fine-tuning) — faster, and keeps what it learned.
    init_from: str = ""

    # ---------------- optimisation ----------------
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-4                # backbone; the new classification head uses 10x this
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 4                          # stop early after this many epochs without improvement
    balance_power: float = 0.5                 # 0 = no rebalancing, 1 = fully equal classes per batch
    loss_weight_power: float = 0.25            # extra loss weight for rare classes
    meta_dropout: float = 0.3                  # hide each metadata field this often, so the model copes without it
    max_hours: float = 0.0                     # 0 = no limit; otherwise stop starting new epochs after this time

    # ---------------- output ----------------
    models_dir: str = "models/trained"         # versions are saved as models/trained/model_v1, model_v2, ...
    checkpoint_dir: str = "models/checkpoints" # in-progress training state, for --resume
    promote: str = "if_better"                 # make the new model the app's model: if_better | always | never
    resume: bool = False                       # continue an interrupted run from checkpoint_dir/last.pt

    # ---------------- hardware ----------------
    device: str = "auto"                       # auto | cpu | cuda | mps
    num_workers: int = 2                       # parallel image-loading processes
    threads: int = 0                           # CPU threads for PyTorch (0 = PyTorch default)
    log_every: int = 50                        # print progress every N batches (0 = once per epoch)

    def path(self, value: str) -> Path:
        """Resolve a configured path relative to the project root."""
        p = Path(value)
        return p if p.is_absolute() else ROOT / p


def _add_argument(parser: argparse.ArgumentParser, f) -> None:
    name = "--" + f.name.replace("_", "-")
    default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
    if isinstance(default, dict):  # edited in this file only
        return
    if isinstance(default, bool):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
    elif isinstance(default, list):
        parser.add_argument(name, nargs="*", default=default)
    else:
        parser.add_argument(name, type=type(default), default=default)


def parse_config(argv: list[str] | None = None, description: str = "") -> TrainingConfig:
    """Build a TrainingConfig from the defaults above plus any command-line overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", dest="data_dir", help="alias for --data-dir")
    for f in fields(TrainingConfig):
        _add_argument(parser, f)
    args = vars(parser.parse_args(argv))
    if args.get("data_dir") is None:
        args["data_dir"] = TrainingConfig.data_dir
    if args["promote"] not in {"if_better", "always", "never"}:
        parser.error("--promote must be one of: if_better, always, never")
    return TrainingConfig(**{k: v for k, v in args.items() if k in TrainingConfig.__dataclass_fields__})
