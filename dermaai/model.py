"""Image + clinical-metadata fusion network built on a timm CNN backbone."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn

from .config import CLASSES, META_DIM

# ImageNet weights mirrored on timm's GitHub releases. Used when the Hugging Face Hub (timm's
# default source) is unreachable, e.g. on locked-down training machines.
_GH = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/"
GITHUB_WEIGHTS = {
    "efficientnet_b0": _GH + "efficientnet_b0_ra-3dd342df.pth",
    "efficientnet_b2": _GH + "efficientnet_b2_ra-bcdf34b7.pth",
    "mobilenetv3_large_100": _GH + "mobilenetv3_large_100_ra-f55367f5.pth",
    "resnet50": _GH + "resnet50_ram-a26f946b.pth",
}


def _pretrained_overlay(arch: str) -> dict | None:
    url = GITHUB_WEIGHTS.get(arch)
    if not url:
        return None
    cache = Path(torch.hub.get_dir()) / "dermaai"
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / url.rsplit("/", 1)[1]
    if not dst.exists():
        torch.hub.download_url_to_file(url, str(dst), progress=False)
    return {"file": str(dst)}


class DermNet(nn.Module):
    """CNN backbone whose pooled features are fused with patient metadata.

    Metadata (age, sex, body site) carries real diagnostic signal — e.g. BCC
    concentrates on the face of older patients — and is optional at inference.
    """

    def __init__(
        self,
        arch: str = "efficientnet_b0",
        num_classes: int = len(CLASSES),
        pretrained: bool = False,
        use_meta: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.arch = arch
        self.use_meta = use_meta
        overlay = _pretrained_overlay(arch) if pretrained else None
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg",
                                          **({"pretrained_cfg_overlay": overlay} if overlay else {}))
        # Some backbones (e.g. MobileNetV3) add a conv head after the pooled features, so the
        # pre-logits width can differ from num_features.
        feat_dim = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        meta_out = 0
        if use_meta:
            meta_out = 64
            self.meta = nn.Sequential(
                nn.Linear(META_DIM, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2),
                nn.Linear(128, meta_out), nn.BatchNorm1d(meta_out), nn.SiLU(),
            )
        self.dropout = nn.Dropout(drop_rate)
        self.head = nn.Linear(feat_dim + meta_out, num_classes)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled image features (before metadata fusion), used for out-of-distribution detection."""
        return self.backbone.forward_head(self.backbone.forward_features(x), pre_logits=True)

    def forward(self, x: torch.Tensor, meta: torch.Tensor | None = None, return_features: bool = False):
        fmap = self.backbone.forward_features(x)
        pooled = self.backbone.forward_head(fmap, pre_logits=True)
        if self.use_meta:
            if meta is None:
                meta = torch.zeros(x.shape[0], META_DIM, device=x.device, dtype=pooled.dtype)
            pooled = torch.cat([pooled, self.meta(meta)], dim=1)
        logits = self.head(self.dropout(pooled))
        return (logits, fmap) if return_features else logits


def save_checkpoint(path: str | Path, model: DermNet, **extra) -> None:
    payload = {
        "arch": model.arch,
        "use_meta": model.use_meta,
        "num_classes": model.head.out_features,
        "state_dict": model.state_dict(),
        **extra,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[DermNet, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = DermNet(arch=ckpt["arch"], num_classes=ckpt["num_classes"], use_meta=ckpt.get("use_meta", True))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model, meta
