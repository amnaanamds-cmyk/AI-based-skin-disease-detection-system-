"""Export a checkpoint to ONNX for mobile / edge deployment.

    python -m dermaai.export --out model.onnx                       # the app's current model
    python -m dermaai.export --checkpoint models/trained/model_v1/model.pt --out model_v1.onnx
"""

from __future__ import annotations

import argparse

import torch

from .config import META_DIM
from .model import load_checkpoint


class _Wrapper(torch.nn.Module):
    def __init__(self, model, temperature: float):
        super().__init__()
        self.model, self.temperature = model, temperature

    def forward(self, image, meta):
        return torch.softmax(self.model(image, meta) / self.temperature, dim=1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=None, help="default: models/trained/<CURRENT>/model.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--opset", type=int, default=17)
    a = p.parse_args()
    from .model_store import model_path
    model, info = load_checkpoint(a.checkpoint or model_path())
    size = int(info.get("img_size", 224))
    wrapper = _Wrapper(model, float(info.get("temperature") or 1.0)).eval()
    torch.onnx.export(
        wrapper, (torch.randn(1, 3, size, size), torch.zeros(1, META_DIM)), a.out,
        input_names=["image", "meta"], output_names=["probabilities"],
        dynamic_axes={"image": {0: "batch"}, "meta": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=a.opset,
    )
    print(f"exported {a.out} (input 1x3x{size}x{size}, meta 1x{META_DIM})")


if __name__ == "__main__":
    main()
