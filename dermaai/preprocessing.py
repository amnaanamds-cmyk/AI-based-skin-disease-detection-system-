"""Image preprocessing shared by training and prediction.

There is exactly one definition of how an image becomes a model input, and
both the training module and the app use it. The parameters (image size,
normalisation, resize ratio) are saved with every trained model, so a model is
always served with the preprocessing it was trained with.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class PreprocessingParams:
    img_size: int = 224                       # model input is img_size x img_size
    resize_ratio: float = 1.14                # resize shorter side to img_size * ratio, then centre-crop
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict | None, img_size: int | None = None) -> "PreprocessingParams":
        d = dict(d or {})
        if img_size and "img_size" not in d:
            d["img_size"] = img_size
        for k in ("mean", "std"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_image(source) -> Image.Image:
    """Open a path or file-like object as an upright RGB image (applies EXIF rotation)."""
    img = source if isinstance(source, Image.Image) else Image.open(source)
    return ImageOps.exif_transpose(img).convert("RGB")


def eval_transform(params: PreprocessingParams | int) -> T.Compose:
    """Deterministic transform used for validation, testing and prediction."""
    p = params if isinstance(params, PreprocessingParams) else PreprocessingParams(img_size=int(params))
    return T.Compose([
        T.Resize(int(p.img_size * p.resize_ratio)),
        T.CenterCrop(p.img_size),
        T.ToTensor(),
        T.Normalize(p.mean, p.std),
    ])


def train_transform(params: PreprocessingParams | int) -> T.Compose:
    """Eval preprocessing plus random augmentation (training only)."""
    p = params if isinstance(params, PreprocessingParams) else PreprocessingParams(img_size=int(params))
    return T.Compose([
        T.RandomResizedCrop(p.img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(180)], p=0.5),
        # Dermoscopy and phone photos vary widely in lighting and white balance.
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.04),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(p.mean, p.std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])


def resize_for_cache(img: Image.Image, shorter_side: int) -> Image.Image:
    """Shrink so the shorter side is ``shorter_side`` px (never enlarges). Used to cache processed data."""
    s = shorter_side / min(img.size)
    if s >= 1:
        return img
    return img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith(".")
