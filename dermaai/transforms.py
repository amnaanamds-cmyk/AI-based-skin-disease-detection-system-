"""Train / eval image transforms."""

from __future__ import annotations

from torchvision import transforms as T

from .config import IMAGENET_MEAN, IMAGENET_STD


def train_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(180)], p=0.5),
        # Dermoscopy and phone photos vary widely in lighting and white balance.
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.04),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])


def eval_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
