"""Conservative fish-preserving multi-view image augmentation."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image


class GaussianNoise:
    """Small sensor-like noise after conversion to a tensor."""

    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma = float(sigma)

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        return (value + torch.randn_like(value) * self.sigma).clamp(0.0, 1.0)


def _normalisation_from_transform(
    transform: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    for item in getattr(transform, "transforms", []):
        mean = getattr(item, "mean", None)
        std = getattr(item, "std", None)
        if mean is not None and std is not None:
            return tuple(float(x) for x in mean), tuple(float(x) for x in std)
    return (
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )


def _image_size(value: Any, fallback: int = 224) -> int:
    if isinstance(value, (tuple, list)):
        return int(value[-1])
    return int(value or fallback)


def _view_transform(
    *,
    size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    scale: tuple[float, float],
    blur_probability: float,
) -> Any:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size,
                scale=scale,
                ratio=(0.75, 1.3333333333),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.RandomGrayscale(p=0.03),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 1.5))],
                p=blur_probability,
            ),
            transforms.ToTensor(),
            transforms.RandomApply([GaussianNoise(0.01)], p=0.15),
            transforms.RandomErasing(
                p=0.10,
                scale=(0.02, 0.08),
                ratio=(0.5, 2.0),
                value="random",
            ),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


class FishMultiCropTransform:
    """Two whole-fish global crops plus several constrained local views."""

    def __init__(
        self,
        *,
        size: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        global_crops: int,
        local_crops: int,
    ) -> None:
        self.global_crops = int(global_crops)
        self.local_crops = int(local_crops)
        self.global_transform = _view_transform(
            size=size,
            mean=mean,
            std=std,
            scale=(0.65, 1.0),
            blur_probability=0.20,
        )
        self.local_transform = _view_transform(
            size=size,
            mean=mean,
            std=std,
            scale=(0.35, 0.65),
            blur_probability=0.35,
        )

    def __call__(self, image: Image.Image) -> list[torch.Tensor]:
        return [self.global_transform(image) for _ in range(self.global_crops)] + [
            self.local_transform(image) for _ in range(self.local_crops)
        ]


class FishValidationViews:
    """Deterministic original and horizontal-flip validation views."""

    def __init__(
        self,
        *,
        size: int,
        mean: tuple[float, ...],
        std: tuple[float, ...],
    ) -> None:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.base = transforms.Compose(
            [
                transforms.Resize(size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: Image.Image) -> list[torch.Tensor]:
        from torchvision.transforms import functional as functional

        return [self.base(image), self.base(functional.hflip(image))]


def build_dino_domain_transforms(
    model: Any,
    *,
    global_crops: int,
    local_crops: int,
) -> tuple[Any, Any]:
    from timm.data import resolve_model_data_config

    config = resolve_model_data_config(model)
    size = _image_size(config.get("input_size", (3, 224, 224)))
    mean = tuple(float(x) for x in config["mean"])
    std = tuple(float(x) for x in config["std"])
    return (
        FishMultiCropTransform(
            size=size,
            mean=mean,
            std=std,
            global_crops=global_crops,
            local_crops=local_crops,
        ),
        FishValidationViews(size=size, mean=mean, std=std),
    )


def build_bioclip_domain_transforms(
    model: Any,
    eval_transform: Any,
) -> tuple[Any, Any]:
    mean, std = _normalisation_from_transform(eval_transform)
    size = _image_size(getattr(getattr(model, "visual", None), "image_size", 224))
    return (
        FishMultiCropTransform(
            size=size,
            mean=mean,
            std=std,
            global_crops=2,
            local_crops=0,
        ),
        FishValidationViews(size=size, mean=mean, std=std),
    )
