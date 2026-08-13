"""Separate DINO and BioCLIP preprocessing paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_dino_transform(
    model: Any,
    *,
    training: bool,
    conservative: bool = True,
    augmentation_profile: str | None = None,
) -> Callable:
    """Build a timm transform matching the encoder's pretrained data config."""
    if training and augmentation_profile == "fish_multiview_global":
        from fish_vlm.domain.transforms import build_dino_domain_transforms

        transform, _ = build_dino_domain_transforms(
            model, global_crops=1, local_crops=0
        )

        def first_view(image: Any) -> Any:
            return transform(image)[0]

        return first_view
    from timm.data import create_transform, resolve_model_data_config

    config = resolve_model_data_config(model)
    if training and not conservative:
        return create_transform(**config, is_training=True)
    return create_transform(**config, is_training=False)


def transform_fingerprint(transform: Any) -> str:
    """Return a deterministic representation for teacher-cache validation."""
    from fish_vlm.utils.hashing import stable_json_hash

    return stable_json_hash({"class": type(transform).__qualname__, "repr": repr(transform)})
