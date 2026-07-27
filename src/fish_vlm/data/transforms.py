"""Separate DINO and BioCLIP preprocessing paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_dino_transform(model: Any, *, training: bool, conservative: bool = True) -> Callable:
    """Build a timm transform matching the encoder's pretrained data config."""
    from timm.data import create_transform, resolve_model_data_config

    config = resolve_model_data_config(model)
    if training and not conservative:
        return create_transform(**config, is_training=True)
    return create_transform(**config, is_training=False)


def transform_fingerprint(transform: Any) -> str:
    """Return a deterministic representation for teacher-cache validation."""
    from fish_vlm.utils.hashing import stable_json_hash

    return stable_json_hash({"class": type(transform).__qualname__, "repr": repr(transform)})

