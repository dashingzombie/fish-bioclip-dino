"""Frozen BioCLIP image/text encoder loading."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def load_bioclip(checkpoint: str) -> tuple[nn.Module, Any, Any, Any, int]:
    """Load BioCLIP, transforms and tokenizer; infer embedding dimension."""
    import open_clip

    model, train_transform, eval_transform = open_clip.create_model_and_transforms(checkpoint)
    tokenizer = open_clip.get_tokenizer(checkpoint)
    model.eval()
    model.requires_grad_(False)
    dimension = _embedding_dimension(model)
    return model, train_transform, eval_transform, tokenizer, dimension


def _embedding_dimension(model: nn.Module) -> int:
    projection = getattr(model, "text_projection", None)
    if isinstance(projection, torch.Tensor):
        return int(projection.shape[-1])
    visual = getattr(model, "visual", None)
    output_dim = getattr(visual, "output_dim", None)
    if output_dim:
        return int(output_dim)
    raise RuntimeError("Cannot infer BioCLIP embedding dimension")


@torch.no_grad()
def encode_bioclip_images(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Encode and normalise images in float32."""
    return F.normalize(model.encode_image(images).float(), dim=-1)


@torch.no_grad()
def encode_bioclip_text(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Encode and normalise token batches in float32."""
    return F.normalize(model.encode_text(tokens).float(), dim=-1)


def assert_frozen_bioclip(model: nn.Module) -> None:
    """Fail if a BioCLIP parameter is accidentally trainable."""
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"BioCLIP must remain frozen; trainable parameters: {trainable[:5]}")

