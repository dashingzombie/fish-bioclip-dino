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


def encode_bioclip_images(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Encode and normalise images in float32, preserving tuning gradients."""
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


def _resolve_module_path(module: nn.Module, path: str) -> Any:
    value: Any = module
    for component in path.split("."):
        value = getattr(value, component, None)
        if value is None:
            return None
    return value


def bioclip_visual_blocks(model: nn.Module) -> list[nn.Module]:
    """Locate the ordered image-encoder blocks used by common OpenCLIP towers."""
    visual = getattr(model, "visual", None)
    if not isinstance(visual, nn.Module):
        raise ValueError("BioCLIP model does not expose a visual encoder")
    for path in (
        "trunk.blocks",
        "blocks",
        "transformer.resblocks",
        "trunk.stages",
        "stages",
    ):
        sequence = _resolve_module_path(visual, path)
        if isinstance(sequence, (nn.ModuleList, nn.Sequential, list, tuple)):
            blocks = list(sequence)
            if blocks and all(isinstance(block, nn.Module) for block in blocks):
                return blocks
    raise ValueError("Cannot locate BioCLIP image-encoder blocks")


def configure_bioclip_tuning(
    model: nn.Module,
    tuning_mode: str,
    *,
    unfreeze_last_blocks: int = 1,
) -> None:
    """Freeze BioCLIP, unfreeze final visual blocks, or unfreeze the visual tower."""
    model.requires_grad_(False)
    if tuning_mode in {"frozen", "linear_probe", "adapter"}:
        model.eval()
        return
    visual = getattr(model, "visual", None)
    if not isinstance(visual, nn.Module):
        raise ValueError("BioCLIP tuning requires a visual encoder")
    if tuning_mode == "full_finetune":
        visual.requires_grad_(True)
        return
    if tuning_mode != "partial_finetune":
        raise ValueError(f"Unknown BioCLIP tuning mode: {tuning_mode}")
    blocks = bioclip_visual_blocks(model)
    count = int(unfreeze_last_blocks)
    if count < 1 or count > len(blocks):
        raise ValueError(
            "unfreeze_last_blocks must be between one and the visual block count"
        )
    for block in blocks[-count:]:
        block.requires_grad_(True)
