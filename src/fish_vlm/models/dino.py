"""DINOv3 loading and trainable-scope control through timm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

DINO_MODEL_ALIASES = {
    # User-facing shorthand retained for configuration compatibility with the
    # canonical current timm registry name.
    "dinov3_vitb16": "vit_base_patch16_dinov3",
}


def load_dino(config: dict[str, Any]) -> tuple[nn.Module, int, str]:
    """Create a DINO model, optionally restore weights, and infer feature size."""
    import timm

    configured_name = str(config["name"])
    model_name = DINO_MODEL_ALIASES.get(configured_name, configured_name)
    model = timm.create_model(
        model_name,
        pretrained=bool(config.get("pretrained", True)),
        num_classes=0,
    )
    source = f"timm:{model_name}:pretrained={bool(config.get('pretrained', True))}"
    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state", checkpoint.get("state_dict", checkpoint))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Incompatible DINO checkpoint; missing={missing}, unexpected={unexpected}")
        source = str(Path(checkpoint_path).resolve())
    feature_dim = int(getattr(model, "num_features", 0))
    if feature_dim <= 0:
        raise RuntimeError("timm model did not expose a valid num_features")
    set_dino_trainable_scope(model, config.get("trainable_scope", "frozen"))
    return model, feature_dim, source


def _last_block(model: nn.Module) -> nn.Module:
    for attribute in ("blocks", "stages", "layers"):
        sequence = getattr(model, attribute, None)
        if sequence is not None and len(sequence):
            return sequence[-1]
    raise ValueError("Cannot locate final DINO block for this timm architecture")


def set_dino_trainable_scope(model: nn.Module, scope: str) -> None:
    """Freeze, unfreeze all, or expose final block and normalisation only."""
    model.requires_grad_(scope == "full")
    if scope == "frozen":
        model.eval()
        return
    if scope == "full":
        model.train()
        return
    if scope != "final_block":
        raise ValueError(f"Unsupported DINO trainable scope: {scope}")
    model.requires_grad_(False)
    _last_block(model).requires_grad_(True)
    for name in ("norm", "fc_norm"):
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            module.requires_grad_(True)


def pooled_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Use timm's established pooled feature interface.

    The pooling strategy is ``forward_features_then_forward_head_pre_logits``
    when available, otherwise the already pooled ``forward_features`` tensor.
    """
    features = model.forward_features(images)
    if hasattr(model, "forward_head"):
        features = model.forward_head(features, pre_logits=True)
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim != 2:
        raise RuntimeError(f"Expected pooled [batch, features] tensor, got {tuple(features.shape)}")
    return features


POOLING_STRATEGY = "forward_features_then_forward_head_pre_logits"
