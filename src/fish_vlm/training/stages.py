"""Stage contracts for trainable components."""

from __future__ import annotations

from torch import nn

from fish_vlm.models.dino import set_dino_trainable_scope
from fish_vlm.models.multimodal import FishMultimodalModel


STAGE_SCOPES = {
    "projection_only": "frozen",
    "final_block": "final_block",
    "joint_supervised_text": "final_block",
    "bioclip_adapter": "frozen",
}


def configure_training_stage(model: FishMultimodalModel, stage: str) -> None:
    """Set requires-grad flags exactly for one named scientific stage."""
    if stage not in STAGE_SCOPES:
        raise ValueError(f"Unknown training stage: {stage}")
    model.requires_grad_(False)
    set_dino_trainable_scope(model.dino, STAGE_SCOPES[stage])
    if stage in {"projection_only", "final_block", "joint_supervised_text"}:
        model.projector.requires_grad_(True)
        model.logit_scale.requires_grad_(True)
    if stage == "joint_supervised_text":
        if model.supervised_head is None:
            raise ValueError("Joint stage requires the supervised head")
        model.supervised_head.requires_grad_(True)
    if stage == "bioclip_adapter":
        if model.bioclip_adapter is None:
            raise ValueError("BioCLIP adapter stage requires an adapter")
        model.bioclip_adapter.requires_grad_(True)
    if model.bioclip is not None:
        model.bioclip.eval()
        model.bioclip.requires_grad_(False)


def trainable_parameter_count(module: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

