"""Stage contracts for trainable components."""

from __future__ import annotations

from torch import nn

from fish_vlm.models.bioclip import configure_bioclip_tuning
from fish_vlm.models.dino import set_dino_trainable_scope
from fish_vlm.models.multimodal import FishMultimodalModel


STAGE_SCOPES = {
    "projection_only": "frozen",
    "final_block": "final_block",
    "joint_supervised_text": "final_block",
    "joint_alignment_preserving": "frozen",
    "joint_alignment_final_block": "final_block",
    "bioclip_linear_probe": "frozen",
    "bioclip_adapter": "frozen",
    "bioclip_partial_finetune": "frozen",
    "bioclip_full_finetune": "frozen",
}


def configure_training_stage(
    model: FishMultimodalModel,
    stage: str,
    *,
    model_config: dict | None = None,
) -> None:
    """Set requires-grad flags exactly for one named scientific stage."""
    if stage not in STAGE_SCOPES:
        raise ValueError(f"Unknown training stage: {stage}")
    model.requires_grad_(False)
    set_dino_trainable_scope(model.dino, STAGE_SCOPES[stage])
    if stage in {
        "projection_only",
        "final_block",
        "joint_supervised_text",
        "joint_alignment_preserving",
        "joint_alignment_final_block",
    }:
        model.projector.requires_grad_(True)
        model.logit_scale.requires_grad_(True)
    if stage in {
        "joint_supervised_text",
        "joint_alignment_preserving",
        "joint_alignment_final_block",
    }:
        if model.supervised_head is None:
            raise ValueError("Joint stage requires the supervised head")
        model.supervised_head.requires_grad_(True)
    if stage == "bioclip_adapter":
        if model.bioclip_adapter is None:
            raise ValueError("BioCLIP adapter stage requires an adapter")
        model.bioclip_adapter.requires_grad_(True)
    if stage == "bioclip_linear_probe":
        if model.bioclip_classifier is None:
            raise ValueError("BioCLIP linear probe requires its classifier")
        model.bioclip_classifier.requires_grad_(True)
    if stage in {"bioclip_partial_finetune", "bioclip_full_finetune"}:
        if model.bioclip is None:
            raise ValueError("BioCLIP fine-tuning requires its image encoder")
        config = model_config or {}
        tuning_mode = str(
            config.get(
                "tuning_mode",
                "partial_finetune"
                if stage == "bioclip_partial_finetune"
                else "full_finetune",
            )
        )
        expected_mode = (
            "partial_finetune"
            if stage == "bioclip_partial_finetune"
            else "full_finetune"
        )
        if tuning_mode != expected_mode:
            raise ValueError(
                f"{stage} requires model.tuning_mode={expected_mode}"
            )
        configure_bioclip_tuning(
            model.bioclip,
            tuning_mode,
            unfreeze_last_blocks=int(
                config.get("unfreeze_last_blocks", 1)
            ),
        )
        if model.bioclip_classifier is None:
            raise ValueError("BioCLIP fine-tuning requires its classifier")
        model.bioclip_classifier.requires_grad_(True)
        if model.bioclip_adapter is not None:
            model.bioclip_adapter.requires_grad_(True)
    elif model.bioclip is not None:
        model.bioclip.eval()
        model.bioclip.requires_grad_(False)


def trainable_parameter_count(module: nn.Module) -> int:
    """Count trainable scalar parameters."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
