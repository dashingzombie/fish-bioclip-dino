"""Explicit weighted composition of independently configured losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fish_vlm.losses.consistency import branch_consistency_loss
from fish_vlm.losses.image_teacher import cosine_teacher_loss, symmetric_contrastive_teacher_loss
from fish_vlm.losses.supervised import supervised_species_loss
from fish_vlm.losses.text_classification import text_classification_loss
from fish_vlm.models.multimodal import ModelOutput


@dataclass
class LossResult:
    """Total objective plus named unweighted components."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]


def compute_total_loss(
    output: ModelOutput,
    targets: torch.Tensor,
    config: dict,
    *,
    teacher_embeddings: torch.Tensor | None = None,
) -> LossResult:
    """Compute enabled losses with explicit weights and no averaging."""
    components: dict[str, torch.Tensor] = {}
    total = output.dino_text_logits.new_zeros((), dtype=torch.float32)

    def add(name: str, value: torch.Tensor) -> None:
        nonlocal total
        section = config[name]
        components[name] = value
        total = total + float(section["weight"]) * value

    section = config["dino_text_classification"]
    if section.get("enabled", False):
        add("dino_text_classification", text_classification_loss(output.dino_text_logits, targets))
    section = config["bioclip_image_teacher"]
    if section.get("enabled", False):
        if teacher_embeddings is None:
            raise ValueError("Enabled image-teacher loss requires teacher embeddings")
        method = section.get("method", "cosine")
        if method == "cosine":
            value = cosine_teacher_loss(output.projected_features, teacher_embeddings)
        elif method == "symmetric_contrastive":
            value = symmetric_contrastive_teacher_loss(
                output.projected_features,
                teacher_embeddings,
                float(section.get("temperature", 0.07)),
            )
        else:
            raise ValueError(f"Unknown teacher loss method: {method}")
        add("bioclip_image_teacher", value)
    section = config["supervised_species"]
    if section.get("enabled", False):
        if output.supervised_logits is None:
            raise ValueError("Enabled supervised loss requires the supervised head")
        add("supervised_species", supervised_species_loss(output.supervised_logits, targets))
    section = config["native_bioclip_text"]
    if section.get("enabled", False):
        if output.bioclip_logits is None:
            raise ValueError("Enabled native BioCLIP loss requires its image branch")
        add("native_bioclip_text", text_classification_loss(output.bioclip_logits, targets))
    section = config["branch_consistency"]
    if section.get("enabled", False):
        if output.bioclip_logits is None:
            raise ValueError("Enabled branch consistency requires BioCLIP logits")
        add(
            "branch_consistency",
            branch_consistency_loss(
                output.dino_text_logits,
                output.bioclip_logits,
                section.get("method", "js"),
            ),
        )
    if not components:
        raise ValueError("At least one loss must be enabled")
    return LossResult(total=total, components=components)

