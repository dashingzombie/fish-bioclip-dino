"""Explicit weighted composition of independently configured losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fish_vlm.losses.consistency import branch_consistency_loss
from fish_vlm.losses.image_teacher import cosine_teacher_loss, symmetric_contrastive_teacher_loss
from fish_vlm.losses.hierarchy import hierarchical_cross_entropy
from fish_vlm.losses.hard_negatives import hard_negative_cross_entropy
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
    representation_teacher_embeddings: torch.Tensor | None = None,
    genus_class_indices: list[int] | torch.Tensor | None = None,
    family_class_indices: list[int] | torch.Tensor | None = None,
    hard_negative_context: dict[str, object] | None = None,
) -> LossResult:
    """Compute enabled losses with explicit weights and no averaging."""
    components: dict[str, torch.Tensor] = {}
    total = output.dino_text_logits.new_zeros((), dtype=torch.float32)

    def add(name: str, value: torch.Tensor) -> None:
        nonlocal total
        section = config[name]
        components[name] = value
        total = total + float(section["weight"]) * value

    def text_loss(
        logits: torch.Tensor,
        section: dict,
    ) -> torch.Tensor:
        hard = section.get("hard_negatives", {})
        if not hard.get("enabled", False):
            return text_classification_loss(logits, targets)
        strategy = str(hard["strategy"])
        context = hard_negative_context or {}
        group_key = (
            "genus_groups"
            if strategy == "same_genus"
            else "family_groups"
        )
        return hard_negative_cross_entropy(
            logits,
            targets,
            strategy=strategy,
            top_k=int(hard.get("top_k", 5)),
            class_groups=context.get(group_key),  # type: ignore[arg-type]
            prototype_similarity=context.get(  # type: ignore[arg-type]
                "visual_similarity"
                if strategy == "visually_similar"
                else "prototype_similarity"
            ),
        )

    section = config.get("dino_text_classification", {})
    if section.get("enabled", False):
        add(
            "dino_text_classification",
            text_loss(output.dino_text_logits, section),
        )
    section = config.get("bioclip_image_teacher", {})
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
    section = config.get("supervised_species", {})
    if section.get("enabled", False):
        if output.supervised_logits is None:
            raise ValueError("Enabled supervised loss requires the supervised head")
        add("supervised_species", supervised_species_loss(output.supervised_logits, targets))
    section = config.get("native_bioclip_text", {})
    if section.get("enabled", False):
        if output.bioclip_logits is None:
            raise ValueError("Enabled native BioCLIP loss requires its image branch")
        add(
            "native_bioclip_text",
            text_loss(output.bioclip_logits, section),
        )
    section = config.get("bioclip_supervised_species", {})
    if section.get("enabled", False):
        if output.bioclip_supervised_logits is None:
            raise ValueError(
                "Enabled BioCLIP supervised loss requires its classifier"
            )
        add(
            "bioclip_supervised_species",
            supervised_species_loss(
                output.bioclip_supervised_logits, targets
            ),
        )
    section = config.get("bioclip_pretrained_distillation", {})
    if section.get("enabled", False):
        if (
            output.bioclip_original_features is None
            or teacher_embeddings is None
        ):
            raise ValueError(
                "BioCLIP distillation requires current and pretrained embeddings"
            )
        add(
            "bioclip_pretrained_distillation",
            cosine_teacher_loss(
                output.bioclip_original_features,
                teacher_embeddings,
            ),
        )
    section = config.get("representation_distillation", {})
    if section.get("enabled", False):
        if representation_teacher_embeddings is None:
            raise ValueError(
                "Representation distillation requires stage-2 embeddings"
            )
        add(
            "representation_distillation",
            cosine_teacher_loss(
                output.projected_features,
                representation_teacher_embeddings,
            ),
        )
    for name, mapping in (
        ("genus_supervised", genus_class_indices),
        ("family_supervised", family_class_indices),
    ):
        section = config.get(name, {})
        if not section.get("enabled", False):
            continue
        if output.supervised_logits is None or mapping is None:
            raise ValueError(
                f"Enabled {name} requires supervised logits and taxonomy"
            )
        add(
            name,
            hierarchical_cross_entropy(
                output.supervised_logits,
                targets,
                mapping,
            ),
        )
    section = config.get("branch_consistency", {})
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
