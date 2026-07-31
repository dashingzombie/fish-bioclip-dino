"""Taxonomy-level objectives derived from ordered species logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hierarchical_cross_entropy(
    species_logits: torch.Tensor,
    species_targets: torch.Tensor,
    class_to_group: list[int] | torch.Tensor,
) -> torch.Tensor:
    """Aggregate species probabilities into taxonomy groups where known.

    A mapping value of ``-1`` marks a species without metadata. Samples whose
    target has no mapping are excluded from this auxiliary loss.
    """
    mapping = torch.as_tensor(
        class_to_group,
        device=species_logits.device,
        dtype=torch.long,
    )
    if mapping.ndim != 1 or len(mapping) != species_logits.shape[1]:
        raise ValueError("Taxonomy mapping must match species-logit columns")
    if len(mapping) == 0 or mapping.min().item() < -1:
        raise ValueError("Taxonomy group indices must be -1 or non-negative")
    known_classes = mapping >= 0
    if not bool(known_classes.any()):
        return species_logits.float().sum() * 0.0
    group_count = int(mapping[known_classes].max().item()) + 1
    probabilities = torch.softmax(species_logits.float(), dim=-1)
    group_probabilities = probabilities.new_zeros(
        (len(probabilities), group_count)
    )
    group_probabilities.scatter_add_(
        1,
        mapping[known_classes].unsqueeze(0).expand(len(probabilities), -1),
        probabilities[:, known_classes],
    )
    group_targets = mapping[species_targets.long()]
    known_targets = group_targets >= 0
    if not bool(known_targets.any()):
        return species_logits.float().sum() * 0.0
    return F.nll_loss(
        group_probabilities[known_targets].clamp_min(1e-12).log(),
        group_targets[known_targets],
    )
