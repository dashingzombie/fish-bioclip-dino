"""Taxonomy-level objectives derived from ordered species logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hierarchical_cross_entropy(
    species_logits: torch.Tensor,
    species_targets: torch.Tensor,
    class_to_group: list[int] | torch.Tensor,
) -> torch.Tensor:
    """Aggregate species probabilities into genus/family groups."""
    mapping = torch.as_tensor(
        class_to_group,
        device=species_logits.device,
        dtype=torch.long,
    )
    if mapping.ndim != 1 or len(mapping) != species_logits.shape[1]:
        raise ValueError("Taxonomy mapping must match species-logit columns")
    if len(mapping) == 0 or mapping.min().item() < 0:
        raise ValueError("Taxonomy group indices must be non-negative")
    group_count = int(mapping.max().item()) + 1
    probabilities = torch.softmax(species_logits.float(), dim=-1)
    group_probabilities = probabilities.new_zeros(
        (len(probabilities), group_count)
    )
    group_probabilities.scatter_add_(
        1,
        mapping.unsqueeze(0).expand(len(probabilities), -1),
        probabilities,
    )
    group_targets = mapping[species_targets.long()]
    return F.nll_loss(
        group_probabilities.clamp_min(1e-12).log(),
        group_targets,
    )
