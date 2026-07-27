"""DINO/BioCLIP image-embedding alignment losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_teacher_loss(projected: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Mean one-minus-cosine alignment in float32."""
    if projected.shape != teacher.shape:
        raise ValueError("Projected and teacher embeddings must have identical shapes")
    return (1.0 - F.cosine_similarity(projected.float(), teacher.float(), dim=-1)).mean()


def symmetric_contrastive_teacher_loss(
    projected: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric in-batch contrastive alignment."""
    if projected.shape != teacher.shape:
        raise ValueError("Projected and teacher embeddings must have identical shapes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    projected = F.normalize(projected.float(), dim=-1)
    teacher = F.normalize(teacher.float(), dim=-1)
    logits = projected @ teacher.T / temperature
    labels = torch.arange(len(projected), device=projected.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

