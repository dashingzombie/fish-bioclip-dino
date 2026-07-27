"""Prototype-based text classification loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def text_classification_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy over the configured candidate prototype matrix."""
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("Expected logits [batch, classes] and targets [batch]")
    return F.cross_entropy(logits.float(), targets)

