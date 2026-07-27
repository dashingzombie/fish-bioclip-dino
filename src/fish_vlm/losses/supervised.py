"""Seen-species supervised loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_species_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ordinary seen-class cross entropy."""
    return F.cross_entropy(logits.float(), targets)

