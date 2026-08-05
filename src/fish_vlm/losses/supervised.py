"""Seen-species supervised loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_species_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Seen-class cross entropy with optional confidence regularisation."""
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    return F.cross_entropy(
        logits.float(),
        targets,
        label_smoothing=float(label_smoothing),
    )
