"""Cosine classifier for labelled seen species."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CosineClassifier(nn.Module):
    """Learn class directions and a bounded positive scale."""

    def __init__(self, feature_dim: int, num_classes: int, initial_scale: float = 20.0) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.weight = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.normal_(self.weight, std=0.02)
        self.log_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features.float(), dim=-1)
        weights = F.normalize(self.weight.float(), dim=-1)
        return self.log_scale.exp().clamp(max=100.0) * features @ weights.T

