"""Residual adapter for frozen BioCLIP image embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BioClipResidualAdapter(nn.Module):
    """Adapt embeddings without updating the BioCLIP image encoder."""

    def __init__(self, embedding_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        original = original.float()
        return F.normalize(original + self.network(original), dim=-1)

