"""DINO-to-BioCLIP projection heads."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class DinoToBioClipProjector(nn.Module):
    """MLP projector ending in unit-normalised BioCLIP-space features."""

    def __init__(
        self,
        dino_dim: int,
        bioclip_dim: int,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bioclip_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.network(features)
        return F.normalize(projected.float(), dim=-1)


class LinearDinoToBioClipProjector(nn.Module):
    """Layer-normalised linear projector."""

    def __init__(self, dino_dim: int, bioclip_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, bioclip_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(features).float(), dim=-1)


class LearnableLogitScale(nn.Module):
    """CLIP-style bounded similarity scale."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def forward(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)


def build_projector(config: dict, dino_dim: int, bioclip_dim: int) -> nn.Module:
    """Build a configured projector without hard-coded dimensions."""
    if config["type"] == "linear":
        return LinearDinoToBioClipProjector(dino_dim, bioclip_dim)
    if config["type"] == "mlp":
        return DinoToBioClipProjector(
            dino_dim,
            bioclip_dim,
            hidden_dim=int(config.get("hidden_dim", 2048)),
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported projector type: {config['type']}")

