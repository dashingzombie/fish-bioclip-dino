"""Small deterministic test doubles; no pretrained downloads."""

from __future__ import annotations

import torch
from torch import nn


class TinyDino(nn.Module):
    """timm-like pooled encoder."""

    num_features = 4

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)

    def forward_features(self, value: torch.Tensor) -> torch.Tensor:
        value = value.reshape(len(value), -1)[:, :4]
        for block in self.blocks:
            value = torch.tanh(block(value))
        return self.norm(value)

    def forward_head(self, value: torch.Tensor, pre_logits: bool = True) -> torch.Tensor:
        return value


class TinyBioClip(nn.Module):
    """open_clip-like deterministic dual encoder."""

    def __init__(self, input_dim: int = 4, embedding_dim: int = 3) -> None:
        super().__init__()
        self.image = nn.Linear(input_dim, embedding_dim, bias=False)
        self.text = nn.Embedding(32, embedding_dim)
        self.text_projection = nn.Parameter(torch.eye(embedding_dim), requires_grad=False)

    def encode_image(self, value: torch.Tensor) -> torch.Tensor:
        return self.image(value.reshape(len(value), -1)[:, :4])

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text(tokens).mean(dim=1)


def tiny_tokenizer(texts: list[str]) -> torch.Tensor:
    """Map text length to deterministic token IDs."""
    return torch.tensor([[len(text) % 31 + 1, 1] for text in texts], dtype=torch.long)

