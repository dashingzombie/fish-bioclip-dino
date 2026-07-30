"""Frozen DINO projection teacher used for alignment-preserving training."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fish_vlm.models.dino import pooled_features
from fish_vlm.models.multimodal import FishMultimodalModel


class DinoProjectionTeacher(nn.Module):
    """Frozen snapshot of the DINO encoder and projection head."""

    def __init__(self, model: FishMultimodalModel) -> None:
        super().__init__()
        self.dino = copy.deepcopy(model.dino)
        self.projector = copy.deepcopy(model.projector)
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_checkpoint(
        cls,
        model: FishMultimodalModel,
        checkpoint_path: str | Path,
        *,
        expected_identity: dict[str, Any],
    ) -> "DinoProjectionTeacher":
        """Load an independently identified DINO/projector teacher."""
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected_identity.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Incompatible representation teacher: {mismatches}"
            )
        model_state = checkpoint.get("model_state")
        projector_state = checkpoint.get("projector_state")
        if not isinstance(model_state, dict) or not isinstance(
            projector_state, dict
        ):
            raise ValueError(
                "Representation teacher checkpoint lacks model state"
            )
        teacher = cls(model)
        dino_state = {
            key.removeprefix("dino."): value
            for key, value in model_state.items()
            if key.startswith("dino.")
        }
        teacher.dino.load_state_dict(dino_state, strict=True)
        teacher.projector.load_state_dict(projector_state, strict=True)
        teacher.requires_grad_(False)
        teacher.eval()
        return teacher

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(pooled_features(self.dino, images))
