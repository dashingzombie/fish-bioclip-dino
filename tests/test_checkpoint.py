from __future__ import annotations

from pathlib import Path

import pytest
import torch

from conftest import TinyDino
from fish_vlm.models.multimodal import FishMultimodalModel
from fish_vlm.models.projector import LearnableLogitScale, LinearDinoToBioClipProjector
from fish_vlm.training.checkpoint import load_checkpoint, save_checkpoint


def _model() -> FishMultimodalModel:
    return FishMultimodalModel(
        TinyDino(), LinearDinoToBioClipProjector(4, 3), LearnableLogitScale()
    )


def test_checkpoint_round_trip_and_order_rejection(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "checkpoint.pt"
    metadata = {
        "dino_model_name": "tiny",
        "dino_checkpoint_source": "mock",
        "bioclip_checkpoint": "mock",
        "text_prototype_hash": "prompts",
        "canonical_prompt_hash": "all-prompts",
        "seen_species": ["A", "B"],
        "unseen_species": ["C"],
        "training_species_hash": "training",
        "pseudo_unseen_split_hash": None,
        "active_losses": ["dino_text_classification"],
    }
    save_checkpoint(
        path, model=model, optimizer=None, scheduler=None, scaler=None, step=300,
        best_metric=0.75, config={"seed": 1}, metadata=metadata,
    )
    loaded = load_checkpoint(
        path, _model(), expected_seen_species=["A", "B"], expected_unseen_species=["C"],
        expected_text_prototype_hash="prompts", expected_training_species_hash="training",
    )
    assert loaded["step"] == 300
    with pytest.raises(ValueError, match="Incompatible"):
        load_checkpoint(
            path, _model(), expected_seen_species=["B", "A"], expected_unseen_species=["C"],
            expected_text_prototype_hash="prompts",
        )
