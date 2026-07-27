"""Scientifically self-describing checkpoint persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.models.dino import POOLING_STRATEGY
from fish_vlm.models.multimodal import FishMultimodalModel
from fish_vlm.utils.hashing import ordered_names_hash
from fish_vlm.utils.io import torch_save_atomic


REQUIRED_METADATA = {
    "dino_model_name", "dino_checkpoint_source", "bioclip_checkpoint",
    "text_prototype_hash", "canonical_prompt_hash", "seen_species",
    "unseen_species", "training_species_hash", "active_losses",
}


def save_checkpoint(
    path: str | Path,
    *,
    model: FishMultimodalModel,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    scaler: Any,
    step: int,
    best_metric: float,
    config: dict[str, Any],
    metadata: dict[str, Any],
    calibration_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write all model/training/scientific identity state atomically."""
    missing = REQUIRED_METADATA - set(metadata)
    if missing:
        raise ValueError(f"Missing checkpoint metadata: {sorted(missing)}")
    payload = {
        "model_state": model.state_dict(),
        "projector_state": model.projector.state_dict(),
        "supervised_head_state": None if model.supervised_head is None else model.supervised_head.state_dict(),
        "bioclip_adapter_state": None if model.bioclip_adapter is None else model.bioclip_adapter.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "gradient_scaler_state": None if scaler is None else scaler.state_dict(),
        "step": int(step),
        "best_metric": float(best_metric),
        "resolved_configuration": config,
        "dino_pooling_strategy": POOLING_STRATEGY,
        "pseudo_unseen_split_hash": metadata.get("pseudo_unseen_split_hash"),
        "calibration_metadata": calibration_metadata,
        **metadata,
    }
    torch_save_atomic(payload, path)
    return payload


def load_checkpoint(
    path: str | Path,
    model: FishMultimodalModel,
    *,
    expected_seen_species: list[str],
    expected_unseen_species: list[str],
    expected_text_prototype_hash: str | None,
    expected_canonical_prompt_hash: str | None = None,
    expected_training_species_hash: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate class/prototype/training identities before loading weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "seen_species": expected_seen_species,
        "unseen_species": expected_unseen_species,
        "dino_pooling_strategy": POOLING_STRATEGY,
    }
    if expected_text_prototype_hash is not None:
        expected["text_prototype_hash"] = expected_text_prototype_hash
    if expected_canonical_prompt_hash is not None:
        expected["canonical_prompt_hash"] = expected_canonical_prompt_hash
    if expected_training_species_hash is not None:
        expected["training_species_hash"] = expected_training_species_hash
    mismatches = {key: (checkpoint.get(key), value) for key, value in expected.items() if checkpoint.get(key) != value}
    if mismatches:
        raise ValueError(f"Incompatible checkpoint: {mismatches}")
    model.load_state_dict(checkpoint["model_state"], strict=strict)
    return checkpoint
