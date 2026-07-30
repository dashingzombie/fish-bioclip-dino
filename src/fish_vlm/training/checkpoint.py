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
    "unseen_species", "training_species", "training_species_hash", "active_losses",
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
        "bioclip_classifier_state": None if model.bioclip_classifier is None else model.bioclip_classifier.state_dict(),
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
    expected_dino_model_name: str | None = None,
    expected_dino_checkpoint_source: str | None = None,
    expected_bioclip_checkpoint: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Validate class/prototype/training identities before loading weights."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_METADATA - set(checkpoint)
    if missing:
        raise ValueError(
            "Checkpoint uses an unsupported metadata schema; missing fields: "
            f"{sorted(missing)}"
        )
    checkpoint_training_species(
        checkpoint,
        seen_species=expected_seen_species,
    )
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
    if expected_dino_model_name is not None:
        expected["dino_model_name"] = expected_dino_model_name
    if expected_dino_checkpoint_source is not None:
        expected["dino_checkpoint_source"] = expected_dino_checkpoint_source
    if expected_bioclip_checkpoint is not None:
        expected["bioclip_checkpoint"] = expected_bioclip_checkpoint
    mismatches = {key: (checkpoint.get(key), value) for key, value in expected.items() if checkpoint.get(key) != value}
    if mismatches:
        raise ValueError(f"Incompatible checkpoint: {mismatches}")
    model_state = checkpoint.get("model_state")
    projector_state = checkpoint.get("projector_state")
    if not isinstance(model_state, dict):
        raise ValueError("Checkpoint model_state is missing or invalid")
    if not isinstance(projector_state, dict) or not projector_state:
        raise ValueError("Checkpoint projector_state is missing or invalid")
    embedded_projector = {
        key.removeprefix("projector."): value
        for key, value in model_state.items()
        if key.startswith("projector.")
    }
    if set(embedded_projector) != set(projector_state):
        raise ValueError(
            "Checkpoint projector_state does not match model_state keys"
        )
    inconsistent_projector = [
        key
        for key in projector_state
        if not torch.equal(embedded_projector[key], projector_state[key])
    ]
    if inconsistent_projector:
        raise ValueError(
            "Checkpoint projector_state conflicts with model_state: "
            f"{inconsistent_projector}"
        )
    incompatible = model.load_state_dict(model_state, strict=strict)
    if not strict:
        required_prefixes = ("dino.", "projector.", "logit_scale.")
        missing_core = [
            key
            for key in incompatible.missing_keys
            if key.startswith(required_prefixes)
        ]
        if missing_core:
            raise ValueError(
                f"Checkpoint is missing core model weights: {missing_core}"
            )
    loaded_projector = model.projector.state_dict()
    failed_to_load = [
        key
        for key in projector_state
        if key not in loaded_projector
        or not torch.equal(
            loaded_projector[key].detach().cpu(),
            projector_state[key].detach().cpu(),
        )
    ]
    if failed_to_load:
        raise RuntimeError(
            "Projection-head weights were not restored from the checkpoint: "
            f"{failed_to_load}"
        )
    return checkpoint


def checkpoint_training_species(
    checkpoint: dict[str, Any],
    *,
    seen_species: list[str],
) -> list[str]:
    """Return the checkpoint's strict ordered supervised-head label space."""
    value = checkpoint.get("training_species")
    if not isinstance(value, list) or not value or not all(
        isinstance(name, str) for name in value
    ):
        raise ValueError(
            "Checkpoint lacks a valid ordered training_species list; "
            "regenerate it with the current training code"
        )
    names = list(value)
    if len(names) != len(set(names)):
        raise ValueError("Checkpoint training_species contains duplicates")
    unknown = sorted(set(names) - set(seen_species))
    if unknown:
        raise ValueError(
            f"Checkpoint training_species contains unknown seen species: {unknown}"
        )
    expected_hash = ordered_names_hash(names)
    if checkpoint.get("training_species_hash") != expected_hash:
        raise ValueError("Checkpoint training_species_hash does not match its class list")
    return names
