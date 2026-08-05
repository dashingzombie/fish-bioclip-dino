"""Validated composition of a fine-tuned BioCLIP visual tower with DINO."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from fish_vlm.training.checkpoint import REQUIRED_METADATA
from fish_vlm.utils.hashing import ordered_names_hash


EXPECTED_FINETUNE_LOSSES = [
    "native_bioclip_text",
    "bioclip_supervised_species",
    "bioclip_pretrained_distillation",
]


def _validate_finetune_contract(checkpoint: dict[str, Any]) -> None:
    resolved = checkpoint.get("resolved_configuration")
    if not isinstance(resolved, dict):
        raise ValueError("BioCLIP checkpoint lacks its resolved configuration")
    if resolved.get("training", {}).get("stage") != "bioclip_full_finetune":
        raise ValueError("BioCLIP checkpoint is not a full fine-tune")
    model = resolved.get("model", {})
    bioclip = model.get("bioclip", {})
    if (
        model.get("tuning_mode") != "full_finetune"
        or bioclip.get("freeze_image_encoder", True)
        or not bioclip.get("freeze_text_encoder", True)
    ):
        raise ValueError(
            "BioCLIP checkpoint does not prove visual-only full fine-tuning"
        )
    if checkpoint.get("active_losses") != EXPECTED_FINETUNE_LOSSES:
        raise ValueError(
            "BioCLIP checkpoint lacks the required text, supervised, and "
            "pretrained-distillation losses"
        )


def load_finetuned_bioclip_visual(
    checkpoint_path: str | Path,
    model: nn.Module,
    *,
    expected_seen_species: list[str],
    expected_unseen_species: list[str],
    expected_training_species: list[str],
    expected_text_prototype_hash: str,
    expected_canonical_prompt_hash: str,
    expected_bioclip_checkpoint: str,
) -> dict[str, Any]:
    """Load only validated visual weights; require text weights to be unchanged."""
    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise TypeError("BioCLIP checkpoint must contain a mapping")
    missing_metadata = REQUIRED_METADATA - set(checkpoint)
    if missing_metadata:
        raise ValueError(
            "BioCLIP checkpoint uses an unsupported metadata schema; missing "
            f"{sorted(missing_metadata)}"
        )
    _validate_finetune_contract(checkpoint)
    expected = {
        "seen_species": expected_seen_species,
        "unseen_species": expected_unseen_species,
        "training_species": expected_training_species,
        "training_species_hash": ordered_names_hash(expected_training_species),
        "text_prototype_hash": expected_text_prototype_hash,
        "canonical_prompt_hash": expected_canonical_prompt_hash,
        "bioclip_checkpoint": expected_bioclip_checkpoint,
    }
    mismatches = {
        key: (checkpoint.get(key), value)
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Incompatible BioCLIP fine-tune checkpoint: {mismatches}")

    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError("BioCLIP checkpoint model_state is missing")
    saved = {
        key.removeprefix("bioclip."): value
        for key, value in model_state.items()
        if key.startswith("bioclip.")
    }
    current = model.state_dict()
    if set(saved) != set(current):
        raise ValueError(
            "Fine-tuned and runtime BioCLIP state dictionaries differ"
        )
    visual = {
        key: value for key, value in saved.items() if key.startswith("visual.")
    }
    if not visual:
        raise ValueError("BioCLIP checkpoint contains no visual-tower weights")
    changed_nonvisual = [
        key
        for key, value in saved.items()
        if not key.startswith("visual.")
        and not torch.equal(value.detach().cpu(), current[key].detach().cpu())
    ]
    if changed_nonvisual:
        raise ValueError(
            "Fine-tuned BioCLIP changed frozen text/nonvisual weights: "
            f"{changed_nonvisual[:10]}"
        )
    incompatible = model.load_state_dict(visual, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected fine-tuned BioCLIP visual keys: "
            f"{incompatible.unexpected_keys}"
        )
    missing_visual = [
        key for key in incompatible.missing_keys if key.startswith("visual.")
    ]
    if missing_visual:
        raise RuntimeError(
            f"Fine-tuned BioCLIP visual weights were not loaded: {missing_visual}"
        )
    return checkpoint
