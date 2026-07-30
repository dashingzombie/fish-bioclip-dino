"""Fail-closed audit for the official unseen-species inference contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.inference.predict import decode_class_scores
from fish_vlm.models.multimodal import normalised_similarity_scores
from fish_vlm.training.checkpoint import load_checkpoint
from fish_vlm.training.train import (
    _data_processed_path,
    build_runtime,
    load_candidate_prototypes,
)
from fish_vlm.utils.hashing import ordered_names_hash, prompts_hash
from fish_vlm.utils.io import read_json


@torch.no_grad()
def audit_unseen_inference(
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Validate every static identity used before unseen image scoring starts."""
    inference = config["inference"]["unseen"]
    if inference["candidate_set"] != "unseen":
        raise ValueError(
            "Official unseen inference must use candidate_set=unseen"
        )
    if inference["mode"] in {
        "supervised",
        "supervised_plus_text",
        "bioclip_supervised",
        "bioclip_supervised_plus_text",
    }:
        raise ValueError(
            "Official unseen inference cannot use supervised classifier scores"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    if (
        bundle.model.supervised_head is not None
        or bundle.model.bioclip_classifier is not None
    ):
        raise ValueError(
            "Unseen-only inference constructed a supervised seen classifier"
        )
    prototypes, species_names, cache = load_candidate_prototypes(
        config,
        bundle,
        "unseen",
        device=device,
    )
    if not species_names:
        raise ValueError("The unseen candidate-species set is empty")
    if cache["species_names"] != species_names:
        raise RuntimeError(
            "Text-prototype columns and unseen output labels have different order"
        )

    canonical_prompts = read_json(
        _data_processed_path(config, "canonical_prompts.json")
    )
    _, _, checkpoint_prototype_cache = load_candidate_prototypes(
        config, bundle, "seen", device=device
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=checkpoint_prototype_cache[
            "prompt_hash"
        ],
        expected_canonical_prompt_hash=prompts_hash(
            canonical_prompts, bundle.partitions.all_species
        ),
        expected_dino_model_name=str(config["model"]["dino"]["name"]),
        expected_dino_checkpoint_source=bundle.dino_source,
        expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        strict=False,
    )

    ordering_probe = torch.arange(
        len(species_names), dtype=torch.float32
    ).unsqueeze(0)
    if decode_class_scores(ordering_probe, species_names) != [species_names[-1]]:
        raise RuntimeError("Prediction decoding does not preserve score-column order")
    formula_probe = normalised_similarity_scores(
        torch.tensor([[10.0, 0.0], [0.0, 5.0]]),
        torch.tensor([[0.0, 3.0], [2.0, 0.0]]),
    )
    if not torch.equal(
        formula_probe,
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    ):
        raise RuntimeError("Image/text score normalisation contract failed")

    candidate_count = len(species_names)
    return {
        "status": "passed",
        "checkpoint": str(Path(checkpoint_path)),
        "checkpoint_step": int(checkpoint["step"]),
        "candidate_set": "unseen",
        "candidate_species": species_names,
        "number_of_candidate_species": candidate_count,
        "random_accuracy": 1.0 / candidate_count,
        "prototype_shape": list(prototypes.shape),
        "prototype_column_order_hash": ordered_names_hash(species_names),
        "score_formula": "normalise(image_embeddings) @ normalise(text_embeddings).T",
        "prediction_mode": inference["mode"],
        "checks": {
            "candidate_species_set": True,
            "label_to_prototype_ordering": True,
            "checkpoint_loading": True,
            "projection_head_loading": True,
            "bioclip_tokenizer_text_encoder_consistency": True,
            "image_text_normalisation_contract": True,
            "text_cache_invalidation": True,
            "prediction_decoding": True,
            "seen_classifier_absent": True,
        },
    }
