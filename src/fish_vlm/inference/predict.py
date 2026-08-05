"""Candidate-restricted model inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path
from fish_vlm.data.catalog import split_filenames
from fish_vlm.evaluation.gate_calibrate import (
    load_gate_calibration,
    validate_hybrid_checkpoint,
)
from fish_vlm.evaluation.gating import gated_prediction_indices
from fish_vlm.inference.bioclip_checkpoint import (
    load_finetuned_bioclip_visual,
)
from fish_vlm.models.fusion import CalibrationParameters
from fish_vlm.training.checkpoint import (
    checkpoint_training_species,
    load_checkpoint,
)
from fish_vlm.training.distributed import DistributedContext
from fish_vlm.training.train import build_runtime, load_candidate_prototypes, make_loader
from fish_vlm.utils.io import read_json, write_json

SUPERVISED_MODES = {
    "supervised",
    "supervised_plus_text",
    "bioclip_supervised",
    "bioclip_supervised_plus_text",
}


def is_training_free_native_bioclip(
    config: dict[str, Any],
    mode: str,
) -> bool:
    """Return whether checkpoint-trained image/text components are unused."""
    image_path = config["model"]["bioclip_image_path"]
    return (
        bool(config["inference"].get("training_free_native", False))
        and mode == "bioclip_native"
        and config["model"].get("tuning_mode", "frozen") == "frozen"
        and image_path.get("mode") == "frozen_zero_shot"
        and not image_path.get("adapter", {}).get("enabled", False)
        and not config["model"].get("bioclip_classifier", {}).get(
            "enabled", False
        )
        and config["model"].get("bioclip", {}).get(
            "freeze_image_encoder", True
        )
    )


def decode_class_scores(
    scores: torch.Tensor,
    class_names: list[str],
) -> list[str]:
    """Decode score columns using the exact ordered class list that defined them."""
    if scores.ndim != 2:
        raise ValueError("Class scores must be a two-dimensional matrix")
    if scores.shape[1] != len(class_names):
        raise ValueError(
            "Score-column count does not match the ordered output labels"
        )
    if len(class_names) != len(set(class_names)):
        raise ValueError("Ordered output labels contain duplicates")
    indices = scores.argmax(dim=-1).cpu().tolist()
    return [class_names[index] for index in indices]


def load_calibration(path: str | Path | None, config: dict[str, Any]) -> CalibrationParameters:
    """Load fitted calibration or explicit configured defaults."""
    if path is None:
        return CalibrationParameters(
            dino_text_weight=float(config["fusion"]["dino_text_weight"]),
            supervised_weight=float(config["fusion"]["supervised_weight"]),
            calibration_gamma=float(
                config["fusion"].get("calibration_gamma", 0.0)
            ),
        )
    value = read_json(path)
    from fish_vlm.utils.hashing import stable_json_hash

    supplied_hash = value.get("hash")
    unhashed = {key: item for key, item in value.items() if key != "hash"}
    if supplied_hash != stable_json_hash(unhashed):
        raise ValueError("Calibration file hash is invalid")
    return CalibrationParameters(**value["parameters"])


@torch.no_grad()
def predict_split(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    calibration_path: str | Path | None = None,
    bioclip_checkpoint_path: str | Path | None = None,
) -> dict[str, str]:
    """Predict one official image-only split under its explicit candidate set."""
    if split not in {"test", "unseen"}:
        raise ValueError("Official split must be test or unseen")
    inference = config["inference"][split]
    candidate_set = inference["candidate_set"]
    mode = inference["mode"]
    if split == "unseen" and mode in SUPERVISED_MODES:
        raise ValueError("The supervised head is forbidden for official unseen inference")
    if candidate_set == "all" and not config["inference"].get("generalised_enabled", False):
        raise ValueError("Generalised seen/unseen candidate mode is disabled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    prototypes, species_names, prototype_cache = load_candidate_prototypes(
        config, bundle, candidate_set, device=device
    )
    from fish_vlm.training.train import _data_processed_path
    from fish_vlm.utils.hashing import prompts_hash

    canonical_prompts = read_json(_data_processed_path(config, "canonical_prompts.json"))
    seen_text_hash = prototype_cache["prompt_hash"]
    if candidate_set != "seen":
        _, _, checkpoint_prototype_cache = load_candidate_prototypes(
            config, bundle, "seen", device=device
        )
        seen_text_hash = checkpoint_prototype_cache["prompt_hash"]
    checkpoint_text_hash: str | None = seen_text_hash
    if is_training_free_native_bioclip(config, mode):
        if bioclip_checkpoint_path is not None:
            raise ValueError(
                "Fine-tuned BioCLIP cannot use training_free_native inference"
            )
        checkpoint_text_hash = None
    canonical_prompt_hash = prompts_hash(
        canonical_prompts, bundle.partitions.all_species
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=checkpoint_text_hash,
        expected_canonical_prompt_hash=canonical_prompt_hash,
        expected_dino_model_name=str(config["model"]["dino"]["name"]),
        expected_dino_checkpoint_source=bundle.dino_source,
        expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        strict=False,
    )
    hybrid_checkpoint = bool(
        config["inference"].get("hybrid_dino_seen_checkpoint", False)
    )
    if hybrid_checkpoint:
        validate_hybrid_checkpoint(checkpoint)
    elif bioclip_checkpoint_path is not None:
        raise ValueError(
            "Fine-tuned BioCLIP composition requires an explicitly marked "
            "hybrid DINO checkpoint"
        )
    checkpoint_species = checkpoint_training_species(
        checkpoint, seen_species=bundle.partitions.seen_species
    )
    if bioclip_checkpoint_path is not None:
        if bundle.model.bioclip is None:
            raise ValueError("Fine-tuned BioCLIP inference is unavailable")
        load_finetuned_bioclip_visual(
            bioclip_checkpoint_path,
            bundle.model.bioclip,
            expected_seen_species=bundle.partitions.seen_species,
            expected_unseen_species=bundle.partitions.unseen_species,
            expected_training_species=checkpoint_species,
            expected_text_prototype_hash=seen_text_hash,
            expected_canonical_prompt_hash=canonical_prompt_hash,
            expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        )
    if prototype_cache["species_names"] != species_names:
        raise RuntimeError(
            "Candidate prototype columns and output label ordering diverged"
        )
    if split == "unseen" and (
        bundle.model.supervised_head is not None
        or bundle.model.bioclip_classifier is not None
    ):
        raise ValueError(
            "Unseen-only inference must construct no supervised seen classifier"
        )
    if mode in {"supervised", "supervised_plus_text"} and checkpoint.get(
        "supervised_head_state"
    ) is None:
        raise ValueError(
            "The selected seen inference mode requires a checkpoint trained "
            "with the DINO supervised head"
        )
    if mode in {
        "bioclip_supervised",
        "bioclip_supervised_plus_text",
    } and checkpoint.get(
        "bioclip_classifier_state"
    ) is None:
        raise ValueError(
            "BioCLIP supervised inference requires its trained classifier"
        )
    supervised_class_indices: list[int] | None = None
    seen_class_indices: list[int] | None = None
    if candidate_set == "all":
        candidate_to_index = {
            name: index for index, name in enumerate(species_names)
        }
        seen_class_indices = [
            candidate_to_index[name]
            for name in bundle.partitions.seen_species
        ]
    if mode in SUPERVISED_MODES:
        supervised_species = checkpoint_species
        candidate_to_index = {
            name: index for index, name in enumerate(species_names)
        }
        missing_supervised = sorted(
            set(supervised_species) - set(candidate_to_index)
        )
        if missing_supervised:
            raise ValueError(
                "Inference candidate set omits supervised-head species: "
                f"{missing_supervised}"
            )
        supervised_class_indices = [
            candidate_to_index[name] for name in supervised_species
        ]
    calibration = load_calibration(calibration_path or config.get("calibration_path"), config)
    filenames = split_filenames(data_path(config, f"{split}_split"))
    context = DistributedContext(0, 1, 0, device)
    loader, _ = make_loader(
        filenames, config, bundle, None, species_names, training=False, context=context
    )
    predictions: dict[str, str] = {}
    bundle.model.eval()
    for batch in loader:
        output = bundle.model(
            batch["dino_image"].to(device),
            prototypes,
            batch["bioclip_image"].to(device),
        )
        if split == "unseen" and (
            output.supervised_logits is not None
            or output.bioclip_supervised_logits is not None
        ):
            raise RuntimeError(
                "Seen-classifier logits were produced during unseen-only prediction"
            )
        probabilities = bundle.model.probabilities(
            output,
            mode,
            calibration,
            supervised_class_indices=supervised_class_indices,
            seen_class_indices=seen_class_indices,
        )
        decoded = decode_class_scores(probabilities, species_names)
        predictions.update(
            dict(zip(batch["filename"], decoded, strict=True))
        )
    if len(predictions) != len(filenames):
        raise RuntimeError("Prediction output lost or duplicated filenames")
    write_json(output_path, predictions)
    return predictions


@torch.no_grad()
def predict_gated_split(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    gate_path: str | Path,
    output_path: str | Path,
    *,
    split: str,
    bioclip_checkpoint_path: str | Path | None = None,
) -> dict[str, str]:
    """Apply the DINO-confidence/BioCLIP-all-candidate gate to one split."""
    if split not in {"test", "unseen"}:
        raise ValueError("Official split must be test or unseen")
    if not config["inference"].get("generalised_enabled", False):
        raise ValueError("Gated inference requires generalised all-class mode")
    if config["model"].get("tuning_mode", "frozen") != "frozen":
        raise ValueError("Gated runtime must construct the frozen BioCLIP architecture")
    image_path = config["model"]["bioclip_image_path"]
    if (
        image_path.get("mode") != "frozen_zero_shot"
        or image_path.get("adapter", {}).get("enabled", False)
    ):
        raise ValueError("Gated fallback must use unadapted native BioCLIP")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    prototypes, species_names, prototype_cache = load_candidate_prototypes(
        config, bundle, "all", device=device
    )
    from fish_vlm.training.train import _data_processed_path
    from fish_vlm.utils.hashing import prompts_hash

    seen_prototype_cache = load_candidate_prototypes(
        config, bundle, "seen", device=device
    )[2]
    canonical_prompt_hash = prompts_hash(
        read_json(_data_processed_path(config, "canonical_prompts.json")),
        bundle.partitions.all_species,
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=seen_prototype_cache["prompt_hash"],
        expected_canonical_prompt_hash=canonical_prompt_hash,
        expected_dino_model_name=str(config["model"]["dino"]["name"]),
        expected_dino_checkpoint_source=bundle.dino_source,
        expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        strict=False,
    )
    validate_hybrid_checkpoint(checkpoint)
    if prototype_cache["species_names"] != species_names:
        raise RuntimeError(
            "All-candidate prototype columns and output labels diverged"
        )
    if checkpoint.get("supervised_head_state") is None:
        raise ValueError("Gated inference requires a trained DINO supervised head")
    supervised_species = checkpoint_training_species(
        checkpoint, seen_species=bundle.partitions.seen_species
    )
    finetuned_bioclip = None
    if bioclip_checkpoint_path is not None:
        if bundle.model.bioclip is None:
            raise ValueError("Fine-tuned BioCLIP inference is unavailable")
        finetuned_bioclip = load_finetuned_bioclip_visual(
            bioclip_checkpoint_path,
            bundle.model.bioclip,
            expected_seen_species=bundle.partitions.seen_species,
            expected_unseen_species=bundle.partitions.unseen_species,
            expected_training_species=supervised_species,
            expected_text_prototype_hash=seen_prototype_cache["prompt_hash"],
            expected_canonical_prompt_hash=canonical_prompt_hash,
            expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        )
    candidate_to_index = {
        name: index for index, name in enumerate(species_names)
    }
    supervised_class_indices = [
        candidate_to_index[name] for name in supervised_species
    ]
    gate = load_gate_calibration(gate_path)
    gate_metadata = gate["metadata"]
    if gate_metadata.get("supervised_species") != supervised_species:
        raise ValueError(
            "Gate calibration supervised species do not match the checkpoint"
        )
    if int(gate_metadata.get("checkpoint_step", -1)) != int(checkpoint["step"]):
        raise ValueError("Gate calibration checkpoint step does not match")
    expected_fallback = (
        None
        if bioclip_checkpoint_path is None
        else str(bioclip_checkpoint_path)
    )
    if gate_metadata.get("bioclip_fallback_checkpoint") != expected_fallback:
        raise ValueError("Gate calibration BioCLIP fallback does not match")
    expected_bioclip_step = (
        None if finetuned_bioclip is None else int(finetuned_bioclip["step"])
    )
    if gate_metadata.get("bioclip_fallback_checkpoint_step") != expected_bioclip_step:
        raise ValueError("Gate calibration BioCLIP checkpoint step does not match")

    filenames = split_filenames(data_path(config, f"{split}_split"))
    loader, _ = make_loader(
        filenames,
        config,
        bundle,
        None,
        species_names,
        training=False,
        context=DistributedContext(0, 1, 0, device),
    )
    predictions: dict[str, str] = {}
    bundle.model.eval()
    for batch in loader:
        output = bundle.model(
            batch["dino_image"].to(device),
            prototypes,
            batch["bioclip_image"].to(device),
        )
        if output.supervised_logits is None or output.bioclip_logits is None:
            raise RuntimeError("Gated inference branches are unavailable")
        prediction_indices, _, _ = gated_prediction_indices(
            output.supervised_logits,
            output.bioclip_logits,
            supervised_class_indices=supervised_class_indices,
            threshold=float(gate["parameters"]["threshold"]),
            supervised_temperature=float(
                gate["parameters"]["supervised_temperature"]
            ),
        )
        decoded = [species_names[index] for index in prediction_indices.cpu()]
        predictions.update(
            dict(zip(batch["filename"], decoded, strict=True))
        )
    if len(predictions) != len(filenames):
        raise RuntimeError("Gated prediction lost or duplicated filenames")
    write_json(output_path, predictions)
    return predictions
