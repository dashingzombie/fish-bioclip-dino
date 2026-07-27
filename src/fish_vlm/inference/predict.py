"""Candidate-restricted model inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path
from fish_vlm.data.catalog import split_filenames
from fish_vlm.models.fusion import CalibrationParameters
from fish_vlm.training.checkpoint import load_checkpoint
from fish_vlm.training.distributed import DistributedContext
from fish_vlm.training.train import build_runtime, load_candidate_prototypes, make_loader
from fish_vlm.utils.io import read_json, write_json


def load_calibration(path: str | Path | None, config: dict[str, Any]) -> CalibrationParameters:
    """Load fitted calibration or explicit configured defaults."""
    if path is None:
        return CalibrationParameters(
            dino_text_weight=float(config["fusion"]["dino_text_weight"]),
            supervised_weight=float(config["fusion"]["supervised_weight"]),
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
) -> dict[str, str]:
    """Predict one official image-only split under its explicit candidate set."""
    if split not in {"test", "unseen"}:
        raise ValueError("Official split must be test or unseen")
    inference = config["inference"][split]
    candidate_set = inference["candidate_set"]
    mode = inference["mode"]
    if split == "unseen" and mode in {"supervised", "supervised_plus_text"}:
        raise ValueError("The supervised head is forbidden for official unseen inference")
    if candidate_set == "all" and not config["inference"].get("generalised_enabled", False):
        raise ValueError("Generalised seen/unseen candidate mode is disabled")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    prototypes, species_names, _ = load_candidate_prototypes(config, bundle, candidate_set, device=device)
    from fish_vlm.training.train import _data_processed_path
    from fish_vlm.utils.hashing import prompts_hash

    canonical_prompts = read_json(_data_processed_path(config, "canonical_prompts.json"))
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=None,
        expected_canonical_prompt_hash=prompts_hash(
            canonical_prompts, bundle.partitions.all_species
        ),
        strict=False,
    )
    if mode in {"supervised", "supervised_plus_text"} and checkpoint.get("supervised_head_state") is None:
        raise ValueError(
            "The selected seen inference mode requires a checkpoint trained with the supervised head"
        )
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
        probabilities = bundle.model.probabilities(output, mode, calibration)
        indices = probabilities.argmax(dim=-1).cpu().tolist()
        predictions.update(
            {filename: species_names[index] for filename, index in zip(batch["filename"], indices, strict=True)}
        )
    if len(predictions) != len(filenames):
        raise RuntimeError("Prediction output lost or duplicated filenames")
    write_json(output_path, predictions)
    return predictions
