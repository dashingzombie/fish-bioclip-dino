"""Collect validation logits and fit calibration metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path
from fish_vlm.data.catalog import load_labels, split_filenames
from fish_vlm.evaluation.calibration import fit_calibration, save_calibration
from fish_vlm.training.checkpoint import load_checkpoint
from fish_vlm.training.distributed import DistributedContext
from fish_vlm.training.train import (
    _data_processed_path,
    _split_labelled_filenames,
    build_runtime,
    load_candidate_prototypes,
    make_loader,
)
from fish_vlm.utils.hashing import prompts_hash
from fish_vlm.utils.io import read_json


@torch.no_grad()
def calibrate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Fit branch calibration using only a held-out labelled training subset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    prototypes, species_names, cache = load_candidate_prototypes(config, bundle, "seen", device=device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=cache["prompt_hash"],
        expected_canonical_prompt_hash=prompts_hash(
            read_json(_data_processed_path(config, "canonical_prompts.json")),
            bundle.partitions.all_species,
        ),
        strict=False,
    )
    labels = load_labels(config)
    _, validation_names = _split_labelled_filenames(
        split_filenames(data_path(config, "train_split")),
        labels,
        int(config["seed"]),
        float(config["validation"].get("seen_fraction", 0.1)),
    )
    loader, _ = make_loader(
        validation_names,
        config,
        bundle,
        labels,
        species_names,
        training=False,
        context=DistributedContext(0, 1, 0, device),
    )
    dino: list[torch.Tensor] = []
    bioclip: list[torch.Tensor] = []
    supervised: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    bundle.model.eval()
    for batch in loader:
        output = bundle.model(
            batch["dino_image"].to(device), prototypes, batch["bioclip_image"].to(device)
        )
        if output.bioclip_logits is None:
            raise ValueError("Calibration requires the BioCLIP-native branch")
        dino.append(output.dino_text_logits.cpu())
        bioclip.append(output.bioclip_logits.cpu())
        targets.append(batch["species_index"])
        if output.supervised_logits is not None:
            supervised.append(output.supervised_logits.cpu())
    parameters = fit_calibration(
        torch.cat(dino),
        torch.cat(bioclip),
        torch.cat(targets),
        torch.cat(supervised) if supervised else None,
    )
    return save_calibration(
        str(output_path),
        parameters,
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint["epoch"],
            "species_names": species_names,
            "text_prototype_hash": cache["prompt_hash"],
        },
    )
