"""Collect validation logits and fit calibration metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path
from fish_vlm.data.catalog import load_labels, split_filenames
from fish_vlm.evaluation.calibration import (
    fit_calibration,
    fit_probability_weight,
    fit_species_disjoint_fusion,
    save_calibration,
)
from fish_vlm.models.fusion import (
    expanded_supervised_probabilities,
    fuse_text_probabilities,
)
from fish_vlm.inference.predict import is_training_free_native_bioclip
from fish_vlm.training.checkpoint import (
    checkpoint_training_species,
    load_checkpoint,
)
from fish_vlm.training.distributed import DistributedContext
from fish_vlm.training.train import (
    _data_processed_path,
    _split_labelled_filenames,
    _pseudo_split_path,
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
    classifier_mode = str(
        config.get("inference", {}).get("test", {}).get("mode", "")
    )
    training_free_native = is_training_free_native_bioclip(
        config, classifier_mode
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=(
            None
            if training_free_native
            else cache["prompt_hash"]
        ),
        expected_canonical_prompt_hash=prompts_hash(
            read_json(_data_processed_path(config, "canonical_prompts.json")),
            bundle.partitions.all_species,
        ),
        expected_dino_model_name=str(config["model"]["dino"]["name"]),
        expected_dino_checkpoint_source=bundle.dino_source,
        expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        strict=False,
    )
    supervised_species = (
        []
        if training_free_native
        else checkpoint_training_species(
            checkpoint,
            seen_species=bundle.partitions.seen_species,
        )
    )
    species_to_index = {
        name: index for index, name in enumerate(species_names)
    }
    missing_supervised = sorted(set(supervised_species) - set(species_to_index))
    if missing_supervised:
        raise ValueError(
            "Calibration candidate set omits supervised-head species: "
            f"{missing_supervised}"
        )
    supervised_class_indices = [
        species_to_index[name] for name in supervised_species
    ]
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
    use_bioclip_classifier = classifier_mode in {
        "bioclip_supervised",
        "bioclip_supervised_plus_text",
    }
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
        if (
            use_bioclip_classifier
            and output.bioclip_supervised_logits is not None
        ):
            supervised.append(output.bioclip_supervised_logits.cpu())
        elif not use_bioclip_classifier and output.supervised_logits is not None:
            supervised.append(output.supervised_logits.cpu())
    parameters = fit_calibration(
        torch.cat(dino),
        torch.cat(bioclip),
        torch.cat(targets),
        torch.cat(supervised) if supervised else None,
        supervised_class_indices=(
            supervised_class_indices if supervised else None
        ),
    )
    pseudo_path = _pseudo_split_path(config)
    species_disjoint = bool(
        config.get("calibration", {}).get("species_disjoint", True)
    )
    if species_disjoint and (
        pseudo_path is None or not pseudo_path.exists()
    ):
        raise FileNotFoundError(
            "Calibration requires the configured species-disjoint "
            "pseudo-unseen split"
        )
    if species_disjoint:
        assert pseudo_path is not None
        from dataclasses import replace

        pseudo = read_json(pseudo_path)
        training_indices = [
            species_to_index[name]
            for name in pseudo["training_species"]
        ]
        pseudo_indices = {
            species_to_index[name]
            for name in pseudo["evaluation_species"]
        }
        all_targets = torch.cat(targets)
        pseudo_mask = torch.tensor(
            [int(target) in pseudo_indices for target in all_targets],
            dtype=torch.bool,
        )
        seen_mask = ~pseudo_mask
        if bool(seen_mask.any()) and bool(pseudo_mask.any()):
            all_dino = torch.cat(dino)
            all_bioclip = torch.cat(bioclip)
            disjoint = fit_species_disjoint_fusion(
                all_dino[seen_mask],
                all_bioclip[seen_mask],
                all_targets[seen_mask],
                all_dino[pseudo_mask],
                all_bioclip[pseudo_mask],
                all_targets[pseudo_mask],
                seen_class_indices=training_indices,
                gamma_values=[
                    float(value)
                    for value in config.get("calibration", {}).get(
                        "gamma_values", []
                    )
                ]
                or None,
            )
            parameters = replace(
                parameters,
                dino_temperature=disjoint.dino_temperature,
                bioclip_temperature=disjoint.bioclip_temperature,
                dino_text_weight=disjoint.dino_text_weight,
                calibration_gamma=disjoint.calibration_gamma,
            )
            if supervised:
                all_supervised = torch.cat(supervised)
                classifier_probabilities = expanded_supervised_probabilities(
                    all_supervised[seen_mask],
                    parameters.supervised_temperature,
                    class_count=all_dino.shape[1],
                    class_indices=supervised_class_indices,
                )
                if use_bioclip_classifier:
                    text_probabilities = torch.softmax(
                        all_bioclip[seen_mask].float()
                        / parameters.bioclip_temperature,
                        dim=-1,
                    )
                else:
                    text_probabilities = fuse_text_probabilities(
                        all_dino[seen_mask],
                        all_bioclip[seen_mask],
                        parameters,
                    )
                parameters = replace(
                    parameters,
                    supervised_weight=fit_probability_weight(
                        classifier_probabilities,
                        text_probabilities,
                        all_targets[seen_mask],
                    ),
                )
    return save_calibration(
        str(output_path),
        parameters,
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint["step"],
            "species_names": species_names,
            "supervised_species": supervised_species,
            "supervised_classifier": (
                "bioclip" if use_bioclip_classifier else "dino"
            ),
            "text_prototype_hash": cache["prompt_hash"],
            "species_disjoint_tuning": bool(
                species_disjoint
            ),
        },
    )
