"""Checkpoint evaluation over labelled seen or pseudo-unseen images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path
from fish_vlm.data.catalog import load_labels, split_filenames
from fish_vlm.data.datasets import BioClipImageDataset
from fish_vlm.data.taxonomy import load_family_mapping
from fish_vlm.inference.predict import is_training_free_native_bioclip
from fish_vlm.models.bioclip import encode_bioclip_images
from fish_vlm.models.multimodal import text_logits
from fish_vlm.prototypes.conditions import (
    PROMPT_CONDITIONS,
    build_prompt_conditions,
    encode_condition_prototypes,
)
from fish_vlm.training.checkpoint import load_checkpoint
from fish_vlm.training.distributed import DistributedContext
from fish_vlm.training.train import (
    _data_processed_path,
    build_runtime,
    _pseudo_split_path,
    evaluate_loader,
    load_candidate_prototypes,
    load_runtime_image_cache,
    make_loader,
)
from fish_vlm.utils.hashing import prompts_hash
from fish_vlm.utils.io import read_json


def evaluate_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Evaluate branch metrics without using official image-only labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    split_name = config.get("evaluation", {}).get("split", "train")
    candidate_set = config.get("evaluation", {}).get("candidate_set", "seen")
    inference_mode = str(
        config.get("inference", {}).get("test", {}).get("mode", "")
    )
    prototypes, species_names, cache = load_candidate_prototypes(config, bundle, candidate_set, device=device)
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=(
            None
            if is_training_free_native_bioclip(
                config, inference_mode
            )
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
    labels = load_labels(config)
    if split_name in {"test", "unseen"}:
        raise ValueError("Official test/unseen labels are not assumed; use infer instead")
    filenames = [name for name in split_filenames(data_path(config, "train_split")) if name in labels and labels[name] in set(species_names)]
    context = DistributedContext(0, 1, 0, device)
    loader, _ = make_loader(
        filenames, config, bundle, labels, species_names, training=False, context=context
    )
    metrics = evaluate_loader(bundle.model, loader, prototypes, device)
    metrics["checkpoint_step"] = float(checkpoint["step"])
    selected_key = f"{inference_mode}_accuracy"
    if selected_key in metrics:
        metrics["selection_branch"] = inference_mode
        metrics["selected_accuracy"] = float(metrics[selected_key])
    return metrics


@torch.no_grad()
def evaluate_bioclip_zero_shot(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate native BioCLIP across all controlled prompt conditions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config["model"]["bioclip_image_path"].get("enabled", False):
        raise ValueError("Stage 0 requires model.bioclip_image_path.enabled=true")
    bundle = build_runtime(config, device=device)
    partitions = bundle.partitions
    bioclip = bundle.model.bioclip
    if bioclip is None:
        raise ValueError("Stage 0 requires a loaded BioCLIP image encoder")
    transform = bundle.bioclip_eval_transform
    canonical_prompts = read_json(
        _data_processed_path(config, "canonical_prompts.json")
    )
    descriptions = read_json(data_path(config, "descriptions_json"))
    if not isinstance(descriptions, dict):
        raise TypeError("descriptions_json must contain a JSON object")
    families = load_family_mapping(config, partitions.seen_species)
    condition_prompts = build_prompt_conditions(
        partitions.seen_species,
        descriptions,
        canonical_prompts,
        family_by_species=families,
    )
    condition_prototypes = {
        condition: encode_condition_prototypes(
            prompts,
            partitions.seen_species,
            bioclip,
            bundle.bioclip_tokenizer,
            device=device,
            batch_size=int(config["training"]["eval_batch_size"]),
        )
        for condition, prompts in condition_prompts.items()
    }
    labels = load_labels(config)
    official_train = split_filenames(data_path(config, "train_split"))

    def encode_species(
        species_names: list[str],
        *,
        filenames: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from torch.utils.data import DataLoader

        if filenames is None:
            filenames = [
                name
                for name in official_train
                if name in labels and labels[name] in set(species_names)
            ]
        if not filenames:
            raise ValueError(
                f"No labelled images found for Stage 0 species: {species_names}"
            )
        dataset = BioClipImageDataset(
            filenames,
            data_path(config, "images_dir"),
            transform,
            labels,
            {name: index for index, name in enumerate(species_names)},
            image_cache=load_runtime_image_cache(
                config,
                bundle,
                filenames,
                training=False,
            ),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(config["training"]["eval_batch_size"]),
            shuffle=False,
            num_workers=int(
                config["training"].get(
                    "eval_num_workers",
                    config["training"].get("num_workers", 4),
                )
            ),
        )
        prototype_indices = torch.tensor(
            [partitions.seen_species.index(name) for name in species_names],
            device=device,
        )
        embeddings: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for batch in loader:
            embeddings.append(
                encode_bioclip_images(
                    bioclip, batch["bioclip_image"].to(device)
                ).cpu()
            )
            targets.append(batch["species_index"])
        return (
            torch.cat(embeddings),
            torch.cat(targets),
            prototype_indices,
        )

    pseudo_path = _pseudo_split_path(config)
    if pseudo_path is None:
        seen_species = partitions.seen_species
        pseudo_species: list[str] | None = None
    else:
        if not pseudo_path.exists():
            raise FileNotFoundError(
                f"Pseudo-unseen split not found: {pseudo_path}. "
                "Run make-pseudo-unseen first."
            )
        pseudo = read_json(pseudo_path)
        seen_species = list(pseudo["training_species"])
        pseudo_species = list(pseudo["evaluation_species"])

    eligible_seen = [
        name
        for name in official_train
        if name in labels and labels[name] in set(seen_species)
    ]
    _, seen_validation = _split_labelled_filenames(
        eligible_seen,
        labels,
        int(config["seed"]),
        float(config["validation"].get("seen_fraction", 0.1)),
    )
    seen_embeddings, seen_targets, seen_indices = encode_species(
        seen_species,
        filenames=seen_validation,
    )
    pseudo_data = (
        None if pseudo_species is None else encode_species(pseudo_species)
    )
    from fish_vlm.training.metrics import classification_metrics, harmonic_mean

    results: dict[str, Any] = {
        "conditions": list(PROMPT_CONDITIONS),
        "family_metadata_coverage": len(families)
        / max(1, len(partitions.seen_species)),
    }
    best_condition = ""
    best_score = -1.0
    temperature = float(config["model"].get("temperature", 0.07))
    for condition in PROMPT_CONDITIONS:
        all_prototypes = condition_prototypes[condition]
        seen_logits = text_logits(
            seen_embeddings.to(device),
            all_prototypes.index_select(0, seen_indices),
            1.0 / temperature,
        ).cpu()
        seen_metrics = classification_metrics(
            seen_logits, seen_targets, prefix="bioclip_native"
        )
        seen_accuracy = seen_metrics["bioclip_native_accuracy"]
        results[f"{condition}_seen_accuracy"] = seen_accuracy
        results[f"{condition}_seen_balanced_accuracy"] = seen_metrics[
            "bioclip_native_balanced_accuracy"
        ]
        if pseudo_data is None:
            score = seen_accuracy
        else:
            pseudo_embeddings, pseudo_targets, pseudo_indices = pseudo_data
            pseudo_logits = text_logits(
                pseudo_embeddings.to(device),
                all_prototypes.index_select(0, pseudo_indices),
                1.0 / temperature,
            ).cpu()
            pseudo_metrics = classification_metrics(
                pseudo_logits,
                pseudo_targets,
                prefix="bioclip_native",
            )
            pseudo_accuracy = pseudo_metrics["bioclip_native_accuracy"]
            score = harmonic_mean(seen_accuracy, pseudo_accuracy)
            results[f"{condition}_pseudo_unseen_accuracy"] = pseudo_accuracy
            results[f"{condition}_harmonic_mean"] = score
        if score > best_score:
            best_condition = condition
            best_score = score
    results["best_condition"] = best_condition
    results["best_selection_score"] = best_score
    return results
