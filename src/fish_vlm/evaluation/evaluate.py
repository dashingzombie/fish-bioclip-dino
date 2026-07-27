"""Checkpoint evaluation over labelled seen or pseudo-unseen images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path
from fish_vlm.data.catalog import load_labels, split_filenames
from fish_vlm.data.datasets import BioClipImageDataset
from fish_vlm.data.catalog import official_split_counts
from fish_vlm.models.bioclip import encode_bioclip_images
from fish_vlm.models.multimodal import text_logits
from fish_vlm.prototypes.text import load_text_prototype_cache
from fish_vlm.evaluation.reports import add_selection_metrics
from fish_vlm.training.checkpoint import load_checkpoint
from fish_vlm.training.distributed import DistributedContext
from fish_vlm.training.train import (
    _data_processed_path,
    build_runtime,
    _cache_path,
    _pseudo_split_path,
    ensure_partitions,
    evaluate_loader,
    load_candidate_prototypes,
    load_runtime_image_cache,
    make_loader,
)
from fish_vlm.utils.hashing import prompts_hash
from fish_vlm.utils.io import read_json


def evaluate_checkpoint(config: dict[str, Any], checkpoint_path: str | Path) -> dict[str, float]:
    """Evaluate branch metrics without using official image-only labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    split_name = config.get("evaluation", {}).get("split", "train")
    candidate_set = config.get("evaluation", {}).get("candidate_set", "seen")
    prototypes, species_names, cache = load_candidate_prototypes(config, bundle, candidate_set, device=device)
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
    if split_name in {"test", "unseen"}:
        raise ValueError("Official test/unseen labels are not assumed; use infer instead")
    filenames = [name for name in split_filenames(data_path(config, "train_split")) if name in labels and labels[name] in set(species_names)]
    context = DistributedContext(0, 1, 0, device)
    loader, _ = make_loader(
        filenames, config, bundle, labels, species_names, training=False, context=context
    )
    metrics = evaluate_loader(bundle.model, loader, prototypes, device)
    metrics["checkpoint_step"] = float(checkpoint["step"])
    return metrics


@torch.no_grad()
def evaluate_bioclip_zero_shot(config: dict[str, Any]) -> dict[str, float]:
    """Run the no-training Stage 0 native BioCLIP baseline on labelled train data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not config["model"]["bioclip_image_path"].get("enabled", False):
        raise ValueError("Stage 0 requires model.bioclip_image_path.enabled=true")
    bundle = build_runtime(config, device=device)
    partitions = bundle.partitions
    checkpoint_name = config["model"]["bioclip"]["checkpoint"]
    bioclip = bundle.model.bioclip
    if bioclip is None:
        raise ValueError("Stage 0 requires a loaded BioCLIP image encoder")
    transform = bundle.bioclip_eval_transform
    embedding_dim = bundle.embedding_dim
    prompts = read_json(_data_processed_path(config, "canonical_prompts.json"))
    cache = load_text_prototype_cache(
        _cache_path(config, "text", "text_prototypes_seen.pt"),
        species_names=partitions.seen_species,
        checkpoint=checkpoint_name,
        prompt_hash=prompts_hash(prompts, partitions.seen_species),
        embedding_dim=embedding_dim,
    )
    all_prototypes = cache["embeddings"].to(device)
    labels = load_labels(config)
    official_train = split_filenames(data_path(config, "train_split"))

    def evaluate_species(species_names: list[str]) -> dict[str, float]:
        from torch.utils.data import DataLoader
        from fish_vlm.training.metrics import classification_metrics

        filenames = [
            name for name in official_train
            if name in labels and labels[name] in set(species_names)
        ]
        if not filenames:
            raise ValueError(f"No labelled images found for Stage 0 species: {species_names}")
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
            num_workers=int(config["training"].get("num_workers", 4)),
        )
        indices = torch.tensor(
            [partitions.seen_species.index(name) for name in species_names],
            device=device,
        )
        prototypes = all_prototypes.index_select(0, indices)
        logits: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        for batch in loader:
            embedding = encode_bioclip_images(bioclip, batch["bioclip_image"].to(device))
            logits.append(text_logits(embedding, prototypes, 1.0 / 0.07).cpu())
            targets.append(batch["species_index"])
        return classification_metrics(
            torch.cat(logits), torch.cat(targets), prefix="bioclip_native"
        )

    pseudo_path = _pseudo_split_path(config)
    if pseudo_path is None:
        return evaluate_species(partitions.seen_species)
    if not pseudo_path.exists():
        raise FileNotFoundError(
            f"Pseudo-unseen split not found: {pseudo_path}. Run make-pseudo-unseen first."
        )
    pseudo = read_json(pseudo_path)
    seen_metrics = evaluate_species(list(pseudo["training_species"]))
    unseen_metrics = evaluate_species(list(pseudo["evaluation_species"]))
    metrics = {
        **{f"seen_{key}": value for key, value in seen_metrics.items()},
        **{f"pseudo_unseen_{key}": value for key, value in unseen_metrics.items()},
    }
    test_count, unseen_count = official_split_counts(config)
    return add_selection_metrics(
        metrics,
        seen_accuracy=seen_metrics["bioclip_native_accuracy"],
        pseudo_unseen_accuracy=unseen_metrics["bioclip_native_accuracy"],
        test_count=test_count,
        unseen_count=unseen_count,
    )
