"""Separate evaluation and purpose-specific selection of DINO checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path, load_config
from fish_vlm.data.catalog import load_labels, split_filenames
from fish_vlm.data.taxonomy import load_family_mapping
from fish_vlm.training.checkpoint import load_checkpoint
from fish_vlm.training.distributed import DistributedContext
from fish_vlm.training.metrics import harmonic_mean
from fish_vlm.training.train import (
    _data_processed_path,
    _pseudo_split_path,
    _split_labelled_filenames,
    build_runtime,
    evaluate_loader,
    load_candidate_prototypes,
    make_loader,
)
from fish_vlm.utils.hashing import ordered_names_hash, prompts_hash
from fish_vlm.utils.io import read_json


@torch.no_grad()
def evaluate_partitioned_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Evaluate one checkpoint on matched seen and species-disjoint partitions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_runtime(config, device=device)
    all_prototypes, all_names, cache = load_candidate_prototypes(
        config, bundle, "seen", device=device
    )
    pseudo_path = _pseudo_split_path(config)
    if pseudo_path is None or not pseudo_path.exists():
        raise FileNotFoundError(
            "Stage evaluation requires the configured pseudo-unseen split"
        )
    pseudo = read_json(pseudo_path)
    seen_species = list(pseudo["training_species"])
    pseudo_species = list(pseudo["evaluation_species"])
    canonical_prompts = read_json(
        _data_processed_path(config, "canonical_prompts.json")
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        bundle.model,
        expected_seen_species=bundle.partitions.seen_species,
        expected_unseen_species=bundle.partitions.unseen_species,
        expected_text_prototype_hash=cache["prompt_hash"],
        expected_canonical_prompt_hash=prompts_hash(
            canonical_prompts, bundle.partitions.all_species
        ),
        expected_training_species_hash=ordered_names_hash(seen_species),
        expected_dino_model_name=str(config["model"]["dino"]["name"]),
        expected_dino_checkpoint_source=bundle.dino_source,
        expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
        strict=False,
    )

    labels = load_labels(config)
    official_train = split_filenames(data_path(config, "train_split"))
    eligible_seen = [
        filename
        for filename in official_train
        if filename in labels and labels[filename] in set(seen_species)
    ]
    _, seen_validation = _split_labelled_filenames(
        eligible_seen,
        labels,
        int(config["seed"]),
        float(config["validation"].get("seen_fraction", 0.1)),
    )
    pseudo_validation = [
        filename
        for filename in official_train
        if filename in labels and labels[filename] in set(pseudo_species)
    ]
    context = DistributedContext(0, 1, 0, device)
    seen_loader, _ = make_loader(
        seen_validation,
        config,
        bundle,
        labels,
        seen_species,
        training=False,
        context=context,
    )
    pseudo_loader, _ = make_loader(
        pseudo_validation,
        config,
        bundle,
        labels,
        pseudo_species,
        training=False,
        context=context,
    )
    seen_indices = torch.tensor(
        [all_names.index(name) for name in seen_species], device=device
    )
    pseudo_indices = torch.tensor(
        [all_names.index(name) for name in pseudo_species], device=device
    )
    families = load_family_mapping(config, all_names)
    seen_metrics = evaluate_loader(
        bundle.model,
        seen_loader,
        all_prototypes.index_select(0, seen_indices),
        device,
        species_names=seen_species,
        family_by_species=families,
    )
    pseudo_metrics = evaluate_loader(
        bundle.model,
        pseudo_loader,
        all_prototypes.index_select(0, pseudo_indices),
        device,
        species_names=pseudo_species,
        family_by_species=families,
    )
    seen_branches = [
        branch
        for branch in (
            "dino_text",
            "bioclip_native",
            "fused_text",
            "supervised",
            "supervised_plus_text",
        )
        if f"{branch}_accuracy" in seen_metrics
    ]
    pseudo_branches = [
        branch
        for branch in ("dino_text", "bioclip_native", "fused_text")
        if f"{branch}_accuracy" in pseudo_metrics
    ]
    seen_branch = max(
        seen_branches,
        key=lambda branch: float(seen_metrics[f"{branch}_accuracy"]),
    )
    pseudo_branch = max(
        pseudo_branches,
        key=lambda branch: float(pseudo_metrics[f"{branch}_accuracy"]),
    )
    seen_accuracy = float(seen_metrics[f"{seen_branch}_accuracy"])
    pseudo_accuracy = float(
        pseudo_metrics[f"{pseudo_branch}_accuracy"]
    )
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "seen_selection_branch": seen_branch,
        "pseudo_unseen_selection_branch": pseudo_branch,
        "seen_accuracy": seen_accuracy,
        "pseudo_unseen_accuracy": pseudo_accuracy,
        "harmonic_mean": harmonic_mean(seen_accuracy, pseudo_accuracy),
        "seen_text_retrieval_accuracy": float(
            seen_metrics["dino_text_accuracy"]
        ),
        "pseudo_unseen_text_retrieval_accuracy": float(
            pseudo_metrics["dino_text_accuracy"]
        ),
        "text_retrieval_accuracy": float(
            pseudo_metrics["dino_text_accuracy"]
        ),
        "seen_genus_accuracy": seen_metrics.get(
            f"{seen_branch}_genus_accuracy"
        ),
        "pseudo_unseen_genus_accuracy": pseudo_metrics.get(
            f"{pseudo_branch}_genus_accuracy"
        ),
        "genus_accuracy": pseudo_metrics.get(
            f"{pseudo_branch}_genus_accuracy"
        ),
        "seen_family_accuracy": seen_metrics.get(
            f"{seen_branch}_family_accuracy"
        ),
        "pseudo_unseen_family_accuracy": pseudo_metrics.get(
            f"{pseudo_branch}_family_accuracy"
        ),
        "family_accuracy": pseudo_metrics.get(
            f"{pseudo_branch}_family_accuracy"
        ),
        "family_coverage": pseudo_metrics.get(
            f"{pseudo_branch}_family_coverage"
        ),
        "seen_metrics": seen_metrics,
        "pseudo_unseen_metrics": pseudo_metrics,
    }


def evaluate_stage_checkpoints(
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the three DINO stages and select separate purpose checkpoints."""
    config_path = Path(pipeline_config["_config_path"])
    root = config_path.parent.parent
    stage_reports: dict[str, dict[str, Any]] = {}
    for stage in pipeline_config["workflow"]["training_stages"]:
        if stage["name"] not in {
            "projection_only",
            "final_block",
            "joint_supervised_text",
        }:
            continue
        stage_config_path = Path(stage["config"])
        checkpoint_path = Path(stage["checkpoint"])
        if not stage_config_path.is_absolute():
            stage_config_path = root / stage_config_path
        if not checkpoint_path.is_absolute():
            checkpoint_path = root / checkpoint_path
        report = evaluate_partitioned_checkpoint(
            load_config(stage_config_path),
            checkpoint_path,
        )
        seen_inference = Path(stage["seen_inference_config"])
        unseen_inference = Path(
            stage.get(
                "unseen_inference_config",
                pipeline_config["workflow"]["unseen_inference_config"],
            )
        )
        if not seen_inference.is_absolute():
            seen_inference = root / seen_inference
        if not unseen_inference.is_absolute():
            unseen_inference = root / unseen_inference
        report["seen_inference_config"] = str(seen_inference)
        report["unseen_inference_config"] = str(unseen_inference)
        stage_reports[str(stage["name"])] = report
    if len(stage_reports) != 3:
        raise ValueError("Pipeline must define all three DINO stage checkpoints")
    best_seen = max(
        stage_reports, key=lambda name: stage_reports[name]["seen_accuracy"]
    )
    best_unseen = max(
        stage_reports,
        key=lambda name: stage_reports[name]["pseudo_unseen_accuracy"],
    )
    best_joint = max(
        stage_reports, key=lambda name: stage_reports[name]["harmonic_mean"]
    )
    return {
        "stages": stage_reports,
        "selection": {
            "seen": {
                "stage": best_seen,
                "checkpoint": stage_reports[best_seen]["checkpoint"],
                "metric": "seen_accuracy",
                "inference_config": stage_reports[best_seen][
                    "seen_inference_config"
                ],
            },
            "unseen": {
                "stage": best_unseen,
                "checkpoint": stage_reports[best_unseen]["checkpoint"],
                "metric": "pseudo_unseen_accuracy",
                "inference_config": stage_reports[best_unseen][
                    "unseen_inference_config"
                ],
            },
            "joint": {
                "stage": best_joint,
                "checkpoint": stage_reports[best_joint]["checkpoint"],
                "metric": "harmonic_mean",
                "inference_config": stage_reports[best_joint][
                    "unseen_inference_config"
                ],
            },
        },
    }
