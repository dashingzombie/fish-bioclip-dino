"""Classifier-free BioCLIP adaptation for paired and image-only samples."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fish_vlm.config import data_path
from fish_vlm.data.catalog import load_labels
from fish_vlm.data.taxonomy import load_family_mapping
from fish_vlm.domain.datasets import DomainMultiViewDataset, collate_domain_views
from fish_vlm.domain.transforms import build_bioclip_domain_transforms
from fish_vlm.models.bioclip import (
    bioclip_visual_blocks,
    configure_bioclip_tuning,
    encode_bioclip_images,
    load_bioclip,
)
from fish_vlm.prototypes.conditions import (
    build_prompt_conditions,
    encode_condition_prototypes,
    ensemble_prototypes,
)
from fish_vlm.prototypes.image_teacher import (
    load_image_teacher_cache,
    lookup_teacher_embeddings,
)
from fish_vlm.training.early_stopping import EarlyStopping
from fish_vlm.utils.hashing import ordered_names_hash, prompts_hash, stable_json_hash
from fish_vlm.utils.io import read_json, torch_save_atomic, write_json
from fish_vlm.utils.seed import seed_everything


DOMAIN_ACTIVE_LOSSES = [
    "native_bioclip_text",
    "bioclip_pretrained_distillation",
    "bioclip_multiview_consistency",
    "prototype_hard_negative",
]


def prototype_hard_negative_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    neighbours: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Margin against the most confusing precomputed text-prototype neighbours."""
    candidates = neighbours.index_select(0, targets)
    negatives = logits.gather(1, candidates).max(dim=1).values
    positives = logits.gather(1, targets[:, None]).squeeze(1)
    return F.relu(float(margin) + negatives - positives).mean()


def _prototype_bundle(
    config: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    species: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, str, str]:
    processed = (
        Path(config["data"]["root_dir"])
        / config["data"].get("processed_dir", "processed")
    )
    canonical = read_json(processed / "canonical_prompts.json")
    descriptions = read_json(data_path(config, "descriptions_json"))
    conditions = build_prompt_conditions(
        species,
        descriptions,
        canonical,
        family_by_species=load_family_mapping(config, species),
    )
    weights = {
        str(name): float(weight)
        for name, weight in config["text"]["prototype_ensemble"]["weights"].items()
        if float(weight) > 0
    }
    encoded = {
        name: encode_condition_prototypes(
            conditions[name],
            species,
            model,
            tokenizer,
            device=device,
            batch_size=int(config["domain"]["bioclip"].get("eval_batch_size", 128)),
        )
        for name in weights
    }
    prototypes = ensemble_prototypes(encoded, weights)
    base_hash = prompts_hash(canonical, species)
    ensemble_hash = stable_json_hash(
        {
            "base_prompt_hash": base_hash,
            "weights": {
                str(name): float(weight)
                for name, weight in config["text"]["prototype_ensemble"]["weights"].items()
            },
            "prompts": {name: conditions[name] for name in sorted(encoded)},
        }
    )
    return prototypes, ensemble_hash, prompts_hash(canonical, species)


def _macro_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    recalls = [
        (predictions[targets == target] == target).float().mean()
        for target in targets.unique(sorted=True)
    ]
    return float(torch.stack(recalls).mean()) if recalls else 0.0


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    prototypes: torch.Tensor,
    candidate_indices: torch.Tensor,
    device: torch.device,
    logit_scale: float,
) -> float:
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    candidate_prototypes = prototypes.index_select(0, candidate_indices)
    absolute_to_local = {
        int(absolute): local
        for local, absolute in enumerate(candidate_indices.cpu().tolist())
    }
    for batch in loader:
        embeddings = torch.stack(
            [
                encode_bioclip_images(model, view.to(device, non_blocking=True))
                for view in batch["views"]
            ]
        ).mean(dim=0)
        embeddings = F.normalize(embeddings, dim=-1)
        logits = float(logit_scale) * embeddings @ candidate_prototypes.T
        predictions.append(logits.argmax(dim=-1).cpu())
        targets.append(
            torch.tensor(
                [absolute_to_local[int(value)] for value in batch["target"]],
                dtype=torch.long,
            )
        )
    if not targets:
        return 0.0
    return _macro_accuracy(torch.cat(predictions), torch.cat(targets))


def _loader(
    dataset: DomainMultiViewDataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=shuffle and len(dataset) >= batch_size,
        collate_fn=collate_domain_views,
    )


def _configure_visual_phase(
    model: torch.nn.Module,
    *,
    full: bool,
    last_blocks: int,
) -> None:
    configure_bioclip_tuning(
        model,
        "full_finetune" if full else "partial_finetune",
        unfreeze_last_blocks=last_blocks,
    )


def train_bioclip_domain(
    config: dict[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    """Adapt only BioCLIP's image tower while retaining frozen text geometry."""
    if config["model"].get("bioclip_classifier", {}).get("enabled", False):
        raise ValueError("BioCLIP domain adaptation forbids classifier heads")
    if not config["model"].get("bioclip", {}).get("freeze_text_encoder", True):
        raise ValueError("BioCLIP domain adaptation requires a frozen text encoder")
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = config["domain"]
    training = domain["bioclip"]
    manifest = read_json(domain["manifest_path"])
    checkpoint_name = str(config["model"]["bioclip"]["checkpoint"])
    model, _, eval_transform, tokenizer, _ = load_bioclip(checkpoint_name)
    model = model.to(device)
    model.requires_grad_(False)
    all_species = list(manifest["species"]["all"])
    seen_species = list(manifest["species"]["seen"])
    unseen_species = list(manifest["species"]["unseen"])
    prototypes, all_text_hash, canonical_all_hash = _prototype_bundle(
        config, model, tokenizer, all_species, device
    )
    _, seen_text_hash, _ = _prototype_bundle(
        config, model, tokenizer, seen_species, device
    )
    train_transform, validation_transform = build_bioclip_domain_transforms(
        model, eval_transform
    )
    source_labels = load_labels(config)
    official_train = set(manifest["splits"]["train"])
    labels = {
        name: label
        for name, label in source_labels.items()
        if name in official_train
    }
    species_to_index = {name: index for index, name in enumerate(all_species)}
    paired_species = set(manifest["paired_training_species"])
    images_dir = data_path(config, "images_dir")
    train_dataset = DomainMultiViewDataset(
        list(manifest["adaptation_train"]),
        images_dir,
        train_transform,
        labels=labels,
        species_to_index=species_to_index,
        paired_species=paired_species,
    )
    seen_validation_names = [
        name
        for name in manifest["labelled_validation"]
        if labels[name] in paired_species
    ]
    pseudo_species = set(seen_species) - paired_species
    pseudo_validation_names = sorted(
        name for name, species in labels.items() if species in pseudo_species
    )
    seen_validation = DomainMultiViewDataset(
        seen_validation_names,
        images_dir,
        validation_transform,
        labels=labels,
        species_to_index=species_to_index,
        paired_species=paired_species,
    )
    pseudo_validation = DomainMultiViewDataset(
        pseudo_validation_names,
        images_dir,
        validation_transform,
        labels=labels,
        species_to_index=species_to_index,
        paired_species=pseudo_species,
    )
    workers = int(training.get("num_workers", 8))
    train_loader = _loader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        workers=workers,
        shuffle=True,
        device=device,
    )
    seen_loader = _loader(
        seen_validation,
        batch_size=int(training.get("eval_batch_size", training["batch_size"])),
        workers=max(0, workers // 2),
        shuffle=False,
        device=device,
    )
    pseudo_loader = _loader(
        pseudo_validation,
        batch_size=int(training.get("eval_batch_size", training["batch_size"])),
        workers=max(0, workers // 2),
        shuffle=False,
        device=device,
    )
    all_filenames = [
        name
        for split in ("train", "test", "unseen")
        for name in manifest["splits"][split]
    ]
    teacher_cache = load_image_teacher_cache(
        Path(os.environ.get("FISH_VLM_CACHE_DIR", config.get("cache_dir", "cache")))
        / "bioclip_images"
        / "all_embeddings.pt",
        expected_filenames=all_filenames,
        checkpoint=checkpoint_name,
        transform_hash=None,
    )
    similarity = prototypes @ prototypes.T
    similarity.fill_diagonal_(float("-inf"))
    neighbours = similarity.topk(
        k=min(int(training["hard_negative_top_k"]), len(all_species) - 1),
        dim=1,
    ).indices
    class_counts: dict[int, int] = {}
    for filename in manifest["labelled_train"]:
        species = labels[filename]
        if species in paired_species:
            index = species_to_index[species]
            class_counts[index] = class_counts.get(index, 0) + 1
    class_weights = torch.ones(len(all_species), device=device)
    for index, count in class_counts.items():
        class_weights[index] = 1.0 / max(1, count)
    if class_counts:
        selected = torch.tensor(sorted(class_counts), device=device)
        class_weights[selected] /= class_weights[selected].mean()

    partial_epochs = int(training["partial_epochs"])
    max_epochs = int(training["max_epochs"])
    _configure_visual_phase(
        model,
        full=partial_epochs == 0,
        last_blocks=int(training["unfreeze_last_blocks"]),
    )

    def make_optimizer() -> torch.optim.Optimizer:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError("BioCLIP adaptation exposed no visual parameters")
        return torch.optim.AdamW(
            parameters,
            lr=float(training["visual_lr"]),
            weight_decay=float(training["weight_decay"]),
        )

    optimizer = make_optimizer()
    early = EarlyStopping(int(training["early_stopping_patience_epochs"]))
    output_dir = Path(domain["bioclip_output_dir"])
    best_path = output_dir / "checkpoints" / "best.pt"
    last_path = output_dir / "checkpoints" / "last.pt"
    start_epoch = 0
    if resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["bioclip_state"])
        start_epoch = int(state["epoch"])
        _configure_visual_phase(
            model,
            full=start_epoch >= partial_epochs,
            last_blocks=int(training["unfreeze_last_blocks"]),
        )
        optimizer = make_optimizer()
        optimizer.load_state_dict(state["optimizer_state"])
        early = EarlyStopping(**state["early_stopping"])
    logit_scale = float(training["logit_scale"])
    use_amp = bool(training.get("use_amp", True)) and device.type == "cuda"
    metrics: dict[str, Any] = {}
    for epoch in range(start_epoch, max_epochs):
        if epoch == partial_epochs and partial_epochs > 0:
            _configure_visual_phase(
                model,
                full=True,
                last_blocks=int(training["unfreeze_last_blocks"]),
            )
            optimizer = make_optimizer()
        model.train()
        running: dict[str, float] = {}
        samples = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            views = [view.to(device, non_blocking=True) for view in batch["views"]]
            paired_mask = batch["has_text"].to(device)
            targets = batch["target"].to(device)
            teacher = lookup_teacher_embeddings(
                teacher_cache, batch["filename"]
            ).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                embeddings = [encode_bioclip_images(model, view) for view in views]
                consistency = (1.0 - (embeddings[0] * embeddings[1]).sum(dim=-1)).mean()
                distillation = torch.stack(
                    [1.0 - (embedding * teacher).sum(dim=-1) for embedding in embeddings]
                ).mean()
                text_alignment = embeddings[0].new_zeros(())
                hard_negative = embeddings[0].new_zeros(())
                if paired_mask.any():
                    paired_embeddings = torch.stack(embeddings).mean(dim=0)[paired_mask]
                    paired_targets = targets[paired_mask]
                    logits = logit_scale * paired_embeddings @ prototypes.T
                    per_sample = F.cross_entropy(
                        logits, paired_targets, reduction="none"
                    )
                    text_alignment = (
                        per_sample * class_weights.index_select(0, paired_targets)
                    ).mean()
                    hard_negative = prototype_hard_negative_loss(
                        logits,
                        paired_targets,
                        neighbours,
                        margin=float(training["hard_negative_margin"]),
                    )
                loss = (
                    float(training["text_alignment_weight"]) * text_alignment
                    + float(training["distillation_weight"]) * distillation
                    + float(training["consistency_weight"]) * consistency
                    + float(training["hard_negative_weight"]) * hard_negative
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(training.get("gradient_clip_norm", 1.0)),
            )
            optimizer.step()
            batch_size = len(batch["filename"])
            samples += batch_size
            for name, value in {
                "loss": loss,
                "text_alignment": text_alignment,
                "distillation": distillation,
                "consistency": consistency,
                "hard_negative": hard_negative,
            }.items():
                running[name] = running.get(name, 0.0) + float(value) * batch_size
        paired_indices = torch.tensor(
            [species_to_index[name] for name in sorted(paired_species)],
            device=device,
        )
        pseudo_indices = torch.tensor(
            [species_to_index[name] for name in sorted(pseudo_species)],
            device=device,
        )
        seen_accuracy = _evaluate(
            model,
            seen_loader,
            prototypes,
            paired_indices,
            device,
            logit_scale,
        )
        pseudo_accuracy = (
            _evaluate(
                model,
                pseudo_loader,
                prototypes,
                pseudo_indices,
                device,
                logit_scale,
            )
            if len(pseudo_indices)
            else seen_accuracy
        )
        harmonic = (
            2.0 * seen_accuracy * pseudo_accuracy / (seen_accuracy + pseudo_accuracy)
            if seen_accuracy + pseudo_accuracy > 0
            else 0.0
        )
        improved, should_stop = early.update(harmonic)
        metrics = {
            "epoch": epoch + 1,
            **{name: value / max(1, samples) for name, value in running.items()},
            "seen_macro_accuracy": seen_accuracy,
            "pseudo_unseen_macro_accuracy": pseudo_accuracy,
            "seen_unseen_harmonic_mean": harmonic,
            "selection_value": harmonic,
            "phase": "partial_visual" if epoch < partial_epochs else "full_visual",
            "manifest_hash": manifest["manifest_hash"],
            "bioclip_classifier_head_used": False,
        }
        early_state = {
            "patience": early.patience,
            "min_delta": early.min_delta,
            "best": early.best,
            "bad_evaluations": early.bad_evaluations,
        }
        torch_save_atomic(
            {
                "bioclip_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch + 1,
                "early_stopping": early_state,
                "metrics": metrics,
            },
            last_path,
        )
        if improved:
            checkpoint = {
                "model_state": {
                    f"bioclip.{name}": value
                    for name, value in model.state_dict().items()
                },
                "projector_state": {"domain_adaptation": torch.ones(1)},
                "supervised_head_state": None,
                "bioclip_adapter_state": None,
                "bioclip_classifier_state": None,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None,
                "gradient_scaler_state": None,
                "step": epoch + 1,
                "best_metric": harmonic,
                "resolved_configuration": {
                    **config,
                    "training": {
                        **config["training"],
                        "stage": "bioclip_domain_adaptation",
                    },
                    "model": {
                        **config["model"],
                        "tuning_mode": "full_finetune",
                        "bioclip": {
                            **config["model"]["bioclip"],
                            "freeze_text_encoder": True,
                            "freeze_image_encoder": False,
                        },
                    },
                },
                "dino_pooling_strategy": "not_used_by_bioclip_domain_adaptation",
                "pseudo_unseen_split_hash": None,
                "calibration_metadata": None,
                "dino_model_name": config["model"]["dino"]["name"],
                "dino_checkpoint_source": "not_used_by_bioclip_domain_adaptation",
                "bioclip_checkpoint": checkpoint_name,
                "text_prototype_hash": seen_text_hash,
                "all_text_prototype_hash": all_text_hash,
                "canonical_prompt_hash": canonical_all_hash,
                "seen_species": seen_species,
                "unseen_species": unseen_species,
                "training_species": sorted(paired_species),
                "training_species_hash": ordered_names_hash(sorted(paired_species)),
                "active_losses": DOMAIN_ACTIVE_LOSSES,
                "domain_manifest_hash": manifest["manifest_hash"],
                "official_test_labels_loaded": False,
                "official_unseen_labels_loaded": False,
            }
            torch_save_atomic(checkpoint, best_path)
            write_json(output_dir / "metrics" / "best.json", metrics)
        if should_stop:
            break
    return metrics
