"""End-to-end staged training entry point."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from fish_vlm.config import data_path
from fish_vlm.data.catalog import load_labels, split_filenames
from fish_vlm.data.collate import collate_multiview
from fish_vlm.data.datasets import FishMultiViewDataset
from fish_vlm.data.image_cache import (
    DeterministicImageCacheCollection,
    load_image_cache_collection,
)
from fish_vlm.data.partitions import (
    ClassPartitions,
    build_class_partitions,
    create_and_save_partitions,
    load_partitions,
)
from fish_vlm.data.transforms import build_dino_transform, transform_fingerprint
from fish_vlm.losses.total import compute_total_loss
from fish_vlm.models.bioclip import assert_frozen_bioclip, load_bioclip
from fish_vlm.models.bioclip_adapter import BioClipResidualAdapter
from fish_vlm.models.cosine_classifier import CosineClassifier
from fish_vlm.models.dino import POOLING_STRATEGY, load_dino
from fish_vlm.models.multimodal import FishMultimodalModel
from fish_vlm.models.fusion import CalibrationParameters, fuse_seen_probabilities, fuse_text_probabilities
from fish_vlm.models.projector import LearnableLogitScale, build_projector
from fish_vlm.models.representation_teacher import DinoProjectionTeacher
from fish_vlm.prototypes.image_teacher import load_image_teacher_cache, lookup_teacher_embeddings
from fish_vlm.prototypes.conditions import (
    build_prompt_conditions,
    encode_condition_prototypes,
    ensemble_prototypes,
)
from fish_vlm.prototypes.text import (
    load_text_prototype_cache,
    validate_bioclip_text_identity,
)
from fish_vlm.training.checkpoint import load_checkpoint, save_checkpoint
from fish_vlm.training.distributed import (
    DistributedContext,
    cleanup_distributed,
    initialise_distributed,
    reduce_sum,
)
from fish_vlm.training.early_stopping import EarlyStopping
from fish_vlm.training.metrics import (
    distributed_classification_metrics,
    distributed_hierarchical_accuracy,
    selection_value,
)
from fish_vlm.training.optimizer import build_optimizer
from fish_vlm.training.stages import configure_training_stage, trainable_parameter_count
from fish_vlm.training.wandb_logging import ScientificWandbLogger
from fish_vlm.utils.hashing import ordered_names_hash, stable_json_hash
from fish_vlm.utils.io import read_json, read_pickle, write_json
from fish_vlm.utils.seed import seed_everything

LOGGER = logging.getLogger(__name__)


def _data_processed_path(config: dict[str, Any], filename: str) -> Path:
    return Path(config["data"]["root_dir"]).expanduser() / config["data"].get("processed_dir", "processed") / filename


def _cache_path(config: dict[str, Any], *parts: str) -> Path:
    root = os.environ.get("FISH_VLM_CACHE_DIR", config.get("cache_dir", "cache"))
    return Path(root).expanduser().joinpath(*parts)


def _pseudo_split_path(config: dict[str, Any]) -> Path | None:
    pseudo = config.get("validation", {}).get("pseudo_unseen", {})
    value = pseudo.get("split_path")
    if value and value != "auto":
        return Path(value)
    if pseudo.get("enabled", False):
        split_seed = int(pseudo.get("split_seed", config["seed"]))
        return _data_processed_path(
            config,
            f"pseudo_unseen/{pseudo['strategy']}_seed_{split_seed}.json",
        )
    return None


@dataclass
class RuntimeBundle:
    """Loaded model and scientific identities shared by commands."""

    model: FishMultimodalModel
    partitions: ClassPartitions
    dino_source: str
    bioclip_checkpoint: str
    bioclip_text_encoder: torch.nn.Module
    bioclip_eval_transform: Any
    bioclip_tokenizer: Any
    dino_eval_transform: Any
    embedding_dim: int


def ensure_partitions(config: dict[str, Any]) -> ClassPartitions:
    """Load partitions and prove they still match the organiser source files."""
    path = _data_processed_path(config, "class_partitions.json")
    if not path.exists():
        create_and_save_partitions(
            data_path(config, "labels_json"),
            data_path(config, "all_classes_pickle"),
            path,
        )
    persisted = load_partitions(path)
    current = build_class_partitions(
        load_labels(config),
        list(read_pickle(data_path(config, "all_classes_pickle"))),
    )
    if persisted != current:
        raise ValueError(
            "Persisted class partitions are stale relative to label_train.json "
            "or all_classes.pkl; rerun prepare-prompts"
        )
    return persisted


def build_runtime(config: dict[str, Any], *, device: torch.device | str) -> RuntimeBundle:
    """Load encoders and construct optional branches from configuration."""
    partitions = ensure_partitions(config)
    dino, dino_dim, dino_source = load_dino(config["model"]["dino"])
    checkpoint = config["model"]["bioclip"]["checkpoint"]
    bioclip, _, bioclip_eval, bioclip_tokenizer, embedding_dim = load_bioclip(
        checkpoint
    )
    bioclip = bioclip.to(device)
    projector = build_projector(config["model"]["projector"], dino_dim, embedding_dim)
    logit_scale = LearnableLogitScale(float(config["model"].get("temperature", 0.07)))
    supervised = None
    supervised_cfg = config["model"]["supervised_head"]
    if supervised_cfg.get("enabled", False):
        supervised_classes = len(partitions.seen_species)
        pseudo_path = _pseudo_split_path(config)
        if pseudo_path and pseudo_path.exists():
            supervised_classes = len(read_json(pseudo_path)["training_species"])
        supervised = CosineClassifier(
            dino_dim, supervised_classes, float(supervised_cfg.get("initial_scale", 20.0))
        )
    adapter = None
    path_cfg = config["model"]["bioclip_image_path"]
    adapter_cfg = path_cfg.get("adapter", {})
    if path_cfg.get("mode") == "adapter" or adapter_cfg.get("enabled", False):
        adapter = BioClipResidualAdapter(
            embedding_dim,
            int(adapter_cfg.get("hidden_dim", 512)),
            float(adapter_cfg.get("dropout", 0.1)),
        )
    bioclip_classifier = None
    bioclip_classifier_cfg = config["model"].get(
        "bioclip_classifier", {}
    )
    if bioclip_classifier_cfg.get("enabled", False):
        classifier_classes = len(partitions.seen_species)
        pseudo_path = _pseudo_split_path(config)
        if pseudo_path and pseudo_path.exists():
            classifier_classes = len(
                read_json(pseudo_path)["training_species"]
            )
        bioclip_classifier = CosineClassifier(
            embedding_dim,
            classifier_classes,
            float(bioclip_classifier_cfg.get("initial_scale", 20.0)),
        )
    native_model = bioclip if path_cfg.get("enabled", False) and path_cfg.get("mode") != "disabled" else None
    model = FishMultimodalModel(
        dino,
        projector,
        logit_scale,
        bioclip=native_model,
        bioclip_adapter=adapter,
        supervised_head=supervised,
        bioclip_classifier=bioclip_classifier,
        bioclip_text_space=str(
            path_cfg.get("text_space", "native")
        ),
        bioclip_classifier_space=str(
            bioclip_classifier_cfg.get("feature_space", "native")
        ),
    ).to(device)
    if native_model is not None:
        assert_frozen_bioclip(native_model)
    return RuntimeBundle(
        model=model,
        partitions=partitions,
        dino_source=dino_source,
        bioclip_checkpoint=checkpoint,
        bioclip_text_encoder=bioclip,
        bioclip_eval_transform=bioclip_eval,
        bioclip_tokenizer=bioclip_tokenizer,
        dino_eval_transform=build_dino_transform(dino, training=False),
        embedding_dim=embedding_dim,
    )


def load_candidate_prototypes(
    config: dict[str, Any],
    bundle: RuntimeBundle,
    candidate_set: str,
    *,
    device: torch.device | str,
) -> tuple[torch.Tensor, list[str], dict[str, Any]]:
    """Load and validate one seen/unseen/all text matrix."""
    names = getattr(bundle.partitions, f"{candidate_set}_species")
    prompts = read_json(_data_processed_path(config, "canonical_prompts.json"))
    from fish_vlm.utils.hashing import prompts_hash

    expected_hash = prompts_hash(prompts, names)
    cache = load_text_prototype_cache(
        _cache_path(config, "text", f"text_prototypes_{candidate_set}.pt"),
        species_names=names,
        checkpoint=bundle.bioclip_checkpoint,
        prompt_hash=expected_hash,
        embedding_dim=bundle.embedding_dim,
    )
    validate_bioclip_text_identity(
        cache,
        bundle.bioclip_text_encoder,
        bundle.bioclip_tokenizer,
        device=device,
    )
    ensemble_cfg = config.get("text", {}).get("prototype_ensemble", {})
    if not ensemble_cfg.get("enabled", False):
        return cache["embeddings"].to(device), names, cache
    weights = {
        str(name): float(weight)
        for name, weight in ensemble_cfg.get("weights", {}).items()
    }
    from fish_vlm.data.taxonomy import load_family_mapping

    descriptions = read_json(data_path(config, "descriptions_json"))
    if not isinstance(descriptions, dict):
        raise TypeError("descriptions_json must contain a JSON object")
    condition_prompts = build_prompt_conditions(
        names,
        descriptions,
        prompts,
        family_by_species=load_family_mapping(config, names),
    )
    unknown = sorted(set(weights) - set(condition_prompts))
    if unknown:
        raise ValueError(f"Unknown prototype ensemble conditions: {unknown}")
    encoded = {
        condition: encode_condition_prototypes(
            condition_prompts[condition],
            names,
            bundle.bioclip_text_encoder,
            bundle.bioclip_tokenizer,
            device=device,
            batch_size=int(config["training"].get("eval_batch_size", 128)),
        )
        for condition, weight in weights.items()
        if weight > 0
    }
    positive_weights = {
        condition: weight
        for condition, weight in weights.items()
        if weight > 0
    }
    prototypes = ensemble_prototypes(encoded, positive_weights)
    ensemble_cache = dict(cache)
    ensemble_cache["prompt_hash"] = stable_json_hash(
        {
            "base_prompt_hash": expected_hash,
            "weights": weights,
            "prompts": {
                condition: condition_prompts[condition]
                for condition in sorted(encoded)
            },
        }
    )
    ensemble_cache["prototype_ensemble"] = weights
    ensemble_cache["embeddings"] = prototypes.cpu()
    return prototypes.to(device), names, ensemble_cache


def load_runtime_image_cache(
    config: dict[str, Any],
    bundle: RuntimeBundle,
    filenames: list[str],
    *,
    training: bool,
) -> DeterministicImageCacheCollection | None:
    """Load deterministic model inputs when enabled for this loader."""
    cache_config = config["data"].get("deterministic_transform_cache", {})
    if not cache_config.get("enabled", False):
        return None
    if training and not config["training"].get("conservative_augmentation", True):
        return None
    return load_image_cache_collection(
        _cache_path(config, "image_transforms"),
        required_filenames=filenames,
        dino_model_name=str(config["model"]["dino"]["name"]),
        bioclip_checkpoint=bundle.bioclip_checkpoint,
        dino_transform_hash=transform_fingerprint(bundle.dino_eval_transform),
        bioclip_transform_hash=transform_fingerprint(bundle.bioclip_eval_transform),
    )


def _split_labelled_filenames(
    filenames: list[str],
    labels: dict[str, str],
    seed: int,
    validation_fraction: float,
) -> tuple[list[str], list[str]]:
    """Deterministically keep at least one training image per class when possible."""
    import random

    grouped: dict[str, list[str]] = {}
    for filename in filenames:
        if filename in labels:
            grouped.setdefault(labels[filename], []).append(filename)
    training: list[str] = []
    validation: list[str] = []
    rng = random.Random(seed)
    for species in sorted(grouped):
        group = sorted(grouped[species])
        rng.shuffle(group)
        count = min(max(1, round(len(group) * validation_fraction)), max(0, len(group) - 1))
        validation.extend(group[:count])
        training.extend(group[count:])
    return sorted(training), sorted(validation)


def make_loader(
    filenames: list[str],
    config: dict[str, Any],
    bundle: RuntimeBundle,
    labels: dict[str, str] | None,
    species_names: list[str],
    *,
    training: bool,
    context: DistributedContext,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Construct one distributed-aware multi-view loader."""
    dataset = FishMultiViewDataset(
        filenames,
        data_path(config, "images_dir"),
        build_dino_transform(
            bundle.model.dino,
            training=training,
            conservative=bool(config["training"].get("conservative_augmentation", True)),
        ),
        bundle.bioclip_eval_transform,
        labels=labels,
        species_to_index={name: index for index, name in enumerate(species_names)},
        image_cache=load_runtime_image_cache(
            config,
            bundle,
            filenames,
            training=training,
        ),
    )
    sampler = DistributedSampler(dataset, shuffle=training) if context.world_size > 1 else None
    worker_key = "num_workers" if training else "eval_num_workers"
    workers = int(
        config["training"].get(
            worker_key,
            config["training"].get("num_workers", 4),
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"] if training else config["training"]["eval_batch_size"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=bool(config["training"].get("persistent_workers", True)) and workers > 0,
        pin_memory=context.device.type == "cuda",
        prefetch_factor=(
            int(config["training"].get("prefetch_factor", 2))
            if workers > 0
            else None
        ),
        collate_fn=collate_multiview,
        drop_last=training and len(dataset) >= int(config["training"]["batch_size"]),
    )
    return loader, sampler


@torch.no_grad()
def evaluate_loader(
    model: FishMultimodalModel,
    loader: DataLoader,
    prototypes: torch.Tensor,
    device: torch.device,
    *,
    species_names: list[str] | None = None,
    family_by_species: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate all available branches over one labelled candidate set."""
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        "dino_text": [],
        "bioclip_native": [],
        "supervised": [],
        "bioclip_supervised": [],
        "targets": [],
    }
    for batch in loader:
        targets = batch["species_index"]
        if targets is None:
            raise ValueError("Evaluation metrics require labels")
        output = model(
            batch["dino_image"].to(device),
            prototypes,
            batch["bioclip_image"].to(device),
        )
        collected["dino_text"].append(output.dino_text_logits.cpu())
        collected["targets"].append(targets)
        if output.bioclip_logits is not None:
            collected["bioclip_native"].append(output.bioclip_logits.cpu())
        if output.supervised_logits is not None and output.supervised_logits.shape[1] == prototypes.shape[0]:
            collected["supervised"].append(output.supervised_logits.cpu())
        if (
            output.bioclip_supervised_logits is not None
            and output.bioclip_supervised_logits.shape[1]
            == prototypes.shape[0]
        ):
            collected["bioclip_supervised"].append(
                output.bioclip_supervised_logits.cpu()
            )
    targets = torch.cat(collected["targets"])
    metrics = distributed_classification_metrics(
        torch.cat(collected["dino_text"]), targets, prefix="dino_text"
    )
    if species_names is not None:
        metrics.update(
            distributed_hierarchical_accuracy(
                torch.cat(collected["dino_text"]),
                targets,
                species_names,
                family_by_species=family_by_species,
                prefix="dino_text",
            )
        )
    if collected["bioclip_native"]:
        dino_logits = torch.cat(collected["dino_text"])
        bioclip_logits = torch.cat(collected["bioclip_native"])
        metrics.update(
            distributed_classification_metrics(bioclip_logits, targets, prefix="bioclip_native")
        )
        if species_names is not None:
            metrics.update(
                distributed_hierarchical_accuracy(
                    bioclip_logits,
                    targets,
                    species_names,
                    family_by_species=family_by_species,
                    prefix="bioclip_native",
                )
            )
        text_prob = fuse_text_probabilities(
            dino_logits, bioclip_logits, CalibrationParameters()
        )
        metrics.update(distributed_classification_metrics(text_prob, targets, prefix="fused_text"))
        if species_names is not None:
            metrics.update(
                distributed_hierarchical_accuracy(
                    text_prob,
                    targets,
                    species_names,
                    family_by_species=family_by_species,
                    prefix="fused_text",
                )
            )
    if collected["supervised"]:
        supervised_logits = torch.cat(collected["supervised"])
        metrics.update(
            distributed_classification_metrics(supervised_logits, targets, prefix="supervised")
        )
        if species_names is not None:
            metrics.update(
                distributed_hierarchical_accuracy(
                    supervised_logits,
                    targets,
                    species_names,
                    family_by_species=family_by_species,
                    prefix="supervised",
                )
            )
        if collected["bioclip_native"]:
            seen_prob = fuse_seen_probabilities(
                supervised_logits,
                fuse_text_probabilities(
                    torch.cat(collected["dino_text"]),
                    torch.cat(collected["bioclip_native"]),
                    CalibrationParameters(),
                ),
                CalibrationParameters(),
            )
            metrics.update(
                distributed_classification_metrics(
                    seen_prob, targets, prefix="supervised_plus_text"
                )
            )
            if species_names is not None:
                metrics.update(
                    distributed_hierarchical_accuracy(
                        seen_prob,
                        targets,
                        species_names,
                        family_by_species=family_by_species,
                        prefix="supervised_plus_text",
                    )
                )
    if collected["bioclip_supervised"]:
        classifier_logits = torch.cat(collected["bioclip_supervised"])
        metrics.update(
            distributed_classification_metrics(
                classifier_logits,
                targets,
                prefix="bioclip_supervised",
            )
        )
        if species_names is not None:
            metrics.update(
                distributed_hierarchical_accuracy(
                    classifier_logits,
                    targets,
                    species_names,
                    family_by_species=family_by_species,
                    prefix="bioclip_supervised",
                )
            )
        if collected["bioclip_native"]:
            native_probabilities = torch.softmax(
                torch.cat(collected["bioclip_native"]).float(),
                dim=-1,
            )
            fused_probabilities = fuse_seen_probabilities(
                classifier_logits,
                native_probabilities,
                CalibrationParameters(),
            )
            metrics.update(
                distributed_classification_metrics(
                    fused_probabilities,
                    targets,
                    prefix="bioclip_supervised_plus_text",
                )
            )
            if species_names is not None:
                metrics.update(
                    distributed_hierarchical_accuracy(
                        fused_probabilities,
                        targets,
                        species_names,
                        family_by_species=family_by_species,
                        prefix="bioclip_supervised_plus_text",
                    )
                )
    return metrics


def train_from_config(config: dict[str, Any]) -> dict[str, float]:
    """Run one staged training configuration under single process or torchrun."""
    distributed_cfg = config["training"].get("distributed", {})
    context = initialise_distributed(
        bool(distributed_cfg.get("enabled", False)),
        str(distributed_cfg.get("backend", "nccl")),
    )
    seed_everything(int(config["seed"]) + context.rank)
    if context.device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    bundle = build_runtime(config, device=context.device)
    configure_training_stage(
        bundle.model,
        config["training"]["stage"],
        model_config=config["model"],
    )
    labels = load_labels(config)
    training_species = bundle.partitions.seen_species
    split_hash = None
    pseudo_path = _pseudo_split_path(config)
    if pseudo_path:
        if not pseudo_path.exists():
            raise FileNotFoundError(
                f"Pseudo-unseen split not found: {pseudo_path}. Run make-pseudo-unseen first."
            )
        pseudo = read_json(pseudo_path)
        training_species = list(pseudo["training_species"])
        split_hash = pseudo["split_hash"]
    seen_prototypes, all_seen_names, prototype_cache = load_candidate_prototypes(
        config, bundle, "seen", device=context.device
    )
    from fish_vlm.utils.hashing import prompts_hash

    canonical_prompts = read_json(_data_processed_path(config, "canonical_prompts.json"))
    canonical_prompt_hash = prompts_hash(canonical_prompts, bundle.partitions.all_species)
    indices = torch.tensor([all_seen_names.index(name) for name in training_species], device=context.device)
    prototypes = seen_prototypes.index_select(0, indices)
    prototype_similarity = (
        prototypes.float() @ prototypes.float().T
    ).detach()
    official_train = split_filenames(data_path(config, "train_split"))
    eligible = [name for name in official_train if name in labels and labels[name] in set(training_species)]
    train_names, validation_names = _split_labelled_filenames(
        eligible,
        labels,
        int(config["seed"]),
        float(config["validation"].get("seen_fraction", 0.1)),
    )
    train_loader, train_sampler = make_loader(
        train_names, config, bundle, labels, training_species, training=True, context=context
    )
    validation_loader, _ = make_loader(
        validation_names, config, bundle, labels, training_species, training=False, context=context
    )
    pseudo_validation_loader = None
    pseudo_prototypes = None
    if pseudo_path:
        evaluation_species = list(pseudo["evaluation_species"])
        pseudo_names = [
            name for name in official_train if name in labels and labels[name] in set(evaluation_species)
        ]
        pseudo_validation_loader, _ = make_loader(
            pseudo_names,
            config,
            bundle,
            labels,
            evaluation_species,
            training=False,
            context=context,
        )
        pseudo_indices = torch.tensor(
            [all_seen_names.index(name) for name in evaluation_species],
            device=context.device,
        )
        pseudo_prototypes = seen_prototypes.index_select(0, pseudo_indices)
    teacher_cache = None
    teacher_cfg = config["loss"].get("bioclip_image_teacher", {})
    bioclip_distillation_cfg = config["loss"].get(
        "bioclip_pretrained_distillation", {}
    )
    hard_negative_sections = [
        config["loss"].get("dino_text_classification", {}),
        config["loss"].get("native_bioclip_text", {}),
    ]
    needs_visual_similarity = any(
        section.get("hard_negatives", {}).get("enabled", False)
        and section.get("hard_negatives", {}).get("strategy")
        == "visually_similar"
        for section in hard_negative_sections
    )
    needs_cached_teacher = (
        teacher_cfg.get("enabled", False)
        and teacher_cfg.get("mode") == "cached"
    ) or bioclip_distillation_cfg.get(
        "enabled", False
    ) or needs_visual_similarity
    if needs_cached_teacher:
        full_names = [name for name in official_train if name in labels]
        teacher_cache = load_image_teacher_cache(
            _cache_path(config, "bioclip_images", "train_embeddings.pt"),
            expected_filenames=full_names,
            checkpoint=bundle.bioclip_checkpoint,
            transform_hash=transform_fingerprint(bundle.bioclip_eval_transform),
        )
    use_bioclip_during_training = (
        bool(config["loss"].get("native_bioclip_text", {}).get("enabled", False))
        or bool(
            config["loss"].get("bioclip_supervised_species", {}).get(
                "enabled", False
            )
        )
        or bool(bioclip_distillation_cfg.get("enabled", False))
        or bool(config["loss"].get("branch_consistency", {}).get("enabled", False))
        or (
            bool(teacher_cfg.get("enabled", False))
            and teacher_cfg.get("mode") == "online"
        )
    )
    resume_path = config["training"].get("resume_checkpoint")
    if resume_path:
        load_checkpoint(
            resume_path,
            bundle.model,
            expected_seen_species=bundle.partitions.seen_species,
            expected_unseen_species=bundle.partitions.unseen_species,
            expected_text_prototype_hash=prototype_cache["prompt_hash"],
            expected_canonical_prompt_hash=canonical_prompt_hash,
            expected_training_species_hash=ordered_names_hash(training_species),
            expected_dino_model_name=str(config["model"]["dino"]["name"]),
            expected_dino_checkpoint_source=bundle.dino_source,
            expected_bioclip_checkpoint=bundle.bioclip_checkpoint,
            strict=False,
        )
    optimizer = build_optimizer(bundle.model, config)
    representation_cfg = config["loss"].get(
        "representation_distillation", {}
    )
    representation_teacher = (
        DinoProjectionTeacher.from_checkpoint(
            bundle.model,
            representation_cfg.get(
                "checkpoint",
                config["training"].get("resume_checkpoint", ""),
            ),
            expected_identity={
                "seen_species": bundle.partitions.seen_species,
                "unseen_species": bundle.partitions.unseen_species,
                "text_prototype_hash": prototype_cache["prompt_hash"],
                "canonical_prompt_hash": canonical_prompt_hash,
                "training_species_hash": ordered_names_hash(
                    training_species
                ),
                "dino_model_name": str(
                    config["model"]["dino"]["name"]
                ),
                "dino_checkpoint_source": bundle.dino_source,
                "bioclip_checkpoint": bundle.bioclip_checkpoint,
            },
        ).to(context.device)
        if representation_cfg.get("enabled", False)
        else None
    )
    from fish_vlm.data.taxonomy import genus_for_species, load_family_mapping

    genus_names = sorted(
        {genus_for_species(name) for name in training_species}
    )
    genus_to_index = {
        name: index for index, name in enumerate(genus_names)
    }
    genus_class_indices = [
        genus_to_index[genus_for_species(name)]
        for name in training_species
    ]
    family_mapping = load_family_mapping(config, training_species)
    family_class_indices = None
    if len(family_mapping) == len(training_species):
        family_names = sorted(set(family_mapping.values()))
        family_to_index = {
            name: index for index, name in enumerate(family_names)
        }
        family_class_indices = [
            family_to_index[family_mapping[name]]
            for name in training_species
        ]
    hard_negative_context: dict[str, object] = {
        "genus_groups": genus_class_indices,
        "family_groups": family_class_indices,
        "prototype_similarity": prototype_similarity,
        "visual_similarity": None,
    }
    if needs_visual_similarity:
        assert teacher_cache is not None
        centroids: list[torch.Tensor] = []
        for species in training_species:
            species_filenames = [
                filename
                for filename in teacher_cache["filenames"]
                if labels[filename] == species
            ]
            if not species_filenames:
                raise ValueError(
                    f"No teacher embeddings for visual negatives: {species}"
                )
            centroids.append(
                lookup_teacher_embeddings(
                    teacher_cache, species_filenames
                ).mean(dim=0)
            )
        visual_centroids = torch.nn.functional.normalize(
            torch.stack(centroids).float(), dim=-1
        ).to(context.device)
        hard_negative_context["visual_similarity"] = (
            visual_centroids @ visual_centroids.T
        )
    max_steps = int(config["training"]["max_steps"])
    validation_interval = int(config["training"]["validation_interval_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps
    )
    amp_dtype = torch.bfloat16 if config["training"].get("amp_dtype") == "bfloat16" else torch.float16
    use_amp = bool(config["training"].get("use_amp", True)) and context.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    model: torch.nn.Module = bundle.model
    if context.world_size > 1:
        ddp_config = config["training"].get("distributed", {})
        model = DistributedDataParallel(
            bundle.model,
            device_ids=[context.local_rank],
            broadcast_buffers=bool(ddp_config.get("broadcast_buffers", False)),
            gradient_as_bucket_view=bool(
                ddp_config.get("gradient_as_bucket_view", True)
            ),
            static_graph=bool(ddp_config.get("static_graph", True)),
        )
    early = EarlyStopping(
        int(config["training"].get("early_stopping_patience_evaluations", 10))
    )
    selection_metric = config["validation"]["selection_metric"]
    output_dir = Path(config.get("output_dir", "outputs"))
    checkpoint_name = str(config["training"].get("checkpoint_name", "best.pt"))
    metrics_name = str(
        config["training"].get("metrics_name", f"{Path(checkpoint_name).stem}.json")
    )
    wandb_logger = None
    if context.is_main and config.get("wandb", {}).get("enabled", False):
        wandb_logger = ScientificWandbLogger(
            config,
            trainable_parameters=trainable_parameter_count(bundle.model),
        )
    best_metrics: dict[str, float] = {}
    accumulation_steps = int(config["training"].get("gradient_accumulation_steps", 1))
    if accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be at least 1")
    global_step = 0
    data_pass = 0
    if train_sampler is not None:
        train_sampler.set_epoch(data_pass)
    train_iterator = iter(train_loader)
    running: dict[str, float] = {}
    samples = 0
    interval_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    print(f"Training for {max_steps} steps with validation every {validation_interval} steps")
    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
    try:
        while global_step < max_steps:
            model.train()
            if (
                bundle.model.bioclip is not None
                and config["model"].get("tuning_mode", "frozen")
                in {"frozen", "linear_probe", "adapter"}
            ):
                bundle.model.bioclip.eval()
            if config["model"]["dino"]["trainable_scope"] == "frozen":
                bundle.model.dino.eval()

            for micro_step in range(accumulation_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    data_pass += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(data_pass)
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                targets = batch["species_index"].to(context.device)
                update_parameters = micro_step + 1 == accumulation_steps
                sync_context = (
                    contextlib.nullcontext()
                    if update_parameters or not isinstance(model, DistributedDataParallel)
                    else model.no_sync()
                )
                with sync_context:
                    with torch.autocast(
                        device_type=context.device.type,
                        dtype=amp_dtype,
                        enabled=use_amp,
                    ):
                        output = model(
                            batch["dino_image"].to(context.device, non_blocking=True),
                            prototypes,
                            (
                                batch["bioclip_image"].to(
                                    context.device, non_blocking=True
                                )
                                if use_bioclip_during_training
                                else None
                            ),
                        )
                        representation_teacher_embeddings = (
                            None
                            if representation_teacher is None
                            else representation_teacher(
                                batch["dino_image"].to(
                                    context.device, non_blocking=True
                                )
                            )
                        )
                        teacher = None
                        if (
                            teacher_cfg.get("enabled", False)
                            or bioclip_distillation_cfg.get("enabled", False)
                        ):
                            if needs_cached_teacher:
                                teacher = lookup_teacher_embeddings(
                                    teacher_cache, batch["filename"]
                                ).to(context.device)
                            elif teacher_cfg.get("mode") == "online":
                                teacher = output.bioclip_features
                            else:
                                raise ValueError("Teacher mode must be cached or online")
                        result = compute_total_loss(
                            output,
                            targets,
                            config["loss"],
                            teacher_embeddings=teacher,
                            representation_teacher_embeddings=(
                                representation_teacher_embeddings
                            ),
                            genus_class_indices=genus_class_indices,
                            family_class_indices=family_class_indices,
                            hard_negative_context=hard_negative_context,
                        )
                        backward_loss = result.total / accumulation_steps
                    scaler.scale(backward_loss).backward()
                batch_size = len(targets)
                samples += batch_size
                running["loss"] = running.get("loss", 0.0) + result.total.item() * batch_size
                for name, value in result.components.items():
                    running[name] = running.get(name, 0.0) + value.item() * batch_size
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            should_validate = (
                global_step % validation_interval == 0 or global_step == max_steps
            )
            if not should_validate:
                continue
            if context.world_size > 1:
                names = sorted(running)
                reduced = reduce_sum(
                    torch.tensor(
                        [running[name] for name in names] + [float(samples)],
                        device=context.device,
                        dtype=torch.float64,
                    )
                ).cpu()
                running = {name: float(reduced[index]) for index, name in enumerate(names)}
                samples = int(reduced[-1])
            metrics = evaluate_loader(bundle.model, validation_loader, prototypes, context.device)
            branch = str(config["validation"].get("selection_branch", "fused_text"))
            seen_key = f"{branch}_accuracy"
            if seen_key not in metrics:
                seen_key = "dino_text_accuracy"
            metrics["seen_accuracy"] = metrics[seen_key]
            if pseudo_validation_loader is not None and pseudo_prototypes is not None:
                pseudo_metrics = evaluate_loader(
                    bundle.model, pseudo_validation_loader, pseudo_prototypes, context.device
                )
                metrics.update({f"pseudo_unseen_{key}": value for key, value in pseudo_metrics.items()})
                pseudo_branch = str(
                    config["validation"].get(
                        "unseen_selection_branch", branch
                    )
                )
                pseudo_key = f"{pseudo_branch}_accuracy"
                if pseudo_key not in pseudo_metrics:
                    pseudo_key = "dino_text_accuracy"
                metrics["pseudo_unseen_accuracy"] = pseudo_metrics[pseudo_key]
                from fish_vlm.data.catalog import official_split_counts
                from fish_vlm.evaluation.reports import add_selection_metrics

                test_count, unseen_count = official_split_counts(config)
                metrics = add_selection_metrics(
                    metrics,
                    seen_accuracy=metrics["seen_accuracy"],
                    pseudo_unseen_accuracy=metrics["pseudo_unseen_accuracy"],
                    test_count=test_count,
                    unseen_count=unseen_count,
                )
            elif selection_metric != "seen_accuracy":
                raise ValueError(
                    f"selection_metric={selection_metric!r} requires validation.pseudo_unseen.split_path"
                )
            selected = selection_value(metrics, selection_metric)
            improved, should_stop = early.update(selected)
            training_losses = {
                name: value / max(1, samples) for name, value in running.items()
            }
            learning_rates = {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            }
            throughput = samples / max(
                time.perf_counter() - interval_start, 1e-9
            )
            gpu_peak_memory = (
                torch.cuda.max_memory_allocated(context.device)
                if context.device.type == "cuda"
                else None
            )
            if context.is_main:
                LOGGER.info(
                    "step=%d selected=%.5f train_loss=%.5f",
                    global_step,
                    selected,
                    training_losses["loss"],
                )
                if wandb_logger is not None:
                    wandb_logger.log_step(
                        step=global_step,
                        training_losses=training_losses,
                        metrics=metrics,
                        learning_rates=learning_rates,
                        throughput=throughput,
                        gpu_peak_memory_bytes=gpu_peak_memory,
                        improved=improved,
                    )
                if improved:
                    metadata = {
                        "dino_model_name": config["model"]["dino"]["name"],
                        "dino_checkpoint_source": bundle.dino_source,
                        "bioclip_checkpoint": bundle.bioclip_checkpoint,
                        "text_prototype_hash": prototype_cache["prompt_hash"],
                        "canonical_prompt_hash": canonical_prompt_hash,
                        "seen_species": bundle.partitions.seen_species,
                        "unseen_species": bundle.partitions.unseen_species,
                        "training_species": training_species,
                        "training_species_hash": ordered_names_hash(training_species),
                        "pseudo_unseen_split_hash": split_hash,
                        "active_losses": [
                            name
                            for name, section in config["loss"].items()
                            if isinstance(section, dict)
                            and section.get("enabled", False)
                        ],
                    }
                    save_checkpoint(
                        output_dir / "checkpoints" / checkpoint_name,
                        model=bundle.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=global_step,
                        best_metric=selected,
                        config=config,
                        metadata=metadata,
                    )
                    best_metrics = {
                        **metrics,
                        "best_step": float(global_step),
                        "selection_value": float(selected),
                    }
                    write_json(
                        output_dir / "metrics" / metrics_name,
                        {
                            **best_metrics,
                            "selection_metric": selection_metric,
                            "checkpoint": str(output_dir / "checkpoints" / checkpoint_name),
                        },
                    )
                    print(f"Saved improved checkpoint at step {global_step} with {selection_metric}={selected:.5f}")
                    if wandb_logger is not None:
                        wandb_logger.record_best(step=global_step, metrics=metrics)
            if should_stop:
                break
            running = {}
            samples = 0
            interval_start = time.perf_counter()
            if context.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(context.device)
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()
        cleanup_distributed()
    return best_metrics
