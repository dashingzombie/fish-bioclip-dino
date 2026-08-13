"""Leakage-audited manifests for all-image DINO and BioCLIP adaptation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from fish_vlm.config import data_path
from fish_vlm.data.catalog import load_labels, split_filenames
from fish_vlm.data.partitions import build_class_partitions
from fish_vlm.data.taxonomy import load_family_mapping
from fish_vlm.prototypes.conditions import build_prompt_conditions
from fish_vlm.utils.hashing import stable_json_hash
from fish_vlm.utils.io import read_json, read_pickle, write_json


def class_aware_validation_split(
    filenames: list[str],
    labels: dict[str, str],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    """Split approximately ``fraction`` per class; singletons remain in train."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation fraction must be between zero and one")
    grouped: dict[str, list[str]] = {}
    for filename in filenames:
        species = labels.get(filename)
        if species is not None:
            grouped.setdefault(species, []).append(filename)
    train: list[str] = []
    validation: list[str] = []
    audit: dict[str, dict[str, int]] = {}
    rng = random.Random(seed)
    for species in sorted(grouped):
        group = sorted(grouped[species])
        rng.shuffle(group)
        validation_count = (
            0
            if len(group) == 1
            else min(len(group) - 1, max(1, round(len(group) * fraction)))
        )
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
        audit[species] = {
            "total": len(group),
            "train": len(group) - validation_count,
            "validation": validation_count,
        }
    return sorted(train), sorted(validation), audit


def _split_union(config: dict[str, Any]) -> dict[str, list[str]]:
    splits = {
        name: split_filenames(data_path(config, f"{name}_split"))
        for name in ("train", "test", "unseen")
    }
    locations: dict[str, str] = {}
    for split, filenames in splits.items():
        for filename in filenames:
            previous = locations.setdefault(filename, split)
            if previous != split:
                raise ValueError(
                    f"Image {filename!r} occurs in both {previous} and {split}"
                )
    return splits


def _pseudo_training_species(config: dict[str, Any], seen: list[str]) -> list[str]:
    pseudo = config.get("validation", {}).get("pseudo_unseen", {})
    if not pseudo.get("enabled", False):
        return list(seen)
    value = pseudo.get("split_path")
    if value and value != "auto":
        path = Path(str(value))
    else:
        seed = int(pseudo.get("split_seed", config["seed"]))
        path = (
            Path(config["data"]["root_dir"])
            / config["data"].get("processed_dir", "processed")
            / "pseudo_unseen"
            / f"{pseudo['strategy']}_seed_{seed}.json"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Pseudo-unseen split not found: {path}; run make-pseudo-unseen first"
        )
    value = read_json(path)
    return list(value["training_species"])


def build_domain_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Write exact image/text roles without consulting official split labels."""
    splits = _split_union(config)
    source_labels = load_labels(config)
    # Only filenames from the organiser's training split may retain labels.
    labels = {
        filename: source_labels[filename]
        for filename in splits["train"]
        if filename in source_labels
    }
    all_classes = list(read_pickle(data_path(config, "all_classes_pickle")))
    partitions = build_class_partitions(labels, all_classes)
    fraction = float(config["domain"]["validation_fraction"])
    train_labelled, validation_labelled, split_audit = (
        class_aware_validation_split(
            splits["train"],
            labels,
            fraction=fraction,
            seed=int(config["seed"]),
        )
    )
    unlabelled_official_train = [
        name for name in splits["train"] if name not in labels
    ]
    images_without_text = [
        *unlabelled_official_train,
        *splits["test"],
        *splits["unseen"],
    ]
    adaptation_train = sorted(set(train_labelled + images_without_text))
    if set(validation_labelled) & set(adaptation_train):
        raise RuntimeError("Domain validation images leaked into adaptation train")

    processed = (
        Path(config["data"]["root_dir"])
        / config["data"].get("processed_dir", "processed")
    )
    canonical_path = processed / "canonical_prompts.json"
    if not canonical_path.is_file():
        raise FileNotFoundError(
            f"Canonical prompts not found: {canonical_path}; run prepare-prompts first"
        )
    canonical = read_json(canonical_path)
    descriptions = read_json(data_path(config, "descriptions_json"))
    families = load_family_mapping(config, partitions.all_species)
    conditions = build_prompt_conditions(
        partitions.all_species,
        descriptions,
        canonical,
        family_by_species=families,
    )
    paired_species = set(_pseudo_training_species(config, partitions.seen_species))
    prompt_inventory: dict[str, Any] = {}
    for species in partitions.all_species:
        all_images = sorted(name for name, label in labels.items() if label == species)
        paired_images = sorted(
            set(all_images) & set(train_labelled)
            if species in paired_species
            else set()
        )
        prompt_inventory[species] = {
            "image_count": len(all_images),
            "paired_training_images": paired_images,
            "text_only": not all_images,
            "prompts": {
                condition: conditions[condition][species]
                for condition in conditions
            },
        }

    manifest: dict[str, Any] = {
        "version": 1,
        "validation": {
            "fraction": fraction,
            "strategy": "per_species_stratified",
            "singleton_policy": "train_only",
            "per_species": split_audit,
        },
        "label_policy": {
            "source": str(data_path(config, "labels_json")),
            "allowed_split": "train",
            "official_test_labels_loaded": False,
            "official_unseen_labels_loaded": False,
        },
        "splits": splits,
        "labelled_train": train_labelled,
        "labelled_validation": validation_labelled,
        "adaptation_train": adaptation_train,
        "adaptation_validation": validation_labelled,
        "images_without_text": sorted(images_without_text),
        "paired_training_species": sorted(paired_species),
        "text_only_species": sorted(set(partitions.all_species) - set(labels.values())),
        "species": {
            "seen": partitions.seen_species,
            "unseen": partitions.unseen_species,
            "all": partitions.all_species,
        },
        "prompt_inventory": prompt_inventory,
    }
    manifest["manifest_hash"] = stable_json_hash(manifest)
    output = Path(str(config["domain"]["manifest_path"]))
    write_json(output, manifest)
    write_json(
        Path(str(config["domain"]["prompt_inventory_path"])),
        {
            "manifest_hash": manifest["manifest_hash"],
            "species": prompt_inventory,
        },
    )
    return manifest
