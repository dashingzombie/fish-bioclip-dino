"""Focused command-line interface for the all-image training workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fish_vlm.config import data_path, load_config
from fish_vlm.data.catalog import load_labels, split_filenames
from fish_vlm.data.collate import collate_multiview
from fish_vlm.data.datasets import FishMultiViewDataset
from fish_vlm.data.descriptions import prepare_canonical_prompts
from fish_vlm.data.image_cache import validate_image_filenames
from fish_vlm.data.partitions import create_and_save_partitions
from fish_vlm.data.pseudo_unseen import save_pseudo_unseen_splits
from fish_vlm.data.transforms import transform_fingerprint
from fish_vlm.inference.predict import predict_split
from fish_vlm.inference.submission import merge_predictions, package_submission
from fish_vlm.inference.validation import validate_submission
from fish_vlm.models.bioclip import load_bioclip
from fish_vlm.prototypes.image_teacher import (
    build_image_teacher_cache,
    load_image_teacher_cache,
)
from fish_vlm.prototypes.text import (
    build_text_prototype_cache,
    load_prompts,
    load_text_prototype_cache,
)
from fish_vlm.training.train import (
    _cache_path,
    _data_processed_path,
    build_runtime,
    ensure_partitions,
    load_runtime_image_cache,
    train_from_config,
)
from fish_vlm.utils.logging import configure_logging


def _config_parser(
    subparsers: Any, name: str, *, checkpoint: bool = False
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.add_argument("--config", required=True)
    if checkpoint:
        parser.add_argument("--checkpoint", required=True)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fish-vlm")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "prepare-prompts",
        "make-pseudo-unseen",
        "prepare-domain-data",
        "build-text-prototypes",
        "build-all-teacher-cache",
        "train",
    ):
        _config_parser(commands, name)
    dino_domain = _config_parser(commands, "train-dino-domain")
    dino_domain.add_argument("--resume", action="store_true")
    bioclip_domain = _config_parser(commands, "train-bioclip-domain")
    bioclip_domain.add_argument("--resume", action="store_true")
    infer = _config_parser(commands, "infer", checkpoint=True)
    infer.add_argument("--split", choices=("test", "unseen"), required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--bioclip-checkpoint")
    merge = commands.add_parser("merge-submission")
    merge.add_argument("--test", required=True)
    merge.add_argument("--unseen", required=True)
    merge.add_argument("--output", required=True)
    package = commands.add_parser("package-submission")
    package.add_argument("--submission", required=True)
    package.add_argument("--output", required=True)
    validate = _config_parser(commands, "validate-submission")
    validate.add_argument("--submission", required=True)
    image_list = _config_parser(commands, "list-images")
    image_list.add_argument("--output", required=True)
    return parser


def _prepare(config: dict[str, Any]) -> dict[str, str]:
    partitions = create_and_save_partitions(
        data_path(config, "labels_json"),
        data_path(config, "all_classes_pickle"),
        _data_processed_path(config, "class_partitions.json"),
    )
    return prepare_canonical_prompts(
        data_path(config, "descriptions_json"),
        data_path(config, "manual_overrides"),
        _data_processed_path(config, "canonical_prompts.json"),
        _data_processed_path(config, "prompt_audit.jsonl"),
        expected_species=partitions.all_species,
        max_tokens=int(config["text"].get("max_tokens", 220)),
    )


def _build_text(config: dict[str, Any]) -> None:
    partitions = ensure_partitions(config)
    prompts = load_prompts(
        _data_processed_path(config, "canonical_prompts.json")
    )
    checkpoint = config["model"]["bioclip"]["checkpoint"]
    cache_specs = [
        (
            candidate_set,
            getattr(partitions, f"{candidate_set}_species"),
            _cache_path(
                config, "text", f"text_prototypes_{candidate_set}.pt"
            ),
        )
        for candidate_set in ("seen", "unseen", "all")
    ]
    from fish_vlm.utils.hashing import prompts_hash

    missing = []
    for candidate_set, names, path in cache_specs:
        if path.exists():
            try:
                load_text_prototype_cache(
                    path,
                    species_names=names,
                    checkpoint=checkpoint,
                    prompt_hash=prompts_hash(prompts, names),
                )
            except ValueError as error:
                raise ValueError(
                    f"Existing {candidate_set} text cache at {path} is invalid: {error}"
                ) from error
        else:
            missing.append((candidate_set, names, path))
    if not missing:
        return
    model, _, _, tokenizer, embedding_dim = load_bioclip(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _, names, path in cache_specs:
        if path.exists():
            load_text_prototype_cache(
                path,
                species_names=names,
                checkpoint=checkpoint,
                prompt_hash=prompts_hash(prompts, names),
                embedding_dim=embedding_dim,
            )
        else:
            build_text_prototype_cache(
                prompts,
                names,
                model,
                tokenizer,
                checkpoint,
                path,
                batch_size=int(config["training"]["eval_batch_size"]),
                device=device,
            )


def _all_image_filenames(config: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            name
            for split in ("train", "test", "unseen")
            for name in validate_image_filenames(
                split_filenames(data_path(config, f"{split}_split"))
            )
        )
    )


def _write_image_list(config: dict[str, Any], output_path: str | Path) -> int:
    names = _all_image_filenames(config)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        b"".join(name.encode("utf-8") + b"\0" for name in names)
    )
    return len(names)


def _build_all_teacher(config: dict[str, Any]) -> None:
    filenames = _all_image_filenames(config)
    output_path = _cache_path(
        config, "bioclip_images", "all_embeddings.pt"
    )
    checkpoint = config["model"]["bioclip"]["checkpoint"]
    if output_path.exists():
        load_image_teacher_cache(
            output_path,
            expected_filenames=filenames,
            checkpoint=checkpoint,
            transform_hash=None,
        )
        return
    bundle = build_runtime(config, device="cpu")
    if bundle.model.bioclip is None:
        raise ValueError("All-image teacher cache requires BioCLIP")
    dataset = FishMultiViewDataset(
        filenames,
        data_path(config, "images_dir"),
        bundle.dino_eval_transform,
        bundle.bioclip_eval_transform,
        labels=None,
        species_to_index=None,
        image_cache=load_runtime_image_cache(
            config, bundle, filenames, training=False
        ),
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 4)),
        collate_fn=collate_multiview,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    build_image_teacher_cache(
        bundle.model.bioclip,
        loader,
        checkpoint=bundle.bioclip_checkpoint,
        transform_hash=transform_fingerprint(bundle.bioclip_eval_transform),
        output_path=output_path,
        device=device,
        storage_dtype=(
            torch.bfloat16
            if config["training"].get("teacher_cache_dtype") == "bfloat16"
            else torch.float16
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.command == "merge-submission":
        result = merge_predictions(args.test, args.unseen, args.output)
        print(json.dumps({"merged": len(result), "output": args.output}))
        return 0
    if args.command == "package-submission":
        output = package_submission(args.submission, args.output)
        print(json.dumps({"output": str(output)}))
        return 0
    config = load_config(args.config)
    if args.command == "list-images":
        print(json.dumps({"images": _write_image_list(config, args.output)}))
    elif args.command == "prepare-prompts":
        print(json.dumps({"prepared": len(_prepare(config))}))
    elif args.command == "make-pseudo-unseen":
        partitions = ensure_partitions(config)
        pseudo = config["validation"]["pseudo_unseen"]
        splits = save_pseudo_unseen_splits(
            partitions.seen_species,
            _data_processed_path(config, "pseudo_unseen"),
            strategy=pseudo["strategy"],
            holdout_fraction=float(pseudo["holdout_fraction"]),
            seeds=[int(seed) for seed in pseudo["seeds"]],
        )
        print(json.dumps({"splits": [split.to_dict() for split in splits]}))
    elif args.command == "prepare-domain-data":
        from fish_vlm.domain.data import build_domain_manifest

        manifest = build_domain_manifest(config)
        print(json.dumps({"manifest": config["domain"]["manifest_path"], "hash": manifest["manifest_hash"]}))
    elif args.command == "build-text-prototypes":
        _build_text(config)
    elif args.command == "build-all-teacher-cache":
        _build_all_teacher(config)
    elif args.command == "train-dino-domain":
        from fish_vlm.domain.dino import train_dino_domain

        print(json.dumps(train_dino_domain(config, resume=args.resume), sort_keys=True))
    elif args.command == "train-bioclip-domain":
        from fish_vlm.domain.bioclip import train_bioclip_domain

        print(json.dumps(train_bioclip_domain(config, resume=args.resume), sort_keys=True))
    elif args.command == "train":
        print(json.dumps(train_from_config(config), sort_keys=True))
    elif args.command == "infer":
        result = predict_split(
            config,
            args.checkpoint,
            args.output,
            split=args.split,
            bioclip_checkpoint_path=args.bioclip_checkpoint,
        )
        print(json.dumps({"predictions": len(result), "output": args.output}))
    elif args.command == "validate-submission":
        print(json.dumps(validate_submission(config, args.submission), sort_keys=True))
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
