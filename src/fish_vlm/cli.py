"""Command-line interface for the complete multimodal pipeline."""

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
from fish_vlm.data.image_cache import (
    build_deterministic_image_cache,
    load_deterministic_image_cache,
    validate_image_filenames,
)
from fish_vlm.data.descriptions import prepare_canonical_prompts
from fish_vlm.data.partitions import create_and_save_partitions
from fish_vlm.data.pseudo_unseen import save_pseudo_unseen_splits
from fish_vlm.data.transforms import transform_fingerprint
from fish_vlm.evaluation.calibrate import calibrate_checkpoint
from fish_vlm.evaluation.evaluate import evaluate_bioclip_zero_shot, evaluate_checkpoint
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
from fish_vlm.slurm.launcher import launch_slurm
from fish_vlm.sweeps.pipeline import run_pipeline
from fish_vlm.training.train import (
    _cache_path,
    _data_processed_path,
    build_runtime,
    ensure_partitions,
    load_runtime_image_cache,
    train_from_config,
)
from fish_vlm.utils.io import write_json
from fish_vlm.utils.logging import configure_logging
from fish_vlm.workflow import run_all, write_pipeline_summary


def _config_parser(subparsers: Any, name: str, *, checkpoint: bool = False) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name)
    parser.add_argument("--config", required=True)
    if checkpoint:
        parser.add_argument("--checkpoint")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Create all stable command contracts."""
    parser = argparse.ArgumentParser(prog="fish-vlm")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    _config_parser(commands, "prepare-prompts")
    _config_parser(commands, "build-text-prototypes")
    _config_parser(commands, "build-image-cache")
    _config_parser(commands, "build-teacher-cache")
    _config_parser(commands, "make-pseudo-unseen")
    _config_parser(commands, "train")
    evaluate = _config_parser(commands, "evaluate", checkpoint=True)
    evaluate.add_argument("--output")
    calibrate = _config_parser(commands, "calibrate", checkpoint=True)
    calibrate.add_argument("--output", default="outputs/metrics/calibration.json")
    infer = _config_parser(commands, "infer", checkpoint=True)
    infer.add_argument("--split", choices=("test", "unseen"))
    infer.add_argument("--output", required=True)
    infer.add_argument("--calibration")
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
    image_list.add_argument("--missing-image-cache-only", action="store_true")
    sweep = _config_parser(commands, "sweep")
    sweep.add_argument("--dry-run", action="store_true")
    joint_sweep = commands.add_parser("joint-sweep")
    joint_sweep.add_argument(
        "--phase",
        choices=("loss", "optimiser", "architecture", "training", "all"),
        default="loss",
    )
    joint_sweep.add_argument("--confirm-top", type=int)
    joint_sweep.add_argument("--submit", action="store_true")
    joint_sweep.add_argument("--dry-run", action="store_true")
    joint_sweep.add_argument("--resume", action="store_true")
    joint_sweep.add_argument("--max-concurrent", type=int, default=8)
    joint_sweep.add_argument(
        "--output-root",
        default="outputs/sweep_pipelines/joint_supervised_text",
    )
    slurm = _config_parser(commands, "slurm")
    slurm.add_argument("--dry-run", action="store_true")
    run_all_parser = _config_parser(commands, "run-all")
    run_all_parser.add_argument("--mode", choices=("local", "slurm"), required=True)
    run_all_parser.add_argument("--dry-run", action="store_true")
    run_all_parser.add_argument("--force", action="store_true")
    run_all_parser.add_argument("--gpus", type=int)
    _config_parser(commands, "pipeline-summary")
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
    prompts = load_prompts(_data_processed_path(config, "canonical_prompts.json"))
    checkpoint = config["model"]["bioclip"]["checkpoint"]
    cache_specs = [
        (
            candidate_set,
            getattr(partitions, f"{candidate_set}_species"),
            _cache_path(config, "text", f"text_prototypes_{candidate_set}.pt"),
        )
        for candidate_set in ("seen", "unseen", "all")
    ]
    missing_specs = []
    from fish_vlm.utils.hashing import prompts_hash

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
            missing_specs.append((candidate_set, names, path))
    if not missing_specs:
        return

    model, _, _, tokenizer, embedding_dim = load_bioclip(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for candidate_set, names, path in cache_specs:
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
                batch_size=int(config["training"].get("eval_batch_size", 128)),
                device=device,
            )


def _build_teacher(config: dict[str, Any]) -> None:
    labels = load_labels(config)
    filenames = [name for name in split_filenames(data_path(config, "train_split")) if name in labels]
    output_path = _cache_path(config, "bioclip_images", "train_embeddings.pt")
    checkpoint = config["model"]["bioclip"]["checkpoint"]
    if output_path.exists():
        try:
            load_image_teacher_cache(
                output_path,
                expected_filenames=filenames,
                checkpoint=checkpoint,
                transform_hash=None,
            )
        except ValueError as error:
            raise ValueError(
                f"Existing image-teacher cache at {output_path} is invalid: {error}"
            ) from error
        return

    bundle = build_runtime(config, device="cpu")
    if bundle.model.bioclip is None:
        raise ValueError("Teacher cache requires the native BioCLIP image path")
    dataset = FishMultiViewDataset(
        filenames,
        data_path(config, "images_dir"),
        bundle.dino_eval_transform,
        bundle.bioclip_eval_transform,
        labels,
        {name: index for index, name in enumerate(bundle.partitions.seen_species)},
        image_cache=load_runtime_image_cache(
            config,
            bundle,
            filenames,
            training=False,
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
        storage_dtype=torch.bfloat16 if config["training"].get("teacher_cache_dtype") == "bfloat16" else torch.float16,
    )


def _image_split_filenames(config: dict[str, Any]) -> dict[str, list[str]]:
    return {
        split: validate_image_filenames(
            split_filenames(data_path(config, f"{split}_split"))
        )
        for split in ("train", "test", "unseen")
    }


def _write_image_list(
    config: dict[str, Any],
    output_path: str | Path,
    *,
    missing_image_cache_only: bool = False,
) -> int:
    """Write the exact required image union as a NUL-delimited tar file list."""
    splits = _image_split_filenames(config)
    if missing_image_cache_only:
        cache_root = _cache_path(config, "image_transforms")
        required_splits: dict[str, list[str]] = {}
        for split, filenames in splits.items():
            path = cache_root / split
            if not path.exists():
                required_splits[split] = filenames
                continue
            load_deterministic_image_cache(
                path,
                expected_filenames=filenames,
                dino_model_name=str(config["model"]["dino"]["name"]),
                bioclip_checkpoint=str(config["model"]["bioclip"]["checkpoint"]),
            )
        splits = required_splits
    names = list(
        dict.fromkeys(
            name
            for filenames in splits.values()
            for name in filenames
        )
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"".join(name.encode("utf-8") + b"\0" for name in names))
    return len(names)


def _build_image_caches(config: dict[str, Any]) -> None:
    """Build missing deterministic split caches before training starts."""
    splits = _image_split_filenames(config)
    dino_name = str(config["model"]["dino"]["name"])
    bioclip_checkpoint = str(config["model"]["bioclip"]["checkpoint"])
    cache_root = _cache_path(config, "image_transforms")
    missing: list[str] = []
    for split, filenames in splits.items():
        path = cache_root / split
        if path.exists():
            try:
                load_deterministic_image_cache(
                    path,
                    expected_filenames=filenames,
                    dino_model_name=dino_name,
                    bioclip_checkpoint=bioclip_checkpoint,
                )
            except ValueError as error:
                raise ValueError(
                    f"Existing {split} image-transform cache is invalid: {error}"
                ) from error
        else:
            missing.append(split)
    if not missing:
        return

    bundle = build_runtime(config, device="cpu")
    dino_hash = transform_fingerprint(bundle.dino_eval_transform)
    bioclip_hash = transform_fingerprint(bundle.bioclip_eval_transform)
    cache_config = config["data"].get("deterministic_transform_cache", {})
    for split, filenames in splits.items():
        path = cache_root / split
        if path.exists():
            load_deterministic_image_cache(
                path,
                expected_filenames=filenames,
                dino_model_name=dino_name,
                bioclip_checkpoint=bioclip_checkpoint,
                dino_transform_hash=dino_hash,
                bioclip_transform_hash=bioclip_hash,
            )
            continue
        build_deterministic_image_cache(
            path=path,
            filenames=filenames,
            images_dir=data_path(config, "images_dir"),
            dino_transform=bundle.dino_eval_transform,
            bioclip_transform=bundle.bioclip_eval_transform,
            dino_model_name=dino_name,
            bioclip_checkpoint=bioclip_checkpoint,
            dino_transform_hash=dino_hash,
            bioclip_transform_hash=bioclip_hash,
            dtype=str(cache_config.get("dtype", "float16")),
            batch_size=int(cache_config.get("batch_size", 128)),
            num_workers=int(cache_config.get("num_workers", 16)),
        )


def main(argv: list[str] | None = None) -> int:
    """Dispatch one command and return a shell exit code."""
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
    if args.command == "joint-sweep":
        from fish_vlm.sweeps.joint import run_joint_sweeps

        result = run_joint_sweeps(
            phase=args.phase,
            confirm_top=args.confirm_top,
            submit=args.submit,
            dry_run=args.dry_run,
            max_concurrent=args.max_concurrent,
            resume=args.resume,
            output_root=args.output_root,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    config = load_config(args.config)
    if args.command == "list-images":
        print(
            json.dumps(
                {
                    "images": _write_image_list(
                        config,
                        args.output,
                        missing_image_cache_only=args.missing_image_cache_only,
                    )
                }
            )
        )
    elif args.command == "prepare-prompts":
        print(json.dumps({"prepared": len(_prepare(config))}))
    elif args.command == "build-text-prototypes":
        _build_text(config)
    elif args.command == "build-image-cache":
        _build_image_caches(config)
    elif args.command == "build-teacher-cache":
        _build_teacher(config)
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
    elif args.command == "train":
        print(json.dumps(train_from_config(config), sort_keys=True))
    elif args.command == "evaluate":
        metrics = evaluate_checkpoint(config, args.checkpoint) if args.checkpoint else evaluate_bioclip_zero_shot(config)
        if args.output:
            write_json(args.output, metrics)
        print(json.dumps(metrics, sort_keys=True))
    elif args.command == "calibrate":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for calibration")
        result = calibrate_checkpoint(config, args.checkpoint, args.output)
        print(json.dumps(result, sort_keys=True))
    elif args.command == "infer":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for inference")
        split = args.split or config.get("evaluation", {}).get("official_split")
        if split not in {"test", "unseen"}:
            raise ValueError("Set --split or evaluation.official_split to test/unseen")
        result = predict_split(
            config, args.checkpoint, args.output, split=split, calibration_path=args.calibration
        )
        print(json.dumps({"predictions": len(result), "output": args.output}))
    elif args.command == "validate-submission":
        print(json.dumps(validate_submission(args.submission, config), sort_keys=True))
    elif args.command == "sweep":
        print(json.dumps(run_pipeline(config, dry_run=args.dry_run), sort_keys=True))
    elif args.command == "slurm":
        print(launch_slurm(config, dry_run=args.dry_run))
    elif args.command == "run-all":
        print(
            json.dumps(
                run_all(
                    config,
                    mode=args.mode,
                    dry_run=args.dry_run,
                    force=args.force,
                    gpus=args.gpus,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "pipeline-summary":
        print(json.dumps(write_pipeline_summary(config), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
