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
from fish_vlm.data.descriptions import prepare_canonical_prompts
from fish_vlm.data.partitions import create_and_save_partitions
from fish_vlm.data.pseudo_unseen import save_pseudo_unseen_splits
from fish_vlm.data.transforms import transform_fingerprint
from fish_vlm.evaluation.calibrate import calibrate_checkpoint
from fish_vlm.evaluation.evaluate import evaluate_bioclip_zero_shot, evaluate_checkpoint
from fish_vlm.inference.predict import predict_split
from fish_vlm.inference.submission import merge_predictions
from fish_vlm.inference.validation import validate_submission
from fish_vlm.models.bioclip import load_bioclip
from fish_vlm.prototypes.image_teacher import build_image_teacher_cache
from fish_vlm.prototypes.text import build_text_prototype_cache, load_prompts
from fish_vlm.slurm.launcher import launch_slurm
from fish_vlm.sweeps.pipeline import run_pipeline
from fish_vlm.training.train import (
    _cache_path,
    _data_processed_path,
    build_runtime,
    ensure_partitions,
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
    validate = _config_parser(commands, "validate-submission")
    validate.add_argument("--submission", required=True)
    sweep = _config_parser(commands, "sweep")
    sweep.add_argument("--dry-run", action="store_true")
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
    model, _, _, tokenizer, _ = load_bioclip(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for candidate_set in ("seen", "unseen", "all"):
        names = getattr(partitions, f"{candidate_set}_species")
        build_text_prototype_cache(
            prompts,
            names,
            model,
            tokenizer,
            checkpoint,
            _cache_path(config, "text", f"text_prototypes_{candidate_set}.pt"),
            batch_size=int(config["training"].get("eval_batch_size", 128)),
            device=device,
        )


def _build_teacher(config: dict[str, Any]) -> None:
    bundle = build_runtime(config, device="cpu")
    if bundle.model.bioclip is None:
        raise ValueError("Teacher cache requires the native BioCLIP image path")
    labels = load_labels(config)
    filenames = [name for name in split_filenames(data_path(config, "train_split")) if name in labels]
    dataset = FishMultiViewDataset(
        filenames,
        data_path(config, "images_dir"),
        bundle.dino_eval_transform,
        bundle.bioclip_eval_transform,
        labels,
        {name: index for index, name in enumerate(bundle.partitions.seen_species)},
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
        output_path=_cache_path(config, "bioclip_images", "train_embeddings.pt"),
        device=device,
        storage_dtype=torch.bfloat16 if config["training"].get("teacher_cache_dtype") == "bfloat16" else torch.float16,
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch one command and return a shell exit code."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    if args.command == "merge-submission":
        result = merge_predictions(args.test, args.unseen, args.output)
        print(json.dumps({"merged": len(result), "output": args.output}))
        return 0
    config = load_config(args.config)
    if args.command == "prepare-prompts":
        print(json.dumps({"prepared": len(_prepare(config))}))
    elif args.command == "build-text-prototypes":
        _build_text(config)
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
