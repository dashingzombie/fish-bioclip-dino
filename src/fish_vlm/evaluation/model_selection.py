"""Purpose-specific selection across all completed model families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from fish_vlm.prototypes.conditions import PROMPT_CONDITIONS
from fish_vlm.training.metrics import harmonic_mean
from fish_vlm.utils.io import atomic_write_text, read_json


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _zero_shot_config(
    *,
    root: Path,
    output_dir: Path,
    condition: str,
    split: str,
) -> Path:
    weights = {
        name: 1.0 if name == condition else 0.0
        for name in PROMPT_CONDITIONS
    }
    config = {
        "defaults": [str(root / "configs/base.yaml")],
        "evaluation": {
            "split": "train",
            "candidate_set": "seen" if split == "test" else "unseen",
            "official_split": split,
        },
        "inference": {
            "training_free_native": True,
            "test": {"candidate_set": "seen", "mode": "bioclip_native"},
            "unseen": {
                "candidate_set": "unseen",
                "mode": "bioclip_native",
            },
        },
        "model": {
            "backbone": "bioclip2",
            "tuning_mode": "frozen",
            "supervised_head": {"enabled": False},
            "bioclip_classifier": {"enabled": False},
            "bioclip_image_path": {
                "enabled": True,
                "mode": "frozen_zero_shot",
                "text_space": "native",
                "adapter": {"enabled": False},
            },
        },
        "text": {
            "prototype_ensemble": {
                "enabled": True,
                "weights": weights,
            }
        },
    }
    path = output_dir / f"zero_shot_{condition}_{split}.yaml"
    atomic_write_text(path, yaml.safe_dump(config, sort_keys=False))
    return path


def select_model_checkpoints(
    pipeline_config: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Select seen, unseen, and joint models without sharing one objective."""
    config_path = Path(pipeline_config["_config_path"])
    root = config_path.parent.parent
    workflow = pipeline_config["workflow"]
    dino_report = read_json(
        _resolve(root, workflow["stage_comparison_output"])
    )
    models: dict[str, dict[str, Any]] = {}
    dino_stage_names = {
        "projection_only",
        "final_block",
        "joint_supervised_text",
    }
    for stage in workflow["training_stages"]:
        name = str(stage["name"])
        checkpoint = str(_resolve(root, stage["checkpoint"]))
        seen_config = str(
            _resolve(root, stage["seen_inference_config"])
        )
        unseen_config = str(
            _resolve(
                root,
                stage.get(
                    "unseen_inference_config",
                    workflow["unseen_inference_config"],
                ),
            )
        )
        if name in dino_stage_names:
            metrics = dino_report["stages"][name]
            seen_accuracy = float(metrics["seen_accuracy"])
            pseudo_accuracy = float(metrics["pseudo_unseen_accuracy"])
            joint_score = float(metrics["harmonic_mean"])
        else:
            metrics = read_json(_resolve(root, stage["metrics"]))
            seen_accuracy = float(metrics["seen_accuracy"])
            pseudo_accuracy = float(metrics["pseudo_unseen_accuracy"])
            joint_score = float(
                metrics.get(
                    "seen_unseen_harmonic_mean",
                    harmonic_mean(seen_accuracy, pseudo_accuracy),
                )
            )
        models[name] = {
            "checkpoint": checkpoint,
            "seen_accuracy": seen_accuracy,
            "pseudo_unseen_accuracy": pseudo_accuracy,
            "harmonic_mean": joint_score,
            "seen_inference_config": seen_config,
            "unseen_inference_config": unseen_config,
        }

    zero_shot = read_json(_resolve(root, workflow["zero_shot_output"]))
    linear_checkpoint = next(
        str(_resolve(root, stage["checkpoint"]))
        for stage in workflow["training_stages"]
        if stage["name"] == "bioclip_linear_probe"
    )
    generated_dir = Path(output_path).resolve().parent / "selected_configs"
    for condition in PROMPT_CONDITIONS:
        seen_accuracy = float(zero_shot[f"{condition}_seen_accuracy"])
        pseudo_accuracy = float(
            zero_shot[f"{condition}_pseudo_unseen_accuracy"]
        )
        models[f"bioclip_zero_shot_{condition}"] = {
            # The linear-probe checkpoint is only an identity carrier here:
            # its native BioCLIP encoder was frozen and the generated configs
            # construct neither classifier nor adapter.
            "checkpoint": linear_checkpoint,
            "checkpoint_role": "frozen_native_bioclip_identity",
            "seen_accuracy": seen_accuracy,
            "pseudo_unseen_accuracy": pseudo_accuracy,
            "harmonic_mean": harmonic_mean(
                seen_accuracy, pseudo_accuracy
            ),
            "seen_inference_config": str(
                _zero_shot_config(
                    root=root,
                    output_dir=generated_dir,
                    condition=condition,
                    split="test",
                )
            ),
            "unseen_inference_config": str(
                _zero_shot_config(
                    root=root,
                    output_dir=generated_dir,
                    condition=condition,
                    split="unseen",
                )
            ),
        }

    def select(metric: str, config_key: str) -> dict[str, Any]:
        name = max(
            sorted(models),
            key=lambda candidate: float(models[candidate][metric]),
        )
        return {
            "model": name,
            "checkpoint": models[name]["checkpoint"],
            "metric": metric,
            "value": models[name][metric],
            "inference_config": models[name][config_key],
        }

    return {
        "models": models,
        "selection": {
            "seen": select("seen_accuracy", "seen_inference_config"),
            "unseen": select(
                "pseudo_unseen_accuracy",
                "unseen_inference_config",
            ),
            "joint": select(
                "harmonic_mean",
                "unseen_inference_config",
            ),
        },
    }
