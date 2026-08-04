"""YAML configuration composition and validation."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration is incomplete or inconsistent."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; non-mappings replace their base values."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"Environment variable {name!r} is not set")
        return _ENV.sub(replace, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return loaded


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML with optional recursive ``defaults`` composition."""
    config_path = Path(path).resolve()
    raw = _read_yaml(config_path)
    defaults = raw.pop("defaults", [])
    if isinstance(defaults, (str, Path)):
        defaults = [defaults]
    if not isinstance(defaults, list):
        raise ConfigError("'defaults' must be a path or a list of paths")
    merged: dict[str, Any] = {}
    for default in defaults:
        default_path = Path(str(default))
        if not default_path.is_absolute():
            default_path = (config_path.parent / default_path).resolve()
        merged = deep_merge(merged, load_config(default_path))
    resolved = _expand_env(deep_merge(merged, raw))
    validate_config(resolved)
    resolved["_config_path"] = str(config_path)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    """Validate cross-field invariants needed by every executable path."""
    required = ("data", "model", "training", "loss", "inference")
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing required configuration sections: {missing}")
    model = config["model"]
    scope = model.get("dino", {}).get("trainable_scope")
    if scope not in {"frozen", "final_block", "full"}:
        raise ConfigError("model.dino.trainable_scope must be frozen, final_block, or full")
    projector_type = model.get("projector", {}).get("type")
    if projector_type not in {"linear", "mlp"}:
        raise ConfigError("model.projector.type must be linear or mlp")
    path_mode = model.get("bioclip_image_path", {}).get("mode", "disabled")
    if path_mode not in {"disabled", "frozen_zero_shot", "adapter"}:
        raise ConfigError("BioCLIP image path mode is invalid")
    tuning_mode = model.get("tuning_mode", "frozen")
    if tuning_mode not in {
        "frozen",
        "linear_probe",
        "adapter",
        "partial_finetune",
        "full_finetune",
    }:
        raise ConfigError("model.tuning_mode is invalid")
    if tuning_mode == "partial_finetune" and int(
        model.get("unfreeze_last_blocks", 0)
    ) < 1:
        raise ConfigError(
            "partial_finetune requires model.unfreeze_last_blocks >= 1"
        )
    bioclip_config = model.get("bioclip", {})
    if tuning_mode in {"partial_finetune", "full_finetune"}:
        if model.get("backbone") != "bioclip2":
            raise ConfigError(
                f"{tuning_mode} requires model.backbone=bioclip2"
            )
        if bioclip_config.get("freeze_image_encoder", True):
            raise ConfigError(
                f"{tuning_mode} requires "
                "model.bioclip.freeze_image_encoder=false"
            )
        if not bioclip_config.get("freeze_text_encoder", True):
            raise ConfigError(
                "BioCLIP text encoder must remain frozen"
            )
    text_space = model.get("bioclip_image_path", {}).get(
        "text_space", "native"
    )
    if text_space not in {"native", "adapter"}:
        raise ConfigError(
            "model.bioclip_image_path.text_space must be native or adapter"
        )
    for split_name in ("test", "unseen"):
        candidate = config["inference"].get(split_name, {}).get("candidate_set")
        if candidate not in {"seen", "unseen", "all"}:
            raise ConfigError(f"inference.{split_name}.candidate_set is invalid")
    unseen_mode = config["inference"].get("unseen", {}).get("mode")
    if unseen_mode in {
        "supervised",
        "supervised_plus_text",
        "bioclip_supervised",
        "bioclip_supervised_plus_text",
    }:
        raise ConfigError("The supervised head cannot be used for unseen inference")
    if config["inference"].get("training_free_native", False):
        modes = {
            config["inference"].get(split_name, {}).get("mode")
            for split_name in ("test", "unseen")
        }
        if modes != {"bioclip_native"} or tuning_mode != "frozen":
            raise ConfigError(
                "inference.training_free_native requires frozen "
                "bioclip_native test and unseen modes"
            )
    pseudo_unseen = config.get("validation", {}).get(
        "pseudo_unseen", {}
    )
    if pseudo_unseen.get("split_seed") is not None:
        split_seed = int(pseudo_unseen["split_seed"])
        available_seeds = {
            int(seed) for seed in pseudo_unseen.get("seeds", [])
        }
        if available_seeds and split_seed not in available_seeds:
            raise ConfigError(
                "validation.pseudo_unseen.split_seed must be one of "
                "validation.pseudo_unseen.seeds"
            )
    stage = config["training"].get("stage")
    required_tuning = {
        "bioclip_linear_probe": "linear_probe",
        "bioclip_adapter": "adapter",
        "bioclip_partial_finetune": "partial_finetune",
        "bioclip_full_finetune": "full_finetune",
    }
    if stage in required_tuning and tuning_mode != required_tuning[stage]:
        raise ConfigError(
            f"training.stage={stage} requires "
            f"model.tuning_mode={required_tuning[stage]}"
        )
    if stage == "dino_seen_classifier":
        if scope != "full":
            raise ConfigError(
                "dino_seen_classifier requires model.dino.trainable_scope=full"
            )
        if not model.get("supervised_head", {}).get("enabled", False):
            raise ConfigError(
                "dino_seen_classifier requires model.supervised_head.enabled=true"
            )
        if tuning_mode != "frozen":
            raise ConfigError(
                "dino_seen_classifier requires frozen BioCLIP"
            )
    enabled_hard_negatives: list[str] = []
    for loss_name in (
        "dino_text_classification",
        "native_bioclip_text",
    ):
        hard = config["loss"].get(loss_name, {}).get(
            "hard_negatives", {}
        )
        if not hard.get("enabled", False):
            continue
        strategy = hard.get("strategy")
        if strategy not in {
            "same_genus",
            "same_family",
            "text_similar",
            "visually_similar",
        }:
            raise ConfigError("Hard-negative strategy is invalid")
        enabled_hard_negatives.append(loss_name)
    if len(enabled_hard_negatives) > 1:
        raise ConfigError(
            "Enable hard-negative mining on only one loss per controlled ablation"
        )
    accumulation_steps = int(
        config["training"].get("gradient_accumulation_steps", 1)
    )
    if accumulation_steps < 1:
        raise ConfigError(
            "training.gradient_accumulation_steps must be at least 1"
        )
    if "epochs" in config["training"]:
        raise ConfigError("training.epochs is no longer supported; use max_steps")
    max_steps = int(config["training"].get("max_steps", 0))
    validation_interval = int(
        config["training"].get("validation_interval_steps", 0)
    )
    if max_steps < 1:
        raise ConfigError("training.max_steps must be at least 1")
    if validation_interval < 1 or validation_interval > max_steps:
        raise ConfigError(
            "training.validation_interval_steps must be between 1 and max_steps"
        )
    strategy = config["training"].get("distributed", {}).get("strategy", "ddp")
    if strategy != "ddp":
        raise ConfigError(
            "training.distributed.strategy must be ddp for the frozen-encoder pipeline"
        )


def data_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a data entry relative to ``data.root_dir``."""
    if key == "images_dir" and os.environ.get("FISH_VLM_IMAGES_DIR"):
        return Path(os.environ["FISH_VLM_IMAGES_DIR"]).expanduser()
    root = Path(config["data"]["root_dir"]).expanduser()
    return root / config["data"][key]
