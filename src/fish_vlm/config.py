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
    for split_name in ("test", "unseen"):
        candidate = config["inference"].get(split_name, {}).get("candidate_set")
        if candidate not in {"seen", "unseen", "all"}:
            raise ConfigError(f"inference.{split_name}.candidate_set is invalid")
    unseen_mode = config["inference"].get("unseen", {}).get("mode")
    if unseen_mode in {"supervised", "supervised_plus_text"}:
        raise ConfigError("The supervised head cannot be used for unseen inference")
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
