from __future__ import annotations

from pathlib import Path

import pytest

from fish_vlm.config import ConfigError, deep_merge, load_config, validate_config


def test_deep_merge_and_base_config_load() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}
    config = load_config(Path("configs/train/projection_only.yaml"))
    assert config["training"]["stage"] == "projection_only"
    assert config["model"]["bioclip"]["checkpoint"] == "hf-hub:imageomics/bioclip-2"
    assert config["training"]["batch_size"] == 512
    assert config["training"]["max_steps"] == 4000
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert "epochs" not in config["training"]
    assert config["slurm"]["gpus"] == 4
    assert config["slurm"]["cpus"] == 128
    assert config["slurm"]["memory"] == "700G"
    assert config["slurm"]["partition"] == "gpu-h200"


def test_epoch_configuration_is_rejected() -> None:
    config = load_config("configs/base.yaml")
    config["training"]["epochs"] = 2
    with pytest.raises(ConfigError, match="no longer supported"):
        validate_config(config)


def test_priority_bioclip_configs_are_explicit_and_loadable() -> None:
    linear = load_config("configs/train/bioclip_linear_probe.yaml")
    partial = load_config("configs/train/bioclip_partial_finetune.yaml")
    full = load_config("configs/train/bioclip_full_finetune.yaml")
    assert linear["model"]["tuning_mode"] == "linear_probe"
    assert partial["model"]["backbone"] == "bioclip2"
    assert partial["model"]["tuning_mode"] == "partial_finetune"
    assert partial["model"]["unfreeze_last_blocks"] == 1
    assert partial["optimiser"]["backbone_lr"] == 1.0e-6
    assert full["model"]["tuning_mode"] == "full_finetune"


def test_hybrid_config_is_full_dino_with_scientific_name_fallback() -> None:
    config = load_config("configs/hybrid/dino_seen.yaml")
    assert config["training"]["stage"] == "dino_seen_classifier"
    assert config["training"]["max_steps"] == 16000
    assert config["model"]["dino"]["trainable_scope"] == "full"
    assert config["loss"]["supervised_species"]["enabled"]
    assert config["loss"]["supervised_species"]["label_smoothing"] == 0.1
    assert not config["loss"]["dino_text_classification"]["enabled"]
    weights = config["text"]["prototype_ensemble"]["weights"]
    assert weights["scientific_name"] == 1.0
    assert all(
        value == 0.0
        for name, value in weights.items()
        if name != "scientific_name"
    )
    assert config["validation"]["pseudo_unseen"]["split_seed"] == 42


def test_long_bioclip_config_preserves_text_and_zero_shot_alignment() -> None:
    config = load_config("configs/hybrid/bioclip_long.yaml")
    assert config["training"]["stage"] == "bioclip_full_finetune"
    assert config["training"]["max_steps"] == 20000
    assert config["model"]["tuning_mode"] == "full_finetune"
    assert config["model"]["bioclip"]["freeze_text_encoder"]
    assert not config["model"]["bioclip"]["freeze_image_encoder"]
    assert config["loss"]["native_bioclip_text"]["enabled"]
    assert config["loss"]["bioclip_pretrained_distillation"]["enabled"]
    assert config["loss"]["bioclip_supervised_species"]["label_smoothing"] == 0.1
    assert config["validation"]["unseen_selection_branch"] == "bioclip_native"
    assert config["validation"]["pseudo_unseen"]["split_seed"] == 42


def test_invalid_unseen_supervised_mode(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "data: {}\nmodel:\n  dino: {trainable_scope: frozen}\n"
        "  projector: {type: linear}\n  bioclip_image_path: {mode: disabled}\n"
        "training: {}\nloss: {}\ninference:\n"
        "  test: {candidate_set: seen}\n"
        "  unseen: {candidate_set: unseen, mode: supervised}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="cannot be used"):
        load_config(path)
