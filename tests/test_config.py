from __future__ import annotations

from pathlib import Path

import pytest

from fish_vlm.config import ConfigError, deep_merge, load_config, validate_config


def test_deep_merge_and_base_config_load() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}
    config = load_config(Path("configs/all_data/common.yaml"))
    assert config["training"]["stage"] == "projection_only"
    assert config["model"]["bioclip"]["checkpoint"] == "hf-hub:imageomics/bioclip-2"
    assert config["training"]["batch_size"] == 32
    assert config["training"]["max_steps"] == 4000
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert "epochs" not in config["training"]
    assert config["slurm"]["gpus"] == 1
    assert config["slurm"]["cpus"] == 16
    assert config["slurm"]["memory"] == "96G"
    assert config["slurm"]["partition"] == "gpu-h200"
    assert config["domain"]["cpu_jobs"]["partition"] == "normal"


def test_epoch_configuration_is_rejected() -> None:
    config = load_config("configs/base.yaml")
    config["training"]["epochs"] = 2
    with pytest.raises(ConfigError, match="no longer supported"):
        validate_config(config)


def test_dino_seen_config_is_full_dino_with_no_bioclip_classifier() -> None:
    config = load_config("configs/all_data/dino_finetune.yaml")
    assert config["training"]["stage"] == "dino_seen_classifier"
    assert config["training"]["max_steps"] == 200000
    assert config["model"]["dino"]["trainable_scope"] == "full"
    assert config["loss"]["supervised_species"]["enabled"]
    assert config["loss"]["supervised_species"]["label_smoothing"] == 0.1
    assert not config["loss"]["dino_text_classification"]["enabled"]
    assert not config["model"]["bioclip_classifier"]["enabled"]
    assert config["training"]["early_stopping_patience_evaluations"] == 30


def test_bioclip_domain_config_preserves_classifier_free_text_alignment() -> None:
    config = load_config("configs/all_data/bioclip_domain.yaml")
    assert config["model"]["tuning_mode"] == "full_finetune"
    assert config["model"]["bioclip"]["freeze_text_encoder"]
    assert not config["model"]["bioclip"]["freeze_image_encoder"]
    assert not config["model"]["bioclip_classifier"]["enabled"]
    assert config["domain"]["bioclip"]["early_stopping_patience_epochs"] == 30


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
