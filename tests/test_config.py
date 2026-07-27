from __future__ import annotations

from pathlib import Path

import pytest

from fish_vlm.config import ConfigError, deep_merge, load_config


def test_deep_merge_and_base_config_load() -> None:
    merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}})
    assert merged == {"a": {"b": 3, "c": 2}}
    config = load_config(Path("configs/train/projection_only.yaml"))
    assert config["training"]["stage"] == "projection_only"
    assert config["model"]["bioclip"]["checkpoint"] == "hf-hub:imageomics/bioclip"


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

