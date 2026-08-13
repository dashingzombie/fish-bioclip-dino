"""Regression coverage for all-image missing-modality adaptation."""

from __future__ import annotations

from pathlib import Path
import json
import pickle

import torch
from PIL import Image
from torch import nn

from fish_vlm.domain.bioclip import prototype_hard_negative_loss
from fish_vlm.domain.data import build_domain_manifest, class_aware_validation_split
from fish_vlm.domain.datasets import DomainMultiViewDataset, collate_domain_views
from fish_vlm.domain.dino import dino_cross_view_loss
from fish_vlm.inference.bioclip_checkpoint import (
    EXPECTED_DOMAIN_ADAPTATION_LOSSES,
    load_finetuned_bioclip_visual,
)
from fish_vlm.utils.hashing import ordered_names_hash


def test_class_aware_ten_percent_keeps_singletons_in_training() -> None:
    filenames = ["singleton.jpg", *[f"many-{index}.jpg" for index in range(20)]]
    labels = {
        "singleton.jpg": "Only one",
        **{f"many-{index}.jpg": "Many fish" for index in range(20)},
    }
    train, validation, audit = class_aware_validation_split(
        filenames, labels, fraction=0.10, seed=42
    )
    assert "singleton.jpg" in train
    assert "singleton.jpg" not in validation
    assert audit["Only one"] == {"total": 1, "train": 1, "validation": 0}
    assert audit["Many fish"] == {"total": 20, "train": 18, "validation": 2}


def test_unlabelled_images_are_explicitly_masked_from_text_loss(
    tmp_path: Path,
) -> None:
    for name in ("paired.jpg", "unknown.jpg"):
        Image.new("RGB", (4, 4), color="white").save(tmp_path / name)

    def transform(_: Image.Image) -> list[torch.Tensor]:
        return [torch.zeros(3, 4, 4), torch.ones(3, 4, 4)]

    dataset = DomainMultiViewDataset(
        ["paired.jpg", "unknown.jpg"],
        tmp_path,
        transform,
        labels={"paired.jpg": "Known fish"},
        species_to_index={"Known fish": 3},
        paired_species={"Known fish"},
    )
    batch = collate_domain_views([dataset[0], dataset[1]])
    assert batch["has_text"].tolist() == [True, False]
    assert batch["target"].tolist() == [3, -1]


def test_manifest_ignores_labels_attached_to_official_test_and_unseen(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    processed = data / "processed"
    processed.mkdir(parents=True)
    (data / "labels.json").write_text(
        json.dumps(
            {
                "train.jpg": "Species alpha",
                "test.jpg": "Species beta",
                "unseen.jpg": "Species beta",
            }
        ),
        encoding="utf-8",
    )
    (data / "descriptions.json").write_text(
        json.dumps(
            {
                "Species alpha": "A silver fish with a dark stripe.",
                "Species beta": "A red fish with a rounded fin.",
            }
        ),
        encoding="utf-8",
    )
    (processed / "canonical_prompts.json").write_text(
        json.dumps(
            {
                "Species alpha": "A photograph of Species alpha.",
                "Species beta": "A photograph of Species beta.",
            }
        ),
        encoding="utf-8",
    )
    for name, value in {
        "train.pkl": ["train.jpg"],
        "test.pkl": ["test.jpg"],
        "unseen.pkl": ["unseen.jpg"],
        "classes.pkl": ["Species alpha", "Species beta"],
    }.items():
        with (data / name).open("wb") as handle:
            pickle.dump(value, handle)
    config = {
        "seed": 42,
        "data": {
            "root_dir": str(data),
            "labels_json": "labels.json",
            "descriptions_json": "descriptions.json",
            "all_classes_pickle": "classes.pkl",
            "train_split": "train.pkl",
            "test_split": "test.pkl",
            "unseen_split": "unseen.pkl",
            "processed_dir": "processed",
        },
        "domain": {
            "validation_fraction": 0.10,
            "manifest_path": str(tmp_path / "manifest.json"),
            "prompt_inventory_path": str(tmp_path / "inventory.json"),
        },
        "validation": {"pseudo_unseen": {"enabled": False}},
    }
    manifest = build_domain_manifest(config)
    assert manifest["label_policy"]["official_test_labels_loaded"] is False
    assert manifest["label_policy"]["official_unseen_labels_loaded"] is False
    assert {"test.jpg", "unseen.jpg"}.issubset(manifest["images_without_text"])
    assert manifest["prompt_inventory"]["Species beta"]["image_count"] == 0


def test_dino_self_distillation_uses_cross_views_without_class_targets() -> None:
    student = [torch.randn(3, 8, requires_grad=True) for _ in range(4)]
    teacher = [torch.randn(3, 8) for _ in range(2)]
    loss = dino_cross_view_loss(
        student,
        teacher,
        center=torch.zeros(1, 8),
        student_temperature=0.1,
        teacher_temperature=0.04,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(value.grad is not None for value in student)


def test_bioclip_hard_negative_loss_uses_only_prototype_neighbours() -> None:
    logits = torch.tensor([[4.0, 3.5, -2.0], [0.0, 1.0, 1.3]])
    targets = torch.tensor([0, 1])
    neighbours = torch.tensor([[1], [2], [1]])
    loss = prototype_hard_negative_loss(
        logits, targets, neighbours, margin=0.2
    )
    assert torch.allclose(loss, torch.tensor(0.25))


class _TinyBioClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Linear(2, 2, bias=False)
        self.text = nn.Linear(2, 2, bias=False)


def test_classifier_free_domain_checkpoint_composes_visual_tower(
    tmp_path: Path,
) -> None:
    model = _TinyBioClip()
    original_text = model.text.weight.detach().clone()
    state = {
        f"bioclip.{name}": value.detach().clone()
        for name, value in model.state_dict().items()
    }
    state["bioclip.visual.weight"] += 1.0
    paired_species = ["Seen one"]
    checkpoint = {
        "model_state": state,
        "dino_model_name": "unused",
        "dino_checkpoint_source": "unused",
        "bioclip_checkpoint": "bioclip-test",
        "text_prototype_hash": "text-hash",
        "canonical_prompt_hash": "prompt-hash",
        "seen_species": ["Seen one", "Seen two"],
        "unseen_species": ["Unseen one"],
        "training_species": paired_species,
        "training_species_hash": ordered_names_hash(paired_species),
        "active_losses": EXPECTED_DOMAIN_ADAPTATION_LOSSES,
        "bioclip_classifier_state": None,
        "resolved_configuration": {
            "training": {"stage": "bioclip_domain_adaptation"},
            "model": {
                "tuning_mode": "full_finetune",
                "bioclip": {
                    "freeze_image_encoder": False,
                    "freeze_text_encoder": True,
                },
            },
        },
    }
    path = tmp_path / "domain.pt"
    torch.save(checkpoint, path)
    load_finetuned_bioclip_visual(
        path,
        model,
        expected_seen_species=["Seen one", "Seen two"],
        expected_unseen_species=["Unseen one"],
        expected_training_species=["Seen one", "Seen two"],
        expected_text_prototype_hash="text-hash",
        expected_canonical_prompt_hash="prompt-hash",
        expected_bioclip_checkpoint="bioclip-test",
    )
    assert torch.equal(model.text.weight, original_text)
    assert torch.equal(
        model.visual.weight,
        checkpoint["model_state"]["bioclip.visual.weight"],
    )
