from __future__ import annotations

import torch
from torch import nn

from fish_vlm.data.taxonomy import genus_for_species
from fish_vlm.evaluation.model_selection import select_model_checkpoints
from fish_vlm.losses.hard_negatives import hard_negative_cross_entropy
from fish_vlm.losses.hierarchy import hierarchical_cross_entropy
from fish_vlm.losses.total import apply_missing_family_fallbacks
from fish_vlm.models.bioclip import configure_bioclip_tuning
from fish_vlm.models.fusion import (
    apply_seen_class_penalty,
    CalibrationParameters,
)
from fish_vlm.models.multimodal import FishMultimodalModel
from fish_vlm.models.projector import (
    LearnableLogitScale,
    LinearDinoToBioClipProjector,
)
from fish_vlm.prototypes.conditions import (
    PROMPT_CONDITIONS,
    build_prompt_conditions,
    ensemble_prototypes,
)
from fish_vlm.training.metrics import hierarchical_accuracy
from fish_vlm.training.optimizer import build_optimizer
from fish_vlm.training.stages import configure_training_stage
from fish_vlm.utils.io import write_json

from conftest import TinyBioClip, TinyDino


def test_five_prompt_conditions_and_weighted_ensemble() -> None:
    species = ["Salmo salar"]
    conditions = build_prompt_conditions(
        species,
        {
            "Salmo salar": (
                "Salmo salar in family Salmonidae has a silver body "
                "and dark spots."
            )
        },
        {"Salmo salar": "Full curated description."},
        family_by_species={"Salmo salar": "Salmonidae"},
    )
    assert tuple(conditions) == PROMPT_CONDITIONS
    assert "Salmonidae" in conditions["taxonomic_hierarchy"]["Salmo salar"]
    assert "Salmo salar" not in conditions["morphology_only"]["Salmo salar"]
    assert "Salmo" not in conditions["morphology_only"]["Salmo salar"]
    assert "Salmonidae" not in conditions["morphology_only"]["Salmo salar"]
    assert (
        conditions["full_description"]["Salmo salar"]
        == "Full curated description."
    )

    first = torch.tensor([[1.0, 0.0]])
    second = torch.tensor([[0.0, 1.0]])
    ensemble = ensemble_prototypes(
        {"morphology_only": first, "taxonomic_hierarchy": second},
        {"morphology_only": 3.0, "taxonomic_hierarchy": 1.0},
    )
    assert torch.allclose(ensemble.norm(dim=-1), torch.ones(1))
    assert ensemble[0, 0] > ensemble[0, 1]


def test_hierarchy_metrics_and_losses() -> None:
    logits = torch.tensor(
        [[3.0, 2.0, 0.0], [0.0, 1.0, 4.0]]
    )
    targets = torch.tensor([1, 2])
    species = ["GenusA one", "GenusA two", "GenusB one"]
    metrics = hierarchical_accuracy(
        logits,
        targets,
        species,
        family_by_species={
            "GenusA one": "FamilyA",
            "GenusA two": "FamilyA",
            "GenusB one": "FamilyB",
        },
    )
    assert metrics["genus_accuracy"] == 1.0
    assert metrics["family_accuracy"] == 1.0
    assert torch.isfinite(
        hierarchical_cross_entropy(logits, targets, [0, 0, 1])
    )
    assert genus_for_species("GenusA one") == "GenusA"


def test_hierarchy_loss_ignores_species_without_family_metadata() -> None:
    logits = torch.tensor(
        [[3.0, 2.0, 0.0], [0.0, 1.0, 4.0]],
        requires_grad=True,
    )
    targets = torch.tensor([1, 2])
    partial = hierarchical_cross_entropy(logits, targets, [0, 0, -1])
    known_only = hierarchical_cross_entropy(
        logits[:1], targets[:1], [0, 0, -1]
    )
    assert torch.allclose(partial, known_only)
    partial.backward()
    assert torch.isfinite(logits.grad).all()

    no_known_targets = hierarchical_cross_entropy(
        logits.detach()[1:], targets[1:], [0, 0, -1]
    )
    assert no_known_targets.item() == 0.0


def test_each_hard_negative_strategy_is_individually_runnable() -> None:
    logits = torch.tensor(
        [[5.0, 4.0, 1.0], [1.0, 5.0, 4.0]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1])
    similarity = torch.tensor(
        [[1.0, 0.9, 0.1], [0.9, 1.0, 0.8], [0.1, 0.8, 1.0]]
    )
    for strategy, groups in (
        ("same_genus", [0, 0, 1]),
        ("same_family", [0, 0, 1]),
        ("text_similar", None),
        ("visually_similar", None),
    ):
        loss = hard_negative_cross_entropy(
            logits,
            targets,
            strategy=strategy,
            top_k=1,
            class_groups=groups,
            prototype_similarity=similarity,
        )
        assert torch.isfinite(loss)


def test_same_family_negatives_fall_back_for_unknown_family() -> None:
    logits = torch.tensor([[5.0, 0.0, 4.0]], requires_grad=True)
    loss = hard_negative_cross_entropy(
        logits,
        torch.tensor([0]),
        strategy="same_family",
        top_k=1,
        class_groups=[-1, -1, 0],
    )
    expected = torch.nn.functional.cross_entropy(
        logits[:, [0, 2]], torch.tensor([0])
    )
    assert torch.allclose(loss, expected)


def test_missing_family_metadata_applies_runnable_fallbacks() -> None:
    losses = {
        "family_supervised": {"enabled": True, "weight": 0.05},
        "dino_text_classification": {
            "enabled": True,
            "weight": 1.0,
            "hard_negatives": {
                "enabled": True,
                "strategy": "same_family",
                "top_k": 5,
            },
        },
    }
    fallbacks = apply_missing_family_fallbacks(losses)
    assert not losses["family_supervised"]["enabled"]
    assert (
        losses["dino_text_classification"]["hard_negatives"]["strategy"]
        == "model_score"
    )
    assert len(fallbacks) == 2


def test_seen_penalty_and_native_bioclip_space() -> None:
    probabilities = torch.tensor([[0.6, 0.4]])
    penalised = apply_seen_class_penalty(probabilities, [0], gamma=1.0)
    assert penalised[0, 0] < probabilities[0, 0]
    assert torch.allclose(penalised.sum(-1), torch.ones(1))

    bioclip = TinyBioClip()
    adapter = nn.Sequential(nn.Linear(3, 3), nn.Softmax(dim=-1))
    model = FishMultimodalModel(
        TinyDino(),
        LinearDinoToBioClipProjector(4, 3),
        LearnableLogitScale(),
        bioclip=bioclip,
        bioclip_adapter=adapter,
        bioclip_text_space="native",
    )
    output = model(torch.randn(2, 4), torch.eye(3), torch.randn(2, 4))
    assert torch.allclose(
        output.bioclip_features,
        output.bioclip_original_features,
    )
    assert CalibrationParameters(calibration_gamma=0.5).calibration_gamma == 0.5


class _VisualTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)]
        )
        self.norm = nn.LayerNorm(2)


class _TunableBioClip(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = _VisualTower()
        self.text = nn.Linear(2, 2)


def test_bioclip_partial_tuning_unfreezes_only_last_visual_block() -> None:
    model = _TunableBioClip()
    configure_bioclip_tuning(
        model, "partial_finetune", unfreeze_last_blocks=1
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.visual.blocks[0].parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.visual.blocks[-1].parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.visual.norm.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.text.parameters()
    )


def test_global_selection_can_choose_training_free_unseen_model(
    tmp_path,
) -> None:
    root = tmp_path
    metrics_dir = root / "outputs/metrics"
    dino_stages = {}
    stages = []
    for index, name in enumerate(
        ("projection_only", "final_block", "joint_supervised_text")
    ):
        dino_stages[name] = {
            "seen_accuracy": 0.5 + index * 0.1,
            "pseudo_unseen_accuracy": 0.4,
            "harmonic_mean": 0.45,
        }
        stages.append(
            {
                "name": name,
                "checkpoint": f"outputs/checkpoints/{name}.pt",
                "metrics": f"outputs/metrics/{name}.json",
                "seen_inference_config": "configs/inference/seen.yaml",
            }
        )
    stages.append(
        {
            "name": "bioclip_linear_probe",
            "checkpoint": "outputs/checkpoints/linear.pt",
            "metrics": "outputs/metrics/linear.json",
            "seen_inference_config": "configs/inference/seen_bioclip.yaml",
            "unseen_inference_config": "configs/inference/unseen_bioclip.yaml",
        }
    )
    write_json(
        metrics_dir / "dino.json",
        {"stages": dino_stages},
    )
    write_json(
        metrics_dir / "linear.json",
        {
            "seen_accuracy": 0.8,
            "pseudo_unseen_accuracy": 0.6,
            "seen_unseen_harmonic_mean": 0.685,
        },
    )
    zero_shot = {}
    for condition in PROMPT_CONDITIONS:
        zero_shot[f"{condition}_seen_accuracy"] = 0.3
        zero_shot[f"{condition}_pseudo_unseen_accuracy"] = 0.5
    zero_shot["morphology_only_pseudo_unseen_accuracy"] = 0.9
    write_json(metrics_dir / "zero.json", zero_shot)
    pipeline = {
        "_config_path": str(root / "configs/pipeline.yaml"),
        "workflow": {
            "training_stages": stages,
            "unseen_inference_config": "configs/inference/unseen.yaml",
            "stage_comparison_output": "outputs/metrics/dino.json",
            "zero_shot_output": "outputs/metrics/zero.json",
        },
    }

    report = select_model_checkpoints(
        pipeline,
        metrics_dir / "selection.json",
    )

    assert (
        report["selection"]["unseen"]["model"]
        == "bioclip_zero_shot_morphology_only"
    )
    assert report["selection"]["seen"]["model"] == "bioclip_linear_probe"


def test_alignment_final_block_is_in_optimizer() -> None:
    model = FishMultimodalModel(
        TinyDino(),
        LinearDinoToBioClipProjector(4, 3),
        LearnableLogitScale(),
        supervised_head=nn.Linear(4, 2),
    )
    configure_training_stage(model, "joint_alignment_final_block")
    optimizer = build_optimizer(
        model,
        {
            "training": {
                "stage": "joint_alignment_final_block",
                "lr": 1.0e-4,
                "weight_decay": 0.01,
            },
            "optimiser": {"classifier_lr": 1.0e-4},
        },
    )
    assert "dino_last_block" in {
        group["name"] for group in optimizer.param_groups
    }


def test_dino_seen_classifier_trains_full_dino_and_head_only() -> None:
    model = FishMultimodalModel(
        TinyDino(),
        LinearDinoToBioClipProjector(4, 3),
        LearnableLogitScale(),
        supervised_head=nn.Linear(4, 2),
    )
    configure_training_stage(model, "dino_seen_classifier")
    assert all(parameter.requires_grad for parameter in model.dino.parameters())
    assert all(
        parameter.requires_grad for parameter in model.supervised_head.parameters()
    )
    assert not any(
        parameter.requires_grad for parameter in model.projector.parameters()
    )
    optimizer = build_optimizer(
        model,
        {
            "training": {
                "stage": "dino_seen_classifier",
                "lr": 1.0e-4,
                "weight_decay": 0.01,
            },
            "optimiser": {
                "backbone_lr": 3.0e-6,
                "classifier_lr": 3.0e-4,
            },
        },
    )
    groups = {group["name"]: group for group in optimizer.param_groups}
    assert set(groups) == {"dino_backbone", "supervised_head"}
    assert groups["dino_backbone"]["lr"] == 3.0e-6
    assert groups["supervised_head"]["lr"] == 3.0e-4
