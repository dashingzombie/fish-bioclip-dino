from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from fish_vlm.evaluation.gating import (
    fit_confidence_gate,
    gated_prediction_indices,
    threshold_for_acceptance_rate,
)
from fish_vlm.hybrid.workflow import (
    materialize_hybrid_plan,
    planned_commands,
    select_best_recipe,
)
from fish_vlm.inference.bioclip_checkpoint import (
    EXPECTED_FINETUNE_LOSSES,
    load_finetuned_bioclip_visual,
)
from fish_vlm.utils.hashing import ordered_names_hash


def test_gate_routes_confident_dino_and_falls_back_to_bioclip() -> None:
    supervised = torch.tensor([[8.0, 0.0], [0.2, 0.0]])
    bioclip = torch.tensor(
        [[0.0, 0.0, 9.0], [0.0, 0.0, 9.0]]
    )
    prediction, use_dino, confidence = gated_prediction_indices(
        supervised,
        bioclip,
        supervised_class_indices=[0, 1],
        threshold=0.8,
        supervised_temperature=1.0,
    )
    assert prediction.tolist() == [0, 2]
    assert use_dino.tolist() == [True, False]
    assert confidence[0] > confidence[1]


def test_gate_threshold_uses_known_and_pseudo_unseen_only(monkeypatch) -> None:
    monkeypatch.setattr(
        "fish_vlm.evaluation.gating.fit_temperature", lambda *args: 1.0
    )
    supervised = torch.tensor(
        [
            [8.0, 0.0],
            [0.0, 8.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    bioclip = torch.tensor(
        [
            [0.0, 8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 8.0],
        ]
    )
    fit = fit_confidence_gate(
        supervised,
        bioclip,
        torch.tensor([0, 1, 2, 3]),
        supervised_class_indices=[0, 1],
        known_mask=torch.tensor([True, True, False, False]),
        thresholds=[0.0, 0.8, 0.99, 1.0],
        selection_metric="seen_unseen_harmonic_mean",
        official_seen_count=2,
        official_unseen_count=2,
    )
    assert fit.threshold == pytest.approx(0.8)
    assert fit.known_accuracy == 1.0
    assert fit.pseudo_unseen_accuracy == 1.0
    assert fit.seen_unseen_harmonic_mean == 1.0


def test_selected_acceptance_rate_transfers_to_final_confidence_scale() -> None:
    confidence = torch.tensor([0.95, 0.9, 0.8, 0.2])
    threshold = threshold_for_acceptance_rate(confidence, 0.5)
    assert threshold == pytest.approx(0.85)
    assert int((confidence >= threshold).sum()) == 2


def test_hybrid_plan_materialises_four_submission_workflow_without_execution(
    tmp_path: Path,
) -> None:
    spec = yaml.safe_load(
        Path("configs/hybrid/sweep.yaml").read_text(encoding="utf-8")
    )
    spec["output_root"] = str(tmp_path / "hybrid")
    spec_path = tmp_path / "sweep.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    plan = materialize_hybrid_plan(spec_path)
    assert len(plan["runs"]) == 6
    assert len({run["name"] for run in plan["runs"]}) == 6
    commands = planned_commands(plan)
    assert sum(item["step"].startswith("train:") for item in commands) == 7
    assert all(Path(run["config"]).is_file() for run in plan["runs"])
    assert not any(Path(run["checkpoint"]).exists() for run in plan["runs"])
    assert set(plan["submission_outputs"]) == {
        "pretrained_bioclip_hard_routed",
        "pretrained_bioclip_confidence_gated",
        "finetuned_bioclip_hard_routed",
        "finetuned_bioclip_confidence_gated",
    }
    assert all(
        Path(path).suffix == ".zip"
        for path in plan["submission_outputs"].values()
    )

    first = yaml.safe_load(
        Path(plan["runs"][0]["config"]).read_text(encoding="utf-8")
    )
    assert first["validation"]["pseudo_unseen"]["split_seed"] == 42
    assert first["training"]["stage"] == "dino_seen_classifier"
    assert first["training"]["max_steps"] == 16000
    assert first["loss"]["supervised_species"]["label_smoothing"] == 0.1

    bioclip = yaml.safe_load(
        Path(plan["finetuned_bioclip"]["config"]).read_text(
            encoding="utf-8"
        )
    )
    assert bioclip["training"]["stage"] == "bioclip_full_finetune"
    assert bioclip["training"]["max_steps"] == 20000
    assert bioclip["model"]["bioclip"]["freeze_text_encoder"]
    assert not bioclip["model"]["bioclip"]["freeze_image_encoder"]
    assert bioclip["validation"]["pseudo_unseen"]["split_seed"] == 42


def test_recipe_selection_uses_gate_objective(tmp_path: Path) -> None:
    runs = []
    for name, score, pseudo in (("a", 0.8, 0.9), ("b", 0.9, 0.7)):
        gate_path = tmp_path / name / "gate.json"
        gate_path.parent.mkdir(parents=True)
        gate_path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "selection_value": score,
                        "pseudo_unseen_accuracy": pseudo,
                        "known_accuracy": 0.95,
                    }
                }
            )
        )
        runs.append(
            {
                "name": name,
                "parameters": {},
                "gate": str(gate_path),
            }
        )
    selected = select_best_recipe({"runs": runs})
    assert selected["name"] == "b"


class _TinyBioCLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visual = nn.Linear(2, 2, bias=False)
        self.text = nn.Linear(2, 2, bias=False)


def _bioclip_checkpoint(model: nn.Module) -> dict[str, object]:
    seen = ["Seen one", "Seen two"]
    unseen = ["Unseen one"]
    state = {
        f"bioclip.{key}": value.detach().clone()
        for key, value in model.state_dict().items()
    }
    state["bioclip.visual.weight"] = state["bioclip.visual.weight"] + 1.0
    return {
        "dino_model_name": "dino-test",
        "dino_checkpoint_source": "test-source",
        "bioclip_checkpoint": "bioclip-test",
        "text_prototype_hash": "text-hash",
        "canonical_prompt_hash": "prompt-hash",
        "seen_species": seen,
        "unseen_species": unseen,
        "training_species": seen,
        "training_species_hash": ordered_names_hash(seen),
        "active_losses": EXPECTED_FINETUNE_LOSSES,
        "resolved_configuration": {
            "training": {"stage": "bioclip_full_finetune"},
            "model": {
                "tuning_mode": "full_finetune",
                "bioclip": {
                    "freeze_image_encoder": False,
                    "freeze_text_encoder": True,
                },
            },
        },
        "model_state": state,
        "step": 123,
    }


def test_finetuned_bioclip_loader_changes_visual_only(tmp_path: Path) -> None:
    model = _TinyBioCLIP()
    original_visual = model.visual.weight.detach().clone()
    original_text = model.text.weight.detach().clone()
    checkpoint = _bioclip_checkpoint(model)
    path = tmp_path / "bioclip.pt"
    torch.save(checkpoint, path)

    loaded = load_finetuned_bioclip_visual(
        path,
        model,
        expected_seen_species=["Seen one", "Seen two"],
        expected_unseen_species=["Unseen one"],
        expected_training_species=["Seen one", "Seen two"],
        expected_text_prototype_hash="text-hash",
        expected_canonical_prompt_hash="prompt-hash",
        expected_bioclip_checkpoint="bioclip-test",
    )

    assert loaded["step"] == 123
    assert torch.equal(model.visual.weight, original_visual + 1.0)
    assert torch.equal(model.text.weight, original_text)


def test_finetuned_bioclip_loader_rejects_changed_text(tmp_path: Path) -> None:
    model = _TinyBioCLIP()
    checkpoint = _bioclip_checkpoint(model)
    checkpoint["model_state"]["bioclip.text.weight"] += 1.0  # type: ignore[index]
    path = tmp_path / "bad-bioclip.pt"
    torch.save(checkpoint, path)

    with pytest.raises(ValueError, match="changed frozen text"):
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
