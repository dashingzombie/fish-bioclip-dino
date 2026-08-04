from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
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


def test_hybrid_plan_materialises_six_configs_and_no_execution(
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
    assert sum(item["step"].startswith("train:") for item in commands) == 6
    assert all(Path(run["config"]).is_file() for run in plan["runs"])
    assert not any(Path(run["checkpoint"]).exists() for run in plan["runs"])

    first = yaml.safe_load(
        Path(plan["runs"][0]["config"]).read_text(encoding="utf-8")
    )
    assert first["validation"]["pseudo_unseen"]["split_seed"] == 42
    assert first["training"]["stage"] == "dino_seen_classifier"


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
