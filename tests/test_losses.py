import torch

from fish_vlm.losses.consistency import branch_consistency_loss
from fish_vlm.losses.image_teacher import cosine_teacher_loss, symmetric_contrastive_teacher_loss
from fish_vlm.losses.total import compute_total_loss
from fish_vlm.models.multimodal import ModelOutput
from fish_vlm.training.metrics import distributed_classification_metrics


def test_losses_are_finite_and_explicitly_weighted() -> None:
    targets = torch.tensor([0, 1])
    projected = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    teacher = projected.clone()
    output = ModelOutput(
        torch.randn(2, 4), projected, torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        teacher, torch.tensor([[2.0, 0.0], [0.0, 2.0]]), None,
    )
    config = {
        "dino_text_classification": {"enabled": True, "weight": 1.0},
        "bioclip_image_teacher": {"enabled": True, "weight": 0.25, "method": "cosine"},
        "supervised_species": {"enabled": False, "weight": 0.5},
        "native_bioclip_text": {"enabled": False, "weight": 1.0},
        "branch_consistency": {"enabled": False, "weight": 0.05},
    }
    result = compute_total_loss(output, targets, config, teacher_embeddings=teacher)
    expected = result.components["dino_text_classification"] + 0.25 * result.components["bioclip_image_teacher"]
    assert torch.allclose(result.total, expected)
    assert cosine_teacher_loss(projected, teacher).item() < 1e-6
    assert torch.isfinite(symmetric_contrastive_teacher_loss(projected, teacher))
    assert branch_consistency_loss(output.dino_text_logits, output.dino_text_logits).abs() < 1e-6


def test_ddp_safe_metric_path_without_process_group() -> None:
    scores = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    targets = torch.tensor([0, 1])
    metrics = distributed_classification_metrics(scores, targets, prefix="branch")
    assert metrics["branch_accuracy"] == 1.0
    assert metrics["branch_top5_accuracy"] == 1.0
