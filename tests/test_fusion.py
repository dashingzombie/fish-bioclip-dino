import pytest
import torch

from fish_vlm.models.fusion import (
    CalibrationParameters,
    expanded_supervised_probabilities,
    fuse_seen_probabilities,
    fuse_text_probabilities,
)


def test_calibrated_probability_fusion() -> None:
    dino = torch.tensor([[4.0, 0.0]])
    bioclip = torch.tensor([[0.0, 4.0]])
    calibration = CalibrationParameters(
        dino_temperature=2.0, bioclip_temperature=1.0,
        supervised_temperature=1.0, dino_text_weight=0.75, supervised_weight=0.6,
    )
    text = fuse_text_probabilities(dino, bioclip, calibration)
    assert torch.allclose(text.sum(-1), torch.ones(1))
    seen = fuse_seen_probabilities(torch.tensor([[3.0, 0.0]]), text, calibration)
    assert torch.allclose(seen.sum(-1), torch.ones(1))
    with pytest.raises(ValueError):
        fuse_text_probabilities(dino, torch.randn(1, 3), calibration)


def test_subset_supervised_probabilities_expand_into_full_seen_order() -> None:
    logits = torch.tensor([[4.0, 1.0]])
    expanded = expanded_supervised_probabilities(
        logits,
        1.0,
        class_count=4,
        class_indices=[1, 3],
    )
    assert expanded.shape == (1, 4)
    assert expanded[0, 0] == 0
    assert expanded[0, 2] == 0
    assert torch.allclose(expanded.sum(-1), torch.ones(1))

    text = torch.softmax(torch.tensor([[1.0, 2.0, 3.0, 4.0]]), -1)
    fused = fuse_seen_probabilities(
        logits,
        text,
        CalibrationParameters(supervised_weight=0.5),
        supervised_class_indices=[1, 3],
    )
    assert fused.shape == text.shape
    assert torch.allclose(fused.sum(-1), torch.ones(1))

    with pytest.raises(ValueError, match="explicit"):
        expanded_supervised_probabilities(
            logits,
            1.0,
            class_count=4,
        )
