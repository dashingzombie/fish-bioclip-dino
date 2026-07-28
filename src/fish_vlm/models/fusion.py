"""Temperature calibration and probability-space branch fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CalibrationParameters:
    """Independent branch temperatures and fusion weights."""

    dino_temperature: float = 1.0
    bioclip_temperature: float = 1.0
    supervised_temperature: float = 1.0
    dino_text_weight: float = 0.5
    supervised_weight: float = 0.7

    def __post_init__(self) -> None:
        if min(self.dino_temperature, self.bioclip_temperature, self.supervised_temperature) <= 0:
            raise ValueError("All temperatures must be positive")
        if not 0 <= self.dino_text_weight <= 1 or not 0 <= self.supervised_weight <= 1:
            raise ValueError("Fusion weights must be in [0, 1]")


def calibrated_probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax calibrated float32 logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(logits.float() / temperature, dim=-1)


def expanded_supervised_probabilities(
    supervised_logits: torch.Tensor,
    temperature: float,
    *,
    class_count: int,
    class_indices: list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Map a subset-trained supervised head into the full candidate ordering."""
    if supervised_logits.ndim != 2:
        raise ValueError("Supervised logits must be a rank-two matrix")
    if class_count < 1:
        raise ValueError("Supervised candidate class_count must be positive")
    probabilities = calibrated_probabilities(supervised_logits, temperature)
    if class_indices is None:
        if probabilities.shape[1] != class_count:
            raise ValueError(
                "Subset-sized supervised logits require explicit full-space class indices"
            )
        return probabilities

    indices = torch.as_tensor(
        class_indices,
        dtype=torch.long,
        device=probabilities.device,
    )
    if indices.ndim != 1 or indices.numel() != probabilities.shape[1]:
        raise ValueError(
            "Supervised class indices must match the supervised-head output width"
        )
    if indices.unique().numel() != indices.numel():
        raise ValueError("Supervised class indices contain duplicates")
    if bool(((indices < 0) | (indices >= class_count)).any()):
        raise ValueError("Supervised class index is outside the candidate label space")
    expanded = probabilities.new_zeros((probabilities.shape[0], class_count))
    expanded.index_copy_(1, indices, probabilities)
    return expanded


def fuse_text_probabilities(
    dino_logits: torch.Tensor,
    bioclip_logits: torch.Tensor,
    calibration: CalibrationParameters,
) -> torch.Tensor:
    """Convexly combine calibrated text-branch probabilities."""
    if dino_logits.shape != bioclip_logits.shape:
        raise ValueError("Text branch logits must use the identical candidate ordering")
    dino = calibrated_probabilities(dino_logits, calibration.dino_temperature)
    bioclip = calibrated_probabilities(bioclip_logits, calibration.bioclip_temperature)
    return calibration.dino_text_weight * dino + (1.0 - calibration.dino_text_weight) * bioclip


def fuse_seen_probabilities(
    supervised_logits: torch.Tensor,
    text_probabilities: torch.Tensor,
    calibration: CalibrationParameters,
    *,
    supervised_class_indices: list[int] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Fuse subset or full-space supervised predictions with seen text predictions."""
    supervised = expanded_supervised_probabilities(
        supervised_logits,
        calibration.supervised_temperature,
        class_count=text_probabilities.shape[1],
        class_indices=supervised_class_indices,
    )
    if supervised.shape[0] != text_probabilities.shape[0]:
        raise ValueError("Supervised and text branches must have equal batch sizes")
    return calibration.supervised_weight * supervised + (1.0 - calibration.supervised_weight) * text_probabilities
