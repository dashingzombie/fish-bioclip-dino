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
) -> torch.Tensor:
    """Fuse a seen-species supervised head with text probabilities."""
    supervised = calibrated_probabilities(supervised_logits, calibration.supervised_temperature)
    if supervised.shape != text_probabilities.shape:
        raise ValueError("Supervised and text branches must use identical seen ordering")
    return calibration.supervised_weight * supervised + (1.0 - calibration.supervised_weight) * text_probabilities

