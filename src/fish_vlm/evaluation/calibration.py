"""Validation-only temperature and fusion calibration."""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from fish_vlm.models.fusion import CalibrationParameters, fuse_seen_probabilities, fuse_text_probabilities
from fish_vlm.utils.hashing import stable_json_hash
from fish_vlm.utils.io import write_json


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor, *, max_iter: int = 100) -> float:
    """Fit one positive scalar temperature by validation NLL."""
    logits = logits.detach().float()
    targets = targets.detach().long()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp().clamp(0.01, 100.0), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.01, 100.0))


def _best_weight(first: torch.Tensor, second: torch.Tensor, targets: torch.Tensor) -> float:
    best_weight, best_accuracy = 0.5, -1.0
    for index in range(101):
        weight = index / 100.0
        accuracy = ((weight * first + (1 - weight) * second).argmax(-1) == targets).float().mean().item()
        if accuracy > best_accuracy:
            best_accuracy, best_weight = accuracy, weight
    return best_weight


def fit_calibration(
    dino_logits: torch.Tensor,
    bioclip_logits: torch.Tensor,
    targets: torch.Tensor,
    supervised_logits: torch.Tensor | None = None,
) -> CalibrationParameters:
    """Fit independent temperatures, then probability-space fusion weights."""
    dino_temperature = fit_temperature(dino_logits, targets)
    bioclip_temperature = fit_temperature(bioclip_logits, targets)
    dino_prob = torch.softmax(dino_logits.float() / dino_temperature, -1)
    bioclip_prob = torch.softmax(bioclip_logits.float() / bioclip_temperature, -1)
    dino_weight = _best_weight(dino_prob, bioclip_prob, targets)
    supervised_temperature = 1.0
    supervised_weight = 0.7
    if supervised_logits is not None:
        supervised_temperature = fit_temperature(supervised_logits, targets)
        provisional = CalibrationParameters(
            dino_temperature, bioclip_temperature, supervised_temperature, dino_weight, 0.5
        )
        text_prob = fuse_text_probabilities(dino_logits, bioclip_logits, provisional)
        supervised_prob = torch.softmax(supervised_logits.float() / supervised_temperature, -1)
        supervised_weight = _best_weight(supervised_prob, text_prob, targets)
    return CalibrationParameters(
        dino_temperature, bioclip_temperature, supervised_temperature, dino_weight, supervised_weight
    )


def save_calibration(path: str, calibration: CalibrationParameters, metadata: dict) -> dict:
    """Persist parameters and a tamper-evident content hash."""
    value = {"parameters": asdict(calibration), "metadata": metadata}
    value["hash"] = stable_json_hash(value)
    write_json(path, value)
    return value

