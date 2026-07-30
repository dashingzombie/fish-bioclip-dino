"""Validation-only temperature and fusion calibration."""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from fish_vlm.models.fusion import (
    CalibrationParameters,
    apply_seen_class_penalty,
    expanded_supervised_probabilities,
    fuse_text_probabilities,
)
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


def fit_probability_weight(
    first: torch.Tensor,
    second: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Grid-fit one convex probability weight on the supplied validation set."""
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
    *,
    supervised_class_indices: list[int] | torch.Tensor | None = None,
) -> CalibrationParameters:
    """Fit temperatures and fusion with an optional subset-trained head."""
    dino_temperature = fit_temperature(dino_logits, targets)
    bioclip_temperature = fit_temperature(bioclip_logits, targets)
    dino_prob = torch.softmax(dino_logits.float() / dino_temperature, -1)
    bioclip_prob = torch.softmax(bioclip_logits.float() / bioclip_temperature, -1)
    dino_weight = fit_probability_weight(
        dino_prob, bioclip_prob, targets
    )
    supervised_temperature = 1.0
    supervised_weight = 0.7
    if supervised_logits is not None:
        class_count = dino_logits.shape[1]
        if supervised_class_indices is None:
            if supervised_logits.shape[1] != class_count:
                raise ValueError(
                    "Subset-sized supervised logits require supervised_class_indices"
                )
            eligible = torch.ones_like(targets, dtype=torch.bool)
            supervised_targets = targets
        else:
            indices = torch.as_tensor(
                supervised_class_indices,
                dtype=torch.long,
                device=targets.device,
            )
            if indices.ndim != 1 or indices.numel() != supervised_logits.shape[1]:
                raise ValueError(
                    "Supervised class indices must match the supervised-head width"
                )
            if indices.unique().numel() != indices.numel():
                raise ValueError("Supervised class indices contain duplicates")
            if bool(((indices < 0) | (indices >= class_count)).any()):
                raise ValueError(
                    "Supervised class index is outside the text candidate space"
                )
            full_to_subset = torch.full(
                (class_count,),
                -1,
                dtype=torch.long,
                device=targets.device,
            )
            full_to_subset[indices] = torch.arange(
                len(indices),
                dtype=torch.long,
                device=targets.device,
            )
            supervised_targets = full_to_subset[targets]
            eligible = supervised_targets >= 0
            if not bool(eligible.any()):
                raise ValueError(
                    "Calibration data contains no targets represented by the supervised head"
                )
        supervised_temperature = fit_temperature(
            supervised_logits[eligible],
            supervised_targets[eligible],
        )
        provisional = CalibrationParameters(
            dino_temperature,
            bioclip_temperature,
            supervised_temperature,
            dino_weight,
            0.5,
        )
        text_prob = fuse_text_probabilities(dino_logits, bioclip_logits, provisional)
        supervised_prob = expanded_supervised_probabilities(
            supervised_logits,
            supervised_temperature,
            class_count=class_count,
            class_indices=supervised_class_indices,
        )
        supervised_weight = fit_probability_weight(
            supervised_prob, text_prob, targets
        )
    return CalibrationParameters(
        dino_temperature, bioclip_temperature, supervised_temperature, dino_weight, supervised_weight
    )


def fit_species_disjoint_fusion(
    seen_dino_logits: torch.Tensor,
    seen_bioclip_logits: torch.Tensor,
    seen_targets: torch.Tensor,
    pseudo_dino_logits: torch.Tensor,
    pseudo_bioclip_logits: torch.Tensor,
    pseudo_targets: torch.Tensor,
    *,
    seen_class_indices: list[int] | torch.Tensor,
    gamma_values: list[float] | None = None,
) -> CalibrationParameters:
    """Tune DINO/BioCLIP fusion and seen penalty by disjoint harmonic mean."""
    if seen_dino_logits.shape[1] != pseudo_dino_logits.shape[1]:
        raise ValueError(
            "Seen and pseudo-unseen logits must share one all-class ordering"
        )
    dino_temperature = fit_temperature(
        torch.cat([seen_dino_logits, pseudo_dino_logits]),
        torch.cat([seen_targets, pseudo_targets]),
    )
    bioclip_temperature = fit_temperature(
        torch.cat([seen_bioclip_logits, pseudo_bioclip_logits]),
        torch.cat([seen_targets, pseudo_targets]),
    )
    gamma_grid = gamma_values or [
        index / 20.0 for index in range(0, 41)
    ]
    best: tuple[float, float, float] | None = None
    for weight_index in range(101):
        weight = weight_index / 100.0
        calibration = CalibrationParameters(
            dino_temperature=dino_temperature,
            bioclip_temperature=bioclip_temperature,
            dino_text_weight=weight,
        )
        seen_probabilities = fuse_text_probabilities(
            seen_dino_logits, seen_bioclip_logits, calibration
        )
        pseudo_probabilities = fuse_text_probabilities(
            pseudo_dino_logits, pseudo_bioclip_logits, calibration
        )
        for gamma in gamma_grid:
            seen_penalised = apply_seen_class_penalty(
                seen_probabilities, seen_class_indices, gamma
            )
            pseudo_penalised = apply_seen_class_penalty(
                pseudo_probabilities, seen_class_indices, gamma
            )
            seen_accuracy = (
                seen_penalised.argmax(-1) == seen_targets
            ).float().mean().item()
            pseudo_accuracy = (
                pseudo_penalised.argmax(-1) == pseudo_targets
            ).float().mean().item()
            denominator = seen_accuracy + pseudo_accuracy
            harmonic = (
                0.0
                if denominator == 0
                else 2
                * seen_accuracy
                * pseudo_accuracy
                / denominator
            )
            candidate = (harmonic, weight, float(gamma))
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    return CalibrationParameters(
        dino_temperature=dino_temperature,
        bioclip_temperature=bioclip_temperature,
        dino_text_weight=best[1],
        calibration_gamma=best[2],
    )


def save_calibration(path: str, calibration: CalibrationParameters, metadata: dict) -> dict:
    """Persist parameters and a tamper-evident content hash."""
    value = {"parameters": asdict(calibration), "metadata": metadata}
    value["hash"] = stable_json_hash(value)
    write_json(path, value)
    return value
