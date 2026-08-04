"""Leakage-safe confidence gating between seen DINO and frozen BioCLIP."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fish_vlm.evaluation.calibration import fit_temperature


@dataclass(frozen=True)
class GateFit:
    """Selected gate parameters and validation metrics."""

    threshold: float
    supervised_temperature: float
    selection_metric: str
    selection_value: float
    known_accuracy: float
    pseudo_unseen_accuracy: float
    seen_unseen_harmonic_mean: float
    estimated_overall_accuracy: float
    dino_route_fraction: float
    known_dino_route_fraction: float
    pseudo_unseen_dino_route_fraction: float


def gated_prediction_indices(
    supervised_logits: torch.Tensor,
    bioclip_logits: torch.Tensor,
    *,
    supervised_class_indices: list[int] | torch.Tensor,
    threshold: float,
    supervised_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route high-confidence DINO predictions, otherwise use BioCLIP."""
    if supervised_logits.ndim != 2 or bioclip_logits.ndim != 2:
        raise ValueError("DINO and BioCLIP logits must be rank-two matrices")
    if supervised_logits.shape[0] != bioclip_logits.shape[0]:
        raise ValueError("DINO and BioCLIP batch sizes differ")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Gate threshold must be in [0, 1]")
    if supervised_temperature <= 0:
        raise ValueError("Supervised temperature must be positive")
    indices = torch.as_tensor(
        supervised_class_indices,
        dtype=torch.long,
        device=supervised_logits.device,
    )
    if indices.ndim != 1 or indices.numel() != supervised_logits.shape[1]:
        raise ValueError(
            "Supervised class indices must match the DINO classifier width"
        )
    if indices.unique().numel() != indices.numel():
        raise ValueError("Supervised class indices contain duplicates")
    if bool(((indices < 0) | (indices >= bioclip_logits.shape[1])).any()):
        raise ValueError("A DINO class is outside the BioCLIP candidate space")

    dino_probabilities = torch.softmax(
        supervised_logits.float() / float(supervised_temperature), dim=-1
    )
    confidence, subset_prediction = dino_probabilities.max(dim=-1)
    dino_prediction = indices[subset_prediction]
    bioclip_prediction = bioclip_logits.float().argmax(dim=-1)
    use_dino = confidence >= float(threshold)
    prediction = torch.where(use_dino, dino_prediction, bioclip_prediction)
    return prediction, use_dino, confidence


def _harmonic_mean(first: float, second: float) -> float:
    denominator = first + second
    return 0.0 if denominator == 0 else 2.0 * first * second / denominator


def threshold_for_acceptance_rate(
    confidence: torch.Tensor,
    acceptance_rate: float,
) -> float:
    """Map a selected known acceptance rate onto a new confidence scale."""
    if confidence.ndim != 1 or confidence.numel() == 0:
        raise ValueError("Confidence must be a non-empty vector")
    if not 0.0 <= acceptance_rate <= 1.0:
        raise ValueError("Acceptance rate must be in [0, 1]")
    count = int(round(float(acceptance_rate) * confidence.numel()))
    if count <= 0:
        return 1.0
    if count >= confidence.numel():
        return 0.0
    ordered = confidence.detach().float().sort(descending=True).values
    accepted_edge = float(ordered[count - 1])
    rejected_edge = float(ordered[count])
    return (accepted_edge + rejected_edge) / 2.0


def fit_confidence_gate(
    supervised_logits: torch.Tensor,
    bioclip_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    supervised_class_indices: list[int] | torch.Tensor,
    known_mask: torch.Tensor,
    thresholds: list[float],
    selection_metric: str,
    official_seen_count: int,
    official_unseen_count: int,
) -> GateFit:
    """Fit temperature and threshold on known/pseudo-unseen validation data."""
    if not thresholds:
        raise ValueError("At least one gate threshold is required")
    if targets.ndim != 1 or known_mask.ndim != 1:
        raise ValueError("Targets and known_mask must be vectors")
    if len(targets) != len(supervised_logits) or len(targets) != len(known_mask):
        raise ValueError("Gate calibration tensors have inconsistent lengths")
    known_mask = known_mask.bool()
    pseudo_mask = ~known_mask
    if not bool(known_mask.any()) or not bool(pseudo_mask.any()):
        raise ValueError(
            "Gate selection requires both known and pseudo-unseen validation images"
        )
    indices = torch.as_tensor(
        supervised_class_indices,
        dtype=torch.long,
        device=targets.device,
    )
    full_to_subset = torch.full(
        (bioclip_logits.shape[1],),
        -1,
        dtype=torch.long,
        device=targets.device,
    )
    full_to_subset[indices] = torch.arange(
        len(indices), dtype=torch.long, device=targets.device
    )
    known_targets = full_to_subset[targets[known_mask]]
    if bool((known_targets < 0).any()):
        raise ValueError("Known validation targets are absent from the DINO head")
    temperature = fit_temperature(
        supervised_logits[known_mask], known_targets
    )
    total_official = official_seen_count + official_unseen_count
    if total_official <= 0:
        raise ValueError("Official split counts must have a positive total")
    supported_metrics = {
        "estimated_overall_accuracy",
        "seen_unseen_harmonic_mean",
    }
    if selection_metric not in supported_metrics:
        raise ValueError(
            f"Unsupported gate selection metric: {selection_metric}"
        )

    best: tuple[tuple[float, float, float, float, float, float], GateFit] | None = None
    for threshold in sorted({float(value) for value in thresholds}):
        prediction, use_dino, _ = gated_prediction_indices(
            supervised_logits,
            bioclip_logits,
            supervised_class_indices=indices,
            threshold=threshold,
            supervised_temperature=temperature,
        )
        known_accuracy = float(
            (prediction[known_mask] == targets[known_mask]).float().mean()
        )
        pseudo_accuracy = float(
            (prediction[pseudo_mask] == targets[pseudo_mask]).float().mean()
        )
        harmonic = _harmonic_mean(known_accuracy, pseudo_accuracy)
        estimated = (
            official_seen_count * known_accuracy
            + official_unseen_count * pseudo_accuracy
        ) / total_official
        selection_value = (
            estimated
            if selection_metric == "estimated_overall_accuracy"
            else harmonic
        )
        fit = GateFit(
            threshold=threshold,
            supervised_temperature=temperature,
            selection_metric=selection_metric,
            selection_value=selection_value,
            known_accuracy=known_accuracy,
            pseudo_unseen_accuracy=pseudo_accuracy,
            seen_unseen_harmonic_mean=harmonic,
            estimated_overall_accuracy=estimated,
            dino_route_fraction=float(use_dino.float().mean()),
            known_dino_route_fraction=float(
                use_dino[known_mask].float().mean()
            ),
            pseudo_unseen_dino_route_fraction=float(
                use_dino[pseudo_mask].float().mean()
            ),
        )
        # At equal predictive metrics, preserve the requested DINO path for as
        # many known images as possible while minimising pseudo-unseen leakage.
        ranking = (
            selection_value,
            pseudo_accuracy,
            known_accuracy,
            fit.known_dino_route_fraction,
            -fit.pseudo_unseen_dino_route_fraction,
            -threshold,
        )
        if best is None or ranking > best[0]:
            best = (ranking, fit)
    assert best is not None
    return best[1]
