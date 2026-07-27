"""Classification and seen/unseen selection metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from fish_vlm.training.distributed import gather_objects


def classification_metrics(
    logits_or_probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Compute top-1, balanced accuracy, macro-F1 and safe top-5."""
    scores = logits_or_probabilities.detach().float().cpu()
    labels = targets.detach().long().cpu()
    if scores.ndim != 2 or len(scores) != len(labels):
        raise ValueError("Scores and targets have incompatible shapes")
    predictions = scores.argmax(dim=-1)
    k = min(5, scores.shape[1])
    topk = scores.topk(k, dim=-1).indices
    accuracy = (predictions == labels).float().mean().item()
    top5 = (topk == labels[:, None]).any(dim=1).float().mean().item()
    y_true = labels.numpy()
    y_pred = predictions.numpy()
    name = f"{prefix}_" if prefix else ""
    return {
        f"{name}accuracy": accuracy,
        f"{name}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{name}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{name}top5_accuracy": top5,
    }


def distributed_classification_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Gather local predictions before exact nonlinear metric calculation."""
    payloads = gather_objects((scores.detach().cpu(), targets.detach().cpu()))
    all_scores = torch.cat([item[0] for item in payloads])  # type: ignore[index]
    all_targets = torch.cat([item[1] for item in payloads])  # type: ignore[index]
    return classification_metrics(all_scores, all_targets, prefix=prefix)


def harmonic_mean(seen_accuracy: float, unseen_accuracy: float) -> float:
    """Seen/pseudo-unseen harmonic mean."""
    denominator = seen_accuracy + unseen_accuracy
    return 0.0 if denominator == 0 else 2.0 * seen_accuracy * unseen_accuracy / denominator


def estimated_overall_accuracy(
    seen_accuracy: float,
    pseudo_unseen_accuracy: float,
    test_count: int,
    unseen_count: int,
) -> float:
    """Competition-weighted estimate from actual official split sizes."""
    total = test_count + unseen_count
    if total <= 0:
        raise ValueError("Official split image counts must sum to a positive value")
    return (test_count * seen_accuracy + unseen_count * pseudo_unseen_accuracy) / total


SELECTION_METRICS = {
    "estimated_overall_accuracy",
    "seen_unseen_harmonic_mean",
    "pseudo_unseen_accuracy",
    "seen_accuracy",
}


def selection_value(metrics: dict[str, Any], name: str) -> float:
    """Extract a supported model-selection score."""
    if name not in SELECTION_METRICS:
        raise ValueError(f"Unsupported selection metric: {name}")
    if name not in metrics:
        raise KeyError(f"Selection metric was not calculated: {name}")
    return float(metrics[name])

