"""Evaluation report composition."""

from __future__ import annotations

from fish_vlm.training.metrics import estimated_overall_accuracy, harmonic_mean


def add_selection_metrics(
    metrics: dict[str, float],
    *,
    seen_accuracy: float,
    pseudo_unseen_accuracy: float,
    test_count: int,
    unseen_count: int,
) -> dict[str, float]:
    """Add the two cross-partition selection measures."""
    return {
        **metrics,
        "seen_accuracy": seen_accuracy,
        "pseudo_unseen_accuracy": pseudo_unseen_accuracy,
        "seen_unseen_harmonic_mean": harmonic_mean(seen_accuracy, pseudo_unseen_accuracy),
        "estimated_overall_accuracy": estimated_overall_accuracy(
            seen_accuracy, pseudo_unseen_accuracy, test_count, unseen_count
        ),
    }

