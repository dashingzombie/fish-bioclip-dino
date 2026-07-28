"""Metric-based early stopping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track maximised validation metrics."""

    patience: int
    min_delta: float = 0.0
    best: float = float("-inf")
    bad_evaluations: int = 0

    def update(self, value: float) -> tuple[bool, bool]:
        """Return ``(improved, should_stop)``."""
        improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.bad_evaluations = 0
            print(f"Early stopping: improved to {self.best:.5f}, resetting bad evaluations")
        else:
            self.bad_evaluations += 1
            print(f"Early stopping: {self.bad_evaluations}/{self.patience} bad evaluations")
        return improved, self.bad_evaluations >= self.patience
