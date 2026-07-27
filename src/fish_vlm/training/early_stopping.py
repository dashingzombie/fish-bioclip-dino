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
        else:
            self.bad_evaluations += 1
        return improved, self.bad_evaluations >= self.patience
