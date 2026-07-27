"""Rank configurations without consulting official labels."""

from __future__ import annotations

from typing import Any


def rank_results(results: list[dict[str, Any]], selection_metric: str) -> list[dict[str, Any]]:
    """Sort completed pseudo-validation results descending."""
    missing = [item.get("name", "<unnamed>") for item in results if selection_metric not in item.get("metrics", {})]
    if missing:
        raise ValueError(f"Runs lack selection metric {selection_metric!r}: {missing}")
    return sorted(results, key=lambda item: float(item["metrics"][selection_metric]), reverse=True)

