"""Batch collation for labelled and image-only splits."""

from __future__ import annotations

from typing import Any

import torch


def collate_multiview(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack both views while representing absent labels explicitly as ``None``."""
    labels = [item["species_index"] for item in batch]
    if any(label is None for label in labels) and not all(label is None for label in labels):
        raise ValueError("A batch may not mix labelled and unlabelled samples")
    return {
        "dino_image": torch.stack([item["dino_image"] for item in batch]),
        "bioclip_image": torch.stack([item["bioclip_image"] for item in batch]),
        "species_index": None if labels[0] is None else torch.tensor(labels, dtype=torch.long),
        "filename": [item["filename"] for item in batch],
    }

