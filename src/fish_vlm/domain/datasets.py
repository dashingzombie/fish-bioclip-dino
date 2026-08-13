"""Raw-image multi-view datasets for domain adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset


class DomainMultiViewDataset(Dataset[dict[str, Any]]):
    """Return augmented views and an optional, explicitly masked text target."""

    def __init__(
        self,
        filenames: list[str],
        images_dir: str | Path,
        transform: Any,
        *,
        labels: dict[str, str] | None = None,
        species_to_index: dict[str, int] | None = None,
        paired_species: set[str] | None = None,
    ) -> None:
        self.filenames = list(filenames)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.labels = labels or {}
        self.species_to_index = species_to_index or {}
        self.paired_species = paired_species

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename = self.filenames[index]
        with Image.open(self.images_dir / filename) as source:
            views = self.transform(source.convert("RGB"))
        if not isinstance(views, list) or not views:
            raise TypeError("Domain transform must return a non-empty view list")
        species = self.labels.get(filename)
        paired = species is not None and (
            self.paired_species is None or species in self.paired_species
        )
        target = self.species_to_index[species] if paired else -1
        return {
            "filename": filename,
            "views": views,
            "target": target,
            "has_text": paired,
        }


def collate_domain_views(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack each crop position while retaining the paired-text mask."""
    view_count = len(batch[0]["views"])
    if any(len(item["views"]) != view_count for item in batch):
        raise ValueError("Every domain sample must have the same view count")
    return {
        "filename": [item["filename"] for item in batch],
        "views": [
            torch.stack([item["views"][view] for item in batch])
            for view in range(view_count)
        ],
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.long),
        "has_text": torch.tensor([item["has_text"] for item in batch], dtype=torch.bool),
    }
