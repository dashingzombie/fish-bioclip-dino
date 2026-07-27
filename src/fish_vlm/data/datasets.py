"""Datasets that decode once and produce independent model views."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset


class FishMultiViewDataset(Dataset[dict[str, Any]]):
    """Decode one RGB image and apply distinct DINO/BioCLIP transforms."""

    def __init__(
        self,
        filenames: list[str],
        images_dir: str | Path,
        dino_transform: Any,
        bioclip_transform: Any,
        labels: dict[str, str] | None = None,
        species_to_index: dict[str, int] | None = None,
    ) -> None:
        self.filenames = list(filenames)
        self.images_dir = Path(images_dir)
        self.dino_transform = dino_transform
        self.bioclip_transform = bioclip_transform
        self.labels = labels or {}
        self.species_to_index = species_to_index or {}
        missing = sorted({self.labels[name] for name in self.filenames if name in self.labels} - set(self.species_to_index))
        if missing:
            raise ValueError(f"Dataset labels absent from candidate mapping: {missing}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename = self.filenames[index]
        path = self.images_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as source:
            image = source.convert("RGB")
            dino_image = self.dino_transform(image)
            bioclip_image = self.bioclip_transform(image)
        species_index = None
        if filename in self.labels:
            species_index = self.species_to_index[self.labels[filename]]
        return {
            "dino_image": dino_image,
            "bioclip_image": bioclip_image,
            "species_index": species_index,
            "filename": filename,
        }


class BioClipImageDataset(Dataset[dict[str, Any]]):
    """BioCLIP-only view used by the no-training Stage 0 baseline."""

    def __init__(
        self,
        filenames: list[str],
        images_dir: str | Path,
        transform: Any,
        labels: dict[str, str],
        species_to_index: dict[str, int],
    ) -> None:
        self.filenames = list(filenames)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.labels = labels
        self.species_to_index = species_to_index

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Any]:
        filename = self.filenames[index]
        with Image.open(self.images_dir / filename) as source:
            image = self.transform(source.convert("RGB"))
        return {
            "bioclip_image": image,
            "species_index": self.species_to_index[self.labels[filename]],
            "filename": filename,
        }
