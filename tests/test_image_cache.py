from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fish_vlm.data.datasets import FishMultiViewDataset
from fish_vlm.data.image_cache import (
    DeterministicImageCacheCollection,
    build_deterministic_image_cache,
    load_deterministic_image_cache,
    validate_image_filenames,
)


def _transform(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def test_deterministic_image_cache_round_trip_and_dataset_use(
    tmp_path: Path,
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(images / "a.png")
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(images / "b.png")
    path = tmp_path / "cache" / "train"
    built = build_deterministic_image_cache(
        path=path,
        filenames=["a.png", "b.png"],
        images_dir=images,
        dino_transform=_transform,
        bioclip_transform=_transform,
        dino_model_name="tiny",
        bioclip_checkpoint="mock",
        dino_transform_hash="dino-transform",
        bioclip_transform_hash="bioclip-transform",
        dtype="float16",
        batch_size=2,
        num_workers=0,
    )
    loaded = load_deterministic_image_cache(
        path,
        expected_filenames=["a.png", "b.png"],
        dino_model_name="tiny",
        bioclip_checkpoint="mock",
        dino_transform_hash="dino-transform",
        bioclip_transform_hash="bioclip-transform",
    )
    assert built.manifest == loaded.manifest
    collection = DeterministicImageCacheCollection([loaded])

    def forbidden(image):
        raise AssertionError("raw transforms must not run when cache is supplied")

    dataset = FishMultiViewDataset(
        ["a.png"],
        images,
        forbidden,
        forbidden,
        labels={"a.png": "A fish"},
        species_to_index={"A fish": 0},
        image_cache=collection,
    )
    sample = dataset[0]
    assert sample["dino_image"].dtype == torch.float32
    assert sample["bioclip_image"].shape == (3, 4, 4)


def test_image_cache_rejects_unsafe_paths() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        validate_image_filenames(["../escape.png"])
