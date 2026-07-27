from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from conftest import TinyBioClip
from fish_vlm.prototypes.image_teacher import (
    build_image_teacher_cache,
    load_image_teacher_cache,
    lookup_teacher_embeddings,
)


def test_teacher_cache_is_ordered_normalised_and_float32_on_lookup(tmp_path: Path) -> None:
    samples = [
        {"bioclip_image": torch.tensor([1.0, 0, 0, 0]), "species_index": 0, "filename": "a.jpg"},
        {"bioclip_image": torch.tensor([0.0, 1, 0, 0]), "species_index": 1, "filename": "b.jpg"},
    ]
    loader = DataLoader(samples, batch_size=2)
    path = tmp_path / "teacher.pt"
    build_image_teacher_cache(
        TinyBioClip(), loader, checkpoint="mock", transform_hash="transform",
        output_path=path, device="cpu",
    )
    cache = load_image_teacher_cache(
        path,
        expected_filenames=["a.jpg", "b.jpg"],
        checkpoint="mock",
        transform_hash="transform",
    )
    selected = lookup_teacher_embeddings(cache, ["b.jpg"])
    assert selected.dtype == torch.float32
    with pytest.raises(ValueError, match="filenames"):
        load_image_teacher_cache(
            path,
            expected_filenames=["b.jpg", "a.jpg"],
            checkpoint="mock",
            transform_hash="transform",
        )

