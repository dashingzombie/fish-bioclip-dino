"""Deterministic BioCLIP image-teacher cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from fish_vlm.models.bioclip import encode_bioclip_images
from fish_vlm.utils.io import torch_save_atomic


@torch.no_grad()
def build_image_teacher_cache(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    checkpoint: str,
    transform_hash: str,
    output_path: str | Path,
    device: torch.device | str,
    storage_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Cache labelled training images only, rejecting duplicate filenames."""
    embeddings: list[torch.Tensor] = []
    filenames: list[str] = []
    model = model.to(device).eval()
    for batch in loader:
        batch_names = list(batch["filename"])
        if batch.get("species_index") is None:
            raise ValueError("Teacher cache accepts only labelled training batches")
        embeddings.append(encode_bioclip_images(model, batch["bioclip_image"].to(device)).cpu())
        filenames.extend(batch_names)
    if len(filenames) != len(set(filenames)):
        raise ValueError("Duplicate filenames in teacher cache input")
    matrix = torch.cat(embeddings).to(storage_dtype) if embeddings else torch.empty((0, 0), dtype=storage_dtype)
    cache = {
        "embeddings": matrix,
        "filenames": filenames,
        "filename_to_index": {name: index for index, name in enumerate(filenames)},
        "checkpoint": checkpoint,
        "transform_hash": transform_hash,
        "normalised": True,
    }
    torch_save_atomic(cache, output_path)
    return cache


def load_image_teacher_cache(
    path: str | Path,
    *,
    expected_filenames: list[str],
    checkpoint: str,
    transform_hash: str,
) -> dict[str, Any]:
    """Validate cache identity and exact training-image coverage."""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in {
        "filenames": expected_filenames,
        "checkpoint": checkpoint,
        "transform_hash": transform_hash,
        "normalised": True,
    }.items():
        if cache.get(key) != expected:
            raise ValueError(f"Incompatible image-teacher cache field: {key}")
    if cache.get("filename_to_index") != {name: i for i, name in enumerate(expected_filenames)}:
        raise ValueError("Image-teacher cache filename mapping is incompatible")
    norms = cache["embeddings"].float().norm(dim=-1)
    if len(norms) and not torch.allclose(norms, torch.ones_like(norms), atol=1e-3):
        raise ValueError("Image-teacher cache embeddings are not normalised")
    return cache


def lookup_teacher_embeddings(cache: dict[str, Any], filenames: list[str]) -> torch.Tensor:
    """Retrieve embeddings in batch order and convert them to float32."""
    mapping = cache["filename_to_index"]
    missing = [name for name in filenames if name not in mapping]
    if missing:
        raise KeyError(f"Teacher cache misses filenames: {missing}")
    indices = torch.tensor([mapping[name] for name in filenames], dtype=torch.long)
    return cache["embeddings"].index_select(0, indices).float()

