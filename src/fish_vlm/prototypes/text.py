"""Build and validate frozen BioCLIP text-prototype caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from fish_vlm.models.bioclip import encode_bioclip_text
from fish_vlm.utils.hashing import prompts_hash
from fish_vlm.utils.io import read_json, torch_save_atomic

REQUIRED_KEYS = {
    "embeddings", "species_names", "species_to_index", "checkpoint",
    "embedding_dim", "prompt_hash", "normalised",
}


@torch.no_grad()
def build_text_prototype_cache(
    prompts: dict[str, str],
    species_names: list[str],
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint: str,
    output_path: str | Path,
    *,
    batch_size: int = 1024,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Encode one canonical prompt per species, reusing a valid cache."""
    names = list(species_names)
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("species_names must be unique and sorted")
    missing = sorted(set(names) - set(prompts))
    if missing:
        raise ValueError(f"Missing canonical prompts: {missing}")
    if Path(output_path).exists():
        try:
            return load_text_prototype_cache(
                output_path,
                species_names=names,
                checkpoint=checkpoint,
                prompt_hash=prompts_hash(prompts, names),
            )
        except ValueError as error:
            raise ValueError(
                f"Existing text prototype cache at {output_path} is invalid: {error}"
            ) from error

    model = model.to(device)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(names), batch_size):
        texts = [prompts[name] for name in names[start : start + batch_size]]
        tokens = tokenizer(texts).to(device)
        chunks.append(encode_bioclip_text(model, tokens).cpu())
    embeddings = torch.cat(chunks) if chunks else torch.empty((0, 0))
    cache = {
        "embeddings": embeddings,
        "species_names": names,
        "species_to_index": {name: index for index, name in enumerate(names)},
        "checkpoint": checkpoint,
        "embedding_dim": int(embeddings.shape[-1]),
        "prompt_hash": prompts_hash(prompts, names),
        "normalised": True,
    }
    torch_save_atomic(cache, output_path)
    return cache


def load_text_prototype_cache(
    path: str | Path,
    *,
    species_names: list[str],
    checkpoint: str,
    prompt_hash: str,
    embedding_dim: int | None = None,
) -> dict[str, Any]:
    """Load a cache only when every scientific identity field matches."""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    missing = REQUIRED_KEYS - set(cache)
    if missing:
        raise ValueError(f"Text prototype cache lacks fields: {sorted(missing)}")
    expected = {
        "species_names": species_names,
        "checkpoint": checkpoint,
        "prompt_hash": prompt_hash,
        "normalised": True,
    }
    if embedding_dim is not None:
        expected["embedding_dim"] = embedding_dim
    mismatches = {key: (cache[key], value) for key, value in expected.items() if cache[key] != value}
    if mismatches:
        raise ValueError(f"Incompatible text prototype cache: {mismatches}")
    expected_map = {name: index for index, name in enumerate(species_names)}
    if cache["species_to_index"] != expected_map:
        raise ValueError("Text prototype cache species_to_index is incompatible")
    embeddings = cache["embeddings"]
    if embeddings.ndim != 2 or embeddings.shape != (
        len(species_names),
        int(cache["embedding_dim"]),
    ):
        raise ValueError("Text prototype cache embedding shape is incompatible")
    norms = embeddings.float().norm(dim=-1)
    if len(norms) and not torch.allclose(norms, torch.ones_like(norms), atol=1e-4):
        raise ValueError("Text prototype cache claims normalisation but embeddings are not normalised")
    return cache


def load_prompts(path: str | Path) -> dict[str, str]:
    """Load and validate canonical prompt JSON."""
    prompts = read_json(path)
    if not isinstance(prompts, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in prompts.items()):
        raise TypeError("Canonical prompts must map species strings to prompt strings")
    return prompts
