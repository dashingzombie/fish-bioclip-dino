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
    "embedding_dim", "prompt_hash", "normalised", "cache_schema_version",
    "tokenizer_probe_tokens", "text_encoder_probe_embeddings",
}
TEXT_CACHE_SCHEMA_VERSION = 2
TEXT_IDENTITY_PROBES = (
    "A photograph of a fish.",
    "Salmo salar",
)


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
    probe_tokens = tokenizer(list(TEXT_IDENTITY_PROBES))
    if not torch.is_tensor(probe_tokens):
        raise TypeError("BioCLIP tokenizer must return a token tensor")
    probe_embeddings = encode_bioclip_text(
        model, probe_tokens.to(device)
    ).cpu()
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
        "cache_schema_version": TEXT_CACHE_SCHEMA_VERSION,
        "tokenizer_probe_tokens": probe_tokens.cpu(),
        "text_encoder_probe_embeddings": probe_embeddings,
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
        "cache_schema_version": TEXT_CACHE_SCHEMA_VERSION,
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
    probe_tokens = cache["tokenizer_probe_tokens"]
    probe_embeddings = cache["text_encoder_probe_embeddings"]
    if not torch.is_tensor(probe_tokens) or probe_tokens.ndim != 2:
        raise ValueError("Text prototype cache tokenizer probe is incompatible")
    if (
        not torch.is_tensor(probe_embeddings)
        or probe_embeddings.shape != (len(TEXT_IDENTITY_PROBES), int(cache["embedding_dim"]))
    ):
        raise ValueError("Text prototype cache text-encoder probe is incompatible")
    probe_norms = probe_embeddings.float().norm(dim=-1)
    if not torch.allclose(
        probe_norms, torch.ones_like(probe_norms), atol=1e-4
    ):
        raise ValueError("Text prototype cache encoder probe is not normalised")
    return cache


@torch.no_grad()
def validate_bioclip_text_identity(
    cache: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    *,
    device: torch.device | str,
) -> None:
    """Prove the live tokenizer and text encoder reproduce the cached identity probes."""
    tokens = tokenizer(list(TEXT_IDENTITY_PROBES))
    if not torch.is_tensor(tokens):
        raise TypeError("BioCLIP tokenizer must return a token tensor")
    cached_tokens = cache["tokenizer_probe_tokens"]
    if not torch.equal(tokens.cpu(), cached_tokens):
        raise ValueError(
            "Live BioCLIP tokenizer is inconsistent with the text-prototype cache"
        )
    embeddings = encode_bioclip_text(model, tokens.to(device)).cpu()
    cached_embeddings = cache["text_encoder_probe_embeddings"].float()
    if not torch.allclose(
        embeddings.float(), cached_embeddings, atol=1e-4, rtol=1e-4
    ):
        raise ValueError(
            "Live BioCLIP text encoder is inconsistent with the text-prototype cache"
        )


def load_prompts(path: str | Path) -> dict[str, str]:
    """Load and validate canonical prompt JSON."""
    prompts = read_json(path)
    if not isinstance(prompts, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in prompts.items()):
        raise TypeError("Canonical prompts must map species strings to prompt strings")
    return prompts
