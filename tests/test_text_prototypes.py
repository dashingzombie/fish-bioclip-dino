from __future__ import annotations

from pathlib import Path

import pytest
import torch

from conftest import TinyBioClip, tiny_tokenizer
from fish_vlm.prototypes.text import build_text_prototype_cache, load_text_prototype_cache
from fish_vlm.utils.hashing import prompts_hash


def test_text_cache_round_trip_and_incompatibility(tmp_path: Path) -> None:
    prompts = {"A fish": "A prompt", "B fish": "B prompt"}
    names = sorted(prompts)
    path = tmp_path / "text.pt"
    cache = build_text_prototype_cache(
        prompts, names, TinyBioClip(), tiny_tokenizer, "mock", path, batch_size=1
    )
    assert cache["embeddings"].shape == (2, 3)
    assert torch.allclose(cache["embeddings"].norm(dim=-1), torch.ones(2), atol=1e-5)
    loaded = load_text_prototype_cache(
        path,
        species_names=names,
        checkpoint="mock",
        prompt_hash=prompts_hash(prompts, names),
        embedding_dim=3,
    )
    assert loaded["species_names"] == names
    with pytest.raises(ValueError, match="Incompatible"):
        load_text_prototype_cache(
            path,
            species_names=names,
            checkpoint="different",
            prompt_hash=prompts_hash(prompts, names),
            embedding_dim=3,
        )

