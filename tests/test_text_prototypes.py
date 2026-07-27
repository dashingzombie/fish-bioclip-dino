from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from conftest import TinyBioClip, tiny_tokenizer
from fish_vlm import cli
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


def test_all_existing_text_caches_skip_bioclip_loading(
    monkeypatch, tmp_path: Path
) -> None:
    prompts = {"A fish": "A prompt", "B fish": "B prompt"}
    partitions = SimpleNamespace(
        seen_species=["A fish"],
        unseen_species=["B fish"],
        all_species=["A fish", "B fish"],
    )
    for candidate_set in ("seen", "unseen", "all"):
        names = getattr(partitions, f"{candidate_set}_species")
        build_text_prototype_cache(
            prompts,
            names,
            TinyBioClip(),
            tiny_tokenizer,
            "mock",
            tmp_path / "text" / f"text_prototypes_{candidate_set}.pt",
        )

    monkeypatch.setattr(cli, "ensure_partitions", lambda config: partitions)
    monkeypatch.setattr(cli, "load_prompts", lambda path: prompts)
    monkeypatch.setattr(
        cli,
        "_cache_path",
        lambda config, *parts: tmp_path.joinpath(*parts),
    )

    def forbidden(checkpoint):
        raise AssertionError("BioCLIP must not load when every text cache is valid")

    monkeypatch.setattr(cli, "load_bioclip", forbidden)
    cli._build_text(
        {
            "data": {"root_dir": str(tmp_path), "processed_dir": "processed"},
            "model": {"bioclip": {"checkpoint": "mock"}},
            "training": {"eval_batch_size": 2},
        }
    )
