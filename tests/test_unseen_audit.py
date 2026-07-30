from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from conftest import TinyDino
from fish_vlm.inference import audit
from fish_vlm.models.multimodal import FishMultimodalModel
from fish_vlm.models.projector import (
    LearnableLogitScale,
    LinearDinoToBioClipProjector,
)


def _config() -> dict:
    return {
        "data": {"root_dir": ".", "processed_dir": "data/processed"},
        "inference": {
            "unseen": {
                "candidate_set": "unseen",
                "mode": "dino_text",
            }
        },
        "model": {"dino": {"name": "tiny"}},
    }


def _bundle(*, supervised: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        model=FishMultimodalModel(
            TinyDino(),
            LinearDinoToBioClipProjector(4, 3),
            LearnableLogitScale(),
            supervised_head=nn.Linear(4, 1) if supervised else None,
        ),
        partitions=SimpleNamespace(
            seen_species=["Seen fish"],
            unseen_species=["First unseen", "Second unseen"],
            all_species=["First unseen", "Second unseen", "Seen fish"],
        ),
        dino_source="mock-dino",
        bioclip_checkpoint="mock-bioclip",
    )


def test_unseen_audit_reports_candidate_random_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    names = ["First unseen", "Second unseen"]
    monkeypatch.setattr(audit, "build_runtime", lambda config, device: bundle)
    monkeypatch.setattr(
        audit,
        "load_candidate_prototypes",
        lambda config, runtime, candidate_set, device: (
            torch.eye(2, 3),
            names,
            {
                "species_names": names,
                "prompt_hash": f"{candidate_set}-prompt-hash",
            },
        ),
    )
    monkeypatch.setattr(
        audit,
        "read_json",
        lambda path: {
            "First unseen": "first",
            "Second unseen": "second",
            "Seen fish": "seen",
        },
    )
    monkeypatch.setattr(
        audit,
        "load_checkpoint",
        lambda *args, **kwargs: {"step": 17},
    )

    result = audit.audit_unseen_inference(_config(), "best.pt")

    assert result["number_of_candidate_species"] == 2
    assert result["random_accuracy"] == 0.5
    assert result["candidate_species"] == names
    assert all(result["checks"].values())


def test_unseen_audit_rejects_constructed_supervised_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "build_runtime",
        lambda config, device: _bundle(supervised=True),
    )
    with pytest.raises(ValueError, match="supervised seen classifier"):
        audit.audit_unseen_inference(_config(), "best.pt")
