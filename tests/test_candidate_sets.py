import pytest
import torch

from conftest import TinyBioClip
from fish_vlm.models.bioclip import encode_bioclip_images
from fish_vlm.models.multimodal import text_logits


def test_candidate_prototypes_restrict_logits() -> None:
    embeddings = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    seen = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    unseen = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    assert text_logits(embeddings, seen, 10.0).shape == (4, 2)
    assert text_logits(embeddings, unseen, 10.0).shape == (4, 5)
    with pytest.raises(ValueError, match="dimensions"):
        text_logits(embeddings, torch.randn(2, 4), 1.0)


def test_native_bioclip_logits_use_the_same_prototypes() -> None:
    model = TinyBioClip()
    images = torch.eye(4)[:2]
    prototypes = torch.nn.functional.normalize(torch.randn(3, 3), dim=-1)
    embedding = encode_bioclip_images(model, images)
    logits = text_logits(embedding, prototypes, 5.0)
    assert embedding.shape == (2, 3)
    assert logits.shape == (2, 3)
